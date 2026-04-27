"""Probe: does the 5v5 multi-agent loop actually run at 1:1 wall time?

Builds the same runner harness.py builds (5v5, 10 PlayerAgents, real
TeamMessageBus, real fallbacks, real motor controllers), but does NOT
start() the per-agent worker threads — so no LLM HTTP calls happen.
This means the per-tick budget consists of exactly:

  - drain skill_queue (always empty -> noop)
  - step_motor for 10 agents (fallback controllers -> ms-level)
  - env.step_actions (gfootball)
  - perception.filter for 10 agents every obs_refresh_every_ticks ticks
  - on_tick / on_frame callbacks if wired
  - sleep to cap target_wall_fps

Reports actual ticks/sec achieved and the dominant cost. Then runs three
configurations (headless, render+frame stream, render+frame stream w/
JPEG encode) so we know which ones drop below 1.0x.

Run:
    wsl.exe -d Ubuntu-22.04 -- bash -lc \\
      "source ~/football-env/bin/activate && cd /mnt/c/Users/dfgfd/Desktop/football_RL \\
       && unset DISPLAY && python3 -u scripts/probe_realtime_pacing.py"
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress gfootball / gym noise
import logging
logging.basicConfig(level=logging.ERROR, format="%(name)s: %(message)s")

from football_agents.env import FootballEnvAdapter
from football_agents.message_bus import TeamMessageBus
from football_agents.multi_agent_runner import MultiAgentRunner
from football_agents.perception import EgocentricFilter
from football_agents.personas import TEAM_BLUE_5V5, TEAM_RED_5V5
from football_agents.player_agent import PlayerAgent


# -------- a stub LLMClient so we never need API keys --------------------

class StubLLMClient:
    """Mimics LLMClient's surface enough that PlayerAgent / LLMPlayer
    construct without env vars. Worker threads are never started, so
    these methods are not actually called during the probe.
    """
    model = "stub-no-llm"
    base_url = "stub://none"
    n_keys = 0


# -------- per-tick instrumentation ---------------------------------------

class TickProfiler:
    """Records per-section wall-clock budget by patching MultiAgentRunner.

    Buckets: motor_step, env_step, perceive_push, on_tick_cb, on_frame_cb,
    sleep. Sums + counts so we can report avg ms / tick.
    """

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, dt: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1

    def report(self, n_ticks: int) -> str:
        lines = [f"  tick budget breakdown over {n_ticks} ticks (avg ms/tick):"]
        for k in sorted(self.sums.keys()):
            avg_ms = (self.sums[k] / max(1, n_ticks)) * 1000.0
            total_s = self.sums[k]
            lines.append(f"    {k:<20s} = {avg_ms:>6.2f} ms   (total {total_s:>5.2f}s)")
        return "\n".join(lines)


def patch_runner_for_profiling(runner: MultiAgentRunner, prof: TickProfiler) -> None:
    """Wrap the methods on the runner / env so we can time each section."""
    env = runner.env
    orig_step_actions = env.step_actions
    def timed_step_actions(actions):
        t0 = time.perf_counter()
        orig_step_actions(actions)
        prof.add("env_step", time.perf_counter() - t0)
    env.step_actions = timed_step_actions  # type: ignore[assignment]

    for a in runner.agents:
        orig_step = a.step_motor
        def make_timed(orig):
            def timed(raw):
                t0 = time.perf_counter()
                r = orig(raw)
                prof.add("motor_step", time.perf_counter() - t0)
                return r
            return timed
        a.step_motor = make_timed(orig_step)  # type: ignore[assignment]

        orig_perceive = a.perceive
        def make_perc(orig):
            def timed(raw, tick):
                t0 = time.perf_counter()
                r = orig(raw, tick)
                prof.add("perceive", time.perf_counter() - t0)
                return r
            return timed
        a.perceive = make_perc(orig_perceive)  # type: ignore[assignment]


# -------- build a runner without starting worker threads -----------------

def build_probe_runner(
    *,
    render: bool,
    on_tick: bool,
    on_frame: bool,
    target_wall_fps: float = 50.0,
    physics_steps_per_frame: int = 2,
):
    client = StubLLMClient()
    env = FootballEnvAdapter(
        scenario="llm_5v5_full",
        render=render,
        n_controlled_left=5,
        n_controlled_right=5,
        primary_player_slot=0,
        physics_steps_per_frame=physics_steps_per_frame,
    )
    env.reset()
    bus = TeamMessageBus()

    raw = env.raw_obs
    role_arr_left = raw["left_team_roles"]
    role_arr_right = raw["right_team_roles"]
    agents = []
    for slot in range(5):
        role_id = int(role_arr_left[slot])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        agents.append(PlayerAgent(
            slot=slot, player_id=slot, team_side="left",
            role=role_name, persona=TEAM_BLUE_5V5[slot],
            llm_client=client, bus=bus,
        ))
    for slot in range(5):
        role_id = int(role_arr_right[slot])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        agents.append(PlayerAgent(
            slot=5 + slot, player_id=slot, team_side="right",
            role=role_name, persona=TEAM_RED_5V5[slot],
            llm_client=client, bus=bus,
        ))

    cb_counters = {"tick": 0, "frame": 0, "frame_bytes": 0}
    on_tick_cb = None
    on_frame_cb = None
    if on_tick:
        def _ot(snap):
            cb_counters["tick"] += 1
        on_tick_cb = _ot
    if on_frame:
        # Match what eval_platform does: JPEG-encode the frame so we measure
        # the realistic stream cost. If cv2 is missing, fall back to a noop.
        try:
            import cv2
            import numpy as np
            jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            def _of(frame):
                cb_counters["frame"] += 1
                # Convert RGB -> BGR for cv2, then encode JPEG q95 (matches
                # the typical UI stream config).
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, jpeg_params)
                if ok:
                    cb_counters["frame_bytes"] += len(buf)
        except Exception:
            def _of(frame):
                cb_counters["frame"] += 1
                cb_counters["frame_bytes"] += int(getattr(frame, "nbytes", 0))
        on_frame_cb = _of

    runner = MultiAgentRunner(
        env=env, agents=agents,
        obs_refresh_every_ticks=25,
        max_decisions_total=10**9,   # never hit
        max_wall_seconds=10**9,      # never hit (probe controls duration)
        target_wall_fps=target_wall_fps,
        on_tick=on_tick_cb,
        tick_stream_every_ticks=5,
        on_frame=on_frame_cb,
        frame_stream_every_ticks=3,
        # Crucial for the probe: with stub LLM, no agent ever produces a
        # first skill, so the default 8s kickoff wait would burn 8s of
        # our 30s budget on noop sleep. Skip it.
        kickoff_wait_seconds=0.0,
    )
    return runner, cb_counters


# -------- the actual probe -----------------------------------------------

def probe(label: str, *, render: bool, on_tick: bool, on_frame: bool,
          duration_s: float = 30.0, target_wall_fps: float = 50.0,
          physics_steps_per_frame: int = 2) -> dict:
    print("=" * 78)
    print(f"PROBE: {label}")
    print(f"  render={render}  on_tick={on_tick}  on_frame={on_frame}  "
          f"target_wall_fps={target_wall_fps}  pps={physics_steps_per_frame}  "
          f"duration={duration_s}s")
    print("=" * 78)

    runner, cb_counters = build_probe_runner(
        render=render, on_tick=on_tick, on_frame=on_frame,
        target_wall_fps=target_wall_fps,
        physics_steps_per_frame=physics_steps_per_frame,
    )
    prof = TickProfiler()
    patch_runner_for_profiling(runner, prof)

    # IMPORTANT: prevent worker threads from spinning up. We override
    # PlayerAgent.start to a no-op so no LLM call ever happens. The
    # main loop still drains the (empty) skill_queue every tick exactly
    # like production -- so we measure the loop, not the LLM.
    for a in runner.agents:
        a.start = lambda: None  # type: ignore[assignment]

    # Stop the run after duration_s wall seconds via a background timer.
    import threading
    def _stop_after():
        time.sleep(duration_s)
        runner.request_stop()
    t = threading.Thread(target=_stop_after, daemon=True)
    t.start()

    t0 = time.monotonic()
    result = runner.run()
    wall = time.monotonic() - t0

    ticks = result["env_ticks"]
    actual_fps = ticks / wall if wall > 0 else 0.0
    # game time (sec) per tick = physics_steps_per_frame / 100
    game_seconds = ticks * (physics_steps_per_frame / 100.0)
    ratio = game_seconds / wall if wall > 0 else 0.0

    print(f"  -> ran {wall:.2f}s wall, {ticks} env ticks")
    print(f"  -> actual {actual_fps:.2f} ticks/wall-sec  (target 50.00)")
    print(f"  -> game-sec / wall-sec = {ratio:.3f}   "
          f"({'1:1 OK' if 0.97 <= ratio <= 1.03 else 'OUT OF SPEC'})")
    if cb_counters["frame"]:
        avg_kb = cb_counters["frame_bytes"] / cb_counters["frame"] / 1024.0
        print(f"  -> on_frame fired {cb_counters['frame']} times, "
              f"avg JPEG {avg_kb:.1f} KB")
    if cb_counters["tick"]:
        print(f"  -> on_tick fired {cb_counters['tick']} times")
    print(prof.report(ticks))
    print()

    runner.env.close()
    return {
        "label": label, "wall": wall, "ticks": ticks,
        "actual_fps": actual_fps, "ratio": ratio,
        "frames": cb_counters["frame"],
    }


def main() -> None:
    DURATION = float(os.environ.get("PROBE_DURATION", "30"))
    print(f"# probe_realtime_pacing.py  duration={DURATION}s/probe\n")

    results = []
    # 1. baseline: headless, no callbacks, no frame stream. If even THIS
    #    can't hit 50 fps, the loop itself is too heavy.
    results.append(probe(
        "headless, no callbacks", render=False, on_tick=False, on_frame=False,
        duration_s=DURATION,
    ))
    # 2. headless + on_tick (cheap dict serialization).
    results.append(probe(
        "headless + on_tick", render=False, on_tick=True, on_frame=False,
        duration_s=DURATION,
    ))
    # 3. render + on_frame + JPEG encode (realistic eval-server load).
    #    NOTE: on WSL with no DISPLAY this should still work via gfootball's
    #    EGL off-screen path -- env.latest_frame returns the RGB array.
    results.append(probe(
        "render + on_frame (JPEG q95)  [current config]",
        render=True, on_tick=True, on_frame=True, duration_s=DURATION,
        target_wall_fps=50.0, physics_steps_per_frame=2,
    ))
    # 4. PROPOSED FIX A: pps=3 + target 33.34 fps. Tick budget 30ms (~22ms
    #    used). 33.34 * 0.03 = 1.000 game-sec/wall-sec.
    results.append(probe(
        "render + on_frame  [pps=3 + 33 fps]",
        render=True, on_tick=True, on_frame=True, duration_s=DURATION,
        target_wall_fps=33.34, physics_steps_per_frame=3,
    ))
    # 5. PROPOSED FIX B: pps=4 + target 25 fps. Tick budget 40ms (~25ms
    #    used). 25 * 0.04 = 1.000 game-sec/wall-sec. Plenty of headroom.
    results.append(probe(
        "render + on_frame  [pps=4 + 25 fps]",
        render=True, on_tick=True, on_frame=True, duration_s=DURATION,
        target_wall_fps=25.0, physics_steps_per_frame=4,
    ))
    # 6. headless + pps=4 + 25 fps — establishes the headless ceiling
    #    for the same recommendation.
    results.append(probe(
        "headless + on_tick   [pps=4 + 25 fps]",
        render=False, on_tick=True, on_frame=False, duration_s=DURATION,
        target_wall_fps=25.0, physics_steps_per_frame=4,
    ))

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'config':<35s}  {'fps':>7s}  {'ratio':>7s}  {'verdict':<12s}")
    for r in results:
        verdict = "1:1 OK" if 0.97 <= r["ratio"] <= 1.03 else "SLOW"
        print(f"{r['label']:<35s}  {r['actual_fps']:>7.2f}  "
              f"{r['ratio']:>7.3f}  {verdict:<12s}")


if __name__ == "__main__":
    main()
