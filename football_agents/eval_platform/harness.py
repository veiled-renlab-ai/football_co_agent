"""Episode harness — run one or more 5v5 matches headless and capture metrics.

Wraps MultiAgentRunner. Headless by default (render=False). Each episode
streams decisions through `on_decision` so the server can push live updates.

Designed to be safe to call from a worker thread (one episode at a time per
process — gfootball env is not thread-safe across episodes).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..env import FootballEnvAdapter
from ..llm_client import LLMClient
from ..message_bus import TeamMessageBus
from ..multi_agent_runner import MultiAgentRunner
from ..perception import EgocentricFilter
from ..personas import TEAM_BLUE_5V5, TEAM_RED_5V5
from ..player_agent import PlayerAgent
from .metrics import serialize_decision_log, summarize_episode


@dataclass
class EpisodeConfig:
    scenario: str = "llm_5v5_full"
    # Decisions cap is effectively unlimited — wall-clock and gfootball's
    # built-in done flag (or user pressing Stop) end the match.
    max_decisions_total: int = 100_000
    # 90 game minutes = 90*60 = 5400s wall (real-time pacing locks 1:1).
    max_wall_seconds: float = 5400.0
    obs_refresh_every_ticks: int = 25
    # Real-time pacing knobs. The MultiAgentRunner uses deadline-carry pacing
    # (compensates for time.sleep overshoot) so the achieved game-sec/wall-sec
    # ratio is the product wall_fps * pps * 0.01.
    #   - HEADLESS:  pps=2 + wall_fps=50  →  50*2*0.01 = 1.00x (verified 0.977
    #     measured by scripts/probe_realtime_pacing.py — within ±3% spec).
    #   - RENDER=True: gfootball's render adds ~16ms to env.step (measured
    #     ~19ms total vs ~4ms headless), so the 50 fps target slips to ~0.95x.
    #     Auto-switch to pps=3 + wall_fps=33.34 when render=True (33.34*3*0.01
    #     = 1.000x exactly, with a 30ms tick budget vs ~22ms used → 8ms slack).
    # If you override these manually, keep wall_fps * pps * 0.01 == 1.0 for 1:1.
    target_wall_fps: float = 50.0
    physics_steps_per_frame: int = 2
    # Stream raw positions for the live pitch view every N ticks (5 = 10 Hz at 50 fps).
    tick_stream_every_ticks: int = 5
    # Stream gfootball 3D rendered frame every N ticks. Only fires when render=True.
    # 3 ticks @ 50fps cap = ~17 fps video — smooth enough, ~25 KB/frame JPEG = ~400 KB/s.
    frame_stream_every_ticks: int = 3
    render: bool = False

    def __post_init__(self) -> None:
        # When render=True, the heavier env.step (~19ms) blows the 20ms tick
        # budget at pps=2/50fps. Auto-switch to pps=3/33.34fps so the runner
        # has a 30ms budget and game-sec/wall-sec stays at 1.0. The user can
        # still override by passing explicit values that aren't the defaults.
        if (self.render
                and self.target_wall_fps == 50.0
                and self.physics_steps_per_frame == 2):
            self.target_wall_fps = 33.34
            self.physics_steps_per_frame = 3


@dataclass
class EpisodeResult:
    config: EpisodeConfig
    started_at: float
    ended_at: float
    summary: dict = field(default_factory=dict)
    decision_log: list[dict] = field(default_factory=list)
    error: Optional[str] = None


def _build_agents(client: LLMClient, env: FootballEnvAdapter, bus: TeamMessageBus) -> tuple[list[PlayerAgent], dict[int, str]]:
    """Mirror the demo's agent assembly. Returns (agents, slot->label)."""
    raw = env.raw_obs
    role_arr_left = raw["left_team_roles"]
    role_arr_right = raw["right_team_roles"]
    agents: list[PlayerAgent] = []
    slot_to_label: dict[int, str] = {}

    for slot in range(5):
        player_id = slot
        role_id = int(role_arr_left[player_id])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        persona = TEAM_BLUE_5V5[slot]
        agents.append(PlayerAgent(
            slot=slot, player_id=player_id, team_side="left",
            role=role_name, persona=persona, llm_client=client, bus=bus,
        ))
        slot_to_label[slot] = f"蓝·{persona.name}#{persona.jersey_number} ({persona.position})"

    for slot in range(5):
        env_slot = 5 + slot
        player_id = slot
        role_id = int(role_arr_right[player_id])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        persona = TEAM_RED_5V5[slot]
        agents.append(PlayerAgent(
            slot=env_slot, player_id=player_id, team_side="right",
            role=role_name, persona=persona, llm_client=client, bus=bus,
        ))
        slot_to_label[env_slot] = f"红·{persona.name}#{persona.jersey_number} ({persona.position})"

    return agents, slot_to_label


def run_episode(
    config: EpisodeConfig,
    *,
    client: Optional[LLMClient] = None,
    on_decision: Optional[Callable[[dict], None]] = None,
    on_status: Optional[Callable[[str, dict], None]] = None,
    on_tick: Optional[Callable[[dict], None]] = None,
    on_frame: Optional[Callable[[object], None]] = None,
    register_runner: Optional[Callable[["MultiAgentRunner"], None]] = None,
) -> EpisodeResult:
    """Run one episode end-to-end. Blocks until done.

    on_decision: invoked once per LLM decision with a JSON-safe dict
                 (skill is already serialized). Use for live UI streaming.
    on_status:   invoked with ("starting"|"running"|"finished", payload).
    """
    started = time.time()
    if client is None:
        client = LLMClient.from_env()

    env = FootballEnvAdapter(
        scenario=config.scenario,
        render=config.render,
        n_controlled_left=5,
        n_controlled_right=5,
        primary_player_slot=0,
        physics_steps_per_frame=config.physics_steps_per_frame,
    )
    env.reset()
    bus = TeamMessageBus()
    agents, slot_to_label = _build_agents(client, env, bus)

    if on_status:
        on_status("starting", {
            "agents": [
                {"slot": s, "label": slot_to_label[s]} for s in sorted(slot_to_label)
            ],
            "scenario": config.scenario,
            "model": client.model,
        })

    fallback_installs = {"n": 0}

    def _serialize_log_entry(entry: dict) -> dict:
        out = {k: v for k, v in entry.items() if k != "skill"}
        from .metrics import serialize_skill
        out["skill"] = serialize_skill(entry["skill"])
        out["label"] = slot_to_label.get(int(entry["slot"]), f"slot {entry['slot']}")
        return out

    def _on_decision(entry: dict) -> None:
        if on_decision is None:
            return
        try:
            on_decision(_serialize_log_entry(entry))
        except Exception:
            pass

    runner = MultiAgentRunner(
        env=env, agents=agents,
        obs_refresh_every_ticks=config.obs_refresh_every_ticks,
        max_decisions_total=config.max_decisions_total,
        max_wall_seconds=config.max_wall_seconds,
        target_wall_fps=config.target_wall_fps,
        on_decision=_on_decision,
        on_tick=on_tick,
        tick_stream_every_ticks=config.tick_stream_every_ticks,
        on_frame=on_frame,
        frame_stream_every_ticks=config.frame_stream_every_ticks,
        # Bubble runner-emitted phase events ("waiting_for_kickoff", "kickoff",
        # ...) up the same on_status pipe the harness already publishes through.
        on_status=on_status,
    )
    # Expose the runner so the server can call request_stop() on user click.
    if register_runner is not None:
        try:
            register_runner(runner)
        except Exception:
            pass

    # Wrap _arm_fallback_for to count installs (no other clean hook exists).
    original_arm = runner._arm_fallback_for
    def _counting_arm(agent):
        fallback_installs["n"] += 1
        return original_arm(agent)
    runner._arm_fallback_for = _counting_arm  # type: ignore[assignment]

    error: Optional[str] = None
    try:
        if on_status:
            on_status("running", {})
        result = runner.run()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        result = {
            "wall_seconds": time.time() - started,
            "env_ticks": env.tick,
            "decisions_total": 0,
            "n_agents": len(agents),
            "cumulative_reward": env.cumulative_reward,
            "log": [],
        }
    finally:
        try:
            env.close()
        except Exception:
            pass

    summary = summarize_episode(
        log=result["log"],
        cumulative_reward=result["cumulative_reward"],
        env_ticks=result["env_ticks"],
        wall_seconds=result["wall_seconds"],
        n_agents=result["n_agents"],
        fallback_installs=fallback_installs["n"],
        slot_to_label=slot_to_label,
    )
    summary["slot_to_label"] = slot_to_label

    ended = time.time()
    serialized_log = serialize_decision_log(result["log"])
    for d in serialized_log:
        d["label"] = slot_to_label.get(int(d["slot"]), f"slot {d['slot']}")

    if on_status:
        on_status("finished", {"summary": summary, "error": error})

    return EpisodeResult(
        config=config,
        started_at=started,
        ended_at=ended,
        summary=summary,
        decision_log=serialized_log,
        error=error,
    )


def run_many_episodes(
    n_episodes: int,
    config: EpisodeConfig,
    *,
    on_decision: Optional[Callable[[dict, int], None]] = None,
    on_episode_end: Optional[Callable[[int, EpisodeResult], None]] = None,
    on_status: Optional[Callable[[str, dict], None]] = None,
) -> list[EpisodeResult]:
    """Run N episodes in series. Reuses one LLMClient across episodes."""
    client = LLMClient.from_env()
    results: list[EpisodeResult] = []
    for i in range(n_episodes):
        ep_index = i + 1

        def _wrap_on_decision(entry: dict, _ep=ep_index) -> None:
            if on_decision is not None:
                on_decision(entry, _ep)

        def _wrap_on_status(phase: str, payload: dict, _ep=ep_index) -> None:
            if on_status is not None:
                on_status(phase, {"episode": _ep, **payload})

        result = run_episode(
            config,
            client=client,
            on_decision=_wrap_on_decision if on_decision else None,
            on_status=_wrap_on_status if on_status else None,
        )
        results.append(result)
        if on_episode_end is not None:
            on_episode_end(ep_index, result)
    return results
