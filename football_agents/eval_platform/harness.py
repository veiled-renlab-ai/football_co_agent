"""Episode harness — run one or more 5v5 matches headless and capture metrics.

Wraps MultiAgentRunner. Headless by default (render=False). Each episode
streams decisions through `on_decision` so the server can push live updates.

Designed to be safe to call from a worker thread (one episode at a time per
process — gfootball env is not thread-safe across episodes).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Callable, Optional

from ..env import FootballEnvAdapter
from ..llm_client import LLMClient, build_channel_pool
from ..message_bus import TeamMessageBus
from ..multi_agent_runner import MultiAgentRunner
from ..perception import EgocentricFilter
from ..players import (
    TEAM_BLUE_5V5, TEAM_BLUE_11V11, TEAM_RED_5V5, TEAM_RED_11V11,
)
from ..player_agent import PlayerAgent
from .metrics import serialize_decision_log, summarize_episode


@dataclass
class EpisodeConfig:
    scenario: str = "llm_11v11_full"
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
    # Stop-world mode: pause env while all agents decide, then execute K ticks.
    # Default False = original async mode (field keeps moving during LLM think).
    stop_world_mode: bool = False
    # Per-agent timeout (seconds) before arming fallback and unblocking.
    stop_world_timeout: float = 8.0
    # User-injected per-slot soul overrides. Each entry: {"slot": int, "soul": str}.
    # When a slot is in this list, its persona's custom_soul field is set to the
    # user's text and that text replaces the default play_style+background block
    # in the system prompt. Slots not listed use their default persona.
    lineup_overrides: list[dict] = field(default_factory=list)

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


def _build_agents(
    clients: list[LLMClient],
    env: FootballEnvAdapter,
    bus: TeamMessageBus,
    n_per_side: int,
    blue_team: tuple,
    red_team: tuple,
    overrides_by_slot: Optional[dict[int, str]] = None,
) -> tuple[list[PlayerAgent], dict[int, str]]:
    """Build N agents per side, binding each to a channel from `clients` round-robin.

    `overrides_by_slot`: optional {global_slot: soul_text}. When a slot has an
    entry, its persona's `custom_soul` is patched with the user's text — the
    prompt builder then uses that text in place of the default
    play_style+background block.
    """
    raw = env.raw_obs
    role_arr_left = raw["left_team_roles"]
    role_arr_right = raw["right_team_roles"]
    agents: list[PlayerAgent] = []
    slot_to_label: dict[int, str] = {}
    n_chan = len(clients)
    overrides_by_slot = overrides_by_slot or {}

    def _chan(global_slot: int) -> LLMClient:
        return clients[global_slot % n_chan]

    def _maybe_patch(persona, global_slot: int):
        soul = overrides_by_slot.get(global_slot)
        if soul and soul.strip():
            return dataclass_replace(persona, custom_soul=soul.strip())
        return persona

    for slot in range(n_per_side):
        player_id = slot
        role_id = int(role_arr_left[player_id])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        persona = _maybe_patch(blue_team[slot], slot)
        agents.append(PlayerAgent(
            slot=slot, player_id=player_id, team_side="left",
            role=role_name, persona=persona, llm_client=_chan(slot), bus=bus,
        ))
        slot_to_label[slot] = f"蓝·{persona.name}#{persona.jersey_number} ({persona.position})"

    for slot in range(n_per_side):
        env_slot = n_per_side + slot
        player_id = slot
        role_id = int(role_arr_right[player_id])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        persona = _maybe_patch(red_team[slot], env_slot)
        agents.append(PlayerAgent(
            slot=env_slot, player_id=player_id, team_side="right",
            role=role_name, persona=persona, llm_client=_chan(env_slot), bus=bus,
        ))
        slot_to_label[env_slot] = f"红·{persona.name}#{persona.jersey_number} ({persona.position})"

    return agents, slot_to_label


def run_episode(
    config: EpisodeConfig,
    *,
    client: Optional[LLMClient] = None,
    clients: Optional[list[LLMClient]] = None,
    on_decision: Optional[Callable[[dict], None]] = None,
    on_status: Optional[Callable[[str, dict], None]] = None,
    on_tick: Optional[Callable[[dict], None]] = None,
    on_frame: Optional[Callable[[object], None]] = None,
    register_runner: Optional[Callable[["MultiAgentRunner"], None]] = None,
) -> EpisodeResult:
    """Run one episode end-to-end. Blocks until done.

    `clients`: optional list of pre-built LLMClient channels for per-agent binding.
        Defaults to build_channel_pool() (4 channels for 11v11 multi-key load
        balancing). Falls back to a 1-element list wrapping `client` for
        single-channel mode (legacy 5v5 demos).
    `client`:  legacy single-client mode. Wrapped into a 1-element pool if
        `clients` is not provided.
    """
    started = time.time()

    # Resolve channel pool: explicit `clients` > explicit `client` > env-driven pool.
    if clients is not None:
        pool = clients
    elif client is not None:
        pool = [client]
    else:
        pool = build_channel_pool()

    # Pick which team rosters to use based on scenario.
    is_11v11 = "11v11" in config.scenario
    n_per_side = 11 if is_11v11 else 5
    blue_team = TEAM_BLUE_11V11 if is_11v11 else TEAM_BLUE_5V5
    red_team  = TEAM_RED_11V11 if is_11v11 else TEAM_RED_5V5

    if on_status:
        on_status("initializing", {})

    env = FootballEnvAdapter(
        scenario=config.scenario,
        render=config.render,
        n_controlled_left=n_per_side,
        n_controlled_right=n_per_side,
        primary_player_slot=0,
        physics_steps_per_frame=config.physics_steps_per_frame,
    )
    env.reset()
    bus = TeamMessageBus()
    overrides_by_slot = {
        int(ov["slot"]): str(ov.get("soul", ""))
        for ov in (config.lineup_overrides or [])
        if "slot" in ov
    }
    agents, slot_to_label = _build_agents(
        pool, env, bus, n_per_side, blue_team, red_team,
        overrides_by_slot=overrides_by_slot,
    )

    if on_status:
        on_status("starting", {
            "agents": [
                {"slot": s, "label": slot_to_label[s]} for s in sorted(slot_to_label)
            ],
            "scenario": config.scenario,
            "model": ", ".join(sorted({c.model for c in pool})),
            "channels": len(pool),
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
        on_status=on_status,
        stop_world_mode=config.stop_world_mode,
        stop_world_timeout=config.stop_world_timeout,
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
    """Run N episodes in series. Reuses one channel pool across episodes."""
    pool = build_channel_pool()
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
            clients=pool,
            on_decision=_wrap_on_decision if on_decision else None,
            on_status=_wrap_on_status if on_status else None,
        )
        results.append(result)
        if on_episode_end is not None:
            on_episode_end(ep_index, result)
    return results
