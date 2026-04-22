"""Phase 4 demo — async LLM + smooth 3D window.

Runs the env tick loop on the main thread (so the gfootball 3D window stays
smooth at native frame rate via WSLg) and the LLM decision loop on a
background thread. Decisions land asynchronously and are atomically swapped
into the active motor controller — no frozen frames during LLM thinking.

Run from WSL with venv active:
    python3 scripts/demo_async_3d.py
"""
from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.agent import LLMPlayer
from football_agents.async_runner import AsyncRunner
from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
# NOTE: fallback policy intentionally NOT used here. The honest async pattern
# is: each agent's LLM is the SOLE source of intent. Between decisions the
# motor controller continues its last skill until completion, then IDLE.
# For Phase 5 multi-agent, each player has its own LLM thread → decisions
# interleave naturally, IDLE gaps shrink. See conversation 2026-04-22.

SCENARIO = "academy_empty_goal_close"  # known-working: attacker in box, empty goal
N_CONTROLLED_LEFT = 2   # left team: GK (#0) + attacker (#1)
PRIMARY_SLOT = 1        # we drive slot 1 (the attacker); GK (slot 0) stays IDLE


def short_skill(skill) -> str:
    name = type(skill).__name__
    args = []
    for slot in ("target_x", "target_y", "target_player_id", "target_zone",
                 "pass_type", "urgency", "opponent_id", "entity_id", "audience"):
        if hasattr(skill, slot):
            v = getattr(skill, slot)
            args.append(f"{slot}={v:+.2f}" if isinstance(v, float) else f"{slot}={v}")
    if hasattr(skill, "message"):
        args.append(f"msg={skill.message!r}")
    return f"{name}({', '.join(args)})"


def main() -> None:
    print("=" * 78)
    print("Phase 4 ASYNC demo — 3D window stays smooth, LLM decides in background")
    print("=" * 78)

    client = LLMClient.from_env()
    print(f"LLM model: {client.model}\n")

    env = FootballEnvAdapter(
        scenario=SCENARIO,
        render=True,
        n_controlled_left=N_CONTROLLED_LEFT,
        primary_player_slot=PRIMARY_SLOT,
    )
    obs = env.reset()
    pid = obs.self_state.player_id
    role = obs.self_state.role
    print(f"Controlling player #{pid} ({role})\n")

    agent = LLMPlayer(player_id=pid, role=role, llm_client=client)

    def on_decision(log: dict) -> None:
        s = log["skill"]
        agent_log = agent.history[-1] if agent.history else None
        reasoning = (agent_log.reasoning[:80] + "...") if agent_log and agent_log.reasoning else ""
        print(
            f"#{log['decision']:>2}  env_tick={log['env_tick']:>3d}  "
            f"obs_tick={log['obs_tick']:>3d}  lag={log['lag_ticks']:>2d}t  "
            f"llm={log['llm_seconds']:.1f}s  →  {short_skill(s)}"
            + (f"   💭 {reasoning}" if reasoning else "")
        )

    runner = AsyncRunner(
        env=env,
        agent=agent,
        fallback_policy=None,  # uses default body_rest_state_fallback (slow dribble / jog)
        obs_refresh_every_ticks=4,
        max_decisions=60,
        max_wall_seconds=180.0,
        on_decision=on_decision,
    )

    print("Running... (3D window should stay smooth this time)")
    print("─" * 78)
    t0 = time.monotonic()
    result = runner.run()
    wall = time.monotonic() - t0

    print("─" * 78)
    print()
    print("=" * 78)
    if result["cumulative_reward"] > 0:
        verdict = f"✅ GOAL! reward={result['cumulative_reward']:+.1f}"
    elif env.done:
        verdict = f"⏹  episode ended  reward={result['cumulative_reward']:+.1f}"
    else:
        verdict = f"⏱  cap hit  reward={result['cumulative_reward']:+.1f}"
    print(
        f"{verdict}\n"
        f"   {result['decisions']} LLM decisions over {result['env_ticks']} env ticks, "
        f"{wall:.1f}s wall"
    )
    if result["decisions"] > 0:
        avg_llm = sum(d["llm_seconds"] for d in result["log"]) / result["decisions"]
        avg_lag = sum(d["lag_ticks"] for d in result["log"]) / result["decisions"]
        print(f"   avg LLM latency: {avg_llm:.2f}s  |  avg obs→action lag: {avg_lag:.1f} ticks")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
