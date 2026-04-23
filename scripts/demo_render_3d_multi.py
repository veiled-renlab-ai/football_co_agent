"""Phase 5a regression demo — N=1 via the NEW MultiAgentRunner pipeline.

Same scenario / persona / render / fallback as scripts/demo_render_3d.py.
Difference: uses PlayerAgent + MultiAgentRunner (the new multi-agent
infrastructure) with N=1. Output should match the v0.walk-fallback
behavior bit-for-bit modulo LLM stochasticity.

Used to verify that the encapsulation refactor introduced ZERO behavior
change before scaling to 5v5.

Run from WSL with venv active:
    python3 -u scripts/demo_render_3d_multi.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient
from football_agents.multi_agent_runner import MultiAgentRunner
from football_agents.perception import EgocentricFilter
from football_agents.player_agent import PlayerAgent
from football_agents.prompts import DEFAULT_PERSONA

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


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
    print("Phase 5a regression — N=1 via MultiAgentRunner (NEW pipeline)")
    print("Should match v0.walk-fallback output (modulo LLM nondeterminism).")
    print("=" * 78)

    client = LLMClient.from_env()
    print(f"LLM model: {client.model}")
    print(f"Pipeline: PlayerAgent + MultiAgentRunner (encapsulated)\n")

    # Same scenario as the legacy demo: 2 controlled left players (slot 0
    # = GK gets IDLE, slot 1 = our LLM-driven attacker).
    env = FootballEnvAdapter(
        scenario="llm_solo_attack",
        render=True,
        n_controlled_left=2,
        primary_player_slot=1,  # legacy field, harmless in multi-agent path
    )
    env.reset()

    # Resolve the role for slot 1 the same way EgocentricFilter would.
    raw = env.raw_obs
    role_id = int(raw["left_team_roles"][1])
    role = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
    print(f"Controlling player #1 ({role}) via PlayerAgent\n")

    # ONE PlayerAgent, slot 1, default persona (same as legacy demo).
    agent = PlayerAgent(
        slot=1,
        player_id=1,
        team_side="left",
        role=role,
        persona=DEFAULT_PERSONA,
        llm_client=client,
    )

    def on_decision(log: dict) -> None:
        last = agent.llm_player.history[-1] if agent.llm_player.history else None
        reasoning = last.reasoning if last and last.reasoning else ""
        err = last.error if last and last.error else None
        print("─" * 78)
        print(f"#{log['decision']:>2}  pid={log['player_id']}  "
              f"env_tick={log['env_tick']:>4d}  obs_tick={log['obs_tick']:>4d}  "
              f"lag={log['lag_ticks']:>3d}t  llm={log['llm_seconds']:.1f}s")
        if reasoning:
            print(f"  💭 {reasoning}")
        if err:
            print(f"  ⚠️  {err}  →  fallback {short_skill(log['skill'])}")
        else:
            print(f"  🎯 {short_skill(log['skill'])}")
        # Show position from same agent's perception (main thread, safe)
        s = agent.perceive(env.raw_obs, env.tick)
        ss = s.self_state
        ball_str = f"my_ball={ss.has_ball}"
        if (b := s.ball()) is not None:
            ball_str += f"  ball@({b.position.x:+.2f},{b.position.y:+.2f}) d={b.distance:.2f}"
        print(f"     reward={env.cumulative_reward:+.1f}  "
              f"pos=({ss.position.x:+.2f},{ss.position.y:+.2f}) {ball_str}")

    runner = MultiAgentRunner(
        env=env,
        agents=[agent],
        # fallback_policy=None → uses body_rest_state_fallback (walk-speed)
        obs_refresh_every_ticks=4,
        max_decisions_total=200,
        max_wall_seconds=300.0,
        on_decision=on_decision,
    )

    print("Running... 3D should be smooth, LLM decisions land async.")
    t0 = time.monotonic()
    result = runner.run()
    wall = time.monotonic() - t0

    print()
    print("=" * 78)
    if result["cumulative_reward"] > 0:
        verdict = f"✅ {int(result['cumulative_reward'])} GOAL(S)  reward={result['cumulative_reward']:+.1f}"
    elif env.done:
        verdict = f"⏹  episode ended  reward={result['cumulative_reward']:+.1f}"
    else:
        verdict = f"⏱  cap hit  reward={result['cumulative_reward']:+.1f}"
    print(
        f"{verdict}  |  {result['decisions_total']} LLM decisions across "
        f"{result['n_agents']} agent(s)  |  "
        f"{result['env_ticks']} env ticks  |  {wall:.1f}s wall"
    )
    if result["decisions_total"] > 0:
        avg_llm = sum(d["llm_seconds"] for d in result["log"]) / result["decisions_total"]
        avg_lag = sum(d["lag_ticks"] for d in result["log"]) / result["decisions_total"]
        print(f"avg LLM latency: {avg_llm:.2f}s  |  avg obs->action lag: {avg_lag:.1f} ticks")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
