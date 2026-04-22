"""Phase 4 demo — one LLM-driven player on academy_empty_goal_close.

THE FIRST VISIBLE DEMO: shows an LLM perceiving the field, reasoning about it,
choosing a Skill, and the engine executing it.

Run from project root with venv active:
    python3 scripts/demo_single_agent.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.agent import LLMPlayer
from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient
from football_agents.perception import EgocentricFilter

# Quiet down the verbose dependencies; keep our own warnings visible.
logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


def short_skill(skill) -> str:
    name = type(skill).__name__
    args = []
    for slot in ("target_x", "target_y", "target_player_id", "target_zone",
                 "pass_type", "urgency", "opponent_id", "entity_id", "audience"):
        if hasattr(skill, slot):
            v = getattr(skill, slot)
            if isinstance(v, float):
                args.append(f"{slot}={v:+.2f}")
            else:
                args.append(f"{slot}={v}")
    if hasattr(skill, "message"):
        args.append(f"msg={skill.message!r}")
    return f"{name}({', '.join(args)})"


def main() -> None:
    print("=" * 78)
    print("Phase 4 demo — LLM agent on academy_empty_goal_close")
    print("=" * 78)

    # 1. Build LLM client from .env (uses 火山方舟 by default)
    client = LLMClient.from_env()
    print(f"LLM model: {client.model}")
    print(f"Endpoint:  {client.base_url}")
    print()

    # 2. Build env
    env = FootballEnvAdapter(scenario="academy_empty_goal_close", render=False)
    obs = env.reset()
    pid = obs.self_state.player_id
    role = obs.self_state.role
    print(f"Controlling player #{pid} ({role})")
    print()

    # 3. Build agent for this player
    agent = LLMPlayer(player_id=pid, role=role, llm_client=client)

    # 4. Loop
    decision_count = 0
    max_decisions = 30
    t0 = time.monotonic()

    while not env.done and decision_count < max_decisions:
        # If gfootball auto-switched to a different player, refresh agent role
        if obs.self_state.player_id != agent.player_id or obs.self_state.role != agent.role:
            agent.player_id = obs.self_state.player_id
            agent.update_role(obs.self_state.role)

        decision_count += 1
        t_dec = time.monotonic()
        skill = agent.choose_skill(obs)
        dec_dt = time.monotonic() - t_dec

        last = agent.history[-1]
        print("─" * 78)
        print(f"Decision #{decision_count}  (tick={env.tick}, LLM took {dec_dt:.1f}s)")
        if last.reasoning:
            print(f"  💭 Reasoning: {last.reasoning}")
        if last.error:
            print(f"  ⚠️  Error: {last.error}  → fallback to {short_skill(skill)}")
        else:
            print(f"  🎯 Skill:    {short_skill(skill)}")

        status = env.dispatch_skill(skill, max_env_ticks=8)
        obs = env.observe()

        s = obs.self_state
        ball_str = f"my_ball={s.has_ball}"
        if (b := obs.ball()) is not None:
            ball_str += f"  ball@({b.position.x:+.2f},{b.position.y:+.2f}) d={b.distance:.2f}"
        print(f"  → {status}  reward={env.cumulative_reward:+.1f}  "
              f"pid={s.player_id} role={s.role} pos=({s.position.x:+.2f},{s.position.y:+.2f}) "
              f"{ball_str}")

    total_dt = time.monotonic() - t0

    print()
    print("=" * 78)
    if env.cumulative_reward > 0:
        print(f"✅ GOAL!  reward={env.cumulative_reward:+.1f}  "
              f"in {decision_count} LLM decisions ({env.tick} env ticks, {total_dt:.1f}s wall)")
    elif env.done:
        print(f"⏹  Episode ended  reward={env.cumulative_reward:+.1f}  "
              f"in {decision_count} decisions ({total_dt:.1f}s wall)")
    else:
        print(f"⏱  Decision cap hit (no goal)  reward={env.cumulative_reward:+.1f}  "
              f"in {total_dt:.1f}s wall")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
