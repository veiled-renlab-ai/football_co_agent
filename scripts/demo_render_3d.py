"""Phase 4 demo with REAL-TIME 3D window via WSLg + ASYNC LLM.

The env ticks at gfootball's native rate on the main thread (3D window
stays smooth). The LLM decides in a background thread; new skills are
swapped in atomically. Between LLM decisions, a 'body rest-state'
fallback (slow dribble / jog) keeps the player moving — mirrors how
real footballers keep moving while their brain considers the next play.

Run from WSL with venv active:
    python3 -u scripts/demo_render_3d.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.agent import LLMPlayer
from football_agents.async_runner import AsyncRunner
from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient

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
    print("Phase 4 demo — REAL-TIME 3D + ASYNC LLM")
    print("Watch your Windows desktop — gfootball window should pop up.")
    print("=" * 78)

    client = LLMClient.from_env()
    print(f"LLM model: {client.model}\n")

    # Custom scenario llm_solo_attack: 3000 ticks, no premature termination.
    # Multi-agent control: explicitly drive slot 1 (the attacker), slot 0
    # (the GK) gets IDLE so it doesn't wander.
    env = FootballEnvAdapter(
        scenario="llm_solo_attack",
        render=True,
        n_controlled_left=2,
        primary_player_slot=1,
    )
    obs = env.reset()
    pid = obs.self_state.player_id
    role = obs.self_state.role
    print(f"Controlling player #{pid} ({role})\n")

    agent = LLMPlayer(player_id=pid, role=role, llm_client=client)

    def on_decision(log: dict) -> None:
        last = agent.history[-1] if agent.history else None
        reasoning = last.reasoning if last and last.reasoning else ""
        err = last.error if last and last.error else None
        print("─" * 78)
        print(f"#{log['decision']:>2}  env_tick={log['env_tick']:>4d}  "
              f"obs_tick={log['obs_tick']:>4d}  lag={log['lag_ticks']:>3d}t  "
              f"llm={log['llm_seconds']:.1f}s")
        if reasoning:
            print(f"  💭 {reasoning}")
        if err:
            print(f"  ⚠️  {err}  →  fallback {short_skill(log['skill'])}")
        else:
            print(f"  🎯 {short_skill(log['skill'])}")
        s = log_obs = env.observe()
        ss = s.self_state
        ball_str = f"my_ball={ss.has_ball}"
        if (b := s.ball()) is not None:
            ball_str += f"  ball@({b.position.x:+.2f},{b.position.y:+.2f}) d={b.distance:.2f}"
        print(f"     reward={env.cumulative_reward:+.1f}  "
              f"pos=({ss.position.x:+.2f},{ss.position.y:+.2f}) {ball_str}")

    runner = AsyncRunner(
        env=env,
        agent=agent,
        # fallback_policy=None → uses body_rest_state_fallback by default
        # (slow dribble if has-ball, slow jog to ball otherwise)
        obs_refresh_every_ticks=4,
        max_decisions=200,
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
        f"{verdict}  |  {result['decisions']} LLM decisions  |  "
        f"{result['env_ticks']} env ticks  |  {wall:.1f}s wall"
    )
    if result["decisions"] > 0:
        avg_llm = sum(d["llm_seconds"] for d in result["log"]) / result["decisions"]
        avg_lag = sum(d["lag_ticks"] for d in result["log"]) / result["decisions"]
        print(f"avg LLM latency: {avg_llm:.2f}s  |  avg obs->action lag: {avg_lag:.1f} ticks")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
