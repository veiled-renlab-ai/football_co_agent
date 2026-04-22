"""Phase 3 verification — run a scripted policy through a full episode of
academy_empty_goal_close to confirm env -> perception -> motor -> action loop.

Policy: 'shoot if close to goal with ball; else dribble toward goal if has ball;
        else sprint to ball'.

Run from project root with venv active:
    python3 scripts/verify_phase3.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.env import FootballEnvAdapter
from football_agents.perception import Observation
from football_agents.skills import DribbleToward, MoveTo, Shoot, Skill


# ---------------------------------------------------------------------------
# Tiny rule-based policy — placeholder for the LLM that lands in Phase 4.
# ---------------------------------------------------------------------------

def scripted_policy(obs: Observation) -> Skill:
    self_pos = obs.self_state.position
    has_ball = obs.self_state.has_ball

    # If we have the ball, decide between shoot vs dribble
    if has_ball:
        # Distance to opponent goal at x = +1 (left team is attacking)
        dist_to_goal = math.hypot(1.0 - self_pos.x, 0.0 - self_pos.y)
        if dist_to_goal < 0.3:
            return Shoot(target_zone="top_center")
        return DribbleToward(target_x=0.85, target_y=0.0)

    # No ball: chase it
    ball = obs.ball()
    if ball is None:
        # Ball not in our perception (shouldn't really happen with v0 sight range)
        return MoveTo(target_x=0.5, target_y=0.0, urgency="sprint")
    return MoveTo(target_x=ball.position.x, target_y=ball.position.y, urgency="sprint")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def short_obs(obs: Observation) -> str:
    s = obs.self_state
    parts = [
        f"pid={s.player_id}",
        f"role={s.role}",
        f"pos=({s.position.x:+.2f},{s.position.y:+.2f})",
        f"face={s.facing_deg:+.0f}°",
        f"ball={s.has_ball}",
    ]
    return " ".join(parts)


def short_skill(skill: Skill) -> str:
    name = type(skill).__name__
    if hasattr(skill, "target_x"):
        return f"{name}(x={skill.target_x:+.2f}, y={skill.target_y:+.2f})"  # type: ignore[attr-defined]
    if hasattr(skill, "target_zone"):
        return f"{name}(zone={skill.target_zone})"  # type: ignore[attr-defined]
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Phase 3 verification — scripted agent on academy_empty_goal_close")
    print("=" * 70)

    env = FootballEnvAdapter(scenario="academy_empty_goal_close", render=False)
    obs = env.reset()

    print()
    print(f"Initial: {short_obs(obs)}")
    print(f"Perceived entities ({len(obs.perceived_entities)}):")
    for e in obs.perceived_entities:
        v = f" vel=({e.velocity.x:+.2f},{e.velocity.y:+.2f})" if e.velocity else " vel=?"
        flag = " [BALL CARRIER]" if e.has_ball else ""
        print(f"  - {e.role:8s} #{e.entity_id:2d}  pos=({e.position.x:+.2f},{e.position.y:+.2f})  "
              f"d={e.distance:.2f}{v}{flag}")

    print()
    print("─" * 70)
    print(f"{'#':>3} {'tick':>4} {'skill':32s} {'->status':12s} {'reward':>7} {'state'}")
    print("─" * 70)

    decision_count = 0
    while not env.done and decision_count < 60:
        skill = scripted_policy(obs)
        decision_count += 1
        status = env.dispatch_skill(skill, max_env_ticks=8)
        obs = env.observe()
        print(f"{decision_count:>3} {env.tick:>4} {short_skill(skill):32s} "
              f"->{status:11s} {env.cumulative_reward:>+7.2f} {short_obs(obs)}")

    print()
    print("=" * 70)
    if env.cumulative_reward > 0:
        print(f"✅ GOAL  reward={env.cumulative_reward:+.1f}  decisions={decision_count}  ticks={env.tick}")
    elif env.done:
        print(f"⏹  episode ended  reward={env.cumulative_reward:+.1f}  decisions={decision_count}")
    else:
        print(f"⏱  decision cap hit  reward={env.cumulative_reward:+.1f}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
