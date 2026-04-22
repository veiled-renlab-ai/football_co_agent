"""Diagnostic — what does an academy scenario actually look like at reset
and how long does it run with IDLE actions only? Helps figure out why
demos are ending early.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from gfootball.env import create_environment

SCENARIOS = [
    "academy_empty_goal_close",
    "academy_empty_goal",
    "academy_run_to_score",
    "academy_run_to_score_with_keeper",
    "academy_3_vs_1_with_keeper",
]


def inspect(scenario: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {scenario}")
    print('=' * 70)

    env = create_environment(env_name=scenario, representation="raw", render=False)
    obs = env.reset()
    if isinstance(obs, list):
        obs = obs[0]

    n_left = len(obs["left_team"])
    n_right = len(obs["right_team"])
    print(f"  left team:  {n_left} players")
    print(f"  right team: {n_right} players")
    print(f"  active player (controlled): #{obs['active']}  "
          f"role_id={obs['left_team_roles'][obs['active']]}")
    print(f"  designated:                #{obs['designated']}")
    print(f"  ball_owned_team:           {obs['ball_owned_team']}  "
          f"(0=left, 1=right, -1=none)")
    print(f"  ball_owned_player:         {obs['ball_owned_player']}")
    print(f"  steps_left at reset:       {obs['steps_left']}")
    print(f"  ball position:             ({obs['ball'][0]:+.2f}, {obs['ball'][1]:+.2f}, {obs['ball'][2]:+.2f})")
    print()
    print("  left team positions (x, y):")
    for i, p in enumerate(obs["left_team"]):
        role_id = int(obs["left_team_roles"][i])
        marker = " ← active" if i == obs["active"] else ""
        carrier = "  [BALL]" if (obs["ball_owned_team"] == 0 and obs["ball_owned_player"] == i) else ""
        print(f"    #{i}  role={role_id}  pos=({p[0]:+.2f}, {p[1]:+.2f}){marker}{carrier}")
    print("  right team positions (x, y):")
    for i, p in enumerate(obs["right_team"]):
        role_id = int(obs["right_team_roles"][i])
        carrier = "  [BALL]" if (obs["ball_owned_team"] == 1 and obs["ball_owned_player"] == i) else ""
        print(f"    #{i}  role={role_id}  pos=({p[0]:+.2f}, {p[1]:+.2f}){carrier}")

    # Step with IDLE only, see how long episode lasts
    print()
    print(f"  Now stepping with IDLE only...")
    ticks = 0
    while ticks < 500:
        result = env.step(0)  # IDLE
        if len(result) == 5:
            _, _, term, trunc, _ = result
            done = term or trunc
        else:
            _, _, done, _ = result
        ticks += 1
        if done:
            break
    print(f"  Episode ended after {ticks} ticks of IDLE (max 500)")

    env.close()


def main():
    for s in SCENARIOS:
        try:
            inspect(s)
        except Exception as e:
            print(f"\n{s}: FAILED - {e!r}")


if __name__ == "__main__":
    main()
