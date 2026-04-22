"""Phase 2 verification — feed each motor controller a synthetic gfootball
obs dict and print the action sequence it produces, simulating naive
forward-Euler movement so the controller's "arrived at target" logic
actually fires.

Run from project root with venv active:
    python3 scripts/verify_motor.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Add project root to sys.path so `football_agents` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from football_agents.motor import ACTION_NAMES, A, make_controller, vector_to_action
from football_agents.skills import (
    DribbleToward, HoldPosition, MoveTo, PassTo, Shoot,
)


# Approximate per-tick movement step for synthetic obs simulation.
# Real gfootball is faster but this gives readable output.
SIM_SPEED_PER_TICK = 0.04
SIM_SPRINT_MULTIPLIER = 1.6

# Action-id -> (dx, dy) sign for movement simulation
_ACTION_DELTA = {
    A.LEFT: (-1, 0), A.TOP_LEFT: (-1, -1), A.TOP: (0, -1), A.TOP_RIGHT: (1, -1),
    A.RIGHT: (1, 0), A.BOTTOM_RIGHT: (1, 1), A.BOTTOM: (0, 1), A.BOTTOM_LEFT: (-1, 1),
}


def make_obs(self_pos, *, ball_owned=False, teammate_pos=None):
    """Build a minimal synthetic gfootball obs dict for a left-team player at index 0."""
    if teammate_pos is not None:
        left_team = np.array([self_pos, teammate_pos], dtype=np.float64)
    else:
        left_team = np.array([self_pos], dtype=np.float64)

    return {
        "active": 0,
        "left_team": left_team,
        "left_team_direction": np.zeros_like(left_team),
        "right_team": np.array([[1.0, 0.0]], dtype=np.float64),
        "right_team_direction": np.zeros((1, 2), dtype=np.float64),
        "ball": np.array([self_pos[0], self_pos[1], 0.0], dtype=np.float64),
        "ball_direction": np.zeros(3, dtype=np.float64),
        "ball_owned_team": 0 if ball_owned else -1,
        "ball_owned_player": 0 if ball_owned else -1,
        "score": [0, 0],
        "sticky_actions": np.zeros(10, dtype=np.uint8),
        "game_mode": 0,
        "steps_left": 400,
    }


def run(label, skill, *, self_pos, ball_owned=False, teammate_pos=None,
        max_ticks=30, simulate_movement=True):
    print()
    print("─" * 60)
    print(f"▶ {label}")
    print(f"  Skill: {skill}")
    print(f"  Start: pos=({self_pos[0]:+.2f}, {self_pos[1]:+.2f})  "
          f"has_ball={ball_owned}  teammate={teammate_pos}")
    print()

    ctrl = make_controller(skill)
    pos = list(self_pos)
    sprinting = False

    for t in range(1, max_ticks + 1):
        obs = make_obs(pos, ball_owned=ball_owned, teammate_pos=teammate_pos)
        action, status = ctrl.step(obs)
        name = ACTION_NAMES[action]
        marker = "✓" if status == "completed" else ("✗" if status == "failed" else "·")
        print(f"  t={t:2d}  pos=({pos[0]:+.2f}, {pos[1]:+.2f})  "
              f"-> action={action:2d} {name:14s}  status={status}  {marker}")
        if status != "in_progress":
            break

        # Naive movement simulation
        if simulate_movement:
            if action == A.SPRINT:
                sprinting = True
            elif action in _ACTION_DELTA:
                dx, dy = _ACTION_DELTA[action]
                speed = SIM_SPEED_PER_TICK * (SIM_SPRINT_MULTIPLIER if sprinting else 1.0)
                # Diagonal: scale so total speed is consistent
                norm = math.hypot(dx, dy) or 1.0
                pos[0] += (dx / norm) * speed
                pos[1] += (dy / norm) * speed


def main() -> None:
    print("=" * 60)
    print("Phase 2 motor controller verification")
    print("=" * 60)

    # Case 1: walk from left side to right side, jog
    run(
        "MoveTo (jog) — across the field",
        MoveTo(target_x=0.5, target_y=0.0),
        self_pos=(-0.5, 0.0),
        max_ticks=30,
    )

    # Case 2: sprint diagonally
    run(
        "MoveTo (sprint) — diagonal upper-right",
        MoveTo(target_x=0.3, target_y=-0.3, urgency="sprint"),
        self_pos=(0.0, 0.0),
        max_ticks=20,
    )

    # Case 3: dribble forward with ball
    run(
        "DribbleToward — carry ball to (0.7, 0.0)",
        DribbleToward(target_x=0.7, target_y=0.0),
        self_pos=(0.4, 0.1),
        ball_owned=True,
        max_ticks=15,
    )

    # Case 4: dribble FAILS if we don't have the ball
    run(
        "DribbleToward — without possession (should fail)",
        DribbleToward(target_x=0.7, target_y=0.0),
        self_pos=(0.4, 0.1),
        ball_owned=False,
        max_ticks=3,
    )

    # Case 5: short pass to teammate at (0.5, 0.3)
    run(
        "PassTo — short pass to teammate #1",
        PassTo(target_player_id=1, pass_type="short"),
        self_pos=(0.0, 0.0),
        ball_owned=True,
        teammate_pos=(0.5, 0.3),
        max_ticks=5,
        simulate_movement=False,  # passes don't move you
    )

    # Case 6: long pass to a far teammate (top-left)
    run(
        "PassTo — long pass to teammate #1 in top-left",
        PassTo(target_player_id=1, pass_type="long"),
        self_pos=(0.2, 0.1),
        ball_owned=True,
        teammate_pos=(-0.5, -0.3),
        max_ticks=5,
        simulate_movement=False,
    )

    # Case 7: shoot at top_right of opponent goal
    run(
        "Shoot — top_right zone",
        Shoot(target_zone="top_right"),
        self_pos=(0.85, 0.0),
        ball_owned=True,
        max_ticks=5,
        simulate_movement=False,
    )

    # Case 8: shoot fails without ball
    run(
        "Shoot — without possession (should fail)",
        Shoot(target_zone="top_center"),
        self_pos=(0.85, 0.0),
        ball_owned=False,
        max_ticks=3,
        simulate_movement=False,
    )

    # Case 9: HoldPosition — release direction then idle
    run(
        "HoldPosition",
        HoldPosition(),
        self_pos=(0.0, 0.0),
        max_ticks=3,
        simulate_movement=False,
    )

    print()
    print("=" * 60)
    print("Quick unit tests for vector_to_action:")
    print("=" * 60)
    cases = [
        (1.0, 0.0, "RIGHT"),
        (-1.0, 0.0, "LEFT"),
        (0.0, 1.0, "BOTTOM"),
        (0.0, -1.0, "TOP"),
        (1.0, 1.0, "BOTTOM_RIGHT"),
        (-1.0, -1.0, "TOP_LEFT"),
        (0.0, 0.0, "IDLE"),
    ]
    for dx, dy, expected in cases:
        a = vector_to_action(dx, dy)
        ok = ACTION_NAMES[a] == expected
        sign = "✓" if ok else "✗"
        print(f"  vector_to_action({dx:+.0f}, {dy:+.0f}) = {ACTION_NAMES[a]:14s} "
              f"(expected {expected})  {sign}")

    print()
    print("=" * 60)
    print("Phase 2 motor verification complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
