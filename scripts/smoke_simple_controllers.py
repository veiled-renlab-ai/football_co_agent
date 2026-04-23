"""Verify simplified MoveTo/DribbleToward controllers — smooth continuous motion.

After reverting walk-cycle and burst-decay, the controllers should be:
  MoveTo:        tick 1 = SPRINT/RELEASE_SPRINT, tick 2+ = direction (continuous)
  DribbleToward: tick 1 = DRIBBLE, tick 2 = SPRINT/RELEASE_SPRINT, tick 3+ = direction

No more push/release/idle×4 patterns, no more 10-tick burst-then-walk decay.
The agent moves smoothly at the LLM-chosen urgency for the entire decision
window. Speed is governed by env-level real-time pacing (pps=2 + 50fps wall).

walk urgency: gfootball has no native walk action — base speed is jog
(no SPRINT). 'walk' here is treated identically to 'jog' (RELEASE_SPRINT).

Run: wsl ... python3 -m scripts.smoke_simple_controllers
"""
from __future__ import annotations

from football_agents.motor import (
    A, DribbleTowardController, MoveToController, make_controller,
)
from football_agents.skills import DribbleToward, MoveTo

ACTION_NAME = {
    A.IDLE: "IDLE", A.LEFT: "LEFT", A.RIGHT: "RIGHT",
    A.SPRINT: "SPR", A.RELEASE_SPRINT: "rSPR",
    A.RELEASE_DIRECTION: "rDIR", A.DRIBBLE: "DRB", A.RELEASE_DRIBBLE: "rDRB",
}


def _obs_with_ball() -> dict:
    return {"active": 1, "left_team": [[0.0, 0.0]] * 11,
            "left_team_direction": [[0.0, 0.0]] * 11,
            "ball_owned_team": 0, "ball_owned_player": 1}


def _obs_no_ball() -> dict:
    return {"active": 1, "left_team": [[0.0, 0.0]] * 11,
            "left_team_direction": [[0.0, 0.0]] * 11,
            "ball_owned_team": -1, "ball_owned_player": -1}


def trace(controller, obs, n: int) -> list[str]:
    return [ACTION_NAME.get(controller.step(obs)[0], "?") for _ in range(n)]


def main() -> None:
    print("=" * 70)
    print("Simplified controllers — should be SMOOTH (no IDLE / rDIR stutters)")
    print("=" * 70)

    # MoveTo
    for urg, expect_setup in [("sprint", "SPR"), ("jog", "rSPR"), ("walk", "rSPR")]:
        skill = MoveTo(target_x=0.5, target_y=0.0, urgency=urg)
        c = make_controller(skill, team_side="left", player_id=1)
        seq = trace(c, _obs_no_ball(), 20)
        print(f"\n  MoveTo({urg!r:>8}): {' '.join(seq)}")
        assert seq[0] == expect_setup, f"tick 1 expected {expect_setup}, got {seq[0]}"
        # Ticks 2-20 must all be RIGHT (continuous direction press)
        for t in range(1, 20):
            assert seq[t] == "RIGHT", f"tick {t+1} should be RIGHT (continuous), got {seq[t]}"
        print(f"    OK: setup={seq[0]}, then 19 ticks of continuous direction press")

    print()
    # DribbleToward
    for urg, expect_setup2 in [("sprint", "SPR"), ("jog", "rSPR"), ("walk", "rSPR")]:
        skill = DribbleToward(target_x=0.5, target_y=0.0, urgency=urg)
        c = make_controller(skill, team_side="left", player_id=1)
        seq = trace(c, _obs_with_ball(), 20)
        print(f"  DribbleToward({urg!r:>8}): {' '.join(seq)}")
        assert seq[0] == "DRB", f"tick 1 should be DRB, got {seq[0]}"
        assert seq[1] == expect_setup2, f"tick 2 expected {expect_setup2}, got {seq[1]}"
        for t in range(2, 20):
            assert seq[t] == "RIGHT", f"tick {t+1} should be RIGHT (continuous), got {seq[t]}"
        print(f"    OK: DRB + {seq[1]} + 18 ticks of continuous direction press")

    print()
    print("=" * 70)
    print("All sequences are NOW smooth. No IDLE/rDIR stutters mid-motion.")
    print("=" * 70)


if __name__ == "__main__":
    main()
