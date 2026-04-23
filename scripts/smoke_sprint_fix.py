"""Smoke test: verify SPRINT / RELEASE_SPRINT logic in MoveTo + DribbleToward.

Walks through tick sequences and asserts the right gfootball action is
emitted at each tick for each urgency. Verifies the bug fix:
  - DribbleToward(sprint) now actually presses SPRINT (was missing before)
  - MoveTo(jog) now releases SPRINT (was leaking sticky from prior sprint)

Run via:
    wsl -d Ubuntu-22.04 -- bash -lc 'source ~/football-env/bin/activate && \\
        cd /mnt/c/Users/dfgfd/Desktop/football && python3 -m scripts.smoke_sprint_fix'
"""
from __future__ import annotations

from football_agents.motor import (
    A, DribbleTowardController, MoveToController, make_controller,
)
from football_agents.skills import DribbleToward, MoveTo

ACTION_NAME = {
    A.IDLE: "IDLE", A.LEFT: "LEFT", A.RIGHT: "RIGHT",
    A.SPRINT: "SPRINT", A.RELEASE_SPRINT: "RELEASE_SPRINT",
    A.RELEASE_DIRECTION: "RELEASE_DIR", A.DRIBBLE: "DRIBBLE",
    A.RELEASE_DRIBBLE: "RELEASE_DRIBBLE",
}


def _fake_obs_with_ball(player_id: int = 1) -> dict:
    """Minimal obs where player_id has the ball at origin."""
    return {
        "active": player_id,
        "left_team": [[0.0, 0.0]] * 11,
        "left_team_direction": [[0.0, 0.0]] * 11,
        "ball_owned_team": 0,
        "ball_owned_player": player_id,
    }


def _fake_obs_no_ball() -> dict:
    return {
        "active": 1,
        "left_team": [[0.0, 0.0]] * 11,
        "left_team_direction": [[0.0, 0.0]] * 11,
        "ball_owned_team": -1,
        "ball_owned_player": -1,
    }


def trace(controller, obs, n_ticks: int) -> list[str]:
    out = []
    for _ in range(n_ticks):
        action, _status = controller.step(obs)
        out.append(ACTION_NAME.get(action, f"?{action}"))
    return out


def main() -> None:
    print("=" * 60)
    print("MoveTo urgency tick sequences (target far away):")
    print("=" * 60)

    # MoveTo(sprint): tick 1 = SPRINT, tick 2+ = direction
    skill = MoveTo(target_x=0.5, target_y=0.0, urgency="sprint")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_no_ball(), 4)
    print(f"  sprint: {seq}")
    assert seq[0] == "SPRINT", f"tick 1 should be SPRINT, got {seq[0]}"
    assert seq[1] == "RIGHT", f"tick 2 should be RIGHT, got {seq[1]}"
    print("    OK: SPRINT pressed at tick 1, then direction")

    # MoveTo(jog): tick 1 = RELEASE_SPRINT (was missing before!)
    skill = MoveTo(target_x=0.5, target_y=0.0, urgency="jog")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_no_ball(), 4)
    print(f"  jog:    {seq}")
    assert seq[0] == "RELEASE_SPRINT", f"tick 1 should be RELEASE_SPRINT, got {seq[0]}"
    assert seq[1] == "RIGHT", f"tick 2 should be RIGHT, got {seq[1]}"
    print("    OK: RELEASE_SPRINT at tick 1 (clears leftover sticky)")

    # MoveTo(walk): tick 1 = RELEASE_SPRINT, then walk cycle from tick 2
    skill = MoveTo(target_x=0.5, target_y=0.0, urgency="walk")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_no_ball(), 8)
    print(f"  walk:   {seq}")
    assert seq[0] == "RELEASE_SPRINT", f"tick 1 should be RELEASE_SPRINT for walk, got {seq[0]}"
    # Walk cycle from tick 2: phase 0 = direction, phase 1 = RELEASE_DIR, 2-5 = IDLE
    assert seq[1] == "RIGHT", f"walk tick 2 should push direction, got {seq[1]}"
    assert seq[2] == "RELEASE_DIR", f"walk tick 3 should release dir, got {seq[2]}"
    assert seq[3] == "IDLE" and seq[4] == "IDLE", "walk should idle for 4 ticks"
    print("    OK: walk cycle preserved (RELEASE_SPRINT + step+release+idle×4)")

    print()
    print("=" * 60)
    print("DribbleToward urgency tick sequences (has ball):")
    print("=" * 60)

    # DribbleToward(sprint): tick 1 = DRIBBLE, tick 2 = SPRINT, tick 3+ = direction
    # ↑ THIS IS THE BUG FIX — previously SPRINT was never pressed
    skill = DribbleToward(target_x=0.5, target_y=0.0, urgency="sprint")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_with_ball(), 4)
    print(f"  sprint: {seq}")
    assert seq[0] == "DRIBBLE", f"tick 1 should be DRIBBLE, got {seq[0]}"
    assert seq[1] == "SPRINT", f"tick 2 should be SPRINT (BUG FIX!), got {seq[1]}"
    assert seq[2] == "RIGHT", f"tick 3 should be RIGHT, got {seq[2]}"
    print("    OK: DRIBBLE + SPRINT both pressed (bug fix verified)")

    # DribbleToward(jog): tick 1 = DRIBBLE, tick 2 = RELEASE_SPRINT, tick 3+ = direction
    skill = DribbleToward(target_x=0.5, target_y=0.0, urgency="jog")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_with_ball(), 4)
    print(f"  jog:    {seq}")
    assert seq[0] == "DRIBBLE", f"tick 1 should be DRIBBLE, got {seq[0]}"
    assert seq[1] == "RELEASE_SPRINT", f"tick 2 should be RELEASE_SPRINT, got {seq[1]}"
    assert seq[2] == "RIGHT", f"tick 3 should be RIGHT, got {seq[2]}"
    print("    OK: DRIBBLE + RELEASE_SPRINT (clears sticky)")

    # DribbleToward(walk): tick 1=DRIBBLE, tick 2=RELEASE_SPRINT, tick 3+=walk cycle
    skill = DribbleToward(target_x=0.5, target_y=0.0, urgency="walk")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_with_ball(), 9)
    print(f"  walk:   {seq}")
    assert seq[0] == "DRIBBLE"
    assert seq[1] == "RELEASE_SPRINT"
    # Walk cycle starting at tick 3 (phase 0): direction, then release, then idle×4
    assert seq[2] == "RIGHT", f"walk tick 3 should push direction, got {seq[2]}"
    assert seq[3] == "RELEASE_DIR", f"walk tick 4 should release dir, got {seq[3]}"
    assert seq[4] == "IDLE" and seq[5] == "IDLE", "walk should idle"
    print("    OK: DRIBBLE + RELEASE_SPRINT + walk cycle")

    print()
    print("--- All sprint/release_sprint logic verified ---")


if __name__ == "__main__":
    main()
