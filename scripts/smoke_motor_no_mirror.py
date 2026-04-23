"""Verify motor controllers no longer have team_side-dependent coord logic."""
from football_agents.motor import (
    A, MotorController, MoveToController, DribbleTowardController,
    ShootController, make_controller,
)
from football_agents.skills import MoveTo, DribbleToward, Shoot

# Helper to build an obs where the self player is at given pos
def obs_at(pos_x, pos_y, has_ball=False):
    return {
        "active": 1,
        "left_team": [[0,0]] + [[pos_x, pos_y]] + [[0,0]]*9,
        "left_team_direction": [[0,0]]*11,
        "right_team": [[0,0]]*11,
        "right_team_direction": [[0,0]]*11,
        "ball_owned_team": 0 if has_ball else -1,
        "ball_owned_player": 1 if has_ball else -1,
    }

# Test 1: MoveTo target_x=+0.5 -> motor should send action toward +x
# regardless of team_side (because slot view is uniform)
for side in ["left", "right"]:
    skill = MoveTo(target_x=0.5, target_y=0.0, urgency="jog")
    c = make_controller(skill, team_side=side, player_id=1)
    c.step(obs_at(0, 0))  # tick 1: RELEASE_SPRINT
    action, _ = c.step(obs_at(0, 0))  # tick 2: should be RIGHT
    assert action == A.RIGHT, f"team_side={side}: expected RIGHT, got {action}"
    print(f"OK: MoveTo(target_x=+0.5) for team_side={side} -> RIGHT (toward +x)")

# Test 2: Shoot zone "top_left" - both teams should aim same way (slot view)
for side in ["left", "right"]:
    skill = Shoot(target_zone="top_left")
    c = make_controller(skill, team_side=side, player_id=1)
    obs = obs_at(0.7, 0, has_ball=True)
    action, _ = c.step(obs)  # tick 1: face goal at (1.0, -0.04)
    # From (0.7, 0) toward (1.0, -0.04): mostly RIGHT, slightly TOP
    print(f"  Shoot top_left from (0.7, 0) for team={side}: action={action}")
    # Both teams should pick the SAME action (because logic is now team-agnostic)

# Test 3: _opponent_goal_x returns 1.0 always
mc1 = make_controller(MoveTo(target_x=0, target_y=0, urgency="jog"), team_side="left", player_id=1)
mc2 = make_controller(MoveTo(target_x=0, target_y=0, urgency="jog"), team_side="right", player_id=1)
assert mc1._opponent_goal_x() == 1.0
assert mc2._opponent_goal_x() == 1.0
print("OK: _opponent_goal_x() returns 1.0 for both teams")

# Test 4: _selfframe_target_to_abs is gone
import inspect
methods = [m for m in dir(MotorController) if not m.startswith("__")]
assert "_selfframe_target_to_abs" not in methods, f"method should be deleted: {methods}"
print("OK: _selfframe_target_to_abs removed")

print("--- motor.py mirror compensation removed ---")
