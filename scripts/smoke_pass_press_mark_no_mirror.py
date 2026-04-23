"""Verify PassTo/Press/Mark for both teams use left_team/right_team
literals correctly under per-slot view convention.
"""
from football_agents.motor import (
    A, ACTION_NAMES, PassToController, PressController, MarkController,
    make_controller,
)
from football_agents.skills import PassTo, Press, Mark


def fake_obs(self_pos, ball_owner_team=-1, ball_owner_player=-1):
    return {
        "active": 1,
        "left_team":           [[0,0]] + [list(self_pos)] + [[0.3, 0.0]] + [[-0.3, 0]] + [[0,0]]*7,
        "left_team_direction": [[0,0]]*11,
        "right_team":          [[0,0]] + [[0.5, 0.1]] + [[0.4, -0.1]] + [[0,0]]*8,
        "right_team_direction": [[0,0]]*11,
        "ball": [0.0, 0.0, 0.1],
        "ball_direction": [0,0,0],
        "ball_owned_team": ball_owner_team,
        "ball_owned_player": ball_owner_player,
    }


# Test 1: PassTo target_player_id=2 — should look up MY teammate (left_team[2])
# In slot view, my teammate at left_team[2] = (0.3, 0.0). Pass action targets that direction.
for side in ["left", "right"]:
    skill = PassTo(target_player_id=2, pass_type="short")
    c = make_controller(skill, team_side=side, player_id=1)
    obs = fake_obs((0.0, 0.0), ball_owner_team=0, ball_owner_player=1)
    action, _ = c.step(obs)  # tick 1: face teammate
    # From (0,0) toward (0.3, 0.0) → RIGHT
    assert action == A.RIGHT, f"team_side={side}: expected RIGHT, got {ACTION_NAMES[action]}"
    print(f"OK: PassTo for team_side={side} faces teammate at left_team[2]={ACTION_NAMES[action]}")


# Test 2: Press opponent_id=1 — should sprint at MY opponent (right_team[1])
# In slot view, opponent at right_team[1] = (0.5, 0.1).
for side in ["left", "right"]:
    skill = Press(opponent_id=1)
    c = make_controller(skill, team_side=side, player_id=1)
    obs = fake_obs((0.0, 0.0))
    c.step(obs)  # tick 1: SPRINT
    action, _ = c.step(obs)  # tick 2: direction
    # From (0,0) toward (0.5, 0.1) → mostly RIGHT, slightly down
    assert action in (A.RIGHT, A.BOTTOM_RIGHT), f"team_side={side}: expected RIGHT-ish, got {ACTION_NAMES[action]}"
    print(f"OK: Press for team_side={side} sprints at opp at right_team[1]={ACTION_NAMES[action]}")


# Test 3: Mark opponent_id=1 — shadow goal-side
for side in ["left", "right"]:
    skill = Mark(opponent_id=1)
    c = make_controller(skill, team_side=side, player_id=1)
    obs = fake_obs((0.0, 0.0))
    action, _ = c.step(obs)
    # opp at (0.5, 0.1), own_goal_x=-1, SHADOW_OFFSET=0.10
    # target_x = 0.5 - 0.10 = 0.4, target_y = 0.1
    # From (0, 0) toward (0.4, 0.1) → RIGHT or BOTTOM_RIGHT
    assert action in (A.RIGHT, A.BOTTOM_RIGHT), f"team_side={side}: got {ACTION_NAMES[action]}"
    print(f"OK: Mark for team_side={side} moves to goal-side of opp={ACTION_NAMES[action]}")


print("\n--- PassTo/Press/Mark all use uniform slot-view convention ---")
