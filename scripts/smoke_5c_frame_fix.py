"""Verify the two self-frame coordinate fixes:

  Bug #1 (perception.py): TeamMessageBus listener_position frame mismatch.
    For team_b agents, self_pos is in self-frame (mirrored), but the bus
    stores sender_position in absolute. Without the fix, distance is
    computed across mismatched frames and audience='nearby' rejects every
    nearby message for the right team.

  Bug #2 (motor.py): ShootController ZONE_Y_BIAS not mirrored for team_b.
    The LLM picks "top_*" expecting its visual top, but team_b's visual
    top is absolute +y while ZONE_Y_BIAS["top_*"] = -0.04 absolute → the
    shot biases toward the wrong corner.

Pure-Python smoke test: no gfootball, no LLM. Run with the project's
Python (the one that has football_agents importable):

    python -m scripts.smoke_5c_frame_fix
"""
from __future__ import annotations

from football_agents.message_bus import Message, TeamMessageBus
from football_agents.motor import A, ACTION_NAMES, ShootController
from football_agents.perception import EgocentricFilter, Vec2
from football_agents.skills import Shoot


# ---------------------------------------------------------------------------
# Fake gfootball god-views / raw_obs
# ---------------------------------------------------------------------------

def _empty_team_arrays(n: int = 5) -> dict:
    """Stand-in arrays for one team (positions, directions, roles, tired)."""
    return {
        "positions": [[0.0, 0.0]] * n,
        "directions": [[0.0, 0.0]] * n,
        "roles": [0] * n,
        "tired": [0.0] * n,
    }


def build_god_view_red_pair(sender_pos_abs: tuple[float, float],
                            listener_pos_abs: tuple[float, float]) -> dict:
    """God-view with two right-team (team_b) players at the given absolute
    positions: index 1 = sender, index 2 = listener. Other slots placeholder.

    Left team is filled with neutral positions far from the action. Ball
    nowhere relevant.
    """
    right_positions = [[0.0, 0.0]] * 5
    right_positions[1] = list(sender_pos_abs)
    right_positions[2] = list(listener_pos_abs)
    return {
        "active": 0,
        "left_team": [[-0.9, 0.0]] * 5,
        "left_team_direction": [[0.0, 0.0]] * 5,
        "left_team_roles": [0, 1, 2, 4, 5],
        "left_team_tired_factor": [0.0] * 5,
        "right_team": right_positions,
        "right_team_direction": [[0.0, 0.0]] * 5,
        "right_team_roles": [0, 1, 2, 4, 5],
        "right_team_tired_factor": [0.0] * 5,
        "ball": [0.0, 0.0],
        "ball_direction": [0.0, 0.0],
        "ball_owned_team": -1,
        "ball_owned_player": -1,
        "score": [0, 0],
    }


def build_obs_with_red_player_at(idx: int, pos_abs: tuple[float, float]) -> dict:
    """Minimal raw_obs sufficient for ShootController._self_pos_vel +
    ShootController._has_ball, with right-team player `idx` at `pos_abs`.
    Caller can override ball_owned_* afterward.
    """
    rt = [[0.0, 0.0]] * 5
    rt[idx] = list(pos_abs)
    return {
        "active": idx,
        "left_team": [[-0.9, 0.0]] * 5,
        "left_team_direction": [[0.0, 0.0]] * 5,
        "right_team": rt,
        "right_team_direction": [[0.0, 0.0]] * 5,
        "ball": [0.0, 0.0],
        "ball_direction": [0.0, 0.0],
        "ball_owned_team": -1,
        "ball_owned_player": -1,
    }


# ---------------------------------------------------------------------------
# Bug #1 — perception.py / TeamMessageBus frame mismatch
# ---------------------------------------------------------------------------

def test_bug1_team_b_nearby_call_received() -> None:
    """team_b sender at absolute (+0.5, +0.10), team_b listener at
    absolute (+0.5, +0.18). True nearby distance = 0.08 (well inside
    NEARBY_RADIUS = 0.20).

    Pre-fix: listener_pos passed to bus was self-frame (-0.5, -0.18);
    sender_position in bus was absolute (+0.5, +0.10); cross-frame
    distance ≈ 1.04 → REJECTED. Listener hears nothing.

    Post-fix: perception un-mirrors listener_pos to absolute before the
    bus read; distance = 0.08 → ACCEPTED. Then mirrors sender_position
    back to self-frame so prompts.py renders zones from listener's POV.
    """
    bus = TeamMessageBus()

    # Sender posts in absolute coords (matches what player_agent.py does).
    msg = Message(
        sender_player_id=1, sender_jersey=7,
        sender_position=Vec2(0.5, 0.10),
        message="传给我",
        audience="nearby",
        tick_posted=0,
    )
    bus.post("right", msg)

    # Listener: team_b player at absolute (+0.5, +0.18). EgocentricFilter
    # mirrors that to self-frame internally (-0.5, -0.18).
    flt = EgocentricFilter(player_id=2, team="team_b", role="CM", bus=bus)
    god_view = build_god_view_red_pair(
        sender_pos_abs=(0.5, 0.10),
        listener_pos_abs=(0.5, 0.18),
    )
    obs = flt.filter(god_view, tick=10)

    assert len(obs.heard_calls) == 1, (
        f"Bug #1 NOT FIXED — team_b listener should hear team_b sender at "
        f"true distance 0.08; got {len(obs.heard_calls)} heard calls. "
        f"(Pre-fix bug: cross-frame distance ≈ 1.04, rejected as out of "
        f"NEARBY_RADIUS=0.20.)"
    )
    print("OK: Bug #1 — team_b 'nearby' Call correctly received "
          "(distance computed in absolute frame).")

    # Returned sender_position must be in self-frame for prompts.py.
    h = obs.heard_calls[0]
    assert h.sender_position.x < 0, (
        f"HeardCall.sender_position should be mirrored to self-frame for "
        f"team_b listener; got x={h.sender_position.x:+.3f} "
        f"(expected negative, since absolute +0.5 → self-frame -0.5)."
    )
    assert h.sender_position.y < 0, (
        f"HeardCall.sender_position.y should be mirrored too; got "
        f"y={h.sender_position.y:+.3f} (expected negative)."
    )
    print(f"OK: HeardCall.sender_position mirrored to self-frame for team_b: "
          f"x={h.sender_position.x:+.2f}, y={h.sender_position.y:+.2f}")


def test_bug1_team_a_unchanged() -> None:
    """team_a (sign=+1) — both transforms are no-ops. Verifies the fix
    didn't regress the left team's behavior.
    """
    bus = TeamMessageBus()
    msg = Message(
        sender_player_id=1, sender_jersey=7,
        sender_position=Vec2(-0.5, 0.10),
        message="传给我",
        audience="nearby",
        tick_posted=0,
    )
    bus.post("left", msg)

    flt = EgocentricFilter(player_id=2, team="team_a", role="CM", bus=bus)
    # Build a left-team god-view inline (mirror of build_god_view_red_pair).
    lt = [[0.0, 0.0]] * 5
    lt[1] = [-0.5, 0.10]
    lt[2] = [-0.5, 0.18]
    god_view = {
        "active": 2,
        "left_team": lt,
        "left_team_direction": [[0.0, 0.0]] * 5,
        "left_team_roles": [0, 1, 2, 4, 5],
        "left_team_tired_factor": [0.0] * 5,
        "right_team": [[0.9, 0.0]] * 5,
        "right_team_direction": [[0.0, 0.0]] * 5,
        "right_team_roles": [0, 1, 2, 4, 5],
        "right_team_tired_factor": [0.0] * 5,
        "ball": [0.0, 0.0],
        "ball_direction": [0.0, 0.0],
        "ball_owned_team": -1,
        "ball_owned_player": -1,
        "score": [0, 0],
    }
    obs = flt.filter(god_view, tick=10)
    assert len(obs.heard_calls) == 1, (
        "team_a regression — left listener should hear left sender."
    )
    h = obs.heard_calls[0]
    # team_a sign=+1, no flip: sender stays at (-0.5, +0.10)
    assert abs(h.sender_position.x - (-0.5)) < 1e-9
    assert abs(h.sender_position.y - 0.10) < 1e-9
    print("OK: team_a (sign=+1) unchanged — sender_position passes through "
          "unmirrored, distance is identity, message received.")


# ---------------------------------------------------------------------------
# Bug #2 — ShootController ZONE_Y_BIAS not mirrored
# ---------------------------------------------------------------------------

# Direction sectors that count as "shooting toward team_b's visual top".
# team_b shoots toward absolute -x (LEFT). Their visual TOP = absolute +y
# = gfootball action BOTTOM family (since gfootball y > 0 = "bottom on
# screen" per the comment in motor.py near _SECTOR_TO_ACTION).
_TEAMB_TOP_LEFTWARD_ACTIONS = {A.LEFT, A.BOTTOM_LEFT, A.BOTTOM}
# Direction sectors that count as "shooting toward team_b's visual bottom".
_TEAMB_BOTTOM_LEFTWARD_ACTIONS = {A.LEFT, A.TOP_LEFT, A.TOP}


def test_bug2_team_b_top_left_aims_top_from_their_view() -> None:
    """team_b shooter at absolute (+0.7, 0). target_zone='top_left' should
    aim at THEIR view's top, which is absolute +y after the mirror.

    Pre-fix: goal_y = -0.04 absolute → from (+0.7, 0) toward (-1, -0.04)
    → action ≈ LEFT or TOP_LEFT (which from team_b's POV is BOTTOM-left).
    Post-fix: goal_y = +0.04 absolute → from (+0.7, 0) toward (-1, +0.04)
    → action ≈ LEFT or BOTTOM_LEFT (which from team_b's POV is TOP-left).
    """
    skill = Shoot(target_zone="top_left")
    c = ShootController(skill=skill, team_side="right", player_id=4)

    obs = build_obs_with_red_player_at(idx=4, pos_abs=(0.7, 0.0))
    obs["ball_owned_team"] = 1
    obs["ball_owned_player"] = 4

    action, status = c.step(obs)
    assert status == "in_progress", f"unexpected status: {status}"
    assert action in _TEAMB_TOP_LEFTWARD_ACTIONS, (
        f"Bug #2 NOT FIXED — team_b Shoot('top_left') from (+0.7, 0) "
        f"should produce LEFT / BOTTOM_LEFT / BOTTOM (toward absolute +y, "
        f"which is team_b's visual TOP); got {ACTION_NAMES[action]}."
    )
    print(f"OK: Bug #2 — team_b Shoot('top_left') tick 1 = "
          f"{ACTION_NAMES[action]} (toward absolute +y = team_b's visual top).")

    # Symmetric check on bottom_right.
    skill_b = Shoot(target_zone="bottom_right")
    c_b = ShootController(skill=skill_b, team_side="right", player_id=4)
    action_b, _ = c_b.step(obs)
    assert action_b in _TEAMB_BOTTOM_LEFTWARD_ACTIONS, (
        f"team_b Shoot('bottom_right') should aim toward absolute -y "
        f"(team_b's visual bottom); got {ACTION_NAMES[action_b]}."
    )
    print(f"OK: Bug #2 — team_b Shoot('bottom_right') tick 1 = "
          f"{ACTION_NAMES[action_b]} (toward absolute -y = team_b's visual bottom).")


def test_bug2_team_a_unchanged() -> None:
    """team_a (sign=+1): ZONE_Y_BIAS applied with no flip. Shooter at
    absolute (-0.7, 0), 'top_left' should still aim toward absolute -y
    (which IS team_a's visual top — they shoot rightward, see top = -y).
    """
    skill = Shoot(target_zone="top_left")
    c = ShootController(skill=skill, team_side="left", player_id=4)

    # team_a player on left half, with ball.
    rt = [[0.9, 0.0]] * 5
    lt = [[0.0, 0.0]] * 5
    lt[4] = [-0.7, 0.0]
    obs = {
        "active": 4,
        "left_team": lt, "left_team_direction": [[0.0, 0.0]] * 5,
        "right_team": rt, "right_team_direction": [[0.0, 0.0]] * 5,
        "ball": [0.0, 0.0], "ball_direction": [0.0, 0.0],
        "ball_owned_team": 0, "ball_owned_player": 4,
    }
    action, status = c.step(obs)
    # team_a's "top" is absolute -y; from (-0.7, 0) toward (+1, -0.04) →
    # action should be RIGHT or TOP_RIGHT or TOP (rightward + slightly up).
    expected = {A.RIGHT, A.TOP_RIGHT, A.TOP}
    assert action in expected, (
        f"team_a regression — Shoot('top_left') from (-0.7, 0) should "
        f"aim RIGHT / TOP_RIGHT / TOP; got {ACTION_NAMES[action]}."
    )
    print(f"OK: team_a Shoot('top_left') unchanged — action = "
          f"{ACTION_NAMES[action]} (toward absolute -y = team_a's visual top).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Self-frame coordinate fixes — smoke test")
    print("=" * 70)
    print(f"NEARBY_RADIUS = {TeamMessageBus.NEARBY_RADIUS}")
    print()
    print("--- Bug #1: TeamMessageBus listener_position frame ---")
    test_bug1_team_b_nearby_call_received()
    test_bug1_team_a_unchanged()
    print()
    print("--- Bug #2: ShootController ZONE_Y_BIAS mirror ---")
    test_bug2_team_b_top_left_aims_top_from_their_view()
    test_bug2_team_a_unchanged()
    print()
    print("=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
