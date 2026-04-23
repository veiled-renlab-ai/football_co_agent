"""Smoke: render_observation now surfaces entity velocity + opponent carrier position.

Covers BUG #3 (velocity dropped) and BUG #4 (opponent carrier missing 位置).

Run:
    python -m scripts.smoke_render_velocity_carrier
"""
from __future__ import annotations

import io
import sys

# Force UTF-8 stdout so Chinese + bullet chars render on Windows cmd/cp936.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from football_agents.perception import EntityView, Observation, SelfState, Vec2
from football_agents.prompts import DEFAULT_PERSONA, render_observation


def _make_obs(entities: list[EntityView]) -> Observation:
    return Observation(
        tick=0,
        match_clock="00:00",
        score=(0, 0),
        self_state=SelfState(
            player_id=8, team="team_a", role="CAM",
            position=Vec2(0.0, 0.0), velocity=Vec2(0.0, 0.0),
            facing_deg=0.0, stamina=1.0, has_ball=False,
        ),
        perceived_entities=entities,
    )


def case_1_teammate_approaching() -> None:
    """Visible teammate at (0.3, 0) moving toward agent at (0.0, 0).
    velocity (-0.003, 0) → component toward self ≈ +1, mid-bucket speed."""
    teammate = EntityView(
        entity_id=10, role="teammate",
        position=Vec2(0.3, 0.0), velocity=Vec2(-0.003, 0.0),
        distance=0.3, in_current_fov=True, age_ticks=0, has_ball=False,
    )
    # Need a ball EntityView so render doesn't take the no-ball branch
    ball = EntityView(
        entity_id=99, role="ball",
        position=Vec2(0.0, 0.0), velocity=Vec2(0.0, 0.0),
        distance=0.0, in_current_fov=True,
    )
    rendered = render_observation(_make_obs([teammate, ball]), DEFAULT_PERSONA)
    print("--- Case 1: teammate approaching ---")
    for line in rendered.split("\n"):
        if "号" in line and "在" in line:
            print(f"  {line}")
    assert "向你跑来" in rendered, "expected '向你跑来' for approaching teammate"
    print("OK\n")


def case_2_opponent_motion() -> None:
    """Visible opponent at (0.5, 0) with velocity (0.003, 0).
    Note: opponent at distance 0.5 is BEYOND SIGHT_DISTANCE_FULL (0.30).
    Real perception layer would set velocity=None, but we manually inject
    velocity here to verify render renders the phrase when given one."""
    opp = EntityView(
        entity_id=4, role="opponent",
        position=Vec2(0.5, 0.0), velocity=Vec2(0.003, 0.0),
        distance=0.5, in_current_fov=True, age_ticks=0, has_ball=False,
    )
    ball = EntityView(
        entity_id=99, role="ball",
        position=Vec2(0.0, 0.0), velocity=Vec2(0.0, 0.0),
        distance=0.0, in_current_fov=True,
    )
    rendered = render_observation(_make_obs([opp, ball]), DEFAULT_PERSONA)
    print("--- Case 2: opponent moving away ---")
    for line in rendered.split("\n"):
        if "号" in line and "在" in line:
            print(f"  {line}")
    assert "正在远离" in rendered, "expected '正在远离' for opponent moving away"
    print("OK\n")


def case_3_loose_ball_motion() -> None:
    """Loose ball at (0.2, 0) with velocity (0.002, 0) — moving away."""
    ball = EntityView(
        entity_id=99, role="ball",
        position=Vec2(0.2, 0.0), velocity=Vec2(0.002, 0.0),
        distance=0.2, in_current_fov=True, has_ball=False,
    )
    rendered = render_observation(_make_obs([ball]), DEFAULT_PERSONA)
    print("--- Case 3: loose ball motion ---")
    for line in rendered.split("\n"):
        if "散球" in line:
            print(f"  {line}")
    assert "球正滚开" in rendered, "expected '球正滚开' for loose ball moving away"
    print("OK\n")


def case_4_opponent_carrier_position() -> None:
    """Opponent carrier (has_ball=True, opponent role) at (0.4, 0) — must show 位置."""
    opp_carrier = EntityView(
        entity_id=7, role="opponent",
        position=Vec2(0.4, 0.0), velocity=None,
        distance=0.4, in_current_fov=True, age_ticks=0, has_ball=True,
    )
    # Ball is owned by opponent — render uses perceived_entities to find carrier
    ball = EntityView(
        entity_id=99, role="ball",
        position=Vec2(0.4, 0.0), velocity=None,
        distance=0.4, in_current_fov=True,
    )
    rendered = render_observation(_make_obs([opp_carrier, ball]), DEFAULT_PERSONA)
    print("--- Case 4: opponent carrier line ---")
    for line in rendered.split("\n"):
        if "对方" in line and "号脚下" in line:
            print(f"  {line}")
    assert "球在对方 7 号脚下" in rendered
    assert "位置 " in rendered, "opponent carrier line missing '位置 ' (BUG #4)"
    # Position string for (0.4, 0): _zone_x=0.4 falls in '对方半场前压', _zone_y(0)='正中'
    assert "对方半场前压正中" in rendered, "expected zone phrase from _describe_position"
    print("OK\n")


def case_5_far_teammate_no_velocity() -> None:
    """Far teammate at (0.7, 0) with velocity=None — should not crash, no motion phrase."""
    far = EntityView(
        entity_id=11, role="teammate",
        position=Vec2(0.7, 0.0), velocity=None,
        distance=0.7, in_current_fov=True, age_ticks=0, has_ball=False,
    )
    ball = EntityView(
        entity_id=99, role="ball",
        position=Vec2(0.0, 0.0), velocity=Vec2(0.0, 0.0),
        distance=0.0, in_current_fov=True,
    )
    rendered = render_observation(_make_obs([far, ball]), DEFAULT_PERSONA)
    print("--- Case 5: far teammate, velocity=None ---")
    for line in rendered.split("\n"):
        if "11 号" in line:
            print(f"  {line}")
    # Find the teammate line and verify no motion phrase tokens appear in it
    teammate_lines = [l for l in rendered.split("\n") if "11 号" in l]
    assert teammate_lines, "teammate line missing"
    for line in teammate_lines:
        for token in ("向你跑来", "正在远离", "横向移动", "原地"):
            assert token not in line, f"unexpected motion '{token}' on velocity=None"
    print("OK (no crash, no motion phrase)\n")


def case_6_static_teammate() -> None:
    """Static teammate at (0.2, 0.1) with velocity=Vec2(0,0) → '原地'."""
    static = EntityView(
        entity_id=5, role="teammate",
        position=Vec2(0.2, 0.1), velocity=Vec2(0.0, 0.0),
        distance=0.22, in_current_fov=True, age_ticks=0, has_ball=False,
    )
    ball = EntityView(
        entity_id=99, role="ball",
        position=Vec2(0.0, 0.0), velocity=Vec2(0.0, 0.0),
        distance=0.0, in_current_fov=True,
    )
    rendered = render_observation(_make_obs([static, ball]), DEFAULT_PERSONA)
    print("--- Case 6: static teammate ---")
    for line in rendered.split("\n"):
        if "5 号" in line:
            print(f"  {line}")
    assert "原地" in rendered, "expected '原地' for static teammate"
    print("OK\n")


def main() -> None:
    case_1_teammate_approaching()
    case_2_opponent_motion()
    case_3_loose_ball_motion()
    case_4_opponent_carrier_position()
    case_5_far_teammate_no_velocity()
    case_6_static_teammate()
    print("=" * 50)
    print("All 6 cases PASSED")


if __name__ == "__main__":
    main()
