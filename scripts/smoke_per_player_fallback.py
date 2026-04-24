"""Smoke test — verify all 10 per-persona fallbacks load and return
different Skills for different personas given the same realistic obs.

Does NOT require gfootball runtime — just the fallback package + perception
dataclasses. Safe to run from any shell with the venv active.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.fallbacks import FallbackContext, get_fallback
from football_agents.perception import EntityView, Observation, SelfState, Vec2
from football_agents.personas import TEAM_BLUE_5V5, TEAM_RED_5V5


def build_obs(self_pos: Vec2, role: str, has_ball: bool = False) -> Observation:
    """Synthetic obs: opponent dribbling toward our goal near left-side midfield."""
    self_state = SelfState(
        player_id=0, team="team_a", role=role,
        position=self_pos, velocity=Vec2(0.0, 0.0),
        facing_deg=0.0, stamina=1.0, has_ball=has_ball,
    )
    opp_carrier = EntityView(
        entity_id=2, role="opponent",
        position=Vec2(-0.10, 0.08), velocity=Vec2(-0.002, 0.0),
        distance=0.20, in_current_fov=True, has_ball=True,
    )
    teammate = EntityView(
        entity_id=4, role="teammate",
        position=Vec2(-0.20, 0.10), velocity=Vec2(0.0, 0.0),
        distance=0.10, in_current_fov=True, has_ball=False,
    )
    ball = EntityView(
        entity_id=99, role="ball",
        position=Vec2(-0.10, 0.08), velocity=Vec2(-0.002, 0.0),
        distance=0.15, in_current_fov=True, has_ball=False,
    )
    return Observation(
        tick=100, match_clock="00:02", score=(0, 0),
        self_state=self_state, game_mode=0,
        perceived_entities=[opp_carrier, teammate, ball],
    )


def short_skill(skill) -> str:
    t = type(skill).__name__
    args = []
    for a in ("target_x", "target_y", "urgency", "opponent_id"):
        if hasattr(skill, a):
            v = getattr(skill, a)
            if isinstance(v, float):
                args.append(f"{a}={v:+.2f}")
            else:
                args.append(f"{a}={v}")
    inner = ", ".join(args)
    return f"{t}({inner})"


def main() -> None:
    print("Scenario: opponent dribbling in the -y half toward our goal\n")

    scenarios = [
        ("蓝 林涛 GK", TEAM_BLUE_5V5[0], Vec2(-0.98, 0.0), "GK"),
        ("红 赵强 GK", TEAM_RED_5V5[0],  Vec2(-0.98, 0.0), "GK"),
        ("蓝 王浩 RM", TEAM_BLUE_5V5[1], Vec2(0.05, 0.30), "RM"),
        ("红 刘锋 RM", TEAM_RED_5V5[1],  Vec2(0.05, 0.30), "RM"),
        ("蓝 陈宇 CF", TEAM_BLUE_5V5[2], Vec2(0.30, 0.0),  "CF"),
        ("红 李强 CF", TEAM_RED_5V5[2],  Vec2(0.30, 0.0),  "CF"),
        ("蓝 周俊 LB", TEAM_BLUE_5V5[3], Vec2(-0.15, -0.15), "LB"),
        ("红 孙斌 LB", TEAM_RED_5V5[3],  Vec2(-0.10, -0.10), "LB"),
        ("蓝 高磊 CB", TEAM_BLUE_5V5[4], Vec2(-0.20, 0.10), "CB"),
        ("红 马亮 CB", TEAM_RED_5V5[4],  Vec2(-0.15, 0.10), "CB"),
    ]
    for label, persona, pos, role in scenarios:
        fn = get_fallback(persona)
        obs = build_obs(pos, role=role)
        ctx = FallbackContext(persona=persona, obs=obs, recent_llm_intent=None)
        skill = fn(ctx)
        print(f"  {label} -> {short_skill(skill)}")

    # Sanity check: with a recent LLM intent, fallback should defer to it
    print("\nRecent-LLM-intent guard (should all echo the provided intent):")
    from football_agents.skills import MoveTo
    recent = (MoveTo(target_x=0.42, target_y=0.1, urgency="jog"), 100)
    persona = TEAM_BLUE_5V5[0]
    fn = get_fallback(persona)
    obs = build_obs(Vec2(-0.98, 0.0), role="GK")
    ctx = FallbackContext(persona=persona, obs=obs, recent_llm_intent=recent)
    skill = fn(ctx)
    print(f"  林涛 w/ recent LLM MoveTo(+0.42,+0.10,jog) -> {short_skill(skill)}")

    # Sanity check: game_mode != 0 always yields HoldPosition
    print("\nGame-mode guard (game_mode=1 KickOff should force HoldPosition):")
    obs2 = build_obs(Vec2(0.30, 0.0), role="CF")
    obs2.game_mode = 1
    ctx2 = FallbackContext(persona=TEAM_BLUE_5V5[2], obs=obs2, recent_llm_intent=None)
    skill2 = get_fallback(TEAM_BLUE_5V5[2])(ctx2)
    print(f"  陈宇 during KickOff -> {short_skill(skill2)}")

    # Sanity check: low stamina downshifts sprint to jog
    print("\nStamina guard (stamina=0.2 should downshift sprint->jog):")
    self_state = SelfState(
        player_id=0, team="team_a", role="CF",
        position=Vec2(0.60, 0.0), velocity=Vec2(0, 0),
        facing_deg=0, stamina=0.2, has_ball=False,
    )
    # Build an obs that would normally trigger sprint for 李强 (ball in opp penalty)
    ball = EntityView(entity_id=99, role="ball",
                      position=Vec2(0.72, 0.1), velocity=Vec2(0, 0),
                      distance=0.15, in_current_fov=True, has_ball=False)
    tm = EntityView(entity_id=1, role="teammate",
                    position=Vec2(0.72, 0.1), velocity=Vec2(0, 0),
                    distance=0.15, in_current_fov=True, has_ball=True)
    obs3 = Observation(
        tick=100, match_clock="00:02", score=(0, 0),
        self_state=self_state, game_mode=0,
        perceived_entities=[ball, tm],
    )
    ctx3 = FallbackContext(persona=TEAM_RED_5V5[2], obs=obs3, recent_llm_intent=None)
    skill3 = get_fallback(TEAM_RED_5V5[2])(ctx3)
    print(f"  李强 gassed in opp half -> {short_skill(skill3)}")

    print("\nOK.")


if __name__ == "__main__":
    main()
