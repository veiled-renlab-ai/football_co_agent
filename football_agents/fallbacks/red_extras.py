"""Red team 11v11 extras — 6 new fallbacks for the 4-3-3 slots not in 5v5.

Same coord convention as blue: gfootball mirrors right-team views so each
agent always thinks of +x as the opponent goal. So anchor coords below are
identical in form to the blue extras (e.g. RCB sits at x=-0.50 from the
agent's own perspective for both teams).

Slot mapping → fallback function:
    slot 3 RCB  → fallback_huang_tao
    slot 4 RB   → fallback_wu_fei
    slot 5 LCM  → fallback_luo_cheng
    slot 6 CCM  → fallback_jiang_hu
    slot 7 RCM  → fallback_xie_yong
    slot 8 LM   → fallback_zhu_hao
"""
from __future__ import annotations

from ..skills import HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    blue_ranked_threats, clip_target, get_defensive_line_x_blue,
    loose_ball, opponent_with_ball,
)
from .context import FallbackContext

# Note: gfootball's per-slot view mirrors right-team players so each agent sees
# self at left_team and +x toward opponent goal. So red fallbacks use the SAME
# helpers as blue (deepest defender = MIN x; threats sorted ASCENDING).


# RCB — 黄涛 #4
def fallback_huang_tao(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    AX, AY = -0.50, 0.06

    def_line_x = min(get_defensive_line_x_blue(obs), self_pos.x)
    threats = blue_ranked_threats(obs)
    target = threats[1] if len(threats) >= 2 else (threats[0] if threats else None)

    if target is not None and target.position.x < def_line_x - 0.02:
        tx, ty = clip_target(AX, AY, -0.85, 0.0, -0.05, 0.20)
        return HoldPosition() if abs(self_pos.x - tx) < 0.04 and abs(self_pos.y - ty) < 0.04 \
            else MoveTo(target_x=tx, target_y=ty, urgency="jog")

    carrier = opponent_with_ball(obs)
    ball = obs.ball()
    if carrier is not None and ball is not None \
       and ball.position.x < -0.75 and carrier.distance < 0.15:
        return Press(opponent_id=carrier.entity_id)

    if carrier is not None and -0.50 < carrier.position.x < 0.0 and carrier.position.y > -0.05:
        if target is not None:
            return Mark(opponent_id=target.entity_id)

    lb = loose_ball(obs)
    if lb is not None and lb.position.x < -0.60 and lb.distance < 0.20:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    if abs(self_pos.x - AX) < 0.04 and abs(self_pos.y - AY) < 0.04:
        return HoldPosition()
    tx, ty = clip_target(AX, AY, -0.85, 0.0, -0.05, 0.20)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")


# RB — 吴飞 #13
def fallback_wu_fei(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    AX, AY = -0.42, 0.20

    carrier = opponent_with_ball(obs)
    if carrier is not None and carrier.position.x < -0.40 and carrier.position.y > 0.05:
        if carrier.distance < 0.20:
            return Press(opponent_id=carrier.entity_id)

    if carrier is not None and carrier.position.x < 0.0 and carrier.position.y > 0.05:
        return Mark(opponent_id=carrier.entity_id)

    ball = obs.ball()
    if ball is not None and ball.position.x < self_pos.x - 0.05 and carrier is not None:
        return MoveTo(target_x=-0.55, target_y=0.15, urgency="sprint")

    lb = loose_ball(obs)
    if lb is not None and lb.position.x < 0.0 and lb.position.y > 0.0 and lb.distance < 0.25:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    if abs(self_pos.x - AX) < 0.04 and abs(self_pos.y - AY) < 0.04:
        return HoldPosition()
    tx, ty = clip_target(AX, AY, -0.85, 0.25, 0.05, 0.40)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")


# CM template — same as blue
def _cm_fallback(ctx: FallbackContext, ax: float, ay: float) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position

    carrier = opponent_with_ball(obs)
    if carrier is not None and -0.40 < carrier.position.x < 0.10 and carrier.distance < 0.20:
        return Press(opponent_id=carrier.entity_id)

    lb = loose_ball(obs)
    if lb is not None and abs(lb.position.x) < 0.35 and lb.distance < 0.22:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    if abs(self_pos.x - ax) < 0.05 and abs(self_pos.y - ay) < 0.05:
        return HoldPosition()
    tx, ty = clip_target(ax, ay, -0.50, 0.40, -0.25, 0.25)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")


def fallback_luo_cheng(ctx: FallbackContext) -> Skill:   # LCM
    return _cm_fallback(ctx, ax=-0.18, ay=-0.10)


def fallback_jiang_hu(ctx: FallbackContext) -> Skill:    # CCM
    return _cm_fallback(ctx, ax=-0.27, ay=0.0)


def fallback_xie_yong(ctx: FallbackContext) -> Skill:    # RCM
    return _cm_fallback(ctx, ax=-0.18, ay=0.10)


# LM — 朱浩 #11
def fallback_zhu_hao(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    AX, AY = 0.15, -0.32

    carrier = opponent_with_ball(obs)
    if carrier is not None and carrier.position.x < -0.20 and carrier.position.y < -0.15:
        return MoveTo(target_x=-0.30, target_y=-0.30, urgency="sprint")

    lb = loose_ball(obs)
    if lb is not None and lb.position.y < -0.10 and lb.distance < 0.30:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    if abs(self_pos.x - AX) < 0.05 and abs(self_pos.y - AY) < 0.05:
        return HoldPosition()
    tx, ty = clip_target(AX, AY, -0.55, 0.75, -0.42, -0.15)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")
