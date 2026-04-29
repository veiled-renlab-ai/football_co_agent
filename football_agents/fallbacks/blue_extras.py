"""Blue team 11v11 extras — 6 new fallbacks for the 4-3-3 slots not in 5v5.

Each fallback is a body-rest-state policy: simple positional defaults that fill
the gap between LLM decisions. They MUST stay short (~30 lines each); deeper
tactics belong in prompts/Skills, not here.

Slot mapping → fallback function:
    slot 3 RCB  → fallback_zhang_wei
    slot 4 RB   → fallback_li_ming
    slot 5 LCM  → fallback_wang_gang
    slot 6 CCM  → fallback_sun_jian
    slot 7 RCM  → fallback_han_lei
    slot 8 LM   → fallback_zhou_kai
"""
from __future__ import annotations

from ..skills import HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    blue_ranked_threats, clip_target, get_defensive_line_x_blue,
    loose_ball, opponent_with_ball,
)
from .context import FallbackContext


# ──────────────────────────────────────────────────────────────────────────
# RCB — 张伟 #5  (mirror of CB, anchor on right half)
# ──────────────────────────────────────────────────────────────────────────
def fallback_zhang_wei(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    AX, AY = -0.50, 0.06

    def_line_x = min(get_defensive_line_x_blue(obs), self_pos.x)
    threats = blue_ranked_threats(obs)
    # RCB takes the SECOND-deepest threat so the two CBs don't both chase #1.
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


# ──────────────────────────────────────────────────────────────────────────
# RB — 李明 #2  (mirror of LB, anchor on right side)
# ──────────────────────────────────────────────────────────────────────────
def fallback_li_ming(ctx: FallbackContext) -> Skill:
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

    if ball is not None and ball.position.y < -0.15:
        return MoveTo(target_x=-0.25, target_y=0.12, urgency="jog")

    lb = loose_ball(obs)
    if lb is not None and lb.position.x < 0.0 and lb.position.y > 0.0 and lb.distance < 0.25:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    if abs(self_pos.x - AX) < 0.04 and abs(self_pos.y - AY) < 0.04:
        return HoldPosition()
    tx, ty = clip_target(AX, AY, -0.85, 0.25, 0.05, 0.40)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")


# ──────────────────────────────────────────────────────────────────────────
# CM template — 3 mids share logic with different y-anchors
# ──────────────────────────────────────────────────────────────────────────
def _cm_fallback(ctx: FallbackContext, ax: float, ay: float) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position

    # If opponent is carrying and they're advancing into our half → press the carrier
    carrier = opponent_with_ball(obs)
    if carrier is not None and -0.40 < carrier.position.x < 0.10 and carrier.distance < 0.20:
        return Press(opponent_id=carrier.entity_id)

    # Loose ball nearby in midfield → contest
    lb = loose_ball(obs)
    if lb is not None and abs(lb.position.x) < 0.35 and lb.distance < 0.22:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    # Default: hold lane
    if abs(self_pos.x - ax) < 0.05 and abs(self_pos.y - ay) < 0.05:
        return HoldPosition()
    tx, ty = clip_target(ax, ay, -0.50, 0.40, -0.25, 0.25)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")


def fallback_wang_gang(ctx: FallbackContext) -> Skill:   # LCM
    return _cm_fallback(ctx, ax=-0.18, ay=-0.10)


def fallback_sun_jian(ctx: FallbackContext) -> Skill:    # CCM
    return _cm_fallback(ctx, ax=-0.27, ay=0.0)


def fallback_han_lei(ctx: FallbackContext) -> Skill:     # RCM
    return _cm_fallback(ctx, ax=-0.18, ay=0.10)


# ──────────────────────────────────────────────────────────────────────────
# LM — 周凯 #7  (left winger, mirror of RM)
# ──────────────────────────────────────────────────────────────────────────
def fallback_zhou_kai(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    AX, AY = 0.15, -0.32

    # Defensive duty: opponent attacking through our left flank → track back
    carrier = opponent_with_ball(obs)
    if carrier is not None and carrier.position.x < -0.20 and carrier.position.y < -0.15:
        return MoveTo(target_x=-0.30, target_y=-0.30, urgency="sprint")

    # Loose ball on the left wing → contest
    lb = loose_ball(obs)
    if lb is not None and lb.position.y < -0.10 and lb.distance < 0.30:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    # Default: hold left lane
    if abs(self_pos.x - AX) < 0.05 and abs(self_pos.y - AY) < 0.05:
        return HoldPosition()
    tx, ty = clip_target(AX, AY, -0.55, 0.75, -0.42, -0.15)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")
