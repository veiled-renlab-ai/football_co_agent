"""李强 — 红队中锋 #10 fallback (速度型，反越位接直塞)."""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, MoveTo, Skill
from ._helpers import clip, nearest_opponent, teammate_with_ball
from .context import FallbackContext

_MIN_DEPTH_X = 0.15      # 永不回到这里以下 (persona: 不做球只接直塞)
_OFFSIDE_HOVER = 0.35    # 默认贴这条线做反越位
_NEVER_CROSS_Y = 0.20    # 中锋保持中路通道


def fallback_li_qiang(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    tm_ball = teammate_with_ball(obs)

    # 1. 我持球 + 对方禁区附近 → 单刀冲门
    if obs.self_state.has_ball and self_pos.x > 0.3:
        return DribbleToward(target_x=1.00, target_y=0.0, urgency="sprint")

    # 2. 我持球 + 被贴身 → 护住别做球
    if obs.self_state.has_ball:
        closest_opp = nearest_opponent(obs)
        if closest_opp is not None and closest_opp.distance < 0.08:
            return HoldPosition()
        return DribbleToward(target_x=0.90, target_y=0.0, urgency="sprint")

    # 3. 反击启动 (队友在中后场拿球) → 冲对方半场中路
    if tm_ball is not None and tm_ball.position.x < -0.10:
        return MoveTo(target_x=0.55, target_y=0.0, urgency="sprint")

    # 4. 对方防线高位 → 贴最深后卫找缝 (反越位)
    opps = obs.opponents()
    if opps:
        deepest = max(opps, key=lambda e: e.position.x)
        if deepest.position.x > -0.10:
            # 贴到他前面一点点，不站死越位
            target_x = clip(deepest.position.x - 0.02, _MIN_DEPTH_X, 0.90)
            target_y = 0.08 if self_pos.y <= 0 else -0.08  # 横向交替，制造出球点
            return MoveTo(target_x=target_x, target_y=target_y, urgency="jog")

    # 5. 队友在对方禁区附近 → 抢点
    ball = obs.ball()
    if ball is not None and ball.position.x > 0.60:
        target_y = 0.10 if ball.position.y < 0 else -0.10  # 跑另一柱
        return MoveTo(target_x=0.85, target_y=target_y, urgency="sprint")

    # 6. 默认 —— 站在越位线附近等直塞
    target_x = max(self_pos.x, _OFFSIDE_HOVER)
    target_y = clip(self_pos.y, -_NEVER_CROSS_Y, _NEVER_CROSS_Y)
    if abs(target_x - _OFFSIDE_HOVER) < 0.03 and abs(target_y) < 0.04:
        return HoldPosition()
    return MoveTo(target_x=_OFFSIDE_HOVER, target_y=0.0, urgency="walk")
