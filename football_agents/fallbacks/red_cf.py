"""李强 — 红队中锋 #10 fallback (速度型，反越位接直塞).

Fix 3: 丢球后立即sprint回防守位置（x≥-0.20），不再缓慢走位。
       对方控球且在朝红队球门推进时CF sprint回防守位置。
"""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, MoveTo, Skill
from ._helpers import (
    ball_owned_by_opponent, clip, nearest_opponent,
    opponent_with_ball, teammate_with_ball,
)
from .context import FallbackContext

_MIN_DEPTH_X = 0.15      # 永不回到这里以下 (persona: 不做球只接直塞)
_OFFSIDE_HOVER = 0.35    # 默认贴这条线做反越位
_NEVER_CROSS_Y = 0.20    # 中锋保持中路通道

# Fix 3: 防守回位线（对方控球时CF必须回到此x以上，红队x坐标系 朝-x进攻）
# 红队防守方向：对方在-x侧进攻，红队守门在+x侧（x=+1）
# 对方控球时，红队CF需要回到x≥-0.20（不能跑太深到-1）
_DEFENSIVE_RETREAT_X = -0.20


def fallback_li_qiang(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    tm_ball = teammate_with_ball(obs)

    # ---- Fix 3: 对方控球时立即sprint回防守位置 ----
    opp_has_ball = ball_owned_by_opponent(obs)
    if opp_has_ball:
        ball = obs.ball()
        # 对方在红队半场推进（ball.x > 0，或球朝+x方向）
        ball_in_red_half = ball is not None and ball.position.x > 0.0
        # 检测对方是否在朝红队球门推进
        carrier = opponent_with_ball(obs)
        carrier_advancing = (
            carrier is not None and carrier.velocity is not None
            and carrier.velocity.x > 0.001
        ) or (carrier is not None and carrier.position.x > -0.30)
        if (ball_in_red_half or carrier_advancing) and self_pos.x < _DEFENSIVE_RETREAT_X:
            # 丢球后sprint回防守位置
            return MoveTo(
                target_x=_DEFENSIVE_RETREAT_X,
                target_y=0.0,
                urgency="sprint",
            )
        elif self_pos.x < _DEFENSIVE_RETREAT_X - 0.10:
            return MoveTo(
                target_x=_DEFENSIVE_RETREAT_X - 0.10,
                target_y=0.0,
                urgency="jog",
            )

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
    if tm_ball is not None and tm_ball.position.x > 0.10:
        return MoveTo(target_x=0.55, target_y=0.0, urgency="sprint")

    # 4. 对方防线高位 → 贴最深后卫找缝 (反越位)
    opps = obs.opponents()
    if opps:
        deepest = min(opps, key=lambda e: e.position.x)  # 红队：对方最深=x最小（靠近-1）
        if deepest.position.x < 0.10:
            target_x = clip(deepest.position.x + 0.02, _MIN_DEPTH_X, 0.90)
            target_y = 0.08 if self_pos.y <= 0 else -0.08
            return MoveTo(target_x=target_x, target_y=target_y, urgency="jog")

    # 5. 队友在对方禁区附近 → 抢点
    ball = obs.ball()
    if ball is not None and ball.position.x < -0.60:
        target_y = 0.10 if ball.position.y < 0 else -0.10
        return MoveTo(target_x=-0.85, target_y=target_y, urgency="sprint")

    # 6. 默认 —— 站在越位线附近等直塞
    target_x = max(self_pos.x, _OFFSIDE_HOVER)
    target_y = clip(self_pos.y, -_NEVER_CROSS_Y, _NEVER_CROSS_Y)
    if abs(target_x - _OFFSIDE_HOVER) < 0.03 and abs(target_y) < 0.04:
        return HoldPosition()
    return MoveTo(target_x=_OFFSIDE_HOVER, target_y=0.0, urgency="walk")
