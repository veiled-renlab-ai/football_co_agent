"""刘锋 — 红队右前卫 #7 fallback (工兵型，跑动量大逼抢凶).

Fix 3: 丢球后立即sprint回防守位置（x≥-0.10），不再缓慢走位。
"""
from __future__ import annotations

from ..skills import HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    ball_owned_by_opponent, clip, clip_target, loose_ball,
    nearest_opponent, opponent_with_ball, teammate_with_ball,
)
from .context import FallbackContext

_WIDE_Y = 0.30           # 工兵型也要拉宽度
_Y_MIN = 0.15            # 永不缩中
_PRESS_RADIUS = 0.16

# Fix 3: RM防守回位线（对方控球时必须回到此x以上，红队x坐标=朝-x进攻方向）
_DEFENSIVE_RETREAT_X = -0.10


def fallback_liu_feng(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    carrier = opponent_with_ball(obs)

    # ---- Fix 3: 对方控球时立即sprint回防守位置 ----
    opp_has_ball = ball_owned_by_opponent(obs)
    if opp_has_ball and carrier is not None:
        ball = obs.ball()
        # 对方在红队半场推进，或carrier朝+x方向运动（红队球门在+x侧）
        ball_in_red_half = ball is not None and ball.position.x > 0.0
        carrier_advancing = (
            carrier.velocity is not None and carrier.velocity.x > 0.001
        ) or (carrier.position.x > -0.30)
        if (ball_in_red_half or carrier_advancing) and self_pos.x < _DEFENSIVE_RETREAT_X:
            return MoveTo(
                target_x=_DEFENSIVE_RETREAT_X,
                target_y=_WIDE_Y,
                urgency="sprint",
            )
        elif self_pos.x < _DEFENSIVE_RETREAT_X - 0.15:
            return MoveTo(
                target_x=_DEFENSIVE_RETREAT_X - 0.10,
                target_y=_WIDE_Y,
                urgency="sprint",
            )

    # 1. 对方在我方右路前/中场持球 → 主动 Press
    if carrier is not None and carrier.position.x < 0.20 and carrier.position.y > -0.10:
        if carrier.distance < _PRESS_RADIUS:
            return Press(opponent_id=carrier.entity_id)
        return MoveTo(
            target_x=clip(carrier.position.x, -0.20, 0.60),
            target_y=clip(carrier.position.y, _Y_MIN, 0.42),
            urgency="sprint",
        )

    # 2. 右半区散球 + 不太远 → 冲抢
    lb = loose_ball(obs)
    if lb is not None and lb.position.y > -0.15 and lb.position.x > -0.30:
        if lb.distance < 0.35:
            return MoveTo(
                target_x=lb.position.x,
                target_y=lb.position.y,
                urgency="sprint",
            )

    # 3. 对方中/左路持球 → 封住右路出球线
    if carrier is not None:
        opp_rb = nearest_opponent(obs)
        if opp_rb is not None and opp_rb.position.y > 0.05:
            return Mark(opponent_id=opp_rb.entity_id)

    # 4. 我方队友控球且在对方半场 → 拉右肋等直塞
    tm_ball = teammate_with_ball(obs)
    if tm_ball is not None and tm_ball.position.x < 0.0:
        if self_pos.x < -0.75:
            return HoldPosition()
        return MoveTo(target_x=-0.60, target_y=0.35, urgency="sprint")

    # 5. 我方半场控球 → 提前站高拉宽
    if tm_ball is not None:
        return MoveTo(target_x=-0.20, target_y=_WIDE_Y, urgency="jog")

    # 6. 静态 → 默认贴右边线拉宽度
    target_y = _WIDE_Y if self_pos.y < _Y_MIN else self_pos.y
    target_x = clip(self_pos.x, -0.10, 0.30)
    return MoveTo(target_x=target_x, target_y=target_y, urgency="walk")
