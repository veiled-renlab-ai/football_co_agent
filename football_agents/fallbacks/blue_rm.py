"""王浩 — 蓝队右前卫 #11 fallback (速度型边路，急脾气爱冒险).

Fix 3: 丢球后立即sprint回防守位置（x≤0.10），不再缓慢走位。
"""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, MoveTo, Press, Skill
from ._helpers import (
    ball_owned_by_opponent, clip, clip_target,
    opponent_with_ball, teammate_with_ball,
)
from .context import FallbackContext

_WIDE_Y = 0.32               # 默认拉到右路边线附近
_Y_MIN = 0.15                # 永不缩到中路以内
_X_MIN = -0.55               # 不过度回撤
_PUSH_X_WHEN_ATTACK = 0.60   # 我方控球推进时 RM 压到的深度（爆发型）

# Fix 3: RM防守回位线（对方控球时必须回到此x以内）
_DEFENSIVE_RETREAT_X = 0.10


def fallback_wang_hao(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    carrier = opponent_with_ball(obs)

    # ---- Fix 3: 对方控球时立即sprint回防守位置 ----
    opp_has_ball = ball_owned_by_opponent(obs)
    if opp_has_ball and carrier is not None:
        ball = obs.ball()
        # 对方在蓝队半场推进，或朝-x方向运动
        ball_in_blue_half = ball is not None and ball.position.x < 0.0
        carrier_advancing = (
            carrier.velocity is not None and carrier.velocity.x < -0.001
        ) or (carrier.position.x < 0.30)
        if (ball_in_blue_half or carrier_advancing) and self_pos.x > _DEFENSIVE_RETREAT_X:
            # sprint回防守位置，保持右路宽度以防对方左边锋
            return MoveTo(
                target_x=_DEFENSIVE_RETREAT_X,
                target_y=_WIDE_Y,
                urgency="sprint",
            )
        elif self_pos.x > _DEFENSIVE_RETREAT_X + 0.15:
            return MoveTo(
                target_x=_DEFENSIVE_RETREAT_X + 0.10,
                target_y=_WIDE_Y,
                urgency="sprint",
            )

    # 1. 我持球在右路 → 继续突破下底
    if obs.self_state.has_ball and self_pos.y > 0.15:
        tx, ty = clip_target(0.90, 0.35, x_max=0.95, y_max=0.40)
        return DribbleToward(target_x=tx, target_y=ty, urgency="sprint")

    # 2. 我方队友控球且在对方半场 → 压上到右肋
    tm_ball = teammate_with_ball(obs)
    if tm_ball is not None and tm_ball.position.x > 0.0:
        return MoveTo(
            target_x=_PUSH_X_WHEN_ATTACK,
            target_y=_WIDE_Y,
            urgency="sprint",
        )

    # 3. 对方在我方右路持球 → 回追协防 (Press 只用于近身)
    if carrier is not None and carrier.position.x < -0.20 and carrier.position.y > 0.05:
        if carrier.distance < 0.20:
            return Press(opponent_id=carrier.entity_id)
        return MoveTo(
            target_x=clip(carrier.position.x + 0.05, _X_MIN, 0.0),
            target_y=_WIDE_Y,
            urgency="sprint",
        )

    # 4. 我方后场控球 → 站高拉宽作为出球点
    if tm_ball is not None and tm_ball.position.x < -0.20:
        return MoveTo(target_x=0.15, target_y=_WIDE_Y, urgency="jog")

    # 5. 球在中场 → 斜插右肋等直塞 (急脾气提前跑位)
    ball = obs.ball()
    if ball is not None and abs(ball.position.y) < 0.15 and -0.20 < ball.position.x < 0.40:
        return MoveTo(
            target_x=clip(ball.position.x + 0.15, _X_MIN, _PUSH_X_WHEN_ATTACK),
            target_y=_WIDE_Y,
            urgency="jog",
        )

    # 6. 默认 —— 回到右路默认锚点
    target_x = clip(self_pos.x, 0.05, 0.35)
    return MoveTo(target_x=target_x, target_y=_WIDE_Y, urgency="walk")
