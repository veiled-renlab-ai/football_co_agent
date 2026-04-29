"""陈宇 — 蓝队中锋 #9 fallback (全能型，习惯抢点会回撤接应).

Fix 3: 丢球后立即sprint回防守位置，不再用walk缓慢回位。
       对方控球且朝蓝队球门推进（ball.x向负方向移动）时CF sprint到x≤0.20。
"""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, MoveTo, Skill
from ._helpers import (
    ball_owned_by_opponent, clip, clip_target,
    opponent_with_ball, teammate_with_ball,
)
from .context import FallbackContext

_ANCHOR_X = 0.55         # 默认站位
_PUSH_X = 0.75           # 我方推进时前顶到这里
_RETREAT_MIN_X = 0.10    # 回撤底限
_Y_LIMIT = 0.20          # 横向不越过这里，给右前卫王浩留通道

# Fix 3: 防守回位线（对方控球时CF必须回到此x以内）
_DEFENSIVE_RETREAT_X = 0.20


def fallback_chen_yu(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    ball = obs.ball()
    tm_ball = teammate_with_ball(obs)
    carrier = opponent_with_ball(obs)

    # ---- Fix 3: 对方控球时立即sprint回防守位置 ----
    opp_has_ball = ball_owned_by_opponent(obs)
    if opp_has_ball and carrier is not None:
        # 判断对方是否在向蓝队球门方向推进（ball在蓝队半场，或carrier朝-x方向）
        ball_in_blue_half = ball is not None and ball.position.x < 0.0
        carrier_advancing = (
            carrier.velocity is not None and carrier.velocity.x < -0.001
        ) or (carrier.position.x < 0.30)
        if (ball_in_blue_half or carrier_advancing) and self_pos.x > _DEFENSIVE_RETREAT_X:
            # 丢球后sprint回防守位置
            return MoveTo(
                target_x=_DEFENSIVE_RETREAT_X,
                target_y=0.0,
                urgency="sprint",
            )
        elif self_pos.x > _DEFENSIVE_RETREAT_X + 0.10:
            # 对方刚拿球但还在对方半场，也要开始回位（jog）
            return MoveTo(
                target_x=_DEFENSIVE_RETREAT_X + 0.10,
                target_y=0.0,
                urgency="jog",
            )

    # 1. 我持球 + 在对方半场 → 背身推进稳节奏
    if obs.self_state.has_ball and self_pos.x > 0.0:
        return DribbleToward(target_x=0.90, target_y=0.0, urgency="jog")

    # 2. 我方控球 + 持球人在中前场 + 我还没到前压线 → 顶到对方中卫高度
    if tm_ball is not None and tm_ball.position.x > 0.0 and self_pos.x < _PUSH_X - 0.05:
        opps = obs.opponents()
        if opps:
            deepest = max(opps, key=lambda e: e.position.x)
            target_x = clip(deepest.position.x - 0.05, _ANCHOR_X, _PUSH_X)
        else:
            target_x = _PUSH_X
        target_y = clip(tm_ball.position.y, -_Y_LIMIT, _Y_LIMIT)
        return MoveTo(target_x=target_x, target_y=target_y, urgency="jog")

    # 3. 我方后场在被紧逼组织 → 回撤接应 (全能中锋的独特技能)
    if tm_ball is not None and tm_ball.position.x < -0.20:
        nearest_opp_to_carrier = None
        for o in obs.opponents():
            if nearest_opp_to_carrier is None or \
               (o.position.x - tm_ball.position.x) ** 2 + (o.position.y - tm_ball.position.y) ** 2 < \
               (nearest_opp_to_carrier.position.x - tm_ball.position.x) ** 2 + (nearest_opp_to_carrier.position.y - tm_ball.position.y) ** 2:
                nearest_opp_to_carrier = o
        if nearest_opp_to_carrier is not None:
            dist_sq = (nearest_opp_to_carrier.position.x - tm_ball.position.x) ** 2 + \
                      (nearest_opp_to_carrier.position.y - tm_ball.position.y) ** 2
            if dist_sq < 0.015:
                return MoveTo(target_x=_RETREAT_MIN_X, target_y=0.0, urgency="jog")

    # 4. 球到对方禁区附近 + 我还没在禁区 → 抢点 (嗅觉敏锐)
    if ball is not None and ball.position.x > 0.60 and self_pos.x < 0.70:
        target_y = 0.12 if ball.position.y > 0 else -0.12
        return MoveTo(target_x=0.85, target_y=target_y, urgency="sprint")

    # 5. 对方控球在我方半场且未触发Fix 3 → 留在防守位置
    if carrier is not None and carrier.position.x < 0.0:
        anchor = clip(_ANCHOR_X - 0.20, _RETREAT_MIN_X, _DEFENSIVE_RETREAT_X)
        return MoveTo(target_x=anchor, target_y=0.0, urgency="jog")

    # 6. 默认 —— 回到进攻箭头位置
    target_x = clip(self_pos.x, _RETREAT_MIN_X, _ANCHOR_X)
    target_y = clip(self_pos.y, -_Y_LIMIT, _Y_LIMIT)
    if abs(target_x - _ANCHOR_X) < 0.03 and abs(target_y) < 0.04:
        return HoldPosition()
    return MoveTo(target_x=_ANCHOR_X, target_y=0.0, urgency="walk")
