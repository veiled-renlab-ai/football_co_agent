"""刘锋 — 红队右前卫 #7 fallback (工兵型，跑动量大逼抢凶)."""
from __future__ import annotations

from ..skills import HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    clip, clip_target, loose_ball, nearest_opponent,
    opponent_with_ball, teammate_with_ball,
)
from .context import FallbackContext

_WIDE_Y = 0.30           # 工兵型也要拉宽度
_Y_MIN = 0.15            # 永不缩中
_PRESS_RADIUS = 0.16     # 比通用大但没王浩那么保守 (刘锋跑不死)


def fallback_liu_feng(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position

    # 1. 对方在我方右路前/中场持球 → 主动 Press (对手威胁触发，不等)
    carrier = opponent_with_ball(obs)
    if carrier is not None and carrier.position.x > -0.20 and carrier.position.y > -0.10:
        if carrier.distance < _PRESS_RADIUS:
            return Press(opponent_id=carrier.entity_id)
        # 稍远的也冲过去 —— 工兵覆盖型
        return MoveTo(
            target_x=clip(carrier.position.x, -0.20, 0.60),
            target_y=clip(carrier.position.y, _Y_MIN, 0.42),
            urgency="sprint",
        )

    # 2. 右半区散球 + 不太远 → 冲抢 (跑不死 = 二点球必到)
    lb = loose_ball(obs)
    if lb is not None and lb.position.y > -0.15 and lb.position.x > -0.30:
        if lb.distance < 0.35:
            return MoveTo(
                target_x=lb.position.x,
                target_y=lb.position.y,
                urgency="sprint",
            )

    # 3. 对方中/左路持球 → 封住右路出球线 (集体压迫)
    if carrier is not None:
        opp_rb = nearest_opponent(obs)  # 最近对手当 Mark 目标兜底
        if opp_rb is not None and opp_rb.position.y > 0.05:
            return Mark(opponent_id=opp_rb.entity_id)

    # 4. 我方队友控球且在对方半场 → 拉右肋等直塞
    tm_ball = teammate_with_ball(obs)
    if tm_ball is not None and tm_ball.position.x > 0.0:
        if self_pos.x > 0.75:   # 已太深，不再前插免越位
            return HoldPosition()
        return MoveTo(target_x=0.60, target_y=0.35, urgency="sprint")

    # 5. 我方半场控球 → 提前站高拉宽，不回撤接球
    if tm_ball is not None:
        return MoveTo(target_x=0.20, target_y=_WIDE_Y, urgency="jog")

    # 6. 静态 → 默认贴右边线拉宽度 (纠正站位偏中)
    target_y = _WIDE_Y if self_pos.y < _Y_MIN else self_pos.y
    target_x = clip(self_pos.x, -0.10, 0.30)
    return MoveTo(target_x=target_x, target_y=target_y, urgency="walk")
