"""孙斌 — 红队左后卫 #2 fallback (硬朗型，逼抢凶最先压上).

Fix 1: LB盯防第二威胁（非CB已盯的目标，与red_cb不重叠）。
Fix 2: 越位线意识 — 对方在越位位置（x > defensive_line_x）时维持防线。
"""
from __future__ import annotations

from ..skills import HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    clip, clip_target, get_defensive_line_x_red, loose_ball,
    opponent_with_ball, red_ranked_threats, teammate_with_ball,
    teammate_by_jersey,
)
from .context import FallbackContext

_ANCHOR_X = -0.10
_ANCHOR_Y = -0.10
_X_MAX = -0.05
_Y_MAX = -0.05

_RM_ENTITY_ID = 1       # 刘锋 RM 在 TEAM_RED_5V5 的 slot 1
# 红队CB entity_id（马亮，slot index in red team array）
_CB_ENTITY_ID = 3


def fallback_sun_bin(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    carrier = opponent_with_ball(obs)

    # ---- Fix 2: 计算红队防线x ----
    def_line_x = max(get_defensive_line_x_red(obs), self_pos.x)

    # ---- Fix 1: LB专责盯防第二威胁（与CB不重叠）----
    threats = red_ranked_threats(obs)  # 按x降序，最危险在前
    cb_target_id: int | None = threats[0].entity_id if threats else None
    secondary_target = None
    for t in threats:
        if t.entity_id != cb_target_id:
            secondary_target = t
            break
    if secondary_target is None and threats:
        secondary_target = threats[0]

    # Fix 2: 越位线检测 — 对方在越位位置时维持防线，不追
    if secondary_target is not None and secondary_target.position.x > def_line_x + 0.02:
        tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, -0.85, _X_MAX, -0.40, _Y_MAX)
        if abs(self_pos.x - tx) < 0.04 and abs(self_pos.y - ty) < 0.04:
            return HoldPosition()
        return MoveTo(target_x=tx, target_y=ty, urgency="jog")

    # 1. 对方左路持球 + 距离近 → Press (硬朗优先 Press 而不是 Mark)
    if carrier is not None and carrier.position.y < 0.05 and carrier.position.x > -0.20:
        if carrier.distance < 0.25:
            if self_pos.x > 0.70:
                return Mark(opponent_id=carrier.entity_id)
            return Press(opponent_id=carrier.entity_id)

    # 2. 对方左路持球但稍远 → 盯防第二威胁（与CB不重叠）
    if carrier is not None and carrier.position.x > 0.0 and carrier.position.y < 0.05:
        # 如果持球人就是CB该盯的第一威胁，LB盯第二威胁
        if carrier.entity_id == cb_target_id and secondary_target is not None \
                and secondary_target.entity_id != carrier.entity_id:
            return Mark(opponent_id=secondary_target.entity_id)
        return MoveTo(
            target_x=clip(carrier.position.x, -0.70, _X_MAX),
            target_y=clip(carrier.position.y, -0.40, _Y_MAX),
            urgency="sprint",
        )

    # 2b. 对方在红队半场且无持球 → LB盯第二威胁
    if secondary_target is not None and secondary_target.position.x > 0.0:
        return Mark(opponent_id=secondary_target.entity_id)

    # 3. 我方左路散球 → sprint 直接抢
    lb = loose_ball(obs)
    if lb is not None and lb.position.x > -0.2 and lb.position.y < 0.10 and lb.distance < 0.30:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    # 4. 我方控球反击 + 刘锋 RM 未压上 → 简单支援出球点
    tm_ball = teammate_with_ball(obs)
    rm_tm = teammate_by_jersey(obs, _RM_ENTITY_ID)
    rm_pushed = rm_tm is not None and rm_tm.position.x < -0.30
    if tm_ball is not None and tm_ball.position.x < -0.10 and not rm_pushed and self_pos.x > 0.0:
        return MoveTo(target_x=-0.05, target_y=-0.15, urgency="jog")

    # 5. 同侧互斥保护：如果 RM 已经很前了，我就守住别压
    if rm_pushed and self_pos.x < 0.05:
        return HoldPosition()

    # 6. 默认 —— 回出生点
    if abs(self_pos.x - _ANCHOR_X) < 0.04 and abs(self_pos.y - _ANCHOR_Y) < 0.04:
        return HoldPosition()
    tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, -0.85, _X_MAX, -0.40, _Y_MAX)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")
