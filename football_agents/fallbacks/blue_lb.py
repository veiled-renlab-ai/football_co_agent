"""周俊 — 蓝队左后卫 #3 fallback (现代攻守平衡型，守纪律).

Fix 1: LB盯防第二威胁（非CB已盯的目标），如果CB已在盯第一威胁，LB取次高。
Fix 2: 越位线意识 — 对方在越位位置时维持防线，不追。
"""
from __future__ import annotations

from ..skills import HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    blue_ranked_threats, clip, clip_target,
    get_defensive_line_x_blue, loose_ball,
    opponent_with_ball, teammate_with_ball, teammate_by_jersey,
)
from .context import FallbackContext

_ANCHOR_X = -0.15           # 默认站位的 x
_ANCHOR_Y = -0.25           # 默认站位的 y (左路)
_X_MAX = 0.25               # 永不过中线超过这个 (套边硬约束)
_Y_MAX = -0.05              # 永不缩到中路
_X_MIN = -0.85              # 不挡 GK 出球

# 蓝队右前卫 player_id (在 TEAM_BLUE_5V5 里 slot 1 → pid 1)
_RM_ENTITY_ID = 1
# 蓝队中后卫 CB entity_id (slot index in blue team array)
_CB_ENTITY_ID = 3  # 高磊的 player_id


def fallback_zhou_jun(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position

    # ---- Fix 2: 计算己方防线x ----
    def_line_x = min(get_defensive_line_x_blue(obs), self_pos.x)

    # ---- Fix 1: LB专责盯防第二威胁（与CB不重叠）----
    threats = blue_ranked_threats(obs)  # 按x升序，最危险在前
    # CB (高磊, entity_id=3) 盯第一威胁，LB取第二
    cb_target_id: int | None = threats[0].entity_id if threats else None
    secondary_target = None
    for t in threats:
        if t.entity_id != cb_target_id:
            secondary_target = t
            break
    # 如果只有一个威胁，两人都盯它（没有第二个可选）
    if secondary_target is None and threats:
        secondary_target = threats[0]

    # Fix 2: 越位线检测 — 对方已在越位位置时维持防线，不追
    if secondary_target is not None and secondary_target.position.x < def_line_x - 0.02:
        tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, _X_MIN, _X_MAX, -0.40, _Y_MAX)
        if abs(self_pos.x - tx) < 0.04 and abs(self_pos.y - ty) < 0.04:
            return HoldPosition()
        return MoveTo(target_x=tx, target_y=ty, urgency="jog")

    # 1. 自家左路有球（近身威胁）→ 顶上去 Press
    carrier = opponent_with_ball(obs)
    if carrier is not None and carrier.position.x < -0.40 and carrier.position.y < 0.0:
        if carrier.distance < 0.20:
            return Press(opponent_id=carrier.entity_id)

    # 2. 对方右边锋带球（我方左路威胁）→ 盯防第二威胁（与CB不重叠）
    if carrier is not None and carrier.position.x < 0.0 and carrier.position.y < -0.05:
        # 如果这个持球人就是CB该盯的第一威胁，LB改盯第二威胁保持站位
        if carrier.entity_id == cb_target_id and secondary_target is not None \
                and secondary_target.entity_id != carrier.entity_id:
            return Mark(opponent_id=secondary_target.entity_id)
        return Mark(opponent_id=carrier.entity_id)

    # 2b. 对方在我方半场但没控球 → LB盯第二威胁
    if secondary_target is not None and secondary_target.position.x < 0.0:
        return Mark(opponent_id=secondary_target.entity_id)

    # 3. 我方在对方半场控球 AND 王浩 RM 还没压上 → 套边助攻
    tm_ball = teammate_with_ball(obs)
    rm_tm = teammate_by_jersey(obs, _RM_ENTITY_ID)
    rm_has_pushed = rm_tm is not None and rm_tm.position.x > 0.30
    if tm_ball is not None and tm_ball.position.x > 0.20 and not rm_has_pushed:
        return MoveTo(target_x=0.15, target_y=_ANCHOR_Y, urgency="jog")

    # 4. 对方反击 (球在我身后) → 冲回本位
    ball = obs.ball()
    if ball is not None and ball.position.x < self_pos.x - 0.05 \
       and carrier is not None:
        return MoveTo(target_x=-0.55, target_y=-0.15, urgency="sprint")

    # 5. 球在远端（我方右路或对方右路）→ 内收半步做第二中卫
    if ball is not None and ball.position.y > 0.15:
        return MoveTo(target_x=-0.25, target_y=-0.12, urgency="jog")

    # 6. 我方左路散球 → 抢 (只限自家半场)
    lb = loose_ball(obs)
    if lb is not None and lb.position.x < 0.0 and lb.position.y < 0.0 and lb.distance < 0.25:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    # 7. 默认 —— 守纪律站住
    if abs(self_pos.x - _ANCHOR_X) < 0.04 and abs(self_pos.y - _ANCHOR_Y) < 0.04:
        return HoldPosition()
    tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, _X_MIN, _X_MAX, -0.40, _Y_MAX)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")
