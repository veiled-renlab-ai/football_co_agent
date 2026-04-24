"""周俊 — 蓝队左后卫 #3 fallback (现代攻守平衡型，守纪律)."""
from __future__ import annotations

from ..skills import HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    clip, clip_target, loose_ball, opponent_with_ball, teammate_with_ball,
    teammate_by_jersey,
)
from .context import FallbackContext

_ANCHOR_X = -0.15           # 默认站位的 x
_ANCHOR_Y = -0.25           # 默认站位的 y (左路)
_X_MAX = 0.25               # 永不过中线超过这个 (套边硬约束)
_Y_MAX = -0.05              # 永不缩到中路
_X_MIN = -0.85              # 不挡 GK 出球

# 蓝队右前卫 player_id (在 TEAM_BLUE_5V5 里 slot 1 → pid 1 在多 agent 模式下)
# 注：perception 里的 entity_id 就是 team-array index，这里用来查 RM 是否已压上
_RM_ENTITY_ID = 1


def fallback_zhou_jun(ctx: FallbackContext) -> Skill:
    obs = ctx.obs

    # 1. 自家左路有球（近身威胁）→ 顶上去 Press
    carrier = opponent_with_ball(obs)
    if carrier is not None and carrier.position.x < -0.40 and carrier.position.y < 0.0:
        if carrier.distance < 0.20:
            return Press(opponent_id=carrier.entity_id)

    # 2. 对方右边锋带球（我方左路威胁）→ Mark 封下底传中
    if carrier is not None and carrier.position.x < 0.0 and carrier.position.y < -0.05:
        return Mark(opponent_id=carrier.entity_id)

    # 3. 我方在对方半场控球 AND 王浩 RM 还没压上 → 套边助攻
    tm_ball = teammate_with_ball(obs)
    rm_tm = teammate_by_jersey(obs, _RM_ENTITY_ID)
    rm_has_pushed = rm_tm is not None and rm_tm.position.x > 0.30
    if tm_ball is not None and tm_ball.position.x > 0.20 and not rm_has_pushed:
        return MoveTo(target_x=0.15, target_y=_ANCHOR_Y, urgency="jog")

    # 4. 对方反击 (球在我身后) → 冲回本位
    ball = obs.ball()
    if ball is not None and ball.position.x < obs.self_state.position.x - 0.05 \
       and carrier is not None:
        return MoveTo(target_x=-0.55, target_y=-0.15, urgency="sprint")

    # 5. 球在远端（我方右路或对方右路）→ 内收半步做第二中卫
    if ball is not None and ball.position.y > 0.15:
        return MoveTo(target_x=-0.25, target_y=-0.12, urgency="jog")

    # 6. 我方左路散球 → 抢 (只限自家半场)
    lb = loose_ball(obs)
    if lb is not None and lb.position.x < 0.0 and lb.position.y < 0.0 and lb.distance < 0.25:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    # 7. 默认 —— 守纪律站住 (区别于孙斌激进型)
    self_pos = obs.self_state.position
    if abs(self_pos.x - _ANCHOR_X) < 0.04 and abs(self_pos.y - _ANCHOR_Y) < 0.04:
        return HoldPosition()
    tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, _X_MIN, _X_MAX, -0.40, _Y_MAX)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")
