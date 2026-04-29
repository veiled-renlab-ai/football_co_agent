"""高磊 — 蓝队中后卫 #4 fallback (经验型队长，位置感优先).

Fix 1: CB盯防威胁最高的对方球员（距蓝门最近=x最小），让LB去盯第二威胁。
Fix 2: 越位线意识 — 当对方前锋已越过己方防线（x < defensive_line_x）时，
       维持防线位置而非追上去，把对方暴露在越位位置。
"""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    behind_me_opponent, blue_ranked_threats, clip_target,
    get_defensive_line_x_blue, loose_ball, opponent_with_ball,
)
from .context import FallbackContext

_ANCHOR_X = -0.20           # 默认站位
_ANCHOR_Y = 0.10
_X_MAX = 0.0                # 硬约束：永不过中线
_X_MIN = -0.85              # 不挡 GK

# CB负责盯防第一威胁（entity_id index 0 in threats list）
# LB entity_id in blue team array = 2 (slot index, matches blue_lb persona)
_LB_ENTITY_ID = 2


def fallback_gao_lei(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position

    # ---- Fix 2: 计算己方防线x ----
    # 蓝队防线x = 己方队友最小x（最深防守者）与自身位置取最小
    def_line_x = min(get_defensive_line_x_blue(obs), self_pos.x)

    # ---- Fix 1: CB专责盯防第一威胁 ----
    threats = blue_ranked_threats(obs)  # 按x升序，最危险在前
    primary_target = threats[0] if threats else None

    # Fix 2: 越位线检测 — 对方已在越位位置（x < def_line_x）时维持防线，不追
    if primary_target is not None and primary_target.position.x < def_line_x - 0.02:
        # 对方在越位位置，维持防线高度，不追上去
        tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, _X_MIN, _X_MAX, -0.15, 0.25)
        if abs(self_pos.x - tx) < 0.04 and abs(self_pos.y - ty) < 0.04:
            return HoldPosition()
        return MoveTo(target_x=tx, target_y=ty, urgency="jog")

    # 0. (硬短路) 有对方前锋在我身后 → 盯人优先，覆盖一切其他规则
    behind = behind_me_opponent(obs, y_tolerance=0.12)
    if behind is not None:
        return Mark(opponent_id=behind.entity_id)

    # 1. 球进自家禁区 + 有持球人 → Press
    carrier = opponent_with_ball(obs)
    ball = obs.ball()
    if carrier is not None and ball is not None \
       and ball.position.x < -0.75 and carrier.distance < 0.15:
        return Press(opponent_id=carrier.entity_id)

    # 2. 我持球在后场 → 往中场带一步找传球
    if obs.self_state.has_ball and self_pos.x < -0.30:
        return DribbleToward(target_x=0.0, target_y=0.10, urgency="jog")

    # 3. 对方控球推进到我方半场 → CB专责盯第一威胁，让LB管第二
    if carrier is not None and -0.50 < carrier.position.x < 0.0:
        if primary_target is not None:
            return Mark(opponent_id=primary_target.entity_id)
        if abs(self_pos.x - _ANCHOR_X) > 0.05 or abs(self_pos.y - _ANCHOR_Y) > 0.05:
            return MoveTo(target_x=_ANCHOR_X, target_y=_ANCHOR_Y, urgency="jog")
        return HoldPosition()

    # 3b. 对方有控球但不在推进区 → 占好位置
    if carrier is not None and carrier.position.x >= 0.0:
        if primary_target is not None and primary_target.position.x < 0.0:
            return Mark(opponent_id=primary_target.entity_id)

    # 4. 自家禁区边缘散球 → 扑 (但只限禁区内，不追出去)
    lb = loose_ball(obs)
    if lb is not None and lb.position.x < -0.60 and lb.distance < 0.20:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    # 5. 默认 —— 回到锚点站着等
    if abs(self_pos.x - _ANCHOR_X) < 0.04 and abs(self_pos.y - _ANCHOR_Y) < 0.04:
        return HoldPosition()
    tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, _X_MIN, _X_MAX, -0.15, 0.25)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")
