"""马亮 — 红队中后卫 #5 fallback (速度型激进，敢压高位线).

Fix 1: CB专责盯防第一威胁（距红队球门最近=x最大的对方球员）。
Fix 2: 越位线意识 — 对方前锋已越过红队防线（x > defensive_line_x）时维持防线。
"""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    behind_me_opponent, red_ranked_threats, clip_target,
    get_defensive_line_x_red, loose_ball, opponent_with_ball,
)
from .context import FallbackContext

_ANCHOR_X = -0.15           # 高位线默认
_ANCHOR_Y = 0.10
_X_MAX_HIGH = -0.05         # 激进上限 —— 硬约束永不过中线
_PRESS_RADIUS = 0.20
_LOOSE_BALL_RADIUS = 0.20

# 红队LB entity_id（孙斌，slot index in red team array）
_LB_ENTITY_ID = 2


def fallback_ma_liang(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position

    # ---- Fix 2: 计算红队防线x ----
    # 红队防线x = 己方队友最大x（最深防守者，红队守门在+x侧）
    def_line_x = max(get_defensive_line_x_red(obs), self_pos.x)

    # ---- Fix 1: CB专责盯防第一威胁 ----
    threats = red_ranked_threats(obs)  # 按x降序，最危险（x最大）在前
    primary_target = threats[0] if threats else None

    # Fix 2: 越位线检测 — 对方在越位位置（x > def_line_x）时维持防线，不追
    if primary_target is not None and primary_target.position.x > def_line_x + 0.02:
        tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, -0.85, _X_MAX_HIGH, -0.15, 0.25)
        if abs(self_pos.x - tx) < 0.04 and abs(self_pos.y - ty) < 0.04:
            return HoldPosition()
        return MoveTo(target_x=tx, target_y=ty, urgency="jog")

    # 0. (硬短路) 身后有人 → 必须 Mark，覆盖所有激进压上
    behind = behind_me_opponent(obs, y_tolerance=0.12)
    if behind is not None:
        return Mark(opponent_id=behind.entity_id)

    # 1. 对方在我方半场持球 + 够近 → 提前出脚 Press (激进预判)
    carrier = opponent_with_ball(obs)
    if carrier is not None and carrier.position.x < 0.0 and carrier.distance < _PRESS_RADIUS:
        return Press(opponent_id=carrier.entity_id)

    # 2. 对方持球朝我球门推进 → 切线封堵
    if carrier is not None and carrier.velocity is not None:
        if carrier.velocity.x > 0.001 and 0.20 < carrier.distance < 0.40:
            return MoveTo(
                target_x=min(carrier.position.x, 0.60),
                target_y=carrier.position.y,
                urgency="sprint",
            )

    # 3. 对方在我方半场 → CB专责盯第一威胁
    if carrier is not None and carrier.position.x > 0.0:
        if primary_target is not None:
            return Mark(opponent_id=primary_target.entity_id)

    # 4. 我方半场散球 → 冲抢 (快速转换)
    lb = loose_ball(obs)
    if lb is not None and lb.position.x > 0.0 and lb.distance < _LOOSE_BALL_RADIUS:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    # 5. 我持球 → 简洁出球 (不喜欢长时间控)
    if obs.self_state.has_ball:
        return DribbleToward(target_x=0.10, target_y=0.15, urgency="jog")

    # 6. 球在对方禁区附近 → 强制归位 (不做无谓长途奔袭)
    ball = obs.ball()
    if ball is not None and ball.position.x < -0.50:
        return MoveTo(target_x=_ANCHOR_X, target_y=_ANCHOR_Y, urgency="jog")

    # 7. 我方队友控球 → 维持高位线
    if obs.self_state.has_ball or any(e.has_ball for e in obs.teammates()):
        if abs(self_pos.x - _ANCHOR_X) < 0.04 and abs(self_pos.y - _ANCHOR_Y) < 0.04:
            return HoldPosition()
        return MoveTo(target_x=_ANCHOR_X, target_y=_ANCHOR_Y, urgency="jog")

    # 8. 默认 —— 高位线站位
    tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, -0.85, _X_MAX_HIGH, -0.15, 0.25)
    if abs(self_pos.x - tx) < 0.04 and abs(self_pos.y - ty) < 0.04:
        return HoldPosition()
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")
