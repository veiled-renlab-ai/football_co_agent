"""马亮 — 红队中后卫 #5 fallback (速度型激进，敢压高位线)."""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    behind_me_opponent, clip_target, loose_ball, opponent_with_ball,
)
from .context import FallbackContext

_ANCHOR_X = -0.15           # 高位线默认 (比高磊 -0.20 更靠前；Q2 建议 -0.15 同时给 LB 留出纵向差)
_ANCHOR_Y = 0.10
_X_MAX_HIGH = -0.05         # 激进上限 —— 硬约束永不过中线
_PRESS_RADIUS = 0.20        # Q2 建议从 0.25 调到 0.20 (13m→10m 更合理)
_LOOSE_BALL_RADIUS = 0.20   # Q2 同理从 0.35 调到 0.20


def fallback_ma_liang(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position

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
        if carrier.velocity.x < -0.001 and 0.20 < carrier.distance < 0.40:
            return MoveTo(
                target_x=max(carrier.position.x, -0.60),
                target_y=carrier.position.y,
                urgency="sprint",
            )

    # 3. 我方半场散球 → 冲抢 (快速转换)
    lb = loose_ball(obs)
    if lb is not None and lb.position.x < 0.0 and lb.distance < _LOOSE_BALL_RADIUS:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    # 4. 我持球 → 简洁出球 (不喜欢长时间控)
    if obs.self_state.has_ball:
        return DribbleToward(target_x=0.10, target_y=0.15, urgency="jog")

    # 5. 球在对方禁区附近 → 强制归位 (不做无谓长途奔袭)
    ball = obs.ball()
    if ball is not None and ball.position.x > 0.50:
        return MoveTo(target_x=_ANCHOR_X, target_y=_ANCHOR_Y, urgency="jog")

    # 6. 我方队友控球 → 维持高位线
    if obs.self_state.has_ball or any(e.has_ball for e in obs.teammates()):
        if abs(self_pos.x - _ANCHOR_X) < 0.04 and abs(self_pos.y - _ANCHOR_Y) < 0.04:
            return HoldPosition()
        return MoveTo(target_x=_ANCHOR_X, target_y=_ANCHOR_Y, urgency="jog")

    # 7. 默认 —— 高位线站位
    tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, -0.85, _X_MAX_HIGH, -0.15, 0.25)
    if abs(self_pos.x - tx) < 0.04 and abs(self_pos.y - ty) < 0.04:
        return HoldPosition()
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")
