"""孙斌 — 红队左后卫 #2 fallback (硬朗型，逼抢凶最先压上)."""
from __future__ import annotations

from ..skills import HoldPosition, Mark, MoveTo, Press, Skill
from ._helpers import (
    clip, clip_target, loose_ball, opponent_with_ball, teammate_with_ball,
    teammate_by_jersey,
)
from .context import FallbackContext

_ANCHOR_X = -0.10
_ANCHOR_Y = -0.10
_X_MAX = -0.05          # 硬约束 (subagent Q2 修正：从 0.1 调到 -0.05 免冲到 CB 前)
_Y_MAX = -0.05

_RM_ENTITY_ID = 1       # 刘锋 RM 在 TEAM_RED_5V5 的 slot 1


def fallback_sun_bin(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    carrier = opponent_with_ball(obs)

    # 1. 对方左路持球 + 距离近 → Press (硬朗优先 Press 而不是 Mark)
    if carrier is not None and carrier.position.y < 0.05 and carrier.position.x < 0.20:
        if carrier.distance < 0.25:
            # 禁区深处不飞铲 —— 用 Mark 卡位免点球
            if self_pos.x < -0.70:
                return Mark(opponent_id=carrier.entity_id)
            return Press(opponent_id=carrier.entity_id)

    # 2. 对方左路持球但稍远 → 仍然顶上去 (而不是 Mark)
    if carrier is not None and carrier.position.x < 0.0 and carrier.position.y < 0.05:
        return MoveTo(
            target_x=clip(carrier.position.x, -0.70, _X_MAX),
            target_y=clip(carrier.position.y, -0.40, _Y_MAX),
            urgency="sprint",
        )

    # 3. 我方左路散球 → sprint 直接抢 (不等、不看、不想)
    lb = loose_ball(obs)
    if lb is not None and lb.position.x < 0.2 and lb.position.y < 0.10 and lb.distance < 0.30:
        return MoveTo(target_x=lb.position.x, target_y=lb.position.y, urgency="sprint")

    # 4. 我方控球反击 + 刘锋 RM 未压上 → 简单支援出球点
    tm_ball = teammate_with_ball(obs)
    rm_tm = teammate_by_jersey(obs, _RM_ENTITY_ID)
    rm_pushed = rm_tm is not None and rm_tm.position.x > 0.30
    if tm_ball is not None and tm_ball.position.x > 0.10 and not rm_pushed and self_pos.x < 0.0:
        return MoveTo(target_x=0.05, target_y=-0.15, urgency="jog")

    # 5. 同侧互斥保护：如果 RM 已经很前了，我就守住别压
    if rm_pushed and self_pos.x > -0.05:
        return HoldPosition()

    # 6. 默认 —— 回出生点 (简洁不墨迹)
    if abs(self_pos.x - _ANCHOR_X) < 0.04 and abs(self_pos.y - _ANCHOR_Y) < 0.04:
        return HoldPosition()
    tx, ty = clip_target(_ANCHOR_X, _ANCHOR_Y, -0.85, _X_MAX, -0.40, _Y_MAX)
    return MoveTo(target_x=tx, target_y=ty, urgency="jog")
