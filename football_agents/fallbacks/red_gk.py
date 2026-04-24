"""赵强 — 红队门将 #1 fallback (现代型，敢出击清道夫)."""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, MoveTo, Press, Skill
from ._helpers import clip, opponent_with_ball
from .context import FallbackContext

_GOAL_LINE_X = -0.95
_SWEEPER_MAX_X = -0.65   # 封顶 —— 越过这里就真离门太远了
_AGGRESSIVE_THRESHOLD_X = -0.80   # 敢出击到禁区线附近


def fallback_zhao_qiang(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    ball = obs.ball()
    carrier = opponent_with_ball(obs)

    # 0. 球回到对方半场 → 立即归位 (防止"敢出击"弱点)
    if ball is not None and ball.position.x > 0.30:
        return MoveTo(target_x=_GOAL_LINE_X, target_y=0.0, urgency="jog")

    # 1. 禁区内对方持球 → 扑上去封
    if carrier is not None and carrier.position.x < -0.75:
        # 禁区内有 ≥2 对方 → 不冲，保持位置
        n_opp_in_box = sum(
            1 for e in obs.opponents() if e.position.x < -0.75
        )
        if n_opp_in_box < 2:
            return Press(opponent_id=carrier.entity_id)

    # 2. 持球人已过半场进我方但还没到禁区 → 清道夫出击
    if carrier is not None and carrier.position.x < -0.55:
        if carrier.distance < 0.25:
            return Press(opponent_id=carrier.entity_id)

    # 3. 我持球 → 带一下找传球角度 (脚下细腻现代门将)
    if obs.self_state.has_ball:
        return DribbleToward(target_x=-0.60, target_y=0.0, urgency="jog")

    # 4. 队友在中前场控球 → 跟球横向小幅移动 (清道夫站位)
    if ball is not None and ball.position.x > -0.30:
        return MoveTo(
            target_x=_GOAL_LINE_X,
            target_y=clip(ball.position.y * 0.3, -0.15, 0.15),
            urgency="jog",
        )

    # 5. 默认 —— 默认站在禁区弧顶附近，比林涛激进一档
    self_pos = obs.self_state.position
    if self_pos.x < -0.92 and abs(self_pos.y) < 0.05:
        return HoldPosition()
    return MoveTo(target_x=-0.90, target_y=0.0, urgency="jog")
