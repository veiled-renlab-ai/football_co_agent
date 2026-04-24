"""林涛 — 蓝队门将 #1 fallback (经验型，不轻易出击)."""
from __future__ import annotations

from ..skills import HoldPosition, MoveTo, Skill
from ._helpers import clip, loose_ball, opponent_with_ball
from .context import FallbackContext

_GOAL_LINE_X = -0.98
_NEVER_PAST_X = -0.80   # 永远不出禁区弧顶 (林涛死守型比赵强的 -0.65 更保守)


def fallback_lin_tao(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    ball = obs.ball()

    # 1. Panic save: 球已在小禁区内向我方球门推进 → 扑救
    if ball is not None and ball.position.x < -0.88 and ball.distance < 0.10:
        if ball.velocity is not None and ball.velocity.x < -0.003:
            tx = clip(ball.position.x - 0.02, _GOAL_LINE_X, -0.85)
            ty = clip(ball.position.y, -0.10, 0.10)
            return MoveTo(target_x=tx, target_y=ty, urgency="sprint")

    # 2. 禁区内散球 → 出来抢点 (只在禁区内，不越线)
    lb = loose_ball(obs)
    if lb is not None and lb.position.x < -0.75 and abs(lb.position.y) < 0.20:
        return MoveTo(
            target_x=clip(lb.position.x, _GOAL_LINE_X, _NEVER_PAST_X),
            target_y=lb.position.y,
            urgency="sprint",
        )

    # 3. 对方在我半场深处持球 → 门线上横向移动封角度 (不出来)
    carrier = opponent_with_ball(obs)
    if carrier is not None and carrier.position.x < -0.40:
        return MoveTo(
            target_x=-0.95,
            target_y=clip(carrier.position.y * 0.5, -0.08, 0.08),
            urgency="jog",
        )

    # 4. 我偏离了门线中心 → 慢慢归位
    if abs(self_pos.x - _GOAL_LINE_X) > 0.05 or abs(self_pos.y) > 0.03:
        return MoveTo(target_x=_GOAL_LINE_X, target_y=0.0, urgency="jog")

    # 5. 默认 —— 站住不动 (经验型门将省体力)
    return HoldPosition()
