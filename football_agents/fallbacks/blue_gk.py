"""林涛 — 蓝队门将 #1 fallback (经验型，不轻易出击).

Fix 4: 对方进攻时（ball.x < -0.70）GK可稍微前出到禁区顶（x≈-0.88），
       但如果对方球员已进入禁区（x < -0.75），立即退到-0.95。
"""
from __future__ import annotations

from ..skills import HoldPosition, MoveTo, Skill
from ._helpers import clip, loose_ball, opponent_with_ball
from .context import FallbackContext

_GOAL_LINE_X = -0.98
_NEVER_PAST_X = -0.80   # 永远不出禁区弧顶
_BOX_EDGE_X = -0.75     # 禁区线（对方球员进入此区域即为禁区威胁）
_ADVANCED_GK_X = -0.88  # Fix 4: 对方进攻时前出到禁区顶附近
_RETREAT_X = -0.95      # Fix 4: 对方进禁区时退到此位置


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
    if lb is not None and lb.position.x < _BOX_EDGE_X and abs(lb.position.y) < 0.20:
        return MoveTo(
            target_x=clip(lb.position.x, _GOAL_LINE_X, _NEVER_PAST_X),
            target_y=lb.position.y,
            urgency="sprint",
        )

    # 3. 对方在我半场深处持球 → 封角度
    carrier = opponent_with_ball(obs)

    # Fix 4: 如果对方已进入禁区，立即退到-0.95
    if carrier is not None and carrier.position.x < _BOX_EDGE_X:
        target_x = clip(
            _RETREAT_X,
            _GOAL_LINE_X,
            -0.85,
        )
        return MoveTo(
            target_x=target_x,
            target_y=clip(carrier.position.y * 0.5, -0.08, 0.08),
            urgency="sprint",
        )

    # Fix 4: 对方在进攻路上但还没到禁区 → 前出到禁区顶封角度
    if ball is not None and ball.position.x < -0.70:
        # 检查是否有对方球员在禁区内
        opp_in_box = any(
            e.position.x < _BOX_EDGE_X for e in obs.opponents()
        )
        if opp_in_box:
            # 有对方在禁区，退到-0.95
            return MoveTo(
                target_x=_RETREAT_X,
                target_y=clip(ball.position.y * 0.4, -0.08, 0.08),
                urgency="sprint",
            )
        else:
            # 禁区外，前出到禁区顶（-0.88）封压角度
            return MoveTo(
                target_x=_ADVANCED_GK_X,
                target_y=clip(ball.position.y * 0.4, -0.08, 0.08),
                urgency="jog",
            )

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
