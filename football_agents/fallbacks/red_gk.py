"""赵强 — 红队门将 #1 fallback (现代型，敢出击清道夫).

Fix 4: 对方进攻时（ball.x > 0.70）GK可稍微前出到禁区顶（x≈0.88），
       但如果对方球员已进入禁区（x > 0.75），立即退到0.95。
"""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, MoveTo, Press, Skill
from ._helpers import clip, opponent_with_ball
from .context import FallbackContext

_GOAL_LINE_X = -0.95       # 红队守门侧实际在+x=0.95（gfootball坐标系：红队为right_team）
                           # 注：在感知层，红队在god_view["right_team"]侧，
                           # 但EgocentricFilter将红队坐标翻转为left_team视角。
                           # 实际上红队GK的目标应守在x=+0.95，但代码中统一按
                           # left_team视角：GK守-0.95（蓝队和红队均如此）。
_SWEEPER_MAX_X = -0.65
_AGGRESSIVE_THRESHOLD_X = -0.80
_BOX_EDGE_X = -0.75        # 禁区线
_ADVANCED_GK_X = -0.88     # Fix 4: 前出到禁区顶
_RETREAT_X = -0.95         # Fix 4: 对方进禁区时退守


def fallback_zhao_qiang(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    ball = obs.ball()
    carrier = opponent_with_ball(obs)
    self_pos = obs.self_state.position

    # 0. 球回到对方半场 → 立即归位
    if ball is not None and ball.position.x > 0.30:
        return MoveTo(target_x=_GOAL_LINE_X, target_y=0.0, urgency="jog")

    # Fix 4: 如果对方已进入禁区，立即退到-0.95
    if carrier is not None and carrier.position.x < _BOX_EDGE_X:
        return MoveTo(
            target_x=_RETREAT_X,
            target_y=clip(carrier.position.y * 0.5, -0.08, 0.08),
            urgency="sprint",
        )

    # Fix 4: 对方在进攻路上但还没到禁区 → 前出到禁区顶封角度
    if ball is not None and ball.position.x < -0.70:
        opp_in_box = any(
            e.position.x < _BOX_EDGE_X for e in obs.opponents()
        )
        if opp_in_box:
            return MoveTo(
                target_x=_RETREAT_X,
                target_y=clip(ball.position.y * 0.4, -0.08, 0.08),
                urgency="sprint",
            )
        else:
            # 禁区外，前出到禁区顶封压角度
            return MoveTo(
                target_x=_ADVANCED_GK_X,
                target_y=clip(ball.position.y * 0.4, -0.08, 0.08),
                urgency="jog",
            )

    # 1. 禁区内对方持球 → 扑上去封
    if carrier is not None and carrier.position.x < _BOX_EDGE_X:
        n_opp_in_box = sum(
            1 for e in obs.opponents() if e.position.x < _BOX_EDGE_X
        )
        if n_opp_in_box < 2:
            return Press(opponent_id=carrier.entity_id)

    # 2. 持球人已过半场进我方但还没到禁区 → 清道夫出击
    if carrier is not None and carrier.position.x < -0.55:
        if carrier.distance < 0.25:
            return Press(opponent_id=carrier.entity_id)

    # 3. 我持球 → 带一下找传球角度
    if obs.self_state.has_ball:
        return DribbleToward(target_x=-0.60, target_y=0.0, urgency="jog")

    # 4. 队友在中前场控球 → 跟球横向小幅移动
    if ball is not None and ball.position.x > -0.30:
        return MoveTo(
            target_x=_GOAL_LINE_X,
            target_y=clip(ball.position.y * 0.3, -0.15, 0.15),
            urgency="jog",
        )

    # 5. 默认 —— 默认站在禁区弧顶附近
    if self_pos.x < -0.92 and abs(self_pos.y) < 0.05:
        return HoldPosition()
    return MoveTo(target_x=-0.90, target_y=0.0, urgency="jog")
