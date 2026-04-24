"""陈宇 — 蓝队中锋 #9 fallback (全能型，习惯抢点会回撤接应)."""
from __future__ import annotations

from ..skills import DribbleToward, HoldPosition, MoveTo, Skill
from ._helpers import clip, clip_target, opponent_with_ball, teammate_with_ball
from .context import FallbackContext

_ANCHOR_X = 0.55         # 默认站位 (subagent 建议从 +0.35 调深至 +0.55)
_PUSH_X = 0.75           # 我方推进时前顶到这里
_RETREAT_MIN_X = 0.10    # 回撤底限 (subagent 建议从 -0.05 抬到 +0.10)
_Y_LIMIT = 0.20          # 横向不越过这里，给右前卫王浩留通道


def fallback_chen_yu(ctx: FallbackContext) -> Skill:
    obs = ctx.obs
    self_pos = obs.self_state.position
    ball = obs.ball()
    tm_ball = teammate_with_ball(obs)

    # 1. 我持球 + 在对方半场 → 背身推进稳节奏
    if obs.self_state.has_ball and self_pos.x > 0.0:
        return DribbleToward(target_x=0.90, target_y=0.0, urgency="jog")

    # 2. 我方控球 + 持球人在中前场 + 我还没到前压线 → 顶到对方中卫高度
    if tm_ball is not None and tm_ball.position.x > 0.0 and self_pos.x < _PUSH_X - 0.05:
        # 找对方最深处的防守者做参考，往他身前挤
        opps = obs.opponents()
        if opps:
            deepest = max(opps, key=lambda e: e.position.x)
            target_x = clip(deepest.position.x - 0.05, _ANCHOR_X, _PUSH_X)
        else:
            target_x = _PUSH_X
        target_y = clip(tm_ball.position.y, -_Y_LIMIT, _Y_LIMIT)
        return MoveTo(target_x=target_x, target_y=target_y, urgency="jog")

    # 3. 我方后场在被紧逼组织 → 回撤接应 (全能中锋的独特技能)
    if tm_ball is not None and tm_ball.position.x < -0.20:
        # 检查持球队友是否被贴身
        nearest_opp_to_carrier = None
        for o in obs.opponents():
            if nearest_opp_to_carrier is None or \
               (o.position.x - tm_ball.position.x) ** 2 + (o.position.y - tm_ball.position.y) ** 2 < \
               (nearest_opp_to_carrier.position.x - tm_ball.position.x) ** 2 + (nearest_opp_to_carrier.position.y - tm_ball.position.y) ** 2:
                nearest_opp_to_carrier = o
        if nearest_opp_to_carrier is not None:
            dist_sq = (nearest_opp_to_carrier.position.x - tm_ball.position.x) ** 2 + \
                      (nearest_opp_to_carrier.position.y - tm_ball.position.y) ** 2
            if dist_sq < 0.015:   # ~ 0.12 距离
                return MoveTo(target_x=_RETREAT_MIN_X, target_y=0.0, urgency="jog")

    # 4. 球到对方禁区附近 + 我还没在禁区 → 抢点 (嗅觉敏锐)
    if ball is not None and ball.position.x > 0.60 and self_pos.x < 0.70:
        target_y = 0.12 if ball.position.y > 0 else -0.12   # 跑远柱
        return MoveTo(target_x=0.85, target_y=target_y, urgency="sprint")

    # 5. 对方控球在我方半场 → 留在中圈附近等反击 (不回防)
    carrier = opponent_with_ball(obs)
    if carrier is not None and carrier.position.x < 0.0:
        anchor = clip(_ANCHOR_X - 0.20, _RETREAT_MIN_X, _ANCHOR_X)
        return MoveTo(target_x=anchor, target_y=0.0, urgency="walk")

    # 6. 默认 —— 回到进攻箭头位置
    target_x = clip(self_pos.x, _RETREAT_MIN_X, _ANCHOR_X)
    target_y = clip(self_pos.y, -_Y_LIMIT, _Y_LIMIT)
    if abs(target_x - _ANCHOR_X) < 0.03 and abs(target_y) < 0.04:
        return HoldPosition()
    return MoveTo(target_x=_ANCHOR_X, target_y=0.0, urgency="walk")
