"""Fallback policies — fast, deterministic skill choosers used between LLM
decisions in async play. Inspired by:

- **Utility AI** (David Mark, "Behavioral Mathematics for Game AI", 2009 —
  the standard pattern in The Sims, Civilization, FIFA NPCs): every candidate
  action gets scored 0..1 from current observation, pick max.
- **Subsumption architecture** (Rodney Brooks, "A Robust Layered Control
  System For A Mobile Robot", IEEE 1986): low-level reactive behaviors run
  always, higher layers (here: the LLM) override.
- **SayCan** (Ahn et al., Google 2022): high-level planner (LLM) is gated by
  a learned/heuristic affordance/value function. Our utility scores ARE the
  affordance function — what's possible AND useful right now.

This module exports a single function `utility_based_policy(obs) -> Skill`
that the AsyncRunner consumes as `fallback_policy=`.
"""
from __future__ import annotations

import math
from typing import Callable

from .perception import Observation
from .skills import (
    DribbleToward, HoldPosition, MoveTo, PassTo, Press, Shoot, Skill,
)

# Field constants (gfootball coords, left team attacking +x)
OPP_GOAL_X = 1.0
OPP_GOAL_Y = 0.0
GOAL_HALF_WIDTH = 0.044   # gfootball goal post is at y = ±0.044


# ---------------------------------------------------------------------------
# Utility scoring — each function returns ∈ [0, 1]. Higher = more desirable.
# ---------------------------------------------------------------------------

def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _shoot_utility(obs: Observation) -> float:
    """Score shooting: high if close to goal AND clear angle."""
    if not obs.self_state.has_ball:
        return 0.0
    sp = obs.self_state.position
    dist = _dist(sp.x, sp.y, OPP_GOAL_X, OPP_GOAL_Y)
    # Closer to goal = higher base utility (1.0 at goal, ~0 beyond 0.5)
    base = max(0.0, 1.0 - dist * 2.0)
    # Penalty if a defender is between us and the goal (within 0.05 of shot line)
    for o in obs.opponents():
        if o.role == "ball":  # safety
            continue
        if o.position.x > sp.x and abs(o.position.y) < 0.10:
            base *= 0.4
            break
    return base


def _dribble_utility(obs: Observation) -> tuple[float, Skill]:
    """Score dribbling forward. Returns (utility, parameterized skill)."""
    if not obs.self_state.has_ball:
        return 0.0, DribbleToward(target_x=0.0, target_y=0.0)
    sp = obs.self_state.position
    # Find nearest opponent ahead of us
    nearest_opp_ahead = 1.0
    for o in obs.opponents():
        if o.position.x > sp.x:
            d = _dist(o.position.x, o.position.y, sp.x, sp.y)
            nearest_opp_ahead = min(nearest_opp_ahead, d)
    # Clear lane → high utility. Crowded → low.
    lane_clearness = min(1.0, nearest_opp_ahead * 4.0)
    # Push toward goal but at most 0.2 per decision
    target_x = min(OPP_GOAL_X - 0.05, sp.x + 0.2)
    # Bias slightly toward center to set up the shot
    target_y = sp.y * 0.7
    util = lane_clearness * 0.7
    return util, DribbleToward(target_x=target_x, target_y=target_y)


def _pass_utilities(obs: Observation) -> list[tuple[float, Skill]]:
    """Score passing to each visible teammate ahead. Returns list of (util, skill)."""
    if not obs.self_state.has_ball:
        return []
    sp = obs.self_state.position
    out: list[tuple[float, Skill]] = []
    for t in obs.teammates():
        # Forward teammates only — back-passes rarely the right call
        if t.position.x <= sp.x + 0.05:
            continue
        # How marked? Distance to nearest opponent
        nearest_opp_to_t = min(
            (_dist(o.position.x, o.position.y, t.position.x, t.position.y)
             for o in obs.opponents()),
            default=1.0,
        )
        openness = min(1.0, nearest_opp_to_t * 6.0)
        # Position weight: closer to goal = better receiver
        forwardness = (t.position.x + 1.0) / 2.0
        util = openness * forwardness * 0.6
        # Pass type by distance
        d = _dist(sp.x, sp.y, t.position.x, t.position.y)
        ptype = "short" if d < 0.25 else "long"
        out.append((util, PassTo(target_player_id=t.entity_id, pass_type=ptype)))  # type: ignore[arg-type]
    return out


def _chase_ball_utility(obs: Observation) -> tuple[float, Skill]:
    """Score sprinting to the ball when we don't have it."""
    if obs.self_state.has_ball:
        return 0.0, MoveTo(target_x=0.0, target_y=0.0)
    ball = obs.ball()
    if ball is None:
        return 0.0, MoveTo(target_x=0.0, target_y=0.0)
    sp = obs.self_state.position
    # Are we closest of all visible players (teammates + opponents)?
    closer_others = 0
    for e in obs.perceived_entities:
        if e.role == "ball":
            continue
        d_e_ball = _dist(e.position.x, e.position.y, ball.position.x, ball.position.y)
        if d_e_ball < ball.distance - 0.02:  # 0.02 margin for noise
            closer_others += 1
    # Higher utility if we're closest, lower if others are closer
    if closer_others == 0:
        util = 0.85
    else:
        util = max(0.2, 0.85 - 0.15 * closer_others)
    return util, MoveTo(
        target_x=float(ball.position.x),
        target_y=float(ball.position.y),
        urgency="sprint",
    )


def _press_utilities(obs: Observation) -> list[tuple[float, Skill]]:
    """Score pressing each opponent who has the ball."""
    if obs.self_state.has_ball:
        return []
    out: list[tuple[float, Skill]] = []
    for o in obs.opponents():
        if not o.has_ball:
            continue
        # Closer = more urgent to press
        util = max(0.0, 1.0 - o.distance * 1.8)
        out.append((util, Press(opponent_id=o.entity_id)))
    return out


def _support_utility(obs: Observation) -> tuple[float, Skill]:
    """Default: drift toward midfield/forward to stay involved."""
    sp = obs.self_state.position
    # Low constant utility — only fires if nothing better
    target_x = max(sp.x, 0.3)  # at least midfield
    target_y = 0.0              # central by default
    return 0.25, MoveTo(target_x=target_x, target_y=target_y, urgency="jog")


# ---------------------------------------------------------------------------
# Top-level policy
# ---------------------------------------------------------------------------

def utility_based_policy(obs: Observation) -> Skill:
    """Score every candidate skill from the observation, return highest-utility one.

    This is the 'cerebellum' — fast, deterministic, always-running. The LLM
    'cortex' provides tactical overrides on top of this.
    """
    candidates: list[tuple[float, Skill, str]] = [
        (_shoot_utility(obs),         Shoot(target_zone="top_center"),     "shoot"),
        (*_dribble_utility(obs),      "dribble"),  # type: ignore[misc]
        (*_chase_ball_utility(obs),   "chase ball"),  # type: ignore[misc]
        (*_support_utility(obs),      "support drift"),  # type: ignore[misc]
    ]
    # Variable-count categories — append individually
    for util, skill in _pass_utilities(obs):
        candidates.append((util, skill, "pass"))
    for util, skill in _press_utilities(obs):
        candidates.append((util, skill, "press"))

    if not candidates:
        return HoldPosition()

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# Re-export type alias for convenience
PolicyFn = Callable[[Observation], Skill]
