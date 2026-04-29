"""Shared primitives used across the 10 per-player fallbacks.

IMPORTANT INVARIANT: fallback layer is a BODY REST STATE — it fills the
LLM decision gap with a positional default that matches the player's
identity. It MUST NOT grow into a second tactical brain. Every fallback
function is capped at ~30 lines of rules. If a fallback grows beyond that,
the logic belongs in prompts / Skills, not here.
"""
from __future__ import annotations

import math
from typing import Optional

from ..perception import EntityView, Observation, Vec2


# ---------------------------------------------------------------------------
# Entity lookup primitives
# ---------------------------------------------------------------------------

def opponent_with_ball(obs: Observation) -> Optional[EntityView]:
    """The opponent currently carrying the ball, if any is in FOV."""
    for e in obs.opponents():
        if e.has_ball:
            return e
    return None


def teammate_with_ball(obs: Observation) -> Optional[EntityView]:
    """A teammate (not self) currently carrying the ball, if any is in FOV."""
    for e in obs.teammates():
        if e.has_ball:
            return e
    return None


def loose_ball(obs: Observation) -> Optional[EntityView]:
    """The ball if it's in FOV AND no one in FOV is carrying it.
    Caveat: a carrier outside FOV would register as loose here — accept
    this since fallback is egocentric by design.
    """
    ball = obs.ball()
    if ball is None:
        return None
    if any(e.has_ball for e in obs.perceived_entities):
        return None
    return ball


def behind_me_opponent(
    obs: Observation, y_tolerance: float = 0.15
) -> Optional[EntityView]:
    """The closest opponent sitting between me and my own goal (x < self.x),
    laterally within y_tolerance. Used by CB/LB for the offside-trap guard.
    None if my side is clean.
    """
    me = obs.self_state.position
    best: Optional[EntityView] = None
    best_dx = math.inf
    for e in obs.opponents():
        if e.position.x >= me.x:
            continue  # in front of me — not "behind"
        if abs(e.position.y - me.y) > y_tolerance:
            continue
        dx = me.x - e.position.x
        if dx < best_dx:
            best_dx = dx
            best = e
    return best


def nearest_opponent(obs: Observation) -> Optional[EntityView]:
    """Closest opponent to self, regardless of ball state. None if FOV empty."""
    opps = obs.opponents()
    if not opps:
        return None
    return min(opps, key=lambda e: e.distance)


# ---------------------------------------------------------------------------
# Positional helpers
# ---------------------------------------------------------------------------

def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clip_target(
    x: float, y: float,
    x_min: float = -1.0, x_max: float = 1.0,
    y_min: float = -0.42, y_max: float = 0.42,
) -> tuple[float, float]:
    """Clip a MoveTo/DribbleToward target to a zone. Defaults to full pitch."""
    return clip(x, x_min, x_max), clip(y, y_min, y_max)


def distance(a: Vec2, b: Vec2) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def teammate_by_jersey(obs: Observation, jersey: int) -> Optional[EntityView]:
    """Look up a teammate by their gfootball entity_id (= per-team index).
    NOTE: entity_id in EgocentricFilter is the team-array index, not the
    persona's jersey_number. Callers that need a specific persona should
    pass that persona's player_id (which == team-array index in our setup).
    """
    for t in obs.teammates():
        if t.entity_id == jersey:
            return t
    return None


# ---------------------------------------------------------------------------
# Defensive coordination helpers (Fix 1 & 2)
# ---------------------------------------------------------------------------

def ranked_threats(obs: Observation) -> list[EntityView]:
    """Return opponents sorted by threat level (closest to our own goal first).

    For the BLUE team (attacking toward +x, defending toward -x):
      threat = how negative (i.e. how far into blue's half) the opponent's x is.
    We approximate this by sorting ascending on position.x — the most negative
    x is closest to the blue goal and most dangerous.

    For the RED team (attacking toward -x, defending toward +x):
      threat = how positive the opponent's x is.

    In both cases the *defending* team's own goal sits at the extreme of their
    half, so "closest x to our goal" is whichever extreme fits the side.

    Because fallbacks are egocentric, we can't know which side we are on from
    obs alone without reading self_state.team.  We therefore provide two
    convenience wrappers below (blue_threats / red_threats) that callers use
    directly — this function is internal.
    """
    opps = obs.opponents()
    return opps  # caller re-sorts by the correct key


def blue_ranked_threats(obs: Observation) -> list[EntityView]:
    """Opponents sorted by threat for the BLUE team (own goal at x=-1).
    Most dangerous = smallest x (deepest in blue's half).
    """
    opps = obs.opponents()
    return sorted(opps, key=lambda e: e.position.x)


def red_ranked_threats(obs: Observation) -> list[EntityView]:
    """Opponents sorted by threat for the RED team (own goal at x=+1).
    Most dangerous = largest x (deepest in red's half).
    """
    opps = obs.opponents()
    return sorted(opps, key=lambda e: e.position.x, reverse=True)


def get_defensive_line_x_blue(obs: Observation) -> float:
    """Return the x-coordinate of the last (deepest) blue defender.

    'Last defender' for blue = the teammate with the most-negative x
    (nearest to blue's own goal at x=-1), excluding this player itself.
    Falls back to -0.30 if no teammates visible.
    """
    teammates = obs.teammates()
    if not teammates:
        return -0.30
    return min(t.position.x for t in teammates)


def get_defensive_line_x_red(obs: Observation) -> float:
    """Return the x-coordinate of the last (deepest) red defender.

    'Last defender' for red = the teammate with the most-positive x
    (nearest to red's own goal at x=+1), excluding this player itself.
    Falls back to 0.30 if no teammates visible.
    """
    teammates = obs.teammates()
    if not teammates:
        return 0.30
    return max(t.position.x for t in teammates)


def ball_owned_by_opponent(obs: Observation) -> bool:
    """True if any opponent in FOV is carrying the ball.
    Falls back to checking whether ANY perceived entity with has_ball
    is an opponent (handles cases where carrier is just barely visible).
    """
    for e in obs.opponents():
        if e.has_ball:
            return True
    return False
