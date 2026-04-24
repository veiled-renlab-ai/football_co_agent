"""Shared guards — fallback-layer hard rules that apply to ALL players.

Kept intentionally small. Only 3 guards here; anything else lives in
the specific player's fallback function as a per-persona clip or
short-circuit.

Call order in every fallback entry point:
    1. apply_shared_short_circuits(ctx) — may return a Skill to use verbatim
    2. personal rules              — per-player logic chooses candidate skill
    3. downshift_urgency(skill, ctx) — stamina-aware post-filter on the result
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from ..skills import (
    DribbleToward, HoldPosition, MoveTo, Skill,
)
from .context import FallbackContext

# gfootball game_mode values: 0=Normal, 1=KickOff, 2=GoalKick, 3=FreeKick,
# 4=Corner, 5=ThrowIn, 6=Penalty.
_NORMAL_PLAY = 0

# Below this stamina, sprint silently downshifts to jog so a tired player
# doesn't burn the last of their legs on a fallback action. 0.3 ≈ "气喘吁吁"
# bucket in prompts.py (_describe_stamina).
_STAMINA_DOWNSHIFT_THRESHOLD = 0.30


def apply_shared_short_circuits(ctx: FallbackContext) -> Optional[Skill]:
    """Return a Skill that overrides everything, or None to pass through.

    Three universal overrides (in evaluation order):
      1. game_mode != Normal — set piece in progress, just stand still so
         we don't wander during kickoff/free-kick/corner etc.
      2. Ball completely out of FOV AND no recent LLM intent — HoldPosition
         at the player's anchor feels more "lost but alert" than sprinting
         toward a phantom ball the runner might not be able to find.
      3. LLM chose something <100 ticks ago — defer to that intent rather
         than the runner yanking the body in a different direction.
    """
    obs = ctx.obs

    # 1. Set piece → freeze
    if obs.game_mode != _NORMAL_PLAY:
        return HoldPosition()

    # 2. Ball fully invisible AND no fresh LLM intent → hold
    if obs.ball() is None and ctx.recent_llm_intent is None:
        return HoldPosition()

    # 3. Respect a fresh LLM decision over any fallback logic
    if ctx.recent_llm_intent is not None:
        skill, _tick = ctx.recent_llm_intent
        return skill

    return None


def downshift_urgency(skill: Skill, ctx: FallbackContext) -> Skill:
    """If the player is gassed, rewrite urgency='sprint' to 'jog' on
    MoveTo / DribbleToward. Return the skill unchanged for other types.
    """
    if ctx.obs.self_state.stamina >= _STAMINA_DOWNSHIFT_THRESHOLD:
        return skill
    if isinstance(skill, (MoveTo, DribbleToward)) and skill.urgency == "sprint":
        return replace(skill, urgency="jog")
    return skill
