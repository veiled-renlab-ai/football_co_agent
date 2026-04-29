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

    Four universal overrides (in evaluation order):
      1. game_mode != Normal — set piece in progress, just stand still so
         we don't wander during kickoff/free-kick/corner etc.
      2. Motor still in_progress — the current skill hasn't finished yet;
         let it run to completion rather than interrupting it. Return None
         to signal "no override, keep the existing controller". This is the
         correct behaviour in stop-world mode where fallback is re-armed on
         every env tick during the K-tick execution window.
      3. Ball completely out of FOV AND no recent LLM intent — HoldPosition
         at the player's anchor feels more "lost but alert" than sprinting
         toward a phantom ball the runner might not be able to find.
      4. LLM chose something recently — only defer to that intent if the
         motor has already finished (motor_status != "in_progress") AND the
         intent was issued very recently (within 5 ticks). In stop-world
         mode recent_llm_intent is always None (passed explicitly by
         _arm_fallback_for_stop_world), so this guard is bypassed entirely
         in that path.
    """
    obs = ctx.obs

    # 1. Set piece → freeze
    if obs.game_mode != _NORMAL_PLAY:
        return HoldPosition()

    # 2. Motor still executing → don't interrupt it; let it finish.
    #    Return None so the caller's personal fallback logic is skipped too
    #    (the whole fallback call is a no-op when the motor is mid-skill).
    if ctx.motor_status == "in_progress":
        return None

    # 3. Ball fully invisible AND no fresh LLM intent → hold
    if obs.ball() is None and ctx.recent_llm_intent is None:
        return HoldPosition()

    # 4. Respect a fresh LLM decision — but only for a very short window
    #    (5 ticks ≈ 0.1s at 50 fps) to avoid the 2-second "intent lockout"
    #    that plagued the async runner.  In stop-world mode recent_llm_intent
    #    is always None, so this branch is never reached from that path.
    if ctx.recent_llm_intent is not None:
        skill, issued_tick = ctx.recent_llm_intent
        if (obs.tick - issued_tick) < 5:
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
