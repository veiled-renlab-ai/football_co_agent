"""Per-player fallback registry.

Usage from MultiAgentRunner:
    from football_agents.fallbacks import get_fallback
    fn = get_fallback(agent.persona)          # Callable[[FallbackContext], Skill]
    skill = fn(FallbackContext(persona, obs, recent_llm_intent))

Registered by (team, jersey_number) — unique per pitch. Personas without
a registered fallback fall through to `_legacy_fallback` (the previous
global body_rest_state_fallback, preserved for backward compat).
"""
from __future__ import annotations

from typing import Callable

from ..perception import Observation
from ..prompts import PlayerPersona
from ..skills import HoldPosition, Skill
from ._guards import apply_shared_short_circuits, downshift_urgency
from .blue_cb import fallback_gao_lei
from .blue_cf import fallback_chen_yu
from .blue_gk import fallback_lin_tao
from .blue_lb import fallback_zhou_jun
from .blue_rm import fallback_wang_hao
from .context import FallbackContext
from .red_cb import fallback_ma_liang
from .red_cf import fallback_li_qiang
from .red_gk import fallback_zhao_qiang
from .red_lb import fallback_sun_bin
from .red_rm import fallback_liu_feng

PersonalFallback = Callable[[FallbackContext], Skill]
FallbackFn = Callable[[FallbackContext], Skill]

# (team, jersey_number) → pure per-persona fallback.
# Keys use team names exactly as PlayerPersona.team (Chinese labels).
_REGISTRY: dict[tuple[str, int], PersonalFallback] = {
    # Blue
    ("蓝队", 1):  fallback_lin_tao,
    ("蓝队", 11): fallback_wang_hao,
    ("蓝队", 9):  fallback_chen_yu,
    ("蓝队", 3):  fallback_zhou_jun,
    ("蓝队", 4):  fallback_gao_lei,
    # Red
    ("红队", 1):  fallback_zhao_qiang,
    ("红队", 7):  fallback_liu_feng,
    ("红队", 10): fallback_li_qiang,
    ("红队", 2):  fallback_sun_bin,
    ("红队", 5):  fallback_ma_liang,
}


def _legacy_fallback(ctx: FallbackContext) -> Skill:
    """Safety net for personas not in the registry. Was body_rest_state_fallback."""
    return HoldPosition()


def _wrap_with_guards(personal: PersonalFallback) -> FallbackFn:
    """Combine the 3 shared-guard pipeline with a per-person rule set.

    Order:
      1. shared short-circuits (game_mode / ball invisible / recent LLM intent)
      2. personal rules
      3. stamina downshift (sprint → jog when gassed)
    """

    def _entry(ctx: FallbackContext) -> Skill:
        override = apply_shared_short_circuits(ctx)
        if override is not None:
            return override
        chosen = personal(ctx)
        return downshift_urgency(chosen, ctx)

    _entry.__name__ = f"wrapped_{personal.__name__}"
    return _entry


def get_fallback(persona: PlayerPersona) -> FallbackFn:
    """Resolve a persona to its fallback function (already wrapped with
    shared guards). Missing personas get the legacy HoldPosition stub.
    """
    key = (persona.team, persona.jersey_number)
    personal = _REGISTRY.get(key)
    if personal is None:
        import logging
        logging.getLogger(__name__).warning(
            "no fallback registered for persona (team=%r, jersey=%d, name=%r); "
            "using HoldPosition legacy stub",
            persona.team, persona.jersey_number, persona.name,
        )
        return _wrap_with_guards(_legacy_fallback)
    return _wrap_with_guards(personal)


__all__ = ["FallbackContext", "get_fallback"]
