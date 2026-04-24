"""FallbackContext — the data packet passed to every fallback function.

Built by MultiAgentRunner once per tick per agent before calling that
agent's fallback. Bundles everything a fallback rule might read:
  - persona:          WHO this player is (identity / style / jersey / team)
  - obs:              egocentric Observation (self + FOV entities + heard_calls + game_mode)
  - recent_llm_intent: (skill, tick) if LLM picked something <2s ago, else None
                       Fallback should defer to this rather than overriding

Kept as a plain frozen dataclass — no methods, no behavior. Rules poke
at the fields directly so they stay easy to read and cheap to construct.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..perception import Observation
from ..prompts import PlayerPersona
from ..skills import Skill


@dataclass(frozen=True)
class FallbackContext:
    persona: PlayerPersona
    obs: Observation
    recent_llm_intent: Optional[tuple[Skill, int]] = None
