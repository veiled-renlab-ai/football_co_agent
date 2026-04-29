"""FallbackContext — the data packet passed to every fallback function.

Built by MultiAgentRunner once per tick per agent before calling that
agent's fallback. Bundles everything a fallback rule might read:
  - persona:          WHO this player is (identity / style / jersey / team)
  - obs:              egocentric Observation (self + FOV entities + heard_calls + game_mode)
  - recent_llm_intent: (skill, tick) if LLM picked something <2s ago, else None
                       Fallback should defer to this rather than overriding
  - motor_status:     Current motor controller status ("in_progress", "completed",
                      "failed", or None if no controller installed yet).
                      In stop-world mode, fallback is only armed when a skill
                      completes, so this will be "completed" or "failed".
                      Guards use this to avoid interrupting a skill mid-execution.

Kept as a plain frozen dataclass — no methods, no behavior. Rules poke
at the fields directly so they stay easy to read and cheap to construct.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ..perception import Observation
from ..prompts import PlayerPersona
from ..skills import Skill

MotorStatus = Optional[Literal["in_progress", "completed", "failed"]]


@dataclass(frozen=True)
class FallbackContext:
    persona: PlayerPersona
    obs: Observation
    recent_llm_intent: Optional[tuple[Skill, int]] = None
    motor_status: MotorStatus = None
