"""LLMPlayer — a player whose brain is an LLM choosing intent-level Skills.

Wraps the LLM client + prompt rendering + tool-call → Skill instantiation
into a single object that the runner can poll for decisions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm_client import LLMClient, LLMDecision
from .perception import Observation
from .prompts import DEFAULT_PERSONA, PlayerPersona, build_system_prompt, render_observation
from .skills import (
    ALL_SKILLS, SKILLS_BY_NAME, Call, DribbleToward, HoldPosition, Mark, MoveTo,
    PassTo, Press, ReceiveBall, ScanBehind, Shoot, Skill, Tackle, Track,
    skill_to_tool_schema,
)


# Action space gating — match what the player can PHYSICALLY do right now.
# This is not behavioral steering; it's the action space of a real player
# (you can't 'kick the ball' if you don't have it).
_REQUIRES_BALL: set[type] = {DribbleToward, PassTo, Shoot}
_REQUIRES_TEAMMATE: set[type] = {PassTo}
_REQUIRES_OPPONENT: set[type] = {Mark, Press, Tackle}

logger = logging.getLogger(__name__)


@dataclass
class TurnLog:
    """One LLM decision turn's audit record — useful for debugging and replay."""
    tick: int
    observation_text: str
    reasoning: str
    tool_name: str
    tool_args: dict[str, Any]
    skill: Optional[Skill]
    error: Optional[str] = None


class LLMPlayer:
    """A football player driven by an LLM.

    Lifecycle:
        player = LLMPlayer(player_id=1, role="CB", llm_client=client)
        skill  = player.choose_skill(observation)   # one LLM call
        # ... runner dispatches the skill, then loops ...
    """

    def __init__(
        self,
        *,
        player_id: int,
        role: str,
        llm_client: LLMClient,
        persona: PlayerPersona = DEFAULT_PERSONA,
    ) -> None:
        self.player_id = player_id
        self.role = role
        self.persona = persona
        self.llm_client = llm_client
        self._system_prompt = build_system_prompt(persona)
        self.history: list[TurnLog] = []

    @staticmethod
    def _valid_skills_for(obs: Observation) -> list[type]:
        """Filter skills to those mechanically possible right now."""
        has_ball = obs.self_state.has_ball
        has_teammate = bool(obs.teammates())
        has_opponent = bool(obs.opponents())
        valid: list[type] = []
        for s in ALL_SKILLS:
            if s in _REQUIRES_BALL and not has_ball:
                continue
            if s in _REQUIRES_TEAMMATE and not has_teammate:
                continue
            if s in _REQUIRES_OPPONENT and not has_opponent:
                continue
            valid.append(s)
        return valid

    # ------------------------------------------------------------------

    def update_role(self, new_role: str) -> None:
        """gfootball role label changed (auto-switch). Persona stays the same —
        the player is still 李大军, just now playing a different on-pitch role."""
        self.role = new_role

    def choose_skill(self, observation: Observation) -> Skill:
        """Run one LLM call, parse, and return a Skill instance.

        On any failure (LLM error, unknown tool, invalid args), logs the error
        and falls back to HoldPosition — the agent never crashes the match.
        """
        obs_text = render_observation(observation, self.persona)
        log_entry = TurnLog(
            tick=observation.tick,
            observation_text=obs_text,
            reasoning="",
            tool_name="",
            tool_args={},
            skill=None,
        )

        # Per-turn dynamic tool filtering — only physically possible actions.
        valid_skill_classes = self._valid_skills_for(observation)
        valid_tools = [skill_to_tool_schema(c) for c in valid_skill_classes]

        try:
            decision = self.llm_client.choose_tool(
                system_prompt=self._system_prompt,
                user_message=obs_text,
                tools=valid_tools,
            )
        except Exception as e:
            log_entry.error = f"llm_call_failed: {e!r}"
            log_entry.skill = HoldPosition()
            self.history.append(log_entry)
            logger.warning("LLM call failed for player %s: %s", self.player_id, e)
            return log_entry.skill

        log_entry.reasoning = decision.reasoning
        log_entry.tool_name = decision.tool_name
        log_entry.tool_args = decision.tool_args

        skill_cls = SKILLS_BY_NAME.get(decision.tool_name)
        if skill_cls is None:
            log_entry.error = f"unknown_skill: {decision.tool_name!r}"
            log_entry.skill = HoldPosition()
            self.history.append(log_entry)
            logger.warning("Unknown skill %r from LLM; fallback to HoldPosition", decision.tool_name)
            return log_entry.skill

        try:
            skill = skill_cls(**decision.tool_args)
        except TypeError as e:
            log_entry.error = f"bad_args: {e!r}"
            log_entry.skill = HoldPosition()
            self.history.append(log_entry)
            logger.warning("Bad args for %s: %s", decision.tool_name, e)
            return log_entry.skill

        log_entry.skill = skill
        self.history.append(log_entry)
        return skill
