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
    ALL_SKILLS, CATEGORY_TOOL_NAMES, SKILLS_BY_NAME, Call, DribbleToward,
    HoldPosition, Mark, MoveTo, PassTo, Press, ReceiveBall, ScanBehind,
    Shoot, Skill, Tackle, Track, layer_1_category_tools, skill_to_tool_schema,
    skills_in_category,
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

    # 3-tier memory:
    #   - last X turns: kept VERBATIM (full message exchange)
    #   - turns X+1..Y: compressed to 1-line summaries ("choose_attack → shoot(top_center)")
    #   - older than Y: dropped
    MAX_RECENT_TURNS: int = 3
    MAX_TOTAL_TURNS: int = 8   # so compressed-zone holds (8-3)=5 summaries

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
        # Recent turns: list of [user_obs, asst_s1, tool_ack_1, asst_s2, tool_ack_2]
        self._recent_turns: list[list[dict[str, Any]]] = []
        # Compressed summaries: one short string per older turn
        self._compressed_summaries: list[str] = []

    # ----------------------------------------------------------------
    # Memory helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _compress_turn(turn_msgs: list[dict[str, Any]]) -> str:
        """Squash one decision turn into a single-line summary string."""
        parts: list[str] = []
        for m in turn_msgs:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                tc = m["tool_calls"][0]
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                if args and args != "{}":
                    parts.append(f"{name}({args})")
                else:
                    parts.append(f"{name}()")
        return " → ".join(parts) if parts else "(no tool call)"

    def _build_messages(
        self, new_user_msg: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Assemble the full message list to send to the LLM:
        system + (compressed-recap if any) + recent verbatim turns + new user.
        """
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]
        if self._compressed_summaries:
            recap = "（你最近几轮的动作回顾，按时间从早到晚）\n" + "\n".join(
                f"  · 第 {-len(self._compressed_summaries) + i + 1} 轮前: {s}"
                for i, s in enumerate(self._compressed_summaries)
            )
            msgs.append({"role": "user", "content": recap})
            msgs.append({"role": "assistant", "content": "好的，我记得这些。"})
        for turn in self._recent_turns:
            msgs.extend(turn)
        msgs.append(new_user_msg)
        return msgs

    def _commit_turn(self, turn_msgs: list[dict[str, Any]]) -> None:
        """After a successful decision turn, push messages into memory and
        compress / drop as needed per 3-tier policy."""
        self._recent_turns.append(turn_msgs)
        # Tier 1 → Tier 2: oldest recent gets compressed
        while len(self._recent_turns) > self.MAX_RECENT_TURNS:
            oldest = self._recent_turns.pop(0)
            self._compressed_summaries.append(self._compress_turn(oldest))
        # Tier 2 → Tier 3 (delete): drop oldest compressed beyond budget
        max_compressed = self.MAX_TOTAL_TURNS - self.MAX_RECENT_TURNS
        while len(self._compressed_summaries) > max_compressed:
            self._compressed_summaries.pop(0)

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
        valid_set = set(valid_skill_classes)

        # Stage 1 categories: only those that contain at least one valid skill
        from .skills import SKILL_CATEGORY  # local import (avoids circular)
        valid_categories = sorted({
            c for s, c in SKILL_CATEGORY.items() if s in valid_set
        })
        layer_1_tools = [
            t for t in layer_1_category_tools()
            if CATEGORY_TOOL_NAMES[t["function"]["name"]] in valid_categories
        ]

        # ---- Build messages with 3-tier memory ----
        new_user_msg: dict[str, Any] = {"role": "user", "content": obs_text}
        msgs_for_stage1 = self._build_messages(new_user_msg)

        try:
            stage1 = self.llm_client.chat_with_messages(
                messages=msgs_for_stage1,
                tools=layer_1_tools,
            )
        except Exception as e:
            log_entry.error = f"llm_stage1_failed: {e!r}"
            log_entry.skill = HoldPosition()
            self.history.append(log_entry)
            logger.warning("LLM stage-1 failed for player %s: %s", self.player_id, e)
            return log_entry.skill

        category = CATEGORY_TOOL_NAMES.get(stage1.tool_name)
        if category is None:
            log_entry.error = f"unknown_category: {stage1.tool_name!r}"
            log_entry.skill = HoldPosition()
            self.history.append(log_entry)
            logger.warning("Unknown category %r from stage-1", stage1.tool_name)
            return log_entry.skill

        stage1_assist_msg = stage1.raw_message.model_dump(exclude_none=True)
        stage1_tool_call_id = stage1.raw_message.tool_calls[0].id
        stage1_tool_ack = {
            "role": "tool",
            "tool_call_id": stage1_tool_call_id,
            "content": f"OK，类目已选定: {category}。请从该类下选具体动作。",
        }

        # ---- Stage 2: pick specific skill within the chosen category ----
        cat_skills = [s for s in skills_in_category(category) if s in valid_set]
        if not cat_skills:
            log_entry.error = f"no_valid_skills_in_category: {category}"
            log_entry.skill = HoldPosition()
            self.history.append(log_entry)
            return log_entry.skill

        stage2_tools = [skill_to_tool_schema(c) for c in cat_skills]
        # Stage 2 sees: system + memory + new_user + stage1_asst + stage1_ack
        msgs_for_stage2 = msgs_for_stage1 + [stage1_assist_msg, stage1_tool_ack]

        try:
            stage2 = self.llm_client.chat_with_messages(
                messages=msgs_for_stage2,
                tools=stage2_tools,
            )
        except Exception as e:
            log_entry.error = f"llm_stage2_failed: {e!r}"
            log_entry.skill = HoldPosition()
            self.history.append(log_entry)
            logger.warning("LLM stage-2 failed for player %s: %s", self.player_id, e)
            return log_entry.skill

        stage2_assist_msg = stage2.raw_message.model_dump(exclude_none=True)
        stage2_tool_call_id = stage2.raw_message.tool_calls[0].id
        stage2_tool_ack = {
            "role": "tool",
            "tool_call_id": stage2_tool_call_id,
            "content": f"动作 {stage2.tool_name} 已经派给 motor 执行。",
        }

        # Commit this turn into 3-tier memory (recent → compress → drop)
        this_turn_msgs = [
            new_user_msg,
            stage1_assist_msg,
            stage1_tool_ack,
            stage2_assist_msg,
            stage2_tool_ack,
        ]
        self._commit_turn(this_turn_msgs)

        # Combine reasoning from both stages for the visible log
        combined = []
        if stage1.reasoning:
            combined.append(f"[{category}] {stage1.reasoning}")
        if stage2.reasoning:
            combined.append(stage2.reasoning)
        log_entry.reasoning = " // ".join(combined)
        log_entry.tool_name = stage2.tool_name
        log_entry.tool_args = stage2.tool_args

        skill_cls = SKILLS_BY_NAME.get(stage2.tool_name)
        if skill_cls is None:
            log_entry.error = f"unknown_skill: {stage2.tool_name!r}"
            log_entry.skill = HoldPosition()
            self.history.append(log_entry)
            logger.warning("Unknown skill %r from stage-2; fallback HoldPosition", stage2.tool_name)
            return log_entry.skill

        try:
            skill = skill_cls(**stage2.tool_args)
        except TypeError as e:
            log_entry.error = f"bad_args: {e!r}"
            log_entry.skill = HoldPosition()
            self.history.append(log_entry)
            logger.warning("Bad args for %s: %s", stage2.tool_name, e)
            return log_entry.skill

        log_entry.skill = skill
        self.history.append(log_entry)
        return skill
