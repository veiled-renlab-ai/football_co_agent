"""LLM client — wraps the `openai` Python SDK to talk to OpenAI-compatible
endpoints (火山方舟 / MiniMax / DeepSeek / OpenAI).

Provider is selected by the LLM_PROVIDER env var. All providers expose the
same OpenAI-style chat-completions + tool-use API, so the rest of the codebase
stays provider-agnostic.

Failure mode: if the LLM call fails or returns no tool call, we surface the
exception to the caller — `LLMPlayer` decides how to fall back (typically
HoldPosition).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env once at module import.
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMDecision:
    """One LLM turn's output: the chosen tool + arguments + any free-text reasoning."""
    tool_name: str
    tool_args: dict[str, Any]
    reasoning: str = ""           # may be empty if the model only emitted a tool call
    raw_message: Optional[Any] = None


class LLMClient:
    """Thin OpenAI-compatible chat client with tool-use."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise RuntimeError("LLM api_key is empty — check .env")
        self.model = model
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    # ---- factory --------------------------------------------------------

    @classmethod
    def from_env(cls) -> "LLMClient":
        provider = os.getenv("LLM_PROVIDER", "volcengine").lower()
        if provider == "volcengine":
            return cls(
                api_key=os.getenv("VOLCENGINE_API_KEY", ""),
                base_url=os.getenv("VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                model=os.getenv("VOLCENGINE_MODEL", "doubao-seed-2-0-lite-260215"),
            )
        if provider == "minimax":
            return cls(
                api_key=os.getenv("MINIMAX_API_KEY", ""),
                base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
                model=os.getenv("MINIMAX_MODEL", "MiniMax-M1"),
            )
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}; expected volcengine|minimax")

    # ---- main call ------------------------------------------------------

    def choose_tool(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        max_retries: int = 1,
    ) -> LLMDecision:
        """Send one chat completion with `tools` and parse the model's tool call.

        On bad-JSON responses (GLM-4.7 occasionally returns truncated
        `arguments='{'`), we retry up to `max_retries` more times with the
        same prompt. Not a state machine — pure transport-layer resilience.

        Other failure modes (no tool call, network, auth) raise immediately.
        """
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return self._call_once(
                    system_prompt, user_message, tools,
                    temperature=temperature, max_tokens=max_tokens,
                )
            except RuntimeError as e:
                if "non-JSON tool arguments" not in str(e):
                    raise
                last_err = e
                if attempt < max_retries:
                    logger.warning(
                        "LLM returned bad JSON (attempt %d/%d), retrying...",
                        attempt + 1, max_retries + 1,
                    )
        # Exhausted retries
        assert last_err is not None
        raise last_err

    def _call_once(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> LLMDecision:
        """Stateless one-shot — system + single user msg, no history."""
        return self.chat_with_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_with_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> LLMDecision:
        """Lower-level — caller supplies the FULL message list (system +
        history + new user). Used by LLMPlayer to maintain conversation
        memory across stages and turns.

        Returns LLMDecision (parses first tool_call) and includes the raw
        assistant message via .raw_message so caller can append to history.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        msg = response.choices[0].message
        reasoning = (msg.content or "").strip()

        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            raise RuntimeError(
                f"LLM did not return a tool call. "
                f"Model={self.model}. Reasoning text was: {reasoning[:200]!r}"
            )

        tc = tcs[0]
        tool_name = tc.function.name
        try:
            tool_args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"LLM returned non-JSON tool arguments: {tc.function.arguments!r}"
            ) from e

        return LLMDecision(
            tool_name=tool_name,
            tool_args=tool_args,
            reasoning=reasoning,
            raw_message=msg,
        )
