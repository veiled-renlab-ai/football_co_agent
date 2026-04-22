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
                model=os.getenv("VOLCENGINE_MODEL", "doubao-seed-1.6-250615"),
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
        max_tokens: int = 2048,
    ) -> LLMDecision:
        """Send one chat completion with `tools` and parse the model's tool call.

        Raises RuntimeError if no tool call comes back (some providers don't
        respect tool_choice='required'; the caller should fall back).

        Notes:
          - max_tokens=2048 because some thinking-capable models (e.g. GLM-4.7
            on 火山方舟 Coding Plan) consume hidden reasoning_tokens that share
            the output budget; with long prompts + many tools the response gets
            truncated to '{' if the cap is too low.
          - extra_body.thinking.type='disabled' attempts to turn off implicit
            thinking on GLM-style providers. Harmless if the provider ignores
            unknown extras; OpenAI / Doubao etc. just pass it through.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
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
        # arguments is a JSON string per OpenAI spec
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
