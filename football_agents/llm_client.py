"""LLM client — wraps the `openai` Python SDK to talk to OpenAI-compatible
endpoints (阿里云百炼 Token Plan / 火山方舟 / MiniMax / DeepSeek / OpenAI).

Provider is selected by the LLM_PROVIDER env var. All providers expose the
same OpenAI-style chat-completions + tool-use API, so the rest of the codebase
stays provider-agnostic.

Key rotation: a provider may configure N API keys (comma-separated). The
client holds one OpenAI instance per key and rotates round-robin per call
to raise effective RPM past a single key's rate limit. Rotation is
thread-safe so multiple PlayerAgent worker threads sharing one LLMClient
still get balanced key use.

Failure mode: if the LLM call fails or returns no tool call, we surface the
exception to the caller — `LLMPlayer` decides how to fall back (typically
HoldPosition).
"""
from __future__ import annotations

import json
import logging
import os
import threading
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


def _parse_key_list(raw: str) -> list[str]:
    """Split a comma-separated env var into a list of non-empty API keys."""
    return [k.strip() for k in raw.split(",") if k.strip()]


class LLMClient:
    """Thin OpenAI-compatible chat client with tool-use + multi-key rotation."""

    def __init__(
        self,
        *,
        api_keys: list[str],
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> None:
        if not api_keys:
            raise RuntimeError("LLM api_keys list is empty — check .env")
        self.model = model
        self.base_url = base_url
        # One SDK client per key. Rotation happens in _pick_client().
        self._clients = [
            OpenAI(api_key=k, base_url=base_url, timeout=timeout_s)
            for k in api_keys
        ]
        self._next_idx = 0
        self._lock = threading.Lock()
        # Provider-specific body fragments merged into every request. For
        # 火山方舟 this carries {"thinking": {"type": "disabled"}} to turn
        # off doubao's extended-thinking (≈3x faster); aliyun Token Plan
        # defaults to empty (qwen3 non-thinking by default).
        self._extra_body = extra_body or {}

    @property
    def n_keys(self) -> int:
        return len(self._clients)

    def _pick_client(self) -> OpenAI:
        """Thread-safe round-robin pick across the configured API keys."""
        with self._lock:
            client = self._clients[self._next_idx]
            self._next_idx = (self._next_idx + 1) % len(self._clients)
        return client

    # ---- factory --------------------------------------------------------

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Build an LLMClient from env vars.

        LLM_PROVIDER selects the provider; each provider pulls its own
        KEY / BASE_URL / MODEL env vars. Default: aliyun_token_plan.

        Multi-key (aliyun_token_plan): ALIYUN_TOKEN_PLAN_API_KEYS is a
        comma-separated list of `pt-...` keys. The client rotates through
        them per request to multiply the effective rate limit.
        """
        provider = os.getenv("LLM_PROVIDER", "aliyun_token_plan").lower()
        if provider == "aliyun_token_plan":
            # Aliyun 百炼 Token Plan 团队版 — OpenAI-compatible endpoint.
            # Supported models: qwen3.6-plus, glm-5, MiniMax-M2.5, deepseek-v3.2
            keys = _parse_key_list(os.getenv("ALIYUN_TOKEN_PLAN_API_KEYS", ""))
            if not keys:
                # Fall back to single-key env var for backward compat
                single = os.getenv("ALIYUN_TOKEN_PLAN_API_KEY", "")
                if single:
                    keys = [single]
            return cls(
                api_keys=keys,
                base_url=os.getenv(
                    "ALIYUN_TOKEN_PLAN_BASE_URL",
                    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                ),
                model=os.getenv("ALIYUN_TOKEN_PLAN_MODEL", "qwen3.6-plus"),
                # qwen3 is non-thinking by default, no extra toggle needed
                extra_body=None,
            )
        if provider == "volcengine":
            key = os.getenv("VOLCENGINE_API_KEY", "")
            return cls(
                api_keys=[key] if key else [],
                base_url=os.getenv(
                    "VOLCENGINE_BASE_URL",
                    "https://ark.cn-beijing.volces.com/api/coding/v3",
                ),
                model=os.getenv("VOLCENGINE_MODEL", "doubao-seed-2-0-lite-260215"),
                # doubao-seed defaults to extended thinking; disable for speed
                extra_body={"thinking": {"type": "disabled"}},
            )
        if provider == "minimax":
            key = os.getenv("MINIMAX_API_KEY", "")
            return cls(
                api_keys=[key] if key else [],
                base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
                model=os.getenv("MINIMAX_MODEL", "MiniMax-M1"),
            )
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}; "
            f"expected aliyun_token_plan|volcengine|minimax"
        )

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
        client = self._pick_client()
        create_kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self._extra_body:
            create_kwargs["extra_body"] = self._extra_body
        response = client.chat.completions.create(**create_kwargs)
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
