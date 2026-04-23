"""Test which `extra_body` spelling actually disables GLM-4.7 thinking.

Baseline call has reasoning_tokens≈125 (thinking ON). We try several
common disable spellings and report reasoning_tokens for each.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SIMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "shoot",
        "description": "Shoot at the goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_zone": {"type": "string", "enum": ["top_left", "top_center", "top_right"]},
            },
            "required": ["target_zone"],
            "additionalProperties": False,
        },
    },
}

MSG = [
    {"role": "system", "content": "You are a footballer. Choose an action by calling a tool."},
    {"role": "user", "content": "You have the ball 5m from goal, no defender. Shoot."},
]


def try_mode(client: OpenAI, model: str, label: str, **kwargs) -> None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=MSG,
            tools=[SIMPLE_TOOL],
            tool_choice="auto",
            temperature=0.4,
            max_tokens=1024,
            **kwargs,
        )
    except Exception as e:
        print(f"  {label:50s}  FAILED: {type(e).__name__}: {str(e)[:80]}")
        return
    usage = resp.usage
    ct = usage.completion_tokens
    rt_obj = getattr(usage, "completion_tokens_details", None)
    rt = getattr(rt_obj, "reasoning_tokens", None) if rt_obj else None
    reasoning = resp.choices[0].message.model_dump().get("reasoning_content")
    has_reasoning = "yes" if reasoning else "no"
    has_tool = "yes" if resp.choices[0].message.tool_calls else "no"
    print(f"  {label:50s}  completion={ct:>4}  reasoning_tokens={rt}  "
          f"reasoning_content={has_reasoning}  tool_call={has_tool}")


def main() -> None:
    client = OpenAI(
        api_key=os.getenv("VOLCENGINE_API_KEY"),
        base_url=os.getenv("VOLCENGINE_BASE_URL"),
        timeout=30.0,
    )
    model = os.getenv("VOLCENGINE_MODEL")
    print(f"Model: {model}")
    print(f"Base URL: {client.base_url}")
    print()
    print("Testing multiple ways to disable GLM thinking:")
    print()

    # Baseline — no extra_body
    try_mode(client, model, "baseline (no extra_body)")

    # 1. Doubao-style
    try_mode(client, model, "extra_body={'thinking': {'type': 'disabled'}}",
             extra_body={"thinking": {"type": "disabled"}})

    # 2. Qwen-style
    try_mode(client, model, "extra_body={'enable_thinking': False}",
             extra_body={"enable_thinking": False})

    # 3. Bool on 'thinking'
    try_mode(client, model, "extra_body={'thinking': False}",
             extra_body={"thinking": False})

    # 4. thinking_config
    try_mode(client, model, "extra_body={'thinking_config': {'enabled': False}}",
             extra_body={"thinking_config": {"enabled": False}})

    # 5. Nested under 'extra_params'
    try_mode(client, model, "extra_body={'chat_template_kwargs': {'enable_thinking': False}}",
             extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    # 6. Some vLLM-style deployments use this
    try_mode(client, model, "extra_body={'reasoning': {'enabled': False}}",
             extra_body={"reasoning": {"enabled": False}})

    # 7. GLM-specific /api/coding docs pattern (speculative)
    try_mode(client, model, "extra_body={'do_reasoning': False}",
             extra_body={"do_reasoning": False})


if __name__ == "__main__":
    main()
