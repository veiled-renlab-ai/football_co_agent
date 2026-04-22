"""Minimal diagnostic — call GLM-4.7 with ONE tool and dump the raw response.
Goal: figure out why agent sees `tool_calls[0].function.arguments == '{'`.
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

    print("Calling with 1 tool, tool_choice='auto'...")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a footballer. Choose an action by calling a tool."},
            {"role": "user", "content": "You have the ball 5m from goal, no defender. Shoot."},
        ],
        tools=[SIMPLE_TOOL],
        tool_choice="auto",
        temperature=0.4,
    )

    print(f"finish_reason: {resp.choices[0].finish_reason}")
    msg = resp.choices[0].message
    print(f"message.role: {msg.role}")
    print(f"message.content: {msg.content!r}")
    print(f"message.tool_calls: {msg.tool_calls!r}")
    if msg.tool_calls:
        for i, tc in enumerate(msg.tool_calls):
            print(f"  tool_call[{i}].id: {tc.id}")
            print(f"  tool_call[{i}].type: {tc.type}")
            print(f"  tool_call[{i}].function.name: {tc.function.name}")
            print(f"  tool_call[{i}].function.arguments: {tc.function.arguments!r}")
            print(f"  tool_call[{i}].function.arguments len: {len(tc.function.arguments or '')}")
    print()
    print("Full raw response (first 4KB):")
    try:
        print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False, default=str)[:4000])
    except Exception:
        print(repr(resp))


if __name__ == "__main__":
    main()
