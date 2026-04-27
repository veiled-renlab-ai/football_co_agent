"""Comprehensive Volcengine ARK Coding Plan model probe.

Tests two things per candidate model:
  1. Auth/availability (does the model accept a request?)
  2. Round-trip latency for a tiny tool-use turn (≈ what the agent does)

Output: a sorted table of {OK | latency} per model.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
key = os.getenv("VOLCENGINE_API_KEY", "").strip()
if not key:
    print("missing VOLCENGINE_API_KEY in .env")
    sys.exit(1)

print(f"Testing key: {key[:10]}...{key[-6:]}")
print(f"Endpoint:    https://ark.cn-beijing.volces.com/api/coding/v3\n")

c = OpenAI(
    api_key=key,
    base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
    timeout=30.0,
)

# Expanded candidate list: code/pro/lite/flash variants of Doubao, plus
# DeepSeek / Kimi / GLM / MiniMax variations seen on 火山方舟 documentation.
CANDIDATES = [
    # Doubao seed 2.0 family
    "ark-code-latest",
    "doubao-seed-code",
    "doubao-seed-2-0-code",
    "doubao-seed-2-0-code-250915",
    "doubao-seed-2-0-pro",
    "doubao-seed-2-0-pro-250915",
    "doubao-seed-2-0-lite",
    "doubao-seed-2-0-lite-250915",
    "doubao-seed-2-0-lite-260215",
    # Doubao seed 1.6 family
    "doubao-seed-1-6",
    "doubao-seed-1-6-250615",
    "doubao-seed-1-6-flash",
    "doubao-seed-1-6-flash-250828",
    "doubao-seed-1-6-flash-250715",
    "doubao-seed-1-6-flash-250615",
    "doubao-seed-1-6-lite-251015",
    # Doubao 1.5 family
    "doubao-1-5-pro-32k-250115",
    "doubao-1-5-pro-256k-250115",
    "doubao-1-5-lite-32k-250115",
    "doubao-1-5-thinking-pro",
    # DeepSeek
    "deepseek-v3-2",
    "deepseek-v3.2",
    "deepseek-v3-1",
    "deepseek-r1",
    # Kimi
    "kimi-k2-5",
    "kimi-k2",
    # GLM
    "glm-4-7",
    "glm-4.7",
    "glm-5",
    "glm-4.6",
    "glm-4-6",
    # MiniMax
    "MiniMax-M1",
    "MiniMax-M2",
    "MiniMax-M2-5",
    "minimax-m2-5",
]

# A simple tool-use probe (mimics what our agent does).
TOOLS = [{
    "type": "function",
    "function": {
        "name": "pick_action",
        "description": "Choose one football action.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["pass", "shoot", "hold"]},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}]


def short_err(es: str) -> tuple[str, str]:
    code = "ERR"
    if "UnsupportedModel" in es: code = "404"
    elif "InvalidParameter" in es or "BadRequest" in es: code = "400"
    elif "401" in es: code = "401"
    elif "ModelNotFound" in es or "404" in es: code = "404"
    elif "RateLimit" in es or "429" in es: code = "429"
    elif "timeout" in es.lower(): code = "TIMEOUT"
    short = es.split("Request")[0][:80] if "Request" in es else es[:80]
    return code, short


results: list[tuple[str, str, float, str, bool]] = []  # (model, status, lat, note, tools_ok)

for m in CANDIDATES:
    t0 = time.monotonic()
    try:
        r = c.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": "You pick football actions."},
                {"role": "user", "content": "Pick any action."},
            ],
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "pick_action"}},
            max_tokens=64,
        )
        dt = time.monotonic() - t0
        msg = r.choices[0].message
        used_tool = bool(msg.tool_calls)
        first_arg = ""
        if used_tool:
            first_arg = msg.tool_calls[0].function.arguments[:30]
        results.append((m, "OK", dt, first_arg, used_tool))
    except Exception as e:
        dt = time.monotonic() - t0
        code, short = short_err(str(e))
        results.append((m, code, dt, short, False))

# Print as a table sorted: OK first by latency, then errors.
ok = sorted([r for r in results if r[1] == "OK"], key=lambda r: r[2])
err = [r for r in results if r[1] != "OK"]

print(f"\n{'='*88}")
print(f"{'MODEL':<35} {'STATUS':<8} {'LAT':>7} {'TOOL':>6}  NOTE")
print("=" * 88)
for m, st, lat, note, used in ok:
    print(f"{m:<35} {st:<8} {lat:>6.2f}s {'✓' if used else '✗':>6}  {note}")
print("-" * 88)
for m, st, lat, note, used in err:
    print(f"{m:<35} {st:<8} {lat:>6.2f}s {'-':>6}  {note}")
print("=" * 88)
print(f"\n{len(ok)} working / {len(results)} tested")
if ok:
    fastest = ok[0]
    print(f"Fastest tool-use:  {fastest[0]}  ({fastest[2]:.2f}s)")
