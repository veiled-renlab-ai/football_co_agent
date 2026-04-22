"""Benchmark candidate LLM models on tool-calling for our football skill schema.

Measures:
  - latency (single call)
  - whether the model returns a valid tool call
  - whether the tool args satisfy our coordinate range [-1, 1]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from football_agents.skills import all_tool_schemas
from football_agents.prompts import build_system_prompt

load_dotenv()

CANDIDATES = [
    "doubao-seed-1-6-flash-250828",      # current (flash)
    "doubao-seed-1-6-flash-250715",      # older flash variant
    "doubao-seed-1-6-flash-250615",      # oldest flash variant
    "doubao-seed-1-6-lite-251015",       # seed-1.6 lite
    "doubao-1-5-lite-32k-250115",        # older lite
    "doubao-lite-32k-240828",            # very old lite
    "doubao-pro-32k-functioncall-241028", # tuned for function-call
]

# Realistic test scenario — like what the agent sees mid-game
SYSTEM = build_system_prompt(player_id=1, role="CB")
USER = """\
Tick 18 | Clock 00:01 | Score 0-0

# Your state
  pos=(+0.74, -0.00)  vel=(+0.00, +0.00)  facing=+0°  stamina=100%  has_ball=True

# Ball:  pos=(+0.77, -0.00)  distance_to_you=0.03  carrier=YOU

# Teammates you see (0)
  (none in your FOV)

# Opponents you see (1)
  #0  pos=(-1.01, +0.00)  d=1.75

Decide your next skill.
"""


def test_model(model_id: str) -> dict:
    base_url = os.getenv("VOLCENGINE_BASE_URL")
    api_key = os.getenv("VOLCENGINE_API_KEY")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    tools = all_tool_schemas()

    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.4,
        )
        dt = time.monotonic() - t0
        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            return {"model": model_id, "latency_s": dt, "ok": False, "why": "no tool call",
                    "reasoning": (msg.content or "")[:120]}
        import json
        tc = tcs[0]
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            return {"model": model_id, "latency_s": dt, "ok": False, "why": "bad json args",
                    "tool_name": tc.function.name, "raw_args": tc.function.arguments}

        # Validate coords if present
        coord_problem = []
        for k in ("target_x", "target_y"):
            if k in args and not -1.05 <= float(args[k]) <= 1.05:
                coord_problem.append(f"{k}={args[k]} out of range")

        return {
            "model": model_id,
            "latency_s": dt,
            "ok": True,
            "tool_name": tc.function.name,
            "tool_args": args,
            "reasoning": (msg.content or "")[:120],
            "coord_problem": coord_problem,
        }
    except Exception as e:
        dt = time.monotonic() - t0
        return {"model": model_id, "latency_s": dt, "ok": False, "why": f"exception: {e!r}"[:200]}


def main() -> None:
    print(f"{'model':<40s} {'lat(s)':>7s}  {'ok':<3s}  details")
    print("─" * 110)
    for m in CANDIDATES:
        r = test_model(m)
        ok = "✓" if r["ok"] and not r.get("coord_problem") else ("△" if r["ok"] else "✗")
        details = ""
        if r["ok"]:
            details = f"{r['tool_name']}({r['tool_args']})"
            if r.get("coord_problem"):
                details += f"  ⚠ {r['coord_problem']}"
        else:
            details = r.get("why", "")
        print(f"{m:<40s} {r['latency_s']:>6.1f}s  {ok:<3s}  {details}")


if __name__ == "__main__":
    main()
