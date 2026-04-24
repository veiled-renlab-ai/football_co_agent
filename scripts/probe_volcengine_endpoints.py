"""Probe which Coding Plan models work with current VOLCENGINE_API_KEY."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
key = os.getenv("VOLCENGINE_API_KEY", "").strip()
print(f"Testing key: {key[:10]}...{key[-6:]}")
print(f"Endpoint:    https://ark.cn-beijing.volces.com/api/coding/v3\n")

c = OpenAI(
    api_key=key,
    base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
    timeout=20.0,
)

# Candidate model strings — variants of names published by 火山方舟 Coding Plan
candidates = [
    "auto",
    "ark-code-latest",
    "doubao-seed-code",
    "doubao-seed-2-0-code",
    "doubao-seed-2-0-code-250915",
    "doubao-seed-2-0-pro",
    "doubao-seed-2-0-pro-250915",
    "doubao-seed-1-6",
    "doubao-seed-1-6-250615",
    "deepseek-v3-2",
    "deepseek-v3.2",
    "kimi-k2-5",
    "glm-4-7",
    "glm-4.7",
]

for m in candidates:
    try:
        r = c.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
        )
        content = r.choices[0].message.content or ""
        print(f"[OK]      {m:<35} -> {content[:40]!r}")
    except Exception as e:
        es = str(e)
        short = es.split("Request")[0][:90] if "Request" in es else es[:90]
        code = "ERR"
        if "UnsupportedModel" in es: code = "404"
        elif "InvalidParameter" in es or "BadRequest" in es: code = "400"
        elif "401" in es: code = "401"
        elif "ModelNotFound" in es or "404" in es: code = "404"
        print(f"[{code}]     {m:<35} -> {short}")
