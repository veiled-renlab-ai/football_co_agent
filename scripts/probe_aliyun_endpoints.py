"""Empirical probe: which Aliyun endpoint accepts our pt- prefix keys?"""
from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.getenv("ALIYUN_TOKEN_PLAN_API_KEYS", "").split(",")[0].strip()
print(f"Testing key: {key[:10]}...{key[-6:]}\n")

endpoints = [
    ("Token Plan cn-beijing",
     "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
     "qwen3.6-plus"),
    ("Token Plan intl sg",
     "https://token-plan-intl.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
     "qwen3.6-plus"),
    ("Coding Plan cn",
     "https://coding.dashscope.aliyuncs.com/v1",
     "qwen3-coder-plus"),
    ("Coding Plan intl",
     "https://coding-intl.dashscope.aliyuncs.com/v1",
     "qwen3-coder-plus"),
    ("Dashscope cn compat",
     "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "qwen-plus"),
    ("Dashscope intl compat",
     "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
     "qwen-plus"),
    ("MaaS cn compat (guess)",
     "https://maas.cn-beijing.aliyuncs.com/compatible-mode/v1",
     "qwen3.6-plus"),
    ("Bailian cn compat (guess)",
     "https://bailian.cn-beijing.aliyuncs.com/compatible-mode/v1",
     "qwen3.6-plus"),
]

for name, url, model in endpoints:
    c = OpenAI(api_key=key, base_url=url, timeout=10.0)
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        content = r.choices[0].message.content or ""
        print(f"[OK]    {name:<28} -> {content[:40]!r}")
    except Exception as e:
        es = str(e)
        code = "???"
        if "401" in es: code = "401"
        elif "404" in es: code = "404"
        elif "403" in es: code = "403"
        elif "resolve" in es.lower() or "name or service" in es.lower() or "connection" in es.lower():
            code = "DNS"
        elif "timeout" in es.lower(): code = "TIMEOUT"
        short = es.split("{")[0][:80] if "{" in es else es[:80]
        print(f"[{code}]  {name:<28} -> {short}")
