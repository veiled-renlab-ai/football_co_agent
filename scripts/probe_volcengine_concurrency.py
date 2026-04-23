"""Empirical concurrency probe for 火山方舟 Coding Plan.

Goal: discover the actual concurrency ceiling for our doubao-seed-2-0-lite
endpoint so we can size the BoundedSemaphore for 5v5 (10 agents) and 11v11
(22 agents) simulations.

Methodology:
  - Sweep N ∈ {1, 3, 5, 8, 10, 15} concurrent calls (one burst per N)
  - Each call mimics LLMPlayer.choose_skill: ~3-5k tok input, ~200 tok out,
    tool definition like make_invoke_skill_tool, thinking-mode disabled
  - Capture per-call latency, HTTP status, exception type, response headers
    (looking for x-ratelimit-* if exposed)
  - Stop the sweep early if 429s appear at any N
  - Print a summary table

Cost discipline: 1+3+5+8+10+15 = 42 calls (under the 60-call budget).
"""
from __future__ import annotations

import os
import statistics
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Allow `from football_agents.skills import ...` if needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()


# ---------------------------------------------------------------------------
# Realistic payload — mimic LLMPlayer.choose_skill prompt size (~3-5k in)
# ---------------------------------------------------------------------------

# A Chinese football persona block, ~1k tokens. We repeat ~3x to get ~3k.
PERSONA_BLOCK = """\
你是一名职业足球运动员，正在参加一场 5v5 业余足球比赛。你扮演的角色是中后卫 (CB)，
球衣号码 4 号，惯用脚右脚，身高 184cm，体重 78kg。你的踢球风格偏向稳健防守、
出球清晰、对抗强硬。你的位置感非常好，擅长预判对方的传球路线，第一时间断球或者
封堵传球角度。在没有压力的时候，你倾向于稳稳地把球交给中场组织者，而不是冒险长传。
当对方持球突进到你身前时，你优先选择延缓 (delay) 而不是贸然铲球——除非你有十足
把握能干净地拿到球，或者对方已经突进到禁区前沿、必须立刻终止他的进攻。

你的体能现在是 100%，刚刚开场。你的搭档中后卫是 5 号，他防守站位偏向你的左侧；
后腰是 6 号，他通常会回撤到你和搭档之间接应你的出球。你的边后卫是 2 号 (右) 和
3 号 (左)，他们在对方反击时往往跟不上你的回防速度。门将是 1 号，他出击范围一般，
对高空球的判断中规中矩，所以禁区内的高球你需要主动去争。

战术上，球队执行的是 4-1-2-1 体系，强调高位逼抢和快速反击。当对方门将持球时，前
锋会从中路压上去，逼迫对方门将开大脚或者出球到边路；这时你需要稍稍前压到中线
附近，准备争抢二点球。当己方持球推进到对方半场时，你保持在中圈靠下的位置，作为
最后一道防线，时刻警惕对方反击。

你的语言习惯：在场上你说中文，沟通简短直接。你会喊“我的”、“身后”、“别让他转身”、
“顶上去”这样的指令；你不喜欢说废话，更不会在比赛中讲冗长的战术分析。

请基于以上人设，结合每一帧给你的场上观察，做出最贴近真实球员决策的选择。每个 tick
你只能选择一个 skill 来执行，参数要符合 schema 描述。如果你判断当前没有更好的选
择，可以选 hold_position 站位不动；这并不是失败的选择，恰恰是优秀防守球员经常做
的事——保持位置而不是被对方调动。
"""

SYSTEM_PROMPT = (PERSONA_BLOCK * 3) + """

# 你能用的动作 (Available Skills)

## `move_to`  (MOVE)
跑到场上指定坐标。无球时调整位置、跑空当、或回防时用。
参数:
  - target_x: float
  - target_y: float
  - urgency: 'walk' | 'jog' | 'sprint' (默认 'jog')

## `hold_position`  (MOVE)
站住不动，保持当前位置。等队友传球或维持阵型时用。
（无参数）

## `dribble_toward`  (ATTACK)
带球朝指定坐标突破推进。需要脚下有球。
参数:
  - target_x: float
  - target_y: float
  - urgency: 'walk' | 'jog' | 'sprint' (默认 'jog')

## `pass_to`  (ATTACK)
把球传给指定球衣号码的队友。
参数:
  - target_player_id: int
  - pass_type: 'short' | 'long' | 'through' (默认 'short')

## `shoot`  (ATTACK)
朝对方球门射门，可瞄准球门的某个区域。需要脚下有球。
参数:
  - target_zone: 'top_left' | 'top_center' | 'top_right' | ...

## `mark`  (DEFEND)
盯防指定对手 —— 站位在他和己方球门之间。
参数:
  - opponent_id: int

## `press`  (DEFEND)
上抢指定对手 —— 全速冲过去逼抢。
参数:
  - opponent_id: int

## `tackle`  (DEFEND)
对身前的持球对手发起铲球。
（无参数）

## `scan_behind`  (PERCEIVE)
回头扫一眼背后。
（无参数）

请严格按 schema 调用 invoke_skill 工具；不要用自然语言回复。
"""

# ~1k token user message — current observation snapshot
USER_MESSAGE = """\
Tick 142 | Clock 00:14 | Score 0-0 | Phase: 防守组织阶段

# Your state
  pos=(+0.32, -0.08)  vel=(+0.04, -0.01)  facing=-12°  stamina=98%
  has_ball=False  role=CB  jersey=#4

# Ball
  pos=(+0.18, +0.12)  距离你=0.27  持球者: 对方 #9 (CF)

# Teammates you see (4)
  #5  CB   pos=(+0.41, +0.06)  d=0.18  stamina=99%  状态: 协防站位
  #6  DM   pos=(+0.05, -0.02)  d=0.28  stamina=95%  状态: 回追中
  #2  RB   pos=(+0.22, -0.32)  d=0.27  stamina=92%  状态: 内收
  #1  GK   pos=(+0.96, +0.00)  d=0.65  stamina=100% 状态: 站位

# Opponents you see (5)
  #9  CF   pos=(+0.18, +0.12)  d=0.27  持球, facing=+170°  正面对你
  #10 AM   pos=(-0.04, -0.18)  d=0.38  无球, 朝禁区跑动
  #7  RW   pos=(+0.10, +0.36)  d=0.50  无球, 拉边
  #11 LW   pos=(+0.05, -0.40)  d=0.43  无球, 拉边
  #8  DM   pos=(-0.32, +0.04)  d=0.65  无球, 后插上准备

# Recent communication (last 3 calls in your audible range)
  T=140  #6 (DM):    "上!"
  T=141  #5 (CB):    "我顶 #10"
  T=141  YOU:        (no call)

# Your last 3 actions
  T=139  hold_position   -> 还在原地
  T=140  mark(#9)        -> 切换到盯人
  T=141  mark(#9)        -> 继续跟随

对方 #9 持球面对你，距离 0.27，正面对峙。#10 在你身后插上禁区方向。#5 已经
喊话他来盯 #10。你的搭档已经分工：你需要立刻处理 #9 这个持球点。

Decide your next skill. Call invoke_skill(skill_name=..., args={...}) with one
of the skills listed above. Reply with the tool call only — no natural language.
"""

# Tool definition — matches make_invoke_skill_tool from skills.py
INVOKE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "invoke_skill",
        "description": (
            "执行一个动作。skill_name 必须是 system prompt 里 'Available Skills' "
            "列表中某一个 name；args 是该 skill 的参数字典。如果 skill 不接受参数，"
            "args 传 {}。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "enum": [
                        "move_to", "hold_position", "dribble_toward",
                        "pass_to", "shoot", "receive_ball",
                        "mark", "press", "tackle",
                        "scan_behind", "track", "call",
                    ],
                    "description": "要执行的动作的 name (snake_case)。",
                },
                "args": {
                    "type": "object",
                    "description": "该动作的参数字典；无参数时传 {}。",
                },
            },
            "required": ["skill_name", "args"],
        },
    },
}


# ---------------------------------------------------------------------------
# Probe machinery
# ---------------------------------------------------------------------------

@dataclass
class CallResult:
    call_id: int
    n_concurrency: int
    latency_s: float
    ok: bool
    http_status: Optional[int] = None
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    retry_after: Optional[str] = None
    rate_headers: dict[str, str] = field(default_factory=dict)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def make_client() -> OpenAI:
    api_key = os.getenv("VOLCENGINE_API_KEY", "")
    base_url = os.getenv("VOLCENGINE_BASE_URL",
                         "https://ark.cn-beijing.volces.com/api/coding/v3")
    if not api_key:
        raise RuntimeError("VOLCENGINE_API_KEY is empty — check .env")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)


def one_call(client: OpenAI, model: str, call_id: int,
             n_concurrency: int) -> CallResult:
    """One chat completion. Uses with_raw_response to capture HTTP headers."""
    t0 = time.monotonic()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_MESSAGE},
            ],
            tools=[INVOKE_SKILL_TOOL],
            tool_choice="auto",
            temperature=0.4,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        dt = time.monotonic() - t0
        # Pull headers
        headers = dict(raw.headers)
        rate_headers = {k: v for k, v in headers.items()
                        if "rate" in k.lower() or "limit" in k.lower()
                        or "remain" in k.lower() or "retry" in k.lower()}
        # Parse the response payload
        parsed = raw.parse()
        usage = getattr(parsed, "usage", None)
        return CallResult(
            call_id=call_id,
            n_concurrency=n_concurrency,
            latency_s=dt,
            ok=True,
            http_status=raw.status_code,
            rate_headers=rate_headers,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )
    except Exception as e:  # noqa: BLE001
        dt = time.monotonic() - t0
        # Try to extract HTTP status / retry-after from openai.APIError
        status = getattr(e, "status_code", None)
        retry_after = None
        rate_headers: dict[str, str] = {}
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                hdrs = dict(resp.headers)
                retry_after = hdrs.get("retry-after") or hdrs.get("Retry-After")
                rate_headers = {k: v for k, v in hdrs.items()
                                if "rate" in k.lower() or "limit" in k.lower()
                                or "remain" in k.lower() or "retry" in k.lower()}
            except Exception:
                pass
        return CallResult(
            call_id=call_id,
            n_concurrency=n_concurrency,
            latency_s=dt,
            ok=False,
            http_status=status,
            error_type=type(e).__name__,
            error_msg=str(e)[:300],
            retry_after=retry_after,
            rate_headers=rate_headers,
        )


def run_burst(client: OpenAI, model: str, n: int,
              start_call_id: int) -> list[CallResult]:
    """Fire n parallel calls and wait for all of them."""
    results: list[CallResult] = []
    print(f"\n--- Burst N={n} ({n} concurrent calls) ---", flush=True)
    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [
            ex.submit(one_call, client, model, start_call_id + i, n)
            for i in range(n)
        ]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            tag = "OK " if r.ok else "ERR"
            extra = ""
            if not r.ok:
                extra = f" [{r.error_type}: {r.error_msg[:80] if r.error_msg else ''}]"
            print(f"  call={r.call_id} {tag} status={r.http_status} "
                  f"lat={r.latency_s:5.2f}s{extra}", flush=True)
    t_total = time.monotonic() - t_start
    print(f"  burst total wall time: {t_total:.2f}s", flush=True)
    return results


def summarize(results: list[CallResult]) -> dict[int, dict[str, Any]]:
    """Group by N and compute stats."""
    by_n: dict[int, list[CallResult]] = {}
    for r in results:
        by_n.setdefault(r.n_concurrency, []).append(r)

    summary: dict[int, dict[str, Any]] = {}
    for n, rs in sorted(by_n.items()):
        ok_rs = [r for r in rs if r.ok]
        bad_rs = [r for r in rs if not r.ok]
        latencies = [r.latency_s for r in ok_rs]
        statuses: dict[Any, int] = {}
        err_types: dict[str, int] = {}
        for r in rs:
            statuses[r.http_status] = statuses.get(r.http_status, 0) + 1
            if r.error_type:
                err_types[r.error_type] = err_types.get(r.error_type, 0) + 1
        prompt_toks = [r.prompt_tokens for r in ok_rs if r.prompt_tokens]
        comp_toks = [r.completion_tokens for r in ok_rs if r.completion_tokens]
        summary[n] = {
            "calls": len(rs),
            "success": len(ok_rs),
            "fail": len(bad_rs),
            "p50_s": statistics.median(latencies) if latencies else None,
            "p95_s": (statistics.quantiles(latencies, n=20)[18]
                      if len(latencies) >= 5 else max(latencies) if latencies else None),
            "min_s": min(latencies) if latencies else None,
            "max_s": max(latencies) if latencies else None,
            "statuses": statuses,
            "err_types": err_types,
            "any_429": any(r.http_status == 429 for r in rs),
            "retry_after": [r.retry_after for r in rs if r.retry_after],
            "rate_headers_seen": {k for r in rs for k in r.rate_headers},
            "avg_prompt_tokens": (sum(prompt_toks) / len(prompt_toks)
                                  if prompt_toks else None),
            "avg_completion_tokens": (sum(comp_toks) / len(comp_toks)
                                      if comp_toks else None),
        }
    return summary


def print_table(summary: dict[int, dict[str, Any]]) -> None:
    print("\n" + "=" * 88)
    print("CONCURRENCY PROBE — SUMMARY")
    print("=" * 88)
    header = (f"{'N':>3} | {'OK/Fail':>8} | {'p50':>7} | {'p95':>7} | "
              f"{'min':>6} | {'max':>6} | {'429?':>4} | statuses")
    print(header)
    print("-" * 88)
    for n, s in summary.items():
        p50 = f"{s['p50_s']:.2f}s" if s['p50_s'] is not None else "  -  "
        p95 = f"{s['p95_s']:.2f}s" if s['p95_s'] is not None else "  -  "
        mn = f"{s['min_s']:.2f}s" if s['min_s'] is not None else "  -  "
        mx = f"{s['max_s']:.2f}s" if s['max_s'] is not None else "  -  "
        flag = "YES" if s['any_429'] else "no"
        print(f"{n:>3} | {s['success']:>3}/{s['fail']:<4} | {p50:>7} | {p95:>7} | "
              f"{mn:>6} | {mx:>6} | {flag:>4} | {s['statuses']}")
        if s['err_types']:
            print(f"    err_types: {s['err_types']}")
        if s['retry_after']:
            print(f"    retry_after: {s['retry_after']}")
        if s['rate_headers_seen']:
            print(f"    rate-related headers: {sorted(s['rate_headers_seen'])}")
        if s['avg_prompt_tokens']:
            print(f"    avg tokens: prompt={s['avg_prompt_tokens']:.0f} "
                  f"completion={s['avg_completion_tokens']:.0f}")
    print("=" * 88)

    # Cost / token totals
    total_prompt = sum(
        (s['avg_prompt_tokens'] or 0) * s['success'] for s in summary.values()
    )
    total_comp = sum(
        (s['avg_completion_tokens'] or 0) * s['success'] for s in summary.values()
    )
    total_calls = sum(s['calls'] for s in summary.values())
    total_ok = sum(s['success'] for s in summary.values())
    total_fail = sum(s['fail'] for s in summary.values())
    print(f"\nTotal probe calls: {total_calls} ({total_ok} ok, {total_fail} fail)")
    print(f"Total prompt tokens (estimated): {total_prompt:.0f}")
    print(f"Total completion tokens (estimated): {total_comp:.0f}")


def main() -> None:
    model = os.getenv("VOLCENGINE_MODEL", "doubao-seed-2-0-lite-260215")
    print(f"Model: {model}")
    print(f"Base URL: {os.getenv('VOLCENGINE_BASE_URL')}")
    print(f"System prompt length (chars): {len(SYSTEM_PROMPT)}")
    print(f"User message length (chars): {len(USER_MESSAGE)}")

    client = make_client()
    sweep = [1, 3, 5, 8, 10, 15]
    all_results: list[CallResult] = []
    next_id = 0
    stop_after_429 = False

    for n in sweep:
        if stop_after_429:
            print(f"\nSkipping N={n} (already saw 429s — staying within budget).")
            continue
        burst = run_burst(client, model, n, start_call_id=next_id)
        next_id += n
        all_results.extend(burst)
        if any(r.http_status == 429 for r in burst):
            print(f"\n!!! 429 detected at N={n} — stopping further bursts. !!!")
            stop_after_429 = True
        # Short pause between bursts to avoid bleed-over with any per-minute window
        time.sleep(2.0)

    summary = summarize(all_results)
    print_table(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
