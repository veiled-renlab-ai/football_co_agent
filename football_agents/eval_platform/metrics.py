"""Derived metrics from a MultiAgentRunner decision log.

The runner returns a list of per-decision dicts with keys:
  decision, player_id, slot, env_tick, obs_tick, lag_ticks, llm_seconds, skill

Plus a final result with cumulative_reward / env_ticks / wall_seconds.

This module is pure: no IO, no globals, deterministic.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def serialize_skill(skill: Any) -> dict[str, Any]:
    """Skill is a frozen dataclass. Convert to plain dict + add tool_name."""
    if is_dataclass(skill):
        d = asdict(skill)
    else:
        d = {}
    d["tool_name"] = getattr(type(skill), "tool_name", type(skill).__name__)
    return d


def serialize_decision_log(log: list[dict]) -> list[dict]:
    """Make the runner's decision log JSON-safe (skill -> dict)."""
    out = []
    for d in log:
        entry = {k: v for k, v in d.items() if k != "skill"}
        entry["skill"] = serialize_skill(d["skill"])
        out.append(entry)
    return out


def latency_stats(log: list[dict]) -> dict[str, float]:
    """LLM latency p50/p95/avg from the log's llm_seconds field."""
    vals = [float(d["llm_seconds"]) for d in log if "llm_seconds" in d]
    if not vals:
        return {"p50": 0.0, "p95": 0.0, "avg": 0.0, "max": 0.0, "n": 0}
    return {
        "p50": _percentile(vals, 0.50),
        "p95": _percentile(vals, 0.95),
        "avg": sum(vals) / len(vals),
        "max": max(vals),
        "n": len(vals),
    }


def lag_stats(log: list[dict]) -> dict[str, float]:
    """Decision lag (env_tick - obs_tick) — observation staleness when LLM committed."""
    vals = [int(d["lag_ticks"]) for d in log if "lag_ticks" in d]
    if not vals:
        return {"avg": 0.0, "p95": 0.0, "max": 0}
    return {
        "avg": sum(vals) / len(vals),
        "p95": _percentile([float(v) for v in vals], 0.95),
        "max": max(vals),
    }


def skill_distribution(log: list[dict]) -> dict[str, int]:
    """Count of each tool_name across the whole run."""
    counter: Counter[str] = Counter()
    for d in log:
        s = d["skill"]
        if isinstance(s, dict):
            name = s.get("tool_name") or "unknown"
        else:
            name = getattr(type(s), "tool_name", type(s).__name__)
        counter[name] += 1
    return dict(counter.most_common())


def per_agent_breakdown(log: list[dict], slot_to_label: dict[int, str]) -> list[dict]:
    """One row per slot: decisions / avg latency / top skills."""
    by_slot: dict[int, list[dict]] = {}
    for d in log:
        by_slot.setdefault(int(d["slot"]), []).append(d)

    rows = []
    for slot in sorted(by_slot):
        sub = by_slot[slot]
        lat = latency_stats(sub)
        skills = skill_distribution(sub)
        rows.append({
            "slot": slot,
            "label": slot_to_label.get(slot, f"slot {slot}"),
            "decisions": len(sub),
            "llm_latency_p50": round(lat["p50"], 3),
            "llm_latency_p95": round(lat["p95"], 3),
            "top_skills": list(skills.items())[:5],  # [(name, count), ...]
        })
    return rows


def fallback_rate(log: list[dict]) -> float:
    """Fraction of decisions that came back with an LLM error (excluded by harness for now).

    Reserved: the runner currently logs only successful decisions. Fallback
    triggers happen between decisions (motor finished, no LLM intent yet).
    The harness counts fallback installs separately and feeds it in.
    """
    if not log:
        return 0.0
    errs = sum(1 for d in log if d.get("error"))
    return errs / len(log)


def summarize_episode(
    log: list[dict],
    cumulative_reward: float,
    env_ticks: int,
    wall_seconds: float,
    n_agents: int,
    fallback_installs: int,
    slot_to_label: dict[int, str],
) -> dict[str, Any]:
    """Bundle one episode's headline numbers — the row we show in the UI."""
    blue = max(int(cumulative_reward), 0)
    red = max(-int(cumulative_reward), 0)
    return {
        "score_blue": blue,
        "score_red": red,
        "outcome": (
            "blue_win" if cumulative_reward > 0
            else "red_win" if cumulative_reward < 0
            else "draw"
        ),
        "decisions_total": len(log),
        "env_ticks": env_ticks,
        "wall_seconds": round(wall_seconds, 2),
        "n_agents": n_agents,
        "llm_latency": latency_stats(log),
        "lag": lag_stats(log),
        "skill_distribution": skill_distribution(log),
        "per_agent": per_agent_breakdown(log, slot_to_label),
        "fallback_installs": fallback_installs,
        "fallback_per_decision": (
            round(fallback_installs / max(1, len(log)), 3)
        ),
    }


def aggregate(episodes: list[dict]) -> dict[str, Any]:
    """Aggregate metrics across N episodes for the run header."""
    if not episodes:
        return {
            "n_episodes": 0,
            "blue_wins": 0, "red_wins": 0, "draws": 0,
            "avg_blue_score": 0.0, "avg_red_score": 0.0,
            "avg_decisions": 0.0, "avg_wall_seconds": 0.0,
            "llm_latency_p50": 0.0, "llm_latency_p95": 0.0,
        }
    blue_wins = sum(1 for e in episodes if e["outcome"] == "blue_win")
    red_wins = sum(1 for e in episodes if e["outcome"] == "red_win")
    draws = sum(1 for e in episodes if e["outcome"] == "draw")
    p50s = [e["llm_latency"]["p50"] for e in episodes]
    p95s = [e["llm_latency"]["p95"] for e in episodes]
    return {
        "n_episodes": len(episodes),
        "blue_wins": blue_wins,
        "red_wins": red_wins,
        "draws": draws,
        "blue_win_rate": round(blue_wins / len(episodes), 3),
        "avg_blue_score": round(sum(e["score_blue"] for e in episodes) / len(episodes), 2),
        "avg_red_score": round(sum(e["score_red"] for e in episodes) / len(episodes), 2),
        "avg_decisions": round(sum(e["decisions_total"] for e in episodes) / len(episodes), 1),
        "avg_wall_seconds": round(sum(e["wall_seconds"] for e in episodes) / len(episodes), 1),
        "llm_latency_p50": round(sum(p50s) / len(p50s), 3),
        "llm_latency_p95": round(sum(p95s) / len(p95s), 3),
    }
