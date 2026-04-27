"""JSON persistence for eval runs.

Layout:
  eval_runs/
    <run_id>/
      meta.json          # config + status + aggregate metrics
      episode_001.json   # one file per episode (heavy: full decision log)
      episode_002.json
      ...
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent / "eval_runs"
_LOCK = threading.Lock()


def _ensure_root() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)


def new_run_id() -> str:
    """Sortable ID: YYYYMMDD-HHMMSS-<short uuid>."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}-{short}"


def _safe_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_id):
        raise ValueError(f"unsafe run_id: {run_id!r}")
    return run_id


def run_dir(run_id: str) -> Path:
    return ROOT / _safe_id(run_id)


def write_meta(run_id: str, meta: dict[str, Any]) -> None:
    _ensure_root()
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "meta.json.tmp"
    with _LOCK:
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(d / "meta.json")


def read_meta(run_id: str) -> Optional[dict[str, Any]]:
    p = run_dir(run_id) / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_episode(run_id: str, ep_index: int, payload: dict[str, Any]) -> None:
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    name = f"episode_{ep_index:03d}.json"
    tmp = d / (name + ".tmp")
    with _LOCK:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(d / name)


def read_episode(run_id: str, ep_index: int) -> Optional[dict[str, Any]]:
    p = run_dir(run_id) / f"episode_{ep_index:03d}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def list_runs() -> list[dict[str, Any]]:
    """Sorted newest-first; returns lightweight summaries (just meta.json)."""
    _ensure_root()
    rows: list[dict[str, Any]] = []
    for d in ROOT.iterdir():
        if not d.is_dir():
            continue
        meta_p = d / "meta.json"
        if not meta_p.exists():
            continue
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "run_id": d.name,
            "status": meta.get("status", "unknown"),
            "started_at": meta.get("started_at", 0),
            "ended_at": meta.get("ended_at"),
            "n_episodes": meta.get("n_episodes", 0),
            "completed_episodes": meta.get("completed_episodes", 0),
            "scenario": meta.get("config", {}).get("scenario", "?"),
            "aggregate": meta.get("aggregate", {}),
        })
    rows.sort(key=lambda r: r.get("started_at", 0), reverse=True)
    return rows
