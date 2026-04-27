"""FastAPI server for the Academy eval platform.

Routes:
  GET  /                       -> static dashboard
  GET  /api/runs               -> list runs (newest first)
  POST /api/runs               -> create + start a new run (background thread)
  GET  /api/runs/{run_id}      -> meta + episode list
  GET  /api/runs/{run_id}/episodes/{ep}  -> single episode (full decision log)
  WS   /ws/runs/{run_id}       -> live event stream (decision / status / episode_end)

A run is one (config, n_episodes) bundle. Episodes run sequentially in a
background thread; the UI subscribes via websocket while it's executing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import storage
from .frame_bus import FRAME_BUS
from .harness import EpisodeConfig, run_episode
from .metrics import aggregate

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Academy — Football RL eval")


# ----- in-process pubsub for live websocket streaming --------------------

class RunBroker:
    """In-memory event fan-out per run_id. Survives until process restart."""

    def __init__(self) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = {}
        self._buffers: dict[str, list[dict]] = {}  # last 200 events per run, replay on connect
        self._lock = threading.Lock()

    def publish(self, run_id: str, event: dict) -> None:
        with self._lock:
            buf = self._buffers.setdefault(run_id, [])
            buf.append(event)
            if len(buf) > 500:
                del buf[: len(buf) - 500]
            queues = list(self._subs.get(run_id, []))
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self, run_id: str) -> tuple[asyncio.Queue, list[dict]]:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subs.setdefault(run_id, []).append(q)
            backlog = list(self._buffers.get(run_id, []))
        return q, backlog

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            if run_id in self._subs:
                try:
                    self._subs[run_id].remove(q)
                except ValueError:
                    pass


BROKER = RunBroker()


# ----- API models --------------------------------------------------------

class RunCreate(BaseModel):
    n_episodes: int = Field(1, ge=1, le=50)
    scenario: str = Field("llm_5v5_full")
    # Decisions cap kept high — match length is governed by wall-clock now.
    max_decisions_total: int = Field(100_000, ge=1, le=1_000_000)
    # Default = 90 game minutes (5400s). Real-time pacing locks game-sec = wall-sec.
    max_wall_seconds: float = Field(5400.0, ge=10.0, le=10_800.0)
    obs_refresh_every_ticks: int = Field(25, ge=1, le=100)
    # 1:1 real-time. Set to 0 for "as fast as possible" (legacy headless eval).
    target_wall_fps: float = Field(50.0, ge=0.0, le=240.0)
    # Position snapshot for the live pitch view: every N env ticks (5 = 10 Hz at 50 fps).
    tick_stream_every_ticks: int = Field(5, ge=1, le=50)
    # 3D frame stream cadence (only meaningful with render=True). 3 = ~17 fps video.
    frame_stream_every_ticks: int = Field(3, ge=1, le=50)
    render: bool = Field(
        True,
        description="Render gfootball 3D scene (EGL off-screen when DISPLAY unset). "
                    "Required for the live video stream in the UI.",
    )
    stop_world_mode: bool = Field(
        False,
        description="Pause env while all agents decide; resume after all respond. "
                    "Agents see a stable world snapshot — no stale-obs problem. "
                    "Tradeoff: match pace = slowest agent's LLM latency.",
    )
    stop_world_timeout: float = Field(
        8.0, ge=1.0, le=60.0,
        description="Seconds to wait per decision cycle before timing out laggard agents.",
    )


# Map run_id -> active MultiAgentRunner so the stop endpoint can request it.
ACTIVE_RUNNERS: dict[str, Any] = {}


# ----- background runner -------------------------------------------------

def _run_in_background(run_id: str, body: RunCreate, loop: asyncio.AbstractEventLoop) -> None:
    """Worker thread: runs N episodes sequentially, publishes events."""

    def _publish(evt: dict) -> None:
        evt["t"] = time.time()
        # Schedule on the FastAPI loop so subscribers get it.
        asyncio.run_coroutine_threadsafe(_async_publish(run_id, evt), loop)

    started = time.time()
    config = EpisodeConfig(
        scenario=body.scenario,
        max_decisions_total=body.max_decisions_total,
        max_wall_seconds=body.max_wall_seconds,
        obs_refresh_every_ticks=body.obs_refresh_every_ticks,
        target_wall_fps=body.target_wall_fps,
        tick_stream_every_ticks=body.tick_stream_every_ticks,
        frame_stream_every_ticks=body.frame_stream_every_ticks,
        render=body.render,
        stop_world_mode=body.stop_world_mode,
        stop_world_timeout=body.stop_world_timeout,
    )

    meta: dict[str, Any] = {
        "run_id": run_id,
        "config": {
            "n_episodes": body.n_episodes,
            "scenario": body.scenario,
            "max_decisions_total": body.max_decisions_total,
            "max_wall_seconds": body.max_wall_seconds,
            "obs_refresh_every_ticks": body.obs_refresh_every_ticks,
            "target_wall_fps": body.target_wall_fps,
            "tick_stream_every_ticks": body.tick_stream_every_ticks,
            "frame_stream_every_ticks": body.frame_stream_every_ticks,
            "render": body.render,
            "stop_world_mode": body.stop_world_mode,
            "stop_world_timeout": body.stop_world_timeout,
        },
        "n_episodes": body.n_episodes,
        "completed_episodes": 0,
        "status": "running",
        "started_at": started,
        "ended_at": None,
        "episodes": [],
        "aggregate": {},
        "error": None,
    }
    storage.write_meta(run_id, meta)
    _publish({"type": "run_started", "run_id": run_id, "meta": meta})

    episode_summaries: list[dict] = []
    error: Optional[str] = None
    try:
        for i in range(body.n_episodes):
            ep_index = i + 1
            _publish({"type": "episode_start", "episode": ep_index})

            def _on_decision(entry: dict, _ep=ep_index) -> None:
                # Stream a trimmed decision (full log written to disk at end).
                light = {
                    "episode": _ep,
                    "decision": entry.get("decision"),
                    "slot": entry.get("slot"),
                    "label": entry.get("label"),
                    "env_tick": entry.get("env_tick"),
                    "lag_ticks": entry.get("lag_ticks"),
                    "llm_seconds": round(float(entry.get("llm_seconds", 0)), 3),
                    "skill": entry.get("skill"),
                }
                _publish({"type": "decision", **light})

            def _on_status(phase: str, payload: dict, _ep=ep_index) -> None:
                _publish({"type": "status", "episode": _ep, "phase": phase, **payload})

            def _on_tick(snap: dict, _ep=ep_index) -> None:
                _publish({"type": "tick", "episode": _ep, **snap})

            def _on_frame(frame_ndarray) -> None:
                """Encode RGB frame to JPEG and publish to /stream subscribers.

                Strict 1:1 with gfootball: native 1280×720, no resize, no
                color/brightness post-processing. JPEG q=95 is visually
                indistinguishable from PNG at this size, but ~5x smaller.
                Skips encode if no one's subscribed.
                """
                if FRAME_BUS.num_subscribers(run_id) == 0:
                    return
                try:
                    import cv2
                    bgr = cv2.cvtColor(frame_ndarray, cv2.COLOR_RGB2BGR)
                    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if ok:
                        FRAME_BUS.publish_threadsafe(run_id, jpg.tobytes(), loop)
                except Exception as e:
                    logger.warning("frame encode failed: %s", e)

            def _register_runner(r) -> None:
                ACTIVE_RUNNERS[run_id] = r

            try:
                result = run_episode(
                    config,
                    on_decision=_on_decision,
                    on_status=_on_status,
                    on_tick=_on_tick,
                    on_frame=_on_frame if body.render else None,
                    register_runner=_register_runner,
                )
            finally:
                ACTIVE_RUNNERS.pop(run_id, None)
            ep_payload = {
                "ep_index": ep_index,
                "config": asdict(result.config),
                "started_at": result.started_at,
                "ended_at": result.ended_at,
                "summary": result.summary,
                "decision_log": result.decision_log,
                "error": result.error,
            }
            storage.write_episode(run_id, ep_index, ep_payload)
            episode_summaries.append(result.summary)

            meta["completed_episodes"] = ep_index
            meta["episodes"] = [
                {
                    "ep_index": j + 1,
                    "summary": s,
                } for j, s in enumerate(episode_summaries)
            ]
            meta["aggregate"] = aggregate(episode_summaries)
            storage.write_meta(run_id, meta)
            _publish({
                "type": "episode_end",
                "episode": ep_index,
                "summary": result.summary,
                "aggregate": meta["aggregate"],
                "error": result.error,
            })
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.exception("run %s failed", run_id)
    finally:
        meta["status"] = "failed" if error else "completed"
        meta["ended_at"] = time.time()
        meta["error"] = error
        meta["aggregate"] = aggregate(episode_summaries)
        storage.write_meta(run_id, meta)
        _publish({"type": "run_end", "status": meta["status"], "error": error, "aggregate": meta["aggregate"]})


async def _async_publish(run_id: str, evt: dict) -> None:
    BROKER.publish(run_id, evt)


# ----- routes ------------------------------------------------------------

@app.get("/api/runs")
def api_list_runs() -> JSONResponse:
    return JSONResponse(storage.list_runs())


@app.post("/api/runs")
async def api_create_run(body: RunCreate) -> JSONResponse:
    run_id = storage.new_run_id()
    loop = asyncio.get_running_loop()
    t = threading.Thread(
        target=_run_in_background,
        args=(run_id, body, loop),
        daemon=True,
        name=f"eval-{run_id}",
    )
    t.start()
    return JSONResponse({"run_id": run_id, "status": "started"})


@app.get("/api/runs/{run_id}")
def api_get_run(run_id: str) -> JSONResponse:
    meta = storage.read_meta(run_id)
    if meta is None:
        raise HTTPException(404, f"run not found: {run_id}")
    return JSONResponse(meta)


@app.get("/api/runs/{run_id}/episodes/{ep}")
def api_get_episode(run_id: str, ep: int) -> JSONResponse:
    payload = storage.read_episode(run_id, ep)
    if payload is None:
        raise HTTPException(404, f"episode {ep} not found in run {run_id}")
    return JSONResponse(payload)


@app.get("/stream/{run_id}")
async def stream_video(run_id: str) -> StreamingResponse:
    """Live MJPEG of gfootball's 3D rendered view for an active run.

    Returns 'multipart/x-mixed-replace; boundary=frame' which any HTML <img>
    tag can render directly. Drops frames if the viewer can't keep up — the
    gfootball loop never blocks on this.
    """
    q = FRAME_BUS.subscribe(run_id)

    async def gen():
        try:
            while True:
                jpg = await q.get()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpg)).encode() + b"\r\n\r\n"
                    + jpg + b"\r\n"
                )
        finally:
            FRAME_BUS.unsubscribe(run_id, q)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store"},
    )


@app.post("/api/runs/{run_id}/stop")
def api_stop_run(run_id: str) -> JSONResponse:
    """Request the runner to end the current episode on the next tick.

    Idempotent — returns 200 even if the run already finished. Latency
    is one env tick (~20ms at 50 fps) before the loop checks the flag.
    """
    runner = ACTIVE_RUNNERS.get(run_id)
    if runner is None:
        return JSONResponse({"run_id": run_id, "status": "not_running"})
    try:
        runner.request_stop()
    except Exception as e:
        raise HTTPException(500, f"stop failed: {e}")
    return JSONResponse({"run_id": run_id, "status": "stop_requested"})


@app.websocket("/ws/runs/{run_id}")
async def ws_run(ws: WebSocket, run_id: str) -> None:
    await ws.accept()
    q, backlog = BROKER.subscribe(run_id)
    try:
        # Replay backlog so late connectors see the run from the start.
        for evt in backlog:
            await ws.send_text(json.dumps(evt, ensure_ascii=False))
        while True:
            evt = await q.get()
            await ws.send_text(json.dumps(evt, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    finally:
        BROKER.unsubscribe(run_id, q)


# Static last so /api/* takes precedence.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
