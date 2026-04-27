"""In-process per-run JPEG frame bus.

Producers: the gfootball worker thread captures `obs['frame']` (RGB ndarray),
encodes it to JPEG via cv2.imencode, and publishes the bytes here.

Consumers: each `/stream/{run_id}` MJPEG StreamingResponse handler subscribes
to a per-run asyncio.Queue and yields the bytes back to the browser as
multipart frames.

Backpressure policy: drop on full. If a viewer's connection lags, we'd rather
drop a frame than slow the gfootball loop.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Optional


class FrameBus:
    """Per-run pub/sub for JPEG frame bytes."""

    def __init__(self) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = {}
        self._lock = threading.Lock()

    def subscribe(self, run_id: str) -> asyncio.Queue:
        # Maxsize=2: keep one in flight + one queued; drop anything older.
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        with self._lock:
            self._subs.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            if run_id in self._subs:
                try:
                    self._subs[run_id].remove(q)
                except ValueError:
                    pass

    def publish_threadsafe(
        self,
        run_id: str,
        jpeg_bytes: bytes,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Called from the worker thread. Schedules a put_nowait on each
        subscriber's queue via the FastAPI loop. Drops if any queue is full
        so the worker never blocks on a slow viewer."""
        with self._lock:
            queues = list(self._subs.get(run_id, []))
        if not queues:
            return
        for q in queues:
            asyncio.run_coroutine_threadsafe(_put_or_drop(q, jpeg_bytes), loop)

    def num_subscribers(self, run_id: str) -> int:
        with self._lock:
            return len(self._subs.get(run_id, []))


async def _put_or_drop(q: asyncio.Queue, item: bytes) -> None:
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        # Drop oldest frame, queue the newest. Keeps the stream "live" for the
        # viewer rather than building up a backlog of stale frames.
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass


FRAME_BUS = FrameBus()
