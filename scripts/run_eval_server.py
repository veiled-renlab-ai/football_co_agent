"""Launch the Academy eval platform.

  cd ~/football_RL && python3 -u scripts/run_eval_server.py

Open http://localhost:8000 in browser. If running inside WSL, the same URL
works from Windows browser thanks to WSL2's automatic port forwarding.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# CRITICAL: pop $DISPLAY BEFORE any gfootball import so the C++ renderer
# auto-falls-back to EGL off-screen rendering. Otherwise it tries to open an
# SDL window in WSLg (which we don't want — we stream the rendered frames
# into the web UI instead). This must be the first thing in the script.
os.environ.pop("DISPLAY", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402  (sys.path setup must come first)

from football_agents.eval_platform.server import app  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true",
                   help="dev mode: auto-reload on file changes")
    args = p.parse_args()

    print("=" * 70)
    print(" ACADEMY · 5v5 LLM agent eval platform")
    print(f" → http://localhost:{args.port}")
    print("=" * 70)

    uvicorn.run(
        "football_agents.eval_platform.server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
