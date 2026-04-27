"""Academy — eval platform for the 5v5 LLM agent football sim.

Submodules:
  metrics   pure functions deriving stats from a decision log
  harness   runs N episodes headless, captures per-episode metrics
  storage   JSON persistence to eval_runs/
  server    FastAPI + websocket frontend
"""
