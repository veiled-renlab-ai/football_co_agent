"""football_agents — multi-agent LLM football simulation.

Each player is modeled as an autonomous LLM agent acting through an intent-level
Skill API on top of Google Research Football's 3D physics engine.

Architecture (mirrors human cognition):

    BRAIN  (LLM, every 5-10 ticks)   — chooses a Skill (intent)
       │
       ▼
    MOTOR  (state machines, per tick) — translates Skill into gfootball atomic actions
       │
       ▼
    BODY   (gfootball, per tick)      — 3D physics simulation

A separate PERCEPTION layer filters the env's god view into egocentric
observations (FOV cone, distance falloff, attention cap, short-term memory)
so each player only "sees" what they realistically could.

See DEV_PLAN.md at the project root for the full development roadmap.
"""

__version__ = "0.0.1"
