"""End-to-end smoke test for 11v11 + multi-channel wiring (no LLM call)."""
import logging
import os
from collections import Counter

logging.basicConfig(level=logging.WARNING)

# Reads ARK_KEYS from .env (auto-loaded by football_agents.llm_client at module import).
# If you don't have a .env yet: ARK_KEYS=key1,key2 in project root.

from football_agents.env import FootballEnvAdapter
from football_agents.eval_platform.harness import _build_agents
from football_agents.llm_client import build_channel_pool
from football_agents.message_bus import TeamMessageBus
from football_agents.players import TEAM_BLUE_11V11, TEAM_RED_11V11

print("[1/4] building env scenario llm_11v11_full ...")
env = FootballEnvAdapter(
    scenario="llm_11v11_full", render=False,
    n_controlled_left=11, n_controlled_right=11,
    primary_player_slot=0, physics_steps_per_frame=2,
)
env.reset()
print(f"  OK env ticks={env.tick} done={env.done}")
print(f"  left_team has {len(env.raw_obs['left_team'])} players")
print(f"  right_team has {len(env.raw_obs['right_team'])} players")
print(f"  left roles: {list(env.raw_obs['left_team_roles'])}")

print("[2/4] building 4-channel LLM pool ...")
pool = build_channel_pool()
print(f"  OK {len(pool)} channels")

print("[3/4] building 22 PlayerAgents ...")
bus = TeamMessageBus()
agents, slot_to_label = _build_agents(pool, env, bus, 11, TEAM_BLUE_11V11, TEAM_RED_11V11)
print(f"  OK built {len(agents)} agents")
for a in [agents[0], agents[5], agents[10], agents[11], agents[15], agents[21]]:
    print(f"    slot={a.slot:2d} pid={a.player_id} side={a.team_side} role={a.role} model={a.llm_player.llm_client.model}")

print("[4/4] verifying channel distribution ...")
chan_counts = Counter(id(a.llm_player.llm_client) for a in agents)
print(f"  agents per channel: {sorted(chan_counts.values(), reverse=True)}  (expect 6,6,5,5)")

# Try ONE motor step (no LLM, just verify action assembly works)
print("[5] one env.step_actions with all-IDLE (sanity check)...")
env.step_actions([0] * 22)
print(f"  OK env.tick={env.tick}")

env.close()
print("[DONE] all wiring OK")
