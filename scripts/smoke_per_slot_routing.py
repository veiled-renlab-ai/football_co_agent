"""Verify each agent now receives its own slot's raw_obs (not the shared
canonical raw[0]). Critical: right-team agents should see themselves at
-x in their slot view (because gfootball rotated them).
"""
from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient
from football_agents.message_bus import TeamMessageBus
from football_agents.multi_agent_runner import MultiAgentRunner
from football_agents.personas import TEAM_BLUE_5V5, TEAM_RED_5V5
from football_agents.player_agent import PlayerAgent

env = FootballEnvAdapter(
    scenario="llm_5v5_full",
    render=False,
    n_controlled_left=5,
    n_controlled_right=5,
)
env.reset()
client = LLMClient.from_env()
bus = TeamMessageBus()

# Build 10 agents
agents = []
for slot in range(5):
    agents.append(PlayerAgent(
        slot=slot, player_id=slot, team_side="left", role="GK" if slot==0 else "CM",
        persona=TEAM_BLUE_5V5[slot], llm_client=client, bus=bus,
    ))
for slot in range(5):
    agents.append(PlayerAgent(
        slot=5+slot, player_id=slot, team_side="right", role="GK" if slot==0 else "CM",
        persona=TEAM_RED_5V5[slot], llm_client=client, bus=bus,
    ))

# Test 1: env.raw_obs_for_slot returns different views for left vs right
left_view = env.raw_obs_for_slot(0)
right_view = env.raw_obs_for_slot(5)
left_self_pos = left_view["left_team"][0]   # left GK from left slot view
right_self_pos = right_view["left_team"][0] # right GK from right slot view (gfootball rotated)
print(f"Left GK position in slot 0 view:  {left_self_pos}")
print(f"Right GK position in slot 5 view: {right_self_pos}")
# Both should be near (-1, 0) because each slot's "self" is at -x in its view
assert left_self_pos[0] < -0.5, "left GK should see themselves at -x"
assert right_self_pos[0] < -0.5, "right GK should see themselves at -x in slot view"
print("OK: both GKs are at -x in their respective slot views (gfootball rotated red)")

# Test 2: agent.perceive uses slot's view (not raw[0])
obs_left_gk = agents[0].perceive(env.raw_obs_for_slot(0), 0)
obs_right_gk = agents[5].perceive(env.raw_obs_for_slot(5), 0)
print(f"Left GK perceived position:  {obs_left_gk.self_state.position}")
print(f"Right GK perceived position: {obs_right_gk.self_state.position}")
# Both should be at -x (slot-view native, no extra mirror)
assert obs_left_gk.self_state.position.x < -0.5
assert obs_right_gk.self_state.position.x < -0.5
print("OK: per-slot perception works without extra mirror")

env.close()
