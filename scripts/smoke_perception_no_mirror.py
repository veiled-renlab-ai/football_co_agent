"""After removing self-frame mirror, both teams should perceive THEIR OWN
slot-view coords (which gfootball already rotated to be self-frame).

Key check: red team agent's perceived self position should be NEGATIVE x
in their slot-view (because gfootball rotated them from +x absolute to -x
self-frame). If perception now correctly READS the slot view without
applying its own mirror, the position is whatever gfootball provided.
"""
from gfootball.env import create_environment
from football_agents.perception import EgocentricFilter

env = create_environment(
    env_name="llm_5v5_full",
    representation="raw",
    render=False,
    number_of_left_players_agent_controls=5,
    number_of_right_players_agent_controls=5,
    other_config_options={"physics_steps_per_frame": 2, "real_time": False},
)
raw = env.reset()

# Build a filter for left GK (slot 0) and inspect
f_left = EgocentricFilter(player_id=0, team="team_a", role="GK")
obs_left = f_left.filter(raw[0], tick=0)
print(f"Left GK slot view position: {obs_left.self_state.position}")
print(f"  Expected: x near -1 (own goal at -x)")

# Build a filter for right GK (slot 5) — but we now use slot 5's raw view!
f_right = EgocentricFilter(player_id=0, team="team_b", role="GK")
obs_right = f_right.filter(raw[5], tick=0)  # raw[5], NOT raw[0]
print(f"Right GK slot view position: {obs_right.self_state.position}")
print(f"  Expected: x near -1 (own goal at -x in slot view)")

# Critical: both should see themselves at NEGATIVE x in self-frame
assert obs_left.self_state.position.x < -0.5, "Left GK should be at -x"
assert obs_right.self_state.position.x < -0.5, "Right GK should be at -x in slot view"
print("OK: both GKs see their own goal at -x in their respective slot views")

env.close()
