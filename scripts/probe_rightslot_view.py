"""Does gfootball provide a pre-rotated observation to right-team slots?
If raw[5] (right slot 0) has positions flipped relative to raw[0] (left slot 0),
then our perception's additional mirror is a DOUBLE mirror.
"""
from gfootball.env import create_environment

env = create_environment(
    env_name="llm_5v5_full",
    representation="raw",
    render=False,
    number_of_left_players_agent_controls=5,
    number_of_right_players_agent_controls=5,
    other_config_options={"physics_steps_per_frame": 2, "real_time": False},
)
raw = env.reset()

for slot in (0, 5):
    side = "LEFT" if slot < 5 else "RIGHT"
    print(f"=== slot {slot} ({side}) view ===")
    print(f"  left_team positions:  {raw[slot]['left_team'].tolist()}")
    print(f"  right_team positions: {raw[slot]['right_team'].tolist()}")
    print(f"  ball:                 {raw[slot]['ball'].tolist()}")
    print(f"  active: {raw[slot]['active']}")
    print()

# Check if step returns different raw per slot too
env.step([0] * 10)
raw = env._env.unwrapped.observation() if hasattr(env._env, "unwrapped") else None
env.close()
