"""Probe: what initial positions and directions does gfootball give
team_b (right) players in llm_5v5_full?"""
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
print(f"raw: list of {len(raw)}")
print()

slot0 = raw[0]
print("LEFT TEAM god-view (slot 0 perspective):")
for i in range(5):
    pos = slot0["left_team"][i]
    direction = slot0["left_team_direction"][i]
    role = int(slot0["left_team_roles"][i])
    print(f"  pid {i} role_id={role}: pos=({pos[0]:+.3f}, {pos[1]:+.3f})  "
          f"dir=({direction[0]:+.4f}, {direction[1]:+.4f})")

print()
print("RIGHT TEAM god-view (slot 0 perspective):")
for i in range(5):
    pos = slot0["right_team"][i]
    direction = slot0["right_team_direction"][i]
    role = int(slot0["right_team_roles"][i])
    print(f"  pid {i} role_id={role}: pos=({pos[0]:+.3f}, {pos[1]:+.3f})  "
          f"dir=({direction[0]:+.4f}, {direction[1]:+.4f})")

print()
print("Per-slot pid/active mapping:")
for s in range(10):
    active = int(raw[s]["active"])
    side = "LEFT " if s < 5 else "RIGHT"
    print(f"  slot {s} ({side}): active={active}")

print()
print("ball:", slot0["ball"], "ball_dir:", slot0["ball_direction"])

env.close()
