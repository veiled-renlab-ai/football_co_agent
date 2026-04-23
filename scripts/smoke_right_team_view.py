"""Probe: does gfootball mirror right-team's per-slot view, or is it absolute?

If MIRRORED: raw[5]['left_team'] holds the right team players (their team
              from their POV), and they attack +x like left team does.
If ABSOLUTE: raw[5]['left_team'] holds the actual left team, and right
             team players still attack -x.
"""
from __future__ import annotations

from gfootball.env import create_environment


def main():
    env = create_environment(
        env_name="llm_5v5_full",
        representation="raw",
        render=False,
        number_of_left_players_agent_controls=5,
        number_of_right_players_agent_controls=5,
        other_config_options={"physics_steps_per_frame": 2, "real_time": False},
    )
    raw = env.reset()
    print(f"Total slots: {len(raw)}")
    print()

    print("--- raw[0] (LEFT slot 0, GK pid=0) ---")
    print(f"  left_team[0] (own GK):  {raw[0]['left_team'][0]}")
    print(f"  right_team[0] (opp GK): {raw[0]['right_team'][0]}")
    print(f"  active: {raw[0]['active']}")
    print()

    print("--- raw[5] (RIGHT slot 0, GK pid=0) ---")
    print(f"  left_team[0]:  {raw[5]['left_team'][0]}")
    print(f"  right_team[0]: {raw[5]['right_team'][0]}")
    print(f"  active: {raw[5]['active']}")
    print()

    # Decision check: if raw[5]['left_team'][0] == raw[0]['left_team'][0]:
    #   → ABSOLUTE coords (no mirroring). Right team needs special handling.
    # If raw[5]['left_team'][0] == raw[0]['right_team'][0] (mirrored):
    #   → gfootball already mirrored. Right team can use raw[5] directly.
    same_left = list(raw[0]['left_team'][0]) == list(raw[5]['left_team'][0])
    print(f"raw[0]['left_team'][0] == raw[5]['left_team'][0]?  {same_left}")
    if same_left:
        print("  → ABSOLUTE coordinates: right team needs perspective flip in our code")
    else:
        print("  → MIRRORED by gfootball: right team can use raw[5+i] directly,")
        print("    their 'left_team' key holds their own (right team) players")

    print()
    print("--- ball ---")
    print(f"  raw[0]['ball']: {raw[0]['ball']}")
    print(f"  raw[5]['ball']: {raw[5]['ball']}")

    env.close()


if __name__ == "__main__":
    main()
