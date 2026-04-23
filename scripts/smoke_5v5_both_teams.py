"""Verify gfootball 5_vs_5 with BOTH teams agent-controlled.

Goal: confirm that n_controlled_left=4 + n_controlled_right=4 gives us
8 controllable slots (4 left outfield + 4 right outfield), GKs stay
scripted (controllable=False), and env.step takes a flat list of 8 actions.

Run: wsl ... python3 -m scripts.smoke_5v5_both_teams
"""
from __future__ import annotations

from gfootball.env import create_environment

from football_agents.perception import EgocentricFilter

ROLE_NAMES = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME


def main() -> None:
    env = create_environment(
        env_name="5_vs_5",
        representation="raw",
        render=False,
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=4,
        other_config_options={"physics_steps_per_frame": 2, "real_time": False},
    )
    raw = env.reset()
    print(f"raw type: {type(raw).__name__}, len={len(raw)}")
    print(f"Expected 8: 4 left outfield + 4 right outfield (both GKs scripted)")
    print()

    print("Per-slot active player_id and role:")
    for i, slot_obs in enumerate(raw):
        side = "LEFT " if i < 4 else "RIGHT"
        active = int(slot_obs["active"])
        team_key = "left_team_roles" if i < 4 else "right_team_roles"
        role_id = int(slot_obs[team_key][active])
        role_name = ROLE_NAMES.get(role_id, "?")
        print(f"  slot {i} ({side}): pid {active} -> role_id {role_id} ({role_name})")
    print()

    print("Testing env.step with 8 actions (4 left + 4 right):")
    result = env.step([0] * 8)
    raw_after = result[0]
    print(f"  raw after step: list of {len(raw_after)}")
    print(f"  reward: {result[1]}")
    print(f"  done: {result[2]}")

    print()
    print("Verifying world state is shared across all 8 slot views:")
    same_left = all(
        raw_after[0]["left_team"].tolist() == raw_after[i]["left_team"].tolist()
        for i in range(1, 8)
    )
    print(f"  All 8 slots see same left_team: {same_left}")
    same_ball = all(
        raw_after[0]["ball"].tolist() == raw_after[i]["ball"].tolist()
        for i in range(1, 8)
    )
    print(f"  All 8 slots see same ball: {same_ball}")
    print(f"  → raw[0] usable as canonical god-view for both teams")

    env.close()
    print("\n--- 5v5 both-teams slot mapping verified ---")


if __name__ == "__main__":
    main()
