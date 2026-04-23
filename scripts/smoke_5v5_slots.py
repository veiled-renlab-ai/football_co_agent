"""Verify gfootball 5_vs_5 multi-agent slot mapping.

In 5_vs_5: GK is controllable=False, the 4 outfield players (RM/CF/LB/CB)
are agent-controllable. With n_controlled_left=4, gfootball returns 4
per-slot raw observations. We need to know:
  - What `active` field each slot reports → tells us which player_id that
    slot drives in the team array
  - The role_ids in left_team_roles for cross-checking

Run via:
    wsl -d Ubuntu-22.04 -- bash -lc 'source ~/football-env/bin/activate && \\
        cd /mnt/c/Users/dfgfd/Desktop/football && python3 -m scripts.smoke_5v5_slots'
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
        number_of_right_players_agent_controls=0,  # right team = scripted bots
    )
    raw = env.reset()
    print(f"raw type: {type(raw).__name__}")
    print(f"len(raw) (one per controlled slot): {len(raw)}")
    print()

    # Each raw[i] is a dict view from slot i's perspective. They share the
    # same world state but may have different `active` fields.
    print("Per-slot 'active' field (= player_id in left_team that slot drives):")
    for i, slot_obs in enumerate(raw):
        active = int(slot_obs["active"])
        roles = slot_obs["left_team_roles"]
        role_id = int(roles[active])
        role_name = ROLE_NAMES.get(role_id, "?")
        print(f"  slot {i} → player_id {active} → role_id {role_id} ({role_name})")
    print()

    # Verify all slots see the same world state (we'll use raw[0] as canonical)
    same_team = (
        raw[0]["left_team"].tolist()
        == raw[1]["left_team"].tolist()
        == raw[2]["left_team"].tolist()
        == raw[3]["left_team"].tolist()
    )
    print(f"All slots agree on left_team positions: {same_team}")

    same_ball = (
        raw[0]["ball"].tolist()
        == raw[1]["ball"].tolist()
        == raw[2]["ball"].tolist()
        == raw[3]["ball"].tolist()
    )
    print(f"All slots agree on ball position: {same_ball}")

    print()
    print("Full left_team roles:")
    roles = raw[0]["left_team_roles"]
    for pid, r in enumerate(roles):
        rname = ROLE_NAMES.get(int(r), "?")
        print(f"  player_id {pid}: role_id {int(r)} ({rname})")

    print()
    print("Full right_team roles:")
    roles = raw[0]["right_team_roles"]
    for pid, r in enumerate(roles):
        rname = ROLE_NAMES.get(int(r), "?")
        print(f"  player_id {pid}: role_id {int(r)} ({rname})")

    env.close()
    print("\n--- 5v5 slot mapping verified ---")


if __name__ == "__main__":
    main()
