"""Verify multi-agent control API: obs shape, step shape, slot-to-player mapping."""
from gfootball.env import create_environment


def main():
    env = create_environment(
        env_name="academy_run_to_score_with_keeper",
        representation="raw",
        render=False,
        number_of_left_players_agent_controls=2,
    )
    obs = env.reset()
    print("=== reset() ===")
    print(f"  type: {type(obs).__name__}, len={len(obs)}")
    for i, o in enumerate(obs):
        active = o["active"]
        role = int(o["left_team_roles"][active])
        pos = o["left_team"][active]
        print(f"  slot {i}: active=#{active}  role={role}  pos=({pos[0]:+.2f}, {pos[1]:+.2f})")

    print()
    print("=== step([IDLE_for_slot0, RIGHT_for_slot1]) ===")
    result = env.step([0, 5])  # GK=IDLE, slot1=RIGHT
    if len(result) == 5:
        obs2, r, _, _, _ = result
    else:
        obs2, r, _, _ = result
    for i, o in enumerate(obs2):
        active = o["active"]
        pos = o["left_team"][active]
        print(f"  slot {i}: active=#{active}  pos=({pos[0]:+.2f}, {pos[1]:+.2f})")
    print(f"  reward: {r}")

    print()
    print("=== step 20 more times with slot1=RIGHT to verify slot1 player moves ===")
    last_pos = None
    for t in range(20):
        result = env.step([0, 5])
        if len(result) == 5:
            obs3, _, term, trunc, _ = result
            done = term or trunc
        else:
            obs3, _, done, _ = result
        if done:
            print(f"  episode ended at step {t+1}")
            break
        active = obs3[1]["active"]
        last_pos = obs3[1]["left_team"][active]
    if last_pos is not None:
        print(f"  slot 1 final position: ({last_pos[0]:+.2f}, {last_pos[1]:+.2f}) "
              "(should have moved RIGHT from initial)")
    env.close()


if __name__ == "__main__":
    main()
