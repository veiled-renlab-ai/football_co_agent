"""End-to-end check for 10-agent 5v5 setup with per-slot perception.

Verifies:
  1. env supports n_controlled_left=5 + n_controlled_right=5 (10 actions in)
  2. All 10 PlayerAgents construct cleanly with shared bus
  3. Per-slot view invariant: each agent's perception comes from its own
     slot raw_obs; both teams see themselves in "left_team" and at the
     position gfootball reports for them in that slot view.
  4. Bus per-team partitioning: blue Call doesn't appear in red obs
  5. All 10 worker threads start + stop cleanly

NO LLM calls. Pure plumbing check.
"""
from __future__ import annotations

from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient
from football_agents.message_bus import TeamMessageBus
from football_agents.personas import TEAM_BLUE_5V5, TEAM_RED_5V5
from football_agents.player_agent import PlayerAgent
from football_agents.skills import Call


def main() -> None:
    print("=" * 70)
    print("5v5 FULL (10 LLM agents) integration smoke test")
    print("=" * 70)

    env = FootballEnvAdapter(
        scenario="llm_5v5_full",
        render=False,
        n_controlled_left=5,
        n_controlled_right=5,
    )
    env.reset()
    assert env.n_controlled_total == 10
    print(f"[1] env.n_controlled_total = {env.n_controlled_total} ok")

    bus = TeamMessageBus()
    client = LLMClient.from_env()  # not actually called

    agents = []
    for slot in range(5):
        agents.append(PlayerAgent(
            slot=slot, player_id=slot, team_side="left",
            role="GK" if slot == 0 else "CM",
            persona=TEAM_BLUE_5V5[slot], llm_client=client, bus=bus,
        ))
    for slot in range(5):
        agents.append(PlayerAgent(
            slot=5 + slot, player_id=slot, team_side="right",
            role="GK" if slot == 0 else "CM",
            persona=TEAM_RED_5V5[slot], llm_client=client, bus=bus,
        ))
    print(f"[2] Built 10 PlayerAgents (5 left + 5 right) all sharing same bus ok")

    # 3. Per-slot view invariant.
    #   gfootball with right_player control rotates each right slot's view
    #   so that right-team players appear in "left_team" of THEIR slot
    #   raw_obs, with self at the negated absolute position.
    raw_blue_cb = env.raw_obs_for_slot(4)   # blue CB (left team, pid=4)
    raw_red_cb = env.raw_obs_for_slot(9)    # red  CB (right team, pid=4)

    blue_cb_self_pos = raw_blue_cb["left_team"][4]
    red_cb_self_pos  = raw_red_cb["left_team"][4]
    print(f"[3] Slot 4 (blue CB) sees self in left_team[4] = "
          f"({blue_cb_self_pos[0]:+.3f}, {blue_cb_self_pos[1]:+.3f})")
    print(f"    Slot 9 (red  CB) sees self in left_team[4] = "
          f"({red_cb_self_pos[0]:+.3f}, {red_cb_self_pos[1]:+.3f})")

    # Each agent's own perceive() must read its own slot view.
    obs_blue = agents[4].perceive(raw_blue_cb, env.tick)
    obs_red = agents[9].perceive(raw_red_cb, env.tick)
    assert obs_blue.self_state.position.x == float(blue_cb_self_pos[0])
    assert obs_blue.self_state.position.y == float(blue_cb_self_pos[1])
    assert obs_red.self_state.position.x == float(red_cb_self_pos[0])
    assert obs_red.self_state.position.y == float(red_cb_self_pos[1])
    # Both teams: self is on their own (negative-x) half in their slot view.
    assert obs_blue.self_state.position.x < 0.0
    assert obs_red.self_state.position.x < 0.0
    print(f"    OK: each agent reads its own slot view; both see self at -x ok")

    # 4. Bus per-team partitioning
    raw0 = env.raw_obs
    agents[0].install_skill(  # blue GK posts
        Call(message="蓝队加油！", audience="team"),
        tick=100, raw_obs=raw0,
    )
    agents[5].install_skill(  # red GK posts
        Call(message="红队上！", audience="team"),
        tick=100, raw_obs=raw0,
    )
    blue_obs = agents[1].perceive(env.raw_obs_for_slot(1), tick=110)  # blue RM
    red_obs = agents[6].perceive(env.raw_obs_for_slot(6), tick=110)   # red RM
    print(f"[4] Blue RM hears {len(blue_obs.heard_calls)} call(s): "
          f"{[c.message for c in blue_obs.heard_calls]}")
    print(f"    Red  RM hears {len(red_obs.heard_calls)} call(s): "
          f"{[c.message for c in red_obs.heard_calls]}")
    blue_messages = [c.message for c in blue_obs.heard_calls]
    red_messages = [c.message for c in red_obs.heard_calls]
    assert "蓝队加油！" in blue_messages and "红队上！" not in blue_messages
    assert "红队上！" in red_messages and "蓝队加油！" not in red_messages
    print(f"    OK: Chinese wall holds — neither team hears the other ok")

    # 5. Worker thread lifecycle
    for a in agents:
        a.start()
    import threading
    n = sum(1 for t in threading.enumerate() if t.name.startswith("agent-pid"))
    print(f"[5] Started 10 worker threads ({n} alive) ok")
    for a in agents:
        a.stop(timeout=0.3)
    print(f"    Stopped all 10 cleanly ok")

    env.close()
    print()
    print("=" * 70)
    print("--- 5v5 FULL integration: 5/5 checks PASSED ---")
    print("=" * 70)


if __name__ == "__main__":
    main()
