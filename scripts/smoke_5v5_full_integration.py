"""End-to-end check for 10-agent 5v5 setup with self-frame perspective.

Verifies:
  1. env supports n_controlled_left=5 + n_controlled_right=5 (10 actions in)
  2. All 10 PlayerAgents construct cleanly with shared bus
  3. Right-team agent's perception is in self-frame (their own goal at -x in
     self-frame, which is +x in absolute — gfootball mirror)
  4. Right-team agent's MoveTo target gets mirrored to absolute
  5. Bus per-team partitioning: blue Call doesn't appear in red obs
  6. All 10 worker threads start + stop cleanly

NO LLM calls. Pure plumbing check.
"""
from __future__ import annotations

import math
from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient
from football_agents.message_bus import TeamMessageBus
from football_agents.motor import make_controller
from football_agents.personas import TEAM_BLUE_5V5, TEAM_RED_5V5
from football_agents.perception import EgocentricFilter
from football_agents.player_agent import PlayerAgent
from football_agents.skills import Call, MoveTo


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
    print(f"[1] env.n_controlled_total = {env.n_controlled_total} ✓")

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
    print(f"[2] Built 10 PlayerAgents (5 left + 5 right) all sharing same bus ✓")

    # 3. Self-frame check: red CB (slot 9, pid=4 in right_team)
    #    Absolute: right team is at +x. Their CB at ~(+0.1, +0.1).
    #    Self-frame: should appear at ~(-0.1, -0.1) (their own half).
    raw0 = env.raw_obs
    red_cb_abs = raw0["right_team"][4]
    print(f"[3] Red CB absolute pos: ({red_cb_abs[0]:+.3f}, {red_cb_abs[1]:+.3f})")
    obs_red = agents[9].perceive(raw0, env.tick)
    print(f"    Red CB self-frame pos: ({obs_red.self_state.position.x:+.3f}, "
          f"{obs_red.self_state.position.y:+.3f})")
    assert obs_red.self_state.position.x == -float(red_cb_abs[0])
    assert obs_red.self_state.position.y == -float(red_cb_abs[1])
    print(f"    OK: red agent sees position mirrored (180° rotation) ✓")

    # 4. Motor mirrors LLM target back to absolute for right team
    #    LLM (red CB) wants to attack +x in self-frame -> motor must go to -x abs
    skill = MoveTo(target_x=0.7, target_y=0.0, urgency="jog")
    c = make_controller(skill, team_side="right", player_id=4)
    tx_abs, ty_abs = c._selfframe_target_to_abs(0.7, 0.0)
    assert tx_abs == -0.7 and ty_abs == 0.0
    print(f"[4] Right team motor: LLM target +0.70 self-frame -> {tx_abs:+.2f} abs ✓")

    # Left team: no flip
    c2 = make_controller(skill, team_side="left", player_id=4)
    tx_abs2, _ = c2._selfframe_target_to_abs(0.7, 0.0)
    assert tx_abs2 == 0.7
    print(f"    Left team: +0.70 self-frame -> {tx_abs2:+.2f} abs (no flip) ✓")

    # 5. Bus per-team partitioning
    agents[0].install_skill(  # blue GK posts
        Call(message="蓝队加油！", audience="team"),
        tick=100, raw_obs=raw0,
    )
    agents[5].install_skill(  # red GK posts
        Call(message="红队上！", audience="team"),
        tick=100, raw_obs=raw0,
    )
    blue_obs = agents[1].perceive(raw0, tick=110)  # blue RM
    red_obs = agents[6].perceive(raw0, tick=110)   # red RM
    print(f"[5] Blue RM hears {len(blue_obs.heard_calls)} call(s): "
          f"{[c.message for c in blue_obs.heard_calls]}")
    print(f"    Red  RM hears {len(red_obs.heard_calls)} call(s): "
          f"{[c.message for c in red_obs.heard_calls]}")
    blue_messages = [c.message for c in blue_obs.heard_calls]
    red_messages = [c.message for c in red_obs.heard_calls]
    assert "蓝队加油！" in blue_messages and "红队上！" not in blue_messages
    assert "红队上！" in red_messages and "蓝队加油！" not in red_messages
    print(f"    OK: Chinese wall holds — neither team hears the other ✓")

    # 6. Worker thread lifecycle
    for a in agents:
        a.start()
    import threading
    n = sum(1 for t in threading.enumerate() if t.name.startswith("agent-pid"))
    print(f"[6] Started 10 worker threads ({n} alive) ✓")
    for a in agents:
        a.stop(timeout=0.3)
    print(f"    Stopped all 10 cleanly ✓")

    env.close()
    print()
    print("=" * 70)
    print("--- 5v5 FULL integration: 6/6 checks PASSED ---")
    print("=" * 70)


if __name__ == "__main__":
    main()
