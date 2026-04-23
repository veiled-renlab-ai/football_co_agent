"""Quick smoke test for PlayerAgent isolation. Run via:

    wsl -d Ubuntu-22.04 -- bash -lc 'source ~/football-env/bin/activate && \\
        cd /mnt/c/Users/dfgfd/Desktop/football && \\
        python3 -m scripts.smoke_player_agent'
"""
from __future__ import annotations

import threading

from football_agents.llm_client import LLMClient
from football_agents.player_agent import PlayerAgent
from football_agents.prompts import PlayerPersona
from football_agents.skills import MoveTo


def main() -> None:
    client = LLMClient.from_env()  # uses .env (real Volcengine config)

    personas = [
        PlayerPersona(
            name=f"测试球员{i}",
            age=20 + i,
            nationality="中国",
            team="蓝队",
            jersey_number=i,
            position="中场",
            play_style=f"风格{i}",
            background=f"背景{i}",
        )
        for i in range(3)
    ]

    agents = [
        PlayerAgent(
            slot=i,
            player_id=i,
            team_side="left",
            role="CM",
            persona=personas[i],
            llm_client=client,
        )
        for i in range(3)
    ]

    # 1. Per-agent state isolation — every owned object must be unique
    for i in range(3):
        for j in range(i + 1, 3):
            assert agents[i].llm_player is not agents[j].llm_player, f"brain leak {i}↔{j}"
            assert agents[i].perception is not agents[j].perception, f"perception leak {i}↔{j}"
            assert agents[i].obs_queue is not agents[j].obs_queue, f"obs_queue leak {i}↔{j}"
            assert agents[i].skill_queue is not agents[j].skill_queue, f"skill_queue leak {i}↔{j}"
            assert agents[i].llm_player._recent_turns is not agents[j].llm_player._recent_turns, f"memory leak {i}↔{j}"
            assert agents[i].llm_player._compressed_summaries is not agents[j].llm_player._compressed_summaries, f"compressed-mem leak {i}↔{j}"
            assert agents[i].perception._tracked_entity_ids is not agents[j].perception._tracked_entity_ids, f"track-set leak {i}↔{j}"
    print("OK: 3 PlayerAgents are fully isolated (brain / perception / queues / memory / tracked set)")

    # 2. install_skill routes player_id correctly into controller
    agents[1].install_skill(MoveTo(target_x=0.5, target_y=0.0, urgency="jog"))
    assert agents[1].current_controller.player_id == 1
    assert agents[0].current_controller is None
    assert agents[2].current_controller is None
    print("OK: install_skill on agent[1] only affects agent[1]; controller has correct player_id")

    # 3. LLMClient is shared (HTTP connection pool reuse)
    assert agents[0].llm_player.llm_client is agents[1].llm_player.llm_client
    assert agents[1].llm_player.llm_client is agents[2].llm_player.llm_client
    print("OK: LLMClient instance is shared across all agents")

    # 4. Personas are per-agent
    assert agents[0].persona.name == "测试球员0"
    assert agents[1].persona.name == "测试球员1"
    assert agents[2].persona.name == "测试球员2"
    print("OK: personas are per-agent (no aliasing)")

    # 5. Worker thread lifecycle
    for a in agents:
        a.start()
    n_threads = sum(1 for t in threading.enumerate() if t.name.startswith("agent-pid"))
    assert n_threads == 3, f"expected 3 worker threads, got {n_threads}"
    print(f"OK: started 3 dedicated worker threads ({n_threads} confirmed alive)")

    # Per-agent track set isolation under concurrent threads
    agents[0].perception.track_entity(77)
    assert 77 in agents[0].perception._tracked_entity_ids
    assert 77 not in agents[1].perception._tracked_entity_ids
    assert 77 not in agents[2].perception._tracked_entity_ids
    print("OK: track_entity(77) on agent[0] does not leak to agents[1,2]")

    for a in agents:
        a.stop(timeout=0.5)
    print("OK: all 3 worker threads stopped cleanly")

    print("--- PlayerAgent isolation: 6/6 checks PASSED ---")


if __name__ == "__main__":
    main()
