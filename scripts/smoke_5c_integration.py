"""End-to-end Phase 5c integration smoke test.

Verifies: bus + PlayerAgent + EgocentricFilter + render_observation +
TeamProfile system prompt injection — all wired together as the 5v5
demo will use them.

DOES NOT call gfootball or LLM — pure Python integration check using
fake observations.

Run: wsl ... python3 -m scripts.smoke_5c_integration
"""
from __future__ import annotations

from football_agents.llm_client import LLMClient
from football_agents.message_bus import TeamMessageBus
from football_agents.personas import TEAM_BLUE_5V5
from football_agents.player_agent import PlayerAgent
from football_agents.prompts import build_system_prompt, render_observation
from football_agents.skills import Call


def _fake_god_view():
    """Minimal raw_obs mimicking gfootball 5_vs_5 god-view."""
    return {
        "active": 1,
        "left_team": [
            [-1.0, 0.0],   # GK pid=0
            [0.0, 0.02],   # RM pid=1 王浩
            [0.0, -0.02],  # CF pid=2 陈宇
            [-0.1, -0.1],  # LB pid=3 周俊
            [-0.1, 0.1],   # CB pid=4 高磊
        ],
        "left_team_direction": [[0.0, 0.0]] * 5,
        "left_team_roles": [0, 7, 9, 2, 1],
        "left_team_tired_factor": [0.0] * 5,
        "right_team": [[1.0, 0.0]] + [[0.5, i * 0.1] for i in range(4)],
        "right_team_direction": [[0.0, 0.0]] * 5,
        "right_team_roles": [0, 7, 9, 2, 1],
        "right_team_tired_factor": [0.0] * 5,
        "ball": [0.0, 0.0],
        "ball_direction": [0.0, 0.0],
        "ball_owned_team": -1,
        "ball_owned_player": -1,
        "score": [0, 0],
    }


def main() -> None:
    print("=" * 70)
    print("Phase 5c End-to-End Integration Smoke Test")
    print("=" * 70)

    # 1. Build infrastructure
    bus = TeamMessageBus()
    client = LLMClient.from_env()  # uses .env for real API config (no actual call)
    print(f"\n[1] Bus + Client created. Lifetime={TeamMessageBus.MESSAGE_LIFETIME_TICKS} ticks")

    # 2. Build 4 PlayerAgents (left team, with bus + team-profile personas)
    agents = []
    for slot in range(4):
        agents.append(
            PlayerAgent(
                slot=slot, player_id=slot + 1, team_side="left", role="CM",
                persona=TEAM_BLUE_5V5[slot], llm_client=client, bus=bus,
            )
        )
    print(f"[2] Built 4 PlayerAgents on left team (王浩/陈宇/周俊/高磊)")
    print(f"    All share the same bus: {all(a.bus is bus for a in agents)}")
    print(f"    All have team_profile: {all(a.persona.team_profile is not None for a in agents)}")

    # 3. Verify system prompt has team-style section
    sp = build_system_prompt(agents[0].persona)
    assert "我们球队（蓝队）的风格" in sp, "team section missing"
    assert "传控渗透" in sp, "team character missing"
    print(f"[3] System prompt for 王浩 includes 蓝队 team-style section ✓")

    # 4. Have agent[0] (王浩 #11) call out
    raw_obs = _fake_god_view()
    agents[0].install_skill(
        Call(message="传给我，禁区前沿！", audience="team"),
        tick=100, raw_obs=raw_obs,
    )
    print(f"[4] 王浩 (pid=1) posted: Call('传给我，禁区前沿！')")

    # 5. Have agent[1] (陈宇 #9) perceive — should hear 王浩's call
    obs = agents[1].perceive(raw_obs, tick=110)
    assert len(obs.heard_calls) == 1, f"expected 1 heard call, got {len(obs.heard_calls)}"
    heard = obs.heard_calls[0]
    assert heard.sender_jersey == 11, f"sender jersey wrong: {heard.sender_jersey}"
    assert heard.message == "传给我，禁区前沿！", f"message wrong: {heard.message}"
    print(f"[5] 陈宇 hears: {heard.sender_jersey} 号 '{heard.message}' (age={heard.age_ticks} ticks)")

    # 6. Render that observation as Chinese prompt — should include the call
    rendered = render_observation(obs, agents[1].persona)
    assert "你听到队友的喊话" in rendered, "heard_calls section missing in render"
    assert "传给我，禁区前沿" in rendered, "call message missing in render"
    print(f"[6] render_observation for 陈宇 includes the heard call ✓")

    # Show the relevant lines
    print("\n--- Excerpt of 陈宇's observation render ---")
    for line in rendered.split("\n"):
        if "听到" in line or "号（在" in line or "陈宇" in line:
            print(f"    {line}")

    # 7. 王浩 himself shouldn't hear his own call
    obs_self = agents[0].perceive(raw_obs, tick=110)
    assert len(obs_self.heard_calls) == 0, "agent should not hear own call"
    print(f"\n[7] 王浩 doesn't hear own call (self-filter works) ✓")

    # 8. After 30+ ticks, message expires
    obs_late = agents[1].perceive(raw_obs, tick=200)
    assert len(obs_late.heard_calls) == 0, "stale messages should be filtered"
    print(f"[8] After 100 ticks, 陈宇 no longer hears stale call (TTL works) ✓")

    # 9. Cleanup
    for a in agents:
        a.stop(timeout=0.1)
    print(f"\n--- Phase 5c integration: 8/8 checks PASSED ---")


if __name__ == "__main__":
    main()
