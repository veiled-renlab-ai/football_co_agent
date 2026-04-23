"""Smoke test for Phase 5c-B: Call skill -> TeamMessageBus routing.

Verifies that PlayerAgent.install_skill(Call(...)) with tick + raw_obs
posts a Message onto the shared bus, and that another agent on the same
team channel can read it back via bus.read_for(...).

Also checks the no-bus and no-tick fallback paths silently no-op
(backward compatibility).

Run via:

    wsl -d Ubuntu-22.04 -- bash -lc 'source ~/football-env/bin/activate && \\
        cd /mnt/c/Users/dfgfd/Desktop/football && \\
        python3 -m scripts.smoke_call_routing'
"""
from __future__ import annotations

from football_agents.llm_client import LLMClient
from football_agents.message_bus import TeamMessageBus
from football_agents.perception import Vec2
from football_agents.player_agent import PlayerAgent
from football_agents.prompts import PlayerPersona
from football_agents.skills import Call, MoveTo


def _make_client() -> LLMClient:
    """Dummy LLMClient — never actually called in this smoke test
    (we only exercise install_skill, which is pure main-thread)."""
    return LLMClient(
        api_key="dummy-not-used",
        base_url="http://localhost:9999",
        model="dummy",
    )


def _make_persona(idx: int) -> PlayerPersona:
    return PlayerPersona(
        name=f"测试球员{idx}",
        age=20 + idx,
        nationality="中国",
        team="蓝队",
        jersey_number=10 + idx,
        position="中场",
        play_style=f"风格{idx}",
        background=f"背景{idx}",
    )


def main() -> None:
    client = _make_client()
    bus = TeamMessageBus()

    agents = [
        PlayerAgent(
            slot=i,
            player_id=i,
            team_side="left",
            role="CM",
            persona=_make_persona(i),
            llm_client=client,
            bus=bus,
        )
        for i in range(2)
    ]

    # Sanity: bus is wired through to perception too.
    assert agents[0].bus is bus, "agent.bus not stored"
    assert agents[0].perception._bus is bus, "perception._bus not wired"
    assert agents[1].perception._bus is bus, "perception._bus not wired (agent[1])"
    print("OK: PlayerAgent stores bus AND passes it into EgocentricFilter")

    # Backward compat: PlayerAgent without bus must still construct.
    bareagent = PlayerAgent(
        slot=0, player_id=0, team_side="left", role="CM",
        persona=_make_persona(99), llm_client=client,
    )
    assert bareagent.bus is None, "default bus should be None"
    assert bareagent.perception._bus is None, "default perception bus should be None"
    print("OK: PlayerAgent(bus=None) backward compat (single-agent mode)")

    # Build a fake raw_obs with agent 0 at (0.0, 0.0).
    fake_raw_obs = {
        "left_team": [[0.0, 0.0], [0.5, 0.1]],
        "right_team": [[-0.5, 0.0], [-0.6, 0.1]],
    }

    # Agent 0 calls Call("传给我", audience="team") at tick=100.
    call = Call(message="传给我", audience="team")
    agents[0].install_skill(call, tick=100, raw_obs=fake_raw_obs)

    # Agent 1 reads its team channel at tick=110 (10 ticks later).
    heard = bus.read_for(
        team="left",
        listener_id=1,
        listener_position=Vec2(0.5, 0.0),
        current_tick=110,
    )
    assert len(heard) == 1, f"expected 1 message, got {len(heard)}: {heard}"
    h = heard[0]
    assert h.sender_player_id == 0, f"sender_player_id={h.sender_player_id}"
    assert h.sender_jersey == 10, f"sender_jersey={h.sender_jersey}"
    assert h.message == "传给我", f"message={h.message!r}"
    assert h.audience == "team", f"audience={h.audience!r}"
    assert h.age_ticks == 10, f"age_ticks={h.age_ticks}"
    assert h.sender_position.x == 0.0 and h.sender_position.y == 0.0, (
        f"sender_position={h.sender_position}"
    )
    print(f"OK: bus.read_for got the message back: {h}")

    # Agent 0 should NOT hear its own message (sender filter).
    own = bus.read_for(
        team="left",
        listener_id=0,
        listener_position=Vec2(0.0, 0.0),
        current_tick=110,
    )
    assert len(own) == 0, f"sender should not hear own call, got {own}"
    print("OK: sender does not hear own Call (self-filter works)")

    # Right-team channel must be empty (team partitioning).
    right = bus.read_for(
        team="right",
        listener_id=0,
        listener_position=Vec2(0.0, 0.0),
        current_tick=110,
    )
    assert len(right) == 0, f"right team should be empty, got {right}"
    print("OK: right-team channel is empty (team partitioning works)")

    # Backward-compat: install_skill on Call with NO tick/raw_obs must NOT crash.
    # (Logs a warning; the message is dropped.)
    bare_bus = TeamMessageBus()
    bagent = PlayerAgent(
        slot=0, player_id=0, team_side="left", role="CM",
        persona=_make_persona(50), llm_client=client, bus=bare_bus,
    )
    bagent.install_skill(Call(message="should-be-dropped", audience="team"))
    leftover = bare_bus.read_for(
        team="left", listener_id=1,
        listener_position=Vec2(0.0, 0.0), current_tick=10,
    )
    assert len(leftover) == 0, (
        f"Call without tick/raw_obs must NOT post; got {leftover}"
    )
    print("OK: install_skill(Call) without tick/raw_obs silently no-ops")

    # Sanity: a non-Call skill still installs a controller (Call branch shouldn't
    # break the existing flow).
    agents[0].install_skill(MoveTo(target_x=0.5, target_y=0.0, urgency="jog"))
    assert agents[0].current_controller is not None, "MoveTo must install controller"
    assert agents[0].last_skill_name == "MoveTo"
    print("OK: non-Call skills still install controllers normally")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
