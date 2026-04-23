"""Smoke test for TeamMessageBus (Phase 5c-A).

Exercises:
1. post() then read_for() round-trip
2. audience='team' broadcast vs audience='nearby' radius gating
3. Stale-message filtering (age >= MESSAGE_LIFETIME_TICKS)
4. Self-message filtering (sender_id == listener_id is dropped)

Run:  python -m scripts.smoke_message_bus
   or python scripts/smoke_message_bus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script (python scripts/smoke_message_bus.py) without
# needing PYTHONPATH set — add repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_agents.message_bus import HeardCall, Message, TeamMessageBus
from football_agents.perception import Vec2


def main() -> int:
    bus = TeamMessageBus()

    # ---- agent 1 (jersey 8) posts a team broadcast at tick 100 ------------
    bus.post("left", Message(
        sender_player_id=1,
        sender_jersey=8,
        sender_position=Vec2(0.0, 0.0),
        message="传给我",
        audience="team",
        tick_posted=100,
    ))
    print("[post] tick=100  agent_id=1 jersey=8  audience=team   '传给我'")

    # ---- agent 4 (jersey 11) posts a nearby call at tick 105 --------------
    bus.post("left", Message(
        sender_player_id=4,
        sender_jersey=11,
        sender_position=Vec2(0.5, 0.0),
        message="身后有人",
        audience="nearby",
        tick_posted=105,
    ))
    print("[post] tick=105  agent_id=4 jersey=11 audience=nearby pos=(0.5,0.0) '身后有人'")
    print()

    # ---- read for listener 2 at tick 110 ----------------------------------
    listener_pos = Vec2(0.6, 0.0)
    heard_at_110 = bus.read_for(
        team="left",
        listener_id=2,
        listener_position=listener_pos,
        current_tick=110,
    )
    print(f"[read] listener_id=2 pos=(0.6,0.0) tick=110  -> {len(heard_at_110)} call(s)")
    for h in heard_at_110:
        print(f"        from #{h.sender_jersey} (age={h.age_ticks} ticks): "
              f"'{h.message}' [{h.audience}]")

    # Acceptance #4: should hear BOTH (agent 1's team broadcast, agent 4's nearby).
    assert len(heard_at_110) == 2, f"expected 2 calls at tick=110, got {len(heard_at_110)}"
    msgs = {h.message for h in heard_at_110}
    assert "传给我" in msgs, "expected to hear '传给我' (team broadcast)"
    assert "身后有人" in msgs, "expected to hear '身后有人' (nearby, dist ~0.1 < 0.2)"
    # Spot-check ages
    ages = {h.message: h.age_ticks for h in heard_at_110}
    assert ages["传给我"] == 10, f"age for '传给我' should be 10, got {ages['传给我']}"
    assert ages["身后有人"] == 5, f"age for '身后有人' should be 5, got {ages['身后有人']}"
    print("        OK: heard both, ages correct (10 / 5)")
    print()

    # ---- read for listener 2 at tick 200 (both stale, > 30 ticks old) ----
    heard_at_200 = bus.read_for(
        team="left",
        listener_id=2,
        listener_position=listener_pos,
        current_tick=200,
    )
    print(f"[read] listener_id=2 pos=(0.6,0.0) tick=200  -> {len(heard_at_200)} call(s)")
    assert heard_at_200 == [], f"expected 0 stale calls at tick=200, got {len(heard_at_200)}"
    print("        OK: both messages stale (>= 30 ticks old), filtered out")
    print()

    # ---- read for listener 4 (sender of '身后有人') ------------------------
    # Should NOT hear own message; should still hear agent 1's team call
    # if within window — pick a tick close enough.
    heard_self_4 = bus.read_for(
        team="left",
        listener_id=4,
        listener_position=Vec2(0.5, 0.0),
        current_tick=110,
    )
    print(f"[read] listener_id=4 pos=(0.5,0.0) tick=110  -> {len(heard_self_4)} call(s)")
    for h in heard_self_4:
        print(f"        from #{h.sender_jersey} (age={h.age_ticks}): '{h.message}' [{h.audience}]")
    own_msgs = [h for h in heard_self_4 if h.sender_player_id == 4]
    assert own_msgs == [], f"agent 4 should not hear own message, got {own_msgs}"
    # Should still hear agent 1's team broadcast (audience='team', not stale)
    assert any(h.message == "传给我" for h in heard_self_4), \
        "agent 4 should still hear agent 1's team broadcast"
    print("        OK: did NOT hear own '身后有人'; still hears teammate's '传给我'")
    print()

    # ---- bonus: nearby gating from far away --------------------------------
    # Listener at (1.0, 0.0); agent 4's '身后有人' was posted from (0.5, 0.0).
    # Distance = 0.5 > NEARBY_RADIUS (0.20) → must NOT hear that nearby call.
    far_pos = Vec2(1.0, 0.0)
    heard_far = bus.read_for(
        team="left",
        listener_id=2,
        listener_position=far_pos,
        current_tick=110,
    )
    print(f"[read] listener_id=2 pos=(1.0,0.0) tick=110  -> {len(heard_far)} call(s)")
    for h in heard_far:
        print(f"        from #{h.sender_jersey} (age={h.age_ticks}): '{h.message}' [{h.audience}]")
    nearby_far = [h for h in heard_far if h.audience == "nearby"]
    assert nearby_far == [], \
        f"listener at dist=0.5 should NOT hear nearby call (radius=0.20), got {nearby_far}"
    assert any(h.message == "传给我" for h in heard_far), \
        "team broadcast must still reach a far listener"
    print("        OK: nearby call dropped at dist=0.5 (>0.20); team broadcast still heard")
    print()

    # ---- bonus: opposite-team channel isolation ---------------------------
    heard_right = bus.read_for(
        team="right",
        listener_id=2,
        listener_position=listener_pos,
        current_tick=110,
    )
    assert heard_right == [], "right channel should be empty (no posts)"
    print(f"[read] team='right' tick=110 -> {len(heard_right)} call(s)  OK: channels isolated")
    print()

    print("ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
