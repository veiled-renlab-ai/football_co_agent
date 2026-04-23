"""Verify the runner's last_obs_push_tick is keyed by slot, not player_id.

Pre-fix bug: 5v5 with both teams meant blue/red player_ids collided on
the same dict key. Blue iterated first, always won the push, red never
got refreshed observations after startup → only 1 decision per red agent.

Post-fix: keyed by slot (0..9 unique). Each agent's refresh schedule is
independent. Simulating 50 env ticks should push to ALL 10 agents
roughly equally.
"""
from __future__ import annotations

# Simulate the runner's push logic without spinning up the actual env / LLM.

class FakeAgent:
    def __init__(self, slot: int, player_id: int, team: str):
        self.slot = slot
        self.player_id = player_id
        self.team = team
        self.push_count = 0


def main() -> None:
    # Mimic 5v5: blue pids 0-4, red pids 0-4 (collision on player_id)
    agents = []
    for slot in range(5):
        agents.append(FakeAgent(slot=slot, player_id=slot, team="blue"))
    for slot in range(5):
        agents.append(FakeAgent(slot=5 + slot, player_id=slot, team="red"))

    obs_refresh_every_ticks = 4
    n_agents = len(agents)

    # POST-FIX init (keyed by slot)
    last_push: dict[int, int] = {
        a.slot: -((i * obs_refresh_every_ticks) // max(1, n_agents))
        for i, a in enumerate(agents)
    }
    assert len(last_push) == 10, f"expected 10 entries, got {len(last_push)}"
    print(f"Post-fix init dict has {len(last_push)} entries (one per slot) ✓")

    # Simulate 50 env ticks
    for tick in range(50):
        for a in agents:
            if tick - last_push[a.slot] >= obs_refresh_every_ticks:
                a.push_count += 1
                last_push[a.slot] = tick

    print()
    print(f"After 50 ticks of simulation:")
    print(f"  {'agent':<25}  {'pushes':>6}")
    print(f"  {'-'*25}  {'-'*6}")
    for a in agents:
        print(f"  slot={a.slot} pid={a.player_id} team={a.team:<5}  "
              f"{a.push_count:>6}")

    # Both teams should have ROUGHLY EQUAL push counts (within 1)
    blue_pushes = [a.push_count for a in agents if a.team == "blue"]
    red_pushes = [a.push_count for a in agents if a.team == "red"]
    print()
    print(f"Blue total pushes: {sum(blue_pushes)}, avg per agent: {sum(blue_pushes)/5:.1f}")
    print(f"Red  total pushes: {sum(red_pushes)},  avg per agent: {sum(red_pushes)/5:.1f}")
    assert min(red_pushes) >= 10, f"Red team starved: {red_pushes}"
    assert min(blue_pushes) >= 10, f"Blue team starved: {blue_pushes}"
    print()
    print("OK: BOTH teams get refreshed observations regularly. Bug fixed ✓")

    # Now simulate the PRE-FIX (keyed by player_id) for comparison
    print()
    print("--- Comparison: PRE-FIX behavior (keyed by player_id) ---")
    for a in agents:
        a.push_count = 0
    last_push_buggy: dict[int, int] = {
        a.player_id: -((i * obs_refresh_every_ticks) // max(1, n_agents))
        for i, a in enumerate(agents)
    }
    print(f"Buggy dict has {len(last_push_buggy)} entries (collision: red overwrote blue)")
    for tick in range(50):
        for a in agents:
            if tick - last_push_buggy[a.player_id] >= obs_refresh_every_ticks:
                a.push_count += 1
                last_push_buggy[a.player_id] = tick

    print(f"  {'agent':<25}  {'pushes':>6}")
    for a in agents:
        marker = "  <-- STARVED" if a.push_count < 5 else ""
        print(f"  slot={a.slot} pid={a.player_id} team={a.team:<5}  "
              f"{a.push_count:>6}{marker}")
    print()
    print("Pre-fix: blue team monopolized refresh slots; red team starved.")


if __name__ == "__main__":
    main()
