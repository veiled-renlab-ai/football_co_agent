"""Phase 5d — 5v5 with ALL 10 players LLM-controlled.

Custom scenario `llm_5v5_full`: every player (incl. both GKs) is
controllable. 10 PlayerAgents:
  - 5 蓝队 LLM (slots 0-4: GK, RM, CF, LB, CB) — TEAM_BLUE_5V5
  - 5 红队 LLM (slots 5-9: GK, RM, CF, LB, CB) — TEAM_RED_5V5

All share one TeamMessageBus (per-team partitioned channels).

Run:
    python3 -u scripts/demo_render_5v5_full.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient
from football_agents.message_bus import TeamMessageBus
from football_agents.multi_agent_runner import MultiAgentRunner
from football_agents.perception import EgocentricFilter
from football_agents.personas import TEAM_BLUE_5V5, TEAM_RED_5V5
from football_agents.player_agent import PlayerAgent

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


def short_skill(skill) -> str:
    name = type(skill).__name__
    args = []
    for slot in ("target_x", "target_y", "target_player_id", "target_zone",
                 "pass_type", "urgency", "opponent_id", "entity_id", "audience"):
        if hasattr(skill, slot):
            v = getattr(skill, slot)
            args.append(f"{slot}={v:+.2f}" if isinstance(v, float) else f"{slot}={v}")
    if hasattr(skill, "message"):
        args.append(f"msg={skill.message!r}")
    return f"{name}({', '.join(args)})"


def main() -> None:
    print("=" * 80)
    print("Phase 5d — 5v5 FULL: 10 LLM agents (蓝队 5 vs 红队 5, all controllable)")
    print("=" * 80)

    client = LLMClient.from_env()
    print(f"LLM model: {client.model}\n")

    env = FootballEnvAdapter(
        scenario="llm_5v5_full",
        render=True,
        n_controlled_left=5,
        n_controlled_right=5,
        primary_player_slot=0,  # legacy field, unused
    )
    env.reset()

    # Shared TeamMessageBus (separate channels per team — Chinese wall holds)
    bus = TeamMessageBus()
    print(f"TeamMessageBus created (per-team partitioned, "
          f"{TeamMessageBus.MESSAGE_LIFETIME_TICKS}-tick TTL)\n")

    # Slot mapping (verified by smoke_5v5_both_teams):
    #   L slots 0..4 -> player_id 0..4 (GK, RM, CF, LB, CB)
    #   R slots 5..9 -> player_id 0..4 (GK, RM, CF, LB, CB) (right-team relative)
    raw = env.raw_obs
    role_arr_left = raw["left_team_roles"]
    role_arr_right = raw["right_team_roles"]
    agents = []

    print("--- LEFT TEAM (蓝队) ---")
    for slot in range(5):
        player_id = slot   # 0..4
        role_id = int(role_arr_left[player_id])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        persona = TEAM_BLUE_5V5[slot]
        agents.append(PlayerAgent(
            slot=slot, player_id=player_id, team_side="left",
            role=role_name, persona=persona, llm_client=client, bus=bus,
        ))
        print(f"  slot {slot} -> pid {player_id} ({role_name}) -> "
              f"{persona.name} #{persona.jersey_number} {persona.position}")

    print("--- RIGHT TEAM (红队) ---")
    for slot in range(5):
        env_slot = 5 + slot
        player_id = slot   # 0..4 in right_team array
        role_id = int(role_arr_right[player_id])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        persona = TEAM_RED_5V5[slot]
        agents.append(PlayerAgent(
            slot=env_slot, player_id=player_id, team_side="right",
            role=role_name, persona=persona, llm_client=client, bus=bus,
        ))
        print(f"  slot {env_slot} -> pid {player_id} ({role_name}) -> "
              f"{persona.name} #{persona.jersey_number} {persona.position}")
    print()

    name_by_slot = {a.slot: a.persona.name for a in agents}
    jersey_by_slot = {a.slot: a.persona.jersey_number for a in agents}
    team_by_slot = {a.slot: ("蓝" if a.team_side == "left" else "红") for a in agents}

    def on_decision(log: dict) -> None:
        slot = log["slot"]
        agent = next(a for a in agents if a.slot == slot)
        last = agent.llm_player.history[-1] if agent.llm_player.history else None
        reasoning = last.reasoning if last and last.reasoning else ""
        err = last.error if last and last.error else None
        team = team_by_slot[slot]
        name = name_by_slot[slot]
        jn = jersey_by_slot[slot]
        print("─" * 80)
        print(f"#{log['decision']:>3}  [{team}{name} #{jn}]  "
              f"env_tick={log['env_tick']:>4d}  obs_tick={log['obs_tick']:>4d}  "
              f"lag={log['lag_ticks']:>3d}t  llm={log['llm_seconds']:.1f}s")
        if reasoning:
            print(f"  💭 {reasoning}")
        if err:
            print(f"  ⚠️  {err}  →  fallback {short_skill(log['skill'])}")
        else:
            print(f"  🎯 {short_skill(log['skill'])}")

    runner = MultiAgentRunner(
        env=env, agents=agents,
        obs_refresh_every_ticks=25,  # ~500ms 同步快照 (50fps × 25 = 0.5s)
        max_decisions_total=400,
        max_wall_seconds=300.0,
        on_decision=on_decision,
    )

    print("Starting 10 worker threads (one per agent)...")
    print("Running... 10 LLMs deciding async.\n")
    t0 = time.monotonic()
    result = runner.run()
    wall = time.monotonic() - t0

    print()
    print("=" * 80)
    if result["cumulative_reward"] > 0:
        verdict = f"✅ score: 蓝 {int(result['cumulative_reward'])} : 0 红"
    elif result["cumulative_reward"] < 0:
        verdict = f"✅ score: 蓝 0 : {abs(int(result['cumulative_reward']))} 红"
    else:
        verdict = f"⏹  no goal scored  reward={result['cumulative_reward']:+.1f}"
    print(
        f"{verdict}  |  {result['decisions_total']} decisions across "
        f"{result['n_agents']} agents  |  {result['env_ticks']} env ticks  |  "
        f"{wall:.1f}s wall"
    )
    if result["decisions_total"] > 0:
        avg_llm = sum(d["llm_seconds"] for d in result["log"]) / result["decisions_total"]
        avg_lag = sum(d["lag_ticks"] for d in result["log"]) / result["decisions_total"]
        print(f"avg LLM latency: {avg_llm:.2f}s  |  avg lag: {avg_lag:.1f} ticks")
        from collections import Counter
        per = Counter(d["slot"] for d in result["log"])
        per_str = "  ".join(
            f"{team_by_slot[s]}{name_by_slot[s]}={c}" for s, c in sorted(per.items())
        )
        print(f"per-agent: {per_str}")
    print("=" * 80)
    env.close()


if __name__ == "__main__":
    main()
