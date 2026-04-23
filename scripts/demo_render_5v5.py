"""Phase 5b — 4 LLM agents on left team vs gfootball bots on right (5v5).

Built-in scenario `5_vs_5`:
  - Both teams have 5 players each (1 GK + 4 outfield)
  - GK is controllable=False → gfootball scripts both GKs
  - We agent-control the 4 left outfield players via n_controlled_left=4
  - Right team: scripted bots (n_controlled_right=0, default difficulty 0.05)

Each LLM agent runs in its OWN dedicated thread (per-agent isolation contract).
4 worker threads + 1 main thread (env tick + motor + filter on main).

NO MessageBus, NO Call wiring yet (Phase 5c, requires prompt-change approval).
Agents will TRY to coordinate via Call but the calls have no recipients.
This is the multi-agent baseline before adding cooperation primitives.

Run:
    python3 -u scripts/demo_render_5v5.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient
from football_agents.multi_agent_runner import MultiAgentRunner
from football_agents.perception import EgocentricFilter
from football_agents.personas import TEAM_BLUE_5V5
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
    print("Phase 5b — 5v5 multi-agent (4 LLM left vs scripted bots right)")
    print("Each LLM agent runs in its OWN dedicated thread; per-agent isolation.")
    print("=" * 80)

    client = LLMClient.from_env()
    print(f"LLM model: {client.model}\n")

    env = FootballEnvAdapter(
        scenario="5_vs_5",
        render=True,
        n_controlled_left=4,         # 4 outfield players controlled
        primary_player_slot=0,       # legacy field; unused in multi-agent path
    )
    env.reset()

    # Build 4 PlayerAgents using the verified slot→player_id map
    raw = env.raw_obs
    roles_arr = raw["left_team_roles"]
    agents = []
    for slot in range(4):
        player_id = slot + 1   # GK is player_id 0; outfield 1..4
        role_id = int(roles_arr[player_id])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        persona = TEAM_BLUE_5V5[slot]
        agents.append(
            PlayerAgent(
                slot=slot,
                player_id=player_id,
                team_side="left",
                role=role_name,
                persona=persona,
                llm_client=client,
            )
        )
        print(
            f"slot {slot} → pid {player_id} ({role_name}) → {persona.name} "
            f"(#{persona.jersey_number} {persona.position})"
        )
    print()

    # on_decision callback — print which agent decided what, with reasoning
    name_by_pid = {a.player_id: a.persona.name for a in agents}
    jersey_by_pid = {a.player_id: a.persona.jersey_number for a in agents}

    def on_decision(log: dict) -> None:
        pid = log["player_id"]
        # Find the agent so we can pull its history (for reasoning text)
        agent = next(a for a in agents if a.player_id == pid)
        last = agent.llm_player.history[-1] if agent.llm_player.history else None
        reasoning = last.reasoning if last and last.reasoning else ""
        err = last.error if last and last.error else None
        name = name_by_pid[pid]
        jn = jersey_by_pid[pid]
        print("─" * 80)
        print(f"#{log['decision']:>3}  [{name} #{jn}]  "
              f"env_tick={log['env_tick']:>4d}  obs_tick={log['obs_tick']:>4d}  "
              f"lag={log['lag_ticks']:>3d}t  llm={log['llm_seconds']:.1f}s")
        if reasoning:
            print(f"  💭 {reasoning}")
        if err:
            print(f"  ⚠️  {err}  →  fallback {short_skill(log['skill'])}")
        else:
            print(f"  🎯 {short_skill(log['skill'])}")
        # Pull a quick state line for context
        s = agent.perceive(env.raw_obs, env.tick)
        ss = s.self_state
        ball = s.ball()
        ball_str = f"my_ball={ss.has_ball}"
        if ball is not None:
            ball_str += f"  ball@({ball.position.x:+.2f},{ball.position.y:+.2f}) d={ball.distance:.2f}"
        print(f"     reward={env.cumulative_reward:+.1f}  "
              f"pos=({ss.position.x:+.2f},{ss.position.y:+.2f}) {ball_str}")

    runner = MultiAgentRunner(
        env=env,
        agents=agents,
        obs_refresh_every_ticks=4,
        max_decisions_total=200,    # 200 across all 4 agents (~50/each)
        max_wall_seconds=300.0,
        on_decision=on_decision,
    )

    print("Starting 4 worker threads (one per agent)...")
    print("Running... 3D should be smooth, decisions land async per-agent.")
    t0 = time.monotonic()
    result = runner.run()
    wall = time.monotonic() - t0

    print()
    print("=" * 80)
    if result["cumulative_reward"] > 0:
        verdict = f"✅ {int(result['cumulative_reward'])} GOAL(S)  reward={result['cumulative_reward']:+.1f}"
    elif env.done:
        verdict = f"⏹  episode ended  reward={result['cumulative_reward']:+.1f}"
    else:
        verdict = f"⏱  cap hit  reward={result['cumulative_reward']:+.1f}"
    print(
        f"{verdict}  |  {result['decisions_total']} decisions across "
        f"{result['n_agents']} agents  |  {result['env_ticks']} env ticks  |  "
        f"{wall:.1f}s wall"
    )
    if result["decisions_total"] > 0:
        avg_llm = sum(d["llm_seconds"] for d in result["log"]) / result["decisions_total"]
        avg_lag = sum(d["lag_ticks"] for d in result["log"]) / result["decisions_total"]
        print(f"avg LLM latency: {avg_llm:.2f}s  |  avg obs->action lag: {avg_lag:.1f} ticks")

        # Per-agent decision count
        from collections import Counter
        per_agent = Counter(d["player_id"] for d in result["log"])
        per_agent_str = "  ".join(
            f"{name_by_pid[pid]}#{jersey_by_pid[pid]}={cnt}"
            for pid, cnt in sorted(per_agent.items())
        )
        print(f"per-agent decisions: {per_agent_str}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
