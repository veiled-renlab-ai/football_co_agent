"""Phase 4 demo with REAL-TIME 3D window via WSLg.

Per gfootball/doc/api.md, calling render() (or create_environment(render=True))
opens an SDL-based real-time 3D window. With WSLg available (DISPLAY=:0,
WAYLAND_DISPLAY=wayland-0), the window appears directly on the Windows desktop.

Tradeoff: rendering slows env.step() significantly — expect maybe 10-20s
per LLM decision (LLM call ~1-3s + 8 rendered ticks ~5-15s).

Run from WSL with venv active:
    python3 scripts/demo_render_3d.py

Watch for the gfootball window popping up on your Windows desktop.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.agent import LLMPlayer
from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient

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
    print("=" * 78)
    print("Phase 4 demo with REAL-TIME 3D WINDOW (via WSLg)")
    print("Watch your Windows desktop — gfootball window should pop up.")
    print("=" * 78)

    client = LLMClient.from_env()
    print(f"LLM model: {client.model}\n")

    # render=True opens the SDL 3D window via WSLg.
    # n_controlled_left=2 + primary_player_slot=1 explicitly drives the attacker
    # (player #1, CB role); the GK (slot #0) gets IDLE. Without this, gfootball
    # auto-control starts on the GK and only switches when the ball moves —
    # demo wastes decisions on the wrong player.
    # Custom scenario llm_solo_attack: 3000 ticks, no possession-change
    # termination, no goal-end termination — continuous football.
    # Defined in venv: site-packages/gfootball/scenarios/llm_solo_attack.py
    env = FootballEnvAdapter(
        scenario="llm_solo_attack",
        render=True,
        n_controlled_left=2,
        primary_player_slot=1,
    )
    obs = env.reset()
    pid = obs.self_state.player_id
    role = obs.self_state.role
    print(f"Controlling player #{pid} ({role})\n")

    agent = LLMPlayer(player_id=pid, role=role, llm_client=client)

    decision_count = 0
    max_decisions = 100000   # let env.done (game_duration reached) be the only stop
    t0 = time.monotonic()

    while not env.done and decision_count < max_decisions:
        if obs.self_state.player_id != agent.player_id or obs.self_state.role != agent.role:
            agent.player_id = obs.self_state.player_id
            agent.update_role(obs.self_state.role)

        decision_count += 1
        t_dec = time.monotonic()
        skill = agent.choose_skill(obs)
        dec_dt = time.monotonic() - t_dec
        last = agent.history[-1]

        print("─" * 78)
        print(f"Decision #{decision_count}  (tick={env.tick}, LLM took {dec_dt:.1f}s)")
        if last.reasoning:
            print(f"  💭 {last.reasoning}")
        if last.error:
            print(f"  ⚠️  {last.error}  → fallback to {short_skill(skill)}")
        else:
            print(f"  🎯 {short_skill(skill)}")

        # max_env_ticks=4 keeps the game clock advancing in small steps
        # (instead of jumping ~14 sim sec per LLM decision); smoother to watch.
        status = env.dispatch_skill(skill, max_env_ticks=4)
        obs = env.observe()
        s = obs.self_state
        ball_str = f"my_ball={s.has_ball}"
        if (b := obs.ball()) is not None:
            ball_str += f"  ball@({b.position.x:+.2f},{b.position.y:+.2f}) d={b.distance:.2f}"
        print(f"  → {status}  reward={env.cumulative_reward:+.1f}  "
              f"pos=({s.position.x:+.2f},{s.position.y:+.2f}) {ball_str}")

    total_dt = time.monotonic() - t0
    print()
    print("=" * 78)
    if env.cumulative_reward > 0:
        print(f"✅ GOAL!  reward={env.cumulative_reward:+.1f}  in {decision_count} decisions, {total_dt:.1f}s wall")
    elif env.done:
        print(f"⏹  Episode ended  reward={env.cumulative_reward:+.1f}")
    else:
        print(f"⏱  Decision cap hit  reward={env.cumulative_reward:+.1f}")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
