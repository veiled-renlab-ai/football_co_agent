"""Phase 4 demo with VIDEO recording.

Per gfootball/doc/saving_replays.md, setting `write_video=True` and
`write_full_episode_dumps=True` makes gfootball write an AVI of a simple
2D animation of the episode — NO display, NO xvfb needed.

Output goes to:  C:\\Users\\dfgfd\\Desktop\\football\\replays\\
                 (from WSL: /mnt/c/Users/dfgfd/Desktop/football/replays/)

Run from WSL with venv active:
    python3 scripts/demo_with_video.py

After the run, open the .avi from Windows in VLC or Windows Media Player.
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

REPLAY_DIR = "/mnt/c/Users/dfgfd/Desktop/football/replays"


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
    print("Phase 4 demo with VIDEO — LLM agent on academy_empty_goal_close")
    print(f"Recording to: {REPLAY_DIR}")
    print("=" * 78)

    client = LLMClient.from_env()
    print(f"LLM model: {client.model}\n")

    # write_video=True alone is enough — gfootball generates a 2D animation video
    # WITHOUT needing a display (per saving_replays.md). render=True would only
    # be needed for the real-time 3D SDL window via WSLg.
    env = FootballEnvAdapter(
        scenario="academy_empty_goal_close",
        write_video=True,
        logdir=REPLAY_DIR,
    )
    obs = env.reset()
    pid = obs.self_state.player_id
    role = obs.self_state.role
    print(f"Controlling player #{pid} ({role})\n")

    agent = LLMPlayer(player_id=pid, role=role, llm_client=client)

    decision_count = 0
    max_decisions = 30
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

        status = env.dispatch_skill(skill, max_env_ticks=8)
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
        print(f"✅ GOAL!  reward={env.cumulative_reward:+.1f}  in {decision_count} LLM decisions, {total_dt:.1f}s wall")
    elif env.done:
        print(f"⏹  Episode ended  reward={env.cumulative_reward:+.1f}  in {decision_count} decisions, {total_dt:.1f}s wall")
    else:
        print(f"⏱  Decision cap hit  reward={env.cumulative_reward:+.1f}  in {total_dt:.1f}s wall")
    print(f"\n📹 Video saved in: {REPLAY_DIR}")
    print(f"   On Windows, open with: explorer.exe \"C:\\Users\\dfgfd\\Desktop\\football\\replays\"")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
