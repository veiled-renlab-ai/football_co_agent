"""What does doubao-seed-2-0-code return when given our real system prompt
+ the invoke_skill meta-tool? Capture tool_name + tool_args shape so we
know whether agent.py's new dual-path parser will work.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.fallbacks import FallbackContext
from football_agents.llm_client import LLMClient
from football_agents.perception import EntityView, Observation, SelfState, Vec2
from football_agents.personas import TEAM_BLUE_5V5
from football_agents.prompts import build_system_prompt, render_observation
from football_agents.skills import ALL_SKILLS, SKILLS_BY_NAME, make_invoke_skill_tool


def build_realistic_obs(persona) -> Observation:
    self_state = SelfState(
        player_id=2, team="team_a", role="CF",
        position=Vec2(0.60, 0.0), velocity=Vec2(0.002, 0.0),
        facing_deg=0.0, stamina=0.8, has_ball=True,
    )
    opp_gk = EntityView(
        entity_id=0, role="opponent",
        position=Vec2(0.95, 0.0), velocity=Vec2(0.0, 0.0),
        distance=0.35, in_current_fov=True, has_ball=False,
    )
    teammate = EntityView(
        entity_id=1, role="teammate",
        position=Vec2(0.50, 0.30), velocity=Vec2(0.0, 0.0),
        distance=0.32, in_current_fov=True, has_ball=False,
    )
    ball = EntityView(
        entity_id=99, role="ball",
        position=Vec2(0.60, 0.0), velocity=Vec2(0.0, 0.0),
        distance=0.0, in_current_fov=True, has_ball=False,
    )
    return Observation(
        tick=200, match_clock="00:04", score=(0, 0),
        self_state=self_state, game_mode=0,
        perceived_entities=[opp_gk, teammate, ball],
    )


def main() -> None:
    persona = TEAM_BLUE_5V5[2]  # 陈宇 CF, has_ball
    system_prompt = build_system_prompt(persona)
    obs = build_realistic_obs(persona)
    user_msg = render_observation(obs, persona)

    valid_names = [c.tool_name for c in ALL_SKILLS]
    invoke_tool = make_invoke_skill_tool(valid_names)

    client = LLMClient.from_env()
    print(f"model: {client.model}\n")

    dec = client.chat_with_messages(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        tools=[invoke_tool],
    )
    print(f"tool_name:  {dec.tool_name!r}")
    print(f"tool_args:  {json.dumps(dec.tool_args, ensure_ascii=False)}")
    print(f"reasoning:  {dec.reasoning[:150]!r}")

    # Show what agent.py's new dual-path parser would do
    if dec.tool_name == "invoke_skill":
        sn = dec.tool_args.get("skill_name")
        ar = dec.tool_args.get("args", {})
        print(f"\n[meta-tool path] skill_name={sn!r}, args={ar}")
    else:
        sn = dec.tool_name
        ar = dec.tool_args
        print(f"\n[direct path] skill_name={sn!r}, args={ar}")

    cls = SKILLS_BY_NAME.get(sn)
    print(f"skill_cls lookup: {cls}")
    if cls is not None:
        try:
            inst = cls(**ar)
            print(f"instantiate ok: {inst}")
        except TypeError as e:
            print(f"instantiate FAIL: {e}")


if __name__ == "__main__":
    main()
