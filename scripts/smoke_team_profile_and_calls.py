from football_agents.personas import TEAM_BLUE_5V5, BLUE_TEAM_PROFILE
from football_agents.prompts import (
    DEFAULT_PERSONA, TeamProfile, build_system_prompt, render_observation
)
from football_agents.perception import Observation, SelfState, Vec2
from football_agents.message_bus import HeardCall


def main():
    # Test 1: DEFAULT_PERSONA has no team_profile
    sp_default = build_system_prompt(DEFAULT_PERSONA)
    assert "我们球队" not in sp_default, "DEFAULT_PERSONA shouldn't get team section"
    print("OK: DEFAULT_PERSONA system prompt has no team section")

    # Test 2: TEAM_BLUE_5V5[0] (王浩) has team_profile injected
    wang = TEAM_BLUE_5V5[0]
    assert wang.team_profile is not None, "wang_hao should have team_profile"
    sp_wang = build_system_prompt(wang)
    assert "蓝队" in sp_wang, "should inject 蓝队 team section"
    assert "传控渗透" in sp_wang, "should include team character text"
    print("OK: 王浩 system prompt includes team-style section")
    print(f"  Excerpt: {[l for l in sp_wang.split(chr(10)) if '蓝队' in l or '传控' in l]}")

    # Test 3: Observation with no heard_calls
    obs = Observation(
        tick=0, match_clock="00:00", score=(0, 0),
        self_state=SelfState(player_id=1, team="team_a", role="CM",
                             position=Vec2(0, 0), velocity=Vec2(0, 0),
                             facing_deg=0, stamina=1.0, has_ball=False),
        heard_calls=[],
    )
    rendered = render_observation(obs, wang)
    assert "你听到" not in rendered, "no heard_calls -> no section"
    print("OK: empty heard_calls -> no section in render")

    # Test 4: Observation WITH heard_calls
    obs.heard_calls = [
        HeardCall(sender_player_id=2, sender_jersey=9,
                  sender_position=Vec2(0.4, 0.1), message="传给我",
                  audience="team", age_ticks=15),  # 0.3s ago
        HeardCall(sender_player_id=3, sender_jersey=3,
                  sender_position=Vec2(-0.5, -0.2), message="身后有人",
                  audience="team", age_ticks=5),   # 0.1s ago = "刚刚"
    ]
    rendered = render_observation(obs, wang)
    assert "听到队友的喊话" in rendered
    assert "传给我" in rendered
    assert "身后有人" in rendered
    assert "刚刚" in rendered
    print("OK: heard_calls render with proper formatting")
    print("Excerpt:")
    for line in rendered.split("\n"):
        if "喊话" in line or "号" in line and "在" in line:
            print(f"  {line}")


if __name__ == "__main__":
    main()
