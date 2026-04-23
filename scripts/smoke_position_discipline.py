"""Verify all 10 players have position_discipline + system prompt
contains both 球场守则 and 位置职责 sections."""
from football_agents.players import TEAM_BLUE_5V5, TEAM_RED_5V5
from football_agents.prompts import build_system_prompt, DEFAULT_PERSONA


def main():
    assert len(TEAM_BLUE_5V5) == 5 and len(TEAM_RED_5V5) == 5
    print(f"OK: 10 players (5 blue + 5 red)")

    # All 10 should have position_discipline
    for p in TEAM_BLUE_5V5 + TEAM_RED_5V5:
        assert p.position_discipline is not None, f"{p.name} missing position_discipline"
        assert p.team_profile is not None, f"{p.name} missing team_profile"
    print("OK: all 10 personas have position_discipline + team_profile")

    # Same-position players in different teams should share the SAME discipline string
    assert TEAM_BLUE_5V5[0].position_discipline is TEAM_RED_5V5[0].position_discipline, "GK discipline should be shared"
    assert TEAM_BLUE_5V5[1].position_discipline is TEAM_RED_5V5[1].position_discipline, "RM discipline should be shared"
    print("OK: same-position players share discipline (via shared module constant)")

    # Different positions = different discipline
    assert TEAM_BLUE_5V5[0].position_discipline != TEAM_BLUE_5V5[1].position_discipline
    print("OK: different positions = different discipline")

    # System prompt for 王浩 should contain BOTH new sections
    sp = build_system_prompt(TEAM_BLUE_5V5[1])  # 王浩 RM
    assert "我对足球的理解" in sp, "missing universal section"
    assert "球队像一张网" in sp, "universal text missing"
    assert "我的位置职责（右前卫）" in sp, "missing position section"
    assert "我守右路宽度" in sp, "RM discipline text missing"
    assert "我们球队（蓝队）的风格" in sp, "team section regression"
    print("OK: 王浩 system prompt has universal守则 + position职责 + team风格")

    # DEFAULT_PERSONA has no position_discipline → no position section
    sp_default = build_system_prompt(DEFAULT_PERSONA)
    assert "我的位置职责" not in sp_default
    print("OK: DEFAULT_PERSONA has no position section (backward compat)")

    # Show excerpt
    print("\n--- 王浩 system prompt position section ---")
    in_section = False
    for line in sp.split("\n"):
        if "我的位置职责" in line:
            in_section = True
        if in_section:
            print(f"  {line}")
            if line.startswith("##") and "位置职责" not in line:
                break

    print("\n--- 5/5 PASS ---")


if __name__ == "__main__":
    main()
