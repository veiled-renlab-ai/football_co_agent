"""End-to-end verification that the self-frame mirror is removed and
right-team players actually attack toward THEIR opponent goal.

Steps:
1. Construct full 10-agent setup (5 blue + 5 red, all LLM but using a
   FAKE LLMClient that always returns DribbleToward(target_x=+0.7, target_y=0,
   urgency='sprint') — telling every agent "go forward to attack").
2. Run for 200 env ticks (no real LLM, just fakes).
3. Check final positions:
   - Left team players should have moved toward absolute +x (their opp goal)
   - Right team players should have moved toward absolute -x (their opp goal)

If both teams advanced toward their respective opponent goals, the bug is
fixed. If right team drifted toward absolute +x (their own goal), bug
remains.
"""
import logging
logging.basicConfig(level=logging.WARNING)

from football_agents.env import FootballEnvAdapter
from football_agents.llm_client import LLMClient, LLMDecision
from football_agents.message_bus import TeamMessageBus
from football_agents.multi_agent_runner import MultiAgentRunner
from football_agents.personas import TEAM_BLUE_5V5, TEAM_RED_5V5
from football_agents.player_agent import PlayerAgent


class FakeFwdLLMClient:
    """LLM that always picks DribbleToward(+0.7, 0, sprint).
    Every agent gets the same forward-attack intent."""
    def __init__(self, real_client):
        self.model = real_client.model
        self.base_url = real_client.base_url
        self._client = real_client._client

    def chat_with_messages(self, messages, tools, **kwargs):
        # Fake a tool call to invoke_skill with DribbleToward
        class FakeMsg:
            content = "go forward"
            tool_calls = [type("TC", (), {
                "id": "fake",
                "function": type("F", (), {
                    "name": "invoke_skill",
                    "arguments": '{"skill_name": "dribble_toward", "args": {"target_x": 0.7, "target_y": 0, "urgency": "sprint"}}',
                })()
            })()]
            def model_dump(self, **kw):
                return {"role": "assistant", "content": self.content,
                        "tool_calls": [{"id": "fake", "type": "function",
                                        "function": {"name": "invoke_skill",
                                                     "arguments": self.tool_calls[0].function.arguments}}]}
        return LLMDecision(
            tool_name="invoke_skill",
            tool_args={"skill_name": "dribble_toward",
                       "args": {"target_x": 0.7, "target_y": 0, "urgency": "sprint"}},
            reasoning="forward",
            raw_message=FakeMsg(),
        )


def main():
    env = FootballEnvAdapter(
        scenario="llm_5v5_full",
        render=False,
        n_controlled_left=5,
        n_controlled_right=5,
    )
    env.reset()
    real_client = LLMClient.from_env()
    fake_client = FakeFwdLLMClient(real_client)
    bus = TeamMessageBus()

    initial_left = [list(env.raw_obs_for_slot(0)["left_team"][i]) for i in range(5)]
    initial_right = [list(env.raw_obs_for_slot(5)["left_team"][i]) for i in range(5)]
    print("Initial positions:")
    print(f"  LEFT  team: {initial_left}")
    print(f"  RIGHT team (slot view): {initial_right}")

    agents = []
    for slot in range(5):
        agents.append(PlayerAgent(
            slot=slot, player_id=slot, team_side="left", role="CM",
            persona=TEAM_BLUE_5V5[slot], llm_client=fake_client, bus=bus,
        ))
    for slot in range(5):
        agents.append(PlayerAgent(
            slot=5+slot, player_id=slot, team_side="right", role="CM",
            persona=TEAM_RED_5V5[slot], llm_client=fake_client, bus=bus,
        ))

    runner = MultiAgentRunner(
        env=env, agents=agents,
        max_decisions_total=100,
        max_wall_seconds=15.0,
        target_wall_fps=0,  # no throttle for fast probe
    )
    runner.run()

    # Read final ABSOLUTE positions from raw[0] (the "left perspective" canonical view)
    final_left_abs = [list(env.raw_obs["left_team"][i]) for i in range(5)]
    final_right_abs = [list(env.raw_obs["right_team"][i]) for i in range(5)]
    print("\nFinal absolute positions:")
    print(f"  LEFT  team:  {final_left_abs}")
    print(f"  RIGHT team:  {final_right_abs}")

    # Compute total displacement in absolute x for each team
    initial_left_abs_x = [p[0] for p in initial_left]
    initial_right_abs_x = [-p[0] for p in initial_right]  # un-rotate the slot view to abs
    final_left_x = [p[0] for p in final_left_abs]
    final_right_x = [p[0] for p in final_right_abs]

    left_drift = sum(f - i for f, i in zip(final_left_x, initial_left_abs_x))
    right_drift = sum(f - i for f, i in zip(final_right_x, initial_right_abs_x))
    print(f"\nTotal abs-x drift: LEFT={left_drift:+.3f}, RIGHT={right_drift:+.3f}")
    print(f"Expected: LEFT > 0 (toward +x = opp goal),  RIGHT < 0 (toward -x = opp goal)")

    assert left_drift > 0.05, f"LEFT team failed to advance toward opp goal: {left_drift}"
    assert right_drift < -0.05, f"RIGHT team failed to advance toward opp goal: {right_drift}"
    print("\nBOTH TEAMS ATTACK THEIR OPP GOAL — mirror bug FIXED")

    env.close()


if __name__ == "__main__":
    main()
