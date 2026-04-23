"""Dump initial obs for all 10 agents in llm_5v5_full to diagnose
why one team runs off field at episode start.

Steps env (with no actions / IDLE) for 100 ticks. At tick 0 and tick 100,
print for EACH agent:
  - absolute position (from raw_obs)
  - self-frame position (from agent.perceive)
  - facing_deg
  - velocity description (from prompts._describe_velocity)
  - ball position (self-frame)
  - what render_observation produces (the full Chinese prompt)
  - what body_rest_state_fallback would pick for them at this state
  - motor un-mirror trace for the fallback's MoveTo
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.env import FootballEnvAdapter
from football_agents.message_bus import TeamMessageBus
from football_agents.multi_agent_runner import body_rest_state_fallback
from football_agents.perception import EgocentricFilter
from football_agents.personas import TEAM_BLUE_5V5, TEAM_RED_5V5
from football_agents.player_agent import PlayerAgent
from football_agents.prompts import (
    _describe_facing,
    _describe_position,
    _describe_velocity,
    render_observation,
)
from football_agents.skills import DribbleToward, MoveTo, HoldPosition


# -------- Fake LLM client that never gets called -------------------------

class _FakeLLM:
    model = "fake-no-call"

    def __init__(self):
        pass


def _short_skill(skill):
    name = type(skill).__name__
    args = []
    for slot in ("target_x", "target_y", "urgency", "opponent_id"):
        if hasattr(skill, slot):
            v = getattr(skill, slot)
            if isinstance(v, float):
                args.append(f"{slot}={v:+.3f}")
            else:
                args.append(f"{slot}={v}")
    return f"{name}({', '.join(args)})"


def _trace_motor_unmirror(skill, team_side: str) -> str:
    """For a MoveTo / DribbleToward, simulate the motor un-mirror step."""
    if not isinstance(skill, (MoveTo, DribbleToward)):
        return "(skill is not MoveTo/DribbleToward — no un-mirror)"
    tx, ty = float(skill.target_x), float(skill.target_y)
    if team_side == "right":
        abs_x, abs_y = -tx, -ty
    else:
        abs_x, abs_y = tx, ty
    return (
        f"LLM target self-frame=({tx:+.3f},{ty:+.3f}) -> "
        f"motor abs=({abs_x:+.3f},{abs_y:+.3f}) (team_side={team_side})"
    )


def _action_for(skill, self_pos_abs, team_side: str) -> str:
    """If skill is MoveTo, what direction action will motor send (after first-tick sprint setup)?"""
    if not isinstance(skill, (MoveTo, DribbleToward)):
        return "(no direction action)"
    tx, ty = float(skill.target_x), float(skill.target_y)
    if team_side == "right":
        tgt_x, tgt_y = -tx, -ty
    else:
        tgt_x, tgt_y = tx, ty
    dx = tgt_x - float(self_pos_abs[0])
    dy = tgt_y - float(self_pos_abs[1])
    angle_deg = math.degrees(math.atan2(dy, dx))
    # Match motor.vector_to_action sector mapping
    sector = round(math.atan2(dy, dx) / (math.pi / 4)) % 8
    names = ["RIGHT", "BOTTOM_RIGHT", "BOTTOM", "BOTTOM_LEFT",
             "LEFT", "TOP_LEFT", "TOP", "TOP_RIGHT"]
    return f"abs heading=({dx:+.3f},{dy:+.3f}) angle={angle_deg:+.1f}° -> {names[sector]}"


def main():
    print("=" * 80)
    print("PROBE: initial observations for all 10 LLM agents in llm_5v5_full")
    print("=" * 80)

    env = FootballEnvAdapter(
        scenario="llm_5v5_full",
        render=False,
        n_controlled_left=5,
        n_controlled_right=5,
        primary_player_slot=0,
    )
    env.reset()

    bus = TeamMessageBus()

    # Build agents using the same logic as demo_render_5v5_full.py
    raw = env.raw_obs
    role_arr_left = raw["left_team_roles"]
    role_arr_right = raw["right_team_roles"]

    agents: list[PlayerAgent] = []
    fake_client = _FakeLLM()

    for slot in range(5):
        player_id = slot
        role_id = int(role_arr_left[player_id])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        persona = TEAM_BLUE_5V5[slot]
        agents.append(PlayerAgent(
            slot=slot, player_id=player_id, team_side="left",
            role=role_name, persona=persona, llm_client=fake_client, bus=bus,
        ))
    for slot in range(5):
        env_slot = 5 + slot
        player_id = slot
        role_id = int(role_arr_right[player_id])
        role_name = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        persona = TEAM_RED_5V5[slot]
        agents.append(PlayerAgent(
            slot=env_slot, player_id=player_id, team_side="right",
            role=role_name, persona=persona, llm_client=fake_client, bus=bus,
        ))

    def dump(tick_label: str):
        print()
        print("#" * 80)
        print(f"# {tick_label}")
        print("#" * 80)
        raw = env.raw_obs
        print(f"\nRAW ball position (abs): {tuple(float(v) for v in raw['ball'])}")
        print(f"RAW left_team positions:")
        for i, p in enumerate(raw["left_team"]):
            print(f"  pid {i}: abs=({float(p[0]):+.3f},{float(p[1]):+.3f})")
        print(f"RAW right_team positions:")
        for i, p in enumerate(raw["right_team"]):
            print(f"  pid {i}: abs=({float(p[0]):+.3f},{float(p[1]):+.3f})")

        for a in agents:
            team_label = "蓝(L,team_a)" if a.team_side == "left" else "红(R,team_b)"
            team_key = "left_team" if a.team_side == "left" else "right_team"
            abs_pos = tuple(float(v) for v in raw[team_key][a.player_id])
            obs = a.perceive(raw, env.tick)

            print()
            print("-" * 80)
            print(
                f"slot={a.slot}  pid={a.player_id}  team={team_label}  "
                f"role={a.role}  persona={a.persona.name} #{a.persona.jersey_number} ({a.persona.position})"
            )
            print(f"  ABS pos          = ({abs_pos[0]:+.3f}, {abs_pos[1]:+.3f})")
            print(
                f"  SELF-FRAME pos   = ({obs.self_state.position.x:+.3f}, "
                f"{obs.self_state.position.y:+.3f})"
            )
            print(
                f"  vel (self-frame) = ({obs.self_state.velocity.x:+.4f}, "
                f"{obs.self_state.velocity.y:+.4f})  "
                f"=> {_describe_velocity(obs.self_state.velocity)}"
            )
            print(
                f"  facing_deg       = {obs.self_state.facing_deg:+.1f}°  "
                f"=> {_describe_facing(obs.self_state.facing_deg)}"
            )
            print(
                f"  position zone    = {_describe_position(obs.self_state.position)}"
            )
            ball = obs.ball()
            if ball is not None:
                print(
                    f"  BALL self-frame  = ({ball.position.x:+.3f}, "
                    f"{ball.position.y:+.3f})  dist={ball.distance:.3f}  "
                    f"=> zone {_describe_position(ball.position)}"
                )
                # ball motion (self-frame)
                if ball.velocity is not None:
                    print(
                        f"  BALL vel (sf)    = ({ball.velocity.x:+.4f}, "
                        f"{ball.velocity.y:+.4f})"
                    )

            # ---- fallback decision -----------------------------------
            fb = body_rest_state_fallback(obs)
            print(f"  fallback skill   = {_short_skill(fb)}")
            print(f"  motor un-mirror  : {_trace_motor_unmirror(fb, a.team_side)}")
            print(f"  direction action : {_action_for(fb, abs_pos, a.team_side)}")

            # ---- full Chinese prompt rendering -----------------------
            prompt_str = render_observation(obs, a.persona)
            print()
            print(f"  --- render_observation() output ---")
            for line in prompt_str.split("\n"):
                print(f"    {line}")

    # ---- TICK 0 dump -----------------------------------------------
    dump("TICK 0  (before any env.step)")

    # ---- step 100 ticks of all-IDLE --------------------------------
    print("\n[stepping 100 IDLE ticks...]")
    for _ in range(100):
        env.step_actions([0] * 10)

    # ---- TICK 100 dump ---------------------------------------------
    dump(f"TICK {env.tick} (after 100 IDLE steps)")

    env.close()


if __name__ == "__main__":
    main()
