"""Verify side-effect skills (Track/Call/ScanBehind/ReceiveBall) preserve
the previously-installed motor controller (don't interrupt body action)."""
from football_agents.llm_client import LLMClient
from football_agents.message_bus import TeamMessageBus
from football_agents.player_agent import PlayerAgent
from football_agents.prompts import DEFAULT_PERSONA
from football_agents.skills import (
    Call, MoveTo, ReceiveBall, ScanBehind, Track, DribbleToward,
)
from football_agents.motor import MoveToController, DribbleTowardController

def main():
    bus = TeamMessageBus()
    client = LLMClient.from_env()
    a = PlayerAgent(slot=0, player_id=1, team_side="left", role="CM",
                    persona=DEFAULT_PERSONA, llm_client=client, bus=bus)

    # Install MoveTo first (motor skill)
    a.install_skill(MoveTo(target_x=0.5, target_y=0.0, urgency="sprint"))
    initial_controller = a.current_controller
    assert isinstance(initial_controller, MoveToController)
    print("Initial: MoveToController installed")

    # Install Track — should NOT replace controller
    a.install_skill(Track(entity_id=7), tick=100, raw_obs={
        "active": 1, "left_team": [[0.0,0.0]]*11, "left_team_direction": [[0,0]]*11,
        "ball_owned_team": -1, "ball_owned_player": -1,
    })
    assert a.current_controller is initial_controller, "Track should NOT replace controller"
    assert a.last_skill_name == "Track"
    assert 7 in a.perception._tracked_entity_ids
    print("OK: Track preserves MoveToController, side-effect applied")

    # Install Call — should NOT replace controller
    a.install_skill(Call(message="传球", audience="team"), tick=110, raw_obs={
        "active": 1, "left_team": [[0.0,0.0]]*11, "left_team_direction": [[0,0]]*11,
        "ball_owned_team": -1, "ball_owned_player": -1,
    })
    assert a.current_controller is initial_controller, "Call should NOT replace controller"
    assert len(bus.read_for("left", listener_id=2,
                            listener_position=__import__("football_agents.perception", fromlist=["Vec2"]).Vec2(0,0),
                            current_tick=120)) > 0
    print("OK: Call preserves controller, message posted to bus")

    # Install ScanBehind — should NOT replace controller, should arm scan flag
    a.install_skill(ScanBehind())
    assert a.current_controller is initial_controller, "ScanBehind should NOT replace controller"
    assert a.perception._scan_behind_pending == True
    print("OK: ScanBehind preserves controller, scan flag armed")

    # Install ReceiveBall — should NOT replace controller (pure intent record)
    a.install_skill(ReceiveBall())
    assert a.current_controller is initial_controller, "ReceiveBall should NOT replace controller"
    assert a.last_skill_name == "ReceiveBall"
    print("OK: ReceiveBall preserves controller (pure intent)")

    # Install DribbleToward — IS a motor skill, SHOULD replace controller
    a.install_skill(DribbleToward(target_x=0.7, target_y=0.0, urgency="sprint"))
    assert a.current_controller is not initial_controller, "DribbleToward SHOULD replace controller"
    assert isinstance(a.current_controller, DribbleTowardController)
    print("OK: DribbleToward replaces with new controller (motor skill)")

    print("\n--- side-effect skill preservation: 5/5 PASS ---")

if __name__ == "__main__":
    main()
