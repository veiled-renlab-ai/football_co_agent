"""Verify burst-then-decay logic in MoveToController + DribbleTowardController.

The whole point: after _BURST_TICKS, the controller stops honoring LLM's
sprint/jog urgency and decays to walk-cycle (push, release, idle×4) so the
agent doesn't blast through the world while LLM is still thinking.

This test prints a 30-tick trace for each urgency and checks the key
inflection points.

Run via:
    wsl -d Ubuntu-22.04 -- bash -lc 'source ~/football-env/bin/activate && \\
        cd /mnt/c/Users/dfgfd/Desktop/football && python3 -m scripts.smoke_burst_decay'
"""
from __future__ import annotations

from football_agents.motor import (
    A, DribbleTowardController, MoveToController, make_controller,
)
from football_agents.skills import DribbleToward, MoveTo

ACTION_NAME = {
    A.IDLE: "IDLE", A.LEFT: "LEFT", A.RIGHT: "RIGHT",
    A.SPRINT: "SPR", A.RELEASE_SPRINT: "rSPR",
    A.RELEASE_DIRECTION: "rDIR", A.DRIBBLE: "DRB",
    A.RELEASE_DRIBBLE: "rDRB",
}


def _fake_obs_with_ball() -> dict:
    return {"active": 1, "left_team": [[0.0, 0.0]] * 11,
            "left_team_direction": [[0.0, 0.0]] * 11,
            "ball_owned_team": 0, "ball_owned_player": 1}


def _fake_obs_no_ball() -> dict:
    return {"active": 1, "left_team": [[0.0, 0.0]] * 11,
            "left_team_direction": [[0.0, 0.0]] * 11,
            "ball_owned_team": -1, "ball_owned_player": -1}


def trace(controller, obs, n_ticks: int) -> list[str]:
    out = []
    for _ in range(n_ticks):
        action, _ = controller.step(obs)
        out.append(ACTION_NAME.get(action, f"?{action}"))
    return out


def visual(seq: list[str]) -> str:
    return " ".join(f"{s:>4}" for s in seq)


def main() -> None:
    BURST = MoveToController._BURST_TICKS
    print(f"Burst length: {BURST} ticks (~{BURST*1000//58}ms wall at 58 tick/s)")
    print(f"After tick {BURST+1}: should decay to walk-cycle\n")
    print("Legend: SPR=SPRINT  rSPR=RELEASE_SPRINT  rDIR=RELEASE_DIR")
    print("        RIGHT=push direction  IDLE=no input  DRB=DRIBBLE\n")

    # ---- MoveTo ----------------------------------------------------------
    print("=" * 100)
    print(f"MoveTo(target_x=0.5, urgency=sprint) — first 30 ticks")
    print("=" * 100)
    skill = MoveTo(target_x=0.5, target_y=0.0, urgency="sprint")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_no_ball(), 25)
    print(f"  ticks 1-25: {visual(seq)}")
    # tick 1: SPRINT
    assert seq[0] == "SPR", f"tick 1 expected SPR, got {seq[0]}"
    # ticks 2 to 1+BURST: RIGHT (sprint phase)
    for t in range(1, 1 + BURST):
        assert seq[t] == "RIGHT", f"tick {t+1} expected RIGHT (burst sprint), got {seq[t]}"
    # tick 2+BURST: rSPR (decay transition)
    assert seq[1 + BURST] == "rSPR", f"tick {2+BURST} expected rSPR (decay), got {seq[1+BURST]}"
    # tick 3+BURST: phase 0 of walk cycle = push direction
    assert seq[2 + BURST] == "RIGHT", f"tick {3+BURST} expected RIGHT (walk push), got {seq[2+BURST]}"
    # tick 4+BURST: phase 1 = rDIR
    assert seq[3 + BURST] == "rDIR", f"tick {4+BURST} expected rDIR (walk release), got {seq[3+BURST]}"
    # ticks 5..7+BURST: IDLE (phase 2-5)
    for t in range(4 + BURST, min(7 + BURST, len(seq))):
        assert seq[t] == "IDLE", f"tick {t+1} expected IDLE (walk idle), got {seq[t]}"
    print(f"  OK: SPR (tick 1) → 10 ticks of sprint → rSPR transition → walk cycle")

    print()
    skill = MoveTo(target_x=0.5, target_y=0.0, urgency="jog")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_no_ball(), 25)
    print(f"MoveTo(urgency=jog) ticks 1-25: {visual(seq)}")
    # tick 1: rSPR
    assert seq[0] == "rSPR"
    # ticks 2 to 1+BURST: RIGHT (jog burst)
    for t in range(1, 1 + BURST):
        assert seq[t] == "RIGHT", f"tick {t+1} jog-burst expected RIGHT, got {seq[t]}"
    # tick 2+BURST onwards: walk cycle (no rSPR transition for jog → walk)
    assert seq[1 + BURST] == "RIGHT", f"tick {2+BURST} walk-cycle phase 0 expected RIGHT, got {seq[1+BURST]}"
    assert seq[2 + BURST] == "rDIR", f"tick {3+BURST} walk-cycle phase 1 expected rDIR, got {seq[2+BURST]}"
    print(f"  OK: rSPR setup → 10 jog ticks → walk cycle (no transition tick needed)")

    print()
    skill = MoveTo(target_x=0.5, target_y=0.0, urgency="walk")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_no_ball(), 25)
    print(f"MoveTo(urgency=walk) ticks 1-25: {visual(seq)}")
    # tick 1: rSPR; tick 2 onward: walk cycle continuously (no decay change)
    assert seq[0] == "rSPR"
    assert seq[1] == "RIGHT"   # phase 0
    assert seq[2] == "rDIR"    # phase 1
    assert seq[3] == seq[4] == seq[5] == seq[6] == "IDLE"  # phase 2-5
    assert seq[7] == "RIGHT"   # next cycle phase 0
    print(f"  OK: walk runs continuously through burst→decay (no break)")

    # ---- DribbleToward ---------------------------------------------------
    print()
    print("=" * 100)
    print(f"DribbleToward(target_x=0.5, urgency=sprint) — first 30 ticks")
    print("=" * 100)
    skill = DribbleToward(target_x=0.5, target_y=0.0, urgency="sprint")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_with_ball(), 25)
    print(f"  ticks 1-25: {visual(seq)}")
    # tick 1: DRB; tick 2: SPR
    assert seq[0] == "DRB" and seq[1] == "SPR"
    # ticks 3 to 2+BURST: RIGHT (sprint burst)
    for t in range(2, 2 + BURST):
        assert seq[t] == "RIGHT", f"tick {t+1} sprint-burst expected RIGHT, got {seq[t]}"
    # tick 3+BURST: rSPR (decay transition for sprint)
    assert seq[2 + BURST] == "rSPR", f"tick {3+BURST} expected rSPR decay, got {seq[2+BURST]}"
    # tick 4+BURST: walk cycle phase 0 = push
    assert seq[3 + BURST] == "RIGHT", f"tick {4+BURST} expected RIGHT walk push, got {seq[3+BURST]}"
    # tick 5+BURST: phase 1 = rDIR
    assert seq[4 + BURST] == "rDIR", f"tick {5+BURST} expected rDIR, got {seq[4+BURST]}"
    print(f"  OK: DRB + SPR → 10 sprint ticks → rSPR transition → walk cycle")

    print()
    skill = DribbleToward(target_x=0.5, target_y=0.0, urgency="walk")
    c = make_controller(skill, team_side="left", player_id=1)
    seq = trace(c, _fake_obs_with_ball(), 25)
    print(f"DribbleToward(urgency=walk) ticks 1-25: {visual(seq)}")
    assert seq[0] == "DRB" and seq[1] == "rSPR"
    # walk cycle from tick 3 (cycle_start=3): phase 0 = RIGHT
    assert seq[2] == "RIGHT"
    assert seq[3] == "rDIR"
    assert seq[4] == seq[5] == seq[6] == seq[7] == "IDLE"
    print(f"  OK: DRB + rSPR setup → continuous walk cycle through burst→decay")

    print()
    print("=" * 100)
    print("--- All burst-then-decay sequences verified ---")
    print("=" * 100)


if __name__ == "__main__":
    main()
