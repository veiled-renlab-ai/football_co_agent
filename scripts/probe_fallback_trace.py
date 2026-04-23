"""Probe: run 100 ticks of pure fallback with NO LLM on llm_5v5_full,
and verify nobody runs off the field."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from football_agents.env import FootballEnvAdapter
from football_agents.perception import EgocentricFilter
from football_agents.motor import make_controller
from football_agents.multi_agent_runner import body_rest_state_fallback


env = FootballEnvAdapter(
    scenario="llm_5v5_full",
    render=False,
    n_controlled_left=5,
    n_controlled_right=5,
    primary_player_slot=0,
)
env.reset()
raw = env.raw_obs

# Build 10 filters, one per player
filters = {}
for side in ("left", "right"):
    team_key = f"{side}_team"
    team_label = "team_a" if side == "left" else "team_b"
    roles = raw[f"{team_key}_roles"]
    for pid in range(5):
        role_id = int(roles[pid])
        role = EgocentricFilter.GFOOTBALL_ROLE_TO_NAME.get(role_id, "CM")
        filters[(side, pid)] = EgocentricFilter(
            player_id=pid, team=team_label, role=role,
        )

# Build initial fallback controllers
controllers = {}
for (side, pid), f in filters.items():
    obs = f.filter(raw, 0)
    skill = body_rest_state_fallback(obs)
    ctrl = make_controller(skill, team_side=side, player_id=pid)
    controllers[(side, pid)] = (ctrl, skill)

print("Initial fallback decisions:")
for (side, pid), (_, skill) in sorted(controllers.items()):
    tx = getattr(skill, "target_x", "-")
    ty = getattr(skill, "target_y", "-")
    name = type(skill).__name__
    tx_s = f"{tx:+.3f}" if isinstance(tx, float) else str(tx)
    ty_s = f"{ty:+.3f}" if isinstance(ty, float) else str(ty)
    print(f"  {side}/{pid}: {name}(tx={tx_s}, ty={ty_s})")
print()

# Run 100 ticks
for tick in range(1, 101):
    actions = []
    for side in ("left", "right"):
        for pid in range(5):
            ctrl, skill = controllers[(side, pid)]
            action, status = ctrl.step(raw)
            actions.append(action)
            if status != "in_progress":
                # Re-arm fallback
                f = filters[(side, pid)]
                obs = f.filter(raw, tick)
                skill2 = body_rest_state_fallback(obs)
                ctrl2 = make_controller(skill2, team_side=side, player_id=pid)
                controllers[(side, pid)] = (ctrl2, skill2)
    env.step_actions(actions)
    raw = env.raw_obs

    if tick in (1, 5, 10, 25, 50, 100):
        print(f"--- tick {tick} ---")
        print(f"  LEFT  positions:")
        for pid in range(5):
            p = raw["left_team"][pid]
            d = raw["left_team_direction"][pid]
            print(f"    pid {pid}: pos=({p[0]:+.3f}, {p[1]:+.3f})  vel=({d[0]:+.4f}, {d[1]:+.4f})")
        print(f"  RIGHT positions:")
        for pid in range(5):
            p = raw["right_team"][pid]
            d = raw["right_team_direction"][pid]
            print(f"    pid {pid}: pos=({p[0]:+.3f}, {p[1]:+.3f})  vel=({d[0]:+.4f}, {d[1]:+.4f})")

env.close()
