"""Inspect gfootball's raw observation format so we know exactly what
the Motor layer has to work with. Used as scratch tooling, not part of the
runtime package.
"""
from __future__ import annotations

import pprint

import numpy as np
from gfootball.env import create_environment


def main() -> None:
    env = create_environment(
        env_name="academy_empty_goal_close",
        representation="raw",
        render=False,
    )
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    print("=" * 60)
    print(f"Top-level type: {type(obs)}")
    print("=" * 60)

    if isinstance(obs, list):
        print(f"obs is a list of {len(obs)} dicts (one per controlled player)")
        sample = obs[0]
    else:
        print("obs is a single dict")
        sample = obs

    print()
    print("=== Keys + shapes ===")
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:30s} ndarray shape={v.shape} dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k:30s} {type(v).__name__} len={len(v)}")
        else:
            print(f"  {k:30s} {type(v).__name__} = {v!r}")

    print()
    print("=== Full sample (depth=3) ===")
    pprint.pprint(sample, depth=3, width=100)

    # Also do one step and re-inspect to see what changes
    print()
    print("=" * 60)
    print("After one step (action=0, idle):")
    print("=" * 60)
    result = env.step(0)
    if len(result) == 5:
        obs2, _, _, _, _ = result
    else:
        obs2, _, _, _ = result
    if isinstance(obs2, list):
        obs2 = obs2[0]
    print(f"  ball position changed: {sample['ball']} -> {obs2['ball']}")
    print(f"  ball_owned_team:       {sample['ball_owned_team']} -> {obs2['ball_owned_team']}")
    print(f"  ball_owned_player:     {sample['ball_owned_player']} -> {obs2['ball_owned_player']}")
    print(f"  active player:         {sample['active']} -> {obs2['active']}")

    env.close()


if __name__ == "__main__":
    main()
