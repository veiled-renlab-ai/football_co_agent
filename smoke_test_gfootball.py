"""
gfootball smoke test — confirms the C++ engine actually runs an episode,
not just that the Python package imports.

Run from WSL Ubuntu shell with the venv activated:
    source ~/football-env/bin/activate
    python3 /mnt/c/Users/dfgfd/Desktop/football/smoke_test_gfootball.py
"""

import sys
import numpy as np

print("=" * 50)
print("gfootball smoke test")
print("=" * 50)

# 1. Import check
import gfootball
try:
    from importlib.metadata import version as _pkg_version
    _gf_version = _pkg_version('gfootball')
except Exception:
    _gf_version = '(version unknown)'
print(f"[1/4] Import OK. gfootball version: {_gf_version}")

from gfootball.env import create_environment

# 2. Create the simplest scenario (1 attacker vs empty goal, no goalkeeper)
print("[2/4] Creating environment 'academy_empty_goal_close' (headless)...")
env = create_environment(
    env_name='academy_empty_goal_close',
    representation='simple115v2',  # flat 115-dim vector observation
    render=False,                  # no graphics — pure compute
)
print(f"        action space: {env.action_space}")

# 3. Reset and inspect observation
print("[3/4] env.reset() ...")
obs = env.reset()
# Newer gym returns (obs, info), older returns just obs
if isinstance(obs, tuple):
    obs = obs[0]
print(f"        obs shape: {np.asarray(obs).shape}")

# 4. Step the engine for up to 50 random actions
print("[4/4] Running up to 50 random steps to exercise the C++ engine...")
ep_reward = 0.0
for step_i in range(50):
    action = np.random.randint(0, env.action_space.n)
    result = env.step(action)
    # Handle both old (4-tuple) and new (5-tuple) gym APIs
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    ep_reward += float(reward)
    if done:
        print(f"        episode ended at step {step_i + 1}, total reward: {ep_reward}")
        break
else:
    print(f"        50 steps completed, no goal yet (expected for random play). total reward: {ep_reward}")

env.close()
print()
print("=" * 50)
print("gfootball smoke test PASSED")
print("=" * 50)
print()
print("The C++ engine compiled and runs end-to-end. Ready to build the agent layer.")
