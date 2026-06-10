"""Generates golden CartPole-v1 trajectories from the reference Gymnasium implementation.

The C# port replays the same initial states and action sequences and must reproduce
these observations bit-for-bit at float64 physics / float32 observation precision.

Usage: python tools/generate_goldens.py  (writes tests/RL.NET.Tests/Fixtures/cartpole_golden.json)
"""
import json
import os

import gymnasium as gym
import numpy as np

SCENARIOS = [
    # (name, initial state [x, x_dot, theta, theta_dot], action pattern)
    ("alternating_from_origin", [0.0, 0.0, 0.0, 0.0], lambda t: t % 2),
    ("push_right_from_offset", [0.01, -0.02, 0.03, 0.04], lambda t: 1),
    ("push_left_to_termination", [-1.0, -0.5, -0.05, -0.1], lambda t: 0),
    ("near_angle_limit", [0.0, 0.0, 0.18, 0.5], lambda t: (t // 3) % 2),
]

def run(initial_state, policy, max_steps=300):
    env = gym.make("CartPole-v1")
    env.reset(seed=0)
    env.unwrapped.state = np.array(initial_state, dtype=np.float64)
    steps = []
    for t in range(max_steps):
        action = int(policy(t))
        obs, reward, terminated, truncated, _ = env.step(action)
        steps.append({
            "action": action,
            "obs": [float(x) for x in obs],
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        if terminated or truncated:
            break
    env.close()
    return steps

fixture = {
    "gymnasium_version": gym.__version__,
    "scenarios": [
        {"name": name, "initial_state": state, "steps": run(state, policy)}
        for name, state, policy in SCENARIOS
    ],
}

out = os.path.join(os.path.dirname(__file__), "..", "tests", "RL.NET.Tests", "Fixtures", "cartpole_golden.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(fixture, f, indent=1)
print(f"wrote {out}: " + ", ".join(f"{s['name']} ({len(s['steps'])} steps)" for s in fixture["scenarios"]))
