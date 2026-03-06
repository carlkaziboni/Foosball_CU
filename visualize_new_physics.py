#!/usr/bin/env python3
"""
Visualize the new foosball physics with random actions.
Shows ball being kicked by virtual kicks, traveling across the field,
and goals being scored — all with stable simulation (no NaN).
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
import numpy as np

NUM_EPISODES = 10

print("=" * 60)
print("  Foosball Physics Visualization — Random Actions")
print("=" * 60)
print("  Fixes applied:")
print("    - All collision disabled (virtual kicks only)")
print("    - Actuator gains stabilised (kp 150k→5k)")
print("    - Armature=1.0 on rotation DOFs")
print("    - KICK_SPEED=120, KICK_RADIUS=10, ball_x limits ±32")
print()
print(f"  Running {NUM_EPISODES} episodes with random actions...")
print("  Close the window or press Ctrl+C to stop.")
print("=" * 60)

env = FoosballEnv()

try:
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_world_y = -999
        min_world_y = 999
        kick_count = 0
        prev_vel = 0

        print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            ball_world_y = env.data.body(env._ball_bid).xpos[1]
            ball_vel_y = env.data.qvel[env._by_dof]
            max_world_y = max(max_world_y, ball_world_y)
            min_world_y = min(min_world_y, ball_world_y)

            # Detect kicks (velocity jumps)
            if abs(ball_vel_y) > abs(prev_vel) + 3.0:
                kick_count += 1
            prev_vel = ball_vel_y

            # Render
            env.render()

        goal_str = ""
        if ball_world_y > 65:
            goal_str = "  ⚽ GOAL SCORED! (yellow → blue goal)"
        elif ball_world_y < -65:
            goal_str = "  ❌ Goal conceded (blue → yellow goal)"

        print(f"  Steps: {steps}  |  Reward: {total_reward:+.1f}  |  Kicks: {kick_count}")
        print(f"  Ball Y range: [{min_world_y:.1f}, {max_world_y:.1f}]  (goals at ±65){goal_str}")

    print("\n" + "=" * 60)
    print("  Visualization complete!")
    print("=" * 60)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
finally:
    env.close()
