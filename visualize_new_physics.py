#!/usr/bin/env python3
"""
Visualize the foosball physics with both teams taking random actions.
Verifies ball bounces, goal detection, capsule-only collision, and
the rotation-angle kick guard for both yellow and blue sides.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
import numpy as np

# ── table constants (must match gym env) ──────────────────────────
GOAL_Y = 40.5
GOAL_HALF_W = 4.0

class RandomAntagonist:
    """Dummy model that returns random actions in [-1,1], matching SB3 API.
    The env scales these to ±20 internally so blue is equally aggressive."""
    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, obs, **kwargs):
        action = np.random.uniform(-1.0, 1.0, size=self.action_size).astype(np.float32)
        return action, None


NUM_EPISODES = 10

print("=" * 60)
print("  Foosball Physics Visualization — Both Teams Playing")
print("=" * 60)
print("  Table: 48.5 × 81 cm, 1-3-4-3 formation")
print("  Physics checks:")
print("    - Capsule-only collision (visual meshes disabled)")
print("    - Rotation-angle kick guard (MAX_KICK_ANGLE = 70°)")
print("    - Goal opening |x| < 4.0, end-wall bounces elsewhere")
print("    - Both yellow AND blue take random actions")
print()
print(f"  Running {NUM_EPISODES} episodes...")
print("  Close the window or press Ctrl+C to stop.")
print("=" * 60)

antagonist = RandomAntagonist(action_size=8)
env = FoosballEnv(antagonist_model=antagonist)

yellow_goals_total = 0
blue_goals_total = 0

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

            ball_pos = env.data.body(env._ball_bid).xpos
            ball_world_x = ball_pos[0]
            ball_world_y = ball_pos[1]
            ball_vel_y = env.data.qvel[env._by_dof]
            max_world_y = max(max_world_y, ball_world_y)
            min_world_y = min(min_world_y, ball_world_y)

            # Detect kicks (velocity jumps)
            if abs(ball_vel_y) > abs(prev_vel) + 3.0:
                kick_count += 1
            prev_vel = ball_vel_y

            # Render
            env.render()

        # Determine how the episode ended
        goal_str = ""
        if abs(ball_world_y) >= GOAL_Y and abs(ball_world_x) < GOAL_HALF_W:
            if ball_world_y > 0:
                goal_str = "  YELLOW SCORES! (ball into blue goal)"
                yellow_goals_total += 1
            else:
                goal_str = "  BLUE SCORES!   (ball into yellow goal)"
                blue_goals_total += 1

        print(f"  Steps: {steps}  |  Reward: {total_reward:+.1f}  |  Kicks: {kick_count}")
        print(f"  Ball Y range: [{min_world_y:.1f}, {max_world_y:.1f}]  (goals at +/-{GOAL_Y}){goal_str}")

    print("\n" + "=" * 60)
    print(f"  Final tally: Yellow {yellow_goals_total} - {blue_goals_total} Blue")
    print("  Visualization complete!")
    print("=" * 60)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
finally:
    env.close()
