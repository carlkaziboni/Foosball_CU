#!/usr/bin/env python3
"""
Visualize a trained SAC agent from the models_kaggle folder.

Usage:
    python3 visualize_kaggle_model.py                     # best_model vs self
    python3 visualize_kaggle_model.py checkpoint_6k       # 6k checkpoint vs self
    python3 visualize_kaggle_model.py best_model random   # best_model vs random

Available checkpoints under models_kaggle/0/sac/:
    best_model        — EvalCallback best + final epoch save
    checkpoint_6k     — milestone save at 6 000 cumulative timesteps
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import SAC
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

# ── Pick checkpoint ────────────────────────────────────────────────────────────
checkpoint_name = sys.argv[1] if len(sys.argv) > 1 else "best_model"
opponent_mode   = sys.argv[2] if len(sys.argv) > 2 else "self"   # "self" or "random"
models_dir_name = sys.argv[3] if len(sys.argv) > 3 else "models_kaggle"  # e.g. "models_new"

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), models_dir_name)

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "0", "sac", checkpoint_name, "model.zip")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"ERROR: Model not found at {CHECKPOINT_PATH}")
    print("\nAvailable checkpoints:")
    sac_dir = os.path.join(MODELS_DIR, "0", "sac")
    if os.path.isdir(sac_dir):
        for name in sorted(os.listdir(sac_dir)):
            candidate = os.path.join(sac_dir, name, "model.zip")
            if os.path.exists(candidate):
                print(f"  {name}")
    sys.exit(1)

NUM_EPISODES = 10


class DeterministicModelWrapper:
    """Wraps an SB3 SAC model so the env receives deterministic actions from it.
    The env calls predict(obs) without deterministic= so we inject it here."""
    def __init__(self, sac_model):
        self.sac_model = sac_model

    def predict(self, obs, **kwargs):
        return self.sac_model.predict(obs, deterministic=True)


class RandomAntagonist:
    """Random blue-team opponent."""
    def __init__(self, action_size=8):
        self.action_size = action_size

    def predict(self, obs, **kwargs):
        action = np.random.uniform(-1.0, 1.0, size=self.action_size).astype(np.float32)
        return action, None


print("=" * 60)
print(f"  Foosball — Kaggle Model Visualization")
print(f"  Models dir : {models_dir_name}")
print(f"  Checkpoint : {checkpoint_name}")
print(f"  Path       : {CHECKPOINT_PATH}")
print(f"  Opponent   : {'self (model vs model)' if opponent_mode == 'self' else 'random'}")
print("=" * 60)
print(f"  Loading SAC model ...")

model = SAC.load(CHECKPOINT_PATH, device="cpu")
print(f"  Model loaded ✓")

if opponent_mode == "self":
    antagonist = DeterministicModelWrapper(model)
    print(f"  Blue team  : same model (deterministic) — self-play")
else:
    antagonist = RandomAntagonist(action_size=8)
    print(f"  Blue team  : random actions")

print(f"  Running {NUM_EPISODES} deterministic episodes")
print("  Close the window or Ctrl+C to stop.")
print("=" * 60)

env = FoosballEnv(antagonist_model=antagonist)

goals_scored   = 0
goals_conceded = 0
episode_rewards = []

try:
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        max_y = -999.0
        min_y = 999.0

        print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")

        while not done:
            # Deterministic action from trained SAC policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            ball_y = env.data.body(env._ball_bid).xpos[1]
            max_y = max(max_y, ball_y)
            min_y = min(min_y, ball_y)

            env.render()

        episode_rewards.append(total_reward)

        outcome = ""
        if ball_y > 65:
            outcome = "  ⚽ GOAL SCORED!"
            goals_scored += 1
        elif ball_y < -65:
            outcome = "  ❌ GOAL CONCEDED"
            goals_conceded += 1
        else:
            outcome = "  ─ truncated"

        print(f"  Steps: {steps}  |  Reward: {total_reward:+.1f}{outcome}")
        print(f"  Ball Y range: [{min_y:.1f}, {max_y:.1f}]")

    mean_r = sum(episode_rewards) / len(episode_rewards)
    print("\n" + "=" * 60)
    print(f"  Results over {NUM_EPISODES} episodes")
    print(f"  Goals scored   : {goals_scored}")
    print(f"  Goals conceded : {goals_conceded}")
    print(f"  Mean reward    : {mean_r:+.1f}")
    print(f"  Min  reward    : {min(episode_rewards):+.1f}")
    print(f"  Max  reward    : {max(episode_rewards):+.1f}")
    print("=" * 60)

except KeyboardInterrupt:
    print("\n\nStopped by user.")
finally:
    env.close()
