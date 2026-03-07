#!/usr/bin/env python3
"""
Visualize a trained SAC model playing foosball.
Loads the best saved checkpoint and runs it against a random antagonist
with GLFW rendering.

Usage:
    python3 visualize_trained_model.py                         # uses default model path
    python3 visualize_trained_model.py /path/to/model.zip      # uses custom model path
"""
import sys
import os
import time
import numpy as np

# ── NumPy compatibility shim ───────────────────────────────────────────────────
# Models saved on Kaggle (NumPy 2.x) reference numpy._core which doesn't exist
# in NumPy 1.x.  Create an alias so cloudpickle can deserialise the checkpoint.
if not hasattr(np, "_core"):
    import numpy.core as _np_core
    np._core = _np_core
    sys.modules["numpy._core"] = _np_core
    # Also alias sub-modules that cloudpickle may reference
    for _sub in ("numeric", "multiarray", "_multiarray_umath", "fromnumeric"):
        _full = f"numpy._core.{_sub}"
        if _full not in sys.modules:
            _old = f"numpy.core.{_sub}"
            try:
                __import__(_old)
                sys.modules[_full] = sys.modules[_old]
            except ImportError:
                pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "0", "sac", "best_model", "model.zip"
)
NUM_EPISODES = 10
RENDER_DELAY = 0.0  # seconds per step (0 = as fast as possible)


class RandomAntagonist:
    """Dummy model that returns random actions in [-1,1], matching SB3 API.
    The env scales these to ±20 internally so blue has equal control authority."""
    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, obs, **kwargs):
        action = np.random.uniform(-1.0, 1.0, size=self.action_size).astype(np.float32)
        return action, None


def main():
    # ── Load model ─────────────────────────────────────────────────────────────
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Train a model first (e.g. via the Kaggle notebook) and place it in models/0/sac/best_model/model.zip")
        sys.exit(1)

    print("=" * 60)
    print("  Foosball — Trained Model Visualization")
    print("=" * 60)
    print(f"  Model        : {model_path}")
    print(f"  Episodes     : {NUM_EPISODES}")
    print(f"  Protagonist  : Trained SAC (deterministic)")
    print(f"  Antagonist   : Random actions")
    print()
    print("  Close the window or press Ctrl+C to stop.")
    print("=" * 60)

    model = SAC.load(model_path)
    print(f"  Model loaded ✓")

    # ── Create env with random antagonist ──────────────────────────────────────
    antagonist = RandomAntagonist(action_size=8)
    env = FoosballEnv(antagonist_model=antagonist)

    # ── Run episodes ───────────────────────────────────────────────────────────
    stats = {"goals_for": 0, "goals_against": 0, "total_reward": 0.0, "total_steps": 0}

    try:
        for episode in range(NUM_EPISODES):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0

            print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")

            while not done:
                # Trained agent picks actions deterministically
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                steps += 1

                env.render()
                if RENDER_DELAY > 0:
                    time.sleep(RENDER_DELAY)

            # ── Detect outcome ─────────────────────────────────────────────────
            ball_world_y = env.data.body(env._ball_bid).xpos[1]
            ball_world_x = env.data.body(env._ball_bid).xpos[0]
            in_goal = abs(ball_world_x) < 15

            if ball_world_y > 65 and in_goal:
                outcome = "⚽ GOAL SCORED (yellow → blue goal)"
                stats["goals_for"] += 1
            elif ball_world_y < -65 and in_goal:
                outcome = "❌ Goal conceded (blue → yellow goal)"
                stats["goals_against"] += 1
            elif steps >= 1500:
                outcome = "⏱  Time limit reached"
            else:
                outcome = "🔄 Episode ended (stagnant/other)"

            stats["total_reward"] += ep_reward
            stats["total_steps"] += steps

            print(f"  Steps: {steps}  |  Reward: {ep_reward:+.1f}  |  {outcome}")

        # ── Summary ────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  Summary")
        print("=" * 60)
        print(f"  Episodes        : {NUM_EPISODES}")
        print(f"  Goals scored    : {stats['goals_for']}")
        print(f"  Goals conceded  : {stats['goals_against']}")
        print(f"  Mean reward     : {stats['total_reward'] / NUM_EPISODES:+.1f}")
        print(f"  Mean steps/ep   : {stats['total_steps'] / NUM_EPISODES:.0f}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
