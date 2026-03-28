"""
HardwareRunner — runs a trained SAC model and forwards its actions to the
physical foosball table via FoosballSerialBridge.

The MuJoCo sim is used as a "digital twin" observation source: the model
receives sim-derived observations while its slide outputs drive real hardware.
When real sensors become available, override _build_obs() to use them instead.

Usage:
    # Dry run — log commands, no serial ports opened:
    python3 -m servo_control.hardware_runner path/to/model.zip --dry-run

    # Live at 30 Hz:
    python3 -m servo_control.hardware_runner path/to/model.zip --hz 30

    # Slower rate for initial hardware testing:
    python3 -m servo_control.hardware_runner path/to/model.zip --hz 5
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# ── NumPy compatibility shim (matches visualize_trained_model.py) ──────────────
if not hasattr(np, "_core"):
    import numpy.core as _np_core
    np._core = _np_core
    sys.modules["numpy._core"] = _np_core
    for _sub in ("numeric", "multiarray", "_multiarray_umath", "fromnumeric"):
        _full = f"numpy._core.{_sub}"
        if _full not in sys.modules:
            _old = f"numpy.core.{_sub}"
            try:
                __import__(_old)
                sys.modules[_full] = sys.modules[_old]
            except ImportError:
                pass

# Add project root to path so imports resolve regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

from .bridge_config import REDIS_HOST, REDIS_PORT, TARGET_HZ
from .foosball_serial_bridge import FoosballSerialBridge

log = logging.getLogger(__name__)


class RandomAntagonist:
    """Random antagonist — same as in visualize_trained_model.py."""
    def __init__(self, action_size: int = 8):
        self.action_size = action_size

    def predict(self, obs, **kwargs):
        action = np.random.uniform(-1.0, 1.0, size=self.action_size).astype(np.float32)
        return action, None


class HardwareRunner:
    """
    Loads a trained SAC model and runs it against the physical foosball table.

    Args:
        model_path: path to a .zip checkpoint produced by stable-baselines3
        bridge:     FoosballSerialBridge instance (already configured)
        target_hz:  command rate in Hz — how often actions are sent to hardware
        render:     if True, open a MuJoCo GLFW window alongside hardware output
    """

    def __init__(
        self,
        model_path: str,
        bridge: FoosballSerialBridge,
        target_hz: float = TARGET_HZ,
        render: bool = False,
    ) -> None:
        self.model_path = model_path
        self.bridge = bridge
        self.target_hz = target_hz
        self.render = render

    def run(self, num_episodes: int = 10) -> None:
        """Load the model and run num_episodes, sending actions to hardware."""
        if not os.path.exists(self.model_path):
            print(f"ERROR: Model not found at {self.model_path}")
            sys.exit(1)

        print("=" * 60)
        print("  Foosball — Hardware Runner")
        print("=" * 60)
        print(f"  Model      : {self.model_path}")
        print(f"  Episodes   : {num_episodes}")
        print(f"  Rate       : {self.target_hz:.0f} Hz")
        print(f"  Dry run    : {self.bridge._dry_run}")
        print()

        model = SAC.load(self.model_path)
        print("  Model loaded ✓")

        antagonist = RandomAntagonist(action_size=8)
        env = FoosballEnv(antagonist_model=antagonist)
        print("  Environment created ✓")

        self.bridge.open()
        print("  Serial bridge opened ✓\n")

        stats = {"goals_for": 0, "goals_against": 0, "total_reward": 0.0, "total_steps": 0}

        try:
            for episode in range(num_episodes):
                obs, _ = env.reset()
                done = False
                ep_reward = 0.0
                steps = 0
                prev_rot = np.zeros(4, dtype=np.float32)

                # Deadline-based rate limiter: self-corrects for inference jitter
                deadline = time.monotonic()

                print(f"--- Episode {episode + 1}/{num_episodes} ---")

                while not done:
                    # Model inference (deterministic)
                    action, _ = model.predict(obs, deterministic=True)

                    # Send to hardware
                    self.bridge.send_action(action, prev_rot)

                    # Save rotation components (odd indices: 1,3,5,7) for next step
                    prev_rot = action[1::2]

                    # Step sim (produces next observation)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    steps += 1

                    if self.render:
                        env.render()

                    # Rate control — sleep until next deadline
                    deadline += 1.0 / self.target_hz
                    sleep_s = deadline - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                # Episode outcome
                ball_y = env.data.body(env._ball_bid).xpos[1]
                ball_x = env.data.body(env._ball_bid).xpos[0]
                in_goal = abs(ball_x) < 15
                if ball_y > 65 and in_goal:
                    outcome = "GOAL SCORED (yellow → blue goal)"
                    stats["goals_for"] += 1
                elif ball_y < -65 and in_goal:
                    outcome = "Goal conceded (blue → yellow goal)"
                    stats["goals_against"] += 1
                elif steps >= 1500:
                    outcome = "Time limit reached"
                else:
                    outcome = "Episode ended"

                stats["total_reward"] += ep_reward
                stats["total_steps"] += steps
                print(f"  Steps: {steps}  |  Reward: {ep_reward:+.1f}  |  {outcome}\n")

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.bridge.close()
            env.close()

        print("=" * 60)
        print("  Summary")
        print("=" * 60)
        print(f"  Episodes       : {num_episodes}")
        print(f"  Goals scored   : {stats['goals_for']}")
        print(f"  Goals conceded : {stats['goals_against']}")
        print(f"  Mean reward    : {stats['total_reward'] / num_episodes:+.1f}")
        print(f"  Mean steps/ep  : {stats['total_steps'] / num_episodes:.0f}")
        print("=" * 60)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run a trained foosball SAC model on physical hardware."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "0", "sac", "best_model", "model.zip",
        ),
        help="Path to .zip checkpoint (default: models/0/sac/best_model/model.zip)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument(
        "--hz", type=float, default=TARGET_HZ,
        help=f"Command rate in Hz (default: {TARGET_HZ})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log Redis commands without connecting"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Open a MuJoCo GLFW render window alongside hardware output"
    )
    parser.add_argument("--redis-host", default=REDIS_HOST)
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT)
    args = parser.parse_args()

    bridge = FoosballSerialBridge(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        dry_run=args.dry_run,
    )
    runner = HardwareRunner(
        model_path=args.model_path,
        bridge=bridge,
        target_hz=args.hz,
        render=args.render,
    )
    runner.run(num_episodes=args.episodes)
