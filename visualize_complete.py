"""
Complete Visualization Suite for Foosball Training Results
Shows trained models, statistics, and performance metrics
"""
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from stable_baselines3 import SAC
import time
import sys
import os
import glob
import numpy as np
from pathlib import Path


def find_all_checkpoints():
    """Find all saved model checkpoints"""
    checkpoints = {}
    model_dir = "./models/0/sac/"
    
    # Find best model
    best_model = os.path.join(model_dir, "best_model", "best_model.zip")
    if os.path.exists(best_model):
        checkpoints['best_model'] = best_model
    
    # Find milestone checkpoints
    for checkpoint_dir in glob.glob(os.path.join(model_dir, "checkpoint_*")):
        name = os.path.basename(checkpoint_dir)
        model_path = os.path.join(checkpoint_dir, "model.zip")
        if os.path.exists(model_path):
            checkpoints[name] = model_path
    
    # Find epoch checkpoints
    for epoch_dir in glob.glob(os.path.join(model_dir, "epoch_*")):
        name = os.path.basename(epoch_dir)
        model_path = os.path.join(epoch_dir, "model.zip")
        if os.path.exists(model_path):
            checkpoints[name] = model_path
    
    return checkpoints


def evaluate_model(model, env, num_episodes=10, render=False):
    """Evaluate a model and return statistics"""
    episode_rewards = []
    episode_lengths = []
    goals_scored = 0
    goals_conceded = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Track goals (large positive/negative rewards)
            if reward > 500:
                goals_scored += 1
            elif reward < -500:
                goals_conceded += 1
            
            if render:
                env.render()
                time.sleep(0.01)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'goal_ratio': goals_scored / max(1, goals_scored + goals_conceded)
    }


def visualize_best_model(num_episodes=5):
    """Visualize the best trained model"""
    print("\n" + "="*70)
    print("🏆 VISUALIZING BEST MODEL")
    print("="*70)
    
    env = FoosballEnv(antagonist_model=None)
    model_path = "./models/0/sac/best_model/best_model.zip"
    
    try:
        model = SAC.load(model_path)
        print(f"✓ Loaded best model from {model_path}\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(0.02)
        
        print(f"  Reward: {total_reward:.2f} | Steps: {steps}")
    
    env.close()
    print("\n✓ Visualization complete!")


def compare_checkpoints():
    """Compare performance across all checkpoints"""
    print("\n" + "="*70)
    print("📊 COMPARING ALL CHECKPOINTS")
    print("="*70)
    
    checkpoints = find_all_checkpoints()
    
    if not checkpoints:
        print("✗ No checkpoints found!")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoint(s):\n")
    
    env = FoosballEnv(antagonist_model=None)
    results = {}
    
    for name, path in sorted(checkpoints.items()):
        print(f"Evaluating {name}...", end=' ')
        try:
            model = SAC.load(path)
            stats = evaluate_model(model, env, num_episodes=5, render=False)
            results[name] = stats
            print(f"✓ Mean Reward: {stats['mean_reward']:.2f}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    env.close()
    
    # Print comparison table
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    print(f"{'Checkpoint':<20} {'Mean Reward':<15} {'Episode Len':<15} {'Goals':<10}")
    print("-"*70)
    
    for name in sorted(results.keys()):
        stats = results[name]
        goals = f"{stats['goals_scored']}/{stats['goals_conceded']}"
        print(f"{name:<20} {stats['mean_reward']:<15.2f} {stats['mean_length']:<15.1f} {goals:<10}")
    
    # Find best performing model
    best_checkpoint = max(results.items(), key=lambda x: x[1]['mean_reward'])
    print("\n" + "="*70)
    print(f"🏆 Best Checkpoint: {best_checkpoint[0]}")
    print(f"   Mean Reward: {best_checkpoint[1]['mean_reward']:.2f}")
    print(f"   Goal Ratio: {best_checkpoint[1]['goal_ratio']*100:.1f}%")
    print("="*70)


def show_training_summary():
    """Show summary of training run"""
    print("\n" + "="*70)
    print("📈 TRAINING SUMMARY")
    print("="*70)
    
    # Check for logs
    log_dir = "./logs/"
    if os.path.exists(log_dir):
        print(f"\n✓ Logs directory: {log_dir}")
        print("  Use: tensorboard --logdir ./logs/")
    
    # Check for checkpoints
    checkpoints = find_all_checkpoints()
    print(f"\n✓ Saved checkpoints: {len(checkpoints)}")
    for name in sorted(checkpoints.keys()):
        print(f"  - {name}")
    
    print("\n" + "="*70)


def main():
    print("\n" + "="*70)
    print("🎮 FOOSBALL AI - COMPLETE VISUALIZATION SUITE")
    print("="*70)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nSelect visualization mode:")
        print("  1. Watch best model play (with rendering)")
        print("  2. Compare all checkpoints (no rendering)")
        print("  3. Show training summary")
        print("  4. Full analysis (summary + comparison + visualization)")
        
        choice = input("\nEnter choice (1-4): ").strip()
        mode = {'1': 'watch', '2': 'compare', '3': 'summary', '4': 'full'}.get(choice, 'full')
    
    try:
        if mode == 'watch' or mode == 'full':
            visualize_best_model(num_episodes=5)
        
        if mode == 'compare' or mode == 'full':
            compare_checkpoints()
        
        if mode == 'summary' or mode == 'full':
            show_training_summary()
        
    except KeyboardInterrupt:
        print("\n\n✗ Visualization interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
