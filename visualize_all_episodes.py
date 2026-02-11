from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from stable_baselines3 import SAC
import time

print("=" * 60)
print("Visualizing All Episodes from 20k Training")
print("=" * 60)

# Load environment with rendering
env = FoosballEnv(antagonist_model=None)

# Load the best model from 20k timestep training
model_path = "./models/0/sac/best_model/model.zip"
try:
    model = SAC.load(model_path)
    print(f"✓ Model loaded: {model_path}")
except:
    # Try alternative path
    model_path = "./models/0/sac/best_model.zip"
    try:
        model = SAC.load(model_path)
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        exit(1)

print("\nPlaying 10 episodes to showcase trained behavior...")
print("Watch the simulation window!")
print("Press Ctrl+C to stop early.\n")

# Play 10 full episodes
episode_stats = []

try:
    for episode in range(1, 11):
        print(f"{'='*60}")
        print(f"Episode {episode}/10")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_ball_y = -999  # Track furthest ball progress
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Track ball progress
            ball_obs = env._get_ball_obs()
            ball_y = ball_obs[0][1]
            max_ball_y = max(max_ball_y, ball_y)
            
            env.render()
            time.sleep(0.01)  # Slight delay for visibility
        
        episode_stats.append({
            'episode': episode,
            'reward': total_reward,
            'steps': steps,
            'max_ball_y': max_ball_y
        })
        
        print(f"Episode {episode} completed:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Max Ball Y Position: {max_ball_y:.2f}")
        print()

except KeyboardInterrupt:
    print("\n\nVisualization stopped by user.")

finally:
    env.close()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Training Summary - 20k Timesteps")
    print("=" * 60)
    
    if episode_stats:
        import numpy as np
        
        rewards = [s['reward'] for s in episode_stats]
        steps_list = [s['steps'] for s in episode_stats]
        max_ys = [s['max_ball_y'] for s in episode_stats]
        
        print(f"\nEpisodes played: {len(episode_stats)}")
        print(f"\nReward Statistics:")
        print(f"  Average: {np.mean(rewards):.2f}")
        print(f"  Std Dev: {np.std(rewards):.2f}")
        print(f"  Min: {np.min(rewards):.2f}")
        print(f"  Max: {np.max(rewards):.2f}")
        
        print(f"\nSteps Statistics:")
        print(f"  Average: {np.mean(steps_list):.1f}")
        print(f"  Min: {np.min(steps_list)}")
        print(f"  Max: {np.max(steps_list)}")
        
        print(f"\nBall Progress:")
        print(f"  Average Max Y: {np.mean(max_ys):.2f}")
        print(f"  Best Progress: {np.max(max_ys):.2f}")
        
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
