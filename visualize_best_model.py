from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from stable_baselines3 import SAC
import time
import numpy as np

print("=" * 60)
print("Visualizing Best Model from Aggressive Training")
print("=" * 60)

# Load environment with rendering
env = FoosballEnv(antagonist_model=None)

# Load the best model
model_path = "./models/0/sac/best_model/model.zip"
try:
    model = SAC.load(model_path)
    print(f"✓ Model loaded: {model_path}")
except Exception as e:
    print(f"✗ Could not load model: {e}")
    exit(1)

print("\nPlaying 5 episodes with the best model...")
print("Watch the simulation window!")
print("Press Ctrl+C to stop early.\n")

# Play 5 episodes
episode_stats = []

try:
    for episode in range(1, 6):
        print(f"{'='*60}")
        print(f"Episode {episode}/5")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        contact_rewards = 0
        movement_rewards = 0
        ball_moved = False
        max_ball_speed = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Track what's happening
            ball_obs = env._get_ball_obs()
            ball_vel = ball_obs[1]
            ball_speed = np.sqrt(ball_vel[0]**2 + ball_vel[1]**2)
            max_ball_speed = max(max_ball_speed, ball_speed)
            
            if ball_speed > 0.5:
                ball_moved = True
            
            # Render
            env.render()
            time.sleep(0.01)
        
        episode_stats.append({
            'episode': episode,
            'reward': total_reward,
            'steps': steps,
            'max_speed': max_ball_speed,
            'ball_moved': ball_moved
        })
        
        print(f"Episode {episode} completed:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Max Ball Speed: {max_ball_speed:.2f}")
        print(f"  Ball Moved: {'Yes' if ball_moved else 'No'}")
        print()

except KeyboardInterrupt:
    print("\n\nVisualization stopped by user.")

finally:
    env.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Best Model Performance Summary")
    print("=" * 60)
    
    if episode_stats:
        rewards = [s['reward'] for s in episode_stats]
        speeds = [s['max_speed'] for s in episode_stats]
        moved_count = sum(1 for s in episode_stats if s['ball_moved'])
        
        print(f"\nEpisodes: {len(episode_stats)}")
        print(f"\nRewards:")
        print(f"  Average: {np.mean(rewards):.2f}")
        print(f"  Range: {np.min(rewards):.2f} to {np.max(rewards):.2f}")
        
        print(f"\nBall Interaction:")
        print(f"  Ball moved in {moved_count}/{len(episode_stats)} episodes")
        print(f"  Average max speed: {np.mean(speeds):.2f}")
        print(f"  Best speed: {np.max(speeds):.2f}")
        
        if np.mean(rewards) > -100:
            print("\n✓ Model showing signs of learning!")
        elif moved_count > 0:
            print("\n⚠ Model making some ball contact but needs more training")
        else:
            print("\n✗ Model not interacting with ball yet - needs more training")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
