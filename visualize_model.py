from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from stable_baselines3 import SAC
import time
import sys

# Load the environment with rendering
env = FoosballEnv(antagonist_model=None)

# Load the trained model
model_path = "./models/0/sac/best_model/best_model.zip"

try:
    model = SAC.load(model_path)
    print(f"✓ Model loaded successfully from {model_path}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Make sure you have trained a model first by running: python3 sac_agent_entry_v2.py")
    sys.exit(1)

print("\n" + "="*60)
print("Visualizing Foosball AI - Best Model")
print("="*60)
print("Playing 5 episodes with the best trained model...")
print("Watch the simulation window!")
print("Press Ctrl+C to stop early.")
print("="*60 + "\n")

# Run episodes
try:
    for episode in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1}/5 ---")
        
        while not done:
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Render the simulation
            env.render()
            
            # Slow down for better visibility
            time.sleep(0.02)
        
        print(f"Episode {episode + 1} completed:")
        print(f"  - Total Reward: {total_reward:.2f}")
        print(f"  - Steps: {steps}")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

except KeyboardInterrupt:
    print("\n\nVisualization stopped by user.")

finally:
    env.close()
