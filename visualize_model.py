from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from stable_baselines3 import SAC
import time
import sys

# Load the environment with rendering
env = FoosballEnv(antagonist_model=None)

# Try to load the most recent model
model_paths = [
    "./models/0/sac/best_model.zip",           # Latest best model
    "./models/0/sac/epoch_2/model.zip",        # Latest epoch
    "./models/0/sac/best_model/best_model.zip" # Old path
]

model = None
model_path = None

for path in model_paths:
    try:
        model = SAC.load(path)
        model_path = path
        print(f"✓ Model loaded successfully from {path}")
        break
    except Exception as e:
        continue

if model is None:
    print(f"✗ Error: Could not load any model")
    print("Available paths tried:")
    for path in model_paths:
        print(f"  - {path}")
    print("\nMake sure you have trained a model first by running: python3 sac_agent_entry_v2.py")
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
