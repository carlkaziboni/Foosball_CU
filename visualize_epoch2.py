from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
from stable_baselines3 import SAC
import time

print("=" * 60)
print("Quick Visualization - Epoch 4 Model")
print("=" * 60)

# Load environment with rendering
env = FoosballEnv(antagonist_model=None)

# Load epoch 4 model
model_path = "./models/0/sac/epoch_4/model.zip"
try:
    model = SAC.load(model_path)
    print(f"✓ Model loaded: {model_path}\n")
except FileNotFoundError:
    print(f"✗ Model not found: {model_path}")
    print("Available models:")
    import os
    for root, dirs, files in os.walk("./models"):
        for file in files:
            if file.endswith(".zip"):
                print(f"  - {os.path.join(root, file)}")
    exit(1)

# Play 3 full-length episodes (no step limit)
for episode in range(1, 4):
    print(f"--- Episode {episode}/3 ---")
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:  # Play until episode naturally ends
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        env.render()
        time.sleep(0.01)  # Slight delay for visibility
    
    print(f"  Reward: {total_reward:.2f}, Steps: {steps}\n")

env.close()
print("=" * 60)
print("Visualization complete!")
print("=" * 60)
