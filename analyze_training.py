import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
import os

print("Analyzing trained models...")
print("=" * 60)

# Find all available models
model_paths = []
for root, dirs, files in os.walk("./models/0/sac/"):
    for file in files:
        if file.endswith("model.zip"):
            model_paths.append(os.path.join(root, file))

model_paths.sort()
print(f"Found {len(model_paths)} models:\n")

# Test each model
results = []
for model_path in model_paths:
    model_name = model_path.replace("./models/0/sac/", "").replace("/model.zip", "").replace("best_model.zip", "best_model")
    
    try:
        env = FoosballEnv(antagonist_model=None)
        model = SAC.load(model_path)
        
        # Run 10 episodes
        rewards = []
        for _ in range(10):
            obs, info = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            rewards.append(total_reward)
        
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        env.close()
        
        results.append({
            'name': model_name,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'rewards': rewards
        })
        
        print(f"{model_name:20s}: {avg_reward:8.2f} ± {std_reward:6.2f}")
        
    except Exception as e:
        print(f"{model_name:20s}: Error - {e}")

print("\n" + "=" * 60)
print("Performance Progression:")
print("=" * 60)

# Plot results
if results:
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r['name'] for r in results]
    avgs = [r['avg_reward'] for r in results]
    stds = [r['std_reward'] for r in results]
    
    x = np.arange(len(names))
    ax.bar(x, avgs, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Model Checkpoint')
    ax.set_ylabel('Average Reward')
    ax.set_title('Training Progression - Reward per Checkpoint')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progression.png', dpi=150)
    print("\n✓ Chart saved to: training_progression.png")
    plt.show()

print("\nAnalysis complete!")
