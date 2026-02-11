from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
import numpy as np

env = FoosballEnv()
obs, info = env.reset()

print('Testing MASSIVELY INCREASED rewards...')
print('='*60)

# Test with random actions to see reward range
rewards = []
for i in range(50):
    action = np.random.uniform(-1, 1, 8)
    obs, reward, term, trunc, info = env.step(action)
    rewards.append(reward)
    if reward > 10:
        print(f'Step {i+1}: reward = {reward:.2f} (POSITIVE!)')
    if term:
        break

avg_reward = np.mean(rewards)
max_reward = np.max(rewards)
min_reward = np.min(rewards)
positive_count = sum(1 for r in rewards if r > 0)

print(f'\n50 random steps:')
print(f'  Average reward: {avg_reward:.2f}')
print(f'  Max reward: {max_reward:.2f}')
print(f'  Min reward: {min_reward:.2f}')
print(f'  Positive rewards: {positive_count}/50')
print('='*60)
print('✓ New reward scale should be 5-10x stronger!')
print('\nNow ready for training with:')
print('  - 3x higher learning rate (0.001)')
print('  - State-dependent exploration (SDE)')
print('  - 10x stronger curriculum rewards')
print('  - Earlier learning start (500 steps)')
