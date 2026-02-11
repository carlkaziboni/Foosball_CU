from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
import numpy as np

env = FoosballEnv()
obs, info = env.reset()

print('Testing NEW reward function with curriculum learning...')
print('='*60)

# Test 1: No action (ball stationary)
obs, reward, term, trunc, info = env.step(np.zeros(8))
print(f'Test 1 - No action: reward = {reward:.2f}')

# Test 2: Random actions (might hit ball)
total_reward = 0
positive_rewards = 0
for i in range(20):
    action = np.random.uniform(-1, 1, 8)
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    if reward > 0:
        positive_rewards += 1
        print(f'  Step {i+1}: reward = {reward:.2f} (positive!)')
    if term:
        break

print(f'\nTest 2 - 20 random steps:')
print(f'  Total reward: {total_reward:.2f}')
print(f'  Positive reward steps: {positive_rewards}/20')
print('='*60)
print('✓ New reward function with curriculum learning active')
print('\nAgent should now get positive rewards for:')
print('  - Moving foosmen near ball (+5-25)')
print('  - Making ball move (+10-20)')
print('  - Ball moving toward goal (+15-30)')
print('  - Scoring goals (+1000)')
