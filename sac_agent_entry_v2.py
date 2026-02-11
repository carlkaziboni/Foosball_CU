import os
from ai_agents.common.train.impl.protagonist_antagonist_training_engine import ProtagonistAntagonistTrainingEngine
from ai_agents.common.train.impl.generic_agent_manager import GenericAgentManager
from ai_agents.common.train.impl.sac_agent import SACFoosballAgent
import sys
import argparse
from stable_baselines3.common.monitor import Monitor

from ai_agents.common.train.impl.single_player_training_engine import SinglePlayerTrainingEngine
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


def sac_foosball_env_factory(x=None):
    env = FoosballEnv(antagonist_model=None)
    env = Monitor(env)
    return env

if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Train or test model.')
    argparse.add_argument('-t', '--test', help='Test mode', action='store_true')
    args = argparse.parse_args()


    model_dir = './models'
    log_dir = './logs'
    
    # Overnight training (~12 hours, 20,000 timesteps)
    total_epochs = 10
    epoch_timesteps = int(2000)  # 10 epochs × 2,000 = 20k timesteps
    cycle_timesteps = 500

    agent_manager = GenericAgentManager(1, sac_foosball_env_factory, SACFoosballAgent)
    agent_manager.initialize_training_agents()
    agent_manager.initialize_frozen_best_models()

    engine = SinglePlayerTrainingEngine(
        agent_manager=agent_manager,
        environment_generator=sac_foosball_env_factory
    )

    # Start training
    if not args.test:
        print(f"Starting overnight training (~12 hours)")
        print(f"Configuration: {total_epochs} epochs × {epoch_timesteps} timesteps = {total_epochs * epoch_timesteps:,} total timesteps")
        print("=" * 60)
        engine.train(total_epochs=total_epochs, epoch_timesteps=epoch_timesteps, cycle_timesteps=cycle_timesteps)

    # Test the trained agent
    engine.test()