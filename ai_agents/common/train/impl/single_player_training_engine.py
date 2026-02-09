from ai_agents.common.train.interface.agent_manager import AgentManager
from typing import List
from ai_agents.common.train.interface.training_engine import TrainingEngine

class SinglePlayerTrainingEngine(TrainingEngine):
    def __init__(
            self,
            agent_manager: AgentManager,
            environment_generator

    ):
        self.agent_manager = agent_manager
        self.current_epoch = 0
        self.best_models: List[str] = []
        self.num_agents_training = len(self.agent_manager.get_training_agents())
        self.environment_generator = environment_generator

    def train(self, total_epochs: int, epoch_timesteps: int, cycle_timesteps: int):
        total_timesteps = 0
        checkpoint_intervals = [100000, 250000, 500000, 750000, 1000000, 1250000, 1500000]  # Save at these timesteps
        
        for epoch in range(total_epochs):
            print(f"\n{'='*60}")
            print(f"Starting epoch {epoch + 1}/{total_epochs}")
            print(f"Training for {epoch_timesteps} timesteps")
            print(f"Total timesteps so far: {total_timesteps:,}")
            print(f"{'='*60}\n")
            
            protagonist_agents = self.agent_manager.get_training_agents()
            self.agent_manager.initialize_frozen_best_models()
            antagonist_agents = self.agent_manager.get_frozen_best_models()

            protagonist_agent = protagonist_agents[0]
            env = self.environment_generator()
            protagonist_agent.change_env(env)
            protagonist_agent.learn(epoch_timesteps)
            
            total_timesteps += epoch_timesteps
            
            print(f"\n✓ Epoch {epoch + 1}/{total_epochs} completed!")
            print(f"Total timesteps: {total_timesteps:,}")
            
            # Always save the latest model (overwrites best_model)
            protagonist_agent.save()
            print(f"✓ Latest model saved to {protagonist_agent.model_dir}/{protagonist_agent.id}/sac/best_model")
            
            # Save checkpoint at specific milestones
            for checkpoint in checkpoint_intervals:
                if total_timesteps >= checkpoint and (total_timesteps - epoch_timesteps) < checkpoint:
                    checkpoint_path = f"{protagonist_agent.model_dir}/{protagonist_agent.id}/sac/checkpoint_{checkpoint//1000}k"
                    protagonist_agent.save_checkpoint(checkpoint_path)
                    print(f"🎯 Milestone checkpoint saved: {checkpoint//1000}k timesteps → {checkpoint_path}")
            
            # Save periodic checkpoints every 10 epochs
            if (epoch + 1) % 10 == 0:
                periodic_path = f"{protagonist_agent.model_dir}/{protagonist_agent.id}/sac/epoch_{epoch+1}"
                protagonist_agent.save_checkpoint(periodic_path)
                print(f"💾 Periodic checkpoint saved: epoch {epoch+1} → {periodic_path}")
            
            print()
            self.current_epoch += 1


    def test(self, num_episodes: int = 100):
        protagonist = self.agent_manager.get_frozen_best_models()[0]
        env = self.environment_generator(protagonist)

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = protagonist.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                done = terminated or truncated
            print(f"Episode {episode + 1}/{num_episodes} completed.")