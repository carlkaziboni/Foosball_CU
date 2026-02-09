from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class DetailedTensorboardCallback(BaseCallback):
    """
    Custom callback for logging detailed metrics to TensorBoard during training.
    """
    
    def __init__(self, verbose=0):
        super(DetailedTensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.goals_scored = 0
        self.goals_conceded = 0
        
    def _on_step(self) -> bool:
        # Log every step's reward
        if len(self.locals.get('rewards', [])) > 0:
            reward = self.locals['rewards'][0]
            
            # Track goals (reward spikes indicate goals)
            if reward > 500:  # Goal scored
                self.goals_scored += 1
                self.logger.record('foosball/goals_scored', self.goals_scored)
            elif reward < -500:  # Goal conceded
                self.goals_conceded += 1
                self.logger.record('foosball/goals_conceded', self.goals_conceded)
        
        # Log at each training step
        if self.num_timesteps % 100 == 0:
            # Log buffer statistics if available
            if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer.size() > 0:
                self.logger.record('replay/buffer_size', self.model.replay_buffer.size())
            
            # Log learning rate
            if hasattr(self.model, 'learning_rate'):
                self.logger.record('train/learning_rate', self.model.learning_rate)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.
        """
        if len(self.episode_rewards) > 0:
            # Episode statistics
            self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
            self.logger.record('rollout/ep_rew_max', np.max(self.episode_rewards[-100:]))
            self.logger.record('rollout/ep_rew_min', np.min(self.episode_rewards[-100:]))
            self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
            
            # Goal statistics
            if self.episode_count > 0:
                self.logger.record('foosball/goals_per_episode', self.goals_scored / max(1, self.episode_count))
                self.logger.record('foosball/goal_ratio', self.goals_scored / max(1, self.goals_scored + self.goals_conceded))
    
    def _on_training_end(self) -> None:
        """
        Called at the end of training.
        """
        print(f"\n{'='*60}")
        print("Training Summary:")
        print(f"  Total episodes: {self.episode_count}")
        print(f"  Goals scored: {self.goals_scored}")
        print(f"  Goals conceded: {self.goals_conceded}")
        if self.goals_scored + self.goals_conceded > 0:
            win_rate = self.goals_scored / (self.goals_scored + self.goals_conceded) * 100
            print(f"  Goal ratio: {win_rate:.1f}%")
        print(f"{'='*60}\n")


class FoosballMonitorCallback(BaseCallback):
    """
    Monitor foosball-specific metrics during training.
    """
    
    def __init__(self, eval_env, verbose=0):
        super(FoosballMonitorCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # Log custom environment info if available
        if hasattr(self.training_env, 'get_attr'):
            try:
                # Get info from all environments
                infos = self.locals.get('infos', [])
                
                if len(infos) > 0:
                    # Log any custom metrics from the environment
                    for key in infos[0].keys():
                        if key.startswith('foosball_'):
                            values = [info.get(key, 0) for info in infos]
                            self.logger.record(f'custom/{key}', np.mean(values))
            except:
                pass
