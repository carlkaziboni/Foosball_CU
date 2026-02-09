from stable_baselines3 import SAC
from ai_agents.common.train.interface.foosball_agent import FoosballAgent
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from ai_agents.common.train.impl.tensorboard_callback import DetailedTensorboardCallback, FoosballMonitorCallback
import torch


class SACFoosballAgent(FoosballAgent):
    def __init__(self, id:int, env=None, log_dir='./logs', model_dir='./models', policy_kwargs = dict(net_arch=[3000, 3000, 3000, 3000, 3000, 3000, 3000])):
        self.env = env
        self.model = None
        self.id = id
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.id_subdir = f'{model_dir}/{id}'
        self.policy_kwargs = policy_kwargs
        
        # Determine best available device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(f"✓ Agent {id} using Apple Silicon GPU (MPS) for training")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"✓ Agent {id} using NVIDIA GPU (CUDA) for training")
        else:
            self.device = "cpu"
            print(f"Agent {id} using CPU for training")

    def get_id(self):
        return self.id

    def initialize_agent(self):
        try:
            self.load()
        except Exception as e:
            print(f"Agent {self.id} could not load model. Initializing new model.")
            self.model = SAC('MlpPolicy', self.env, policy_kwargs=self.policy_kwargs, device='cuda', buffer_size=1000000)
        print(f"Agent {self.id} initialized.")

    def predict(self, observation, deterministic=False):
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def learn(self, total_timesteps):
        if self.model is None:
            # Enable TensorBoard logging with descriptive name
            tensorboard_log = f"./tensorboard_logs/sac_agent_{self.id}"
            self.model = SAC(
                'MlpPolicy', 
                self.env, 
                policy_kwargs=self.policy_kwargs, 
                device=self.device, 
                buffer_size=1000000,
                tensorboard_log=tensorboard_log,
                verbose=1
            )
        
        callback = self.create_callback(self.env)
        tb_log_name = f'run_{self.id}'
        print(f"Agent {self.id} starting learning for {total_timesteps} timesteps...")
        print(f"TensorBoard logs: ./tensorboard_logs/sac_agent_{self.id}/{tb_log_name}")
        self.model.learn(
            total_timesteps=total_timesteps, 
            callback=callback, 
            tb_log_name=tb_log_name, 
            progress_bar=True,
            log_interval=10  # Log every 10 episodes
        )
        print(f"Agent {self.id} finished learning!")

    def create_callback(self, env):
        # Evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=self.id_subdir + '/sac/best_model',
            log_path=self.log_dir,
            eval_freq=5000,
            n_eval_episodes=3,
            render=False,
            deterministic=True,
            verbose=1,
        )
        
        # Custom detailed logging callback
        tensorboard_callback = DetailedTensorboardCallback(verbose=1)
        
        # Foosball-specific monitoring callback
        monitor_callback = FoosballMonitorCallback(env, verbose=1)
        
        # Combine all callbacks
        callback_list = CallbackList([eval_callback, tensorboard_callback, monitor_callback])
        
        return callback_list

    def save(self):
        self.model.save(self.id_subdir + '/sac/best_model')
    
    def save_checkpoint(self, path):
        """Save model to a specific checkpoint path"""
        import os
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/model")
        print(f"  Checkpoint saved to: {path}/model.zip")

    def load(self):
        self.model = SAC.load(self.id_subdir + '/sac/best_model/best_model.zip')
        print(f"Agent {self.id} loaded model from {self.id_subdir}/sac/best_model/best_model.zip")

    def change_env(self, env):
        self.env = env
        self.model.set_env(env)