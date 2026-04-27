import os
import logging
import numpy as np
from typing import List, Optional
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from src.environment import ForexEnv

logger = logging.getLogger("rl_agent")

class RLAgent:
    """
    Deep Reinforcement Learning Agent using Proximal Policy Optimization (PPO).
    Wraps the stable-baselines3 model and the vectorized custom gym environment.
    """
    
    def __init__(self, model_path: str = "models/ppo_agent.zip", device: str = "auto"):
        self.model_path = model_path
        self.model = None
        self.device = device
        
    def _make_env(self, df: pd.DataFrame, spread: float = 0.0001, window_size: int = 60):
        def _init():
            return ForexEnv(df=df, spread=spread, window_size=window_size)
        return _init

    def train(self, data_dict: dict, total_timesteps: int = 1_000_000, n_envs: int = 4):
        """
        Train the PPO agent on multiple instruments concurrently using vectorized environments.
        
        Parameters:
        - data_dict: dict mapping instrument names to their feature DataFrames
        - total_timesteps: Total number of steps to simulate across all envs
        - n_envs: Number of parallel environments (matches CPU cores on Kaggle)
        """
        if not data_dict:
            logger.error("No data provided for RL training.")
            return

        # Combine all dfs into one large list for random env selection, or just use the first for simplicity
        # In a full implementation, you'd multiplex the instruments
        # For this prototype, we'll use the combined continuous dataframe or just pick the largest one
        
        first_inst = list(data_dict.keys())[0]
        df = data_dict[first_inst]
        
        logger.info(f"Setting up {n_envs} vectorized environments...")
        # Use SubprocVecEnv for true multiprocessing on Kaggle CPUs
        vec_env = SubprocVecEnv([self._make_env(df) for _ in range(n_envs)])
        
        logger.info("Initializing PPO Neural Network (Actor-Critic)...")
        self.model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            device=self.device
        )
        
        logger.info(f"Starting heavy RL simulation ({total_timesteps:,} steps)...")
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save the .zip containing the PyTorch .pt weights
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        logger.info(f"RL PPO Agent saved to {self.model_path}")

    def load(self):
        """Load the pre-trained PPO model from disk."""
        if os.path.exists(self.model_path) or os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path, device=self.device)
            logger.info(f"Loaded RL Agent from {self.model_path}")
            return True
        logger.warning(f"Could not find RL Agent at {self.model_path}")
        return False

    def predict(self, obs: np.ndarray) -> int:
        """
        Predict the best action given the current environment observation.
        Returns the action (0: Hold, 1: Buy, 2: Sell, 3: Close)
        """
        if self.model is None:
            return 0
            
        action, _states = self.model.predict(obs, deterministic=True)
        return int(action)
