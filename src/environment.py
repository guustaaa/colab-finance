import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("rl_env")

class ForexEnv(gym.Env):
    """
    Custom Vectorized Reinforcement Learning Environment for Forex Trading.
    Compatible with stable-baselines3 and gymnasium.
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, df: pd.DataFrame, initial_balance=1000.0, spread=0.0001, window_size=60):
        super(ForexEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.spread = spread
        
        # Determine features size
        # Assuming df contains 'close' and other technical indicators
        # State = [window_size, num_features] + [balance, position, unrealized_pnl]
        self.num_features = len(self.df.columns)
        
        # Action space: 0: Hold, 1: Buy, 2: Sell, 3: Close
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 
        # A 1D array flattened from the window, plus 3 account variables
        obs_dim = (self.window_size * self.num_features) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Environment state variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.position = 0 # 1 = Long, -1 = Short, 0 = Flat
        self.entry_price = 0.0
        self.units = 0.0
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.units = 0.0
        self.current_step = self.window_size
        self.history = []
        return self._get_observation(), {}

    def _get_observation(self):
        # Slice the current window of features
        window = self.df.iloc[self.current_step - self.window_size : self.current_step].values
        window_flat = window.flatten()
        
        # Account state
        unrealized_pnl = 0.0
        if self.position != 0:
            current_price = self.df.loc[self.current_step, 'close']
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) * self.units
            elif self.position == -1:
                unrealized_pnl = (self.entry_price - current_price) * self.units
                
        account_state = np.array([
            self.balance / self.initial_balance,  # normalized balance
            float(self.position), 
            unrealized_pnl / self.initial_balance # normalized PnL
        ], dtype=np.float32)
        
        return np.concatenate((window_flat, account_state))

    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True, False, {"msg": "End of data"}
            
        current_price = self.df.loc[self.current_step, 'close']
        reward = 0.0
        done = False
        
        # Calculate intermediate PnL for holding
        step_pnl = 0.0
        if self.position == 1:
            step_pnl = (current_price - self.df.loc[self.current_step - 1, 'close']) * self.units
        elif self.position == -1:
            step_pnl = (self.df.loc[self.current_step - 1, 'close'] - current_price) * self.units
            
        self.net_worth += step_pnl
        
        # Execute Action
        trade_penalty = 0.0
        
        if action == 1: # Buy
            if self.position <= 0:
                # Close short if exists
                if self.position == -1:
                    trade_pnl = (self.entry_price - current_price) * self.units
                    self.balance += trade_pnl
                # Open Long
                self.position = 1
                self.entry_price = current_price + (self.spread / 2)
                # Fixed risk sizing for simulation (e.g. risk 2% of balance)
                self.units = (self.balance * 0.02) / (self.entry_price * 0.01) # simple synthetic leverage
                trade_penalty = - (self.spread * self.units) # spread cost
                
        elif action == 2: # Sell
            if self.position >= 0:
                # Close long if exists
                if self.position == 1:
                    trade_pnl = (current_price - self.entry_price) * self.units
                    self.balance += trade_pnl
                # Open Short
                self.position = -1
                self.entry_price = current_price - (self.spread / 2)
                self.units = (self.balance * 0.02) / (self.entry_price * 0.01)
                trade_penalty = - (self.spread * self.units)
                
        elif action == 3: # Close
            if self.position == 1:
                trade_pnl = (current_price - self.entry_price) * self.units
                self.balance += trade_pnl
            elif self.position == -1:
                trade_pnl = (self.entry_price - current_price) * self.units
                self.balance += trade_pnl
            self.position = 0
            self.units = 0.0

        # Update Net Worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            
        # ── REWARD SHAPING ──
        # 1. Base step reward is the step PnL
        reward = step_pnl 
        
        # 2. Add transaction costs
        reward += trade_penalty
        
        # 3. Drawdown punishment
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if drawdown > 0.10:
            reward -= (drawdown * 10) # Heavy penalty for >10% DD
            
        # 4. Ruin condition
        if self.net_worth <= self.initial_balance * 0.5:
            reward -= 1000.0 # Game over penalty
            done = True
            
        # Normalize reward for PPO stability
        reward = reward / self.initial_balance 
            
        obs = self._get_observation()
        info = {
            "net_worth": self.net_worth,
            "drawdown": drawdown,
            "position": self.position
        }
        
        return obs, reward, done, False, info
