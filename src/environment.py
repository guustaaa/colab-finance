import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, Dict

@struct.dataclass
class EnvState:
    current_step: jnp.ndarray
    balance: jnp.ndarray
    net_worth: jnp.ndarray
    max_net_worth: jnp.ndarray
    position: jnp.ndarray
    entry_price: jnp.ndarray
    units: jnp.ndarray
    total_trades: jnp.ndarray
    winning_trades: jnp.ndarray

class JaxForexEnv:
    def __init__(self, data_matrix: jnp.ndarray, initial_balance: float = 1000.0, spread: float = 0.0001, window_size: int = 30):
        self.data_matrix = data_matrix
        self.initial_balance = initial_balance
        self.spread = spread
        self.window_size = window_size
        self.num_features = self.data_matrix.shape[1]
        self.max_steps = self.data_matrix.shape[0] - 1

    def reset(self) -> Tuple[jnp.ndarray, EnvState]:
        state = EnvState(
            current_step=jnp.array(self.window_size, dtype=jnp.int32),
            balance=jnp.array(self.initial_balance, dtype=jnp.float32),
            net_worth=jnp.array(self.initial_balance, dtype=jnp.float32),
            max_net_worth=jnp.array(self.initial_balance, dtype=jnp.float32),
            position=jnp.array(0, dtype=jnp.int32),
            entry_price=jnp.array(0.0, dtype=jnp.float32),
            units=jnp.array(0.0, dtype=jnp.float32),
            total_trades=jnp.array(0, dtype=jnp.int32),
            winning_trades=jnp.array(0, dtype=jnp.int32)
        )
        return self._get_obs(state), state

    def _get_obs(self, state: EnvState) -> jnp.ndarray:
        start_idx = state.current_step - self.window_size
        window = jax.lax.dynamic_slice(self.data_matrix, (start_idx, 0), (self.window_size, self.num_features))
        window_flat = window.flatten()

        w_mean = jnp.mean(window_flat)
        w_std = jnp.std(window_flat) + 1e-8
        window_flat = (window_flat - w_mean) / w_std

        current_price = self.data_matrix[state.current_step, 3] 
        
        unrealized_pnl_long = (current_price - state.entry_price) * state.units
        unrealized_pnl_short = (state.entry_price - current_price) * state.units
        
        unrealized_pnl = jnp.where(
            state.position == 1, unrealized_pnl_long,
            jnp.where(state.position == -1, unrealized_pnl_short, 0.0)
        )

        account_state = jnp.array([
            state.balance / self.initial_balance,
            state.position.astype(jnp.float32),
            unrealized_pnl / self.initial_balance
        ], dtype=jnp.float32)

        return jnp.concatenate((window_flat, account_state))

    def step(self, state: EnvState, action: jnp.ndarray) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        next_step = state.current_step + 1
        done = next_step >= self.max_steps

        current_price = self.data_matrix[state.current_step, 3]
        prev_price = self.data_matrix[state.current_step - 1, 3]

        step_pnl_long = (current_price - prev_price) * state.units
        step_pnl_short = (prev_price - current_price) * state.units
        step_pnl = jnp.where(state.position == 1, step_pnl_long, jnp.where(state.position == -1, step_pnl_short, 0.0))
        
        new_net_worth = state.net_worth + step_pnl

        close_long_pnl = (current_price - state.entry_price) * state.units
        close_short_pnl = (state.entry_price - current_price) * state.units

        def do_buy(args):
            b, p, ep, u, tp, tt, wt, real_pnl = args
            is_reversal = (p == -1)
            real_pnl = jnp.where(is_reversal, close_short_pnl, 0.0)
            b = b + real_pnl
            tt = tt + jnp.where(is_reversal, 1, 0)
            wt = wt + jnp.where(is_reversal & (real_pnl > 0), 1, 0)
            
            p = 1
            ep = current_price + (self.spread / 2)
            u = (b * 0.05) / (ep * 0.01) # Increased position sizing
            tp = -(self.spread * u)
            return b, p, ep, u, tp, tt, wt, real_pnl

        def do_sell(args):
            b, p, ep, u, tp, tt, wt, real_pnl = args
            is_reversal = (p == 1)
            real_pnl = jnp.where(is_reversal, close_long_pnl, 0.0)
            b = b + real_pnl
            tt = tt + jnp.where(is_reversal, 1, 0)
            wt = wt + jnp.where(is_reversal & (real_pnl > 0), 1, 0)
            
            p = -1
            ep = current_price - (self.spread / 2)
            u = (b * 0.05) / (ep * 0.01)
            tp = -(self.spread * u)
            return b, p, ep, u, tp, tt, wt, real_pnl

        def do_close(args):
            b, p, ep, u, tp, tt, wt, real_pnl = args
            is_closing_long = (p == 1)
            is_closing_short = (p == -1)
            real_pnl = jnp.where(is_closing_long, close_long_pnl, jnp.where(is_closing_short, close_short_pnl, 0.0))
            
            b = b + real_pnl
            is_closing = is_closing_long | is_closing_short
            tt = tt + jnp.where(is_closing, 1, 0)
            wt = wt + jnp.where(is_closing & (real_pnl > 0), 1, 0)
            
            p = 0
            ep = 0.0
            u = 0.0
            return b, p, ep, u, tp, tt, wt, real_pnl

        def do_hold(args):
            return args

        args = (state.balance, state.position, state.entry_price, state.units, jnp.array(0.0, dtype=jnp.float32), state.total_trades, state.winning_trades, jnp.array(0.0, dtype=jnp.float32))
        args = jax.lax.switch(action, [do_hold, do_buy, do_sell, do_close], args)
        new_balance, new_position, new_entry_price, new_units, trade_penalty, new_total_trades, new_winning_trades, realized_pnl = args

        new_max_net_worth = jnp.maximum(state.max_net_worth, new_net_worth)

        # -------------------------------------------------------------
        # 🛡️ DYNAMIC VOLATILITY & STRICT REWARDS (Research Point #3)
        # -------------------------------------------------------------
        window_closes = jax.lax.dynamic_slice(self.data_matrix, (state.current_step - self.window_size, 3), (self.window_size, 1))
        volatility = jnp.std(window_closes) / (jnp.mean(window_closes) + 1e-8)

        trade_incentive = jnp.where(
            realized_pnl > 0, 
            realized_pnl * 20.0, # Massive multiplier for taking profit
            jnp.where(realized_pnl < 0, (realized_pnl * 20.0) - (0.01 * self.initial_balance), 0.0) # Flat penalty for losing
        )
        
        base_reward = step_pnl + trade_penalty + trade_incentive
        
        # Scale reward by current market volatility to prevent hyper-active trading during chaos
        reward = base_reward / (1.0 + (volatility * 100.0))
        
        drawdown = (new_max_net_worth - new_net_worth) / new_max_net_worth
        reward = jnp.where(drawdown > 0.10, reward - (drawdown * 20), reward)
        
        ruin = new_net_worth <= (self.initial_balance * 0.5)
        reward = jnp.where(ruin, reward - 1000.0, reward)
        done = jnp.logical_or(done, ruin)
        
        reward = reward / self.initial_balance

        new_state = EnvState(
            current_step=next_step,
            balance=new_balance,
            net_worth=new_net_worth,
            max_net_worth=new_max_net_worth,
            position=new_position,
            entry_price=new_entry_price,
            units=new_units,
            total_trades=new_total_trades,
            winning_trades=new_winning_trades
        )
        
        obs = self._get_obs(new_state)
        info = {"net_worth": new_net_worth, "drawdown": drawdown, "position": new_position}

        return obs, new_state, reward, done, info