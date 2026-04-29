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
        
        # Independent Column-Wise Z-Score Normalization
        w_mean = jnp.mean(window, axis=0)
        w_std = jnp.std(window, axis=0) + 1e-8
        window_norm = (window - w_mean) / w_std
        window_flat = window_norm.flatten()

        current_mid = self.data_matrix[state.current_step, 3] # 'close' column
        ask = current_mid + (self.spread / 2.0)
        bid = current_mid - (self.spread / 2.0)
        
        pnl_long = (bid - state.entry_price) * state.units
        pnl_short = (state.entry_price - ask) * state.units
        
        unrealized_pnl = jnp.where(
            state.position == 1, pnl_long,
            jnp.where(state.position == -1, pnl_short, 0.0)
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

        current_mid = self.data_matrix[state.current_step, 3]
        ask = current_mid + (self.spread / 2.0)
        bid = current_mid - (self.spread / 2.0)

        pnl_long = (bid - state.entry_price) * state.units
        pnl_short = (state.entry_price - ask) * state.units
        current_unrealized = jnp.where(state.position == 1, pnl_long, jnp.where(state.position == -1, pnl_short, 0.0))
        current_net_worth = state.balance + current_unrealized

        def do_buy(args):
            b, p, ep, u, tt, wt = args
            is_rev = (p == -1)
            realized = jnp.where(is_rev, pnl_short, 0.0)
            b = b + realized
            tt = tt + jnp.where(is_rev, 1, 0)
            wt = wt + jnp.where(is_rev & (realized > 0), 1, 0)

            p = 1
            ep = ask
            u = (b * 0.05) / (current_mid * 0.01) # 5% Position Sizing
            return b, p, ep, u, tt, wt

        def do_sell(args):
            b, p, ep, u, tt, wt = args
            is_rev = (p == 1)
            realized = jnp.where(is_rev, pnl_long, 0.0)
            b = b + realized
            tt = tt + jnp.where(is_rev, 1, 0)
            wt = wt + jnp.where(is_rev & (realized > 0), 1, 0)

            p = -1
            ep = bid
            u = (b * 0.05) / (current_mid * 0.01)
            return b, p, ep, u, tt, wt

        def do_close(args):
            b, p, ep, u, tt, wt = args
            is_long = (p == 1)
            is_short = (p == -1)
            realized = jnp.where(is_long, pnl_long, jnp.where(is_short, pnl_short, 0.0))
            
            b = b + realized
            is_closing = is_long | is_short
            tt = tt + jnp.where(is_closing, 1, 0)
            wt = wt + jnp.where(is_closing & (realized > 0), 1, 0)

            p = 0
            ep = 0.0
            u = 0.0
            return b, p, ep, u, tt, wt

        def do_hold(args):
            return args

        args = (state.balance, state.position, state.entry_price, state.units, state.total_trades, state.winning_trades)
        args = jax.lax.switch(action, [do_hold, do_buy, do_sell, do_close], args)
        new_balance, new_position, new_entry_price, new_units, new_total_trades, new_winning_trades = args

        new_pnl_long = (bid - new_entry_price) * new_units
        new_pnl_short = (new_entry_price - ask) * new_units
        new_unrealized = jnp.where(new_position == 1, new_pnl_long, jnp.where(new_position == -1, new_pnl_short, 0.0))
        new_net_worth = new_balance + new_unrealized
        new_max_net_worth = jnp.maximum(state.max_net_worth, new_net_worth)

        # -------------------------------------------------------------
        # 🔬 THE PURE MATHEMATICAL REWARD (Logarithmic Returns)
        # -------------------------------------------------------------
        # Log returns symmetrically and beautifully represent portfolio growth
        log_return = jnp.log(new_net_worth / jnp.maximum(current_net_worth, 1.0))
        
        # Strict Transaction Penalty: Punish the agent specifically for switching states (overtrading)
        is_transaction = (new_position != state.position)
        txn_penalty = jnp.where(is_transaction, 0.0005, 0.0) # Represents a 0.05% portfolio hit for trading
        
        # Scale up for neural network gradient visibility
        reward = (log_return - txn_penalty) * 1000.0
        
        # Hard Stop: Account Blown
        ruin = new_net_worth <= (self.initial_balance * 0.5)
        reward = jnp.where(ruin, reward - 100.0, reward)
        done = jnp.logical_or(done, ruin)

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
        info = {"net_worth": new_net_worth, "position": new_position}

        return obs, new_state, reward, done, info