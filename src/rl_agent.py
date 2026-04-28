import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["JAX_PLATFORMS"] = "cuda,cpu" 

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import pandas as pd
import logging
import pickle
import time
from functools import partial
from src.environment import JaxForexEnv

logger = logging.getLogger("rl_agent")

# =====================================================================
# 🧠 PST-TRADER WITH MIXED-PRECISION & SEQUENTIAL MINI-BATCHING
# =====================================================================

class Actor(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        
        # Mixed-Precision Execution (bfloat16) to save VRAM
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        init_fn = nn.initializers.orthogonal(np.sqrt(2))
        
        x = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(x)
        x = nn.tanh(x)
        x = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(x)
        x = nn.tanh(x)
        
        # Cast back to float32 for stable probability calculations
        logits = nn.Dense(self.action_dim, dtype=jnp.float32, kernel_init=nn.initializers.orthogonal(0.01))(x)
        return logits

class TwinCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        init_fn = nn.initializers.orthogonal(np.sqrt(2))
        
        # Critic 1 (bfloat16)
        c1 = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(x)
        c1 = nn.relu(c1)
        c1 = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(c1)
        c1 = nn.relu(c1)
        v1 = nn.Dense(1, dtype=jnp.float32, kernel_init=nn.initializers.orthogonal(1.0))(c1)
        
        # Critic 2 (bfloat16)
        c2 = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(x)
        c2 = nn.relu(c2)
        c2 = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(c2)
        c2 = nn.relu(c2)
        v2 = nn.Dense(1, dtype=jnp.float32, kernel_init=nn.initializers.orthogonal(1.0))(c2)
        
        return jnp.squeeze(v1, axis=-1), jnp.squeeze(v2, axis=-1)

class HybridNetwork(nn.Module):
    action_dim: int
    def setup(self):
        self.actor = Actor(action_dim=self.action_dim)
        self.critic = TwinCritic()

    def __call__(self, x):
        logits = self.actor(x)
        v1, v2 = self.critic(x)
        return logits, v1, v2

class RLAgent:
    def __init__(self, model_path: str = "/kaggle/working/ForexAI_State/models/rl_pst_trader_v4.pkl", device: str = "auto"):
        self.model_path = model_path
        self.action_dim = 4
        self.network = HybridNetwork(action_dim=self.action_dim)
        
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=3e-4)
        )
        self.params = None
        
    def train(self, data_dict: dict, total_timesteps: int = 500_000_000):
        # 🛠️ Rebalanced for Dual T4 Stability (1 Million steps per rollout)
        N_ENVS = 8192
        CHUNK_SIZE = 128
        NUM_MINIBATCHES = 16 # Splits the chunk into 16 digestible VRAM blocks
        
        logger.info(f"Aggregating data from {len(data_dict)} currency pairs...")
        combined_df = pd.concat(list(data_dict.values()), ignore_index=True)
        data_matrix = jnp.array(combined_df.values, dtype=jnp.float32)
        
        env = JaxForexEnv(data_matrix=data_matrix)
        num_devices = jax.local_device_count()
        logger.info(f"⚙️ Compute Engine: {num_devices}x GPU | Matrix Size: {N_ENVS}x{CHUNK_SIZE}")
        
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        step_rngs = jax.random.split(rng, num_devices) 
        
        dummy_obs, _ = env.reset()
        self.params = self.network.init(init_rng, dummy_obs)
        opt_state = self.optimizer.init(self.params)
        
        if self.load():
            logger.info("Resuming from safe v4 Checkpoint...")
            
        replicated_params = jax.tree.map(lambda x: jnp.stack([x] * num_devices), self.params)
        replicated_opt_state = jax.tree.map(lambda x: jnp.stack([x] * num_devices), opt_state)

        def huber_loss(x, delta=1.0):
            return jnp.where(jnp.abs(x) < delta, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))

        @partial(jax.pmap, axis_name='p')
        def process_chunk(params, opt_state, step_rngs_mapped, states, obs):
            
            # PHASE 1: CONTINUOUS STREAM ROLLOUT
            def rollout_step(carry, unused):
                carry_rng, current_states, current_obs = carry
                carry_rng, step_rng = jax.random.split(carry_rng)
                
                logits, v1, v2 = self.network.apply(params, current_obs)
                v_curr = jnp.minimum(v1, v2)
                
                actions = jax.random.categorical(step_rng, logits)
                log_probs = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], axis=-1).squeeze(-1)
                
                next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(current_states, actions)
                
                # Iterable Pipeline / Continuous Stream
                def single_reset(s, k):
                    rs = jax.random.randint(k, (), minval=env.window_size, maxval=env.max_steps - 1000)
                    return s.replace(
                        current_step=rs,
                        balance=jnp.array(env.initial_balance, dtype=jnp.float32),
                        net_worth=jnp.array(env.initial_balance, dtype=jnp.float32),
                        max_net_worth=jnp.array(env.initial_balance, dtype=jnp.float32),
                        position=jnp.array(0, dtype=jnp.int32),
                        entry_price=jnp.array(0.0, dtype=jnp.float32),
                        units=jnp.array(0.0, dtype=jnp.float32)
                    )
                
                reset_keys = jax.random.split(step_rng, dones.shape[0])
                reset_states_batch = jax.vmap(single_reset)(next_states, reset_keys)
                
                next_states = jax.tree.map(lambda r, n: jnp.where(dones, r, n), reset_states_batch, next_states)
                next_obs = jax.vmap(env._get_obs)(next_states)
                
                transition = (current_obs, actions, log_probs, rewards, v_curr, dones)
                return (carry_rng, next_states, next_obs), transition

            (step_rngs_mapped, states, next_obs), trajectories = jax.lax.scan(
                rollout_step, (step_rngs_mapped, states, obs), None, length=CHUNK_SIZE
            )
            obs_batch, actions_batch, log_probs_batch, rewards_batch, values_batch, dones_batch = trajectories
            
            # PHASE 2: REVERSE BELLMAN ESTIMATION
            _, next_v1, next_v2 = self.network.apply(params, next_obs)
            next_v = jnp.minimum(next_v1, next_v2)
            
            def compute_returns(carry, transition):
                r, d = transition
                ret = r + 0.99 * carry * (1.0 - d)
                return ret, ret
                
            _, returns_batch = jax.lax.scan(compute_returns, next_v, (rewards_batch, dones_batch), reverse=True)
            
            advantages_batch = returns_batch - values_batch
            adv_mean = jnp.mean(advantages_batch)
            adv_std = jnp.std(advantages_batch) + 1e-8
            norm_advantages_batch = (advantages_batch - adv_mean) / adv_std
            
            # 🛠️ PHASE 3: SEQUENTIAL MINI-BATCHING (Prevents Compiler Hangs & OOM)
            def reshape_to_minibatches(x):
                flat_x = x.reshape(-1, *x.shape[2:])
                mb_size = flat_x.shape[0] // NUM_MINIBATCHES
                return flat_x.reshape(NUM_MINIBATCHES, mb_size, *flat_x.shape[1:])

            mb_obs = reshape_to_minibatches(obs_batch)
            mb_actions = reshape_to_minibatches(actions_batch)
            mb_log_probs = reshape_to_minibatches(log_probs_batch)
            mb_returns = reshape_to_minibatches(returns_batch)
            mb_advs = reshape_to_minibatches(norm_advantages_batch)

            def update_minibatch(carry, batch_data):
                p_carry, opt_carry = carry
                b_obs, b_actions, b_log_probs, b_returns, b_advs = batch_data
                
                def loss_fn(params_to_update):
                    logits, v1, v2 = self.network.apply(params_to_update, b_obs)
                    new_log_probs_full = jax.nn.log_softmax(logits)
                    new_log_probs = jnp.take_along_axis(new_log_probs_full, b_actions[..., None], axis=-1).squeeze(-1)
                    
                    ratio = jnp.exp(new_log_probs - b_log_probs)
                    clip_adv = jnp.clip(ratio, 0.8, 1.2) * b_advs
                    loss_actor = -jnp.mean(jnp.minimum(ratio * b_advs, clip_adv))
                    
                    loss_critic = 0.5 * jnp.mean(huber_loss(v1 - b_returns)) + 0.5 * jnp.mean(huber_loss(v2 - b_returns))
                    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * new_log_probs_full, axis=-1))
                    
                    return loss_actor + 0.5 * loss_critic - 0.05 * entropy

                grad_fn = jax.value_and_grad(loss_fn)
                loss, grads = grad_fn(p_carry)
                grads = jax.lax.pmean(grads, axis_name='p')
                updates, opt_carry = self.optimizer.update(grads, opt_carry)
                p_carry = optax.apply_updates(p_carry, updates)
                
                return (p_carry, opt_carry), loss

            # Sequentially process the 16 minibatches within the compiled graph
            (params, opt_state), batch_losses = jax.lax.scan(
                update_minibatch, 
                (params, opt_state), 
                (mb_obs, mb_actions, mb_log_probs, mb_returns, mb_advs)
            )
            
            return params, opt_state, step_rngs_mapped, states, next_obs, jnp.mean(batch_losses), jnp.mean(rewards_batch)
        
        steps_per_epoch = N_ENVS * CHUNK_SIZE
        state_keys = jax.random.split(rng, num_devices)
        
        @partial(jax.pmap, axis_name='p')
        def init_envs(key):
            keys = jax.random.split(key, N_ENVS // num_devices)
            return jax.vmap(lambda k: env.reset()[1])(keys)

        @partial(jax.pmap, axis_name='p')
        def get_obs_pmap(states):
            return jax.vmap(env._get_obs)(states)

        states = init_envs(state_keys)
        obs = get_obs_pmap(states)

        logger.info(f"🚀 SEQUENTIAL MINI-BATCH ENGINE ONLINE.")
        start_time = time.time()
        
        epoch = 0
        while True:
            replicated_params, replicated_opt_state, step_rngs, states, obs, loss_mean, reward_mean = process_chunk(
                replicated_params, replicated_opt_state, step_rngs, states, obs
            )

            current_loss = jnp.mean(loss_mean)
            current_reward = jnp.mean(reward_mean)

            if current_loss > 1000.0 or jnp.isnan(current_loss):
                logger.error(f"🚨 Math Integrity Warning ({current_loss}). Rebooting safely...")
                if self.load():
                    replicated_params = jax.tree.map(lambda x: jnp.stack([x] * num_devices), self.params)
                    opt_state = self.optimizer.init(self.params)
                    replicated_opt_state = jax.tree.map(lambda x: jnp.stack([x] * num_devices), opt_state)
                    states = init_envs(state_keys)
                    obs = get_obs_pmap(states)
                    continue

            if epoch % 5 == 0:
                elapsed = time.time() - start_time
                fps = int((epoch * steps_per_epoch) / elapsed) if elapsed > 0 else 0
                
                total_trades_count = jnp.sum(states.total_trades)
                winning_trades_count = jnp.sum(states.winning_trades)
                win_rate = (winning_trades_count / jnp.maximum(1, total_trades_count)) * 100.0

                print(f"----------------------------------------")
                print(f"| DECOUPLED PST-TRADER    |            |")
                print(f"|    fps                  | {fps:<10} |")
                print(f"|    total_timesteps      | {epoch * steps_per_epoch:<10} |")
                print(f"| accuracy/               |            |")
                print(f"|    win_rate (%)         | {win_rate:<10.2f} |")
                print(f"|    total_trades         | {total_trades_count:<10} |")
                print(f"| train/                  |            |")
                print(f"|    mean_reward          | {current_reward:<10.5f} |")
                print(f"|    loss                 | {current_loss:<10.5f} |")
                print(f"----------------------------------------")

            if epoch % 20 == 0 and epoch > 0:
                self.params = jax.tree.map(lambda x: x[0], replicated_params)
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.params, f)
                logger.info(f"💾 v4 Checkpoint Saved.")
            epoch += 1

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.params = pickle.load(f)
            return True
        return False

    def predict(self, obs: np.ndarray) -> int:
        if self.params is None:
            return 0
        obs_jnp = jnp.array(obs, dtype=jnp.float32)
        logits, _, _ = self.network.apply(self.params, obs_jnp)
        return int(jnp.argmax(logits))