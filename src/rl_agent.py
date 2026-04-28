import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# 🛑 Fixes the "Unable to initialize backend 'tpu'" warning
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
# 🧠 HYBRID PST-TRADER ARCHITECTURE (PPO + SAC + TD3)
# =====================================================================

class Actor(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        # MASSIVE RAM UTILIZATION: 1024-width layers
        x = nn.Dense(1024)(x)
        x = nn.tanh(x)
        x = nn.Dense(1024)(x)
        x = nn.tanh(x)
        x = nn.Dense(512)(x)
        x = nn.tanh(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits

class TwinCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        # 🛡️ TD3 ELEMENT: Twin Critics to prevent Overestimation Bias
        # Critic 1
        c1 = nn.Dense(1024)(x)
        c1 = nn.relu(c1)
        c1 = nn.Dense(512)(c1)
        c1 = nn.relu(c1)
        v1 = nn.Dense(1)(c1)
        
        # Critic 2
        c2 = nn.Dense(1024)(x)
        c2 = nn.relu(c2)
        c2 = nn.Dense(512)(c2)
        c2 = nn.relu(c2)
        v2 = nn.Dense(1)(c2)
        
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
    def __init__(self, model_path: str = "/kaggle/working/ForexAI_State/models/rl_hybrid_agent.pkl", device: str = "auto"):
        self.model_path = model_path
        self.action_dim = 4
        self.network = HybridNetwork(action_dim=self.action_dim)
        
        # 🛡️ PPO ELEMENT: Trust Region Optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=1e-4)
        )
        self.params = None
        
    def train(self, data_dict: dict, total_timesteps: int = 500_000_000):
        # 🚀 RAM MAXIMIZATION PARAMETERS 
        N_ENVS = 16384       # 16k parallel universes
        CHUNK_SIZE = 500     # 500 steps per universe per loop
        
        logger.info(f"Aggregating data from {len(data_dict)} currency pairs...")
        combined_df = pd.concat(list(data_dict.values()), ignore_index=True)
        data_matrix = jnp.array(combined_df.values, dtype=jnp.float32)
        
        env = JaxForexEnv(data_matrix=data_matrix)
        num_devices = jax.local_device_count()
        logger.info(f"⚙️ Compute Engine: {num_devices}x GPU")
        
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        
        dummy_obs, _ = env.reset()
        self.params = self.network.init(init_rng, dummy_obs)
        opt_state = self.optimizer.init(self.params)
        
        if self.load():
            logger.info("Resuming from existing Hybrid Checkpoint...")
            
        replicated_params = jax.tree.map(lambda x: jnp.stack([x] * num_devices), self.params)
        replicated_opt_state = jax.tree.map(lambda x: jnp.stack([x] * num_devices), opt_state)

        # ---------------------------------------------------------
        # THE HYBRID LOSS FUNCTION (PPO + SAC + TD3)
        # ---------------------------------------------------------
        def hybrid_loss(params, obs, actions, advantages, returns, old_log_probs):
            # 🛠️ FIX: Call the main network apply function directly
            logits, v1, v2 = self.network.apply(params, obs)
            
            log_probs = jax.nn.log_softmax(logits)
            action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)
            
            # 1. PPO Policy Update
            ratio = jnp.exp(action_log_probs - old_log_probs)
            clip_adv = jnp.clip(ratio, 0.8, 1.2) * advantages
            loss_actor = -jnp.mean(jnp.minimum(ratio * advantages, clip_adv))
            
            # 2. TD3 Twin Critic Update
            loss_critic1 = 0.5 * jnp.mean((returns - v1) ** 2)
            loss_critic2 = 0.5 * jnp.mean((returns - v2) ** 2)
            loss_critic = loss_critic1 + loss_critic2
            
            # 3. SAC Soft-Entropy Maximization (Forces aggressive exploration)
            entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1))
            sac_alpha = 0.05 # SAC Temperature
            
            total_loss = loss_actor + (0.5 * loss_critic) - (sac_alpha * entropy)
            return total_loss

        @partial(jax.pmap, axis_name='p')
        def process_chunk(params, opt_state, rng, states, obs):
            def _step(carry, unused):
                params, opt_state, rng, states, obs = carry
                rng, step_rng = jax.random.split(rng)
                
                # Forward Pass
                logits, v1, v2 = self.network.apply(params, obs)
                
                # 🛡️ TD3 ELEMENT: Pessimistic Bound Calculation
                v_pessimistic = jnp.minimum(v1, v2)
                
                actions = jax.random.categorical(step_rng, logits)
                log_probs = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], axis=-1).squeeze(-1)
                
                # Step parallel worlds
                next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(states, actions)
                
                # Calculate Advantage using the Pessimistic Bound
                advantages = rewards - v_pessimistic
                returns = rewards + 0.99 * v_pessimistic
                
                # Backprop
                grad_fn = jax.value_and_grad(hybrid_loss)
                loss, grads = grad_fn(params, obs, actions, advantages, returns, log_probs)
                grads = jax.lax.pmean(grads, axis_name='p')
                updates, opt_state = self.optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                
                return (params, opt_state, rng, next_states, next_obs), (loss, rewards)

            carry = (params, opt_state, rng, states, obs)
            carry, (losses, rewards) = jax.lax.scan(_step, carry, None, length=CHUNK_SIZE)
            params, opt_state, rng, states, obs = carry
            return params, opt_state, rng, states, obs, jnp.mean(losses), jnp.mean(rewards)
        
        steps_per_epoch = N_ENVS * CHUNK_SIZE
        state_keys = jax.random.split(rng, num_devices)
        step_rngs = jax.random.split(jax.random.split(rng)[1], num_devices)
        
        @partial(jax.pmap, axis_name='p')
        def init_envs(key):
            keys = jax.random.split(key, N_ENVS // num_devices)
            return jax.vmap(lambda k: env.reset()[1])(keys)

        @partial(jax.pmap, axis_name='p')
        def get_obs_pmap(states):
            return jax.vmap(env._get_obs)(states)

        states = init_envs(state_keys)
        obs = get_obs_pmap(states)

        logger.info(f"🚀 PST-TRADER IGNITION: Ram Overdrive Mode Enabled.")
        start_time = time.time()
        
        epoch = 0
        while True:
            replicated_params, replicated_opt_state, step_rngs, states, obs, loss_mean, reward_mean = process_chunk(
                replicated_params, replicated_opt_state, step_rngs, states, obs
            )

            current_loss = jnp.mean(loss_mean)
            current_reward = jnp.mean(reward_mean)

            # Auto-Recovery
            if current_loss > 1000.0 or jnp.isnan(current_loss):
                logger.error("🚨 MATH EXPLOSION DETECTED. Reverting to safe checkpoint...")
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
                print(f"| PST-TRADER PERFORMANCE  |            |")
                print(f"|    fps                  | {fps:<10} |")
                print(f"|    total_timesteps      | {epoch * steps_per_epoch:<10} |")
                print(f"| accuracy/               |            |")
                print(f"|    win_rate (%)         | {win_rate:<10.2f} |")
                print(f"|    total_trades         | {total_trades_count:<10} |")
                print(f"| train/                  |            |")
                print(f"|    mean_reward          | {current_reward:<10.5f} |")
                print(f"|    loss                 | {current_loss:<10.5f} |")
                print(f"----------------------------------------")

            # Local Save
            if epoch % 20 == 0 and epoch > 0:
                self.params = jax.tree.map(lambda x: x[0], replicated_params)
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.params, f)
                logger.info(f"💾 Hybrid Weights Checkpoint Saved.")
            
            epoch += 1

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.params = pickle.load(f)
            return True
        return False