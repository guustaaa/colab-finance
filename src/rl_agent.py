import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        
        logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)
        return logits, jnp.squeeze(value, axis=-1)

class RLAgent:
    def __init__(self, model_path: str = "/kaggle/working/ForexAI_State/models/rl_ppo_agent.pkl", device: str = "auto"):
        self.model_path = model_path
        self.action_dim = 4
        self.network = ActorCritic(action_dim=self.action_dim)
        self.optimizer = optax.adam(learning_rate=3e-4)
        self.params = None
        
    def train(self, data_dict: dict, total_timesteps: int = 50_000_000, n_envs: int = 4096, batch_size: int = 8192):
        logger.info(f"Aggregating data from {len(data_dict)} currency pairs...")
        combined_df = pd.concat(list(data_dict.values()), ignore_index=True)
        data_matrix = jnp.array(combined_df.values, dtype=jnp.float32)
        
        env = JaxForexEnv(data_matrix=data_matrix)
        
        num_devices = jax.local_device_count()
        device_platform = jax.devices()[0].platform.upper()
        
        logger.info(f"⚙️ Compute Engine: {num_devices}x {device_platform}")
        
        rng = jax.random.PRNGKey(42)
        rng, init_rng = jax.random.split(rng)
        
        dummy_obs, _ = env.reset()
        self.params = self.network.init(init_rng, dummy_obs)
        opt_state = self.optimizer.init(self.params)
        
        # Load existing weights if they exist (for true continuous continuation)
        if self.load():
            logger.info("Resuming training from existing local checkpoint...")
            
        replicated_params = jax.tree.map(lambda x: jnp.stack([x] * num_devices), self.params)
        replicated_opt_state = jax.tree.map(lambda x: jnp.stack([x] * num_devices), opt_state)

        def ppo_loss(params, obs, actions, advantages, returns, old_log_probs):
            logits, values = self.network.apply(params, obs)
            log_probs = jax.nn.log_softmax(logits)
            action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)
            
            ratio = jnp.exp(action_log_probs - old_log_probs)
            clip_adv = jnp.clip(ratio, 0.8, 1.2) * advantages
            loss_actor = -jnp.mean(jnp.minimum(ratio * advantages, clip_adv))
            loss_critic = 0.5 * jnp.mean((returns - values) ** 2)
            entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1))
            
            return loss_actor + 0.5 * loss_critic - 0.01 * entropy

        CHUNK_SIZE = 100 
        
        @partial(jax.pmap, axis_name='p')
        def process_chunk(params, opt_state, rng, states, obs):
            def _step(carry, unused):
                params, opt_state, rng, states, obs = carry
                rng, step_rng = jax.random.split(rng)
                
                logits, values = self.network.apply(params, obs)
                actions = jax.random.categorical(step_rng, logits)
                log_probs = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], axis=-1).squeeze(-1)
                
                next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(states, actions)
                
                advantages = rewards - values
                returns = rewards + 0.99 * values
                
                grad_fn = jax.value_and_grad(ppo_loss)
                loss, grads = grad_fn(params, obs, actions, advantages, returns, log_probs)
                grads = jax.lax.pmean(grads, axis_name='p')
                updates, opt_state = self.optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                
                return (params, opt_state, rng, next_states, next_obs), (loss, rewards)

            carry = (params, opt_state, rng, states, obs)
            carry, (losses, rewards) = jax.lax.scan(_step, carry, None, length=CHUNK_SIZE)
            params, opt_state, rng, states, obs = carry
            return params, opt_state, rng, states, obs, jnp.mean(losses), jnp.mean(rewards)
        
        steps_per_epoch = n_envs * CHUNK_SIZE
        state_keys = jax.random.split(rng, num_devices)
        step_rngs = jax.random.split(jax.random.split(rng)[1], num_devices)
        
        @partial(jax.pmap, axis_name='p')
        def init_envs(key):
            keys = jax.random.split(key, n_envs // num_devices)
            return jax.vmap(lambda k: env.reset()[1])(keys)

        @partial(jax.pmap, axis_name='p')
        def get_obs_pmap(states):
            return jax.vmap(env._get_obs)(states)

        states = init_envs(state_keys)
        obs = get_obs_pmap(states)

        logger.info(f"🚀 IGNITION: Starting CONTINUOUS Matrix Simulation...")
        logger.info(f"🔥 Chunk Size: {CHUNK_SIZE} steps per loop. Auto-saving locally every 500 chunks.")
        
        start_time = time.time()
        
        epoch = 0
        while True: # Infinite Continuous Loop
            replicated_params, replicated_opt_state, step_rngs, states, obs, loss_mean, reward_mean = process_chunk(
                replicated_params, replicated_opt_state, step_rngs, states, obs
            )

            # Log metrics every 10 chunks
            if epoch % 10 == 0:
                elapsed = time.time() - start_time
                fps = int((epoch * steps_per_epoch) / elapsed) if elapsed > 0 else 0
                
                # Calculate the global win rate across all 4096 matrix instances
                total_trades_count = jnp.sum(states.total_trades)
                winning_trades_count = jnp.sum(states.winning_trades)
                win_rate = (winning_trades_count / jnp.maximum(1, total_trades_count)) * 100.0

                print(f"----------------------------------------")
                print(f"| time/                   |            |")
                print(f"|    fps                  | {fps:<10} |")
                print(f"|    chunks_processed     | {epoch:<10} |")
                print(f"|    time_elapsed (s)     | {int(elapsed):<10} |")
                print(f"|    total_timesteps      | {epoch * steps_per_epoch:<10} |")
                print(f"| accuracy/               |            |")
                print(f"|    win_rate (%)         | {win_rate:<10.2f} |")
                print(f"|    total_trades         | {total_trades_count:<10} |")
                print(f"| train/                  |            |")
                print(f"|    mean_reward          | {jnp.mean(reward_mean):<10.5f} |")
                print(f"|    loss                 | {jnp.mean(loss_mean):<10.5f} |")
                print(f"----------------------------------------")

            # Local Periodic Save every 500 chunks
            if epoch % 500 == 0 and epoch > 0:
                self.params = jax.tree.map(lambda x: x[0], replicated_params)
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.params, f)
                logger.info(f"💾 Checkpoint Saved to {self.model_path}")
            
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
        logits, _ = self.network.apply(self.params, obs_jnp)
        return int(jnp.argmax(logits))