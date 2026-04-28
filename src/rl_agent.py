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
    def __init__(self, model_path: str = "models/ppo_agent.pkl", device: str = "auto"):
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

        # =====================================================================
        # 🚀 THE GPU SATURATION ENGINE
        # This compiles 100 environment steps into a single GPU instruction.
        # Python CPU does nothing while the GPU blasts through this loop.
        # =====================================================================
        CHUNK_SIZE = 100 
        
        @partial(jax.pmap, axis_name='p')
        def process_chunk(params, opt_state, rng, states, obs):
            def _step(carry, unused):
                params, opt_state, rng, states, obs = carry
                rng, step_rng = jax.random.split(rng)
                
                # 1. Act (100% on GPU)
                logits, values = self.network.apply(params, obs)
                actions = jax.random.categorical(step_rng, logits)
                log_probs = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], axis=-1).squeeze(-1)
                
                # 2. Step Envs (100% on GPU)
                next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(states, actions)
                
                # 3. PPO Math (100% on GPU)
                advantages = rewards - values
                returns = rewards + 0.99 * values
                
                # 4. Gradient Update (100% on GPU)
                grad_fn = jax.value_and_grad(ppo_loss)
                loss, grads = grad_fn(params, obs, actions, advantages, returns, log_probs)
                grads = jax.lax.pmean(grads, axis_name='p') # Sync between the 2x T4s
                updates, opt_state = self.optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                
                return (params, opt_state, rng, next_states, next_obs), (loss, rewards)

            # jax.lax.scan loops purely in VRAM without returning to the Python CPU
            carry = (params, opt_state, rng, states, obs)
            carry, (losses, rewards) = jax.lax.scan(_step, carry, None, length=CHUNK_SIZE)
            
            params, opt_state, rng, states, obs = carry
            # Only return the averaged metrics to the CPU at the very end
            return params, opt_state, rng, states, obs, jnp.mean(losses), jnp.mean(rewards)

        # =====================================================================
        
        # Calculate Epochs
        steps_per_epoch = n_envs * CHUNK_SIZE
        epochs = total_timesteps // steps_per_epoch
        
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

        logger.info(f"🚀 IGNITION: Simulating {total_timesteps:,} steps across {n_envs} Matrix Environments...")
        logger.info(f"🔥 Chunk Size: {CHUNK_SIZE} steps per Python iteration (GPU is locked for {steps_per_epoch} decisions per tick)")
        
        import time
        start_time = time.time()
        
        for epoch in range(epochs):
            # 💥 The Python CPU fires one command, and the GPUs process 409,600 steps.
            replicated_params, replicated_opt_state, step_rngs, states, obs, loss_mean, reward_mean = process_chunk(
                replicated_params, replicated_opt_state, step_rngs, states, obs
            )

            # Log metrics (now happening 100x less frequently on the CPU side)
            if epoch % 10 == 0 or epoch == epochs - 1:
                elapsed = time.time() - start_time
                fps = int((epoch * steps_per_epoch) / elapsed) if elapsed > 0 else 0
                
                print(f"----------------------------------------")
                print(f"| time/                   |            |")
                print(f"|    fps                  | {fps:<10} |")
                print(f"|    chunks_processed     | {epoch:<10} |")
                print(f"|    time_elapsed (s)     | {int(elapsed):<10} |")
                print(f"|    total_timesteps      | {epoch * steps_per_epoch:<10} |")
                print(f"| train/                  |            |")
                print(f"|    mean_reward          | {jnp.mean(reward_mean):<10.5f} |")
                print(f"|    loss                 | {jnp.mean(loss_mean):<10.5f} |")
                print(f"----------------------------------------")

        self.params = jax.tree.map(lambda x: x[0], replicated_params)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.params, f)
        logger.info(f"✅ Training Complete. RL Agent saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.params = pickle.load(f)
            logger.info(f"Loaded RL Agent from {self.model_path}")
            return True
        logger.warning(f"Could not find RL Agent at {self.model_path}")
        return False

    def predict(self, obs: np.ndarray) -> int:
        if self.params is None:
            return 0
        obs_jnp = jnp.array(obs, dtype=jnp.float32)
        logits, _ = self.network.apply(self.params, obs_jnp)
        return int(jnp.argmax(logits))