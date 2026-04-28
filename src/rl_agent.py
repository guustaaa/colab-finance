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
        # 1. Combine all pairs into one massive super-matrix
        logger.info(f"Aggregating data from {len(data_dict)} currency pairs...")
        combined_df = pd.concat(list(data_dict.values()), ignore_index=True)
        data_matrix = jnp.array(combined_df.values, dtype=jnp.float32)
        
        env = JaxForexEnv(data_matrix=data_matrix)
        
        # 2. Enforce Dual-GPU Usage
        num_devices = jax.local_device_count()
        device_platform = jax.devices()[0].platform.upper()
        
        logger.info(f"⚙️ Compute Engine: {num_devices}x {device_platform}")
        if device_platform != 'GPU':
            logger.error("🚨 CRITICAL WARNING: JAX IS RUNNING ON CPU!")
            logger.error("If running on Kaggle, ensure 'T4 x2' is selected and install: !pip install -U \"jax[cuda12]\"")
            
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

        @partial(jax.pmap, axis_name='p')
        def update_step(params, opt_state, obs, actions, advantages, returns, old_log_probs):
            grad_fn = jax.value_and_grad(ppo_loss)
            loss, grads = grad_fn(params, obs, actions, advantages, returns, old_log_probs)
            grads = jax.lax.pmean(grads, axis_name='p')
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        @partial(jax.pmap, axis_name='p')
        def act(params, rng, obs):
            logits, val = self.network.apply(params, obs)
            action = jax.random.categorical(rng, logits)
            log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], axis=-1).squeeze(-1)
            return action, log_prob, val
        
        # Calculate Epochs based on n_envs and num_devices
        steps_per_epoch = n_envs
        epochs = total_timesteps // steps_per_epoch
        
        state_keys = jax.random.split(rng, num_devices)
        
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
        
        import time
        start_time = time.time()
        
        # 3. Main Training Loop with Restored Logging
        for epoch in range(epochs):
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_devices)
            
            actions, log_probs, values = act(replicated_params, step_rngs, obs)
            
            @partial(jax.pmap, axis_name='p')
            def step_envs(states, actions):
                return jax.vmap(env.step)(states, actions)

            next_obs, next_states, rewards, dones, infos = step_envs(states, actions)
            
            advantages = rewards - values
            returns = rewards + 0.99 * values
            
            replicated_params, replicated_opt_state, loss = update_step(
                replicated_params, replicated_opt_state, obs, actions, advantages, returns, log_probs
            )
            
            obs = next_obs
            states = next_states

            # Log metrics every 100 epochs
            if epoch % 100 == 0 or epoch == epochs - 1:
                elapsed = time.time() - start_time
                fps = int((epoch * steps_per_epoch) / elapsed) if elapsed > 0 else 0
                mean_reward = jnp.mean(rewards)
                
                print(f"----------------------------------------")
                print(f"| time/                   |            |")
                print(f"|    fps                  | {fps:<10} |")
                print(f"|    iterations           | {epoch:<10} |")
                print(f"|    time_elapsed (s)     | {int(elapsed):<10} |")
                print(f"|    total_timesteps      | {epoch * steps_per_epoch:<10} |")
                print(f"| train/                  |            |")
                print(f"|    mean_reward          | {mean_reward:<10.5f} |")
                print(f"|    loss                 | {jnp.mean(loss):<10.5f} |")
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