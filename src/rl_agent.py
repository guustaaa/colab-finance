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

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0

class SACActor(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        
        mu = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        
        # 🛡️ PPO STABILITY: Bound the log_std to prevent Math NaNs
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

class SACCritic(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.LayerNorm()(x)
        
        # Twin Critic 1
        q1 = nn.Dense(512)(x)
        q1 = nn.relu(q1)
        q1 = nn.Dense(512)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)
        
        # Twin Critic 2
        q2 = nn.Dense(512)(x)
        q2 = nn.relu(q2)
        q2 = nn.Dense(512)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        
        return jnp.squeeze(q1, axis=-1), jnp.squeeze(q2, axis=-1)

class RLAgent:
    def __init__(self, model_path: str = "/kaggle/working/ForexAI_State/models/rl_custom_sac_v1.pkl"):
        self.model_path = model_path
        self.action_dim = 4
        self.actor = SACActor(action_dim=self.action_dim)
        self.critic = SACCritic()
        
        self.opt_actor = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=3e-4))
        self.opt_critic = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate=3e-4))
        self.opt_alpha = optax.adam(learning_rate=3e-4)
        
        self.target_entropy = -float(self.action_dim)
        self.tau = 0.005 # Soft target update rate
        self.params = None
        
    def train(self, data_dict: dict, total_timesteps: int = 500_000_000):
        N_ENVS = 2048   
        CHUNK_SIZE = 64
        NUM_MINIBATCHES = 8
        
        logger.info(f"Aggregating Multi-Timeframe matrices...")
        combined_df = pd.concat(list(data_dict.values()), ignore_index=True)
        data_matrix = jnp.array(combined_df.values, dtype=jnp.float32)
        
        env = JaxForexEnv(data_matrix=data_matrix)
        num_devices = jax.local_device_count()
        logger.info(f"⚙️ Custom SAC Engine: {num_devices}x GPU | Horizon: {N_ENVS}x{CHUNK_SIZE}")
        
        rng = jax.random.PRNGKey(42)
        rng, key_a, key_c = jax.random.split(rng, 3)
        
        dummy_obs, _ = env.reset()
        dummy_action = jnp.zeros((self.action_dim,))
        
        actor_params = self.actor.init(key_a, dummy_obs)
        critic_params = self.critic.init(key_c, dummy_obs, dummy_action)
        critic_target_params = critic_params 
        log_alpha = jnp.array(0.0) 
        
        opt_state_a = self.opt_actor.init(actor_params)
        opt_state_c = self.opt_critic.init(critic_params)
        opt_state_alpha = self.opt_alpha.init(log_alpha)
        
        self.params = {'actor': actor_params, 'critic': critic_params, 'target': critic_target_params, 'log_alpha': log_alpha}
        
        if self.load():
            logger.info("Resuming from Custom SAC Checkpoint...")
            actor_params, critic_params, critic_target_params, log_alpha = self.params['actor'], self.params['critic'], self.params['target'], self.params['log_alpha']
            
        # Replicate across devices
        def rep(x): return jnp.stack([x] * num_devices)
        p_actor = jax.tree.map(rep, actor_params)
        p_critic = jax.tree.map(rep, critic_params)
        p_target = jax.tree.map(rep, critic_target_params)
        p_alpha = rep(log_alpha)
        
        o_actor = jax.tree.map(rep, opt_state_a)
        o_critic = jax.tree.map(rep, opt_state_c)
        o_alpha = jax.tree.map(rep, opt_state_alpha)
        
        step_rngs_mapped = jax.random.split(rng, num_devices)

        
        opt_state_a = self.opt_actor.init(actor_params)
        opt_state_c = self.opt_critic.init(critic_params)
        opt_state_alpha = self.opt_alpha.init(log_alpha)
        
        self.params = {'actor': actor_params, 'critic': critic_params, 'target': critic_target_params, 'log_alpha': log_alpha}
        
        if self.load():
            logger.info("Resuming from Custom SAC Checkpoint...")
            actor_params, critic_params, critic_target_params, log_alpha = self.params['actor'], self.params['critic'], self.params['target'], self.params['log_alpha']
            
        # Replicate across devices
        def rep(x): return jnp.stack([x] * num_devices)
        p_actor = jax.tree.map(rep, actor_params)
        p_critic = jax.tree.map(rep, critic_params)
        p_target = jax.tree.map(rep, critic_target_params)
        p_alpha = rep(log_alpha)
        
        o_actor = jax.tree.map(rep, opt_state_a)
        o_critic = jax.tree.map(rep, opt_state_c)
        o_alpha = jax.tree.map(rep, opt_state_alpha)
        
        step_rngs_mapped = jax.random.split(rng, num_devices)

        # -------------------------------------------------------------
        # CUSTOM SAC ALGORITHM LOGIC
        # -------------------------------------------------------------
        def sample_action(actor_params, obs, rng_key):
            mu, log_std = self.actor.apply(actor_params, obs)
            std = jnp.exp(log_std)
            noise = jax.random.normal(rng_key, mu.shape)
            pi = mu + noise * std
            
            # Reparameterization Trick to bind actions between [-1, 1]
            action = jnp.tanh(pi)
            log_prob = jnp.sum(jax.scipy.stats.norm.logpdf(pi, mu, std) - 2 * (jnp.log(2) - pi - jax.nn.softplus(-2 * pi)), axis=-1)
            return action, log_prob

        @partial(jax.pmap, axis_name='p')
        def process_chunk(p_actor, p_critic, p_target, p_alpha, o_actor, o_critic, o_alpha, step_rngs_mapped, states, obs, global_step):
            
            # PHASE 1: EXPLORATORY ROLLOUT
            def rollout_step(carry, unused):
                carry_rng, current_states, current_obs = carry
                carry_rng, step_rng = jax.random.split(carry_rng)
                
                action_continuous, _ = sample_action(p_actor, current_obs, step_rng)
                
                # Convert continuous action back to discrete space for the environment
                action_discrete = jnp.argmax(action_continuous, axis=-1)
                
                # Action Masking
                valid_hold = jnp.ones_like(current_states.position, dtype=jnp.bool_)
                valid_buy = current_states.position <= 0
                valid_sell = current_states.position >= 0 
                valid_close = current_states.position != 0 
                action_mask = jnp.stack([valid_hold, valid_buy, valid_sell, valid_close], axis=-1)
                
                # Override invalid choices with Hold (0)
                is_valid = jnp.take_along_axis(action_mask, action_discrete[:, None], axis=-1).squeeze(-1)
                action_discrete = jnp.where(is_valid, action_discrete, 0)
                
                next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(current_states, action_discrete)
                
                transition = (current_obs, action_continuous, rewards, next_obs, dones)
                return (carry_rng, next_states, next_obs), transition

            (step_rngs_mapped, states, next_obs), trajectories = jax.lax.scan(
                rollout_step, (step_rngs_mapped, states, obs), None, length=CHUNK_SIZE
            )
            obs_b, act_b, r_b, next_obs_b, d_b = trajectories
            
            # Flatten for Replay Buffer simulation
            flat_size = CHUNK_SIZE * (N_ENVS // jax.local_device_count())
            step_rngs_mapped, shuffle_rng = jax.random.split(step_rngs_mapped)
            indices = jax.random.permutation(shuffle_rng, flat_size)
            
            def flat(x): return x.reshape(flat_size, -1).squeeze()[indices]
            f_obs, f_act, f_r, f_nobs, f_d = flat(obs_b), flat(act_b), flat(r_b), flat(next_obs_b), flat(d_b)
            
            def reshape_to_minibatches(x):
                mb_size = x.shape[0] // NUM_MINIBATCHES
                return x.reshape(NUM_MINIBATCHES, mb_size, *x.shape[1:])

            mb_obs, mb_act, mb_r, mb_nobs, mb_d = reshape_to_minibatches(f_obs), reshape_to_minibatches(f_act), reshape_to_minibatches(f_r), reshape_to_minibatches(f_nobs), reshape_to_minibatches(f_d)

            # PHASE 2: SAC OPTIMIZATION
            def update_minibatch(carry, batch_data):
                ca_act, ca_crit, ca_targ, ca_alpha, o_act, o_crit, o_al, minirng, mb_step = carry
                b_obs, b_act, b_r, b_nobs, b_d = batch_data
                minirng, next_act_rng = jax.random.split(minirng)
                
                alpha = jnp.exp(ca_alpha)
                
                # 1. Critic Update (TD3 Inspired Minimum Q-Target)
                next_action, next_log_prob = sample_action(ca_act, b_nobs, next_act_rng)
                q1_next, q2_next = self.critic.apply(ca_targ, b_nobs, next_action)
                q_next = jnp.minimum(q1_next, q2_next) - alpha * next_log_prob
                q_target = jax.lax.stop_gradient(b_r + 0.99 * (1.0 - b_d) * q_next)

                def critic_loss_fn(p_c):
                    q1, q2 = self.critic.apply(p_c, b_obs, b_act)
                    return 0.5 * jnp.mean((q1 - q_target)**2) + 0.5 * jnp.mean((q2 - q_target)**2)

                c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(ca_crit)
                c_grads = jax.lax.pmean(c_grads, axis_name='p')
                c_updates, o_crit = self.opt_critic.update(c_grads, o_crit)
                ca_crit = optax.apply_updates(ca_crit, c_updates)

                # 2. Delayed Actor & Alpha Update (TD3 Inspired)
                def actor_update(ca_act, o_act, ca_alpha, o_al, rng_a):
                    def actor_loss_fn(p_a):
                        act_new, logp = sample_action(p_a, b_obs, rng_a)
                        q1_n, q2_n = self.critic.apply(jax.lax.stop_gradient(ca_crit), b_obs, act_new)
                        q_n = jnp.minimum(q1_n, q2_n)
                        return jnp.mean(alpha * logp - q_n), logp

                    (a_loss, logp), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(ca_act)
                    a_grads = jax.lax.pmean(a_grads, axis_name='p')
                    a_updates, o_act = self.opt_actor.update(a_grads, o_act)
                    ca_act = optax.apply_updates(ca_act, a_updates)
                    
                    def alpha_loss_fn(p_al):
                        return -jnp.mean(jnp.exp(p_al) * (logp + self.target_entropy))
                    
                    al_loss, al_grads = jax.value_and_grad(alpha_loss_fn)(ca_alpha)
                    al_grads = jax.lax.pmean(al_grads, axis_name='p')
                    al_updates, o_al = self.opt_alpha.update(al_grads, o_al)
                    ca_alpha = optax.apply_updates(ca_alpha, al_updates)
                    
                    # 3. Soft Target Polyak Update
                    ca_targ_new = jax.tree.map(lambda t, c: self.tau * c + (1 - self.tau) * t, ca_targ, ca_crit)
                    return ca_act, o_act, ca_alpha, o_al, ca_targ_new, a_loss

                # Only update Actor every 2 steps
                do_actor_update = (mb_step % 2 == 0)
                ca_act, o_act, ca_alpha, o_al, ca_targ, a_loss = jax.lax.cond(
                    do_actor_update,
                    lambda _: actor_update(ca_act, o_act, ca_alpha, o_al, next_act_rng),
                    lambda _: (ca_act, o_act, ca_alpha, o_al, ca_targ, 0.0),
                    operand=None
                )

                return (ca_act, ca_crit, ca_targ, ca_alpha, o_act, o_crit, o_al, minirng, mb_step + 1), c_loss

            carry = (p_actor, p_critic, p_target, p_alpha, o_actor, o_critic, o_alpha, step_rngs_mapped, global_step)
            carry, batch_losses = jax.lax.scan(update_minibatch, carry, (mb_obs, mb_act, mb_r, mb_nobs, mb_d))
            p_actor, p_critic, p_target, p_alpha, o_actor, o_critic, o_alpha, step_rngs_mapped, new_global_step = carry
            
            return p_actor, p_critic, p_target, p_alpha, o_actor, o_critic, o_alpha, step_rngs_mapped, states, next_obs, jnp.mean(batch_losses), jnp.mean(r_b), new_global_step
        
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
        global_step = jnp.zeros(num_devices, dtype=jnp.int32)

        logger.info(f"🚀 CUSTOM SAC ONLINE (Multi-Timeframe M5/M15/H1).")
        start_time = time.time()
        
        epoch = 0
        while True:
            p_actor, p_critic, p_target, p_alpha, o_actor, o_critic, o_alpha, step_rngs_mapped, states, obs, loss_mean, reward_mean, global_step = process_chunk(
                p_actor, p_critic, p_target, p_alpha, o_actor, o_critic, o_alpha, step_rngs_mapped, states, obs, global_step
            )

            current_loss = jnp.mean(loss_mean)
            current_reward = jnp.mean(reward_mean)

            if epoch % 5 == 0:
                elapsed = time.time() - start_time
                fps = int((epoch * N_ENVS * CHUNK_SIZE) / elapsed) if elapsed > 0 else 0
                
                total_trades_count = jnp.sum(states.total_trades)
                winning_trades_count = jnp.sum(states.winning_trades)
                win_rate = (winning_trades_count / jnp.maximum(1, total_trades_count)) * 100.0

                print(f"----------------------------------------")
                print(f"| CUSTOM SAC ENGINE       |            |")
                print(f"|    fps                  | {fps:<10} |")
                print(f"|    total_timesteps      | {epoch * N_ENVS * CHUNK_SIZE:<10} |")
                print(f"| accuracy/               |            |")
                print(f"|    win_rate (%)         | {win_rate:<10.2f} |")
                print(f"|    total_trades         | {total_trades_count:<10} |")
                print(f"| train/                  |            |")
                print(f"|    mean_reward          | {current_reward:<10.5f} |")
                print(f"|    critic_loss          | {current_loss:<10.5f} |")
                print(f"----------------------------------------")

            if epoch % 20 == 0 and epoch > 0:
                self.params = {
                    'actor': jax.tree.map(lambda x: x[0], p_actor),
                    'critic': jax.tree.map(lambda x: x[0], p_critic),
                    'target': jax.tree.map(lambda x: x[0], p_target),
                    'log_alpha': jax.tree.map(lambda x: x[0], p_alpha)
                }
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.params, f)
                logger.info(f"💾 Custom SAC Checkpoint Saved.")
            epoch += 1

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.params = pickle.load(f)
            return True
        return False