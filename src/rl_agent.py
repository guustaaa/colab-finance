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
# 🧠 DISCRETE SOFT ACTOR-CRITIC (Proper Math Formulation)
# =====================================================================

class DiscreteActor(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x, mask):
        x = nn.LayerNorm()(x)
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        init_fn = nn.initializers.orthogonal(np.sqrt(2))
        
        x = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(x)
        x = nn.tanh(x)
        x = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(x)
        x = nn.tanh(x)
        
        logits = nn.Dense(self.action_dim, dtype=jnp.float32, kernel_init=nn.initializers.orthogonal(0.01))(x)
        
        # 🛡️ Native Masking: Set invalid actions to -infinity before softmax
        logits = jnp.where(mask, logits, -1e9)
        return logits

class DiscreteTwinCritic(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        # 🛡️ THE FIX: The critic no longer takes 'action' as an input. 
        # It natively outputs 4 distinct Q-values for the 4 discrete states.
        x = nn.LayerNorm()(x)
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        init_fn = nn.initializers.orthogonal(np.sqrt(2))
        
        # Q1 Network
        q1 = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(x)
        q1 = nn.relu(q1)
        q1 = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(q1)
        q1 = nn.relu(q1)
        q1_out = nn.Dense(self.action_dim, dtype=jnp.float32)(q1)
        
        # Q2 Network
        q2 = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(x)
        q2 = nn.relu(q2)
        q2 = nn.Dense(512, dtype=jnp.bfloat16, kernel_init=init_fn)(q2)
        q2 = nn.relu(q2)
        q2_out = nn.Dense(self.action_dim, dtype=jnp.float32)(q2)
        
        return q1_out, q2_out

class RLAgent:
    def __init__(self, model_path: str = "/kaggle/working/ForexAI_State/models/rl_discrete_sac_v1.pkl"):
        self.model_path = model_path
        self.action_dim = 4
        
        self.actor = DiscreteActor(action_dim=self.action_dim)
        self.critic = DiscreteTwinCritic(action_dim=self.action_dim)
        
        self.opt_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=3e-4))
        self.opt_critic = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=3e-4))
        self.opt_alpha = optax.adam(learning_rate=3e-4)
        
        # Target entropy for 4 discrete actions is typically around -0.98 * ln(4)
        self.target_entropy = -1.0 
        self.tau = 0.005 # Soft target update rate
        self.gamma = 0.99
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
        logger.info(f"⚙️ Discrete SAC Engine: {num_devices}x GPU | Horizon: {N_ENVS}x{CHUNK_SIZE}")
        
        rng = jax.random.PRNGKey(42)
        rng, key_a, key_c = jax.random.split(rng, 3)
        
        dummy_obs, _ = env.reset()
        dummy_mask = jnp.ones((self.action_dim,), dtype=jnp.bool_)
        
        actor_params = self.actor.init(key_a, dummy_obs, dummy_mask)
        critic_params = self.critic.init(key_c, dummy_obs)
        critic_target_params = critic_params 
        log_alpha = jnp.array(0.0) 
        
        opt_state_a = self.opt_actor.init(actor_params)
        opt_state_c = self.opt_critic.init(critic_params)
        opt_state_alpha = self.opt_alpha.init(log_alpha)
        
        self.params = {'actor': actor_params, 'critic': critic_params, 'target': critic_target_params, 'log_alpha': log_alpha}
        
        if self.load():
            logger.info("Resuming from Checkpoint...")
            actor_params, critic_params, critic_target_params, log_alpha = self.params['actor'], self.params['critic'], self.params['target'], self.params['log_alpha']
            
        def rep(x): return jnp.stack([x] * num_devices)
        p_actor = jax.tree.map(rep, actor_params)
        p_critic = jax.tree.map(rep, critic_params)
        p_target = jax.tree.map(rep, critic_target_params)
        p_alpha = rep(log_alpha)
        
        o_actor = jax.tree.map(rep, opt_state_a)
        o_critic = jax.tree.map(rep, opt_state_c)
        o_alpha = jax.tree.map(rep, opt_state_alpha)
        
        step_rngs_mapped = jax.random.split(rng, num_devices)

        def get_mask(position):
            valid_hold = jnp.ones_like(position, dtype=jnp.bool_)
            valid_buy = position <= 0
            valid_sell = position >= 0
            valid_close = position != 0
            return jnp.stack([valid_hold, valid_buy, valid_sell, valid_close], axis=-1)

        @partial(jax.pmap, axis_name='p')
        def process_chunk(p_actor, p_critic, p_target, p_alpha, o_actor, o_critic, o_alpha, step_rngs_mapped, states, obs, global_step):
            
            # PHASE 1: EXPLORATORY ROLLOUT
            def rollout_step(carry, unused):
                carry_rng, current_states, current_obs = carry
                carry_rng, step_rng = jax.random.split(carry_rng)
                
                current_mask = get_mask(current_states.position)
                
                # Get action probabilities
                logits = self.actor.apply(p_actor, current_obs, current_mask)
                probs = jax.nn.softmax(logits)
                
                # Sample discrete action
                actions = jax.random.categorical(step_rng, logits)
                
                next_obs, next_states, rewards, dones, infos = jax.vmap(env.step)(current_states, actions)
                next_mask = get_mask(next_states.position)
                
                # Environment Continuous Streaming
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
                
                transition = (current_obs, current_mask, actions, rewards, next_obs, next_mask, dones)
                return (carry_rng, next_states, next_obs), transition

            (step_rngs_mapped, states, next_obs), trajectories = jax.lax.scan(
                rollout_step, (step_rngs_mapped, states, obs), None, length=CHUNK_SIZE
            )
            obs_b, mask_b, act_b, r_b, next_obs_b, next_mask_b, d_b = trajectories
            
            # Flatten & Shuffle
            flat_size = CHUNK_SIZE * (N_ENVS // jax.local_device_count())
            step_rngs_mapped, shuffle_rng = jax.random.split(step_rngs_mapped)
            indices = jax.random.permutation(shuffle_rng, flat_size)
            
            def flat(x): return x.reshape(flat_size, -1).squeeze()[indices]
            f_obs, f_mask, f_act, f_r, f_nobs, f_nmask, f_d = flat(obs_b), flat(mask_b), flat(act_b), flat(r_b), flat(next_obs_b), flat(next_mask_b), flat(d_b)
            
            def reshape_to_minibatches(x):
                mb_size = x.shape[0] // NUM_MINIBATCHES
                return x.reshape(NUM_MINIBATCHES, mb_size, *x.shape[1:])

            mb_obs, mb_mask, mb_act, mb_r, mb_nobs, mb_nmask, mb_d = map(
                reshape_to_minibatches, (f_obs, f_mask, f_act, f_r, f_nobs, f_nmask, f_d)
            )

            # PHASE 2: DISCRETE SAC 1-STEP TD OPTIMIZATION
            def update_minibatch(carry, batch_data):
                ca_act, ca_crit, ca_targ, ca_alpha, o_act, o_crit, o_al, mb_step = carry
                b_obs, b_mask, b_act, b_r, b_nobs, b_nmask, b_d = batch_data
                
                alpha = jnp.exp(ca_alpha)
                
                # 1. Calculate Target Q-Value using 1-Step TD
                next_logits = self.actor.apply(ca_act, b_nobs, b_nmask)
                next_probs = jax.nn.softmax(next_logits)
                next_log_probs = jnp.where(b_nmask, jax.nn.log_softmax(next_logits), 0.0) # Mask log(0) NaNs
                
                next_q1, next_q2 = self.critic.apply(ca_targ, b_nobs)
                next_q_min = jnp.minimum(next_q1, next_q2)
                
                # V(s') = Sum over a [ pi(a|s') * (Q(s', a) - alpha * log_pi(a|s')) ]
                next_v = jnp.sum(next_probs * (next_q_min - alpha * next_log_probs), axis=-1)
                q_target = jax.lax.stop_gradient(b_r + self.gamma * (1.0 - b_d) * next_v)

                # 2. Critic Update
                def critic_loss_fn(p_c):
                    q1, q2 = self.critic.apply(p_c, b_obs)
                    # Extract the Q-values for the specific actions that were actually taken
                    q1_a = jnp.take_along_axis(q1, b_act[..., None], axis=-1).squeeze(-1)
                    q2_a = jnp.take_along_axis(q2, b_act[..., None], axis=-1).squeeze(-1)
                    return 0.5 * jnp.mean((q1_a - q_target)**2) + 0.5 * jnp.mean((q2_a - q_target)**2)

                c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(ca_crit)
                c_grads = jax.lax.pmean(c_grads, axis_name='p')
                c_updates, o_crit = self.opt_critic.update(c_grads, o_crit)
                ca_crit = optax.apply_updates(ca_crit, c_updates)

                # 3. Actor & Alpha Update (Delayed)
                def actor_update(ca_act, o_act, ca_alpha, o_al):
                    def actor_loss_fn(p_a):
                        logits = self.actor.apply(p_a, b_obs, b_mask)
                        probs = jax.nn.softmax(logits)
                        log_probs = jnp.where(b_mask, jax.nn.log_softmax(logits), 0.0)
                        
                        q1, q2 = self.critic.apply(jax.lax.stop_gradient(ca_crit), b_obs)
                        q_min = jnp.minimum(q1, q2)
                        
                        # Actor minimizes KL-Divergence: pi * (alpha * log_pi - Q)
                        loss = jnp.sum(probs * (alpha * log_probs - q_min), axis=-1).mean()
                        
                        # Calculate policy entropy for the Alpha update
                        entropy = -jnp.sum(probs * log_probs, axis=-1).mean()
                        return loss, entropy

                    (a_loss, entropy), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(ca_act)
                    a_grads = jax.lax.pmean(a_grads, axis_name='p')
                    a_updates, o_act = self.opt_actor.update(a_grads, o_act)
                    ca_act = optax.apply_updates(ca_act, a_updates)
                    
                    # Update Temperature (Alpha)
                    def alpha_loss_fn(p_al):
                        return -jnp.exp(p_al) * jax.lax.stop_gradient(entropy - self.target_entropy)
                    
                    al_loss, al_grads = jax.value_and_grad(alpha_loss_fn)(ca_alpha)
                    al_grads = jax.lax.pmean(al_grads, axis_name='p')
                    al_updates, o_al = self.opt_alpha.update(al_grads, o_al)
                    ca_alpha = optax.apply_updates(ca_alpha, al_updates)
                    
                    # Soft Target Polyak Update
                    ca_targ_new = jax.tree.map(lambda t, c: self.tau * c + (1 - self.tau) * t, ca_targ, ca_crit)
                    return ca_act, o_act, ca_alpha, o_al, ca_targ_new, a_loss

                # Delayed update logic
                do_actor_update = (mb_step % 2 == 0)
                ca_act, o_act, ca_alpha, o_al, ca_targ, a_loss = jax.lax.cond(
                    do_actor_update,
                    lambda _: actor_update(ca_act, o_act, ca_alpha, o_al),
                    lambda _: (ca_act, o_act, ca_alpha, o_al, ca_targ, 0.0),
                    operand=None
                )

                return (ca_act, ca_crit, ca_targ, ca_alpha, o_act, o_crit, o_al, mb_step + 1), c_loss

            carry = (p_actor, p_critic, p_target, p_alpha, o_actor, o_critic, o_alpha, global_step)
            carry, batch_losses = jax.lax.scan(update_minibatch, carry, (mb_obs, mb_mask, mb_act, mb_r, mb_nobs, mb_nmask, mb_d))
            p_actor, p_critic, p_target, p_alpha, o_actor, o_critic, o_alpha, new_global_step = carry
            
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

        logger.info(f"🚀 STRICT DISCRETE SAC ONLINE.")
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
                print(f"| DISCRETE SAC v1         |            |")
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
                logger.info(f"💾 Discrete SAC Checkpoint Saved.")
            epoch += 1

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.params = pickle.load(f)
            return True
        return False