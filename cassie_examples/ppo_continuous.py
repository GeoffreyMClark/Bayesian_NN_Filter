# import wandb
import tensorflow as tf
from tensorflow import random
from tensorflow.keras.layers import Input, Dense, Lambda
from cassie import CassieEnv

import gym
import argparse
import numpy as np
from numba import jit, njit

from time import perf_counter

tf.keras.backend.set_floatx('float64')
# Try this to see if its faster
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=64)
parser.add_argument('--actor_lr', type=float, default=0.00005)
parser.add_argument('--critic_lr', type=float, default=0.0001)
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--lmbda', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=2)

args = parser.parse_args()


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def get_action(self, state):
        state = tf.reshape(state, [1, self.state_dim])
        mu, std = self.model(state, training=False)
        action = np.random.normal(mu[0], std[0], size=self.action_dim)
        action = np.clip(action, -self.action_bound, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)
        return log_policy, action

    @tf.function(jit_compile=True)
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(128, activation='relu')(state_input)
        dense_2 = Dense(128, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    @tf.function(jit_compile=True)
    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(ratio, 1.0-args.clip_ratio, 1.0+args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    @tf.function(jit_compile=True)
    def train(self, log_old_policy, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            # Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        # model.compile(optimizer='adam', loss="mse")
        return model

    @tf.function(jit_compile=True)
    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    @tf.function(jit_compile=True)
    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.actor_opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(args.critic_lr)
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + args.gamma * forward_val - v_values[k]
            gae_cumulative = args.gamma * args.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        return np.asarray(list).reshape(len(list), len(list[0][0]))

    def train(self, max_episodes=100000):
        for ep in range(max_episodes):
            time0=perf_counter()
            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []

            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                self.env.render()
                log_old_policy, action = self.actor.get_action(state)

                next_state, reward, done, info = self.env.step(action)
                # print('Cost Motion - ',info['cost_motion'])
                # print('Cost Ctrl   - ',info['cost_ctrl'])
                # print('Reward stand- ',info['reward_stand'])
                # print('')

                state = state.reshape(1,-1)
                action = action.reshape(1,-1)
                next_state = next_state.reshape(1,-1)
                reward = np.asarray([reward]).reshape(1,-1)
                log_old_policy = np.reshape(log_old_policy, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                old_policy_batch.append(log_old_policy)

                if len(state_batch) >= args.update_interval or done:
                    
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.critic.model(states, training=False)
                    next_v_value = self.critic.model(next_state, training=False)
                    
                    gaes, td_targets = self.gae_target(rewards, v_values, next_v_value, done)

                    for epoch in range(args.epochs):
                        actor_loss = self.actor.train(old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]
            timeend=perf_counter()
            print('EP{} EpisodeReward={}   in {}seconds'.format(ep, episode_reward, timeend-time0))
            


def main():
    # env_name = 'CartPole-v1'
    # env_name = 'Pendulum-v1'
    # env_name = 'MountainCarContinuous-v0'
    # env = gym.make(env_name)
    env = CassieEnv()
    agent = Agent(env)
    agent.train()


if __name__ == "__main__":
    main()

# TO DO:
# 4. change observations to encoder and imu sensors


# Before optimization
# 21.805s average

# Basic optimization
# 5.14s average
# 3.04s with update_interval=10 instead of 5
