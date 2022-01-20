from __future__ import absolute_import, division, print_function
import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import reverb
import zlib
import os
import cv2 as cv
import glob

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from cartpole_noise import CartPoleEnvNoise

np.random.seed(2021)

gym_env = CartPoleEnvNoise()
env = suite_gym.wrap_env(gym_env)

# env_name = 'CartPole-v0'
# env = suite_gym.load(env_name)
eval_env = tf_py_environment.TFPyEnvironment(env)


class CustomLossNLL(tf.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        mean, log_sigma = tf.split(y_pred, 2, axis=-1)
        y_target, temp =tf.split(y_true,2,axis=-1)
        sigma = tf.nn.softplus(log_sigma)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        loss = -tf.reduce_mean(dist.log_prob(y_target))
        return loss


def build_dynamics_model():
    model = tf.keras.Sequential([
        layers.Dense(128, activation=tf.nn.relu, input_shape=[5]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(5*2)
    ])
    model.compile(optimizer='adam', loss=[CustomLossNLL()])
    model.summary()
    model.load_weights("models/dyn2/dyn_test2.ckpt")
    return model


def build_observation_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=5, strides=(3,3), padding='valid', activation='relu', input_shape=(75,300,2)),
        layers.Conv2D(64, kernel_size=4, strides=(2,2), padding='valid', activation='relu'),
        layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='valid', activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation=tf.nn.relu, kernel_initializer='he_uniform'),
        layers.Dense(32, activation=tf.nn.relu, kernel_initializer='he_uniform'),
        layers.Dense(5*2)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss= [CustomLossNLL()])
    model.summary()
    model.load_weights("models/obs2/obs_test2.ckpt")
    return model

def build_timedistributed_observation_model():
    input_layer = tf.keras.Input(shape=(2,75,300,1))

    encode_1 = layers.TimeDistributed(layers.Conv2D(64, kernel_size=5, strides=(3,3), padding='valid', activation='relu'))(input_layer)
    encode_2 = layers.TimeDistributed(layers.Conv2D(64, kernel_size=4, strides=(2,2), padding='valid', activation='relu'))(encode_1)
    encode_3 = layers.TimeDistributed(layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='valid', activation='relu'))(encode_2)
    flatten_1 = layers.Flatten()(encode_3)
    deep_1 = layers.Dense(64, activation=tf.nn.relu, kernel_initializer='he_uniform')(flatten_1)
    deep_2 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer='he_uniform')(deep_1)
    output_layer = layers.Dense(5*2)(deep_2)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss= [CustomLossNLL()])
    # model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    model.summary()
    model.load_weights("models/obs3/obs_test3.ckpt")
    return model

def run_model(env, dyn_model, obs_model):
    zero_vec = np.zeros((1,5))
    time_step = env.reset()
    obs=time_step.observation.numpy()
    state = np.concatenate((obs, np.array([0]).reshape(1,1)), axis=1)
    test_state=[]; test_dyn=[]; test_obs=[]
    # video_full = cv.VideoWriter('video_full_3.avi', 0, 30, (600,400))
    # video_sliced = cv.VideoWriter('video_sliced_3.avi', 0, 30, (300,75))

    for j in range(200):
        if not time_step.is_last():
            # use ground truth state
            img_full=env.render(mode='rgb_array').numpy().reshape(400,600,3)
            img_full = cv.pyrDown(img_full[167:317,:,:])
            gray = cv.cvtColor(img_full, cv.COLOR_BGR2GRAY)
            img = gray/256
            if j == 0:
                prev_img = img

            obs = np.concatenate((img.reshape(1,75,300,1), prev_img.reshape(1,75,300,1)), axis=0)
            obs = obs.reshape(1,2,75,300,1)

            obs_pred = obs_model(obs, training=False)
            state_pred = dyn_model(state, training=False)
            action = state_pred.numpy()[0,4]
            if action >=  .5:
                action = int(1)
            elif action < .5:
                action = int(0)
            time_step = env.step(action)
            obs=time_step.observation.numpy()
            state = np.concatenate((obs, np.array([action]).reshape(1,1)), axis=1)

            prev_img = img
            test_state.append(state)
            test_dyn.append(state_pred)
            test_obs.append(obs_pred)

            # video_full.write(img_full)
            # video_sliced.write(gray)

        else:
            break

    # cv.destroyAllWindows()
    # video_full.release()
    # video_sliced.release()

    test_state = np.asarray(test_state).reshape(-1,5)
    test_dyn = np.asarray(test_dyn).reshape(-1,10)
    dynamics_mean, dynamics_log_sigma = tf.split(test_dyn, 2, axis=-1)
    dynamics_sigma = np.sqrt(tf.nn.softplus(dynamics_log_sigma))

    obs_mean, obs_log_sigma = tf.split(test_obs, 2, axis=-1)
    obs_mean=obs_mean.numpy().reshape(-1,5)
    obs_sigma = np.sqrt(tf.nn.softplus(obs_log_sigma)).reshape(-1,5)


    plt.figure(10)
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(test_state[:,i], color='k')
        plt.plot(dynamics_mean[:,i], color='b')
        plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,i]+dynamics_sigma[:,i], dynamics_mean[:,i]-dynamics_sigma[:,i], facecolor='b', alpha=.2)
        plt.plot(obs_mean[:,i], color='g')
        plt.fill_between(np.linspace(0,obs_mean.shape[0],obs_mean.shape[0]), obs_mean[:,i]+obs_sigma[:,i], obs_mean[:,i]-obs_sigma[:,i], facecolor='g', alpha=.2)
    plt.show()




if __name__=='__main__':
    dyn_model = build_dynamics_model()
    obs_model = build_timedistributed_observation_model()

    run_model(eval_env, dyn_model, obs_model)