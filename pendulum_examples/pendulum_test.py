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



gym_env = CartPoleEnvNoise(1.0)
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
    model.load_weights("/home/local/ASUAD/gmclark1/Research/git/Bayesian_NN_Filter/models/dyn2/dyn_test2.ckpt")
    return model


def build_timedistributed_observation_model():
    input_layer = tf.keras.Input(shape=(2,75,300,1))

    encode_1 = layers.TimeDistributed(layers.Conv2D(32, kernel_size=5, strides=(3,3), padding='same', activation='relu', kernel_initializer='he_uniform'))(input_layer)
    encode_2 = layers.TimeDistributed(layers.Conv2D(32, kernel_size=4, strides=(2,2), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))(encode_1)
    encode_3 = layers.TimeDistributed(layers.Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))(encode_2)
    flaten_4 = layers.TimeDistributed(layers.Flatten())(encode_3)
    deeeep_5 = layers.TimeDistributed(layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform'))(flaten_4)

    flaten_6 = layers.Flatten()(deeeep_5)

    deeeep_7 = layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform')(flaten_6)
    deeeep_8 = layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform')(deeeep_7)
    deeeep_9 = layers.Dense(64, activation=tf.nn.relu, kernel_initializer='he_uniform')(deeeep_8)
    deeeep_10 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer='he_uniform')(deeeep_9)
    output_layer = layers.Dense(5*2)(deeeep_10)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss= [CustomLossNLL()])
    model.summary()
    model.load_weights("/home/local/ASUAD/gmclark1/Research/data/pendulum/models/test_0/test_0")
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
            img = np.abs((gray/256)-1)
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