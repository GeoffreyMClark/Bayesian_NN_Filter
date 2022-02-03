from __future__ import absolute_import, division, print_function
import base64
from cmath import nan
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
from tensorflow.keras import layers, Model, models
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
    model.load_weights("/home/geoffrey/Research/data/inv_pendulum/models/dyn0/dyn0")
    return model


def build_timedistributed_observation_model():
    input_layer = tf.keras.Input(shape=(2,75,300,1))

    encode_1 = layers.TimeDistributed(layers.Conv2D(32, kernel_size=5, strides=(3,3), padding='same', activation='relu', kernel_initializer='he_uniform'))(input_layer)
    encode_2 = layers.TimeDistributed(layers.Conv2D(32, kernel_size=4, strides=(1,1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))(encode_1)
    encode_3 = layers.TimeDistributed(layers.Conv2D(32, kernel_size=3, strides=(1,1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))(encode_2)
    flaten_4 = layers.TimeDistributed(layers.Flatten())(encode_3)
    deeeep_5 = layers.TimeDistributed(layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform'))(flaten_4)

    flaten_6 = layers.Flatten()(deeeep_5)

    deeeep_7 = layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform')(flaten_6)
    deeeep_8 = layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform')(deeeep_7)
    deeeep_9 = layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform')(deeeep_8)
    deeeep_10 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer='he_uniform')(deeeep_9)
    output_layer = layers.Dense(5*2)(deeeep_10)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss= [CustomLossNLL()])
    model.summary()
    model.load_weights("/home/geoffrey/Research/data/inv_pendulum/models/obs3/obs3")
    return model

def run_models(env, dyn_model, obs_model):
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




def run_filter(env, dyn_model, obs_model):
    savename = 'Pendulum_nonoise'
    video_full = cv.VideoWriter(savename+'.avi', 0, 30, (600,400))

    time_step = env.reset()
    state = np.concatenate((time_step.observation.numpy(), np.array([0]).reshape(1,1), np.zeros([1,5])), axis=1)
    test_state=[]; test_dyn=[]; test_obs=[]; test_filter=[]

    cycle_state = state

    for j in range(500):
        if not time_step.is_last():
            img_full=env.render(mode='rgb_array').numpy().reshape(400,600,3)
            gray = cv.cvtColor(img_full, cv.COLOR_BGR2GRAY).reshape(400,600,1)

            # speckle noise
            # gauss = np.random.normal(0,.1,gray.size)
            # gauss = gauss.reshape(gray.shape[0],gray.shape[1], gray.shape[2]).astype('uint8')
            # noise = gray + gray * gauss

            # Salt noise
            # gauss = np.random.normal(0,1,gray.size)
            # gauss = gauss.reshape(gray.shape[0],gray.shape[1], gray.shape[2]).astype('uint8')
            # noise = cv.add(gray,gauss).reshape(400,600,1)

            # Gaussian Blur
            # noise = cv.GaussianBlur(gray.reshape(400,600), (7, 7), 0).reshape(400,600,1)

            # No noise
            noise = gray

            video_full.write(noise)

            # cv.imshow("full_img", noise)
            # cv.waitKey(1)

            cut = cv.pyrDown(noise[167:317,:,:])
            img = np.abs((cut/255)-1)
            if j == 0:
                prev_img = img
            obs = np.concatenate((img.reshape(1,75,300,1), prev_img.reshape(1,75,300,1)), axis=0)
            obs = obs.reshape(1,2,75,300,1)
            obs_pred = obs_model(obs, training=False)
            obs_mean, obs_logvar = np.split(obs_pred, 2, axis=-1)
            obs_var = tf.nn.softplus(obs_logvar).numpy()

            # state_pred = dyn_model(cycle_state[-1,0:5].reshape(1,-1), training=False)
            state_pred = dyn_model(state[0,0:5].reshape(1,-1), training=False)
            dyn_mean, dyn_logvar = np.split(state_pred, 2, axis=-1)
            dyn_var = tf.nn.softplus(dyn_logvar).numpy()

            proc_var = dyn_var + cycle_state[-1,5:10]
            filtered_state = (obs_var/(proc_var+obs_var))*dyn_mean + (proc_var/(proc_var+obs_var))*obs_mean
            filtered_var = proc_var*(proc_var+obs_var)*obs_var
            new_state = np.concatenate((filtered_state, filtered_var), axis=1)

            cycle_state = np.concatenate((cycle_state, new_state), axis=0)

            action = state_pred.numpy()[0,4]
            if action >=  .5:
                action = int(1)
            elif action < .5:
                action = int(0)
            elif np.isnan(action):
                action = int(0)
            print(action)
            time_step = env.step(action)
            state = np.concatenate((time_step.observation.numpy(), np.array([action]).reshape(1,1), np.zeros([1,5])), axis=1)

            prev_img = img
            test_state.append(state)
            test_dyn.append(state_pred)
            test_obs.append(obs_pred)
            test_filter.append(new_state)
        else:
            break

    cv.destroyAllWindows()
    video_full.release()

    test_state = np.asarray(test_state).reshape(-1,10)
    test_dyn = np.asarray(test_dyn).reshape(-1,10)
    test_obs = np.asarray(test_obs).reshape(-1,10)
    test_filter = np.asarray(test_filter).reshape(-1,10)

    plt.figure(1, figsize=(10,10))
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(test_state[:,i], c='#173f5f')

        # plt.plot(test_dyn[:,i], c='b')
        # plt.fill_between(np.linspace(0,test_dyn.shape[0]-1,test_dyn.shape[0]), test_dyn[:,i]+tf.nn.softplus(test_dyn[:,i+5]), test_dyn[:,i]-tf.nn.softplus(test_dyn[:,i+5]), facecolor='b', alpha=.2)

        plt.plot(test_obs[:,i], c=[.4705, .7921, .6470])
        plt.fill_between(np.linspace(0,test_obs.shape[0]-1,test_obs.shape[0]), test_obs[:,i]+tf.nn.softplus(test_obs[:,i+5]), test_obs[:,i]-tf.nn.softplus(test_obs[:,i+5]), facecolor='g', alpha=.2)

        plt.plot(test_filter[:,i], c='b')
        plt.fill_between(np.linspace(0,test_filter.shape[0]-1,test_filter.shape[0]), test_filter[:,i]+test_filter[:,i+5], test_filter[:,i]-test_filter[:,i+5], facecolor='b', alpha=.2)
    


    plt.subplot(5,1,1)
    plt.ylabel('Cart Position')
    plt.subplot(5,1,2)
    plt.ylabel('Cart Velocity')
    plt.subplot(5,1,3)
    plt.ylabel('Pole Position')
    plt.subplot(5,1,4)
    plt.ylabel('Pole Velocity')
    plt.subplot(5,1,5)
    i=4
    plt.plot(test_dyn[:,i], c='b')
    plt.fill_between(np.linspace(0,test_dyn.shape[0]-1,test_dyn.shape[0]), test_dyn[:,i]+tf.nn.softplus(test_dyn[:,i+5]), test_dyn[:,i]-tf.nn.softplus(test_dyn[:,i+5]), facecolor='b', alpha=.2)
    plt.ylabel('Control Force')
    plt.ylim([-.25, 1.25])
    plt.xlabel('Time (t)')
    plt.savefig(savename+'.pdf', bbox_inches='tight')
    plt.show()








if __name__=='__main__':
    dyn_model = build_dynamics_model()
    # obs_model = build_timedistributed_observation_model()
    obs_model = models.load_model('/home/geoffrey/Research/data/inv_pendulum/models/obs3/obs3', custom_objects={'CustomLossNLL': CustomLossNLL()})

    # run_models(eval_env, dyn_model, obs_model)

    run_filter(eval_env, dyn_model, obs_model)
