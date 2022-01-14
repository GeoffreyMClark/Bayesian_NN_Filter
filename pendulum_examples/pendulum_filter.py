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

data_dir = 'data/inv_pendulum/test19/'
# data_dir = 'data/stored/'

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


class CustomLossMSEReduced(tf.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        # mean1, mean2, mean3, mean4, mean5, log_sigma1, log_sigma2, log_sigma3, log_sigma4, log_sigma5 = tf.split(y_pred, 10, axis=-1)
        # true1, true2, true3, true4, true5, true_sigma1, true_sigma2, true_sigma3, true_sigma4, true_sigma5 = tf.split(y_pred, 10, axis=-1)

        mean1, log_sigma1 = tf.split(y_pred, 2, axis=-1)
        true1, true_sigma1 = tf.split(y_pred, 2, axis=-1)

        loss = tf.reduce_mean(tf.math.sqrt(mean1 - true1))
        return loss


# @tf.function()
def parse_tfr_dynamics(element):
  data = {
      'img_height': tf.io.FixedLenFeature([], tf.int64),
      'img_width':tf.io.FixedLenFeature([], tf.int64),
      'img_depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
    #   'prev_raw_image' : tf.io.FixedLenFeature([], tf.string),
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'prev_state_size':tf.io.FixedLenFeature([], tf.int64),
      'prev_state' : tf.io.FixedLenFeature([], tf.string),}
  content = tf.io.parse_single_example(element, data)
#   content = tf.io.parse_example(element, data)
  state_size = content['state_size']
  prev_state_size = content['prev_state_size']
  raw_state = content['state']
  raw_prev_state = content['prev_state']
  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])
  prev_state = tf.io.parse_tensor(raw_prev_state, out_type=tf.float64)
  prev_state = tf.reshape(prev_state, shape=[prev_state_size])
  return (prev_state, state)


def get_dynamics_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_dynamics)#, num_parallel_calls=tf.data.AUTOTUNE)
    return pendulum_dataset


def parse_tfr_observation(element):
  data = {
      'img_height': tf.io.FixedLenFeature([], tf.int64),
      'img_width':tf.io.FixedLenFeature([], tf.int64),
    #   'img_depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'prev_raw_image' : tf.io.FixedLenFeature([], tf.string),
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'prev_state' : tf.io.FixedLenFeature([], tf.string),}
  content = tf.io.parse_single_example(element, data)
  height = content['img_height']
  width = content['img_width']
#   depth = content['img_depth']
  raw_image = content['raw_image']
  prev_raw_image = content['prev_raw_image']
  state_size = content['state_size']
  raw_state = content['state']
  image = tf.io.parse_tensor(raw_image, out_type=tf.float16)
  image = tf.reshape(image, shape=[height,width,1])

  prev_image = tf.io.parse_tensor(prev_raw_image, out_type=tf.float16)
  prev_image = tf.reshape(prev_image, shape=[height,width,1])

  img = tf.concat((image, prev_image), axis=2)

  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])
  return (img, state)


def get_observation_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_observation)
    return pendulum_dataset


def build_dynamics_model(model_name):
    model = tf.keras.Sequential([
        layers.Dense(128, activation=tf.nn.relu, input_shape=[5]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(5*2)
    ])
    model.compile(optimizer='adam', loss=[CustomLossNLL()])
    model.summary()
    tf_callback = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_name + "_{epoch}",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            verbose=0,
        )
    ]
    return model, tf_callback


def build_observation_model(model_name):
    model = tf.keras.Sequential([
        layers.Conv2D(128, kernel_size=5, strides=(3,3), padding='valid', activation='relu', input_shape=(75,300,2)),
        layers.Conv2D(64, kernel_size=4, strides=(2,2), padding='valid', activation='relu'),
        # layers.Conv2D(64, kernel_size=3, strides=(1,1), padding='valid', activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation=tf.nn.relu, kernel_initializer='he_uniform'),
        layers.Dense(32, activation=tf.nn.relu, kernel_initializer='he_uniform'),
        layers.Dense(5*2)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss= [CustomLossNLL()])
    # model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    model.summary()
    tf_callback = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_name + "_{epoch}",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            verbose=0,
        )
    ]
    return model, tf_callback


def run_model(env, dyn_model):
    zero_vec = np.zeros((1,5))
    time_step = env.reset()
    obs=time_step.observation.numpy()
    state = np.concatenate((obs, np.array([0]).reshape(1,1)), axis=1)
    test_state=[]; test_dyn=[]; test_obs=[]
    video_full = cv.VideoWriter('video_full_3.avi', 0, 30, (600,400))
    video_sliced = cv.VideoWriter('video_sliced_3.avi', 0, 30, (300,75))

    while not time_step.is_last():
    # for i in range(500):
        # use ground truth state
        img_full=env.render(mode='rgb_array').numpy().reshape(400,600,3)
        video_full.write(img_full)

        img_full = cv.pyrDown(img_full[167:317,:,:])
        gray = cv.cvtColor(img_full, cv.COLOR_BGR2GRAY)
        img = gray/256

        video_sliced.write(gray)


        # obs_pred = obs_model(img.reshape(1,75,300,1))
        state_pred = dyn_model(state, training=False)
        action = state_pred.numpy()[0,4]
        if action >=  .5:
            action = int(1)
        elif action < .5:
            action = int(0)
        time_step = env.step(action)
        obs=time_step.observation.numpy()
        state = np.concatenate((obs, np.array([action]).reshape(1,1)), axis=1)

        test_state.append(state)
        test_dyn.append(state_pred)
        # test_obs.append(obs_pred)

    cv.destroyAllWindows()
    video_full.release()
    video_sliced.release()

    test_state = np.asarray(test_state).reshape(-1,5)
    test_dyn = np.asarray(test_dyn).reshape(-1,10)
    dynamics_mean, dynamics_log_sigma = tf.split(test_dyn, 2, axis=-1)
    dynamics_sigma = np.sqrt(tf.nn.softplus(dynamics_log_sigma))

    # obs_mean, obs_log_sigma = tf.split(test_obs, 2, axis=-1)
    # obs_mean=obs_mean.numpy().reshape(-1,5)
    # obs_sigma = np.sqrt(tf.nn.softplus(obs_log_sigma)).reshape(-1,5)


    plt.figure(10)
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(test_state[:,i], color='k')
        plt.plot(dynamics_mean[:,i], color='b')
        plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,i]+dynamics_sigma[:,i], dynamics_mean[:,i]-dynamics_sigma[:,i], facecolor='b', alpha=.2)
        # plt.plot(obs_mean[:,i], color='g')
        # plt.fill_between(np.linspace(0,obs_mean.shape[0],obs_mean.shape[0]), obs_mean[:,i]+obs_sigma[:,i], obs_mean[:,i]-obs_sigma[:,i], facecolor='b', alpha=.2)
    plt.show()

    pass










if __name__=='__main__':
    # data14 totally works for the dynamics model, can stay up pretty well
    # dyn_model_name = 'dynamics'
    # val_size = 2048
    # dyn_dataset = get_dynamics_dataset()
    # dyn_dataset = dyn_dataset.apply(tf.data.experimental.ignore_errors())
    # dyn_valid = dyn_dataset.take(val_size).batch(val_size)
    # # dyn_train = dyn_dataset.skip(val_size).shuffle(buffer_size=22000).batch(256).prefetch(tf.data.AUTOTUNE)
    # dyn_train = dyn_dataset.skip(val_size).batch(32).cache().prefetch(tf.data.AUTOTUNE)
    # dyn_model, tf_callback1 = build_dynamics_model(dyn_model_name)
    # dyn_model.fit(dyn_train, validation_data=dyn_valid, epochs=150, verbose=1) #, callbacks=tf_callback)

    obs_model_name = 'observation'
    val_size = 2048
    obs_dataset = get_observation_dataset()
    obs_dataset = obs_dataset.apply(tf.data.experimental.ignore_errors())
    obs_valid = obs_dataset.take(val_size).batch(val_size)
    # dyn_train = dyn_dataset.skip(val_size).shuffle(buffer_size=22000).batch(256).prefetch(tf.data.AUTOTUNE)
    obs_train = obs_dataset.skip(val_size).batch(32).cache().prefetch(tf.data.AUTOTUNE)
    obs_model, tf_callback2 = build_observation_model(obs_model_name)
    obs_model.fit(obs_train, validation_data=obs_valid, epochs=80, verbose=1) #, callbacks=tf_callback)







    # run_model(eval_env, dyn_model)




pass
