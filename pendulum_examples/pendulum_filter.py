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

data_dir = 'data/inv_pendulum/working1/'
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


# @tf.function()
def parse_tfr_dynamics(element):
  data = {
      'img_height': tf.io.FixedLenFeature([], tf.int64),
      'img_width':tf.io.FixedLenFeature([], tf.int64),
      'img_depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
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
    total_images=0
    for idx, file in enumerate(files):
        try:
            total_images += sum([1 for _ in tf.io.tf_record_iterator(file)]) # Check corrupted tf records
        except:
            print("{}: {} is corrupted".format(idx, file))
    print("Succeed, no corrupted tf records found for {} images".format(total_images))
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_dynamics)#, num_parallel_calls=tf.data.AUTOTUNE)
    return pendulum_dataset





# def parse_tfr_observation(element):
#   data = {
#       'img_height': tf.io.FixedLenFeature([], tf.int64),
#       'img_width':tf.io.FixedLenFeature([], tf.int64),
#       'img_depth':tf.io.FixedLenFeature([], tf.int64),
#       'raw_image' : tf.io.FixedLenFeature([], tf.string),
#       'state_size':tf.io.FixedLenFeature([], tf.int64),
#       'state' : tf.io.FixedLenFeature([], tf.string),
#       'prev_state' : tf.io.FixedLenFeature([], tf.string),}
#   content = tf.io.parse_single_example(element, data)
#   height = content['img_height']
#   width = content['img_width']
#   depth = content['img_depth']
#   raw_image = content['raw_image']
#   state_size = content['state_size']
#   raw_state = content['state']
#   image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
#   image = tf.reshape(image, shape=[height,width,depth])
#   state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
#   state = tf.reshape(state, shape=[state_size])
#   return (image, state)


# def get_observation_dataset(tfr_dir:str=data_dir, pattern:str="pendulum.tfrecords"):
#     files = glob.glob(tfr_dir+pattern, recursive=False)
#     pendulum_dataset = tf.data.TFRecordDataset(files)
#     pendulum_dataset = pendulum_dataset.map(parse_tfr_observation)
#     return pendulum_dataset


def build_dynamics_model(model_name):
    model = tf.keras.Sequential([
        layers.Dense(128, activation=tf.nn.relu, input_shape=[5]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(5*2)
    ])
    model.compile(optimizer='adam', loss= [CustomLossNLL()])
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


def run_model(environment, model):
    zero_vec = np.zeros((1,5))
    time_step = environment.reset()
    obs=time_step.observation.numpy()
    state = np.concatenate((obs, np.array([0]).reshape(1,1)), axis=1)
    while not time_step.is_last():
    # for i in range(1000):
        # use ground truth state
        environment.render()
        state_pred = model(state, training=False)
        action = state_pred.numpy()[0,4]
        if action >=  .5:
            action = int(1)
        elif action < .5:
            action = int(0)
        time_step = environment.step(action)
        obs=time_step.observation.numpy()
        state = np.concatenate((obs, np.array([action]).reshape(1,1)), axis=1)

        # use predicted state as next state
        # environment.render()
        # state_pred = model(state, training=False)
        # action = int(state_pred.numpy()[0,4])
        # environment.step(action)
        # state=state_pred.numpy()[0,0:5]

        







if __name__=='__main__':
    # obs_dataset = get_observation_dataset()

    dyn_model_name = 'working1'
    val_size = 2048
    dyn_dataset = get_dynamics_dataset()
    dyn_valid = dyn_dataset.take(val_size).batch(val_size)
    # dyn_train = dyn_dataset.skip(val_size).shuffle(buffer_size=22000).batch(256).prefetch(tf.data.AUTOTUNE)
    dyn_train = dyn_dataset.skip(val_size).batch(32).cache().prefetch(tf.data.AUTOTUNE)
    # dyn_train = dyn_dataset.skip(val_size).batch(64)
    dyn_model, tf_callback = build_dynamics_model(dyn_model_name)

    dyn_model.fit(dyn_train, validation_data=dyn_valid, epochs=150, verbose=1, callbacks=tf_callback)
    run_model(eval_env, dyn_model)

 


pass
# 39 40 41 49