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

np.random.seed(2021)

data_dir = 'data/inv_pendulum/'

env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


class CustomLossNLL(tf.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        mean, log_sigma = tf.split(y_pred, 2, axis=-1)
        y_target, temp =tf.split(y_true,2,axis=-1)
        sigma = tf.nn.softplus(log_sigma)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        loss = -tf.reduce_mean(dist.log_prob(y_target))
        return loss


def parse_tfr_dynamics(element):
  data = {
      'img_height': tf.io.FixedLenFeature([], tf.int64),
      'img_width':tf.io.FixedLenFeature([], tf.int64),
      'img_depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'prev_state' : tf.io.FixedLenFeature([], tf.string),}
  content = tf.io.parse_single_example(element, data)
  state_size = content['state_size']
  raw_state = content['state']
  raw_prev_state = content['prev_state']
  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])
  prev_state = tf.io.parse_tensor(raw_prev_state, out_type=tf.float64)
  prev_state = tf.reshape(prev_state, shape=[state_size])
  return (prev_state, state)


def get_dynamics_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_dynamics)
    return pendulum_dataset


def parse_tfr_observation(element):
  data = {
      'img_height': tf.io.FixedLenFeature([], tf.int64),
      'img_width':tf.io.FixedLenFeature([], tf.int64),
      'img_depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'prev_state' : tf.io.FixedLenFeature([], tf.string),}
  content = tf.io.parse_single_example(element, data)
  height = content['img_height']
  width = content['img_width']
  depth = content['img_depth']
  raw_image = content['raw_image']
  state_size = content['state_size']
  raw_state = content['state']
  image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
  image = tf.reshape(image, shape=[height,width,depth])
  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])
  return (image, state)


def get_observation_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_observation)
    return pendulum_dataset


def build_dynamics_model():
    model = tf.keras.Sequential([
        layers.Dense(32, activation=tf.nn.relu, input_shape=[5]),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(5*2)
    ])
    model.compile(optimizer='adam', loss= [CustomLossNLL()])
    model.summary()
    return model



if __name__=='__main__':
    obs_dataset = get_observation_dataset()
    dyn_dataset = get_dynamics_dataset()
    for element in obs_dataset.take(1):
        print(element[0])
    for element in dyn_dataset.take(1):
        print(element[0])

    # dyn_dataset = get_dynamics_dataset()
    # dyn_model = build_dynamics_model()
    # # dynamics_model.fit(train_prev_state[permutation,0:5], train_state[permutation,:],epochs=EPOCHS, validation_split = 0.8, verbose=1)
    # dyn_model.fit(dyn_dataset, verbose=1)




pass