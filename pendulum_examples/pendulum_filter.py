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
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment


data_dir = 'data/inv_pendulum/'

env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'img_height': tf.io.FixedLenFeature([], tf.int64),
      'img_width':tf.io.FixedLenFeature([], tf.int64),
      'img_depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'action_size':tf.io.FixedLenFeature([], tf.int64),
      'action' : tf.io.FixedLenFeature([], tf.string),
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'next_state' : tf.io.FixedLenFeature([], tf.string),
    }
  content = tf.io.parse_single_example(element, data)
  height = content['img_height']
  width = content['img_width']
  depth = content['img_depth']
  raw_image = content['raw_image']
  action_size = content['action_size']
  raw_action = content['action']
  state_size = content['state_size']
  raw_state = content['state']
  raw_next_state = content['next_state']

  #get our 'feature'-- our image -- and reshape it appropriately
  image = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
  image = tf.reshape(image, shape=[height,width,depth])

  action = tf.io.parse_tensor(raw_action, out_type=tf.float64)
  action = tf.reshape(action, shape=[action_size])

  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])

  next_state = tf.io.parse_tensor(raw_next_state, out_type=tf.float64)
  next_state = tf.reshape(next_state, shape=[state_size])

  return (image, action, state, next_state)


def get_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    #create the dataset
    pendulum_dataset = tf.data.TFRecordDataset(files)
    #pass every single feature through our mapping function
    pendulum_dataset = pendulum_dataset.map(parse_tfr_element)
    return pendulum_dataset



if __name__=='__main__':
    pendulum_dataset = get_dataset()
    list(pendulum_dataset.as_numpy_iterator())
    for element in pendulum_dataset.take(100):
        print(element[2])





pass