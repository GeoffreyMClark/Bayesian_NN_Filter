from __future__ import absolute_import, division, print_function
import base64
# import imageio
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
# import PIL.Image
# import reverb
import zlib
import os
import cv2 as cv
import glob

import tensorflow as tf
import tensorflow_probability as tfp
import tensorboard
tfd = tfp.distributions
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
# from tf_agents.environments import suite_gym
# from tf_agents.environments import tf_py_environment
# from cartpole_noise import CartPoleEnvNoise

# np.random.seed(2021)

test_num = '0*/'
data_dir = '/home/local/ASUAD/gmclark1/Research/data/pendulum_high/test_'+test_num
valid_dir = '/home/local/ASUAD/gmclark1/Research/data/pendulum_high/valid/'

# gym_env = CartPoleEnvNoise()
# env = suite_gym.wrap_env(gym_env)

# env_name = 'CartPole-v0'
# env = suite_gym.load(env_name)
# eval_env = tf_py_environment.TFPyEnvironment(env)

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
  state = tf.reshape(state[0:5], shape=[5])
  prev_state = tf.io.parse_tensor(raw_prev_state, out_type=tf.float64)
  prev_state = tf.reshape(prev_state, shape=[prev_state_size])

  multiplier = tf.constant([1,1,1,1,0], dtype=tf.float64)
  state = tf.math.multiply(state, multiplier)

  adder = tf.constant([0,0,0,0,.5], dtype=tf.float64)
  state = tf.math.add(state, adder)

  return (prev_state, state)


def get_dynamics_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_dynamics) #, num_parallel_calls=tf.data.AUTOTUNE)
    return pendulum_dataset


def parse_tfr_observation(element):
  data = {
      'img_height': tf.io.FixedLenFeature([], tf.int64),
      'img_width':tf.io.FixedLenFeature([], tf.int64),
      'img_depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'prev_raw_image' : tf.io.FixedLenFeature([], tf.string),
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'prev_state' : tf.io.FixedLenFeature([], tf.string),}
  content = tf.io.parse_single_example(element, data)
  height = content['img_height']
  width = content['img_width']
  depth = content['img_depth']
  raw_image = content['raw_image']
  prev_raw_image = content['prev_raw_image']
  state_size = content['state_size']
  raw_state = content['state']
  image = tf.io.parse_tensor(raw_image, out_type=tf.float16)
#   image = tf.image.rgb_to_grayscale(image)
#   image = tf.scalar_mul(1/255, image)
  image = tf.math.abs(tf.math.subtract(image, 1))

  prev_image = tf.io.parse_tensor(prev_raw_image, out_type=tf.float16)
#   prev_image = tf.image.rgb_to_grayscale(prev_image)
#   prev_image = tf.scalar_mul(1/255, prev_image)
  prev_image = tf.math.abs(tf.math.subtract(prev_image, 1))

  img = tf.concat((image, prev_image), 0)

  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])

  multiplier = tf.constant([1,1,1,1,0], dtype=tf.float64)
  state = tf.math.multiply(state, multiplier)

  adder = tf.constant([0,0,0,0,.5], dtype=tf.float64)
  state = tf.math.add(state, adder)
  return (img, state)


def get_observation_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_observation)
    return pendulum_dataset


def build_dynamics_model():
    model = tf.keras.Sequential([
        layers.Dense(256, activation=tf.nn.relu, input_shape=[5]),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(5*2)
    ])
    model.compile(optimizer='adam', loss=[CustomLossNLL()])
    model.summary()
    filepath = '/home/local/ASUAD/gmclark1/Research/data/pendulum_high/models/dyn0'
    tf_callback = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch',options=None)
                , keras.callbacks.TensorBoard(log_dir='logs')]
    return model, tf_callback


def build_timedistributed_observation_model():
    input_layer = tf.keras.Input(shape=(2,75,300,1))

    encode_1 = layers.TimeDistributed(layers.Conv2D(8, kernel_size=5, strides=(3,3), padding='same', activation='relu', kernel_initializer='he_uniform'))(input_layer)
    encode_2 = layers.TimeDistributed(layers.Conv2D(8, kernel_size=4, strides=(2,1), padding='same', activation='relu', kernel_initializer='he_uniform'))(encode_1)
    encode_3 = layers.TimeDistributed(layers.Conv2D(8, kernel_size=3, strides=(2,1), padding='same', activation='relu', kernel_initializer='he_uniform'))(encode_2) #, kernel_regularizer=tf.keras.regularizers.l2(l=0.01)
    flaten = layers.TimeDistributed(layers.Flatten())(encode_3)
    deeeep_5 = layers.TimeDistributed(layers.Dense(2048, activation=tf.nn.relu, kernel_initializer='he_uniform'))(flaten)
    # deeeep_6 = layers.TimeDistributed(layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_uniform'))(deeeep_5)

    flaten_7 = layers.Flatten()(deeeep_5)

    deeeep_8 = layers.Dense(1024, activation=tf.nn.relu, kernel_initializer='he_uniform')(flaten_7)
    deeeep_9 = layers.Dense(256, activation=tf.nn.relu, kernel_initializer='he_uniform')(deeeep_8)
    deeeep_10 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer='he_uniform')(deeeep_9)
    output_layer = layers.Dense(5)(deeeep_10)

    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=[tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanAbsoluteError()])
    model.summary()
    filepath = '/home/local/ASUAD/gmclark1/Research/data/pendulum_high/models/fullobs5'
    tf_callback = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch',options=None)
                , keras.callbacks.TensorBoard(log_dir='logs')]
    return model, tf_callback



# dyn_dataset = get_dynamics_dataset()
# dyn_dataset = dyn_dataset.apply(tf.data.experimental.ignore_errors()).shuffle(buffer_size=100000)
# dyn_valid = dyn_dataset.take(val_size).batch(val_size)
# # dyn_train = dyn_dataset.skip(val_size).shuffle(buffer_size=22000).batch(256).prefetch(tf.data.AUTOTUNE)
# dyn_train = dyn_dataset.skip(val_size).shuffle(buffer_size=100000).batch(32).cache().prefetch(tf.data.AUTOTUNE)
# dyn_model, tf_callback1 = build_dynamics_model()
# dyn_model.fit(dyn_train, validation_data=dyn_valid, epochs=200, verbose=1, callbacks=tf_callback1)

# dyn_model_path = "/home/local/ASUAD/gmclark1/Research/data/pendulum_high/models/dyn0/dyn0.ckpt"
# dyn_model.save_weights(dyn_model_path)



obs_model_name = 'observation'
val_dataset = get_observation_dataset(tfr_dir=valid_dir)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
obs_valid = val_dataset.batch(1024)

obs_dataset = get_observation_dataset()
obs_dataset = obs_dataset.apply(tf.data.experimental.ignore_errors())
obs_train = obs_dataset.shuffle(buffer_size=10000).batch(32).cache().prefetch(tf.data.AUTOTUNE)
obs_model, tf_callback2 = build_timedistributed_observation_model()
obs_model.fit(obs_train, validation_data=obs_valid, epochs=300, verbose=1, callbacks=tf_callback2)

# ck_path = "models/obs_test4/obs_test4.ckpt"
# obs_model.save_weights(ck_path)



# for x,y in obs_valid.as_numpy_iterator():
#     print(x)

# run_model(eval_env, dyn_model)





pass