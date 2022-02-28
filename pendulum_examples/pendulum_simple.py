from __future__ import absolute_import, division, print_function
import base64
from cgi import test
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
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from matplotlib import rcParams, rc
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

import tensorflow as tf
import tensorflow_probability as tfp
import tensorboard
tfd = tfp.distributions
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
# from cartpole_noise import CartPoleEnvNoise
from cartpole_high import CartPoleEnvNoise

# test1 = np.abs(np.random.normal(0,.0008,50000))
# test2 = np.abs(np.random.normal(0,.00011,50000))
# test3 = np.abs(np.random.normal(0,.00023,50000))
# test4 = np.abs(np.random.normal(0,.001,50000))
# ans = np.mean(np.concatenate((test1, test2, test3, test4), axis=0))
# print(ans)

np.random.seed(2021)

test_num = '0*/'
data_dir = '/home/geoffrey/Research/data/pendulum_high/test_'+test_num
valid_dir = '/home/geoffrey/Research/data/pendulum_high/valid/'


gym_env = suite_gym.wrap_env(CartPoleEnvNoise(1.0, .0001))
eval_env = tf_py_environment.TFPyEnvironment(gym_env)

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
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'prev_state_size':tf.io.FixedLenFeature([], tf.int64),
      'prev_state' : tf.io.FixedLenFeature([], tf.string),}
  content = tf.io.parse_single_example(element, data)
  state_size = content['state_size']
  prev_state_size = content['prev_state_size']
  raw_state = content['state']
  raw_prev_state = content['prev_state']
  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])
  prev_state = tf.io.parse_tensor(raw_prev_state, out_type=tf.float64)
  prev_state = tf.reshape(prev_state, shape=[prev_state_size])

  state = tf.math.multiply(state, tf.constant([1,1,1,1,0], dtype=tf.float64))

  return (prev_state, state)


def get_dynamics_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_dynamics) #, num_parallel_calls=tf.data.AUTOTUNE)
    return pendulum_dataset


def parse_tfr_observation(element):
  data = {
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'prev_state_size':tf.io.FixedLenFeature([], tf.int64),
      'prev_state' : tf.io.FixedLenFeature([], tf.string),}
  content = tf.io.parse_single_example(element, data)
  state_size = content['state_size']
  prev_state_size = content['prev_state_size']
  raw_state = content['state']
  raw_prev_state = content['prev_state']
  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])
  prev_state = tf.io.parse_tensor(raw_prev_state, out_type=tf.float64)
  prev_state = tf.reshape(prev_state, shape=[prev_state_size])

  rand1 = tfp.distributions.MultivariateNormalDiag(loc=[0,0,0,0,0], scale_diag=[1,.001,.001,.001, .00000001]).sample()
  obs = tf.math.add(state, tf.cast(rand1, dtype=tf.float64))
  state = tf.math.multiply(state, tf.constant([1,1,1,1,0], dtype=tf.float64))
  return (obs, state)


def get_observation_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_observation)
    return pendulum_dataset


def build_dynamics_model():
    NUM_TRAIN_EXAMPLES = 1311296
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    model = tf.keras.Sequential([
        tfp.layers.DenseFlipout(64, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu, input_shape=[5]), #, kernel_divergence_fn=kl_divergence_function
        # tfp.layers.DenseFlipout(128, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
        # tfp.layers.DenseFlipout(64, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
        tfp.layers.DenseFlipout(32, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
        tfp.layers.DenseFlipout(16, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
        layers.Dense(5)
    ])
    model.compile(optimizer='adam', loss=[tf.keras.losses.MeanAbsoluteError()])
    model.summary()
    filepath = '/home/geoffrey/Research/data/pendulum_high/models/dyn_simple0'
    tf_callback = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch',options=None)
                , keras.callbacks.TensorBoard(log_dir='logs')]
    return model, tf_callback


def build_simple_observation_model():
    NUM_TRAIN_EXAMPLES = 2233056
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    input_layer = tf.keras.Input(shape=(5))
    # layer1 = tfp.layers.DenseFlipout(256, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu)(input_layer)
    # layer2 = tfp.layers.DenseFlipout(128, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu)(layer1)
    # layer3 = tfp.layers.DenseFlipout(64, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu)(layer2)
    layer4 = tfp.layers.DenseFlipout(32, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu)(input_layer)
    layer5 = tfp.layers.DenseFlipout(16, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu)(layer4)
    output_layer = tfp.layers.DenseFlipout(5, kernel_divergence_fn=kl_divergence_function)(layer5)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam', loss=[tf.keras.losses.MeanSquaredError()])
    model.summary()
    filepath = '/home/geoffrey/Research/data/pendulum_high/models/obs_simple0'
    tf_callback = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch',options=None)
                , keras.callbacks.TensorBoard(log_dir='logs')]
    return model, tf_callback

def observation_model_noise_clean(state, prev_state, n):
    rand1 = np.random.multivariate_normal([0,0,0,0,0], [[.00007,0,0,0,0], [0,.000169,0,0,0], [0,0,.000065,0,0], [0,0,0,.00042,0], [0,0,0,0,0.000000001]], n)
    rand2 = np.random.multivariate_normal([0,0,0,0,0], [[.00007,0,0,0,0], [0,.000169,0,0,0], [0,0,.000065,0,0], [0,0,0,.00042,0], [0,0,0,0,0.000000001]], n)
    observation = np.add(np.tile(((state)).reshape(1,1,5), (1,n,1)), rand1.reshape(1,n,5))
    observation[0,:,4] = np.zeros((n))
    return observation


def run_models(env, dyn_model, obs_model, saved_policy):
    num_ensembles = 100
    state_size = 5
    time_step = env.reset()
    obs=time_step.observation.numpy()
    state = np.concatenate((obs, np.array([0]).reshape(1,1)), axis=1)
    test_state=[]; test_dyn=[]; test_obs=[]
    policy_state = saved_policy.get_initial_state(batch_size=1)
    prev_state = state

    obs_pred_ensemble = np.asarray([]).reshape(0,num_ensembles,state_size)
    dyn_pred_ensemble = np.asarray([]).reshape(0,num_ensembles,state_size)
    flt_pred_ensemble = np.asarray([]).reshape(0,num_ensembles,state_size)
    obs_mean_time = np.asarray([]).reshape(0,state_size)
    true_state = np.asarray([]).reshape(0,state_size)
    ensemble_filtered = np.asarray([]).reshape(0,state_size)

    test_time = []
    extra_test = []
    extra_test2 = []

    for i in range(num_ensembles):
        ensemble_filtered = np.concatenate((ensemble_filtered, state.reshape(1,state_size)), axis=0)

    for j in range(500):
        if not time_step.is_last():
            policy_step = saved_policy.action(time_step, policy_state)
            policy_state = policy_step.state
            action = policy_step.action.numpy()
            # action = np.array([10])

            time_step = env.step(action)
            obs=time_step.observation.numpy()
            state = np.concatenate((obs, np.array([(action-10)*.1]).reshape(1,1)), axis=1)

            env.render()

            start_time = timer()

            ensemble_obs = observation_model_noise_clean(state, prev_state,num_ensembles)
            obs_pred_ensemble = np.concatenate((obs_pred_ensemble, ensemble_obs), axis = 0)
            mean_obs = np.mean(ensemble_obs, axis=1)
            obs_mean_time = np.concatenate((obs_mean_time, mean_obs), axis=0)
            ensemble_obs_mean = np.tile(mean_obs.reshape(1,1,5), (1,num_ensembles,1))
            cov_obs = np.cov(ensemble_obs.reshape(num_ensembles,state_size).T) * np.identity(5)

            ensemble_filtered[:,4]=prev_state[0,4]
            ensemble_dyn = dyn_model(ensemble_filtered, training=False).numpy().reshape(1,num_ensembles,state_size)
            ensemble_dyn[0,:,4] = 0
            dyn_pred_ensemble = np.concatenate((dyn_pred_ensemble, ensemble_dyn), axis = 0)
            mean_dyn = np.mean(ensemble_dyn, axis=1)
            cov_dyn = np.cov(ensemble_dyn.reshape(num_ensembles,state_size).T) * (np.identity(5)*2)

            # print(cov_obs)
            # print(cov_dyn)
            S = np.add(cov_obs, cov_dyn)
            ensemble_filtered = (np.dot(ensemble_dyn.reshape(num_ensembles,state_size), np.dot(cov_obs,np.linalg.pinv(S))) + np.dot(ensemble_obs.reshape(num_ensembles,state_size), np.dot(cov_dyn,np.linalg.pinv(S))))
            # ensemble_filtered = .5*ensemble_dyn.reshape(num_ensembles,5) + .5*ensemble_obs.reshape(num_ensembles,5)
            flt_pred_ensemble = np.concatenate((flt_pred_ensemble, ensemble_filtered.reshape(1,num_ensembles,5)), axis=0)

            end_time = timer()
            test_time.append(end_time-start_time)

            prev_state = state
            true_state = np.concatenate((true_state, state), axis=0)

        else:
            break


    # mean_time = np.mean(np.asarray(test_time)[1:])
    # std_time = np.std(np.asarray(test_time)[1:])
    # print('Average time = ', mean_time)
    # print('Standard deviation time = ', std_time)

    # mae_obs = np.mean(np.tile(np.abs(true_state.reshape(200,1,5)-obs_pred_ensemble),(1,100,1)).reshape(-1,5), 0)
    # std_obs = np.std(np.tile(np.abs(true_state.reshape(200,1,5)-obs_pred_ensemble),(1,100,1)).reshape(-1,5), 0)
    # print('Observation Error1 - ', mae_obs[0], ' +- ',std_obs[0])
    # print('Observation Error2 - ', mae_obs[1], ' +- ',std_obs[1])
    # print('Observation Error3 - ', mae_obs[2], ' +- ',std_obs[2])
    # print('Observation Error4 - ', mae_obs[3], ' +- ',std_obs[3])

    dyn_mean = np.mean(dyn_pred_ensemble, axis=1)
    # mae_dyn = np.mean(np.tile(np.abs(true_state.reshape(200,1,5)-dyn_pred_ensemble),(1,100,1)).reshape(-1,5), 0)
    # std_dyn = np.std(np.tile(np.abs(true_state.reshape(200,1,5)-dyn_pred_ensemble),(1,100,1)).reshape(-1,5), 0)
    # print('Dynamics Error1 - ', mae_dyn[0], ' +- ', std_dyn[0])
    # print('Dynamics Error2 - ', mae_dyn[1], ' +- ', std_dyn[1])
    # print('Dynamics Error3 - ', mae_dyn[2], ' +- ', std_dyn[2])
    # print('Dynamics Error4 - ', mae_dyn[3], ' +- ', std_dyn[3])

    filt_mean = np.mean(flt_pred_ensemble, axis=1)
    # mae_filt = np.mean(np.abs(true_state-filt_mean), axis=0)
    # std_filt = np.std(np.abs(true_state-filt_mean), axis=0)
    # print('Filtered Error1 - ', mae_filt[0], ' +- ', std_filt[0])
    # print('Filtered Error2 - ', mae_filt[1], ' +- ', std_filt[1])
    # print('Filtered Error3 - ', mae_filt[2], ' +- ', std_filt[2])
    # print('Filtered Error4 - ', mae_filt[3], ' +- ', std_filt[3])

    rc('xtick', labelsize=8) 
    rc('ytick', labelsize=8)
    x = np.linspace(0,499*.02,500)


    plt.figure(10, figsize=(3.5, 6))
    for i in range(4):
        plt.subplot(4,1,i+1)
        for j in range(num_ensembles):
            plt.plot(x, flt_pred_ensemble[:,j,i], color=[.7,.7,.7], linewidth=2, alpha=.3)
        plt.plot(x, true_state[:,i], color='k', linewidth=2)
        plt.plot(x, filt_mean[:,i], color='#1f77b4', linewidth=1)
        plt.xlim([0,max(x)+.02])

    plt.subplot(4,1,1)
    plt.ylabel('Cart Position')
    plt.subplot(4,1,2)
    plt.ylabel('Cart Velocity')
    plt.subplot(4,1,3)
    plt.ylabel('Pole Position')
    plt.subplot(4,1,4)
    plt.ylabel('Pole Velocity')
    plt.xlabel('Time (s)')
    plt.tight_layout()

    # color='#173f5f'
    # Color=[.4705, .7921, .6470]


    # plt.figure(10, figsize=(3.5, 4))
    # for i in range(4):
    #     plt.subplot(4,1,i+1)
    #     plt.plot(true_state[:,i], color='k')
    #     plt.plot(obs_mean_time[:,i], color='g')
    #     plt.plot(dyn_mean[:,i], color='b')
    #     plt.plot(filt_mean[:,i], color='r')
    # plt.figure(11, figsize=(8, 6), dpi=100)
    # for i in range(4):
    #     plt.subplot(4,1,i+1)
    #     plt.plot(true_state[:,i], color='k')
    #     for j in range(num_ensembles):
    #         plt.plot(dyn_pred_ensemble[:,j,i], color='b')
    # plt.figure(12, figsize=(8, 6), dpi=100)
    # for i in range(4):
    #     plt.subplot(4,1,i+1)
    #     plt.plot(true_state[:,i], color='k')
    #     for j in range(num_ensembles):
    #         plt.plot(obs_pred_ensemble[:,j,i], color='g')
    # plt.figure(13, figsize=(8, 6), dpi=100)
    # for i in range(4):
    #     plt.subplot(4,1,i+1)
    #     plt.plot(true_state[:,i], color='k')
    #     for j in range(num_ensembles):
    #         plt.plot(flt_pred_ensemble[:,j,i], color='r')
    plt.show()




if __name__=='__main__':
    # Train dynamics model
    # val_dataset = get_dynamics_dataset(tfr_dir=valid_dir)
    # val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors()).shuffle(buffer_size=100000).batch(10000)
    # dyn_dataset = get_dynamics_dataset()
    # dyn_dataset = dyn_dataset.apply(tf.data.experimental.ignore_errors())
    # dyn_train = dyn_dataset.shuffle(buffer_size=100000).batch(32).cache().prefetch(tf.data.AUTOTUNE)
    # dyn_model, tf_callback1 = build_dynamics_model()
    # dyn_model.fit(dyn_train, validation_data=val_dataset, epochs=100, verbose=1, callbacks=tf_callback1)


    # Train observation model
    # val_dataset = get_observation_dataset(tfr_dir=valid_dir)
    # val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors()).shuffle(buffer_size=100000).batch(10000)
    # obs_dataset = get_observation_dataset()
    # obs_dataset = obs_dataset.apply(tf.data.experimental.ignore_errors())
    # obs_train = obs_dataset.shuffle(buffer_size=100000).batch(128).prefetch(tf.data.AUTOTUNE)
    # obs_model, tf_callback2 = build_simple_observation_model()
    # obs_model.fit(obs_train, validation_data=val_dataset, epochs=100, verbose=1, callbacks=tf_callback2)


    # control model
    ctrl_model = tf.compat.v2.saved_model.load('/home/geoffrey/Research/data/pendulum_high/models/ctrl0')
    # dyn_model = build_dynamics_model()
    dyn_model = models.load_model('/home/geoffrey/Research/data/pendulum_high/models/dyn_simple0')
    # obs_model = build_simple_observation_model()
    obs_model = models.load_model('/home/geoffrey/Research/data/pendulum_high/models/obs_simple0')
    run_models(eval_env, dyn_model, obs_model, ctrl_model)














    # action = []
    # i=0
    # for x,y in dyn_dataset.as_numpy_iterator():
    #     action.append(x[4])
    #     i=i+1
    #     print(i)
    #     print(x[4])
    #     if i>3000:
    #         break
    # plt.hist(action, 21)
    # plt.show()


    # for x,y in dyn_dataset.as_numpy_iterator():
    #     print(x)

    # run_model(eval_env, dyn_model)
    pass