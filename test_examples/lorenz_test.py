from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense
# import tensorboard
import datetime

np.random.seed(2021)

class CustomLossNLL(tf.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        mean, log_sigma = tf.split(y_pred, 2, axis=-1)
        y_target, temp =tf.split(y_true,2,axis=-1)
        sigma = tf.nn.softplus(log_sigma)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        loss = -tf.reduce_mean(dist.log_prob(y_target))
        return loss

def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def generateData(num_steps, noise, window, dt):
    observations = np.array([]).reshape(num_steps, -1)
    # Need one more for the initial values
    xs = np.zeros((num_steps + window))
    ys = np.zeros((num_steps + window))
    zs = np.zeros((num_steps + window))
    var = np.zeros(num_steps).reshape(-1,1)
    # Set initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    # Step through "time", calculating the partial derivatives at the current point and using them to estimate the next point
    for i in range(num_steps + window-1):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = (xs[i] + (x_dot * dt))
        ys[i + 1] = (ys[i] + (y_dot * dt))
        zs[i + 1] = (zs[i] + (z_dot * dt))
    # Add noise
    xsn = xs + np.random.normal(0,noise,[num_steps+window])
    ysn = ys + np.random.normal(0,noise,[num_steps+window])
    zsn = zs + np.random.normal(0,noise,[num_steps+window])
    # Arrange data into observations, state, and previous state vectors
    for i in range(1,window):
        observations = np.concatenate((observations, xsn[i:-window+i].reshape(-1,1), ysn[i:-window+i].reshape(-1,1), zsn[i:-window+i].reshape(-1,1)), axis=1)
    observations = np.concatenate((observations, xsn[window:].reshape(-1,1), ysn[window:].reshape(-1,1), zsn[window:].reshape(-1,1)), axis=1)
    state = np.concatenate((xs[window:].reshape(-1,1), ys[window:].reshape(-1,1), zs[window:].reshape(-1,1), var, var, var), axis=1)
    prev_state = np.concatenate((xs[window-1:-1].reshape(-1,1), ys[window-1:-1].reshape(-1,1), zs[window-1:-1].reshape(-1,1), var, var, var), axis=1)
    state_dynamics = state-prev_state 
    # plot data
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(observations[:,-3], observations[:,-2], observations[:,-1], s=8, Color=[.4705, .7921, .6470])
    # ax.plot(state[:,0], state[:,1], state[:,2], Color='#173f5f')
    # plt.show()
    return observations/40, state/40, prev_state/40, state_dynamics


def buildInverseObsModel(window):
    model = tf.keras.Sequential([
        layers.Dense(32, activation=tf.nn.relu, input_shape=[3*window]),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(3*2)
    ])
    model.compile(optimizer='adam', loss= [CustomLossNLL()])
    model.summary()
    return model


def buildDynamicsModel():
    model = tf.keras.Sequential([
        layers.Dense(32, activation=tf.nn.relu, input_shape=[3]),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(3*2)
    ])
    model.compile(optimizer='adam', loss= [CustomLossNLL()])
    model.summary()
    return model



def plot_everything(obs_model, dynamics_model):
    state_pred = obs_model(test_obs, training=False)
    state_mean, state_sigma = prediction_split(state_pred)
    next_state = dynamics_model(prev_test_state[:,0:3], training=False)
    dynamics_mean, dynamics_sigma = prediction_split(next_state)

    # Plot Everything
    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(test_state[:,0], Color='k')
    plt.plot(state_mean[:,0], Color='g')
    plt.fill_between(np.linspace(0,state_mean.shape[0],state_mean.shape[0]), state_mean[:,0]+state_sigma[:,0], state_mean[:,0]-state_sigma[:,0], facecolor='g', alpha=.2)
    plt.plot(dynamics_mean[:,0], Color='b')
    plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,0]+dynamics_sigma[:,0], dynamics_mean[:,0]-dynamics_sigma[:,0], facecolor='b', alpha=.2)
    plt.subplot(3,1,2)
    plt.plot(test_state[:,1], Color='k')
    plt.plot(state_mean[:,1], Color='g')
    plt.fill_between(np.linspace(0,state_mean.shape[0],state_mean.shape[0]), state_mean[:,1]+state_sigma[:,1], state_mean[:,1]-state_sigma[:,1], facecolor='g', alpha=.2)
    plt.plot(dynamics_mean[:,1], Color='b')
    plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,1]+dynamics_sigma[:,1], dynamics_mean[:,1]-dynamics_sigma[:,1], facecolor='b', alpha=.2)
    plt.subplot(3,1,3)
    plt.plot(test_state[:,2], Color='k')
    plt.plot(state_mean[:,2], Color='g')
    plt.fill_between(np.linspace(0,state_mean.shape[0],state_mean.shape[0]), state_mean[:,2]+state_sigma[:,2], state_mean[:,2]-state_sigma[:,2], facecolor='g', alpha=.2)
    plt.plot(dynamics_mean[:,2], Color='b')
    plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,2]+dynamics_sigma[:,2], dynamics_mean[:,2]-dynamics_sigma[:,2], facecolor='b', alpha=.2)
    
    

def prediction_split(prediction):
    mean, log_sigma = tf.split(prediction, 2, axis=-1)
    sigma = np.sqrt(tf.nn.softplus(log_sigma))
    return mean, sigma


if __name__ =="__main__":
    window=10; train_len=1500000; test_len=500; noise=5; dt=0.01
    train_obs, train_state, train_prev_state, train_dynamics = generateData(train_len, noise, window, dt)
    test_obs, test_state, prev_test_state, test_dynamics = generateData(test_len, noise, window, dt)
    permutation = np.random.permutation(train_obs.shape[0])
    EPOCHS = 2

    # obs model, observations -> state
    obs_model = buildInverseObsModel(window)
    obs_model.fit(train_obs[permutation,:], train_state[permutation,:],epochs=EPOCHS, validation_split = 0.8, verbose=1)

    # Dynamics Model, state -> state
    dynamics_model = buildDynamicsModel()
    dynamics_model.fit(train_prev_state[permutation,0:3], train_state[permutation,:],epochs=EPOCHS, validation_split = 0.8, verbose=1)

    # Plot all the stuff (probably not working)
    plot_everything(obs_model, dynamics_model)


    cycle_state = prev_test_state[0].reshape(1,-1)
    filtered_var = np.zeros((1,3))
    for i in range(test_state.shape[0]):
        temp1 = dynamics_model(cycle_state[-1,0:3].reshape(1,-1), training=False)
        dyn_mean, dyn_var = prediction_split(temp1)

        temp2 = obs_model(test_obs[i].reshape(1,-1), training=False)
        obs_mean, obs_var = prediction_split(temp2)

        proc_var = dyn_var + filtered_var
        filtered_state = (obs_var/(proc_var+obs_var))*dyn_mean + (proc_var/(proc_var+obs_var))*obs_mean
        filtered_var = proc_var*(proc_var+obs_var)*obs_var
        new_state = np.concatenate((filtered_state, filtered_var), axis=1)

        cycle_state = np.concatenate((cycle_state, new_state), axis=0)

    cycle_mean, cycle_sigma = tf.split(cycle_state, 2, axis=-1)

    

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(test_obs[:,-3]*40, test_obs[:,-2]*40, test_obs[:,-1]*40, Color=[.4705, .7921, .6470], s=15)
    plt.plot(test_state[:,0]*40, test_state[:,1]*40, test_state[:,2]*40, Color='#173f5f')
    plt.plot(cycle_mean[:,0]*40, cycle_mean[:,1]*40, cycle_mean[:,2]*40, Color='b')
    # plt.savefig('dynamics_only.pdf', bbox_inches='tight')
    plt.show()
    pass
        









