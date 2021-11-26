from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
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

class CustomLossNLL(tf.losses.Loss):
    @tf.function
    def call(self, y_true, y_pred):
        mean, log_sigma = tf.split(y_pred, 2, axis=-1)
        y_target, temp =tf.split(y_true,2,axis=-1)
        sigma = tf.nn.softplus(log_sigma)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        loss = -tf.reduce_mean(dist.log_prob(y_target))
        return loss

def generateData(demos, num_steps, noise, window):
    observations = np.array([]).reshape(-1,2*window)
    state = np.array([]).reshape(-1,5*2)
    prev_state = np.array([]).reshape(-1,5*2)
    for demo in range(demos):
        position = np.random.uniform(-2, 2, 2)
        size = np.random.uniform(0.01, 2, 2)
        start_phase = np.random.uniform(0,2*np.pi)

        noise_1 = np.random.normal(0,noise, num_steps)
        noise_2 = np.random.normal(0,noise, num_steps)
        t = np.linspace(0, num_steps, num_steps)/200

        y = position[0]+(size[0]*np.sin(t*2*np.pi+start_phase)).reshape(-1,1)
        x = position[1]+(size[1]*np.cos(t*2*np.pi+start_phase)).reshape(-1,1)
        grad_y = np.gradient(y[:,0])
        grad_x = np.gradient(x[:,0])
        orientation = np.arctan2(grad_y,grad_x)
        orient_sin = np.sin(orientation).reshape(-1,1)
        orient_cos = np.cos(orientation).reshape(-1,1)
        a = size[0]
        b = size[1]
        perimeter = np.pi * (3*(a+b)-np.sqrt((3*a+b)*(a+3*b)))
        velocity = np.ones((num_steps,1))*perimeter

        sin_nt = y + noise_1.reshape(-1,1)
        cos_nt = x + noise_2.reshape(-1,1)
        var = np.zeros((num_steps-window,1))
        demo_obs = np.array([]).reshape(num_steps-window,-1)
        for i in range(window):
            demo_obs = np.concatenate((demo_obs, sin_nt[i:-window+i], cos_nt[i:-window+i]), axis=1)
        demo_state = np.concatenate((y[window:], x[window:], orient_cos[window:], orient_sin[window:], velocity[window:], var, var, var, var, var),axis=1)
        prev_demo_state = np.concatenate((y[window-1:-1], x[window-1:-1], orient_cos[window-1:-1], orient_sin[window-1:-1], velocity[window-1:-1], var, var, var, var, var),axis=1)
        # plt.scatter(demo_obs[:,0], demo_obs[:,1], Color='b')
        # plt.plot(demo_state[:,0], demo_state[:,1], Color='r')
        # plt.show()
        observations = np.concatenate((observations, demo_obs), axis=0)
        state = np.concatenate((state, demo_state), axis=0)
        prev_state = np.concatenate((prev_state, prev_demo_state), axis=0)
    # plt.show()
    return observations, state, prev_state

def buildModel(obs_size, state_size):
    model = tf.keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[obs_size]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(state_size+state_size**2)
    ])
    model.compile(optimizer='adam', loss= [CustomLossNLL()])
    model.summary()
    return model


def buildDynamicsModel():
    model = tf.keras.Sequential([
        layers.Dense(32, activation=tf.nn.relu, input_shape=[5]),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(5*2)
    ])
    model.compile(optimizer='adam', loss= [CustomLossNLL()])
    model.summary()
    return model




if __name__ =="__main__":
    window=1; length_demo=200; num_demos=1000
    train_obs, train_state, train_prev_state = generateData(num_demos, length_demo, 0.1, window)
    test_obs, test_state, prev_test_state = generateData(1, length_demo, 0.1, window)

    # state model, observations -> state
    state_model = buildModel(window)
    permutation = np.random.permutation(train_obs.shape[0])
    EPOCHS = 10
    state_model.fit(train_obs[permutation,:], train_state[permutation,:],epochs=EPOCHS, validation_split = 0.8, verbose=1)
    state_pred = state_model(test_obs, training=False)
    state_mean, state_log_sigma = tf.split(state_pred, 2, axis=-1)
    state_sigma = np.sqrt(tf.nn.softplus(state_log_sigma))

    # Dynamics Model, state -> state
    dynamics_model = buildDynamicsModel()
    dynamics_model.fit(train_prev_state[permutation,0:5], train_state[permutation,:],epochs=EPOCHS, validation_split = 0.8, verbose=1)
    next_state = dynamics_model(prev_test_state[:,0:5], training=False)
    dynamics_mean, dynamics_log_sigma = tf.split(next_state, 2, axis=-1)
    dynamics_sigma = np.sqrt(tf.nn.softplus(dynamics_log_sigma))

    # Plot Everything
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(test_state[:,0], Color='k')
    plt.plot(state_mean[:,0], Color='g')
    plt.fill_between(np.linspace(0,state_mean.shape[0],state_mean.shape[0]), state_mean[:,0]+state_sigma[:,0], state_mean[:,0]-state_sigma[:,0], facecolor='g', alpha=.2)
    plt.plot(dynamics_mean[:,0], Color='b')
    plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,0]+dynamics_sigma[:,0], dynamics_mean[:,0]-dynamics_sigma[:,0], facecolor='b', alpha=.2)
    plt.subplot(2,1,2)
    plt.plot(test_state[:,1], Color='k')
    plt.plot(state_mean[:,1], Color='g')
    plt.fill_between(np.linspace(0,state_mean.shape[0],state_mean.shape[0]), state_mean[:,1]+state_sigma[:,1], state_mean[:,1]-state_sigma[:,1], facecolor='g', alpha=.2)
    plt.plot(dynamics_mean[:,1], Color='b')
    plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,1]+dynamics_sigma[:,1], dynamics_mean[:,1]-dynamics_sigma[:,1], facecolor='b', alpha=.2)
    

    plt.figure(2)
    plt.subplot(3,1,1)
    plt.plot(test_state[:,2], Color='k')
    plt.plot(state_mean[:,2], Color='g')
    plt.fill_between(np.linspace(0,state_mean.shape[0],state_mean.shape[0]), state_mean[:,2]+state_sigma[:,2], state_mean[:,2]-state_sigma[:,2], facecolor='g', alpha=.2)
    plt.plot(dynamics_mean[:,2], Color='b')
    plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,2]+dynamics_sigma[:,2], dynamics_mean[:,2]-dynamics_sigma[:,0], facecolor='b', alpha=.2)
    
    plt.subplot(3,1,2)
    plt.plot(test_state[:,3], Color='k')
    plt.plot(state_mean[:,3], Color='g')
    plt.fill_between(np.linspace(0,state_mean.shape[0],state_mean.shape[0]), state_mean[:,3]+state_sigma[:,3], state_mean[:,3]-state_sigma[:,3], facecolor='g', alpha=.2)
    plt.plot(dynamics_mean[:,3], Color='b')
    plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,3]+dynamics_sigma[:,3], dynamics_mean[:,3]-dynamics_sigma[:,3], facecolor='b', alpha=.2)
    
    plt.subplot(3,1,3)
    plt.plot(test_state[:,4], Color='k')
    plt.plot(state_mean[:,4], Color='g')
    plt.fill_between(np.linspace(0,state_mean.shape[0],state_mean.shape[0]), state_mean[:,4]+state_sigma[:,4], state_mean[:,4]-state_sigma[:,4], facecolor='g', alpha=.2)
    plt.plot(dynamics_mean[:,4], Color='b')
    plt.fill_between(np.linspace(0,dynamics_mean.shape[0],dynamics_mean.shape[0]), dynamics_mean[:,4]+dynamics_sigma[:,4], dynamics_mean[:,4]-dynamics_sigma[:,0], facecolor='b', alpha=.2)
    
    plt.show()
    pass





