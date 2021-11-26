from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib
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

def generateData(demos, num_steps, noise, window):
    observations = np.array([]).reshape(-1,2*window)
    state = np.array([]).reshape(-1,5*2)
    prev_state = np.array([]).reshape(-1,5*2)
    for demo in range(demos):
        position = np.random.uniform(.99999, 1.00001, 2)
        size = np.random.uniform(.9, 1.1, 2)
        start_phase = np.random.uniform(0,2*np.pi)

        noise_1 = np.random.normal(0,noise, num_steps+window)
        noise_2 = np.random.normal(0,noise, num_steps+window)
        t = np.linspace(start_phase-(2*np.pi/num_steps*window), 2*np.pi+start_phase, num_steps+window)

        y = position[0]+(size[0]*np.sin(t)).reshape(-1,1)
        x = position[1]+(size[1]*np.cos(t)).reshape(-1,1)
        grad_y = np.gradient(y[:,0])
        grad_x = np.gradient(x[:,0])
        orientation = np.arctan2(grad_y,grad_x)
        orient_sin = np.sin(orientation).reshape(-1,1)
        orient_cos = np.cos(orientation).reshape(-1,1)
        a = size[0]
        b = size[1]
        perimeter = np.pi * (3*(a+b)-np.sqrt((3*a+b)*(a+3*b)))
        velocity = np.ones((num_steps+window,1))*perimeter

        sin_nt = y + noise_1.reshape(-1,1)
        cos_nt = x + noise_2.reshape(-1,1)
        var = np.zeros((num_steps,1))
        demo_obs = np.array([]).reshape(num_steps,-1)
        for i in range(1,window):
            demo_obs = np.concatenate((demo_obs, sin_nt[i:-window+i], cos_nt[i:-window+i]), axis=1)
        demo_obs = np.concatenate((demo_obs, sin_nt[window:], cos_nt[window:]), axis=1)
        demo_state = np.concatenate((y[window:], x[window:], orient_cos[window:], orient_sin[window:], velocity[window:], var, var, var, var, var),axis=1)
        prev_demo_state = np.concatenate((y[window-1:-1], x[window-1:-1], orient_cos[window-1:-1], orient_sin[window-1:-1], velocity[window-1:-1], var, var, var, var, var),axis=1)
        # plt.scatter(demo_obs[:,0], demo_obs[:,1], s=8, Color=[.4705, .7921, .6470])
        # plt.plot(demo_state[:,0], demo_state[:,1], Color='#173f5f')
        # plt.show()
        observations = np.concatenate((observations, demo_obs), axis=0)
        state = np.concatenate((state, demo_state), axis=0)
        prev_state = np.concatenate((prev_state, prev_demo_state), axis=0)
    # plt.xlim([-6,6])
    # plt.ylim([-6,6])
    # plt.show()
    # plt.savefig('data.pdf', bbox_inches='tight')
    return observations, state, prev_state

def buildModel(window):
    model = tf.keras.Sequential([
        layers.Dense(32, activation=tf.nn.relu, input_shape=[2*window]),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(5*2)
    ])
    model.compile(optimizer='adam', loss= [CustomLossNLL()])
    model.summary()
    return model


def buildDynamicsModel():
    model = tf.keras.Sequential([
        layers.Dense(32, activation=tf.nn.relu, input_shape=[5]),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(5*2)
    ])
    model.compile(optimizer='adam', loss= [CustomLossNLL()])
    model.summary()
    return model

def buildObservationModel():
    model = tf.keras.Sequential([
        layers.Dense(32, activation=tf.nn.relu, input_shape=[5]),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(2*2)
    ])
    model.compile(optimizer='adam', loss= [CustomLossNLL()])
    model.summary()
    return model


def plot_everything(test_state, state_mean, state_sigma, dynamics_mean, dynamics_sigma):
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


if __name__ =="__main__":
    window=10; length_demo=100; num_demos=10000; noise=.01
    train_obs, train_state, train_prev_state = generateData(num_demos, length_demo, noise, window)
    test_obs, test_state, prev_test_state = generateData(1, length_demo, noise, window)

    # state model, observations -> state
    state_model = buildModel(window)
    permutation = np.random.permutation(train_obs.shape[0])
    EPOCHS = 1
    # state_model.fit(train_obs[permutation,:], train_state[permutation,:],epochs=EPOCHS, validation_split = 0.8, verbose=1)
    # state_pred = state_model(test_obs, training=False)
    # state_mean, state_log_sigma = tf.split(state_pred, 2, axis=-1)
    # state_sigma = np.sqrt(tf.nn.softplus(state_log_sigma))

    # Dynamics Model, state -> state
    dynamics_model = buildDynamicsModel()
    dynamics_model.fit(train_prev_state[permutation,0:5], train_state[permutation,:],epochs=EPOCHS, validation_split = 0.8, verbose=1)
    next_state = dynamics_model(prev_test_state[:,0:5], training=False)
    dynamics_mean, dynamics_log_sigma = tf.split(next_state, 2, axis=-1)
    dynamics_sigma = np.sqrt(tf.nn.softplus(dynamics_log_sigma))

    # plot_everything(test_state, state_mean, state_sigma, dynamics_mean, dynamics_sigma)

    cycle_state = prev_test_state[0].reshape(1,-1)
    filtered_var = np.zeros((1,5))
    # for i in range(test_state.shape[0]):
    for i in range(1000):
        temp1 = dynamics_model(cycle_state[-1,0:5].reshape(1,-1), training=False)
        dyn_mean, dyn_logvar = np.split(temp1, 2, axis=-1)
        dyn_var = tf.nn.softplus(dyn_logvar).numpy()

        # temp2 = state_model(test_obs[i].reshape(1,-1), training=False)
        # obs_mean, obs_logvar = np.split(temp2, 2, axis=-1)
        # obs_var = tf.nn.softplus(obs_logvar).numpy()

        # proc_var = dyn_var + filtered_var
        # filtered_state = (obs_var/(proc_var+obs_var))*dyn_mean + (proc_var/(proc_var+obs_var))*obs_mean
        # filtered_var = proc_var*(proc_var+obs_var)*obs_var
        # new_state = np.concatenate((filtered_state, filtered_var), axis=1)


        cycle_state = np.concatenate((cycle_state, temp1), axis=0)

    cycle_mean, cycle_sigma = tf.split(cycle_state, 2, axis=-1)

    
    # Plot Cyclical
    # plt.figure(3)
    # for i in range(5):
    #     plt.subplot(5,1,i+1)
    #     plt.plot(test_state[:,i], Color=[.4705, .7921, .6470])
    #     plt.plot(state_mean[:,i], Color='#173f5f')
    #     plt.fill_between(np.linspace(0,state_mean.shape[0],state_mean.shape[0]), state_mean[:,i]+state_sigma[:,i], state_mean[:,i]-state_sigma[:,i], facecolor='g', alpha=.2)
    #     plt.plot(cycle_mean[:,i], Color='b')
    #     plt.fill_between(np.linspace(0,cycle_mean.shape[0],cycle_mean.shape[0]), cycle_mean[:,i]+cycle_sigma[:,i], cycle_mean[:,i]-cycle_sigma[:,i], facecolor='b', alpha=.2)


    plt.figure(4)
    plt.scatter(test_obs[:,18], test_obs[:,19], Color=[.4705, .7921, .6470], s=15)
    plt.plot(test_state[:,0], test_state[:,1], Color='#173f5f')
    plt.plot(cycle_mean[:,0], cycle_mean[:,1], Color='b')
    plt.savefig('dynamics_only.pdf', bbox_inches='tight')
    plt.show()
    pass
        







    # Observation model, state -> observation
    # obs_model = buildObservationModel()
    # permutation = np.random.permutation(observations.shape[0])
    # EPOCHS = 5
    # obs_model.fit(prev_state[permutation,0:5], observations[permutation,:],epochs=EPOCHS, validation_split = 0.8, verbose=1)
    # observations, state, prev_state = generateData(1, length_demo, 0.05, window)
    # obs_prediction = obs_model(prev_state[:,0:5], training=False)
    # mean, log_sigma = tf.split(obs_prediction, 2, axis=-1)
    # sigma = tf.nn.softplus(log_sigma)
    # plt.subplot(2,1,1)
    # plt.plot(observations[:,0], Color='b')
    # plt.plot(mean[:,0], Color='g')
    # plt.fill_between(np.linspace(0,mean.shape[0],mean.shape[0]), mean[:,0]+sigma[:,0], mean[:,0]-sigma[:,0], facecolor='g', alpha=.2)
    # plt.subplot(2,1,2)
    # plt.plot(observations[:,1], Color='b')
    # plt.plot(mean[:,1], Color='g')
    # plt.fill_between(np.linspace(0,mean.shape[0],mean.shape[0]), mean[:,1]+sigma[:,1], mean[:,1]-sigma[:,1], facecolor='g', alpha=.2)
    # plt.show()
    # pass



