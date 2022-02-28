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

np.random.seed(2021)







def run_models(dyn_model, obs_model, saved_policy):
    # num_ensembles = 1000
    # state_size = 1000


    for num_ensembles in [1]:
        for state_size in [5000]:
            test_time = []
            ensemble_filtered = np.random.normal(0, 1, num_ensembles*state_size).reshape(num_ensembles,state_size)

            for j in range(200):

                start_time = timer()

                ensemble_obs = np.random.normal(0, 1, num_ensembles*state_size).reshape(1,num_ensembles,state_size)
                cov_obs = np.cov(ensemble_obs.reshape(num_ensembles,state_size).T)

                state = np.tile(np.array([[1,1,1,1,1]]),(num_ensembles,1))
                temp = dyn_model(state, training=False).numpy()
                ensemble_dyn = np.random.normal(0, 1, num_ensembles*state_size).reshape(1,num_ensembles,state_size)
                cov_dyn = np.cov(ensemble_dyn.reshape(num_ensembles,state_size).T)

                S = np.add(cov_obs, cov_dyn)
                S_inv = np.dot(cov_obs,np.linalg.pinv(S))
                # ensemble_filtered = (np.dot(ensemble_dyn.reshape(num_ensembles,state_size), S_inv) + np.dot(ensemble_obs.reshape(num_ensembles,state_size), S_inv))
                ensemble_filtered = .5*ensemble_dyn.reshape(num_ensembles,state_size) + .5*ensemble_obs.reshape(num_ensembles,state_size)

                end_time = timer()
                test_time.append(end_time-start_time)

            mean_time = np.mean(np.asarray(test_time)[1:])
            std_time = np.std(np.asarray(test_time)[1:])
            print(num_ensembles,'x',state_size,'--', mean_time, '+-', std_time)








  



if __name__=='__main__':
    # control model
    ctrl_model = tf.compat.v2.saved_model.load('/home/geoffrey/Research/data/pendulum_high/models/ctrl0')
    # dyn_model = build_dynamics_model()
    dyn_model = models.load_model('/home/geoffrey/Research/data/pendulum_high/models/dyn_simple0')
    # obs_model = build_simple_observation_model()
    obs_model = models.load_model('/home/geoffrey/Research/data/pendulum_high/models/obs_simple0')
    run_models(dyn_model, obs_model, ctrl_model)








