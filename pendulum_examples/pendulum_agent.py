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

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

# tf dataset info from:
# https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c


num_iterations = 20000 # @param {type:"integer"}
initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}
batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
num_data_collection_episodes = 5  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
fc_layer_params = (100, 50) #NN layer sizes

data_dir = 'data/inv_pendulum/'

env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

print('Observation Spec:')
print(env.time_step_spec().observation)

print('Reward Spec:')
print(env.time_step_spec().reward)

print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
# print('Time step:')
# print(time_step)

action = np.array(1, dtype=np.int32)
# print('Action:')
# print(action)

next_time_step = env.step(action)
# print('Next time step:')
# print(next_time_step)






def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))


def create_model(env):
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    return q_net


def create_agent(train_env, q_net):
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0))
    agent.initialize()
    return agent


def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0
  for _ in range(num_episodes):
    time_step = environment.reset()
    episode_return = 0.0
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
    total_return += episode_return
  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]
# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_images(img, action, state, next_state):
    img_arr = np.asarray(img, dtype=np.uint8)
    action_arr = np.asarray(action, dtype=np.float64)
    state_arr = np.asarray(state, dtype=np.float64).reshape(-1,4)
    next_state_arr = np.asarray(next_state, dtype=np.float64).reshape(-1,4)
    data = {
        "img_height" : _int64_feature(img_arr.shape[-3]),
        "img_width" : _int64_feature(img_arr.shape[-2]),
        "img_depth" : _int64_feature(img_arr.shape[-1]),
        "raw_image" : _bytes_feature(serialize_array(img_arr)),
        "action_size" : _int64_feature(action_arr.shape[-1]),
        "action": _bytes_feature(serialize_array(action_arr)),
        "state_size" : _int64_feature(state_arr.shape[-1]),
        "state": _bytes_feature(serialize_array(state_arr)),
        "next_state": _bytes_feature(serialize_array(next_state_arr)),
    }
    return data


def collect_data(environment, policy, num_episodes=1):
    step_count = int(0)
    # img=[]; action=[]; state=[]; next_state=[]
    for i in range(num_episodes):
        time_step = environment.reset()
        step_count = step_count + 1
        current_shard_name = "{}{}_{}{}.tfrecords".format(data_dir, i+1, num_episodes, 'pendulum')
        file_writer = tf.io.TFRecordWriter(current_shard_name)

        while not time_step.is_last():
            action_step = policy.action(time_step)
            # img.append(env.render(mode='rgb_array'))
            # action.append(action_step.action.numpy())
            # state.append(time_step.observation.numpy())
            # time_step = environment.step(action_step.action)
            # next_state.append(time_step.observation.numpy())
            img=env.render(mode='rgb_array')
            action=action_step.action.numpy()
            state=time_step.observation.numpy()
            time_step = environment.step(action_step.action)
            next_state=time_step.observation.numpy()
            # with tf.io.TFRecordWriter(current_shard_name) as file_writer:
            data = parse_images(img, action, state, next_state)
            record_bytes = tf.train.Example(features=tf.train.Features(feature=data)).SerializeToString()
            file_writer.write(record_bytes)
        file_writer.close()



def create_replay(agent):
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=2)

    return replay_buffer, rb_observer

def plot_data(iterations, returns):
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.show()












if __name__=='__main__':
    q_net = create_model(env)
    agent = create_agent(train_env, q_net)
    replay_buffer, rb_observer = create_replay(agent)

    example_environment = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))
    time_step = example_environment.reset()
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())

    py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps).run(train_py_env.reset())

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    # collect_data(eval_env, agent.policy, num_data_collection_episodes)

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    for _ in range(num_iterations):

        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    collect_data(eval_env, agent.policy, num_data_collection_episodes)
    iterations = range(0, num_iterations + 1, eval_interval)
    plot_data(iterations, returns)


    pass


# TO DO
# 2. add optical flow image to dataset so i can calculate velocities
# 3. add perturbations to the gym env. Probably want to use sporatic uniformly distributed perturbations centered on zero.
# 4. ASU spring registration.
# 5. send personalized instructor evaluation for Dr. Holman
# 6. tf.data.AUTOTUNE - optimize dataset performance https://www.tensorflow.org/guide/data_performance

# Done
# 1. finish adding and checking the tf dataset pipeline
