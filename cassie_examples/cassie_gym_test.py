import gym
import numpy as np
from cassie import CassieEnv
from tf_agents.environments import suite_gym

# This should always work
env = CassieEnv()
env.reset()
for i in range(1000000000):
    env.render()
    test = np.sin(i/1000)*2
    print(test)
    action = np.asarray([0,0,0,test,0, 0,0,0,0,0])
    env.step(action)
    # env.step(env.action_space.sample()) # take a random action
env.close()


# Test tf-agents
# env = CassieEnv()
# env.reset()

# print('Observation Spec:')
# print(env.time_step_spec().observation)
pass