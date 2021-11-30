import gym
from cassie import CassieEnv
from tf_agents.environments import suite_gym

# This should always work
env = CassieEnv()
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()


# Test tf-agents
# env = CassieEnv()
# env.reset()

# print('Observation Spec:')
# print(env.time_step_spec().observation)
pass