import gym
from cassie import CassieEnv


# # Instantiate the env   
env = CassieEnv()
env.reset()
for _ in range(100000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

pass