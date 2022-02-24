import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register


class InvertedPendulumEnv3DNoise(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "/home/local/ASUAD/gmclark1/Research/git/Bayesian_NN_Filter/pendulum_examples3d/inverted_pendulum_back.xml", 2)
        register(id='3DCartPole-v2', entry_point='pendulum_examples3d.cartpole_noise3d:InvertedPendulumEnv3DNoise')
        self.action_max  = 1
        self.action_space = spaces.Discrete(20)

    def step(self, a):
        action = (a-10)*(1/10)
        reward = 1.0
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 1
        v.cam.distance = self.model.stat.extent
        v.cam.elevation = -10
        pass