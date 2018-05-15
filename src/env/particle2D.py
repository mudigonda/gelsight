import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Particle2DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dir_path, 'assets/particle2D.xml'), 5)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, 0, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        # Let' set (roughly) how far the viewer is from the robot arena
        self.viewer.cam.distance = 6.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # Observation is equivalent to state
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
        ])
