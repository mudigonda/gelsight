# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
from past.builtins import basestring
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
# import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# import R.data as rdata
import progressbar
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# import scipyplot as spp

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from RGB2video import RGB2video
from gelsightrenderer import GelSightRender

import gym
# from gym.monitoring import VideoRecorder
import src.env
import scipyplot as spp

import logging
logging.basicConfig(
    filename="test.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# # To log file
# fh = logging.FileHandler('example.log')
# fh.setLevel(logging.DEBUG)
# logger.addHandler(fh)


class mypolicy():

    def __init__(self):
        pass

    def act(self, state):
        return [0.7, 0.3]


class simulator():
    def __init__(self):

        self.env = 'myparticle2D-v0'
        self._env = []
        self.policy = []
        self._renderer = None
        self.horizon = 100
        self.resolutionIn = [100, 100]
        self.resolutionOut = [100, 100]

    def run_controller(self, env, horizon, policy):

        logs = DotMap()
        logs.states = []
        logs.actions = []
        logs.rewards = []
        logs.times = []

        observation = env.reset()
        for t in range(horizon):
            # env.render()
            state = observation
            # print(state)
            action = policy.act(state)

            observation, reward, done, info = env.step(action)

            # Log
            # logs.times.append()
            logs.actions.append(action)
            logs.rewards.append(reward)
            logs.states.append(observation)

        # Cluster state
        logs.actions = np.array(logs.actions)
        logs.rewards = np.array(logs.rewards)
        logs.states = np.array(logs.states)
        return logs

    def run(self, horizon=100, seed=0):

        env = gym.make(self.env)
        env.seed(seed)
        logging.info('Initializing env: %s' % self.env)

        self._renderer = GelSightRender()

        logs = []

        # target = [0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5]
        policy = mypolicy()
        logs.append(self.run_controller(env, horizon=horizon, policy=policy))

        plt.figure()
        plt.plot(logs[0].states)
        plt.legend()
        plt.show()

        plt.figure()
        plt.imshow(self.state2touch(logs[0].states[5]))
        plt.show()

        video = []
        for i in range(logs[0].states.shape[0]):
                video.append(self.state2touch(logs[0].states[i]))
        RGB2video(nameFile='gelsight_simulator', data=np.array(video))

    def state2touch(self, state, resolution=[100, 100]):
        """
        :param state: 4D
        :param resolution:
        :return:
        """
        # Convert state to depth
        xyz = [state[0], state[1]]  # move to xy coordinates [-1,+1]
        x = np.linspace(-1, 1, self.resolutionIn[0])
        y = np.linspace(-1, 1, self.resolutionIn[1])
        xv, yv = np.meshgrid(x, y)

        # Gaussian
        depthmap = multivariate_normal.pdf(np.stack((xv, yv), axis=2), mean=xyz, cov=0.15)  # Gaussian
        depthmap = 230 * depthmap / depthmap.max()

        # Sphere

        depthmap = np.uint8(np.maximum(np.minimum(depthmap, 255), 0))

        # plt.figure()
        # plt.imshow(depthmap)
        # plt.show()

        rgb = self._renderer.render(depthmap=depthmap)

        return rgb


if __name__ == '__main__':
    a = simulator()
    a.run()
