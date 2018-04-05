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


def run_controller(env, horizon, policy):

    logs = DotMap()
    logs.states = []
    logs.actions = []
    logs.rewards = []
    logs.times = []

    observation = env.reset()
    for t in range(horizon):
        env.render()
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


def main(horizon=100, seed=0):

    env_model = 'myparticle2D-v0'
    env = gym.make(env_model)
    env.seed(seed)
    logging.info('Initializing env: %s' % env_model)

    logs = []

    # target = [0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5]
    policy = mypolicy()
    logs.append(run_controller(env, horizon=horizon, policy=policy))

    plt.figure()
    plt.plot(logs[0].states)
    plt.legend()
    plt.show()

    plt.figure()
    plt.imshow(state2touch(logs[0].states[5]))
    plt.show()

    video = []
    for i in range(logs[0].states.shape[0]):
            video.append(state2touch(logs[0].states[i]))
    RGB2video(nameFile='gelsight_simulator', data=np.array(video))


def compute_bg_channel(dim, light_coordinates):
    """

    :param dim:
    :param light_coordinates:
    :return:
    """
    # Convert state to depth
    xyz = [light_coordinates[0], light_coordinates[1]]
    x = np.linspace(-1, 1, dim[0])
    y = np.linspace(-1, 1, dim[1])
    xv, yv = np.meshgrid(x, y)
    cov = 0.5
    luminance = 2
    channel = 256 * luminance * multivariate_normal.pdf(np.stack((xv, yv), axis=2), mean=xyz, cov=cov)
    return channel

def add_markers(depthmap, angle=0, xy_shift=0, size_marker=5, distance_marker=30):
    """

    :param depthmap:
    :param angle:
    :param xy_shift:
    :param size_marker:
    :param distance_marker:
    :return:
    """
    im = []
    return im


def depthmap2touch(depthmap):
    """

    :param depthmap:
    :return:
    """
    dim = depthmap.shape
    im = np.zeros((dim[0], dim[1], 3))

    light_pos = np.array([[0, -1 / 3],
                          [1 / 3, 1 / 3],
                          [-1 / 3, 1 / 3],
                          ])
    im[:, :, 0] += compute_bg_channel(dim=dim, light_coordinates=light_pos[0])  # R
    im[:, :, 1] += compute_bg_channel(dim=dim, light_coordinates=light_pos[1])  # G
    im[:, :, 2] += compute_bg_channel(dim=dim, light_coordinates=light_pos[2])  # B

    # Add contact
    im[:, :, 0] += depthmap
    im[:, :, 0] += depthmap
    im[:, :, 0] += depthmap

    # Add markers
    # TODO:

    im = np.clip(im, 0, 255)
    im = im.astype(np.uint8)

    return im


def state2touch(state, resolution=[256, 256]):
    """
    :param state: 4D
    :param resolution:
    :return:
    """
    # Convert state to depth
    xyz = [state[0], state[2]]  # move to xy coordinates
    x = np.linspace(-1, 1, resolution[0])
    y = np.linspace(-1, 1, resolution[1])
    xv, yv = np.meshgrid(x, y)
    pressure = 5
    depthmap = pressure * multivariate_normal.pdf(np.stack((xv, yv), axis=2), mean=xyz, cov=0.01)
    depthmap = np.maximum(np.minimum(depthmap, 50), 10)

    im = depthmap2touch(depthmap=depthmap)

    return im


if __name__ == '__main__':
    main()
