# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
from past.builtins import basestring
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
# import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# import progressbar
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import src
import src.env

# import progressbar
import argparse
import gym
import time
import os
# from gym.monitoring import VideoRecorder
from collections import OrderedDict

# import scipyplot as spp

import logging
logging.basicConfig(
    filename="test.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# # To log file
# fh = logging.FileHandler('example.log')
# fh.setLevel(logging.DEBUG)
# logger.addHandler(fh)


class mypolicy(object):

    def __init__(self):
        pass

    def act(self, state):
        return np.array([0.7, 0.3])


class randpolicy(object):

    def __init__(self):
        pass

    def act(self, state):
        return np.random.uniform(-10, 10, 2)


class simulator(object):
    def __init__(self):

        self.env = 'myparticle2D-v0'
        self._env = []
        self.policy = []
        self._renderer = None
        self.horizon = 100
        self.resolutionIn = [100, 100]
        #self.resolutionOut = [126, 126]
        self.resolutionOut = self.resolutionIn

    def run_controller(self, horizon, policy):

        logs = DotMap()
        logs.states = []
        logs.actions = []
        logs.rewards = []
        logs.times = []
        logs.obs = []

        observation = self._env.reset()
        print("Env has been reset")
        for t in range(horizon):
            # env.render()
            state = self.state2touch(observation)
            print("Go from state to touch")
            # print(state)
            action = policy.act(state)
            print("Get an action based on policy")

            observation, reward, done, info = self._env.step(action)
            print("Perform an action")

            # Log
            # logs.times.append()
            logs.actions.append(action.tolist())
            logs.rewards.append(reward)
            logs.states.append(state)
            logs.obs.append(observation)

        # Cluster state
        logs.actions = np.array(logs.actions)
        logs.rewards = np.array(logs.rewards)
        logs.states = np.array(logs.states)
        logs.obs = np.array(logs.obs)
        return logs

    def run(self, horizon=100, seed=0, policy=mypolicy()):

        self._env = gym.make(self.env)
        self._env.seed(seed)
        logger.info('Initializing env: %s' % self.env)
        p = DotMap()
        p.resolutionOutput = self.resolutionOut
        print("src.GelSightRender with parameters p")
        self._renderer = src.GelSightRender(parameters=p)  # Init GelSight renderer

        log = []
        # target = [0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5]
        self.policy = policy
        log = self.run_controller(horizon=horizon, policy=policy)
        print("controller has run")

        # plt.figure()
        # plt.plot(log.obs)
        # plt.legend()
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(log.states[5])
        # plt.show()

        return log

    def state2touch(self, state):
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

        # # Gaussian
        # depthmap = multivariate_normal.pdf(np.stack((xv, yv), axis=2), mean=xyz, cov=0.15)  # Gaussian
        # depthmap = 230 * depthmap / depthmap.max()

        # Sphere
        radius = 0.2
        z = np.power(radius, 2) - np.power(xv-xyz[0], 2) - np.power(yv-xyz[1], 2)
        depthmap = np.clip(z, 0, np.inf)
        depthmap = 200*depthmap / depthmap.max()
        depthmap = np.uint8(np.maximum(np.minimum(depthmap, 255), 0))

        rgb = self._renderer.render(depthmap=depthmap)

        # plt.figure()
        # plt.imshow(depthmap)
        # plt.show()
        # plt.figure()
        # plt.imshow(rgb)
        # plt.show()

        return rgb


def log2forwardDataset(log):
    input = []
    output = []
    for i in range(log.actions.shape[0]-1):
        input.append([np.rollaxis(log.states[i], 2, 0), log.actions[i]])  # Note the 3 x H x W format!
        output.append(log.states[i+1])
    return [input, output]


def saveLog2Video(log, nameFile='gelsight_simulator'):
    video = []
    for i in range(log.states.shape[0]):
        video.append(log.states[i])
    src.RGB2video(nameFile=nameFile, data=np.array(video))
    return


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nc = 3
        self.nz = 32
        self.encoder = nn.Sequential(OrderedDict([
            # 3 x 256 x 256
            ('conv1', nn.Conv2d(self.nc, 8, kernel_size=4, stride=2, padding=1)),
            ('conv2', nn.LeakyReLU(0.2, inplace=True)),
            # ('conv3', nn.MaxPool2d(2, stride=2)),  # b, 16, 5, 5
            ('conv4', nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)),  # b, 8, 3, 3
            ('conv5', nn.LeakyReLU(0.2, inplace=True)),
            # ('conv6', nn.MaxPool2d(2, stride=1)),  # b, 8, 2, 2
            ('conv7', nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)),  # b, 8, 3, 3
            ('conv8', nn.LeakyReLU(0.2, inplace=True)),
            # 32 x 32 x 32
        ]))
        self.decoder = nn.Sequential(
            #
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # b, 16, 5, 5
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.ConvTranspose2d(8, self.nc, kernel_size=4, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, tactile, action):
        x = self.encoder(tactile)  # 32 x 32 x 32
        x = 256*self.decoder(x)
        # x = tactile
        return x

    def predict(self, datapoint, useGPU=False):
        self.eval()
        # TODO: accept multiple datapoints
        tactile = torch.from_numpy(datapoint[0]).float()
        action = torch.from_numpy(datapoint[1]).float()
        tactile = Variable(tactile[None, :])
        action = Variable(action[None, :])  # temp
        # if useGPU:
        #       self.cuda()
        #     tactile.cuda()
        #     action.cuda()
        pred = self.forward(tactile=tactile, action=action)
        return pred.data.numpy().squeeze()

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


def train_network(dataset, model, parameters=DotMap()):
    import torch.optim as optim

    p = DotMap()
    p.opt.n_epochs = parameters.get('n_epochs', 10)
    p.opt.optimizer = optim.Adam
    p.opt.batch_size = parameters.get('batch_size', 100)
    p.opt.learning_rate = parameters.get('learning_rate', 0.0001)
    p.criterion = parameters.get('criterion', nn.MSELoss())
    p.useGPU = parameters.get('useGPU', True)
    p.verbosity = parameters.get('verbosity', 1)
    p.logs = parameters.get('logs', None)

    # Init logs
    if p.logs is None:
        logs = DotMap()
        logs.training_error = []
        logs.time = None
    else:
        logs = p.logs

    # Optimizer
    optimizer = p.opt.optimizer(model.parameters(), lr=p.opt.learning_rate)

    if p.useGPU:
        cudnn.benchmark = True
        model.cuda()
        p.criterion.cuda()

    class PytorchDataset(Dataset):
        def __init__(self, dataset):
            self.inputs = dataset[0]
            self.outputs = dataset[1]
            self.n_data = len(dataset[0])
            # self.n_inputs = dataset[0].shape[1]
            # self.n_outputs = dataset[1].shape[1]

        def __getitem__(self, index):
            # print('\tcalling Dataset:__getitem__ @ idx=%d' % index)
            input = [torch.from_numpy(self.inputs[index][0]).float(), torch.from_numpy(self.inputs[index][1]).float()]
            output = torch.from_numpy(self.outputs[index]).float()
            return input, output

        def __len__(self):
            # print('\tcalling Dataset:__len__')
            return self.n_data

    logger.info('Training NN from dataset')
    dataset = PytorchDataset(dataset=dataset)
    loader = DataLoader(dataset, batch_size=p.opt.batch_size, shuffle=True)  ##shuffle=True #False
        # pin_memory=True
        # drop_last=False

    startTime = timer()
    if logs.time is None:
        logs.time = [0]

    for epoch in range(p.opt.n_epochs):
        for i, data in enumerate(loader, 0):
            # Load data
            inputs, targets = data
            #
            tactile = Variable(inputs[0]).cuda()
            action = Variable(inputs[1]).cuda()
            targets = Variable(targets).cuda()
            if p.useGPU:
                tactile = tactile.cuda()
                action = action.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model.forward(tactile=tactile, action=action)
            loss = p.criterion(outputs, targets)

            e = loss.data[0]
            logs.training_error.append(e)
            logger.info('Iter %010d - %f ' % (epoch, e))
            loss.backward()
            optimizer.step()  # Does the update
            logs.time.append(timer() - logs.time[-1])

    endTime = timer()
    logger.info('Optimization completed in %f[s]' % (endTime - startTime))
    return model.cpu(), logs

def str2bool(varName):
    if varName == "False":
        return False
    else:
        return True

def HWC2CHW(input):
   return np.rollaxis(input, 0, 3).astype(np.uint8)
if __name__ == '__main__':

    COLLECT_DATA = True
    CREATE_DATASET = True
    TRAIN_MODEL = True

    folderData = 'data/pointmass'
    extension = '.log'
    AP = argparse.ArgumentParser()
    AP.add_argument('--N_REPS',default=1,type=int,help="Number of Episodes")
    AP.add_argument('--COLLECT_DATA',default="True",type=str,help="Number of Episodes")
    AP.add_argument('--CREATE_DATASET',default="True",type=str,help="Number of Episodes")
    AP.add_argument('--TRAIN_MODEL',default="False",type=str,help="Number of Episodes")
    parsed = AP.parse_args()

    parsed.COLLECT_DATA = str2bool(parsed.COLLECT_DATA)
    parsed.CREATE_DATASET = str2bool(parsed.CREATE_DATASET)
    parsed.TRAIN_MODEL = str2bool(parsed.TRAIN_MODEL)

    # Collect random data
    if COLLECT_DATA:
        src.create_folder(folderData)
        N_REPS = parsed.N_REPS
        HORIZON = 50
        POLICY = randpolicy()
        print("Starting simulator")
        a = simulator()
        for i in range(N_REPS):
            print("Starting to run the simulator")
            log = a.run(horizon=HORIZON, policy=POLICY)
            nameFile = time.strftime("%Y-%m-%d_%H%M%S")
            src.SaveData(log, fileName='%s/%s.log' % (folderData, nameFile))

    # Create dataset
    if parsed.CREATE_DATASET:
        logger.info('Creating Dataset')
        inputs = []
        outputs = []
        listLogs = [each for each in os.listdir(folderData) if each.endswith(extension)]
        for nameFile in listLogs:
            log = src.LoadData('%s/%s' % (folderData, nameFile))
            t_dataset = log2forwardDataset(log)
            inputs = inputs + t_dataset[0]
            outputs = outputs + t_dataset[1]

        dataset = [inputs, outputs]
        src.SaveData(dataset, fileName='%s/dataset.pkl' % folderData)

    # Train Model
    if parsed.TRAIN_MODEL:

        def init_weights(m):
            if type(m) == nn.Linear:
                m.weight.data.fill_(1.0)

        dataset = src.LoadData('%s/dataset.pkl' % folderData)
        dataset = [dataset[0][0:2], dataset[1][0:2]]
        model = Net()
        model.train()
        model.apply(init_weights)
        p = DotMap()
        p.useGPU = True
        p.n_epochs = 10000
        p.learning_rate = 0.0005
        model, logs = train_network(dataset=dataset, model=model, parameters=p)
        torch.save(model, '%s/model.pt' % folderData)

        dataset = src.LoadData('%s/dataset.pkl' % folderData)
        model = torch.load('%s/model.pt' % folderData)
        idx = 0
        pred = model.predict(dataset[0][idx])

        plt.figure()
        plt.imshow(HWC2CHW(dataset[0][idx][0]))

        plt.figure()
        plt.imshow(HWC2CHW(pred))
        plt.show()
