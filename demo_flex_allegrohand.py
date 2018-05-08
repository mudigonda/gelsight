# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt

import os
import random
from ctypes import *

# Load Flex Gym library
debug = 0
if (os.name == "nt"):
    flexGymPath = os.path.dirname(os.path.realpath(__file__)) + "/../../bin/win64/"
    os.chdir(flexGymPath)
    if (debug):
        flexGym = cdll.LoadLibrary("NvFlexGymDebugCUDA_x64")
    else:
        flexGym = cdll.LoadLibrary("NvFlexGymReleaseCUDA_x64")
else:
    flexGymPath = os.path.dirname(os.path.realpath(__file__)) + "/../../bin/linux64/"
    os.chdir(flexGymPath)
    if (debug):
        flexGym = cdll.LoadLibrary(flexGymPath + "NvFlexGymDebugCUDA_x64.so")
    else:
        flexGym = cdll.LoadLibrary(flexGymPath + "NvFlexGymReleaseCUDA_x64.so")

# Initialize Flex Gym
flexGym.NvFlexGymInit(None)

# Parameters
loadPath = '"../../data/ant.xml"'
numAgents = 500
numObservations = 39
numActions = 8
numSubsteps = 4
numIterations = 25
pause = 'false'
doLearning = 'true'

# Load a scene
flexGym.NvFlexGymLoadScene('RL Ant', f'''
                                     {{
                                         "LoadPath": {loadPath},
                                         "NumAgents": {numAgents},
                                         "NumObservations": {numObservations},
                                         "NumActions": {numActions},
                                         "NumSubsteps": {numSubsteps},
                                         "NumIterations": {numIterations},
                                         "Pause": {pause},
                                         "DoLearning": {doLearning}
                                     }}
                                     ''')

# Buffers
totalActions = numAgents * numActions
ActionBuffType = c_float * totalActions
actionBuff = ActionBuffType()
totalObservations = numAgents * numObservations
ObservationBuffType = c_float * totalObservations
observationBuff = ObservationBuffType()
RewardBuffType = c_float * numAgents
rewardBuff = RewardBuffType()
DeathBuffType = c_byte * numAgents
deathBuff = DeathBuffType()

# Simulation loop
quit = 0
while (quit == 0):
    for agent in range(0, numAgents):
        for action in range(0, numActions):
            actionBuff[agent * numActions + action] = random.uniform(-1, 1)
    flexGym.NvFlexGymSetActions(actionBuff, 0, totalActions)
    quit = flexGym.NvFlexGymUpdate()
    flexGym.NvFlexGymGetRewards(rewardBuff, deathBuff, 0, numAgents)
    for agent in range(0, numAgents):
        if (deathBuff[agent]):
            flexGym.NvFlexGymResetAgent(agent)
    flexGym.NvFlexGymGetObservations(observationBuff, 0, totalObservations)
    # flexGym.NvFlexGymGetExtras(extraBuff, 0, totalExtras) # Number ???
    # flexGym.NvFlexGymResetAllAgents()
    # flexGym.NvFlexGymGetObservations(agentObservationBuff, agent * numObservations, numObservations) # Get agent observations

# Shutdown Flex Gym
flexGym.NvFlexGymShutdown()
