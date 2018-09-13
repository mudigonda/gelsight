import numpy as np
import sys
sys.path.append('..')
import argparse
from demo_mujoco_pointmass import simulator
from demo_mujoco_pointmass import randpolicy 
import IPython;
import matplotlib.pyplot as plt
import cv2
from dotmap import DotMap


def compute_action(goal_img,curr_img):
    curr_img = cv2.cvtColor(curr_img,cv2.COLOR_RGB2GRAY)
    goal_img = cv2.cvtColor(goal_img,cv2.COLOR_RGB2GRAY)
    #plt.subplot(1,2,1)
    #plt.ion()
    #plt.imshow(curr_img)
    #plt.subplot(1,2,2)
    #plt.imshow(goal_img)
    #plt.show()
    #compute moments
    M_curr = cv2.moments(curr_img)
    currX= M_curr["m10"]/M_curr["m00"]
    currY= M_curr["m01"]/M_curr["m00"]
    M_goal = cv2.moments(goal_img)
    goalX = M_goal["m10"]/M_goal["m00"]
    goalY = M_goal["m01"]/M_goal["m00"]

    action = np.zeros(2)
    action[0] = goalX - currX
    action[1] = goalY - currY
    error = np.linalg.norm(goal_img - curr_img)/(100**2)
    print(action/100.)
    return error,action/100.


if __name__ == "__main__":
    AP = argparse.ArgumentParser()
    AP.add_argument("--type",type=str,default=None,help="The default mode is with init velocity, quasi runs the quasi static mode")
    AP.add_argument("--thresh",type=float,default=1e-3,help="Threshold for the L2 norm ")
    AP.add_argument("--horizon",type=int,default=5,help="Horizon for the episode ")
    AP.add_argument("--debug",type=str,default="False",help="Horizon for the episode ")
    parsed = AP.parse_args()
    
    #Get our final goal pose by running the sim once
    sim = simulator()
    goal_policy = randpolicy()
    goal_log = sim.run(horizon=parsed.horizon,policy=goal_policy)
    if parsed.debug == "True":
        plt.subplot(1,2,1)
        plt.title('begin')
        plt.imshow(goal_log['states'][0])
        plt.subplot(1,2,2)
        plt.title('end')
        plt.imshow(goal_log['states'][-1])
    #goal_img = goal_log['states'][-1]
    #goal_diff_img
    goal_img = goal_log['states'][-1] - goal_log['states'][-2]
    obs_init = sim._env.reset()
    state_init = sim.state2touch(obs_init)

    #Init error to be massive
    error = 1e+3
    logs = DotMap()
    logs.states = []
    logs.actions = []
    logs.rewards = []
    logs.times = []
    logs.obs = []
    ii = 0
    while(error > parsed.thresh):
    #for ii in range(200):
        #we take random action for 0th step
        if ii == 0:
            action = np.random.uniform(-10,10,2)
            error = 1e+3
        elif ii ==1:
            curr_img = logs['states'][-1] - state_init
            error,action = compute_action(goal_img,curr_img)
        else:
            curr_img = logs['states'][-1] - logs['states'][-2]
            error,action = compute_action(goal_img,curr_img)
        #apply action
        obs, reward, done, info = sim._env.step(action)
        #go from obs to state
        state = sim.state2touch(obs)
        logs.actions.append(action.tolist())
        logs.rewards.append(reward)
        logs.states.append(state)
        logs.obs.append(obs)
        ii += 1
        print("Current Error is {}".format(error))
    # Cluster state
    logs.actions = np.array(logs.actions)
    logs.rewards = np.array(logs.rewards)
    logs.states = np.array(logs.states)
    logs.obs = np.array(logs.obs)
    if parsed.debug == "True":
        plt.subplot(1,2,1)
        plt.imshow(curr_img)
        plt.plot(currY,currX,"r+")
        plt.subplot(1,2,2)
        plt.imshow(goal_img)
        plt.plot(goalY,goalX,"g*")
        plt.show()
    IPython.embed()
