"""Interface to the baxter data (todo)"""

import sys
import os
#sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import scipy
import random
import math
import pickle

data_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
DATASET = '/media/mudigonda/Projects/tactile-servo/data/pointmass/'
print(DATASET)
EPISODE_LENGTH = 50
TOT_SAMPLES = 49000
TRAIN_SAMPLES = 45000
IM_SIZE = 100
CHANNELS = 3
ACTION_DIM = 2


def load_data():
    data = pickle.load(open(DATASET+'dataset.pkl','rb'))
    inputs = data[0]
    outputs = data[1]
    #Apply off-shifts between inputs and outputs
    return inputs, outputs

def get_batch(batch_size, MAX_TRAIN):
    val = np.random.randint(0,MAX_TRAIN,batch_size)
    return val

def preprocess_data(inputs, outputs):
    input_im = np.zeros((TOT_SAMPLES,CHANNELS, IM_SIZE, IM_SIZE))
    input_action = np.zeros((TOT_SAMPLES, ACTION_DIM))
    outputs = np.asarray(outputs) #just going from list to np array
    import IPython; IPython.embed()
    for ii in range(TOT_SAMPLES):
        input_im[ii,...] = inputs[ii][0]
        input_action[ii,...] = inputs[ii][1]
    return input_im, input_action, outputs

def compute_mean_std(input_im,outputs):
    #preprocess data
    mean_im = input_im.mean(axis=0)

    #subtracting mean and div by std
    input_im = input_im - mean_im 
    print("subtracting the mean done")
    std_im = input_im.std(axis=0)
    print("computing std done")
    input_im = input_im/ std_im
    print("dividing by std div")
    outputs = outputs - mean_im.transpose(1,2,0)
    outputs = outputs/std_im.transpose(1,2,0)
    #align data

    return input_im, input_action,  outputs
