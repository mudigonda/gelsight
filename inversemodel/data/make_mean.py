import sys
sys.path.append('/home/mudigonda/anaconda3/envs/gelsight/lib/python3.5/site-packages')
from gelsight import load_data, preprocess_data
import numpy as np


PATH = '/media/mudigonda/Projects/tactile-servo/data/pointmass/'


inputs = np.load(PATH + 'inputs.npy')

print("Inputs and Outputs")
print(inputs.shape)
print(outputs.shape)
input_mean = inputs.mean(axis=0)
np.save('input_means.npy',input_mean)
inputs = inputs - input_mean
np.save('mean_subtr_input.npy',inputs)

import IPython; IPython.embed()
