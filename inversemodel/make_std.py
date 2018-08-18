import sys
sys.path.append('/home/mudigonda/anaconda3/envs/gelsight/lib/python3.5/site-packages')
from gelsight import load_data, preprocess_data
import numpy as np


PATH = '/media/mudigonda/Projects/tactile-servo/data/pointmass/'


inputs = np.load('mean_subtr_input.npy')

print("Inputs and Outputs")
print(inputs.shape)
input_std = inputs.std(axis=0)
np.save('input_std.npy',input_std)
inputs = inputs/input_std
np.save('mean_subtr_std_input.npy',inputs)

import IPython; IPython.embed()
