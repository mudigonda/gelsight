import sys
sys.path.append('/home/mudigonda/anaconda3/envs/gelsight/lib/python3.5/site-packages')
from gelsight import load_data, preprocess_data
import numpy as np


PATH = '/media/mudigonda/Projects/tactile-servo/data/pointmass/'


outputs = np.load(PATH + 'outputs.npy')

print("Inputs and Outputs")
print(outputs.shape)
input_mean = np.load('input_means.npy')
outputs = outputs - input_mean.transpose(1,2,0)
np.save('mean_subtr_output.npy',outputs)

import IPython; IPython.embed()
