import sys
sys.path.append('/home/mudigonda/anaconda3/envs/gelsight/lib/python3.5/site-packages')
from gelsight import load_data, preprocess_data
import numpy as np


PATH = '/media/mudigonda/Projects/tactile-servo/data/pointmass/'


outputs = np.load('mean_subtr_output.npy')
input_std = np.load('input_std.npy')

outputs = outputs/input_std
np.save('mean_subtr_std_output.npy',outputs)

import IPython; IPython.embed()
