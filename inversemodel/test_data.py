import sys
sys.path.append('/home/mudigonda/anaconda3/envs/gelsight/lib/python3.5/site-packages')
from gelsight import load_data, preprocess_data
import numpy as np


PATH = '/media/mudigonda/Projects/tactile-servo/data/pointmass/'

inputs, outputs = load_data()
print("Loaded data")

inputs, input_actions, outputs = preprocess_data(inputs,outputs)


np.save(PATH + 'inputs.npy',inputs)
np.save(PATH + 'input_actions.npy',input_actions)
np.save(PATH + 'outputs.npy',outputs)

print("Preprocessing data done")

import IPython; IPython.embed()
