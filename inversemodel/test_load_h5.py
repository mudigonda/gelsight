import deepdish as dd
import numpy as np
import os

path = '/home/ubuntu/Data/hd5/'
fnames = os.listdir(path)

images = np.zeros((len(fnames),18,48,64,3))
actions = np.zeros((len(fnames),3))

for ii, fname in enumerate(fnames):
  data = dd.io.load(path+fname,'/')
  for jj in range(18):
    images[ii,jj,...] = data['img_'+str(jj)]
    actions[ii,...] = data['action_'+str(jj)]

