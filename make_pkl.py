import sys
sys.path.append('/home/mudigonda/anaconda3/envs/gelsight/lib/python3.5/site-packages')

import src
import os
import numpy as np
#import src.env
import logging
logging.basicConfig(
    filename="makepkl.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )
logger = logging.getLogger()



def log2forwardDataset(log):
    input = []
    output = []
    for i in range(log.actions.shape[0]-1):
        input.append([np.rollaxis(log.states[i], 2, 0), log.actions[i]])  # Note the 3 x H x W format!
        output.append(log.states[i+1])
    return [input, output]

folderData =  '/media/mudigonda/Projects/tactile-servo/data/pointmass/'
extension = '.log'

if __name__ == "__main__":

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

