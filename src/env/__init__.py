# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt

from gym.envs.registration import register

register(
    id='myparticle2D-v0',
    entry_point='src.env.particle2D:Particle2DEnv'
)