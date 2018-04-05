# # Compatibility Python 2/3
# from __future__ import division, print_function, absolute_import
# from builtins import range
# # ----------------------------------------------------------------------------------------------------------------------
#
# import numpy as np
# from dotmap import DotMap
# import matplotlib.pyplot as plt

import bpy
import mathutils


resolution = 640, 480
namefile = '/home/rcalandra/Dropbox/Research/tactile-servo/gelsight.png'

# Change shape mesh
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.object.mode_set(mode='OBJECT')
idx_mesh = 1
scene = list(bpy.data.objects)
scene[1]

# Save to file
bpy.data.scenes["Scene"].render.filepath = namefile
bpy.ops.render.render(write_still=True)
