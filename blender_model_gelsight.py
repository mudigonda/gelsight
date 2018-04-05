# TODO: Delete every other object
import bpy

resolution_sensor = [640, 480]

bpy.ops.object.camera_add(view_align=False, enter_editmode=False, location=(0.0, 0.0, 4.0),
                          rotation=(0.0, 0.0, 0.0))
bpy.ops.object.lamp_add(type='POINT', radius=1.0, view_align=False, location=(-1.0, -1.0, 1.0),
                        rotation=(0.0, 0.0, 0.0))
bpy.ops.object.lamp_add(type='POINT', radius=1.0, view_align=False, location=(-1.0, 1.0, 1.0),
                        rotation=(0.0, 0.0, 0.0))
bpy.ops.object.lamp_add(type='POINT', radius=1.0, view_align=False, location=(1.0, 0.0, 1.0),
                        rotation=(0.0, 0.0, 0.0))
bpy.ops.mesh.primitive_grid_add(x_subdivisions=resolution_sensor[0], y_subdivisions=resolution_sensor[1], radius=1.0,
                                view_align=False,
                                enter_editmode=False, location=(0.0, 0.0, 0.1), rotation=(0.0, 0.0, 0.0))
# TODO: change colors lights

scene = list(bpy.data.objects)
scene[1].name = 'light_r'
scene[2].name = 'light_g'
scene[3].name = 'light_b'
