# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
from PIL import Image
from math import *

# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")


class GelSightRender:
    """
    GelSight Renderer
    """

    def __init__(self, p={}):
        """

        """
        self.resolution = [256, 256]  # Resolution window
        self.resolutionInput = [100, 100]  # Resolution of the input DepthMap
        self.depthmapClip = 0.5  # Maximum deformation of the gel
        self.markers = False  #

        # Position Camera
        self.camera = DotMap()
        self.camera.pos = [2, 0, 0]

        # Position and intensity of the lights
        self.radius = 1.5
        self.intensity = 0.8
        self.lights = [DotMap(), DotMap(), DotMap()]
        self.lights[0].angle = 0
        self.lights[0].direction = [3.0, self.radius*np.sin(self.lights[0].angle), self.radius*np.cos(self.lights[0].angle), self.intensity]
        self.lights[0].rgba = [0.0, 0.0, 1.0, 1.0]
        self.lights[1].angle = 2*np.pi*1/3
        self.lights[1].direction = [3.0, self.radius*np.sin(self.lights[1].angle), self.radius*np.cos(self.lights[1].angle), self.intensity]
        self.lights[1].rgba = [0.0, 1.0, 0.0, 1.0]
        self.lights[2].angle = 2*np.pi*2/3
        self.lights[2].direction = [3.0, self.radius*np.sin(self.lights[2].angle), self.radius*np.cos(self.lights[2].angle), self.intensity]
        self.lights[2].rgba = [1.0, 0.0, 0.0, 1.0]

        # Intensity of ambient light
        self.ambient_intensity = [0.1, 0.1, 0.1, 0.0]
        self.surface = GL_SMOOTH  # The surface type(Flat or Smooth)

        self.init()

    def init(self):
        """
        Initialiaze rendering
        :return:
        """

        glutInit(sys.argv)  # Initialize the OpenGL pipeline
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)  # Set OpenGL display mode
        # Set the Window size and position
        glutInitWindowSize(self.resolution[0], self.resolution[1])
        glutInitWindowPosition(50, 100)
        glutCreateWindow('GelSight Renderer')  # Create the window with given title

        # Set background color to black
        glClearColor(0.0, 0.0, 0.0, 0.0)

        d = sqrt(self.camera.pos[0]*self.camera.pos[0] +
                 self.camera.pos[1]*self.camera.pos[1] +
                 self.camera.pos[2]*self.camera.pos[2])

        # Set matrix mode
        glMatrixMode(GL_PROJECTION)

        # Reset matrix
        glLoadIdentity()
        glFrustum(-d * 0.5, d * 0.5, -d * 0.5, d * 0.5, d - 1.0, d + 1.0)

        # Set camera
        gluLookAt(self.camera.pos[0], self.camera.pos[1], self.camera.pos[2], 0, 0, 0, 0, 0, 1)

        # Set OpenGL parameters
        glEnable(GL_DEPTH_TEST)

        # Enable lighting
        glEnable(GL_LIGHTING)

        # Set light model
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, self.ambient_intensity)

        # Enable light number 0
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, self.lights[0].direction)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.lights[0].rgba)
        # Enable light number 1
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, self.lights[1].direction)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, self.lights[1].rgba)
        # Enable light number 2
        glEnable(GL_LIGHT2)
        glLightfv(GL_LIGHT2, GL_POSITION, self.lights[2].direction)
        glLightfv(GL_LIGHT2, GL_DIFFUSE, self.lights[2].rgba)

        # Setup the material
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

    def extractImage(self):
        """
        Extract the RGB pixels from the window
        :return:
        """
        buffer = (GLubyte * (3 * self.resolution[0] * self.resolution[1]))(0)
        glReadPixels(0, 0, self.resolution[0], self.resolution[1], GL_RGB, GL_UNSIGNED_BYTE, buffer)
        # Use PIL to convert raw RGB buffer and flip the right way up
        image = Image.frombytes(mode="RGB", size=(self.resolution[0], self.resolution[1]), data=buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def save2file(self, nameFile):
        """
        Save the OpenGL buffer to file
        :param nameFile:
        :return:
        """
        image = self.extractImage()
        image.save(nameFile)  # Save image to disk

    def draw(self, depthmap=None):
        """
        Draw the surface
        :param depthmap: array [x,y] in [0, 256]
        :return:
        """

        x = np.linspace(-1, 1, num=self.resolutionInput[0])
        y = np.linspace(-1, 1, num=self.resolutionInput[1])
        # TODO: add curvature to the surface
        # TODO: Add markers to the surface
        # Make gel
        z = np.ones((self.resolutionInput[0], self.resolutionInput[1]))  # Rest position of the surface
        # Add normalized depthmap
        z = z - self.depthmapClip * depthmap/256

        for j in range(self.resolutionInput[0]-1):
            glBegin(GL_QUAD_STRIP)  # Begin a strip
            for i in range(self.resolutionInput[1]):
                glNormal3f(z[i, j], x[j],  y[i])
                glVertex3f(z[i, j], x[j],  y[i])
                glNormal3f(z[i, j+1], x[j+1], y[i])
                glVertex3f(z[i, j+1], x[j+1], y[i])
            glEnd()  # End of the strip


    # def special(self, key, x, y):
    #     # Keyboard controller for sphere
    #
    #     # # Scale the sphere up or down
    #     # if key == GLUT_KEY_UP:
    #     #     self.user_height += 0.1
    #     # if key == GLUT_KEY_DOWN:
    #     #     self.user_height -= 0.1
    #     #
    #     # # Rotate the cube
    #     # if key == GLUT_KEY_LEFT:
    #     #     self.user_theta += 0.1
    #     # if key == GLUT_KEY_RIGHT:
    #     #     self.user_theta -= 0.1
    #
    #     # Toggle the surface
    #     if key == GLUT_KEY_F1:
    #         if self.surface == GL_FLAT:
    #             self.surface = GL_SMOOTH
    #         else:
    #             self.surface = GL_FLAT
    #
    #     glutPostRedisplay()
    #
    # # The idle callback
    # def idle(self):
    #     global last_time
    #     time = glutGet(GLUT_ELAPSED_TIME)
    #
    #     if last_time == 0 or time >= last_time + 40:
    #         last_time = time
    #         glutPostRedisplay()
    #         self.save2file()
    #
    # # The visibility callback
    # def visible(self, vis):
    #     if vis == GLUT_VISIBLE:
    #         glutIdleFunc(self.idle)
    #     else:
    #         glutIdleFunc(None)

    def render(self, depthmap=None):
        """
        Render a depthmap as a GelSight contact
        :param depthmap:
        :return:
        """

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set color to white
        glColor3f(1.0, 1.0, 1.0)

        # Set shade model
        glShadeModel(self.surface)

        self.draw(depthmap)
        glutSwapBuffers()

    def close(self):
        """
        Close rendering
        :return:
        """
        # TODO: implement me!
        pass


if __name__ == '__main__':

    a = GelSightRender()  # initialize renderer

    # -----------------------

    # Generate test depthmap
    from scipy.stats import multivariate_normal
    resolution = [100, 100]
    xyz = [0.5, 0]  # move to xy coordinates [-1,+1]
    x = np.linspace(-1, 1, resolution[0])
    y = np.linspace(-1, 1, resolution[1])
    xv, yv = np.meshgrid(x, y)
    pressure = 12
    depthmap = multivariate_normal.pdf(np.stack((xv, yv), axis=2), mean=xyz, cov=0.3)  # Gaussian
    depthmap = depthmap * 200 / depthmap.max()
    depthmap = np.uint8(np.maximum(np.minimum(depthmap, 255), 0))

    plt.figure()
    plt.imshow(depthmap)
    plt.show()

    a.render(depthmap=depthmap)
    a.save2file(nameFile='test_01.png')

    # -----------------------

    resolution = [100, 100]
    depthmap = np.random.uniform(0, 100, (resolution[0], resolution[1]))

    plt.figure()
    plt.imshow(depthmap)
    plt.show()

    a.render(depthmap=depthmap)
    a.save2file(nameFile='test_02.png')
    print('Done')

    # -----------------------
