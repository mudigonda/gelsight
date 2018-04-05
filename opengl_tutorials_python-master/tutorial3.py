#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" Tutorial 3: Matrices 

	Note(s):
		- Spend several days/weeks going over matrices.  If your linear-algebra-foo
		is weak you are going to have a painful time with OpenGL.
"""

from __future__ import print_function

from OpenGL.GL import *
from OpenGL.GL.ARB import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.special import *
from OpenGL.GL.shaders import *
from glew_wish import *
from csgl import *

import common
import glfw
import sys
import os

# Global window
window = None
null = c_void_p(0)


def opengl_init():
    global window
    # Initialize the library
    if not glfw.init():
        print("Failed to initialize GLFW\n", file=sys.stderr)
        return False

    # Open Window and create its OpenGL context
    window = glfw.create_window(1024, 768, "Tutorial 03", None,
                                None)  # (in the accompanying source code this variable will be global)
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    if not window:
        print(
            "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n",
            file=sys.stderr)
        glfw.terminate()
        return False

    # Initialize GLEW
    glfw.make_context_current(window)
    glewExperimental = True

    # GLEW is a framework for testing extension availability.  Please see tutorial notes for
    # more information including why can remove this code.
    if glewInit() != GLEW_OK:
        print("Failed to initialize GLEW\n", file=sys.stderr);
        return False
    return True


def main():
    if not opengl_init():
        return

    glfw.set_input_mode(window, glfw.STICKY_KEYS, GL_TRUE)

    # Set opengl clear color to something other than red (color used by the fragment shader)
    glClearColor(0, 0, 0.4, 0)

    vertex_array_id = glGenVertexArrays(1)
    glBindVertexArray(vertex_array_id)

    program_id = common.LoadShaders("./shaders/Tutorial3/SimpleTransform.vertexshader",
                                    "./shaders/Tutorial3/SingleColor.fragmentshader")

    # Get a handle for our "MVP" uniform
    matrix_id = glGetUniformLocation(program_id, "MVP");

    # Projection matrix : 45 Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    projection = mat4.perspective(45.0, 4.0 / 3.0, 0.1, 100.0)

    # Camera matrix
    view = mat4.lookat(vec3(4, 3, 3),  # Camera is at (4,3,3), in World Space
                       vec3(0, 0, 0),  # and looks at the origin
                       vec3(0, 1, 0))

    # Model matrix : an identity matrix (model will be at the origin)
    model = mat4.identity()

    # Our ModelViewProjection : multiplication of our 3 matrices
    mvp = projection * view * model

    vertex_data = [-1.0, -1.0, 0.0,
                   1.0, -1.0, 0.0,
                   0.0, 1.0, 0.0]

    vertex_buffer = glGenBuffers(1);

    # GLFloat = c_types.c_float
    array_type = GLfloat * len(vertex_data)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, len(vertex_data) * 4, array_type(*vertex_data), GL_STATIC_DRAW)

    while glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(program_id)

        # Send our transformation to the currently bound shader,
        # in the "MVP" uniform
        glUniformMatrix4fv(matrix_id, 1, GL_FALSE, mvp.data)
        # Bind vertex buffer data to the attribute 0 in our shader.
        # Note:  This can also be done in the VAO itself (see vao_test.py)

        # Enable the vertex attribute at element[0], in this case that's the triangle's vertices
        # this could also be color, normals, etc.  It isn't necessary to disable these
        #
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glVertexAttribPointer(
            0,  # attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,  # len(vertex_data)
            GL_FLOAT,  # type
            GL_FALSE,  # ormalized?
            0,  # stride
            null  # array buffer offset (c_type == void*)
        )

        # Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, 3)  # 3 indices starting at 0 -> 1 triangle

        # Not strictly necessary because we only have
        glDisableVertexAttribArray(0)

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    # note braces around vertex_buffer and vertex_array_id.
    # These 2 functions expect arrays of values
    glDeleteBuffers(1, [vertex_buffer])
    glDeleteProgram(program_id)
    glDeleteVertexArrays(1, [vertex_array_id])
    glfw.terminate()


if __name__ == "__main__":
    main()
