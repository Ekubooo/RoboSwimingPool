"""CUDA-fed instanced squares, independent of Newton and simulation ownership.

The caller owns the OpenGL context and must make it current for every method.
Instances occupy 32 bytes: center/radius followed by RGBA. A single Warp kernel
writes a registered GL buffer; no particle data crosses the CPU each frame.
"""

import ctypes

import numpy as np
import warp as wp
from pyglet import gl

from .shaders import BILLBOARD_FRAGMENT_SHADER, BILLBOARD_VERTEX_SHADER
from .interop import gl_integer
from .particles import ParticleInstances


class BillboardBatch:
    """Draw a CUDA particle instance buffer with the original opaque shaders."""

    def __init__(self, device, **settings):
        self._closed = False
        self._program = None
        self.instances = None
        # Importing pyglet.graphics can create a GL context.
        from pyglet.graphics.shader import Shader, ShaderProgram

        previous_program = gl_integer(gl.GL_CURRENT_PROGRAM)
        try:
            self.instances = ParticleInstances(device, **settings)
            self.device = self.instances.device
            self._program = ShaderProgram(Shader(BILLBOARD_VERTEX_SHADER, "vertex"),
                                          Shader(BILLBOARD_FRAGMENT_SHADER, "fragment"))
            self._view_location = gl.glGetUniformLocation(self._program.id, b"view")
            self._projection_location = gl.glGetUniformLocation(self._program.id, b"projection")
        except BaseException:
            self.close()
            raise
        finally:
            gl.glUseProgram(previous_program)

    @property
    def count(self):
        return self.instances.count

    def update(self, positions, velocities, radii):
        """Borrow contiguous GPU arrays with matching lengths; never mutate them."""
        if self._closed:
            raise RuntimeError("Cannot update a closed BillboardBatch")
        for name, array, dtype in (("positions", positions, wp.vec3),
                                   ("velocities", velocities, wp.vec3), ("radii", radii, wp.float32)):
            if not isinstance(array, wp.array) or array.dtype != dtype or array.ndim != 1:
                raise ValueError(f"Billboard {name} must be a one-dimensional Warp array of {dtype}")
            if array.device != self.device or not array.is_contiguous:
                raise ValueError(f"Billboard {name} must be contiguous and on {self.device}")
        count = len(positions)
        if len(velocities) != count or len(radii) != count:
            raise ValueError("Billboard positions, velocities and radii must have matching lengths")
        self.instances.upload(positions, velocities, radii)

    def draw(self, view, projection):
        """Draw into the caller's framebuffer, preserving the GL state we change.

        Matrices use OpenGL column-major order, like Newton's camera matrices.
        Depth testing and depth writes are enabled; no blending or lighting is
        applied. Framebuffers, viewport and textures remain owned by the caller.
        """
        if self._closed or not self.count:
            return
        previous_program = gl_integer(gl.GL_CURRENT_PROGRAM)
        previous_vao = gl_integer(gl.GL_VERTEX_ARRAY_BINDING)
        previous_depth_mask = gl_integer(gl.GL_DEPTH_WRITEMASK)
        previous_depth_func = gl_integer(gl.GL_DEPTH_FUNC)
        polygon_mode = (gl.GLint * 2)()
        gl.glGetIntegerv(gl.GL_POLYGON_MODE, polygon_mode)
        capabilities = (gl.GL_BLEND, gl.GL_CULL_FACE, gl.GL_DEPTH_TEST)
        enabled = [gl.glIsEnabled(capability) for capability in capabilities]
        try:
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glUseProgram(self._program.id)
            for location, matrix in ((self._view_location, view), (self._projection_location, projection)):
                values = np.ascontiguousarray(matrix, dtype=np.float32)
                gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE,
                                      values.ctypes.data_as(ctypes.POINTER(gl.GLfloat)))
            self.instances.draw()
        finally:
            gl.glBindVertexArray(previous_vao)
            gl.glUseProgram(previous_program)
            gl.glDepthMask(previous_depth_mask)
            gl.glDepthFunc(previous_depth_func)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, polygon_mode[0])
            for capability, was_enabled in zip(capabilities, enabled):
                (gl.glEnable if was_enabled else gl.glDisable)(capability)

    def close(self):
        """Release CUDA registration before deleting the GL buffer."""
        if self._closed:
            return
        self._closed = True
        if self.instances is not None:
            self.instances.close()
        if self._program is not None:
            self._program.delete()
            self._program = None
