"""CUDA-fed instanced squares, independent of Newton and simulation ownership.

The caller owns the OpenGL context and must make it current for every method.
Instances occupy 32 bytes: center/radius followed by RGBA. A single Warp kernel
writes a registered GL buffer; no particle data crosses the CPU each frame.
"""

import ctypes
import math

import numpy as np
import warp as wp
from pyglet import gl

from .shaders import BILLBOARD_FRAGMENT_SHADER, BILLBOARD_VERTEX_SHADER

# WarpLab's speed palette, in linear shader values (no lighting or tone mapping).
LOW_SPEED_COLOR = (29.0 / 255.0, 119.0 / 255.0, 231.0 / 255.0)
MID_SPEED_COLOR = (160.0 / 255.0, 221.0 / 255.0, 1.0)
HIGH_SPEED_COLOR = (1.0, 1.0, 1.0)


@wp.kernel
def _pack_instances(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    speed_mid: float,
    speed_max: float,
    low_color: wp.vec3,
    mid_color: wp.vec3,
    high_color: wp.vec3,
    instances: wp.array(dtype=wp.vec4),
):
    i = wp.tid()
    p = positions[i]
    speed = wp.clamp(wp.length(velocities[i]), 0.0, speed_max)
    color = low_color
    if speed <= speed_mid:
        color = wp.lerp(low_color, mid_color, speed / speed_mid)
    else:
        color = wp.lerp(mid_color, high_color, (speed - speed_mid) / (speed_max - speed_mid))
    instances[2 * i] = wp.vec4(p[0], p[1], p[2], radii[i])
    instances[2 * i + 1] = wp.vec4(color[0], color[1], color[2], 1.0)


def _gl_integer(name):
    value = gl.GLint()
    gl.glGetIntegerv(name, ctypes.byref(value))
    return value.value


class BillboardBatch:
    """One opaque particle batch on a CUDA device and the current GL context.

    Input arrays are borrowed only during update(). draw() consumes owned GL
    storage, so cameras can move without updating simulation data. CPU fallback
    is deliberately disabled: silently copying would hide interop failures.
    """

    def __init__(self, device, *, speed_color_mid=2.0, speed_color_max=6.5):
        self._closed = False
        self._program = None
        self._vao = gl.GLuint()
        self._vertices = gl.GLuint()
        self._indices = gl.GLuint()
        self._instances = gl.GLuint()
        self._registered = None
        self.count = 0
        self._capacity = 0
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("BillboardBatch requires a CUDA device with OpenGL interoperability")
        if not (math.isfinite(speed_color_mid) and math.isfinite(speed_color_max)
                and 0.0 < speed_color_mid < speed_color_max):
            raise ValueError("Billboard speed thresholds must satisfy 0 < mid < max and be finite")
        self.speed_color_mid = float(speed_color_mid)
        self.speed_color_max = float(speed_color_max)

        # Importing pyglet.graphics creates a GL context. Keep it out of the
        # module import path so non-visual simulation needs no display server.
        from pyglet.graphics.shader import Shader, ShaderProgram

        previous_vao = _gl_integer(gl.GL_VERTEX_ARRAY_BINDING)
        previous_buffer = _gl_integer(gl.GL_ARRAY_BUFFER_BINDING)
        previous_program = _gl_integer(gl.GL_CURRENT_PROGRAM)
        try:
            self._program = ShaderProgram(Shader(BILLBOARD_VERTEX_SHADER, "vertex"),
                                          Shader(BILLBOARD_FRAGMENT_SHADER, "fragment"))
            self._view_location = gl.glGetUniformLocation(self._program.id, b"view")
            self._projection_location = gl.glGetUniformLocation(self._program.id, b"projection")
            gl.glGenVertexArrays(1, ctypes.byref(self._vao))
            gl.glGenBuffers(1, ctypes.byref(self._vertices))
            gl.glGenBuffers(1, ctypes.byref(self._indices))
            gl.glGenBuffers(1, ctypes.byref(self._instances))
            gl.glBindVertexArray(self._vao)
            vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vertices)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, ctypes.c_void_p(0))
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._indices)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instances)
            for attribute, offset in ((1, 0), (2, 16)):
                gl.glEnableVertexAttribArray(attribute)
                gl.glVertexAttribPointer(attribute, 4, gl.GL_FLOAT, gl.GL_FALSE, 32, ctypes.c_void_p(offset))
                gl.glVertexAttribDivisor(attribute, 1)
        except BaseException:
            self.close()
            raise
        finally:
            gl.glBindVertexArray(previous_vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, previous_buffer)
            gl.glUseProgram(previous_program)

    def _resize(self, count):
        # RegisteredGLBuffer releases CUDA registration when the last reference
        # is dropped. Do this BEFORE reallocating or deleting its GL storage.
        self._registered = None
        self._capacity = 0
        self.count = 0
        previous_buffer = _gl_integer(gl.GL_ARRAY_BUFFER_BINDING)
        try:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instances)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, count * 32, None, gl.GL_DYNAMIC_DRAW)
            self._registered = wp.RegisteredGLBuffer(
                self._instances.value, device=self.device,
                flags=wp.RegisteredGLBuffer.WRITE_DISCARD, fallback_to_copy=False)
            self._capacity = count
        finally:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, previous_buffer)

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
        self.count = 0
        if not count:
            return
        if count != self._capacity:
            self._resize(count)
        mapped = self._registered.map(dtype=wp.vec4, shape=(2 * count,))
        try:
            wp.launch(_pack_instances, dim=count,
                      inputs=[positions, velocities, radii, self.speed_color_mid, self.speed_color_max,
                              wp.vec3(*LOW_SPEED_COLOR), wp.vec3(*MID_SPEED_COLOR), wp.vec3(*HIGH_SPEED_COLOR)],
                      outputs=[mapped], device=self.device, record_tape=False)
        finally:
            self._registered.unmap()
        self.count = count

    def draw(self, view, projection):
        """Draw into the caller's framebuffer, preserving the GL state we change.

        Matrices use OpenGL column-major order, like Newton's camera matrices.
        Depth testing and depth writes are enabled; no blending or lighting is
        applied. Framebuffers, viewport and textures remain owned by the caller.
        """
        if self._closed or not self.count:
            return
        previous_program = _gl_integer(gl.GL_CURRENT_PROGRAM)
        previous_vao = _gl_integer(gl.GL_VERTEX_ARRAY_BINDING)
        previous_depth_mask = _gl_integer(gl.GL_DEPTH_WRITEMASK)
        previous_depth_func = _gl_integer(gl.GL_DEPTH_FUNC)
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
            gl.glBindVertexArray(self._vao)
            gl.glDrawElementsInstanced(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None, self.count)
        finally:
            gl.glBindVertexArray(previous_vao)
            gl.glUseProgram(previous_program)
            gl.glDepthMask(previous_depth_mask)
            gl.glDepthFunc(previous_depth_func)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, polygon_mode[0])
            for capability, was_enabled in zip(capabilities, enabled):
                (gl.glEnable if was_enabled else gl.glDisable)(capability)

    def close(self):
        """Release CUDA registration before GL buffers; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        self.count = 0
        self._registered = None
        for buffer in (self._instances, self._indices, self._vertices):
            if buffer.value:
                gl.glDeleteBuffers(1, ctypes.byref(buffer))
                buffer.value = 0
        if self._vao.value:
            gl.glDeleteVertexArrays(1, ctypes.byref(self._vao))
            self._vao.value = 0
        if self._program is not None:
            self._program.delete()
            self._program = None
