"""Quad geometry and CUDA-written instance records shared by fluid passes."""
import ctypes
import math

import numpy as np
import warp as wp
from pyglet import gl

from .interop import RegisteredBuffer, gl_integer


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


@wp.kernel
def opacity(velocity: wp.array(dtype=wp.vec3), packed: wp.array(dtype=wp.vec4),
            low: float, high: float, a: float, b: float):
    i = wp.tid()
    t = wp.clamp((wp.length(velocity[i])-low)/(high-low), 0., 1.)
    c = packed[2*i+1]
    packed[2*i+1] = wp.vec4(c[0], c[1], c[2], a+(b-a)*t*t*(3.-2.*t))


class ParticleInstances:
    """Own the vertex layout; the caller selects the shader and draw state."""

    def __init__(self, device, *, speed_color_mid=2.0, speed_color_max=6.5):
        self._vao = gl.GLuint()
        self._vertices = gl.GLuint()
        self._indices = gl.GLuint()
        self.buffer = None
        self.count = 0
        self._capacity = 0
        self._closed = False
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("Particle instances require a CUDA device with OpenGL interoperability")
        if not (math.isfinite(speed_color_mid) and math.isfinite(speed_color_max)
                and 0.0 < speed_color_mid < speed_color_max):
            raise ValueError("Billboard speed thresholds must satisfy 0 < mid < max and be finite")
        self.speed_color_mid = float(speed_color_mid)
        self.speed_color_max = float(speed_color_max)

        previous_vao = gl_integer(gl.GL_VERTEX_ARRAY_BINDING)
        previous_buffer = gl_integer(gl.GL_ARRAY_BUFFER_BINDING)
        try:
            gl.glGenVertexArrays(1, ctypes.byref(self._vao))
            gl.glGenBuffers(1, ctypes.byref(self._vertices))
            gl.glGenBuffers(1, ctypes.byref(self._indices))
            self.buffer = RegisteredBuffer(self.device)
            gl.glBindVertexArray(self._vao)
            vertices = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vertices)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, ctypes.c_void_p(0))
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._indices)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffer.name)
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

    def reserve(self, count):
        if self._closed:
            raise RuntimeError("Cannot resize closed particle instances")
        if count and count != self._capacity:
            self._capacity = self.count = 0
            self.buffer.resize(count * 32)
            self._capacity = count

    def upload(self, positions, velocities, radii, *, opacity_settings=None):
        count = len(positions)
        self.count = 0
        self.reserve(count)
        if not count:
            return
        with self.buffer.mapped(dtype=wp.vec4, shape=(2 * count,)) as data:
            wp.launch(_pack_instances, dim=count,
                      inputs=[positions, velocities, radii, self.speed_color_mid, self.speed_color_max,
                              wp.vec3(*LOW_SPEED_COLOR), wp.vec3(*MID_SPEED_COLOR), wp.vec3(*HIGH_SPEED_COLOR)],
                      outputs=[data], device=self.device, record_tape=opacity_settings is not None)
            if opacity_settings is not None:
                wp.launch(opacity, count, inputs=[velocities, data, *opacity_settings], device=self.device)
        self.count = count

    def draw(self):
        gl.glBindVertexArray(self._vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None, self.count)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._capacity = self.count = 0
        if self.buffer is not None:
            self.buffer.close()
        for buffer in (self._indices, self._vertices):
            if buffer.value:
                gl.glDeleteBuffers(1, ctypes.byref(buffer))
                buffer.value = 0
        if self._vao.value:
            gl.glDeleteVertexArrays(1, ctypes.byref(self._vao))
            self._vao.value = 0
