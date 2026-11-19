"""Lifetime of a CUDA-registered GL array buffer in the caller's context."""
import ctypes
from contextlib import contextmanager

import warp as wp
from pyglet import gl


def gl_integer(name):
    value = gl.GLint()
    gl.glGetIntegerv(name, ctypes.byref(value))
    return value.value


class RegisteredBuffer:
    def __init__(self, device):
        self.device = device
        self.name = gl.GLuint()
        self.registration = None
        gl.glGenBuffers(1, ctypes.byref(self.name))

    def resize(self, nbytes):
        if not self.name.value:
            raise RuntimeError("Cannot allocate a closed interop buffer")
        # Drop the registration before GL changes or destroys its storage.
        self.registration = None
        previous = gl_integer(gl.GL_ARRAY_BUFFER_BINDING)
        try:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.name)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, nbytes, None, gl.GL_DYNAMIC_DRAW)
            self.registration = wp.RegisteredGLBuffer(
                self.name.value, device=self.device, flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                fallback_to_copy=False)
        finally:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, previous)

    @contextmanager
    def mapped(self, *, dtype, shape):
        registration = self.registration
        array = registration.map(dtype=dtype, shape=shape)
        try:
            yield array
        finally:
            registration.unmap()

    def close(self):
        self.registration = None
        if self.name.value:
            gl.glDeleteBuffers(1, ctypes.byref(self.name))
            self.name.value = 0
