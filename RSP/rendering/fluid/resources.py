"""Private resources for optional modes; the legacy batch stays unchanged."""
import ctypes
import numpy as np
import warp as wp
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from ..billboard import BillboardBatch, _pack_instances, LOW_SPEED_COLOR, MID_SPEED_COLOR, HIGH_SPEED_COLOR


@wp.kernel
def opacity(velocity: wp.array(dtype=wp.vec3), packed: wp.array(dtype=wp.vec4),
            low: float, high: float, a: float, b: float):
    i = wp.tid()
    t = wp.clamp((wp.length(velocity[i])-low)/(high-low), 0., 1.)
    c = packed[2*i+1]
    packed[2*i+1] = wp.vec4(c[0], c[1], c[2], a+(b-a)*t*t*(3.-2.*t))


class ParticleBuffer(BillboardBatch):
    def update(self, positions, velocities, radii, config):
        count = len(positions)
        if count != self._capacity:
            self._resize(count)
        self.count = 0
        if not count:
            return
        data = self._registered.map(dtype=wp.vec4, shape=(2*count,))
        try:
            wp.launch(_pack_instances, count, inputs=[positions, velocities, radii,
                self.speed_color_mid, self.speed_color_max, wp.vec3(*LOW_SPEED_COLOR),
                wp.vec3(*MID_SPEED_COLOR), wp.vec3(*HIGH_SPEED_COLOR)], outputs=[data], device=self.device)
            wp.launch(opacity, count, inputs=[velocities, data, config.speed_low, config.speed_high,
                      config.alpha_low, config.alpha_high], device=self.device)
        finally:
            self._registered.unmap()
        self.count = count

    def draw_instances(self):
        gl.glBindVertexArray(self._vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None, self.count)


def program(vertex, fragment):
    return ShaderProgram(Shader(vertex, 'vertex'), Shader(fragment, 'fragment'))


def matrix(shader, name, value):
    loc = gl.glGetUniformLocation(shader.id, name.encode())
    array = np.ascontiguousarray(value, dtype=np.float32)
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, array.ctypes.data_as(ctypes.POINTER(gl.GLfloat)))


def texture(shader, name, target, unit):
    gl.glActiveTexture(gl.GL_TEXTURE0+unit)
    gl.glBindTexture(gl.GL_TEXTURE_2D, target)
    shader[name] = unit


class Target:
    def __init__(self, width, height, formats):
        self.fbo = gl.GLuint()
        self.textures = []
        gl.glGenFramebuffers(1, ctypes.byref(self.fbo))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        try:
            for i, (internal, external) in enumerate(formats):
                tex = gl.GLuint()
                gl.glGenTextures(1, ctypes.byref(tex))
                self.textures.append(tex)
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal, width, height, 0, external, gl.GL_FLOAT, None)
                for prop in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, prop, gl.GL_NEAREST)
                for prop in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T):
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, prop, gl.GL_CLAMP_TO_EDGE)
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0+i, gl.GL_TEXTURE_2D, tex, 0)
            attachments = (gl.GLenum*len(formats))(*(gl.GL_COLOR_ATTACHMENT0+i for i in range(len(formats))))
            gl.glDrawBuffers(len(formats), attachments)
            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError('Fluid framebuffer is incomplete')
        except BaseException:
            self.close()
            raise

    def bind(self, values=None):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        if values:
            for i, value in enumerate(values):
                gl.glClearBufferfv(gl.GL_COLOR, i, (gl.GLfloat*4)(*value))

    def close(self):
        for tex in self.textures:
            gl.glDeleteTextures(1, ctypes.byref(tex))
        self.textures.clear()
        if self.fbo.value:
            gl.glDeleteFramebuffers(1, ctypes.byref(self.fbo))
            self.fbo.value = 0
