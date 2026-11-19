"""Resources for the optional fluid passes."""
import ctypes
import numpy as np
from pyglet import gl
from pyglet.graphics.shader import Shader, ShaderProgram
from ..particles import ParticleInstances


class ParticleBuffer:
    def __init__(self, device, **settings):
        self.instances = ParticleInstances(device, **settings)

    @property
    def count(self):
        return self.instances.count

    def reserve(self, count):
        self.instances.reserve(count)

    def update(self, positions, velocities, radii, config, *, transparent=False):
        alpha_low = config.transparent_alpha_low if transparent else config.alpha_low
        alpha_high = config.transparent_alpha_high if transparent else config.alpha_high
        self.instances.upload(positions, velocities, radii,
                              opacity_settings=(config.speed_low, config.speed_high, alpha_low, alpha_high))

    def draw_instances(self):
        self.instances.draw()

    def close(self):
        self.instances.close()


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
    def __init__(self, width, height, formats, *, depth=False):
        self.fbo = gl.GLuint()
        self.depth = gl.GLuint()
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
            if depth:
                gl.glGenRenderbuffers(1, ctypes.byref(self.depth))
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.depth)
                gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT32F, width, height)
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                             gl.GL_RENDERBUFFER, self.depth)
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
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
        if self.depth.value:
            gl.glDeleteRenderbuffers(1, ctypes.byref(self.depth))
            self.depth.value = 0
        for tex in self.textures:
            gl.glDeleteTextures(1, ctypes.byref(tex))
        self.textures.clear()
        if self.fbo.value:
            gl.glDeleteFramebuffers(1, ctypes.byref(self.fbo))
            self.fbo.value = 0
