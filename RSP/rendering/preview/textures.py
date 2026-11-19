"""Selection-owned OpenGL thumbnails. No model, window or disk cache ownership."""
from collections import OrderedDict
from contextlib import contextmanager
import ctypes
import logging


@contextmanager
def preserved_gl_state(gl):
    def integer(name, count=1):
        values = (gl.GLint*count)()
        gl.glGetIntegerv(name, values)
        return tuple(values)
    bindings = {name: integer(name)[0] for name in (
        gl.GL_DRAW_FRAMEBUFFER_BINDING, gl.GL_READ_FRAMEBUFFER_BINDING,
        gl.GL_RENDERBUFFER_BINDING, gl.GL_CURRENT_PROGRAM, gl.GL_VERTEX_ARRAY_BINDING,
        gl.GL_ARRAY_BUFFER_BINDING, gl.GL_TEXTURE_BINDING_2D, gl.GL_PIXEL_UNPACK_BUFFER_BINDING)}
    viewport = integer(gl.GL_VIEWPORT, 4)
    flags = {flag: gl.glIsEnabled(flag) for flag in
             (gl.GL_DEPTH_TEST, gl.GL_BLEND, gl.GL_CULL_FACE, gl.GL_SCISSOR_TEST, gl.GL_FRAMEBUFFER_SRGB)}
    clear = (gl.GLfloat*4)()
    gl.glGetFloatv(gl.GL_COLOR_CLEAR_VALUE, clear)
    line_width = (gl.GLfloat*1)()
    gl.glGetFloatv(gl.GL_LINE_WIDTH, line_width)
    depth_clear = (gl.GLdouble*1)()
    gl.glGetDoublev(gl.GL_DEPTH_CLEAR_VALUE, depth_clear)
    depth_func = integer(gl.GL_DEPTH_FUNC)[0]
    depth_mask = (gl.GLboolean*1)()
    color_mask = (gl.GLboolean*4)()
    gl.glGetBooleanv(gl.GL_DEPTH_WRITEMASK, depth_mask)
    gl.glGetBooleanv(gl.GL_COLOR_WRITEMASK, color_mask)
    try:
        yield
    finally:
        for flag, enabled in flags.items():
            (gl.glEnable if enabled else gl.glDisable)(flag)
        gl.glLineWidth(line_width[0])
        gl.glClearDepth(depth_clear[0])
        gl.glDepthFunc(depth_func)
        gl.glDepthMask(depth_mask[0])
        gl.glColorMask(*color_mask)
        gl.glClearColor(*clear)
        gl.glViewport(*viewport)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, bindings[gl.GL_DRAW_FRAMEBUFFER_BINDING])
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, bindings[gl.GL_READ_FRAMEBUFFER_BINDING])
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, bindings[gl.GL_RENDERBUFFER_BINDING])
        gl.glUseProgram(bindings[gl.GL_CURRENT_PROGRAM])
        gl.glBindVertexArray(bindings[gl.GL_VERTEX_ARRAY_BINDING])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, bindings[gl.GL_ARRAY_BUFFER_BINDING])
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, bindings[gl.GL_PIXEL_UNPACK_BUFFER_BINDING])
        gl.glBindTexture(gl.GL_TEXTURE_2D, bindings[gl.GL_TEXTURE_BINDING_2D])


class AssetPreviews:
    """Requests originate in UI drawing; process_one executes outside that callback."""
    width, height = 512, 384

    def __init__(self, assets):
        self.assets = assets
        self.textures, self.errors = {}, {}
        self.pending = OrderedDict()
        self.closed = False

    def request(self, ref, color=None):
        key = (ref, None if color is None else tuple(color))
        if not self.closed and key not in self.textures and key not in self.errors:
            self.pending[key] = None
        return self.textures.get(key), self.errors.get(key)

    def process_one(self):
        if self.closed or not self.pending:
            return
        key, _ = self.pending.popitem(last=False)
        try:
            from .geometry import load_geometry, project_geometry
            triangles, lines = project_geometry(load_geometry(self.assets[key[0]], key[1]))
            self.textures[key] = self._render(triangles, lines)
        except Exception as error:
            self.errors[key] = str(error)
            logging.exception('Asset preview failed: %s', key[0])

    def _render(self, triangles, lines):
        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram
        texture, framebuffer, depth = gl.GLuint(), gl.GLuint(), gl.GLuint()
        program = None
        shaders = []
        vao, buffer = gl.GLuint(), gl.GLuint()
        success = False
        with preserved_gl_state(gl):
            try:
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
                gl.glGenTextures(1, ctypes.byref(texture))
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, self.width, self.height,
                                0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
                for parameter in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, gl.GL_LINEAR)
                gl.glGenFramebuffers(1, ctypes.byref(framebuffer))
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                         gl.GL_TEXTURE_2D, texture, 0)
                gl.glGenRenderbuffers(1, ctypes.byref(depth))
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth)
                gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height)
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth)
                if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError('Preview framebuffer is incomplete')
                shaders.append(Shader('''#version 330 core
layout(location=0) in vec3 position; layout(location=1) in vec3 color; out vec3 rgb;
void main(){ gl_Position=vec4(position,1.0); rgb=color; }''', 'vertex'))
                shaders.append(Shader('''#version 330 core
in vec3 rgb; out vec4 result;
void main(){ result=vec4(rgb,1.0); }''', 'fragment'))
                program = ShaderProgram(*shaders)
                program.use()
                gl.glViewport(0, 0, self.width, self.height)
                for flag in (gl.GL_BLEND, gl.GL_CULL_FACE, gl.GL_SCISSOR_TEST, gl.GL_FRAMEBUFFER_SRGB):
                    gl.glDisable(flag)
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthFunc(gl.GL_LESS)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glColorMask(True, True, True, True)
                gl.glLineWidth(4.)
                gl.glClearDepth(1.)
                gl.glClearColor(.075, .09, .115, 1.)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                gl.glGenVertexArrays(1, ctypes.byref(vao))
                gl.glGenBuffers(1, ctypes.byref(buffer))
                gl.glBindVertexArray(vao)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
                for index in (0, 1):
                    gl.glEnableVertexAttribArray(index)
                    gl.glVertexAttribPointer(index, 3, gl.GL_FLOAT, False, 24, ctypes.c_void_p(index*12))
                for vertices, mode in ((triangles, gl.GL_TRIANGLES), (lines, gl.GL_LINES)):
                    if len(vertices):
                        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes,
                                        vertices.ctypes.data_as(ctypes.c_void_p), gl.GL_STREAM_DRAW)
                        gl.glDrawArrays(mode, 0, len(vertices))
                success = True
                return texture.value
            finally:
                gl.glDeleteBuffers(1, ctypes.byref(buffer))
                gl.glDeleteVertexArrays(1, ctypes.byref(vao))
                if program is not None:
                    program.delete()
                for shader in shaders:
                    shader.delete()
                gl.glDeleteRenderbuffers(1, ctypes.byref(depth))
                gl.glDeleteFramebuffers(1, ctypes.byref(framebuffer))
                if not success:
                    gl.glDeleteTextures(1, ctypes.byref(texture))

    def close(self):
        if self.closed:
            return
        from pyglet import gl
        for value in self.textures.values():
            texture = gl.GLuint(value)
            gl.glDeleteTextures(1, ctypes.byref(texture))
        self.textures.clear()
        self.pending.clear()
        self.errors.clear()
        self.closed = True
