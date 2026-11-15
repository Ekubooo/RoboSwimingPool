"""Post-scene fluid composition, before Newton's UI and displayed-frame copy."""
import ctypes
import time
import numpy as np
from pyglet import gl
from .config import MODES, SSFR_PRESETS, WATER_COLORS, FluidRenderConfig
from . import shaders as s
from .resources import ParticleBuffer, Target, program, matrix, texture


class FluidPipeline:
    def __init__(self, viewer):
        self.viewer = viewer
        self.mode = 'Original'
        self.pending = None
        self.ssfr_preset = 'Water'
        self.pending_preset = None
        self.water_color = 'Current'
        self.pending_water_color = None
        self.error = None
        self.config = FluidRenderConfig()
        self.batch = None
        self.visible = False
        self.programs = {}
        self.targets = {}
        self.size = None
        self.last_switch_ms = 0.
        self.original_render = viewer.renderer.render
        viewer.renderer.render = self.render
        self.vao = gl.GLuint()

    def request(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        self.pending = mode

    def apply_pending(self):
        if self.pending is None and self.pending_preset is None and self.pending_water_color is None:
            return
        mode = self.pending or self.mode
        preset = self.pending_preset or self.ssfr_preset
        water_color = self.pending_water_color or self.water_color
        self.pending = self.pending_preset = None
        self.pending_water_color = None
        start = time.perf_counter()
        try:
            self.viewer.renderer._make_current()
            if mode != 'Original':
                self.prepare(mode, preset)
                # Validate CUDA registration before committing the switch. It
                # must fail here, rather than after hiding the previous mode.
                count = self.viewer.model.particle_count
                if count and self.batch._capacity != count:
                    self.batch._resize(count)
            self.mode, self.error = mode, None
            self.ssfr_preset = preset
            self.water_color = water_color
        except Exception as error:
            self.error = f'Cannot enable {mode} / {preset}: {error}'
        finally:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self.last_switch_ms = (time.perf_counter()-start)*1000

    def request_ssfr_preset(self, preset):
        if preset not in SSFR_PRESETS:
            raise ValueError(preset)
        self.pending_preset = preset

    def request_water_color(self, color):
        if color not in WATER_COLORS:
            raise ValueError(color)
        self.pending_water_color = color

    def prepare(self, mode, preset=None):
        velocity = mode == 'SSFR' and (preset or self.ssfr_preset) == 'Velocity'
        if self.batch is None:
            self.batch = ParticleBuffer(self.viewer.device, **self.viewer._billboard_settings)
            gl.glGenVertexArrays(1, ctypes.byref(self.vao))
        definitions = {'transparent': (s.PARTICLE, s.TRANSPARENT), 'resolve': (s.QUAD, s.RESOLVE)} if mode == 'Transparent' else {
            'depth': (s.PARTICLE, s.DEPTH), 'thickness': (s.PARTICLE, s.THICKNESS),
            'blur': (s.QUAD, s.BLUR), 'surface': (s.QUAD, s.SURFACE)}
        if velocity:
            definitions.update(thickness_velocity=(s.PARTICLE, s.THICKNESS_VELOCITY),
                               surface_velocity=(s.QUAD, s.SURFACE_VELOCITY))
        for name, sources in definitions.items():
            if name not in self.programs:
                self.programs[name] = program(*sources)
        renderer = self.viewer.renderer
        size = (renderer._screen_width, renderer._screen_height)
        if size != self.size:
            for target in self.targets.values():
                target.close()
            self.targets.clear()
            self.size = size
        formats = {'output': [(gl.GL_RGBA16F, gl.GL_RGBA)]}
        if mode == 'Transparent':
            formats['oit'] = [(gl.GL_RGBA32F, gl.GL_RGBA), (gl.GL_R16F, gl.GL_RED)]
        else:
            for name in ('depth', 'horizontal', 'smooth'):
                formats[name] = [(gl.GL_RG32F, gl.GL_RG)]
            formats['thickness'] = [(gl.GL_R16F, gl.GL_RED)]
            if velocity:
                formats['thickness_velocity'] = [
                    (gl.GL_R16F, gl.GL_RED),     # Unchanged fluid thickness.
                    (gl.GL_RGBA16F, gl.GL_RGBA), # Weighted RGB and weight.
                    (gl.GL_R16F, gl.GL_RED),     # Weighted speed coverage.
                ]
        for name, fmt in formats.items():
            if name not in self.targets:
                self.targets[name] = Target(*size, fmt)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def update(self, state):
        points = self.viewer._apply_layer_transform_to_points(state.particle_q)
        self.batch.update(points, state.particle_qd, self.viewer.model.particle_radius, self.config)

    def full_screen(self):
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    def particle_pass(self, name, radius):
        r = self.viewer.renderer
        shader = self.programs[name]
        with shader:
            matrix(shader, 'view', r._view_matrix)
            matrix(shader, 'projection', r._projection_matrix)
            # Camera matrices are column-major flat arrays.
            inv = np.linalg.inv(np.asarray(r._projection_matrix).reshape(4, 4, order='F'))
            matrix(shader, 'inv_projection', inv.T)
            shader['size'] = self.size
            shader['radius_scale'] = radius
            texture(shader, 'scene_depth', r._frame_depth_texture, 0)
            if name == 'transparent':
                shader['soft_width'] = self.config.soft_intersection
            elif name == 'thickness_velocity':
                texture(shader, 'nearest_depth', self.targets['depth'].textures[0], 1)
                shader['alpha_range'] = (self.config.alpha_low, self.config.alpha_high)
                shader['attribute_depth_range'] = self.config.attribute_depth_range
            self.batch.draw_instances()

    def transparent(self):
        from .transparent import render
        render(self)

    def ssfr(self):
        from .ssfr import render
        render(self)

    def render(self, *args, **kwargs):
        self.original_render(*args, **kwargs)
        if self.mode == 'Original' or not self.visible or self.batch is None or not self.batch.count:
            return
        self.prepare(self.mode)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDepthMask(False)
        try:
            if self.mode == 'Transparent':
                self.transparent()
            else:
                self.ssfr()
            # Commit the completed image to Newton's frame texture and window.
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.targets['output'].fbo)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            for destination in (self.viewer.renderer._frame_fbo, 0):
                gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, destination)
                gl.glBlitFramebuffer(0,0,*self.size,0,0,*self.size,gl.GL_COLOR_BUFFER_BIT,gl.GL_NEAREST)
        finally:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glBindVertexArray(0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glDisable(gl.GL_BLEND)
            gl.glBlendEquation(gl.GL_FUNC_ADD)
            gl.glDepthMask(True)
            gl.glEnable(gl.GL_DEPTH_TEST)

    def close(self):
        if self.batch:
            self.batch.close()
            self.batch = None
        for target in self.targets.values():
            target.close()
        self.targets.clear()
        for shader in self.programs.values():
            shader.delete()
        self.programs.clear()
        if self.vao.value:
            gl.glDeleteVertexArrays(1, ctypes.byref(self.vao))
            self.vao.value = 0
        self.size = None
