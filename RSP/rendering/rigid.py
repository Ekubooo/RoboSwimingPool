"""Shared CUDA-written rigid instances, using Newton's existing mesh shaders.

One 96-byte record contains a column-major transform, RGB/opacity, and material.
Meshes are borrowed; each draw group owns only its VAO. CUDA registration and
the shared instance buffer belong to RigidBatch, not to the individual groups.
"""
import ctypes

import warp as wp
from pyglet import gl

from .interop import RegisteredBuffer, gl_integer


@wp.kernel
def _pack_rigids(
    shape_ids: wp.array(dtype=int),
    extra: wp.array(dtype=wp.transform),
    scales: wp.array(dtype=wp.vec3),
    materials: wp.array(dtype=wp.vec4),
    shape_body: wp.array(dtype=int),
    shape_local: wp.array(dtype=wp.transform),
    shape_world: wp.array(dtype=int),
    colors: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    world_offsets: wp.array(dtype=wp.vec3),
    layer: wp.transform,
    packed: wp.array(dtype=wp.vec4),
):
    i = wp.tid()
    shape = shape_ids[i]
    pose = shape_local[shape]
    body = shape_body[shape]
    if body >= 0:
        pose = wp.transform_multiply(body_q[body], pose)
    world = shape_world[shape]
    if world >= 0 and world < world_offsets.shape[0]:
        pose = wp.transform(wp.transform_get_translation(pose) + world_offsets[world],
                            wp.transform_get_rotation(pose))
    pose = wp.transform_multiply(layer, wp.transform_multiply(pose, extra[i]))
    p = wp.transform_get_translation(pose)
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    scale = scales[i]
    for column in range(3):
        packed[6 * i + column] = wp.vec4(rotation[0, column] * scale[column],
                                        rotation[1, column] * scale[column],
                                        rotation[2, column] * scale[column], 0.0)
    packed[6 * i + 3] = wp.vec4(p[0], p[1], p[2], 1.0)
    color = colors[shape]
    packed[6 * i + 4] = wp.vec4(color[0], color[1], color[2], wp.clamp(opacities[shape], 0.0, 1.0))
    packed[6 * i + 5] = materials[i]


class RigidDrawable:
    """A mesh draw range in the shared buffer; follows Newton's scene protocol."""
    def __init__(self, batch, mesh, offset, count, *, fallback_texture, transparent=False, draw_edge=True):
        self.mesh = mesh
        self.hidden = False
        self.cast_shadow = True
        self.draw_edge = draw_edge
        self._fallback_texture = fallback_texture
        self.active_instances = count
        self.transparent = transparent
        self.vao = gl.GLuint()
        previous_vao = gl_integer(gl.GL_VERTEX_ARRAY_BINDING)
        previous_buffer = gl_integer(gl.GL_ARRAY_BUFFER_BINDING)
        try:
            gl.glGenVertexArrays(1, ctypes.byref(self.vao))
            gl.glBindVertexArray(self.vao)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh.ebo)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh.vbo)
            for attribute, size, start in ((0, 3, 0), (1, 3, 12), (2, 2, 24)):
                gl.glVertexAttribPointer(attribute, size, gl.GL_FLOAT, gl.GL_FALSE,
                                         mesh.vertex_byte_size, ctypes.c_void_p(start))
                gl.glEnableVertexAttribArray(attribute)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, batch.buffer.name)
            fields = [(3 + i, 4, i * 16) for i in range(4)] + [(7, 3, 64), (8, 4, 80), (9, 1, 76)]
            for attribute, size, start in fields:
                gl.glVertexAttribPointer(attribute, size, gl.GL_FLOAT, gl.GL_FALSE,
                                         96, ctypes.c_void_p(offset * 96 + start))
                gl.glEnableVertexAttribArray(attribute)
                gl.glVertexAttribDivisor(attribute, 1)
        except BaseException:
            self.destroy()
            raise
        finally:
            gl.glBindVertexArray(previous_vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, previous_buffer)

    def has_transparency(self):
        return self.transparent and not self.hidden and self.active_instances > 0

    def render(self):
        if self.hidden or not self.vao.value or not self.active_instances:
            return
        (gl.glEnable if self.mesh.backface_culling else gl.glDisable)(gl.GL_CULL_FACE)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        texture = self.mesh.texture_id
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture if texture is not None else self._fallback_texture())
        gl.glBindVertexArray(self.vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.mesh.num_indices, gl.GL_UNSIGNED_INT,
                                   None, self.active_instances)
        gl.glBindVertexArray(0)

    def destroy(self):
        if self.vao.value:
            gl.glDeleteVertexArrays(1, ctypes.byref(self.vao))
            self.vao.value = 0


class RigidBatch:
    """One registration/map/kernel/unmap per layer, regardless of draw count."""
    def __init__(self, model, shape_ids, extras, scales, materials):
        self.model = model
        self.device = model.device
        self.count = len(shape_ids)
        self.buffer = None
        self.drawables = []
        self.closed = False
        self.valid = False
        if not self.device.is_cuda:
            raise ValueError("RigidBatch requires CUDA/OpenGL interoperability")
        self.shape_ids = wp.array(shape_ids, dtype=int, device=self.device)
        self.extras = wp.array(extras, dtype=wp.transform, device=self.device)
        self.scales = wp.array(scales, dtype=wp.vec3, device=self.device)
        self.materials = wp.array(materials, dtype=wp.vec4, device=self.device)
        self.empty_offsets = wp.empty(0, dtype=wp.vec3, device=self.device)
        try:
            if self.count:
                self.buffer = RegisteredBuffer(self.device)
                self.buffer.resize(self.count * 96)
        except BaseException:
            self.close()
            raise

    def add_drawable(self, mesh, offset, count, **options):
        drawable = RigidDrawable(self, mesh, offset, count, **options)
        self.drawables.append(drawable)
        return drawable

    def update(self, state, world_offsets, layer):
        if self.closed:
            raise RuntimeError("Cannot update a closed rigid batch")
        self.valid = False
        if not self.count:
            return
        with self.buffer.mapped(dtype=wp.vec4, shape=(self.count * 6,)) as packed:
            model = self.model
            wp.launch(_pack_rigids, self.count,
                      inputs=[self.shape_ids, self.extras, self.scales, self.materials,
                              model.shape_body, model.shape_transform, model.shape_world,
                              model.shape_color, model.shape_opacity, state.body_q,
                              world_offsets if world_offsets is not None else self.empty_offsets, layer],
                      outputs=[packed], device=self.device, record_tape=False)
        self.valid = True

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.valid = False
        for drawable in self.drawables:
            drawable.destroy()
        if self.buffer is not None:
            self.buffer.close()
