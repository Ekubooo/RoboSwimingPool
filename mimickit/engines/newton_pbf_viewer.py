"""Newton rendering with one instanced cube per PBF particle."""
import numpy as np
import trimesh
import warp as wp
import newton

from engines.newton_recorder import NewtonVideoRecorder
from util import resource_util


@wp.kernel
def particle_cubes(positions: wp.array(dtype=wp.vec3), radii: wp.array(dtype=float),
                   transforms: wp.array(dtype=wp.transform), scales: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    transforms[i] = wp.transform(positions[i], wp.quat_identity())
    scales[i] = wp.vec3(2.0 * radii[i])


class PBFViewer(newton.viewer.ViewerGL):
    """Keep the native particle visibility control, replacing only its geometry."""
    _cube_mesh = "/pbf/particle_cube"
    _cube_instances = "/model/particles"

    def _log_particles(self, state):
        if not self.model.particle_count:
            return
        hidden = not self.show_particles or self._layer_force_hidden()
        if hidden:
            if hasattr(self, "_particle_transforms"):
                self.log_instances(self._cube_instances, self._cube_mesh,
                                   None, None, None, None, hidden=True)
            return
        if not hasattr(self, "_particle_transforms"):
            # Duplicate vertices per face for flat cube normals. The prototype
            # is shared by all particles; no per-particle CPU objects are made.
            mesh = trimesh.creation.box(extents=[1., 1., 1.])
            points = wp.array(mesh.triangles.reshape(-1, 3), dtype=wp.vec3, device=self.device)
            indices = wp.array(np.arange(len(points)), dtype=wp.int32, device=self.device)
            normals = wp.array(np.repeat(mesh.face_normals, 3, axis=0), dtype=wp.vec3, device=self.device)
            self.log_mesh(self._cube_mesh, points, indices, normals=normals, hidden=True)
            n = self.model.particle_count
            self._particle_transforms = wp.empty(n, dtype=wp.transform, device=self.device)
            self._particle_scales = wp.empty(n, dtype=wp.vec3, device=self.device)
            self._particle_colors = wp.full(n, wp.vec3(.7, .6, .4), dtype=wp.vec3, device=self.device)
            self._particle_materials = wp.full(n, wp.vec4(.5, 0., 0., 0.), dtype=wp.vec4, device=self.device)
        points = self._apply_layer_transform_to_points(state.particle_q)
        wp.launch(particle_cubes, self.model.particle_count, inputs=[points, self.model.particle_radius],
                  outputs=[self._particle_transforms, self._particle_scales], device=self.device)
        self.log_instances(self._cube_instances, self._cube_mesh, self._particle_transforms,
                           self._particle_scales, self._particle_colors, self._particle_materials)

    def close(self):
        super().close()
        for name in ("_particle_transforms", "_particle_scales", "_particle_colors", "_particle_materials"):
            if hasattr(self, name):
                delattr(self, name)


class PBFVideoRecorder(NewtonVideoRecorder):
    def _build_viewer(self):
        self._viewer = resource_util.construct(PBFViewer, width=self._resolution[0],
                                                height=self._resolution[1], headless=True)
        self._engine.init_viewer(self._viewer)
