"""Scene assembly and lifecycle glue for the copied Newton-PBF backend."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import trimesh
import warp as wp

from RSP.sim_ready.pipelines.scenes import pbf_play as assets
from RSP.sim_ready.lib.storage.store import AssetStore
from RSP.sim_ready.lib.asset_validation import (ValidationReport, diagnostic_scope,
    validate_physical_asset, validate_skeleton)
from RSP.sim_ready.adapters.newton import describe_articulations, validate_model_for_mimickit
from RSP.sim_ready.pipelines.articulation import prepare_articulated_asset
from RSP.sim_ready.config import BakeOptions
from solvers.pbf import boundary_model, pbf_helper


@wp.kernel
def sample_sdf(volume: wp.uint64, points: wp.array(dtype=wp.vec3), distances: wp.array(dtype=float)):
    i = wp.tid()
    index = wp.volume_world_to_index(volume, points[i])
    distances[i] = wp.volume_sample_f(volume, index, wp.Volume.LINEAR)


def runtime_config(control_dt, substeps, *, pressure_iterations=None, time_scale=1.0):
    """Keep physical scalars; use the play scene's Z-up particle layout."""
    original = pbf_helper.PBFConfig()
    transform = assets.SCENE_TRANSFORM
    return replace(original, frame_dt=control_dt * time_scale, substeps=substeps,
                   pressure_iterations=(original.pressure_iterations if pressure_iterations is None
                                        else pressure_iterations),
                   gravity=tuple(transform[:3, :3] @ original.gravity),
                   container_size=tuple(np.abs(transform[:3, :3]) @ original.container_size),
                   block_start=assets.FLUID_BLOCK_START, block_end=assets.FLUID_BLOCK_END,
                   initial_velocity=tuple(transform[:3, :3] @ original.initial_velocity))


class PBFScene:
    def __init__(self, directory, device, control_dt, substeps, *, fluid_substeps=None,
                 pressure_iterations=None, fluid_time_scale=1.0):
        self.store = AssetStore(directory)
        self.directory = self.store.root
        self.device = device
        self.control_dt = control_dt
        self.physics_dt = control_dt / substeps
        self.fluid_time_scale = fluid_time_scale
        self.config = runtime_config(control_dt, substeps if fluid_substeps is None else fluid_substeps,
                                     pressure_iterations=pressure_iterations, time_scale=fluid_time_scale)
        self.particles = SimpleNamespace(config=self.config)
        pbf_helper.init_scene_parameters(self.particles)
        self.primitive_files = assets.primitive_files(self.store)
        self.body_meshes = {}
        self.boundaries = None
        self.validation = ValidationReport()
        self._skeletons = {}
        self._excluded_links = {}
        self._validated_model = None

    def register_character(self, obj_id, skeleton, excluded_links=()):
        self._skeletons[obj_id] = skeleton
        self._excluded_links[obj_id] = tuple(excluded_links)

    def validate_model(self, model):
        if self._validated_model is model:
            return
        with diagnostic_scope(self.validation, "newton.describe", "scene"):
            bindings = describe_articulations(model)
        self.validation.extend(validate_model_for_mimickit(model, bindings))
        for obj_id, binding in enumerate(bindings):
            self.validation.extend(validate_physical_asset(binding.asset, self._excluded_links.get(obj_id, ())))
            if obj_id in self._skeletons:
                self.validation.extend(validate_skeleton(binding.asset, self._skeletons[obj_id]))
        if set(self._skeletons) - set(range(len(bindings))):
            self.validation.add("error", "skeleton.object", "scene", "Registered character has no physical articulation")
        self.validation.require_valid()
        self._bindings = bindings
        self._validated_model = model

    def add_static(self, builder):
        with diagnostic_scope(self.validation, "static.prepare", "pool and bunny"):
            self.static_path, self.bunny, hulls = assets.prepare_static(
                self.store, self.device, self.particles.smoothing_length, report=self.validation)
        self.initial_clearances = self._validate_initial_layout()
        assets.add_static_collisions(builder, self.bunny, hulls)
        self.static_hull_count = len(hulls)

    def _validate_initial_layout(self):
        """AABB separation is conservative for these spheres and aligned boxes."""
        water = np.array([self.config.block_start, self.config.block_end])
        obstacles = [("initial_water", water), ("static_bunny", self.bunny.bounds)]
        records = []
        for spec in assets.RIGIDS:
            radius = spec.size / 2 if spec.kind == "box" else spec.size
            bounds = np.array([np.array(spec.center) - radius, np.array(spec.center) + radius])
            if np.any(bounds[0] <= 0) or np.any(bounds[1] >= assets.POOL_SIZE):
                raise ValueError(f"{spec.name} intersects the container")
            for name, other in obstacles:
                separation = max(float(np.max(bounds[0] - other[1])),
                                 float(np.max(other[0] - bounds[1])))
                if separation <= 0:
                    raise ValueError(f"Initial {spec.name} overlaps {name}")
                records.append(dict(body=spec.name, obstacle=name, aabb_gap=separation))
            obstacles.append((spec.name, bounds))
        return records

    def add_particles(self, builder):
        n = self.particles.n
        mass = .8 * self.particles.particle_spacing**3 * self.config.rest_density
        builder.add_particles(pos=[(0., 0., 0.)] * n,
                              vel=[self.config.initial_velocity] * n,
                              mass=[mass] * n, radius=[self.config.particle_radius] * n)

    def bind_model(self, model, body_ids):
        self.validate_model(model)
        self.particles.model = model
        centers = model.body_com.numpy()
        masses = model.body_mass.numpy()
        paths, mapped_ids, entries = [], [], []
        selected = set(body_ids)
        known = {i for binding in self._bindings for i in binding.body_ids}
        if len(selected) != len(body_ids) or selected - known:
            self.validation.add("error", "binding.bodies", "scene", "Duplicate or unknown runtime body IDs")
        for binding in self._bindings:
            if selected.intersection(binding.body_ids) and not set(binding.body_ids) <= selected:
                self.validation.add("error", "binding.partial", binding.asset.name, "Select an entire asset; exclude links explicitly")
        self.validation.require_valid()
        for obj_id, binding in enumerate(self._bindings):
            if not selected.intersection(binding.body_ids):
                continue
            prepared = prepare_articulated_asset(binding.asset, store=self.store, device=self.device,
                bake_options=BakeOptions(voxel_size=.005, max_support_radius=self.particles.smoothing_length),
                support_radius=self.particles.smoothing_length,
                excluded_links=self._excluded_links.get(obj_id, ()))
            self.validation.extend(prepared.report)
            for local_id, path in prepared.boundaries.items():
                body_id = binding.body_ids[local_id]
                mesh = prepared.meshes[local_id]
                self.body_meshes[body_id] = mesh
                mapped_ids.append(body_id)
                paths.append(path)
                entries.append(dict(body_id=body_id, label=model.body_label[body_id],
                                    asset=str(path), local_transform=[0, 0, 0, 0, 0, 0, 1],
                                    center_of_mass=centers[body_id].tolist(), mass=float(masses[body_id]),
                                    bounds=mesh.bounds.tolist()))
        self.boundaries = boundary_model.load_boundaries(
            self.static_path, paths,
            boundary_model.BoundaryLoadOptions(self.particles.smoothing_length, self.device))
        self.boundaries.bind_bodies(model, mapped_ids)
        probes = {"cavity": [.5, .5, 1.], "bottom": [1., 1., -.05],
                  "left_wall": [-.05, 1., 1.], "right_wall": [3.05, 1., 1.],
                  "front_wall": [1., -.05, 1.], "back_wall": [1., 2.05, 1.],
                  "ceiling": [1., 1., 5.05], "bunny": self.bunny.center_mass.tolist()}
        points = wp.array(list(probes.values()), dtype=wp.vec3, device=self.device)
        distances = wp.empty(len(probes), dtype=float, device=self.device)
        wp.launch(sample_sdf, len(probes), inputs=[self.boundaries.assets[0].sdf.id, points],
                  outputs=[distances], device=self.device)
        values = distances.numpy()
        if not np.isfinite(values).all() or not (values[0] > 0 and np.all(values[1:] < 0)):
            raise ValueError(f"Static boundary sign validation failed: {values}")
        self.manifest = dict(world_up_axis="Z", particle_count=self.particles.n,
                             static_asset=str(self.static_path), bunny_body_id=None,
                             static_collision_hulls=self.static_hull_count,
                             static_sign_distances=dict(zip(probes, values.tolist())), dynamic_boundaries=entries,
                             initial_clearances=self.initial_clearances,
                             particle_radius=self.config.particle_radius,
                             support_radius=self.particles.smoothing_length,
                             control_dt=self.control_dt, physics_dt=self.physics_dt,
                             fluid_frame_dt=self.config.frame_dt, fluid_time_scale=self.fluid_time_scale,
                             fluid_dt=self.config.frame_dt / self.config.substeps,
                             fluid_substeps=self.config.substeps,
                             pressure_iterations=self.config.pressure_iterations)
        self.manifest["asset_validation"] = self.validation.to_dict()
        destination = self.store.path("scene_manifest.json")
        with self.store.staging(destination) as stage:
            source = stage / "scene_manifest.json"
            source.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
            self.store.publish([(source, destination)])
        return self.boundaries

    def initialize_particles(self, *states):
        with wp.ScopedDevice(self.device):
            pbf_helper.init_particle_state(self.particles)
            for state in states:
                wp.copy(state.particle_q, self.particles.model.particle_q)
                wp.copy(state.particle_qd, self.particles.model.particle_qd)

    def character_translation(self, state, body_ids, root_position):
        poses = state.body_q.numpy()
        points = []
        for body_id in body_ids:
            if body_id not in self.body_meshes:
                continue
            pose = poses[body_id]
            matrix = trimesh.transformations.quaternion_matrix([pose[6], pose[3], pose[4], pose[5]])
            matrix[:3, 3] = pose[:3]
            points.append(trimesh.transform_points(self.body_meshes[body_id].vertices, matrix))
        points = np.concatenate(points)
        low, high = points.min(axis=0), points.max(axis=0)
        desired = np.array([.8, 1.]) - np.asarray(root_position)[:2]
        minimum = .05 - low[:2]
        maximum = assets.POOL_SIZE[:2] - .05 - high[:2]
        if np.any(minimum > maximum):
            raise ValueError("Sampled humanoid pose cannot fit inside the play pool")
        shift = np.array([*np.clip(desired, minimum, maximum), self.config.block_end[2] + .1 - low[2]])
        if high[2] + shift[2] > assets.POOL_SIZE[2] - .05:
            raise ValueError("Sampled humanoid pose cannot fit above the initial fluid block")
        return shift.astype(np.float32)

    def draw_pool(self, viewer):
        if not hasattr(self, "_line_starts"):
            corners = np.array([[x, y, z] for x in (0., 3.) for y in (0., 2.) for z in (0., 5.)])
            edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                     if np.count_nonzero(corners[a] != corners[b]) == 1]
            self._line_starts = wp.array([corners[a] for a, _ in edges], dtype=wp.vec3, device=self.device)
            self._line_ends = wp.array([corners[b] for _, b in edges], dtype=wp.vec3, device=self.device)
            self._line_colors = wp.array([[.35, .55, .65]] * len(edges), dtype=wp.vec3, device=self.device)
        viewer.log_lines("pbf_pool_outline", starts=self._line_starts, ends=self._line_ends,
                         colors=self._line_colors, width=.008)
