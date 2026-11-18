"""Scene assembly and lifecycle glue for the copied Newton-PBF backend."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import trimesh
import warp as wp

from RSP.sim_ready.pipelines.scenes import pbf_play as assets
from pathlib import Path
from uuid import uuid4
from RSP.sim_ready.paths import PUBLISHED_ROOT, PROJECT_ROOT
from RSP.sim_ready.catalog import physical_signature
from RSP.sim_ready.pipelines.scenes.assembly import resolve_scene
from RSP.sim_ready.pipelines.scenes.container_recipe import add_collisions
from RSP.sim_ready.pipelines.scenes.spec import load_scene
from RSP.sim_ready.lib.storage.hull_io import load_hulls
from RSP.sim_ready.lib.asset_validation import (ValidationReport, diagnostic_scope,
    validate_physical_asset, validate_skeleton)
from RSP.sim_ready.adapters.newton import describe_articulations, validate_model_for_mimickit
from solvers.pbf import boundary_model, pbf_helper
from RSP.runtime import fluid_clock


@wp.kernel
def sample_sdf(volume: wp.uint64, points: wp.array(dtype=wp.vec3), distances: wp.array(dtype=float)):
    i = wp.tid()
    index = wp.volume_world_to_index(volume, points[i])
    distances[i] = wp.volume_sample_f(volume, index, wp.Volume.LINEAR)


def runtime_config(control_dt, substeps, settings=None):
    """Keep physical scalars; express the authored Y-up scene in the Z-up world."""
    settings = fluid_clock.DEFAULT_SETTINGS if settings is None else settings
    original = settings.integrator_config(settings.schedule(control_dt, substeps))
    transform = assets.SCENE_TRANSFORM
    if not np.allclose(np.abs(transform[:3, :3]) @ original.container_size, assets.POOL_SIZE):
        raise ValueError("Fluid container_size must match the existing pool assets; asset geometry is unchanged")
    endpoints = np.array([original.block_start, original.block_end]) @ transform[:3, :3].T + transform[:3, 3]
    return replace(original,
                   gravity=tuple(transform[:3, :3] @ original.gravity),
                   container_size=tuple(np.abs(transform[:3, :3]) @ original.container_size),
                   block_start=tuple(endpoints.min(axis=0)), block_end=tuple(endpoints.max(axis=0)),
                   initial_velocity=tuple(transform[:3, :3] @ original.initial_velocity))


class PBFScene:
    def __init__(self, directory, device, control_dt, substeps, *, scene_spec=None, scene_assets=None):
        self.directory = Path(directory or PUBLISHED_ROOT).resolve()
        spec = load_scene() if scene_spec is None else scene_spec
        self.spec = spec
        resolved = resolve_scene(spec, self.directory) if scene_assets is None else scene_assets
        self.static_asset = resolved.static
        self.resolved_assets = resolved.dynamic
        self.layout = resolved.layout
        self.container_size = np.asarray(self.layout["size"])
        self.initial_clearances = resolved.clearances
        self.dynamic_assets = tuple(self.resolved_assets[ref] for ref in (spec.character, *spec.props))
        self.report_directory = PROJECT_ROOT / "output" / "pbf_play" / "runs" / uuid4().hex
        self.device = device
        self.control_dt = control_dt
        self.rigid_dt = control_dt / substeps
        self.fluid_settings = fluid_clock.DEFAULT_SETTINGS
        self.fluid_schedule = self.fluid_settings.schedule(control_dt, substeps)
        self.config = replace(runtime_config(control_dt, substeps, self.fluid_settings),
                              container_size=tuple(self.container_size),
                              block_start=spec.fluid_start, block_end=spec.fluid_end)
        self.particles = SimpleNamespace(config=self.config)
        pbf_helper.init_scene_parameters(self.particles)
        self.primitive_files = [str(asset.file(asset.record["model"])) for asset in self.dynamic_assets[1:]]
        self.body_meshes = {}
        self.boundaries = None
        self.validation = ValidationReport()
        self._skeletons = {}
        self._object_assets = {}
        self._excluded_links = {}
        self._validated_model = None

    def register_instance(self, obj_id, asset_ref):
        if obj_id in self._object_assets:
            raise ValueError(f"Duplicate runtime object registration: {obj_id}")
        self._object_assets[obj_id] = self.resolved_assets[asset_ref]

    def register_character(self, obj_id, skeleton, excluded_links=()):
        self.register_instance(obj_id, self.spec.character)
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
        record = self.static_asset.record
        self.static_path = self.static_asset.file(record["boundary"])
        self.bunny = trimesh.load(self.static_asset.file(record["bunny"]), force="mesh", process=False)
        hulls = load_hulls(self.static_asset.file(record["hulls"]))
        add_collisions(builder, self.layout, self.bunny, hulls)
        self.static_hull_count = len(hulls)

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
            asset = self._object_assets[obj_id]
            if physical_signature(binding.asset) != asset.record["physical_signature"]:
                raise ValueError(f"Published physical model differs from runtime object {obj_id}")
            links = asset.record["links"]
            if set(links) != {body.name for body in binding.asset.bodies}:
                raise ValueError("Published boundary link coverage differs from runtime model")
            if self._excluded_links.get(obj_id):
                raise ValueError("Reference published assets do not support excluded links")
            for local_id, body in enumerate(binding.asset.bodies):
                path = asset.file(links[body.name])
                boundary_record = json.loads(path.read_text(encoding="utf-8"))
                mesh_path = (path.parent / boundary_record["files"]["mesh"]).resolve()
                mesh_path = asset.file(mesh_path.relative_to(asset.root).as_posix())
                mesh = trimesh.load(mesh_path, force="mesh", process=False)
                body_id = binding.body_ids[local_id]
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
        probes = self.layout["probes"]
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
                             control_dt=self.control_dt, physics_dt=self.rigid_dt,
                             fluid_frame_dt=self.fluid_schedule.fluid_frame_dt,
                             fluid_dt=self.fluid_schedule.fluid_dt,
                             fluid_substeps=self.fluid_schedule.fluid_steps,
                             fluid_time_scale=self.fluid_settings.time_scale)
        self.manifest["asset_validation"] = self.validation.to_dict()
        self.report_directory.mkdir(parents=True, exist_ok=True)
        (self.report_directory / "scene_manifest.json").write_text(
            json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
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
        maximum = self.container_size[:2] - .05 - high[:2]
        if np.any(minimum > maximum):
            raise ValueError("Sampled humanoid pose cannot fit inside the play pool")
        shift = np.array([*np.clip(desired, minimum, maximum), self.config.block_end[2] + .1 - low[2]])
        if high[2] + shift[2] > self.container_size[2] - .05:
            raise ValueError("Sampled humanoid pose cannot fit above the initial fluid block")
        return shift.astype(np.float32)

    def draw_pool(self, viewer):
        if not hasattr(self, "_line_starts"):
            corners = np.array([[x, y, z] for x in (0., self.container_size[0])
                                for y in (0., self.container_size[1]) for z in (0., self.container_size[2])])
            edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                     if np.count_nonzero(corners[a] != corners[b]) == 1]
            self._line_starts = wp.array([corners[a] for a, _ in edges], dtype=wp.vec3, device=self.device)
            self._line_ends = wp.array([corners[b] for _, b in edges], dtype=wp.vec3, device=self.device)
            self._line_colors = wp.array([[.35, .55, .65]] * len(edges), dtype=wp.vec3, device=self.device)
        viewer.log_lines("pbf_pool_outline", starts=self._line_starts, ends=self._line_ends,
                         colors=self._line_colors, width=.008)
