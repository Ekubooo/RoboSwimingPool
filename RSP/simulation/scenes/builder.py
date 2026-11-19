"""Declare objects and assemble the model without application state views."""

import newton
import numpy as np

from .build import PBFScene


class ObjCfg():
    def __init__(self, asset_file, fix_root, is_visual, enable_self_collisions):
        self.asset_file = asset_file
        self.fix_root = fix_root
        self.is_visual = is_visual
        self.enable_self_collisions = enable_self_collisions
        return

    def __hash__(self):
        values = (self.asset_file, self.fix_root, self.is_visual, self.enable_self_collisions)
        obj_hash = hash(values)
        return obj_hash

    def __eq__(self, other):
        same = self.asset_file == other.asset_file
        same &= self.fix_root == other.fix_root
        same &= self.is_visual == other.is_visual
        same &= self.enable_self_collisions == other.enable_self_collisions
        return same


class SceneBuilder:
    """Declare and assemble objects; delegate asset/boundary work to PBFScene."""

    def __init__(self, config, num_envs, device, timestep, substeps, *, scene_type=PBFScene, log=None, require_dfs=False):
        self.device = device
        self.play_scene = scene_type(
            config.get("pbf_asset_dir"), device, timestep, substeps,
            scene_spec=config.get("scene_spec"), scene_assets=config.get("scene_assets"),
            settings=config.get("fluid_settings"),
        )
        self._log = log
        self._require_dfs = require_dfs
        self.num_envs = num_envs
        self.builder = self._create_model_builder()
        self._builder_cache = {}
        self.object_types = []
        self._obj_builders = []
        self._obj_names = []
        self._obj_start_pos = []
        self._obj_start_rot = []
        self.colors = []
        self.visual = []


    def finalize(self):
        self.validate_envs()
        self.play_scene.add_static(self.builder)
        self.build_objects(self.play_scene.add_particles)
        # Fluid contacts and pressure feedback belong exclusively to PBF.
        for i in range(len(self.builder.shape_flags)):
            self.builder.shape_flags[i] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
        if getattr(self, "_log", None) is not None:
            self._log("Initializing simulation...")
        from RSP.sim_ready.adapters.newton import finalize_model
        model = finalize_model(self.builder, device=self.device, report=self.play_scene.validation,
                               require_dfs=getattr(self, "_require_dfs", False))
        self.play_scene.validate_model(model)
        bindings = iter(self.play_scene.bindings)
        self.object_body_ids = tuple(
            tuple(next(bindings).body_ids for _ in objects)
            for objects in self._obj_builders
        )
        if next(bindings, None) is not None:
            raise ValueError("The model contains undeclared articulations")
        return model

    def coupled_body_ids(self):
        return tuple(body_id for ids, visual in zip(self.object_body_ids[0], self.visual[0])
                     if not visual for body_id in ids)

    def start_transforms(self):
        for env_id, (positions, rotations) in enumerate(zip(self._obj_start_pos, self._obj_start_rot)):
            for obj_id, (position, rotation) in enumerate(zip(positions, rotations)):
                yield env_id, obj_id, position, rotation


    def create_env(self):
        env_id = len(self._obj_builders)
        assert(env_id < self.num_envs)

        self._obj_builders.append([])
        self._obj_names.append([])
        self.object_types.append([])
        self._obj_start_pos.append([])
        self._obj_start_rot.append([])
        self.colors.append([])
        self.visual.append([])

        return env_id


    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True,
                   fix_root=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        if (start_rot is None):
            start_rot = np.array([0.0, 0.0, 0.0, 1.0])

        if (start_pos is None):
            start_pos = np.array([0.0, 0.0, 0.0])

        obj_builder = self._create_obj_builder(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual,
                                               enable_self_collisions=enable_self_collisions)

        obj_id = len(self._obj_builders[env_id])
        self.object_types[env_id].append(obj_type)
        self._obj_builders[env_id].append(obj_builder)
        self._obj_names[env_id].append(name)
        self._obj_start_pos[env_id].append(start_pos)
        self._obj_start_rot[env_id].append(start_rot)
        self.colors[env_id].append(color)
        self.visual[env_id].append(is_visual)

        return obj_id


    def get_obj_type(self, obj_id):
        obj_type = self.object_types[0][obj_id]
        return obj_type


    def validate_envs(self):
        num_envs = self.num_envs
        objs_per_env = len(self._obj_builders[0])

        for i in range(0, num_envs):
            curr_objs = self._obj_builders[i]
            num_objs = len(curr_objs)
            assert(num_objs == objs_per_env), "All envs must have the same number of objects."
        return


    def build_objects(self, add_particles):
        num_envs = self.num_envs
        objs_per_env = len(self._obj_builders[0])
        total_objs = num_envs * objs_per_env

        for env_id in range(num_envs):
            self.builder.begin_world()

            for obj_id in range(objs_per_env):
                obj_builder = self._obj_builders[env_id][obj_id]
                self.builder.add_builder(obj_builder, label_prefix=self._obj_names[env_id][obj_id])

                num_created = env_id * objs_per_env + obj_id + 1
                if self._log is not None:
                    self._log("Building {:d}/{:d} objs".format(num_created, total_objs), end='\r')

            add_particles(self.builder)
            self.builder.end_world()

        if self._log is not None:
            self._log("")
        return


    def _create_obj_builder(self, asset_file, fix_root, is_visual, enable_self_collisions):
        obj_cfg = ObjCfg(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual,
                         enable_self_collisions=enable_self_collisions)

        if (obj_cfg in self._builder_cache):
            obj_builder = self._builder_cache[obj_cfg]
        else:
            obj_builder = self._create_model_builder()

            from RSP.sim_ready.adapters.newton import import_physical_asset
            import_physical_asset(obj_builder, asset_file, fix_root=fix_root,
                                  enable_self_collisions=enable_self_collisions,
                                  report=self.play_scene.validation)
            if obj_builder.articulation_count != 1:
                self.play_scene.validation.add("unsupported", "consumer.articulation_count", asset_file,
                    f"Expected one articulation per object; imported {obj_builder.articulation_count}")
                self.play_scene.validation.require_valid()

            if (is_visual):
                for i in range(len(obj_builder.shape_flags)):
                    obj_builder.shape_flags[i] &= ~newton.ShapeFlags.COLLIDE_SHAPES

            self._builder_cache[obj_cfg] = obj_builder

        return obj_builder


    def _create_model_builder(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 1.0
        return builder


    def build_ground(self):
        shape_cfg = self.builder.ShapeConfig(mu=1.0, restitution=0)
        self.builder.add_ground_plane(cfg=shape_cfg)
        return
