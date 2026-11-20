"""Continuous SMP policy playback in the single-world fluid scene."""
from contextlib import ExitStack
import logging
import math
import numpy as np
import torch

from engines.engine import ObjType
from envs.smp_env import SMPEnv
from envs.sim_env import SimEnv
from envs.task_dodgeball_env import compute_dodgeball_observations
from RSP.sim_ready.lib.asset_validation import SkeletonDescription
from RSP.physics import fluid_clock
from util.camera import Camera, CameraMode


class PBFPlayEnv(SMPEnv):
    continuous_playback = True

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False, *,
                 asset_progress=None):
        if engine_config["engine_name"] != "newton_pbf" or num_envs != 1:
            raise ValueError("pbf_play requires one newton_pbf environment")
        from RSP.sim_ready.lib.storage.runtime_slot import runtime_asset_lease
        from RSP.sim_ready.paths import SCENE_ROOT, configured_runtime_root
        from RSP.sim_ready.pipelines.runtime import AssetPreparationError, ensure_runtime_assets
        from RSP.simulation.scenes.fluid import boundary_support_radius
        from RSP.simulation.scenes.spec import load_scene
        from RSP.simulation.scenes.resolve import resolve_scene

        scene = load_scene(env_config.get("scene_preset", SCENE_ROOT / "pbf_play.yaml"))
        env_config, engine_config = dict(env_config), dict(engine_config)
        settings = engine_config.get("fluid_settings")
        if settings is None:
            settings = fluid_clock.DEFAULT_SETTINGS
        engine_config["fluid_settings"] = settings
        self._policy_hz = float(env_config.get("playback_policy_hz", 30))
        if not math.isfinite(self._policy_hz) or self._policy_hz <= 0:
            raise ValueError("Playback policy frequency must be finite and positive")

        try:
            asset_root = configured_runtime_root(engine_config.get("pbf_asset_dir"))
        except (OSError, ValueError) as error:
            raise AssetPreparationError(f"Cannot access configured runtime assets: {error}") from error
        support_radius = boundary_support_radius(settings)
        self._startup_resources = None
        with ExitStack() as startup:
            try:
                asset_root = startup.enter_context(runtime_asset_lease(asset_root))
            except (OSError, ValueError) as error:
                raise AssetPreparationError(f"Cannot access runtime assets at {asset_root}: {error}") from error

            assets = ensure_runtime_assets(
                scene.asset_refs, support_radius=support_radius, device=device,
                root=asset_root, progress=asset_progress, terrain_prop=scene.terrain_prop,
            )
            try:
                resolved = resolve_scene(scene, asset_root, support_radius=support_radius, assets=assets)
            except ValueError as error:
                raise AssetPreparationError(f"Scene placement is invalid: {error}") from error
            if asset_progress is not None:
                asset_progress("Scene assets are ready. Loading the simulation...")

            engine_config.update(pbf_asset_dir=asset_root, scene_assets=resolved, scene_spec=scene)
            env_config.update(char_file=str(resolved.dynamic[scene.character].model_path),
                              play_spawn_xy=scene.spawn_xy, policy_use_scene_sphere=scene.policy_use_scene_sphere)
            self.policy_use_scene_sphere = bool(env_config["policy_use_scene_sphere"])
            self._spawn_xy = scene.spawn_xy
            self._scene_spec = scene
            self._instance_ids = {}
            self._policy_sphere_id = None
            self._scene_obj_ids = []
            self._pending_reset = False
            self._pending_sphere = None
            self._play_observations = None
            # Needed even during base construction, when observation size is queried.
            self._inactive_sphere_pos = torch.tensor([[[10., 10., .1]]], device=device)
            self._inactive_sphere_vel = torch.zeros_like(self._inactive_sphere_pos)
            # Register cleanup before construction so the engine closes before
            # releasing the asset lease on both partial and successful startup.
            startup.callback(super().close)
            super().__init__(env_config, engine_config, num_envs, device, visualize, record_video)
            self._play_observations = self._create_playback_observations()
            self._startup_resources = startup.pop_all()

    def close(self):
        resources = getattr(self, "_startup_resources", None)
        if resources is not None:
            self._startup_resources = None
            resources.close()

    def _create_playback_observations(self):
        """Bind playback inputs after the environment owns all persistent buffers."""
        from .observations import CharacterStateViews, ObservationBuilder

        if self._enable_phase_obs or self._enable_tar_obs:
            raise ValueError("PBF playback observations require phase/target observations disabled")
        engine, kin = self._engine, self._kin_char_model
        char, sphere = self._get_char_id(), self._policy_sphere_id
        state = CharacterStateViews(
            root_pos=engine.get_root_pos(char), root_rot=engine.get_root_rot(char),
            root_vel=engine.get_root_vel(char), root_ang_vel=engine.get_root_ang_vel(char),
            dof_pos=engine.get_dof_pos(char), dof_vel=engine.get_dof_vel(char),
            body_pos=engine.get_body_pos(char))
        return ObservationBuilder(
            state_views=state, joints=[kin.get_joint(j) for j in range(1, kin.get_num_joints())],
            key_body_ids=self._key_body_ids, output=self._obs_buf,
            inactive_sphere_state=(self._inactive_sphere_pos, self._inactive_sphere_vel),
            scene_sphere_state=((self._inactive_sphere_pos, self._inactive_sphere_vel) if sphere is None else
                                (engine.get_root_pos(sphere).unsqueeze(1), engine.get_root_vel(sphere).unsqueeze(1))),
            global_obs=self._global_obs, root_height_obs=self._root_height_obs,
            use_scene_sphere=self.policy_use_scene_sphere)

    def get_policy_timestep(self):
        return 1.0 / getattr(self, "_policy_hz", 30)

    def _build_data_buffers(self):
        super()._build_data_buffers()
        # SMP's reset/GSI support still owns its history buffers. Playback does
        # not publish a stale training feature as if it were current state.
        self._info.pop("disc_obs", None)

    def _update_misc(self):
        """No training history or reference-character work during playback."""

    def _update_observations(self, env_ids=None):
        if self._play_observations is None:
            # Construction queries observation dimensions before buffers exist.
            SimEnv._update_observations(self, env_ids)
        else:
            self._play_observations.update(env_ids, use_scene_sphere=self.policy_use_scene_sphere)

    def _update_info(self, env_ids=None):
        """Playback has no tracking-error or reward diagnostics."""

    def _post_physics_step(self):
        self._update_time()
        self._update_observations()
        # Reward/done buffers remain zero from initialization/reset. Keeping
        # their return slots preserves the ordinary environment step interface.

    def _build_env(self, env_id, env_config):
        super()._build_env(env_id, env_config)
        for spec, path in zip(self._scene_spec.instances, self._engine.get_play_scene().primitive_files):
            obj_id = self._engine.create_obj(
                env_id, ObjType.rigid, path, spec.name, start_pos=np.array(spec.position), start_rot=np.array(spec.rotation),
                color=np.array(spec.color))
            self._scene_obj_ids.append(obj_id)
            self._instance_ids[spec.name] = obj_id
            self._engine.get_play_scene().register_instance(obj_id, spec.asset)
            if spec.role == "policy_sphere":
                self._policy_sphere_id = obj_id

    def _get_char_color(self):
        return np.array([1.0, 0.0, 0.0])

    def _build_character(self, env_id, env_config, color=None):
        obj_id = super()._build_character(env_id, env_config, color)
        kin = self._kin_char_model
        n = kin.get_num_joints()
        kinds = {"HINGE": "REVOLUTE", "SPHERICAL": "BALL", "FIXED": "FIXED"}
        mode = self._engine.get_control_mode().name
        skeleton = SkeletonDescription(
            tuple(kin.get_body_names()), tuple(int(kin.get_parent_id(i)) for i in range(n)),
            ("ROOT",) + tuple(kinds.get(kin.get_joint(i).joint_type.name, kin.get_joint(i).joint_type.name)
                              for i in range(1, n)),
            (0,) + tuple(kin.get_joint_dof_idx(i) for i in range(1, n)),
            (0,) + tuple(kin.get_joint_dof_dim(i) for i in range(1, n)),
            tuple(range(kin.get_dof_size())) if mode != "none" else (), mode,
            self._clip_position_targets_to_physical_limits)
        self._engine.get_play_scene().register_character(
            obj_id, skeleton, env_config.get("pbf_excluded_links", ()))
        return obj_id

    def set_policy_use_scene_sphere(self, enabled):
        if enabled and self._policy_sphere_id is None:
            logging.warning("No policy_sphere instance; scene sphere observations remain disabled")
            enabled = False
        self.policy_use_scene_sphere = bool(enabled)
        self._update_observations()

    def get_policy_use_scene_sphere(self):
        return self.policy_use_scene_sphere

    def request_policy_use_scene_sphere(self, enabled):
        """Queue an observation option for the next playback frame boundary."""
        self._pending_sphere = bool(enabled)

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        if self.policy_use_scene_sphere:
            sphere_id = self._policy_sphere_id
            pos = self._engine.get_root_pos(sphere_id).unsqueeze(1)
            vel = self._engine.get_root_vel(sphere_id).unsqueeze(1)
        else:
            # The original task's inactive projectile input, without an entity.
            pos = self._inactive_sphere_pos
            vel = self._inactive_sphere_vel
        if env_ids is not None:
            root_pos, root_rot, pos, vel = (value[env_ids] for value in (root_pos, root_rot, pos, vel))
        task_obs = compute_dodgeball_observations(root_pos, root_rot, pos, vel)
        return torch.cat([obs, task_obs], dim=-1)

    def _update_reward(self):
        self._reward_buf.zero_()

    def _update_done(self):
        # Falling, contacts, time and motion-library boundaries never reset play.
        self._done_buf.zero_()

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) == 0:
            return
        for spec in self._scene_spec.instances:
            obj_id = self._instance_ids[spec.name]
            self._engine.set_root_pos(env_ids, obj_id, torch.tensor(spec.position, device=self._device))
            self._engine.set_root_rot(env_ids, obj_id, torch.tensor(spec.rotation, device=self._device))
            self._engine.set_root_vel(env_ids, obj_id, torch.tensor(spec.linear_velocity, device=self._device))
            self._engine.set_root_ang_vel(env_ids, obj_id, torch.tensor(spec.angular_velocity, device=self._device))
        # Preserve MimicKit's sampled pose, root height and velocity. Only move
        # its horizontal origin to the left clearing; no lift above the water.
        char_id = self._get_char_id()
        root = self._engine.get_root_pos(char_id)[env_ids].clone()
        root[:, :2] = root.new_tensor(self._spawn_xy)
        self._engine.set_root_pos(env_ids, char_id, root)
        self._engine.reset_play_scene()

    def _build_camera(self, env_config):
        self._camera = Camera(CameraMode.still, self._engine,
                              pos=np.array([5., -5., 4.1]), target=np.array([1.5, 1., 1.5]))

    def _setup_gui(self):
        """Playback bindings own UI setup; suppress SimEnv's keyboard hooks."""

    def request_reset(self):
        self._pending_reset = True

    def consume_reset_request(self) -> bool:
        """Consume coalesced requests; the runner owns full playback reset."""
        reset = self._pending_reset
        self._pending_reset = False
        return reset

    def apply_playback_commands(self):
        """Apply observation options after reset, before the next decision."""
        if self._pending_sphere is not None:
            enabled, self._pending_sphere = self._pending_sphere, None
            self.set_policy_use_scene_sphere(enabled)

    def render_only(self):
        if self._visualize:
            self._update_camera()
            self._render_scene()

    def _render(self):
        # The runner owns stepping; a paused redraw must never block env.step.
        self.render_only()
