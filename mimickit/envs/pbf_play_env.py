"""Continuous SMP policy playback in the single-world fluid scene."""
import numpy as np
import torch

from engines.engine import ObjType
from envs.smp_env import SMPEnv
from envs.task_dodgeball_env import compute_dodgeball_observations
from sim_ready.pipelines.pbf_play import RIGIDS
from util.camera import Camera, CameraMode


class PBFPlayEnv(SMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        if engine_config["engine_name"] != "newton_pbf" or num_envs != 1:
            raise ValueError("pbf_play requires one newton_pbf environment")
        self.policy_use_scene_sphere = bool(env_config.get("policy_use_scene_sphere", False))
        self._scene_obj_ids = []
        super().__init__(env_config, engine_config, num_envs, device, visualize, record_video)

    def _build_env(self, env_id, env_config):
        super()._build_env(env_id, env_config)
        for spec, path in zip(RIGIDS, self._engine.get_play_scene().primitive_files):
            obj_id = self._engine.create_obj(
                env_id, ObjType.rigid, path, spec.name, start_pos=np.array(spec.center),
                color=np.array(spec.color))
            self._scene_obj_ids.append(obj_id)

    def set_policy_use_scene_sphere(self, enabled):
        self.policy_use_scene_sphere = bool(enabled)
        self._update_observations()

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        if self.policy_use_scene_sphere:
            sphere_id = self._scene_obj_ids[-1]
            pos = self._engine.get_root_pos(sphere_id).unsqueeze(1)
            vel = self._engine.get_root_vel(sphere_id).unsqueeze(1)
        else:
            # The original task's inactive projectile input, without an entity.
            pos = torch.tensor([[[10., 10., .1]]], device=self._device)
            vel = torch.zeros_like(pos)
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
        for obj_id, spec in zip(self._scene_obj_ids, RIGIDS):
            self._engine.set_root_pos(env_ids, obj_id, torch.tensor(spec.center, device=self._device))
            self._engine.set_root_rot(env_ids, obj_id, torch.tensor([0., 0., 0., 1.], device=self._device))
            self._engine.set_root_vel(env_ids, obj_id, 0.)
            self._engine.set_root_ang_vel(env_ids, obj_id, 0.)
        self._engine.place_character(self._get_char_id())
        self._engine.reset_play_scene()

    def _build_camera(self, env_config):
        self._camera = Camera(CameraMode.still, self._engine,
                              pos=np.array([5., -5., 4.1]), target=np.array([1.5, 1., 1.5]))

    def _setup_gui(self):
        super()._setup_gui()
        self._engine.register_keyboard_callback("R", self.reset)
        self._engine.get_viewer().set_reset_callback(self.reset)
        self._engine.register_ui_callback(self._draw_play_controls)

    def _draw_play_controls(self, imgui):
        imgui.text("PBF play / SMP policy")
        changed, enabled = imgui.checkbox("Policy: scene sphere", self.policy_use_scene_sphere)
        if changed:
            self.set_policy_use_scene_sphere(enabled)
        if imgui.button("Reset scene (R)"):
            self.reset()
        imgui.text("No automatic reset.")
        imgui.text("Enter: pause; Space: step.")
