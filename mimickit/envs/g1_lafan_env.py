"""Opt-in canonical LaFAN interface; actor and physics remain native G1."""
import gymnasium.spaces as spaces
import numpy as np

from envs.g1_lafan_adapter import G1LafanAdapter


class G1LafanEnvMixin:
    """Share canonical prior observations and GSI across native G1 SMP tasks."""
    def __init__(self, env_config, *args, **kwargs):
        required = {"global_obs": False, "root_height_obs": True,
                    "disc_dof_vel_obs": False, "num_disc_obs_steps": 10,
                    "enable_tar_obs": False, "pose_termination": False}
        if any(env_config.get(k) != v for k, v in required.items()):
            raise ValueError("G1 LaFAN requires the original local 10-frame pose representation")
        self._gsi_buffer = None
        super().__init__(env_config, *args, **kwargs)
        print("LaFAN adapter: native G1 actor/physics, canonical 10x114 prior observations. "
              "Motion files are construction bootstrap only; the training recipe enables LaFAN GSI.")

    def _build_kin_char_model(self, char_file):
        super()._build_kin_char_model(char_file)
        self._lafan_adapter = G1LafanAdapter(self._kin_char_model, self._device, char_file)

    def get_disc_obs_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(1140,), dtype=np.float32)

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        p, r, v, w, j, _, _ = self._fetch_disc_demo_data(motion_ids, motion_times0)
        return self._lafan_adapter.observations(p, r, v, w, j, self._engine.get_timestep())

    def _update_disc_obs(self, env_ids=None):
        histories = [self._disc_hist_root_pos, self._disc_hist_root_rot,
                     self._disc_hist_root_vel, self._disc_hist_root_ang_vel,
                     self._disc_hist_joint_rot]
        values = [h.get_all() for h in histories]
        if env_ids is not None:
            values = [v[env_ids] for v in values]
        obs = self._lafan_adapter.observations(*values, self._engine.get_timestep())
        if env_ids is None:
            self._disc_obs_buf[:] = obs
        else:
            self._disc_obs_buf[env_ids] = obs

    def _disc_obs_to_motion_frames(self, sample):
        return self._lafan_adapter.from_samples(sample)
