"""Observation/actor equivalence without loading a solver or opening a window."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]
wp.config.kernel_cache_dir = str(ROOT / "output/newton_pbf_refactor_checks/warp_cache")
tempfile.tempdir = str(ROOT / "output/newton_pbf_refactor_checks/tmp")
Path(tempfile.tempdir).mkdir(parents=True, exist_ok=True)

from anim.mjcf_char_model import MJCFCharModel
from envs.char_env import compute_char_obs
from envs.task_dodgeball_env import compute_dodgeball_observations
from learning.base_agent import AgentMode
from learning.ppo_agent import PPOAgent
from learning.normalizer import Normalizer
from learning.distribution_gaussian_diag import DistributionGaussianDiagBuilder, StdType
from RSP.runtime.inference import PolicyInference
from RSP.runtime.observations import ObservationBuilder


@unittest.skipUnless(torch.cuda.is_available(), "CUDA observation path")
class InferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_observations_match_joint_and_heading_edge_cases(self):
        torch.manual_seed(1234)
        device, count = "cuda:0", 9
        kin = MJCFCharModel(device)
        kin.load(str(ROOT / "data/assets/humanoid/humanoid.xml"))
        joints, dofs = kin.get_num_joints() - 1, kin.get_dof_size()
        poses = torch.randn(count, 7, device=device)
        poses[:, 3:] = torch.nn.functional.normalize(poses[:, 3:], dim=-1)
        positions = torch.randn(count, dofs + 6, device=device)[:, 6:]
        for j in range(1, joints + 1):
            joint = kin.get_joint(j)
            if joint.get_dof_dim() == 3:
                start = joint.dof_idx
                positions[:5, start:start + 3] = 0.
                positions[:5, start] = torch.tensor([0., 1e-7, 1e-5, torch.pi - 1e-6, torch.pi + 1e-6], device=device)
        state = dict(root_pos=poses[:, :3], root_rot=poses[:, 3:],
                     root_vel=torch.randn(count, 3, device=device),
                     root_ang_vel=torch.randn(count, 3, device=device),
                     dof_pos=positions, dof_vel=torch.randn(count, dofs, device=device),
                     body_pos=torch.randn(count, joints + 1, 7, device=device)[..., :3])
        ball_pos, ball_vel = torch.randn(count, 3, device=device), torch.randn(count, 3, device=device)
        engine = SimpleNamespace()
        for name, value in state.items():
            ball = ball_pos if name == "root_pos" else ball_vel
            setattr(engine, "get_" + name, lambda obj, v=value, b=ball: v if obj == 0 else b)
        keys = torch.tensor([kin.get_body_names().index(name) for name in
                             ("head", "right_hand", "left_hand", "right_foot", "left_foot")], device=device)
        width = 1 + 12 + 6 * joints + dofs + 3 * len(keys) + 6
        env = SimpleNamespace(_engine=engine, _kin_char_model=kin, _get_char_id=lambda: 0,
                              _scene_obj_ids=[1], _key_body_ids=keys,
                              _enable_phase_obs=False, _enable_tar_obs=False,
                              _root_height_obs=True, _global_obs=False,
                              _obs_buf=torch.empty(count, width, device=device),
                              _inactive_sphere_pos=torch.tensor([[[10., 10., .1]]], device=device).expand(count, 1, 3),
                              _inactive_sphere_vel=torch.zeros(count, 1, 3, device=device))
        with wp.ScopedStream(wp.stream_from_torch(device), sync_exit=True):
            builder = ObservationBuilder(env)
            for use_ball in (False, True):
                expected = compute_char_obs(state["root_pos"], state["root_rot"], state["root_vel"],
                    state["root_ang_vel"], kin.dof_to_rot(state["dof_pos"]), state["dof_vel"],
                    state["body_pos"][:, keys], False, True)
                task = compute_dodgeball_observations(state["root_pos"], state["root_rot"],
                    ball_pos[:, None] if use_ball else env._inactive_sphere_pos,
                    ball_vel[:, None] if use_ball else env._inactive_sphere_vel)
                builder.update(use_scene_sphere=use_ball)
                torch.testing.assert_close(env._obs_buf, torch.cat((expected, task), -1), rtol=1e-5, atol=1e-6)
            saved = env._obs_buf.clone()
            builder.update(torch.empty(0, dtype=torch.long, device=device))
            torch.testing.assert_close(env._obs_buf, saved, rtol=0, atol=0)

    def test_actor_uses_original_normalization_and_action_inverse(self):
        torch.manual_seed(1234)
        device = "cuda:0"
        model = SimpleNamespace(_actor_layers=torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.ReLU()).to(device),
            _action_dist=DistributionGaussianDiagBuilder(32, 4, StdType.FIXED, .05, .01).to(device))
        model.eval_actor = lambda obs: model._action_dist(model._actor_layers(obs))
        agent = SimpleNamespace(_model=model, _mode=AgentMode.TEST,
            _obs_norm=Normalizer((12,), device, init_mean=torch.arange(12, device=device).float(),
                                 init_std=torch.linspace(.1, 3., 12, device=device), clip=10.),
            _a_norm=Normalizer((4,), device, init_mean=torch.ones(4, device=device),
                               init_std=torch.arange(1, 5, device=device).float()))
        obs = torch.randn(9, 12, device=device) * 30
        with torch.no_grad():
            expected, _ = PPOAgent._decide_action(agent, obs, {})
            actual = PolicyInference(agent).act(obs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
