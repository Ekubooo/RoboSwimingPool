"""PBF cadence, held forces, particle continuity and standing playback reset."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import warp as wp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
wp.config.kernel_cache_dir = str(ROOT / "output/newton_pbf_refactor_checks/warp_cache")

from anim.mjcf_char_model import MJCFCharModel
from engines.newton_pbf_engine import SimulationRuntime
from engines.newton_pbf_runtime import runtime_config
from envs.char_env import CharEnv
from envs.pbf_play_env import PBFPlayEnv
from envs.smp_env import SMPEnv
from solvers.pbf import coupling, pbf_helper
from util.circular_buffer import CircularBuffer


class TimingContracts(unittest.TestCase):
    config = dict(control_freq=30, sim_freq=240, env_spacing=5.)

    def test_defaults_and_invalid_cadences(self):
        runtime = SimulationRuntime(self.config, 1, "cuda:0")
        self.assertEqual(runtime._fluid_steps, 8)
        self.assertEqual(runtime.fluid_timestep, runtime.sim_timestep)
        for invalid in (0, -1, 3, 16, 2.5, True, None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "pbf_substeps"):
                SimulationRuntime(self.config | {"pbf_substeps": invalid}, 1, "cuda:0")
        for invalid in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "pressure_iterations"):
                runtime_config(1 / 30, 2, pressure_iterations=invalid)

    def test_runtime_schedule_preserves_rigid_control(self):
        for steps in (1, 2, 4, 8):
            with self.subTest(fluid_steps=steps):
                runtime = SimulationRuntime(self.config | {"pbf_substeps": steps}, 1, "cuda:0")
                a, b, control, contacts, observables = (object() for _ in range(5))
                runtime._state = SimpleNamespace(raw_state=a, swap_state=b)
                runtime._control = SimpleNamespace(controls=SimpleNamespace(control=control))
                runtime._contacts, runtime._observables = contacts, observables
                runtime._solver = Mock()
                runtime._prepare_step, runtime._pre_sim_step, runtime._finish_simulation = Mock(), Mock(), Mock()
                runtime._simulate()
                runtime._prepare_step.assert_called_once_with()
                runtime._finish_simulation.assert_called_once_with(a)
                self.assertEqual(runtime._pre_sim_step.call_count, 8)
                calls = runtime._solver.step.call_args_list
                self.assertEqual(len(calls), 8)
                for i, call in enumerate(calls):
                    self.assertEqual(call.args, ((a, b) if i % 2 == 0 else (b, a)) +
                                     (control, contacts, 1 / 240))
                    self.assertEqual(call.kwargs["advance_fluid"], i % (8 // steps) == 0)
                    self.assertEqual(call.kwargs["fluid_dt"], 1 / (30 * steps))
                    self.assertIs(call.kwargs["observables"], observables if i == 7 else None)
                self.assertEqual(sum(c.kwargs["advance_fluid"] for c in calls), steps)
                self.assertAlmostEqual(steps * runtime.fluid_timestep, 8 * runtime.sim_timestep)

    def test_graph_uses_fluid_parity_and_rigid_counter(self):
        for steps in (1, 2, 4):
            with self.subTest(fluid_steps=steps):
                a, b = SimpleNamespace(ptr=1), SimpleNamespace(ptr=2)
                fluid = SimpleNamespace(pre_Pos=a, temp_pPos=b, grid=Mock())
                solver = SimpleNamespace(model=SimpleNamespace(device="cuda:0", particle_count=4),
                                         fluid=fluid, rigid=SimpleNamespace(_step=10))

                def step():
                    for _ in range(steps):
                        fluid.pre_Pos, fluid.temp_pPos = fluid.temp_pPos, fluid.pre_Pos
                    solver.rigid._step += 8

                with patch.object(wp, "load_module"), patch.object(wp, "ScopedCapture"), \
                        patch.object(wp, "capture_launch"):
                    graph = coupling.CoupledStepGraph(solver, step, 8, fluid_substeps=steps)
                    self.assertEqual(len(graph._graphs), 2 if steps % 2 else 1)
                    self.assertIs(fluid.pre_Pos, a)
                    self.assertEqual(solver.rigid._step, 10)
                    for frame in range(1, 5):
                        graph.launch()
                        self.assertIs(fluid.pre_Pos, (a, b)[frame * steps % 2])
                        self.assertEqual(solver.rigid._step, 10 + frame * 8)

    def test_held_force_impulse_and_gpu_particle_continuity(self):
        device = "cuda:0"
        def state():
            return SimpleNamespace(particle_q=wp.zeros(3, dtype=wp.vec3, device=device),
                                   particle_qd=wp.zeros(3, dtype=wp.vec3, device=device),
                                   body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device))
        a, b = state(), state()
        forces = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        solver = SimpleNamespace(model=SimpleNamespace(device=device, body_count=1),
                                 fluid=Mock(), rigid=Mock(), boundaries=Mock(), boundary_forces=forces)
        updates, impulses = [], []
        force = np.arange(1, 7, dtype=np.float32).reshape(1, 6)

        def fluid_step(state_in, state_out, control, contacts, dt):
            np.testing.assert_array_equal(forces.numpy(), 0.)
            updates.append(dt)
            forces.assign(force * len(updates))
            state_out.particle_q.fill_(wp.vec3(float(len(updates))))
            state_out.particle_qd.fill_(wp.vec3(float(len(updates) + 10)))

        def rigid_step(state_in, state_out, control, contacts, dt, **kwargs):
            # Each step also starts with an independent external body force.
            actual = state_in.body_f.numpy() - 7.
            np.testing.assert_array_equal(actual, force * len(updates))
            impulses.append(actual * dt)

        solver.fluid.step.side_effect = fluid_step
        solver.rigid.step.side_effect = rigid_step
        for i in range(8):
            a.body_f.fill_(wp.spatial_vector(7., 7., 7., 7., 7., 7.))
            coupling.SolverPBFMuJoCo.step(solver, a, b, None, None, 1 / 240,
                                         advance_fluid=i % 4 == 0, fluid_dt=1 / 60)
            a, b = b, a
            np.testing.assert_array_equal(a.particle_q.numpy(), 1. if i < 4 else 2.)
            np.testing.assert_array_equal(a.particle_qd.numpy(), 11. if i < 4 else 12.)
        self.assertEqual(updates, [1 / 60, 1 / 60])
        np.testing.assert_allclose(np.sum(impulses, axis=0), force * 3 / 60, rtol=1e-6)

    def test_scene_lattice_preserves_particle_count(self):
        config = runtime_config(1 / 30, 2)
        particles = SimpleNamespace(config=config)
        pbf_helper.init_scene_parameters(particles)
        self.assertEqual(particles.n, 195525)
        self.assertEqual(tuple(particles.block_counts), (99, 79, 25))
        np.testing.assert_allclose(np.array(config.block_start) + particles.particle_spacing, .025)
        np.testing.assert_allclose(np.array(config.block_start) +
                                   np.array(particles.block_counts) * particles.particle_spacing,
                                   (2.475, 1.975, .625))
        self.assertEqual(config.pressure_iterations, 3)


class StandingReset(unittest.TestCase):
    def make_env(self):
        env = PBFPlayEnv.__new__(PBFPlayEnv)
        env._playback_mode, env._play_init_mode = True, "standing"
        env._kin_char_model = MJCFCharModel("cpu")
        env._kin_char_model.load(str(ROOT / "data/assets/humanoid/humanoid.xml"))
        config = yaml.safe_load((ROOT / "data/envs/pbf_play_env.yaml").read_text())
        CharEnv._parse_init_pose(env, config["init_pose"], "cpu")
        kin = env._kin_char_model
        shapes = dict(root_pos=(3,), root_rot=(4,), root_vel=(3,), root_ang_vel=(3,),
                      dof_pos=(kin.get_dof_size(),), dof_vel=(kin.get_dof_size(),),
                      joint_rot=(kin.get_num_joints() - 1, 4),
                      body_pos=(kin.get_num_joints(), 3), body_rot=(kin.get_num_joints(), 4))
        env._motion_ids = torch.full((2,), 9, dtype=torch.int64)
        env._motion_time_offsets = torch.full((2,), 9.)
        for name, shape in shapes.items():
            setattr(env, "_ref_" + name, torch.full((2, *shape), 9.))
            if name not in ("dof_pos", "body_rot"):
                buffer = CircularBuffer(2, 10, shape, torch.float32, "cpu")
                buffer.push(torch.full((2, *shape), 9.))  # Exercise a nonzero history head.
                setattr(env, "_disc_hist_" + name, buffer)
        return env

    def test_fixed_pose_zero_velocity_and_consistent_histories(self):
        env = self.make_env()
        ids = torch.tensor([1])
        env._reset_ref_motion(ids)
        env._reset_disc_hist(ids)
        torch.testing.assert_close(env._ref_root_pos[1], env._init_root_pos)
        torch.testing.assert_close(env._ref_root_rot[1], env._init_root_rot)
        torch.testing.assert_close(env._ref_dof_pos[1], env._init_dof_pos)
        for name in ("root_vel", "root_ang_vel", "dof_vel"):
            self.assertEqual(torch.count_nonzero(getattr(env, "_ref_" + name)[1]), 0)
        for name in ("root_pos", "root_rot", "root_vel", "root_ang_vel", "joint_rot", "dof_vel", "body_pos"):
            value = getattr(env, "_ref_" + name)
            torch.testing.assert_close(value[0], torch.full_like(value[0], 9.))
            history = getattr(env, "_disc_hist_" + name).get_all()[1]
            torch.testing.assert_close(history, value[1].expand_as(history))
        # Repeated and empty reset must not depend on the previous history head.
        env._reset_ref_motion(ids)
        env._reset_disc_hist(ids)
        env._reset_ref_motion(torch.tensor([], dtype=torch.int64))
        env._reset_disc_hist(torch.tensor([], dtype=torch.int64))

    def test_sampled_and_nonplay_paths_delegate_and_standing_stays_grounded(self):
        env = self.make_env()
        env._scene_obj_ids = []
        env._engine, env._get_char_id = Mock(), Mock(return_value=0)
        ids = torch.tensor([0])
        for play, mode in ((False, "standing"), (True, "sampled"), (True, "standing")):
            env._playback_mode, env._play_init_mode = play, mode
            standing = play and mode == "standing"
            env._engine.reset_mock()
            with patch.object(SMPEnv, "_reset_ref_motion") as ref, \
                    patch.object(SMPEnv, "_reset_disc_hist") as hist, patch.object(SMPEnv, "_reset_envs"):
                env._reset_ref_motion(ids)
                env._reset_disc_hist(ids)
                self.assertEqual(ref.call_count, 0 if standing else 1)
                self.assertEqual(hist.call_count, 0 if standing else 1)
                env._reset_envs(ids)
                self.assertEqual(env._engine.place_character.call_count, 0 if standing else 1)
                env._engine.reset_play_scene.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
