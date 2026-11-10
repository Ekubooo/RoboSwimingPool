"""Backend contracts; run with Newton 1.7/CUDA via unittest discovery."""
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
wp.config.kernel_cache_dir = str(ROOT / "output/newton_pbf_refactor_checks/warp_cache")
_test_temp = ROOT / "output/newton_pbf_refactor_checks/tmp"
_test_temp.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_test_temp)

import newton
from engines import engine
from engines import newton_pbf_engine as current
from engines import newton_pbf_engine_before_refactor as original
from solvers.pbf import coupling
from util import resource_util


def build_model(with_ball):
    """Two worlds, each with a fixed-root and a free-root mechanism."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    for world in range(2):
        builder.begin_world()
        for fixed in (True, False):
            links = [builder.add_link(mass=1., inertia=wp.mat33(1., 0., 0., 0., 1., 0., 0., 0., 1.),
                                      label=f"world{world}/{'fixed' if fixed else 'free'}/link{i}")
                     for i in range(3)]
            root = (builder.add_joint_fixed(-1, links[0]) if fixed
                    else builder.add_joint_free(links[0]))
            hinge = builder.add_joint_revolute(links[0], links[1], axis=(0., 0., 1.))
            tip = (builder.add_joint_ball(links[1], links[2]) if with_ball
                   else builder.add_joint_revolute(links[1], links[2], axis=(1., 0., 0.)))
            builder.add_articulation([root, hinge, tip])
        builder.end_world()
    model = builder.finalize(device="cuda:0", requires_grad=False)
    model.mujoco.dof_passive_stiffness.fill_(37.)
    model.joint_damping.fill_(2.)
    model.joint_effort_limit.fill_(5.)
    return model


def component_outputs(module, with_ball, mode):
    model = build_model(with_ball)
    state = module.SimState(model, 2)
    control = module.Control(model, state, mode, 2)
    control.initialize_force_buffer()
    for obj in range(2):
        shape = state.get_dof_pos(obj).shape
        pos = torch.linspace(-.4, .5, shape.numel(), device="cuda:0").reshape(shape)
        state.set_dof_pos(None, obj, pos)
        state.set_dof_vel(None, obj, pos * -.7)
        state.set_root_pos(None, obj, torch.tensor([[.2, .3, 1.], [.4, .5, 1.2]], device="cuda:0"))
        state.set_root_vel(None, obj, .1)
        state.set_root_ang_vel(None, obj, -.2)
        control.set_cmd(obj, pos * .4 + .2)
    state.pre_step_update()
    state.eval_fk()
    control.update_targets()
    control.apply_control(state.raw_state, control.controls.control)
    control.set_body_forces(torch.tensor([1], device="cuda:0"), 1, 0,
                            torch.tensor([[3., 2., 1.]], device="cuda:0"))
    control.apply_external_forces(state.raw_state)
    state.post_step_update()
    result = {name: getattr(state.raw_state, name).numpy().copy()
              for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f")}
    result.update({name: getattr(control.controls.control, name).numpy().copy()
                   for name in ("joint_target_q", "joint_target_qd", "joint_f")})
    for obj in range(2):
        for name in ("root_pos", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"):
            result[f"{obj}/{name}"] = getattr(state, "get_" + name)(obj).cpu().numpy().copy()
        result[f"{obj}/mass"] = np.asarray(state.calc_obj_mass(1, obj))
        result[f"{obj}/gains"] = np.asarray(state.get_obj_pd_gains(1, obj))
        result[f"{obj}/limits"] = np.asarray(state.get_obj_dof_limits(1, obj))
        result[f"{obj}/torque_limits"] = state.get_obj_torque_limits(1, obj)
    # One-shot external commands clear without replacing the borrowed storage.
    pointer = state._wp_body_force.ptr
    control.finish_step()
    assert state._wp_body_force.ptr == pointer
    assert not state._wp_body_force.numpy().any()
    assert not control.has_body_forces.numpy().any()
    # Scalars alias Newton coordinates; BALL coordinates intentionally use a cache.
    assert (state.get_dof_pos(0).untyped_storage().data_ptr() == state.raw_state.joint_q.ptr) == (not with_ball)
    assert state.get_body_pos(0).untyped_storage().data_ptr() == state.raw_state.body_q.ptr
    assert state.get_root_pos(1).untyped_storage().data_ptr() == state.raw_state.joint_q.ptr
    if module is current and hasattr(state, "layout"):
        assert control.layout is state.layout is control.controls.layout
    return result


class BackendContracts(unittest.TestCase):
    def test_public_api(self):
        def signatures(cls):
            return {name: str(inspect.signature(method))
                    for name, method in inspect.getmembers(cls, inspect.isfunction)
                    if not name.startswith("_") or name == "__init__"}
        self.assertEqual(signatures(original.NewtonEngine), signatures(current.NewtonEngine))

    def test_state_and_control_modes(self):
        for with_ball in (False, True):
            for mode in engine.ControlMode:
                with self.subTest(ball=with_ball, mode=mode):
                    before = component_outputs(original, with_ball, mode)
                    after = component_outputs(current, with_ball, mode)
                    self.assertEqual(before.keys(), after.keys())
                    for key in before:
                        np.testing.assert_allclose(after[key], before[key], rtol=1e-5, atol=1e-6,
                                                   err_msg=f"{mode.name}/{with_ball}/{key}")

    def test_live_model_metadata(self):
        state = current.SimState(build_model(True), 2)
        state._sim_model.body_mass.fill_(4.)
        state._sim_model.joint_target_ke.fill_(19.)
        state._sim_model.joint_effort_limit.fill_(11.)
        self.assertEqual(state.calc_obj_mass(1, 1), 12.)
        np.testing.assert_array_equal(state.get_obj_pd_gains(1, 1)[0], 19.)
        np.testing.assert_array_equal(state.get_obj_torque_limits(1, 1), 11.)

    def test_partial_and_repeated_close(self):
        eng = current.NewtonEngine.__new__(current.NewtonEngine)
        eng.close()
        eng.close()
        presentation = current.Presentation(1, 5., False, False)
        presentation._video_recorder = Mock()
        eng = current.NewtonEngine.__new__(current.NewtonEngine)
        eng._presentation = presentation
        recorder = presentation._video_recorder
        eng.close()
        eng.close()
        recorder.close.assert_called_once_with()
        with patch.object(current.SceneBuilder, "__init__", side_effect=RuntimeError("build failed")):
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                resource_util.construct(current.NewtonEngine,
                                        {"control_freq": 30, "sim_freq": 240, "env_spacing": 5.},
                                        1, "cuda:0", False)


class GraphLifecycle(unittest.TestCase):
    def solver(self):
        a, b = SimpleNamespace(ptr=1), SimpleNamespace(ptr=2)
        return SimpleNamespace(model=SimpleNamespace(device="cuda:0", particle_count=4),
                               fluid=SimpleNamespace(pre_Pos=a, temp_pPos=b, grid=Mock()),
                               rigid=SimpleNamespace(_step=10))

    def test_replay_tracks_parity(self):
        for substeps in (1, 2, 3, 8):
            with self.subTest(substeps=substeps):
                solver = self.solver()
                initial = (solver.fluid.pre_Pos, solver.fluid.temp_pPos)

                def step():
                    for _ in range(substeps):
                        solver.fluid.pre_Pos, solver.fluid.temp_pPos = solver.fluid.temp_pPos, solver.fluid.pre_Pos
                        solver.rigid._step += 1

                with patch.object(wp, "load_module"), patch.object(wp, "ScopedCapture"), patch.object(wp, "capture_launch"):
                    graph = coupling.CoupledStepGraph(solver, step, substeps)
                    self.assertEqual((solver.fluid.pre_Pos, solver.fluid.temp_pPos), initial)
                    self.assertEqual(solver.rigid._step, 10)
                    graph.launch()
                    self.assertEqual(solver.fluid.pre_Pos, initial[substeps % 2])
                    self.assertEqual(solver.rigid._step, 10 + substeps)
                    graph.launch()
                    self.assertEqual(solver.fluid.pre_Pos, initial[0])
                    self.assertEqual(solver.rigid._step, 10 + 2 * substeps)
                solver.fluid.grid.reserve.assert_called_once_with(4)

    def test_failed_capture_restores_host_state(self):
        solver = self.solver()
        initial = (solver.fluid.pre_Pos, solver.fluid.temp_pPos)

        def fail():
            solver.fluid.pre_Pos, solver.fluid.temp_pPos = initial[::-1]
            solver.rigid._step += 1
            raise RuntimeError("capture failed")

        with patch.object(wp, "load_module"), patch.object(wp, "ScopedCapture"):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                coupling.CoupledStepGraph(solver, fail, 3)
        self.assertEqual((solver.fluid.pre_Pos, solver.fluid.temp_pPos), initial)
        self.assertEqual(solver.rigid._step, 10)


if __name__ == "__main__":
    unittest.main()
