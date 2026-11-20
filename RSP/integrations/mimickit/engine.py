"""Assemble the Newton simulation and adapt its API to MimicKit."""

import warp as wp

wp.config.enable_backward = False

import torch
import engines.engine as engine
from util import resource_util
from util.logger import Logger

from RSP.simulation.runtime import SimulationRuntime, StepHooks
from RSP.simulation.scenes.builder import SceneBuilder
from RSP.simulation.state import NewtonStatePair

from . import actuation, state_views
from .actuation import Actuation
from .engine_api import NewtonEngineAPI
from .recorder import EnginePresentation
from .scene import MimicKitScene
from .state_views import SimState


class SceneReset:
    """Coordinate runtime placement/reset; borrow components, own no buffers.

    Placement and reset remain separate operations: the environment first writes
    its sampled joints and scene objects, then places the character, then resets
    solver history. Keep that order and the existing FK synchronization points.
    """

    def __init__(self, state, actuation, runtime, play_scene, clear_dof_feedback):
        self._state = state
        self._actuation = actuation
        self._runtime = runtime
        self._play_scene = play_scene
        self._clear_dof_feedback = clear_dof_feedback

    def place_character(self, obj_id):
        self._state.pre_step_update()
        self._state.eval_fk()
        root = self._state.get_root_pos(obj_id)
        shift = self._play_scene.character_translation(
            self._state.raw_state, self._state.get_obj_body_ids(obj_id), root[0].cpu().numpy())
        root.add_(torch.as_tensor(shift, device=root.device))
        return shift

    def reset(self):
        self._state.pre_step_update()
        self._runtime.reset_solver()
        self._state.synchronize_buffers()
        self._actuation.clear_external_forces()
        self._runtime.reset_clock()
        self._state.post_step_update()
        self._clear_dof_feedback()
        self._runtime.clear_contact_feedback()

    def place_on_terrain(self, obj_id):
        self._state.pre_step_update()
        self._state.eval_fk()
        shift = self._play_scene.terrain_character_translation(
            self._state.raw_state, self._state.get_obj_body_ids(obj_id))
        root = self._state.get_root_pos(obj_id)
        root.add_(torch.as_tensor(shift, device=root.device))
        return shift


class NewtonEngine(NewtonEngineAPI):
    """Assemble components and run workflows behind NewtonEngineAPI.

    SceneBuilder declares the model. SimState adapts Newton buffers and owns views;
    Actuation owns command processing. SimulationRuntime drives the coupled
    solver and feedback. SceneReset coordinates runtime reset, and Presentation
    owns display/recording. Components never call back into this facade, except
    for the existing video recorder's public Engine API adapter.
    """

    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)
        self._runtime = SimulationRuntime(config, num_envs, device)
        self._scene = SceneBuilder(config, num_envs, device,
                                   self._runtime.timestep, self._runtime._sim_steps,
                                   scene_type=MimicKitScene, log=Logger.print, require_dfs=True)
        self._control_mode = engine.ControlMode[config.get("control_mode", "none")]
        self._scene.build_ground()
        self._presentation = resource_util.construct(
            EnginePresentation, num_envs, self._runtime.env_spacing, visualize, record_video,
            viewer=config.get("playback_viewer"))

    def get_name(self):
        return "newton_pbf"

    def initialize_sim(self):
        model = self._scene.finalize()
        states = NewtonStatePair(model)
        self._state = SimState(model, self.get_num_envs(), states=states)
        for env_id, obj_id, position, rotation in self._scene.start_transforms():
            self._state.set_root_pos(env_id, obj_id, torch.tensor(position, device=self._runtime.device))
            self._state.set_root_rot(env_id, obj_id, torch.tensor(rotation, device=self._runtime.device))
        states.eval_fk()
        self._control = Actuation(model, self._state, self._control_mode, self.get_num_envs())
        hooks = StepHooks(
            control=self._control.controls.control,
            needs_actuator_force=self._control_mode in (engine.ControlMode.pos, engine.ControlMode.vel),
            prepare_frame=self._prepare_frame,
            apply_external_forces=self._control.apply_external_forces,
            apply_control=self._control.apply_control,
            finish_frame=self._finish_frame,
        )
        self._feedback = self._runtime.initialize(
            states, self._scene.play_scene, self._scene.coupled_body_ids(), hooks, self._presentation.apply_forces,
        )
        self._state.initialize_auxiliary_views()
        self._control.initialize_force_buffer()
        self._bind_feedback()
        self._scene_reset = SceneReset(self._state, self._control, self._runtime,
                                       self._scene.play_scene, self._clear_dof_feedback)
        self._presentation.initialize(model, self._scene.object_body_ids, self._scene.colors)
        self._presentation.initialize_recording(self)
        self._runtime.build_graphs(self._preload_step_kernels)

    def _bind_feedback(self):
        num_envs = self.get_num_envs()
        layout = self._state.layout
        objects = range(layout.objs_per_env)
        contact = wp.to_torch(self._feedback.total_force).view(num_envs, -1, 3)
        ground = wp.to_torch(self._feedback.ground_force_matrix)
        # The model has one global ground plane.
        assert ground.shape[1] == 1
        ground = ground.view(num_envs, -1, 3)
        self._wp_dof_forces = None
        if self._feedback.actuator_force is not None:
            self._wp_dof_forces = wp.clone(self._feedback.actuator_force)
            forces = wp.to_torch(self._wp_dof_forces).view(num_envs, -1)
            dof_forces = [forces[:, layout.dof_slices[obj_id]] for obj_id in objects]
        else:
            dof_forces = self._control.controls.joint_force
        self._state.bind_feedback(
            dof_forces,
            [contact[:, layout.body_slices[obj_id], :] for obj_id in objects],
            [ground[:, layout.body_slices[obj_id], :] for obj_id in objects],
        )

    def _prepare_frame(self):
        self._state.pre_step_update()
        self._control.update_targets()

    def _finish_frame(self, final_state):
        self._state.post_step_update()
        if self._wp_dof_forces is not None:
            wp.copy(self._wp_dof_forces, self._feedback.actuator_force)
            self._state.rotate_dof_vectors(
                final_state.joint_q, self._wp_dof_forces, self._wp_dof_forces, to_newton=False,
            )
        self._control.finish_step()

    def _clear_dof_feedback(self):
        if self._wp_dof_forces is not None:
            self._wp_dof_forces.zero_()
        else:
            self._control.controls.dof_force.zero_()

    def _preload_step_kernels(self):
        device = self._state.model.device
        for module in (state_views.__name__, actuation.__name__):
            wp.load_module(module=module, device=device)


    def step(self):
        self._presentation.capture_before_step()
        self._runtime.advance()
        self._runtime.finish_step()

    # Placement does not reset the world; the environment calls these in order.

    def place_character(self, obj_id):
        return self._scene_reset.place_character(obj_id)

    def reset_play_scene(self):
        self._scene_reset.reset()

    def place_character_on_terrain(self, obj_id):
        return self._scene_reset.place_on_terrain(obj_id)

    def render(self):
        self._presentation.render(self.get_sim_time(), self._state.raw_state)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        presentation = getattr(self, "_presentation", None)
        resource_util.cleanup(presentation.close if presentation is not None else None)
        return
