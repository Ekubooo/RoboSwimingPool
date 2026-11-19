"""Step a Newton model with the existing rigid and fluid solvers."""

from collections.abc import Callable
from dataclasses import dataclass

import newton
import warp as wp

from RSP.physics.clocked_coupling import ClockedPBFMuJoCo


@dataclass(frozen=True, eq=False)
class StepHooks:
    """Frame and substep work submitted by the simulation's caller."""

    control: newton.Control
    needs_actuator_force: bool
    prepare_frame: Callable[[], None]
    apply_external_forces: Callable[[newton.State], None]
    apply_control: Callable[[newton.State, newton.Control], None]
    finish_frame: Callable[[newton.State], None]


@dataclass(frozen=True, eq=False)
class RuntimeFeedback:
    """Raw Newton arrays borrowed by the caller for the runtime's lifetime."""

    total_force: wp.array
    ground_force_matrix: wp.array
    actuator_force: wp.array | None


class SimulationRuntime:
    """Own solver execution, contact feedback, Graphs and the physics clock.

    The caller supplies its frame and substep hooks. Contacts, observables and
    solver resources remain local to this runtime.
    """

    def __init__(self, config, num_envs, device):
        if num_envs != 1 or not wp.get_device(device).is_cuda:
            raise ValueError("newton_pbf currently supports one CUDA environment")
        self.device = device
        self.num_envs = num_envs
        self.env_spacing = config["env_spacing"]
        # Preserve contact-rich GSI initialization capacity.
        self._solver_njmax = int(config.get("newton_njmax", 450))
        self._solver_nconmax = int(config.get("newton_nconmax", 150))
        if self._solver_njmax <= 0 or self._solver_nconmax <= 0:
            raise ValueError("Newton constraint/contact capacities must be positive.")
        sim_freq = config.get("sim_freq", 60)
        control_freq = config.get("control_freq", 10)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"
        self.timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        self.sim_timestep = 1.0 / sim_freq
        self._sim_step_count = 0
        self._enable_cuda_graph = bool(config.get("enable_cuda_graph", True))

    def initialize(self, states, play_scene, boundary_body_ids, hooks, interaction_forces):
        self._state = states
        self._sim_model = states.model
        self._hooks = hooks
        self._interaction_forces = interaction_forces
        self._build_solver(play_scene, boundary_body_ids)
        play_scene.initialize_particles(states.raw_state, states.swap_state)
        self._solver.boundaries.sync_body_poses(states.raw_state)
        self._build_contact_feedback()
        self._build_observables()
        return RuntimeFeedback(
            self._ground_contact_sensor.total_force,
            self._ground_contact_sensor.force_matrix,
            self._observables.qfrc_actuator if hooks.needs_actuator_force else None,
        )

    def advance(self):
        if self._graph is not None:
            self._graph.launch()
        else:
            self._simulate()

    def finish_step(self):
        self._ground_contact_sensor.update(self._state.raw_state, self._contacts,
                                           observables=self._observables)
        self._sim_step_count += 1

    def get_sim_time(self):
        return self.timestep * self._sim_step_count

    def reset_solver(self):
        # The composed solver preserves the freshly sampled GSI joint state.
        self._solver.reset(self._state.raw_state)

    def reset_clock(self):
        self._sim_step_count = 0

    def clear_contact_feedback(self):
        self._ground_contact_sensor.total_force.zero_()
        self._ground_contact_sensor.force_matrix.zero_()

    # Initialization: rigid/fluid solvers, contact observations and readout views.
    def _build_solver(self, play_scene, boundary_body_ids):
        rigid = newton.solvers.SolverMuJoCo(
            self._sim_model,
            solver="newton",
            njmax=self._solver_njmax,
            nconmax=self._solver_nconmax,
            impratio=10,
            iterations=100,
            ls_iterations=50
        )

        boundaries = play_scene.bind_model(self._sim_model, boundary_body_ids)

        self._solver = ClockedPBFMuJoCo(
            self._sim_model,
            config=play_scene.config,
            boundaries=boundaries, rigid=rigid,
            settings=play_scene.fluid_settings, schedule=play_scene.fluid_schedule
        )
        return


    def _build_contact_feedback(self):
        self._ground_contact_sensor = newton.sensors.SensorContact(self._sim_model,
                                                                   sensing_bodies=list(range(self._sim_model.body_count)),
                                                                   counterpart_shapes="ground*",
                                                                   measure_total=True,
                                                                   request_contact_attributes=False,
                                                                   verbose=True)


    def _build_observables(self):
        self._collision_pipeline = newton.CollisionPipeline(
            self._sim_model, rigid_contact_max=self._solver.get_max_contact_count(),
            soft_contact_max=0)
        self._contacts = self._collision_pipeline.contacts()
        kinds = set(self._ground_contact_sensor.solver_observable_kinds)
        if self._hooks.needs_actuator_force:
            kinds.add(self._solver.ObservableKind.QFRC_ACTUATOR)
        self._observables = self._solver.observables(kinds, contacts=self._contacts)



    # Execution: capture the same substep sequence that eager mode runs.
    def set_fluid_controls(self, *, gravity, feedback_scale, slow_motion):
        self._solver.set_fluid_controls(
            gravity=gravity, feedback_scale=feedback_scale, slow_motion=slow_motion)

    def prepare_playback_step(self):
        """Publish current captured work; eager execution needs no preparation."""
        if self._graph is not None:
            self._graph.ensure_current()

    def build_graphs(self, preload_adapter_kernels):
        self._graph = None
        if not self._enable_cuda_graph:
            return
        # Compile adapter kernels before capture, including scalar-only scenes.
        preload_adapter_kernels()
        self._graph = self._solver.capture_step_graph(self._simulate, self._sim_steps)
        return


    def _simulate(self):
        state0 = self._state.raw_state
        state1 = self._state.swap_state
        control = self._hooks.control

        self._hooks.prepare_frame()

        for i in range(self._sim_steps):
            self._pre_sim_step(state0, control)
            self._solver.step(state0, state1, control, self._contacts, self.sim_timestep,
                              observables=self._observables if i == self._sim_steps - 1 else None)
            state0, state1 = state1, state0

        self._finish_simulation(state0)



    def _pre_sim_step(self, raw_state, control):
        raw_state.clear_forces()
        self._hooks.apply_external_forces(raw_state)
        self._interaction_forces(raw_state)
        self._hooks.apply_control(raw_state, control)
        return


    def _finish_simulation(self, final_state):
        if self._sim_steps % 2 != 0:
            self._state.copy(self._state.swap_state)

        self._hooks.finish_frame(final_state)
