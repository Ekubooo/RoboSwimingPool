"""Continuous policy playback without training/test statistics or auto-reset.

The original agent still owns loading/GSI and the trained weights. The hot path
uses its actor and scalers through PolicyInference. Viewer.should_step() is the
single source of pause/single-step state; the profiler uses this same loop.
"""
import math
import logging
from contextlib import contextmanager
from functools import partial
from time import perf_counter

import torch
import warp as wp

from RSP.simulation.scenes.layout import ScenePlacementError

from .playback_stats import PlaybackStats
from .ui.interaction import FluidInteraction
from .ui.playback import PlaybackPanel, draw_policy_controls


class PlaybackRunner:
    def __init__(self, session, *, fluid_controls=None, stats_sink=None):
        self.session = session
        self.policy = session.policy
        self.viewer = session.viewer
        self.fluid_controls = fluid_controls
        self._stats_sink = stats_sink
        self._stats = PlaybackStats() if stats_sink is not None else None
        if stats_sink is not None:
            stats_sink(None)
        self.obs, self.info = session.obs, session.info
        frame_dt, policy_hz = session.frame_dt, session.policy_hz
        ratio = 1.0 / (policy_hz * frame_dt) if policy_hz > 0 and frame_dt > 0 else 0.0
        if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("Policy period must be a positive whole number of simulation frames")
        self.policy_interval = round(ratio)
        self._clear_action()
        self._reset_error = None
        self._torch_stream = self._warp_stream = None
        if torch.device(session.device).type == "cuda":
            self._torch_stream = torch.cuda.current_stream(session.device)
            self._warp_stream = wp.stream_from_torch(self._torch_stream)

    @property
    def env(self):
        """Expose the environment to diagnostics without coupling the loop to it."""
        return self.session.env

    def reset(self):
        if self._stats is not None:
            self._stats.clear()
            self._stats_sink(None)
        with torch.cuda.stream(self._torch_stream), wp.ScopedStream(self._warp_stream, sync_exit=True):
            self.session.consume_reset_request()
            if self.fluid_controls is not None:
                self.fluid_controls.restore_defaults()
                self.fluid_controls.apply_pending()
            self.obs, self.info = self.session.reset()
            self._clear_action()
            self._reset_error = None

    def _clear_action(self):
        self._policy_phase = 0
        self._action = None

    def tick(self):
        """Process queued commands, then advance once or render a paused frame."""
        started = perf_counter() if self._stats is not None else None
        viewer = self.viewer
        was_paused = self._stats is not None and viewer is not None and viewer.is_paused()
        # Warp's enter/exit dependencies use device events, not a host wait.
        # This also covers direct tick/reset calls made outside run().
        with torch.cuda.stream(self._torch_stream), wp.ScopedStream(self._warp_stream, sync_exit=True):
            stepped = self._tick()
        if self._stats is not None:
            elapsed = perf_counter() - started
            # Exclude paused drawing, single steps and pause/resume transitions.
            continuous = stepped and not was_paused and (viewer is None or not viewer.is_paused())
            self._stats_sink(self._stats.record(elapsed, continuous=continuous))
        return stepped

    def _tick(self):
        if self.session.consume_reset_request():
            if self.viewer is not None:
                self.viewer.set_paused(True)
            try:
                self.reset()
            except ScenePlacementError as error:
                if self.viewer is None:
                    raise
                self._reset_error = f"Reset failed: {error}. Try Reset again."
                logging.exception("Scene reset failed; playback paused")
        if self._reset_error is not None:
            # A failed reset may have written only part of the sampled state.
            # Only another successful reset can enable simulation again.
            self.viewer.set_paused(True)
            if self.fluid_controls is not None:
                self.fluid_controls.error = self._reset_error
            self.session.render_only()
            return False
        if self.fluid_controls is not None:
            self.fluid_controls.apply_pending()
        self.session.apply_commands()
        if self.viewer is not None and not self.viewer.should_step():
            self.session.render_only()
            return False
        if self.fluid_controls is not None:
            try:
                self.session.prepare_step()
                self.fluid_controls.error = None
            except Exception as error:
                if self.viewer is None:
                    raise
                self.viewer.set_paused(True)
                self.fluid_controls.error = str(error)
                logging.exception("Fluid graph update failed; playback paused")
                self.session.render_only()
                return False
        if self._policy_phase == 0:
            self._action = self.policy.act(self.obs)
        # Hold the same GPU action between decisions; env.step retains its
        # original clipping/target handling, and PD still runs every rigid step.
        self.obs, self.info = self.session.step(self._action)
        self._policy_phase = (self._policy_phase + 1) % self.policy_interval
        return True

    def run(self, *, reset=True):
        with torch.no_grad():
            if reset:
                self.reset()
            while self.viewer is None or self.viewer.is_running():
                self.tick()


@contextmanager
def bind_playback(session):
    """Bind controls and UI to an already constructed session and viewer."""
    controls = None
    if session.set_fluid_controls is not None:
        defaults = {}
        if session.fluid_gravity is not None:
            defaults["gravity"] = session.fluid_gravity
        if session.fluid_settings is not None:
            defaults["feedback_scale"] = session.fluid_settings.feedback_scale
        controls = FluidInteraction(session.set_fluid_controls, **defaults)

    viewer = session.viewer
    gui = None if viewer is None else viewer.install_gui()
    panel = None
    if gui is not None:
        panel = PlaybackPanel(viewer, fluid_controls=controls, request_reset=session.request_reset,
                              draw_policy=partial(draw_policy_controls,
                                  get_scene_sphere=session.get_policy_use_scene_sphere,
                                  request_scene_sphere=session.request_policy_use_scene_sphere))
        gui.set_panel(panel)
    try:
        yield PlaybackRunner(session, fluid_controls=controls,
                             stats_sink=None if panel is None else panel.set_playback_fps)
    finally:
        if gui is not None:
            gui.set_panel(None)


def play_mimickit(env, agent):
    """Run an environment and policy whose resources belong to the caller."""
    from RSP.integrations.mimickit.session import MimicKitSession, configure_test_agent

    configure_test_agent(agent)
    with bind_playback(MimicKitSession(env, agent, policy_hz=1.0 / env.get_policy_timestep())) as runner:
        runner.run()
