"""Continuous policy playback without training/test statistics or auto-reset.

The original agent still owns loading/GSI and the trained weights. The hot path
uses its actor and scalers through PolicyInference. Viewer.should_step() is the
single source of pause/single-step state; the profiler uses this same loop.
"""
import math
import logging
from time import perf_counter

import torch
import warp as wp

from .mimickit_bindings import build_playback_components
from .playback_stats import PlaybackStats


# Policy frequency is measured in simulated time, independently of rendering.
POLICY_HZ = 30


class PlaybackRunner:
    def __init__(self, env, agent, *, policy=None, policy_hz=POLICY_HZ):
        self.env = env
        components = build_playback_components(env, agent, policy=policy)
        self.policy = components.policy
        self.viewer = components.viewer
        self.fluid_controls = components.fluid_controls
        self._stats_gui = getattr(self.viewer, "gui", None)
        self._stats = PlaybackStats() if self._stats_gui is not None else None
        if self._stats_gui is not None:
            self._stats_gui.set_playback_fps(None)
        self.obs = env._obs_buf
        self.info = env._info
        frame_dt = env._engine.get_timestep()
        ratio = 1.0 / (policy_hz * frame_dt) if policy_hz > 0 and frame_dt > 0 else 0.0
        if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("Policy period must be a positive whole number of simulation frames")
        self.policy_interval = round(ratio)
        self._clear_action()
        self._torch_stream = self._warp_stream = None
        if torch.device(env._device).type == "cuda":
            self._torch_stream = torch.cuda.current_stream(env._device)
            self._warp_stream = wp.stream_from_torch(self._torch_stream)

    def reset(self):
        if self._stats is not None:
            self._stats.clear()
            self._stats_gui.set_playback_fps(None)
        with torch.cuda.stream(self._torch_stream), wp.ScopedStream(self._warp_stream, sync_exit=True):
            self.env.consume_reset_request()
            if self.fluid_controls is not None:
                self.fluid_controls.restore_defaults()
                self.fluid_controls.apply_pending()
            self.obs, self.info = self.env.reset()
            self._clear_action()

    def _clear_action(self):
        self._policy_phase = 0
        self._action = None

    def tick(self):
        """Process queued commands, then advance once or render a paused frame."""
        started = perf_counter() if self._stats is not None else None
        was_paused = self.viewer.is_paused() if self._stats is not None else False
        # Warp's enter/exit dependencies use device events, not a host wait.
        # This also covers direct tick/reset calls made outside run().
        with torch.cuda.stream(self._torch_stream), wp.ScopedStream(self._warp_stream, sync_exit=True):
            stepped = self._tick()
        if self._stats is not None:
            elapsed = perf_counter() - started
            # Exclude paused drawing, single steps and pause/resume transitions.
            continuous = stepped and not was_paused and not self.viewer.is_paused()
            self._stats_gui.set_playback_fps(self._stats.record(elapsed, continuous=continuous))
        return stepped

    def _tick(self):
        if self.env.consume_reset_request():
            if self.viewer is not None:
                self.viewer.set_paused(True)
            self.reset()
        if self.fluid_controls is not None:
            self.fluid_controls.apply_pending()
        self.env.apply_playback_commands()
        if self.viewer is not None and not self.viewer.should_step():
            self.env.render_only()
            return False
        if self.fluid_controls is not None:
            try:
                self.env._engine.prepare_playback_step()
                self.fluid_controls.error = None
            except Exception as error:
                if self.viewer is None:
                    raise
                self.viewer.set_paused(True)
                self.fluid_controls.error = str(error)
                logging.exception("Fluid graph update failed; playback paused")
                self.env.render_only()
                return False
        if self._policy_phase == 0:
            self._action = self.policy.act(self.obs)
        # Hold the same GPU action between decisions; env.step retains its
        # original clipping/target handling, and PD still runs every rigid step.
        self.obs, _, _, self.info = self.env.step(self._action)
        self._policy_phase = (self._policy_phase + 1) % self.policy_interval
        return True

    def run(self, *, reset=True):
        with torch.no_grad():
            if reset:
                self.reset()
            while self.viewer is None or self.viewer.is_running():
                self.tick()


def play(env, agent):
    """Entry-point adapter; model loading and cleanup stay with the caller."""
    from learning.base_agent import AgentMode

    agent.eval()
    agent.set_mode(AgentMode.TEST)
    PlaybackRunner(env, agent).run()
