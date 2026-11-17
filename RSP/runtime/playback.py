"""Continuous policy playback without training/test statistics or auto-reset.

The original agent still owns loading/GSI and the trained weights. The hot path
uses its actor and scalers through PolicyInference. Viewer.should_step() is the
single source of pause/single-step state; the profiler uses this same loop.
"""
import math
import logging

import torch
import warp as wp

from .inference import PolicyInference
from .interaction import FluidInteraction


# Policy frequency is measured in simulated time, independently of rendering.
POLICY_HZ = 30


class PlaybackRunner:
    def __init__(self, env, agent, *, policy=None, policy_hz=POLICY_HZ):
        self.env = env
        self.policy = policy if policy is not None else PolicyInference(**agent.get_playback_policy_components())
        self.viewer = env._engine.get_viewer()
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
        # This single composition boundary is the existing MimicKit demo glue.
        # The controller, UI and coupled solver themselves do not depend on Env.
        apply_settings = getattr(env._engine, "set_fluid_controls", None)
        self.fluid_controls = FluidInteraction(apply_settings) if apply_settings is not None else None
        if self.viewer is not None and self.fluid_controls is not None:
            self.viewer.gui.fluid_controls = self.fluid_controls
            self.viewer.set_reset_callback(self.fluid_controls.request_reset)

    def reset(self):
        with torch.cuda.stream(self._torch_stream), wp.ScopedStream(self._warp_stream, sync_exit=True):
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
        # Warp's enter/exit dependencies use device events, not a host wait.
        # This also covers direct tick/reset calls made outside run().
        with torch.cuda.stream(self._torch_stream), wp.ScopedStream(self._warp_stream, sync_exit=True):
            return self._tick()

    def _tick(self):
        if self.fluid_controls is not None and self.fluid_controls.apply_pending():
            if self.viewer is not None:
                self.viewer._paused = True
                self.viewer._step_requested = False
            self.env.request_reset()
        if self.env.apply_playback_commands():
            self.obs, self.info = self.env._obs_buf, self.env._info
            self._clear_action()
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
                self.viewer._paused = True
                self.viewer._step_requested = False
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
