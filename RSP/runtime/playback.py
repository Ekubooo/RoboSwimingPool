"""Continuous policy playback without training/test statistics or auto-reset.

The original agent still owns loading/GSI and the trained weights. The hot path
uses its actor and scalers through PolicyInference. Viewer.should_step() is the
single source of pause/single-step state; the profiler uses this same loop.
"""
import torch
import warp as wp

from .inference import PolicyInference


class PlaybackRunner:
    def __init__(self, env, agent, *, policy=None):
        self.env = env
        self.policy = policy if policy is not None else PolicyInference(agent)
        self.viewer = env._engine.get_viewer()
        self.obs = env._obs_buf
        self.info = env._info
        self._torch_stream = self._warp_stream = None
        if torch.device(env._device).type == "cuda":
            self._torch_stream = torch.cuda.current_stream(env._device)
            self._warp_stream = wp.stream_from_torch(self._torch_stream)

    def reset(self):
        with torch.cuda.stream(self._torch_stream), wp.ScopedStream(self._warp_stream, sync_exit=True):
            self.obs, self.info = self.env.reset()

    def tick(self):
        """Process queued commands, then advance once or render a paused frame."""
        # Warp's enter/exit dependencies use device events, not a host wait.
        # This also covers direct tick/reset calls made outside run().
        with torch.cuda.stream(self._torch_stream), wp.ScopedStream(self._warp_stream, sync_exit=True):
            return self._tick()

    def _tick(self):
        if self.env.apply_playback_commands():
            self.obs, self.info = self.env._obs_buf, self.env._info
        if self.viewer is not None and not self.viewer.should_step():
            self.env.render_only()
            return False
        action = self.policy.act(self.obs)
        self.obs, _, _, self.info = self.env.step(action)
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
