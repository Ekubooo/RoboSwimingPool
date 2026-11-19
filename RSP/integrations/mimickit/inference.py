"""Deterministic playback using the already-loaded MimicKit actor and scalers.

Loading, GSI initialization and ownership remain with the original agent.
Only its per-frame training distribution/statistics path is bypassed.
"""
import torch


class PolicyInference:
    def __init__(self, *, actor, mean, observation_normalizer, action_normalizer):
        """Borrow loaded policy components without changing their ownership."""
        self.actor = actor
        self.mean = mean
        self.observation_normalizer = observation_normalizer
        self.action_normalizer = action_normalizer

    @torch.no_grad()
    def act(self, observations):
        normalized = self.observation_normalizer.normalize(observations)
        action = self.mean(self.actor(normalized))
        # Keep the original inverse normalization. Environment action clipping
        # and physical-limit handling still run through env.step(action).
        return self.action_normalizer.unnormalize(action)
