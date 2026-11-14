"""Deterministic playback using the already-loaded MimicKit actor and scalers.

Loading, GSI initialization and ownership remain with the original agent.
Only its per-frame training distribution/statistics path is bypassed.
"""
import torch


class PolicyInference:
    def __init__(self, agent):
        from learning.distribution_gaussian_diag import DistributionGaussianDiagBuilder

        model = agent._model
        if not isinstance(model._action_dist, DistributionGaussianDiagBuilder):
            raise ValueError("Playback inference requires the existing continuous Gaussian actor")
        self.actor = model._actor_layers
        self.mean = model._action_dist._mean_net
        self.observation_normalizer = agent._obs_norm
        self.action_normalizer = agent._a_norm

    @torch.no_grad()
    def act(self, observations):
        normalized = self.observation_normalizer.normalize(observations)
        action = self.mean(self.actor(normalized))
        # Keep the original inverse normalization. Environment action clipping
        # and physical-limit handling still run through env.step(action).
        return self.action_normalizer.unnormalize(action)
