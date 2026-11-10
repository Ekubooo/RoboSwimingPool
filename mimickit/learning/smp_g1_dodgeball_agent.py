"""Backward-compatible dodgeball entry point for shared G1 target protection."""
from learning.smp_g1_agent import SMPG1Agent, physical_target_loss, joint_limit_pressure


class SMPG1DodgeballAgent(SMPG1Agent):
    def __init__(self, config, env, device):
        if not hasattr(env, "_proj_ids"):
            raise ValueError("Use the opt-in G1 dodgeball position-target-limit environment.")
        super().__init__(config, env, device)
