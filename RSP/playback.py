"""Continuous policy playback for the single-world PBF scene."""
import torch

from envs.pbf_play_env import PBFPlayEnv
from learning.base_agent import AgentMode


def run_playback(agent, env, *, num_steps=None):
    """Run until viewer exit/interrupt; an optional step budget supports checks.

    The caller owns construction and cleanup, as in MimicKit's train/test entry.
    TEST selects the existing deterministic actor and enables configured recording.
    """
    if not isinstance(env, PBFPlayEnv):
        raise ValueError("play mode requires PBFPlayEnv")
    if num_steps is not None and num_steps < 0:
        raise ValueError("num_steps must be non-negative or None")

    agent.eval()
    agent.set_mode(AgentMode.TEST)
    env.set_playback_mode(True)
    with torch.no_grad():
        obs, info = env.reset()
        step = 0
        while num_steps is None or step < num_steps:
            action, _ = agent._decide_action(obs, info)
            obs, _, _, info = env.step(action)
            step += 1
