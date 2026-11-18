"""Native G1 playback with the existing LaFAN initialization adapter."""
from envs.g1_lafan_env import G1LafanEnvMixin
from envs.pbf_play_env import PBFPlayEnv
from learning.agent_builder import load_agent_file
from learning.smp_g1_agent import SMPG1Agent
from util import resource_util


class G1PBFPlayEnv(G1LafanEnvMixin, PBFPlayEnv):
    def _get_char_color(self):
        return None  # Use authored shape colors rather than a character-wide tint.


def build_agent(agent_file, env, device):
    """The dodgeball subclass adds only a training-task guard, no saved state."""
    config = load_agent_file(agent_file)
    if config["agent_name"] not in {"SMPG1", "SMPG1Dodgeball"}:
        raise ValueError("G1 playback requires its SMPG1 policy configuration")
    return resource_util.construct(SMPG1Agent, config, env, device)
