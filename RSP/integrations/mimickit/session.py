"""Load existing MimicKit policies and expose their active playback session."""

from contextlib import contextmanager
from pathlib import Path

from .inference import PolicyInference


class MimicKitSession:
    """Borrow the environment, agent and their persistent observation buffers."""

    def __init__(self, env, agent, *, policy=None, policy_hz=30):
        self._env = env
        self._agent = agent
        engine = env._engine
        self._engine = engine
        self.device = env._device
        self.frame_dt = engine.get_timestep()
        self.policy_hz = policy_hz
        self.viewer = engine.get_viewer()
        self.policy = PolicyInference(**agent.get_playback_policy_components()) if policy is None else policy
        self.set_fluid_controls = getattr(engine, "set_fluid_controls", None)
        get_scene = getattr(engine, "get_play_scene", None)
        scene = get_scene() if get_scene is not None else None
        self.fluid_settings = getattr(scene, "fluid_settings", None)
        self.fluid_gravity = getattr(getattr(scene, "config", None), "gravity", None)

    @property
    def env(self):
        """Expose the environment to diagnostics."""
        return self._env

    @property
    def agent(self):
        """Expose the loaded agent to diagnostics."""
        return self._agent

    @property
    def obs(self):
        return self._env._obs_buf

    @property
    def info(self):
        return self._env._info

    def reset(self):
        return self._env.reset()

    def step(self, action):
        obs, _, _, info = self._env.step(action)
        return obs, info

    def render_only(self):
        self._env.render_only()

    def prepare_step(self):
        self._engine.prepare_playback_step()

    def request_reset(self):
        self._env.request_reset()

    def consume_reset_request(self):
        return self._env.consume_reset_request()

    def apply_commands(self):
        self._env.apply_playback_commands()

    def get_policy_use_scene_sphere(self):
        return self._env.get_policy_use_scene_sphere()

    def request_policy_use_scene_sphere(self, enabled):
        self._env.request_policy_use_scene_sphere(enabled)


def configure_test_agent(agent):
    from learning.base_agent import AgentMode

    agent.eval()
    agent.set_mode(AgentMode.TEST)


@contextmanager
def playback_process(device):
    """Own MimicKit's share of process and display resources."""
    from util import display, mp_util, resource_util

    try:
        mp_util.init(0, 1, device, None)
        yield
    finally:
        resource_util.cleanup(mp_util.shutdown, display.close_virtual_display)


@contextmanager
def open_session(profile, scene_path, viewer, *, visualize=True, fluid_settings=None, policy_hz=30):
    """Build and release the environment and agent while borrowing a window."""
    from envs.env_builder import load_configs
    from learning.agent_builder import build_agent
    from util import resource_util, util

    from .environment import PBFPlayEnv

    env = agent = None
    try:
        util.set_rand_seed(profile.seed)
        env_config, engine_config = load_configs(str(profile.env_config), str(profile.engine_config))
        env_config = dict(env_config, scene_preset=str(Path(scene_path).resolve()), playback_policy_hz=policy_hz)
        engine_config = dict(engine_config, playback_viewer=viewer, fluid_settings=fluid_settings)
        env_type, agent_factory = PBFPlayEnv, build_agent
        if profile.profile_id == "g1":
            from .g1_playback import G1PBFPlayEnv, build_agent as build_g1_agent

            env_type, agent_factory = G1PBFPlayEnv, build_g1_agent
            env_config = dict(env_config, env_name="pbf_play", enable_early_termination=False)
        env = resource_util.construct(env_type, env_config, engine_config, 1, profile.device, visualize, False)
        agent = agent_factory(str(profile.agent_config), env, profile.device)
        agent.load(str(profile.model_file))
        configure_test_agent(agent)
        yield MimicKitSession(env, agent, policy_hz=policy_hz)
    finally:
        resource_util.cleanup(env.close if env is not None else None,
                              agent.close if agent is not None else None)
