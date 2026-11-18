"""Own one Viewer; borrow it to a single environment after selection completes."""
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
import yaml

from RSP.sim_ready.pipelines.scenes.spec import parse_scene
from .profiles import load_presets, read_yaml, load_characters
from .assembly import SceneChoices
from .window import SelectionPanel


def scene_path(selection, profile):
    """Presets remain in place; only custom scenes get a generated snapshot."""
    if isinstance(selection, Path):
        return selection
    parse_scene(selection)
    path = profile.out_dir / "scenes" / uuid4().hex / "scene.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(selection, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def select_scene(viewer, panel):
    """Finish the UI frame before any simulation construction."""
    from pyglet.event import EVENT_HANDLED

    window = viewer.renderer.window
    def cancel_close():
        panel.cancel()
        return EVENT_HANDLED  # Keep GL alive until the selection resources are gone.

    window.push_handlers(on_close=cancel_close)
    try:
        panel.attach(viewer)
        viewer.gui.set_selection_callback(panel.draw, cancel_callback=panel.cancel)
        while viewer.is_running() and panel.result is None and not panel.cancelled:
            panel.prepare_previews()
            viewer.begin_frame(0.)
            viewer.end_frame()
        return panel.result if viewer.is_running() and not panel.cancelled else None
    finally:
        viewer.gui.set_selection_callback(None)
        try:
            panel.detach()
        finally:
            window.remove_handlers(on_close=cancel_close)


@contextmanager
def simulation(profile, selection, viewer, *, visualize=True):
    """Retain MimicKit loading, GSI and cleanup; the caller owns the window."""
    from envs.env_builder import load_configs
    from envs.pbf_play_env import PBFPlayEnv
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import resource_util, util

    env = agent = None
    try:
        util.set_rand_seed(profile.seed)
        env_config, engine_config = load_configs(str(profile.env_config), str(profile.engine_config))
        record = read_yaml(selection) if isinstance(selection, Path) else selection
        if record["character"] != read_yaml(profile.scene_preset)["character"]:
            raise ValueError("Selected character does not match its playback profile")
        env_config = dict(env_config, scene_preset=str(scene_path(selection, profile).resolve()))
        engine_config = dict(engine_config, playback_viewer=viewer)
        env_type, agent_factory = PBFPlayEnv, build_agent
        if profile.profile_id == "g1":
            from RSP.runtime.g1_playback import G1PBFPlayEnv, build_agent as build_g1_agent
            env_type, agent_factory = G1PBFPlayEnv, build_g1_agent
            env_config = dict(env_config, env_name="pbf_play", enable_early_termination=False)
        env = resource_util.construct(env_type, env_config, engine_config, 1, profile.device, visualize, False)
        agent = agent_factory(str(profile.agent_config), env, profile.device)
        agent.load(str(profile.model_file))
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        yield env, agent
    finally:
        resource_util.cleanup(env.close if env is not None else None,
                              agent.close if agent is not None else None)


def run(*, profile_id="humanoid", direct=False):
    import warp as wp
    from util import mp_util, resource_util, display
    from RSP.rendering.newton_gl import NewtonBillboardViewer
    from RSP.runtime.playback import PlaybackRunner

    characters = load_characters()
    profile = characters[profile_id].require_available()
    viewer = None
    try:
        mp_util.init(0, 1, profile.device, None)
        with wp.ScopedDevice(profile.device):
            viewer = resource_util.construct(NewtonBillboardViewer, headless=False)
            selection = profile.scene_preset
            if not direct:
                choices = SceneChoices(read_yaml(profile.scene_preset))
                panel = SelectionPanel(choices, load_presets(), profile.scene_preset, characters=characters)
                request = select_scene(viewer, panel)
                del panel, choices
                if request is None:
                    return
                selected_profile = characters[request.profile_id].require_available()
                if selected_profile.device != profile.device:
                    raise ValueError("同窗选择需要人形配套使用相同设备")
                profile, selection = selected_profile, request.scene
            if selection is None:
                return
            viewer.show_loading_splash("Loading scene and policy...")
            viewer.begin_frame(0.)
            viewer.end_frame()
            if not viewer.is_running():
                return
            with simulation(profile, selection, viewer) as (env, agent):
                viewer.hide_loading_splash()
                viewer.set_paused(False)
                PlaybackRunner(env, agent).run()
    finally:
        resource_util.cleanup(lambda: resource_util.close_viewer(viewer),
                              mp_util.shutdown, display.close_virtual_display)
