"""Own one Viewer; borrow it to a single environment after selection completes."""
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
import yaml

from RSP.sim_ready.pipelines.scenes.spec import parse_scene
from .profiles import load_profile, load_presets, read_yaml
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
    panel.attach(viewer)
    viewer.gui.set_selection_callback(panel.draw)
    try:
        while viewer.is_running() and panel.result is None and not panel.cancelled:
            viewer.begin_frame(0.)
            viewer.end_frame()
        return panel.result if viewer.is_running() and not panel.cancelled else None
    finally:
        viewer.gui.set_selection_callback(None)
        panel.detach()


@contextmanager
def simulation(profile, selection, viewer):
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
        env_config = dict(env_config, scene_preset=str(scene_path(selection, profile).resolve()))
        engine_config = dict(engine_config, playback_viewer=viewer)
        env = resource_util.construct(PBFPlayEnv, env_config, engine_config, 1, profile.device, True, False)
        agent = build_agent(str(profile.agent_config), env, profile.device)
        agent.load(str(profile.model_file))
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        yield env, agent
    finally:
        resource_util.cleanup(env.close if env is not None else None,
                              agent.close if agent is not None else None)


def run():
    import warp as wp
    from util import mp_util, resource_util, display
    from RSP.rendering.newton_gl import NewtonBillboardViewer
    from RSP.runtime.playback import PlaybackRunner

    profile = load_profile()
    choices = SceneChoices(read_yaml(profile.scene_preset))
    panel = SelectionPanel(choices, load_presets(), profile.scene_preset)
    viewer = None
    try:
        mp_util.init(0, 1, profile.device, None)
        with wp.ScopedDevice(profile.device):
            viewer = resource_util.construct(NewtonBillboardViewer, headless=False)
            selection = select_scene(viewer, panel)
            del panel, choices
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
