"""Own one Viewer; borrow it to a single environment after selection completes."""
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import yaml

from RSP.integrations.mimickit.profiles import load_presets, read_yaml, load_characters
from RSP.simulation.scenes.spec import parse_scene

from .config import POLICY_HZ, get_fluid_settings
from .ui.scene_choices import SceneChoices
from .ui.selection import SelectionPanel


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

    gui = viewer.install_gui()
    if gui is None:
        raise RuntimeError("Scene selection requires the Newton ImGui interface")
    window = viewer.renderer.window

    def cancel_close():
        panel.cancel()
        return EVENT_HANDLED  # Keep GL alive until the selection resources are gone.

    window.push_handlers(on_close=cancel_close)
    try:
        panel.attach(viewer)
        gui.set_selection_callback(panel.draw, cancel_callback=panel.cancel)
        while viewer.is_running() and panel.result is None and not panel.cancelled:
            panel.prepare_previews()
            viewer.begin_frame(0.)
            viewer.end_frame()
        return panel.result if viewer.is_running() and not panel.cancelled else None
    finally:
        try:
            try:
                gui.set_selection_callback(None)
            finally:
                panel.detach()
        finally:
            window.remove_handlers(on_close=cancel_close)


@contextmanager
def open_playback(profile, selection, viewer, *, visualize=True):
    """Validate the selection before opening its matching policy session."""
    from RSP.integrations.mimickit.session import open_session

    record = read_yaml(selection) if isinstance(selection, Path) else selection
    if record["character"] != read_yaml(profile.scene_preset)["character"]:
        raise ValueError("Selected character does not match its playback profile")
    path = scene_path(selection, profile).resolve()
    with open_session(profile, path, viewer, visualize=visualize,
                      fluid_settings=get_fluid_settings(), policy_hz=POLICY_HZ) as session:
        yield session


def run(*, profile_id="humanoid", direct=False):
    import warp as wp

    from RSP.integrations.mimickit.session import playback_process
    from RSP.rendering.newton_gl import open_viewer

    from .playback import bind_playback

    characters = load_characters()
    profile = characters[profile_id].require_available()
    with playback_process(profile.device), wp.ScopedDevice(profile.device), open_viewer(headless=False) as viewer:
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
                raise ValueError("Humanoid profiles must use the same device to share a window")
            profile, selection = selected_profile, request.scene
        if selection is None:
            return
        viewer.show_loading_splash("Loading scene and policy...")
        viewer.begin_frame(0.)
        viewer.end_frame()
        if not viewer.is_running():
            return
        with open_playback(profile, selection, viewer) as session, bind_playback(session) as runner:
            viewer.hide_loading_splash()
            viewer.set_paused(False)
            runner.run()
