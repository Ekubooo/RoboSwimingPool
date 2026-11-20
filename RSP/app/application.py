"""Own one Viewer; borrow it to a single environment after selection completes."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import yaml

from RSP.integrations.mimickit.profiles import load_presets, read_yaml, load_characters
from RSP.simulation.scenes.spec import parse_scene
from RSP.simulation.scenes.assembly import SceneAssembly
from RSP.simulation.scenes.layout import ScenePlacementError

from .config import POLICY_HZ, boundary_support_radius, get_fluid_settings
from .ui.selection import SelectionPanel


class _StartupCancelled(Exception):
    """The window closed while assets were being prepared."""


def scene_path(selection, profile):
    """Presets remain in place; only custom scenes get a generated snapshot."""
    if isinstance(selection, Path):
        return selection
    spec = parse_scene(selection)
    record = deepcopy(selection)
    record["container"] = asdict(spec.container)
    record["character"] = asdict(spec.character)
    if spec.terrain_prop is not None:
        record["terrain_prop"] = asdict(spec.terrain_prop)
    for item, instance in zip(record["instances"], spec.instances, strict=True):
        item["asset"] = asdict(instance.asset)
    path = profile.out_dir / "scenes" / uuid4().hex / "scene.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(record, sort_keys=False, allow_unicode=True), encoding="utf-8")
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
        gui.set_selection_callback(panel.draw, cancel_callback=panel.handle_escape)
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
def open_playback(profile, selection, viewer, *, visualize=True, asset_progress=None):
    """Validate the selection before opening its matching policy session."""
    from RSP.integrations.mimickit.session import open_session

    record = read_yaml(selection) if isinstance(selection, Path) else selection
    if parse_scene(record).character != parse_scene(read_yaml(profile.scene_preset)).character:
        raise ValueError("Selected character does not match its playback profile")
    path = scene_path(selection, profile).resolve()
    with open_session(profile, path, viewer, visualize=visualize,
                      fluid_settings=get_fluid_settings(), policy_hz=POLICY_HZ,
                      asset_progress=asset_progress) as session:
        yield session


def run(*, profile_id="humanoid", direct=False):
    import torch
    import warp as wp

    from RSP.integrations.mimickit.session import playback_process
    from RSP.rendering.newton_gl import open_viewer
    from RSP.sim_ready.catalog import scan_sources
    from RSP.sim_ready.config import BakeOptions
    from RSP.sim_ready.paths import configured_runtime_root
    from RSP.sim_ready.pipelines.runtime import prepare_sources
    from RSP.sim_ready.runtime_store import AssetPreparationError

    from .playback import bind_playback

    sources = scan_sources()
    characters = load_characters(sources=sources)
    profile = characters[profile_id].require_available()
    with playback_process(profile.device), wp.ScopedDevice(profile.device), open_viewer(headless=False) as viewer:
        def show_progress(message):
            if not viewer.is_running():
                raise _StartupCancelled()
            viewer.show_loading_splash(message)
            viewer.begin_frame(0.)
            viewer.end_frame()
            if not viewer.is_running():
                raise _StartupCancelled()

        try:
            prepared = prepare_sources(sources=sources, device=profile.device,
                support_radius=max(BakeOptions().max_support_radius, boundary_support_radius()),
                root=configured_runtime_root(read_yaml(profile.engine_config).get("pbf_asset_dir")),
                progress=show_progress)
            show_progress("Asset checks complete. Opening scene selection...")
        except _StartupCancelled:
            return
        for error in prepared.errors:
            print(f"Asset unavailable: {error}", flush=True)
        panel = None
        if not direct:
            panel = SelectionPanel(SceneAssembly(read_yaml(profile.scene_preset), sources=prepared.sources,
                                                prepared_assets=prepared.assets),
                                   load_presets(), profile.scene_preset, characters=characters)
            viewer.hide_loading_splash()

        while True:
            selection = profile.scene_preset
            if panel is not None:
                request = select_scene(viewer, panel)
                if request is None:
                    return
                selected_profile = characters[request.profile_id].require_available()
                if selected_profile.device != profile.device:
                    raise ValueError("Humanoid profiles must use the same device to share a window")
                profile, selection = selected_profile, request.scene
            try:
                show_progress("Checking scene assets...")
                with open_playback(profile, selection, viewer, asset_progress=show_progress) as session:
                    viewer.hide_loading_splash()
                    viewer.set_paused(False)
                    with bind_playback(session) as runner:
                        # Keep the selection available until the first sampled pose fits.
                        with torch.no_grad():
                            runner.reset()
                        panel = None
                        runner.run(reset=False)
            except _StartupCancelled:
                return
            except (AssetPreparationError, ScenePlacementError) as error:
                if panel is None:
                    raise
                if not viewer.is_running():
                    return
                viewer.set_model(None)
                viewer.hide_loading_splash()
                panel.resume_after_failure(error)
                continue
            return
