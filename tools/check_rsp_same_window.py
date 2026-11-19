"""Short real-GL selection-to-playback check; no recording or asset preparation."""
from pathlib import Path
import argparse
import os
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--ui-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    output = ROOT / "output/same_window_checks"
    output.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(output)
    import warp as wp
    wp.config.kernel_cache_dir = str(ROOT / "output/scene_checks/warp_cache")
    import torch
    from util import mp_util, resource_util
    from RSP.app.application import open_playback, select_scene
    from RSP.app.ui.selection import SelectionPanel
    from RSP.app.ui.scene_choices import SceneChoices
    from RSP.integrations.mimickit.profiles import load_profile, load_presets, read_yaml
    from RSP.rendering.newton_gl import NewtonBillboardViewer
    from RSP.app.playback import bind_playback

    profile = load_profile()
    panel = SelectionPanel(SceneChoices(read_yaml(profile.scene_preset)), load_presets(), profile.scene_preset)
    if args.custom:
        panel.mode = "custom"
        for prop in panel.props.values():
            prop.count = 3
        assert panel.generate()
    viewer = None
    mp_util.init(0, 1, profile.device, None)
    try:
        with wp.ScopedDevice(profile.device):
            viewer = NewtonBillboardViewer(width=1100, height=800, headless=False)
            viewer.ui.io.set_ini_filename(None)
            window, context = viewer.renderer.window, viewer.renderer.window.context
            # Drive the real draw callback, then submit the same request as the button.
            draw = panel.draw
            frames = 0
            def select(imgui):
                nonlocal frames
                draw(imgui)
                frames += 1
                if frames == 3:
                    panel.request_start()
            panel.draw = select
            request = select_scene(viewer, panel)
            assert frames == 3 and viewer.gui.selection_callback is None
            if args.ui_only:
                print("PASS: real ImGui draw, font, selection and callback removal")
                return
            assert request.profile_id == profile.profile_id
            with open_playback(profile, request.scene, viewer) as session, bind_playback(session) as runner:
                env, agent = session.env, session.agent
                assert env._engine.get_viewer() is viewer
                assert viewer.renderer.window is window and window.context is context
                runner.reset()
                assert runner.policy_interval == round(read_yaml(profile.engine_config)["control_freq"] / session.policy_hz)
                count = 2 if args.custom else 11
                with patch.object(runner.policy, "act", wraps=runner.policy.act) as act:
                    for _ in range(count):
                        assert runner.tick()
                    assert act.call_count == (count + runner.policy_interval - 1) // runner.policy_interval
                assert torch.isfinite(runner.obs).all()
                runner.reset()
                assert torch.all(env._time_buf == 0)
            assert viewer.is_running() and window.context is context
            print(f"PASS: same viewer/window/context, {count} frames, cadence, reset, borrowed lifetime")
    finally:
        resource_util.cleanup(lambda: resource_util.close_viewer(viewer), mp_util.shutdown)


if __name__ == "__main__":
    main()
