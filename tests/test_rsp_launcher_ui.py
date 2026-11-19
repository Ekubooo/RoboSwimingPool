"""Selection requests, frame boundaries and borrowed viewer ownership."""
from pathlib import Path
import sys
import unittest
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock, patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'mimickit')]
from RSP.integrations.mimickit.profiles import load_profile, load_presets, read_yaml, load_characters, LaunchRequest
from RSP.app.ui.scene_choices import SceneChoices
from RSP.app.ui.selection import SelectionPanel
from RSP.app.application import select_scene
from RSP.app import application
from RSP.rendering.newton_gl import NewtonBillboardViewer
from RSP.rendering.newton_gui import NewtonGuiHost
from RSP.rendering.presentation import Presentation
from RSP.integrations.mimickit.recorder import EnginePresentation


class SelectionChecks(unittest.TestCase):
    def panel(self):
        profile = load_profile()
        return SelectionPanel(SceneChoices(read_yaml(profile.scene_preset)), load_presets(), profile.scene_preset)

    def test_presets_and_custom_requests(self):
        panel = self.panel()
        for name, path in panel.presets.items():
            panel.preset = name
            panel.request_start()
            self.assertEqual(panel.result.scene, path)
            self.assertEqual(panel.result.profile_id, 'g1' if name == 'G1 demo' else 'humanoid')
        panel.mode = 'custom'
        for p in panel.props.values():
            p.enabled = False
        panel.invalidate()
        panel.request_start()
        self.assertEqual(panel.result.scene['instances'], [])
        for p in panel.props.values():
            p.enabled, p.count = True, 3
        panel.invalidate()
        self.assertIsNone(panel.layout)
        panel.placement = 'random'
        self.assertTrue(panel.generate())
        layout = panel.layout
        panel.request_start()
        self.assertEqual(panel.result.scene, layout)
        self.assertEqual(len(layout['instances']), 9)

    def test_selection_clears_after_complete_frame_and_close(self):
        for start in (True, False):
            viewer = Mock()
            viewer.is_running.return_value = True
            panel = Mock(result=None, cancelled=False)
            panel.result = None
            def frame():
                if start:
                    panel.result = 'selected'
                else:
                    viewer.is_running.return_value = False
            viewer.end_frame.side_effect = frame
            result = select_scene(viewer, panel)
            self.assertEqual(result, 'selected' if start else None)
            viewer.end_frame.assert_called_once()
            viewer.install_gui.return_value.set_selection_callback.assert_called_with(None)
            panel.detach.assert_called_once()

    def test_selection_has_no_playback_keys(self):
        from pyglet.window import key
        base = Mock()
        gui = NewtonGuiHost(base, Mock())
        gui.set_selection_callback(Mock())
        for symbol in (key.SPACE, key.ENTER, key.R):
            gui.handle_key_press(symbol)
        base._viewer.set_paused.assert_not_called()
        base._viewer.request_step.assert_not_called()
        base.handle_key_press.assert_not_called()

    def test_borrowed_and_owned_viewer_close(self):
        viewer = Mock()
        with patch('RSP.integrations.mimickit.recorder.resource_util.construct', return_value=viewer) as factory:
            presentation = EnginePresentation(1, 1., True, False, viewer=viewer)
            factory.assert_not_called()
            self.assertIs(presentation.get_viewer(), viewer)
            presentation.close()
            presentation.close()
            viewer.close.assert_not_called()
            owned = EnginePresentation(1, 1., True, False)
            factory.assert_called_once()
            owned.close()
            owned.close()
            viewer.close.assert_called_once_with()

    def test_borrowed_viewer_routes_keys_only_to_the_current_presentation(self):
        viewer = NewtonBillboardViewer.__new__(NewtonBillboardViewer)
        viewer._keyboard_binding = None
        viewer.renderer = Mock()
        viewer.gui = Mock()
        old_key, new_key = Mock(), Mock()

        with patch("RSP.rendering.presentation.str_to_key_code", return_value=42), \
             patch.object(viewer, "close") as close:
            old = Presentation(1, 1., True, viewer=viewer)
            old.register_keyboard_callback("r", old_key)
            viewer.on_key_press(42, 0)
            old_key.assert_called_once_with()

            current = Presentation(1, 1., True, viewer=viewer)
            current.register_keyboard_callback("r", new_key)
            old_ref = weakref.ref(old)
            old.close()
            del old
            self.assertIsNone(old_ref())
            viewer.on_key_press(42, 0)
            old_key.assert_called_once_with()
            new_key.assert_called_once_with()

            current_ref = weakref.ref(current)
            current.close()
            del current
            self.assertIsNone(current_ref())
            viewer.on_key_press(42, 0)
            new_key.assert_called_once_with()
            self.assertEqual(viewer.gui.handle_key_press.call_count, 3)
            viewer.renderer.register_key_press.assert_not_called()
            close.assert_not_called()

    def test_application_closes_on_cancel_or_load_failure(self):
        profile = replace(load_profile(), device='cpu')
        characters = load_characters()
        characters['humanoid'] = replace(characters['humanoid'], profile=profile)
        for selection in (None, LaunchRequest('humanoid', profile.scene_preset)):
            viewer = Mock()
            viewer.is_running.return_value = True
            @contextmanager
            def fail(*args):
                raise ValueError('load failed')
                yield
            @contextmanager
            def create(**options):
                try:
                    yield viewer
                finally:
                    viewer.close()
            with patch.object(application, 'load_characters', return_value=characters), \
                 patch.object(application, 'select_scene', return_value=selection), \
                 patch.object(application, 'open_playback', fail), \
                 patch('RSP.rendering.newton_gl.open_viewer', side_effect=create) as open_window, \
                 patch('util.mp_util.init'), patch('util.mp_util.shutdown') as shutdown:
                if selection is None:
                    application.run()
                else:
                    with self.assertRaisesRegex(ValueError, 'load failed'):
                        application.run()
                open_window.assert_called_once()
                viewer.close.assert_called_once_with()
                shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()
