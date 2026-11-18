"""Selection requests, frame boundaries and borrowed viewer ownership."""
from pathlib import Path
import sys
import unittest
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'mimickit')]
from RSP.launcher.profiles import load_profile, load_presets, read_yaml
from RSP.launcher.assembly import SceneChoices
from RSP.launcher.window import SelectionPanel
from RSP.launcher.application import select_scene
from RSP.launcher import application
from RSP.rendering.playback_ui import PlaybackGui
from engines.newton_pbf_engine import Presentation


class SelectionChecks(unittest.TestCase):
    def panel(self):
        profile = load_profile()
        return SelectionPanel(SceneChoices(read_yaml(profile.scene_preset)), load_presets(), profile.scene_preset)

    def test_presets_and_custom_requests(self):
        panel = self.panel()
        for name, path in panel.presets.items():
            panel.preset = name
            panel.request_start()
            self.assertEqual(panel.result, path)
        panel.mode = 'custom'
        for p in panel.props.values():
            p.enabled = False
        panel.invalidate()
        panel.request_start()
        self.assertEqual(panel.result['instances'], [])
        for p in panel.props.values():
            p.enabled, p.count = True, 3
        panel.invalidate()
        self.assertIsNone(panel.layout)
        panel.placement = 'random'
        self.assertTrue(panel.generate())
        layout = panel.layout
        panel.request_start()
        self.assertEqual(panel.result, layout)
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
            viewer.gui.set_selection_callback.assert_called_with(None)
            panel.detach.assert_called_once()

    def test_selection_has_no_playback_keys(self):
        from pyglet.window import key
        base = Mock()
        gui = PlaybackGui(base)
        gui.set_selection_callback(Mock())
        for symbol in (key.SPACE, key.ENTER, key.R):
            gui.handle_key_press(symbol)
        base._viewer.set_paused.assert_not_called()
        base._viewer.request_step.assert_not_called()
        base.handle_key_press.assert_not_called()

    def test_borrowed_and_owned_viewer_close(self):
        viewer = Mock()
        with patch('engines.newton_pbf_engine.resource_util.construct', return_value=viewer) as factory, \
             patch('engines.newton_pbf_engine.resource_util.close_viewer') as close:
            presentation = Presentation(1, 1., True, False, viewer=viewer)
            factory.assert_not_called()
            self.assertIs(presentation.get_viewer(), viewer)
            presentation.close()
            presentation.close()
            close.assert_not_called()
            owned = Presentation(1, 1., True, False)
            factory.assert_called_once()
            owned.close()
            owned.close()
            close.assert_called_once_with(viewer)

    def test_application_closes_on_cancel_or_load_failure(self):
        profile = replace(load_profile(), device='cpu')
        for selection in (None, profile.scene_preset):
            viewer = Mock()
            viewer.is_running.return_value = True
            @contextmanager
            def fail(*args):
                raise ValueError('load failed')
                yield
            with patch.object(application, 'load_profile', return_value=profile), \
                 patch.object(application, 'select_scene', return_value=selection), \
                 patch.object(application, 'simulation', fail), \
                 patch('util.resource_util.construct', return_value=viewer) as create, \
                 patch('util.resource_util.close_viewer') as close, \
                 patch('util.mp_util.init'), patch('util.mp_util.shutdown') as shutdown:
                if selection is None:
                    application.run()
                else:
                    with self.assertRaisesRegex(ValueError, 'load failed'):
                        application.run()
                create.assert_called_once()
                close.assert_called_once_with(viewer)
                shutdown.assert_called_once()


if __name__ == '__main__':
    unittest.main()
