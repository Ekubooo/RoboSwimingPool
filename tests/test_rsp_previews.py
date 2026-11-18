"""Focused preview geometry, texture ownership and cancellation checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'mimickit')]
from RSP.sim_ready.catalog import scan_published, AssetRef
from RSP.launcher.preview_assets import load_geometry, project_geometry
from RSP.rendering.asset_preview import AssetPreviews
from RSP.rendering.playback_ui import PlaybackGui


class PreviewChecks(unittest.TestCase):
    def test_geometry_and_camera_fit(self):
        assets = scan_published()
        refs = [AssetRef('articulations', 'g1', '1'), AssetRef('articulations', 'humanoid', '2'),
                AssetRef('containers', 'pbf_pool', '2'), AssetRef('props', 'light_box', '2')]
        for ref in refs:
            geometry = load_geometry(assets[ref])
            triangles, lines = project_geometry(geometry)
            points = np.concatenate((triangles[:, :3], lines[:, :3]))
            self.assertTrue(np.isfinite(points).all())
            self.assertLessEqual(np.abs(points).max(), 1.)
            self.assertGreater(len(triangles), 0)
            if ref.category == 'containers':
                self.assertEqual(len(lines), 24)

    def test_failure_is_cached_and_close_cancels_pending(self):
        ref = AssetRef('props', 'absent', '1')
        previews = AssetPreviews({})
        previews.request(ref)
        with self.assertLogs(level='ERROR'):
            previews.process_one()
        self.assertIsNotNone(previews.request(ref)[1])
        self.assertFalse(previews.pending)
        previews.close()
        previews.close()
        previews.request(AssetRef('props', 'another', '1'))
        self.assertFalse(previews.pending)

    def test_escape_requests_cancel_without_destroying_context(self):
        from pyglet.window import key
        gui, cancel, close = PlaybackGui(Mock()), Mock(), Mock()
        gui.set_selection_callback(Mock(), cancel_callback=cancel)
        gui.handle_key_press(key.ESCAPE, close_fn=close)
        cancel.assert_called_once()
        close.assert_not_called()
        gui.set_selection_callback(None)
        self.assertIsNone(gui.selection_cancel)

    def test_system_close_is_deferred_and_handlers_removed(self):
        from RSP.launcher.application import select_scene
        viewer = Mock()
        viewer.is_running.return_value = True
        panel = Mock(result=None, cancelled=False)
        panel.result = None
        panel.cancel.side_effect = lambda: setattr(panel, 'cancelled', True)
        handlers = {}
        viewer.renderer.window.push_handlers.side_effect = lambda **kw: handlers.update(kw)
        viewer.end_frame.side_effect = lambda: handlers['on_close']()
        self.assertIsNone(select_scene(viewer, panel))
        panel.detach.assert_called_once()
        viewer.close.assert_not_called()
        viewer.renderer.window.remove_handlers.assert_called_once_with(**handlers)

    def test_attach_failure_still_detaches(self):
        from RSP.launcher.application import select_scene
        viewer, panel = Mock(), Mock()
        panel.attach.side_effect = RuntimeError('font/setup failure')
        with self.assertRaisesRegex(RuntimeError, 'font/setup'):
            select_scene(viewer, panel)
        panel.detach.assert_called_once()
        viewer.renderer.window.remove_handlers.assert_called_once()

    def test_real_gl_cards_and_release(self):
        import ctypes
        from pyglet import gl
        from RSP.rendering.newton_gl import NewtonBillboardViewer
        from RSP.launcher.window import SelectionPanel
        from RSP.launcher.assembly import SceneChoices
        from RSP.launcher.profiles import load_profile, load_presets, read_yaml
        viewer = NewtonBillboardViewer(width=1000, height=800, vsync=False)
        viewer.renderer.window.set_visible(False)
        profile = load_profile()
        panel = SelectionPanel(SceneChoices(read_yaml(profile.scene_preset)), load_presets(), profile.scene_preset)
        textures = []
        try:
            panel.attach(viewer)
            viewer.gui.set_selection_callback(panel.draw, cancel_callback=panel.cancel)
            for mode in ('preset', 'custom'):
                panel.mode = mode
                for _ in range(16):
                    panel.prepare_previews()
                    viewer.begin_frame(0.)
                    viewer.end_frame()
            self.assertFalse(panel.previews.errors, panel.previews.errors)
            textures = list(panel.previews.textures.values())
            self.assertGreaterEqual(len(textures), 5)
            for texture in textures:
                self.assertTrue(gl.glIsTexture(texture))
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
                pixels = (gl.GLubyte*(512*384*4))()
                gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels)
                self.assertGreater(np.asarray(pixels).reshape(-1, 4)[:, :3].std(), 5.)
            self.assertIsNone(panel.result)  # Drawing cards never launches a simulation.
            # A shader failure after texture/FBO allocation must delete its texture.
            deleted = []
            original_delete = gl.glDeleteTextures
            def record_delete(count, pointer):
                deleted.append(ctypes.cast(pointer, ctypes.POINTER(gl.GLuint))[0])
                original_delete(count, pointer)
            empty = np.empty((0, 6), dtype=np.float32)
            with patch('pyglet.graphics.shader.Shader', side_effect=RuntimeError('shader failed')), \
                    patch.object(gl, 'glDeleteTextures', side_effect=record_delete):
                with self.assertRaisesRegex(RuntimeError, 'shader failed'):
                    panel.previews._render(empty, empty)
            self.assertEqual(len(deleted), 1)
            self.assertFalse(gl.glIsTexture(deleted[0]))
        finally:
            viewer.gui.set_selection_callback(None)
            panel.detach()
            for texture in textures:
                self.assertFalse(gl.glIsTexture(texture))
            viewer.close()


if __name__ == '__main__':
    unittest.main()
