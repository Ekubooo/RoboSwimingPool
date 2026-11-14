"""Real CUDA/OpenGL checks for billboard data, depth, lifecycle and Newton UI.

Run in the WarpSim environment:
  python -m unittest discover -s tests -p test_rsp_rendering.py -v
"""

import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/rsp_render_checks"
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "tmp").mkdir(exist_ok=True)
tempfile.tempdir = str(OUTPUT / "tmp")
wp.config.kernel_cache_dir = str(ROOT / "output/newton_pbf_refactor_checks/warp_cache")

import newton
from pyglet import app, gl
from RSP.rendering.billboard import BillboardBatch, _gl_integer
from RSP.rendering.newton_gl import NewtonBillboardViewer


def particle_model(positions=((0., 0., 0.),), velocities=None, occluder=False):
    builder = newton.ModelBuilder()
    if velocities is None:
        velocities = [(0., 0., 0.)] * len(positions)
    for position, velocity in zip(positions, velocities):
        builder.add_particle(pos=position, vel=velocity, mass=1., radius=.4)
    if occluder:
        builder.add_shape_box(body=-1, xform=wp.transform((0., -.6, 0.), wp.quat_identity()),
                              hx=.2, hy=.1, hz=.2)
    return builder.finalize(device="cuda:0")


class BillboardChecks(unittest.TestCase):
    def setUp(self):
        # Pyglet's process-global exit flag survives closing the previous test's
        # last window. Each case starts a fresh viewer session.
        app.event_loop.has_exit = False
        self.viewer = NewtonBillboardViewer(width=192, height=192, headless=True, vsync=False)
        self.viewer.renderer.draw_sky = False
        self.viewer.renderer.sky_upper = (0., 0., 0.)

    def tearDown(self):
        self.viewer.close()
        self.viewer.close()

    def setup_model(self, model):
        self.viewer.set_model(model)
        self.viewer.show_particles = True
        self.viewer.show_visual = True
        self.viewer.camera.pos = np.array((0., -5., 0.), dtype=np.float32)
        self.viewer.camera.look_at((0., 0., 0.))
        self.state = model.state()
        return self.state

    def frame(self, update=True):
        self.viewer.begin_frame(0.)
        if update:
            self.viewer.log_state(self.state)
        self.viewer.end_frame()
        return self.viewer.get_frame().numpy()

    def test_palette_input_ownership_and_stable_buffer(self):
        speeds = (0., 1., 2., 4.25, 6.5, 20.)
        model = particle_model([(i * .9 - 2.25, 0., 0.) for i in range(6)],
                               [(speed, 0., 0.) for speed in speeds])
        self.setup_model(model)
        originals = [a.numpy().copy() for a in (self.state.particle_q, self.state.particle_qd, model.particle_radius)]
        self.frame()
        batch = self.viewer.objects["/model/particles"].batch
        buffer_id = batch._instances.value
        registration_id = id(batch._registered)
        with patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected GPU readback")):
            batch.update(self.state.particle_q, self.state.particle_qd, model.particle_radius)
        self.assertEqual(buffer_id, batch._instances.value)
        self.assertEqual(registration_id, id(batch._registered))
        packed = np.empty((6, 8), dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, batch._instances)
        gl.glGetBufferSubData(gl.GL_ARRAY_BUFFER, 0, packed.nbytes, packed.ctypes.data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        np.testing.assert_allclose(packed[:, :3], originals[0])
        np.testing.assert_allclose(packed[:, 3], originals[2])
        expected_rgb = np.array([[29, 119, 231], [94.5, 170, 243], [160, 221, 255],
                                 [207.5, 238, 255], [255, 255, 255], [255, 255, 255]]) / 255
        np.testing.assert_allclose(packed[:, 4:7], expected_rgb, atol=1e-7)
        np.testing.assert_equal(packed[:, 7], 1.)
        for source, original in zip((self.state.particle_q, self.state.particle_qd, model.particle_radius), originals):
            np.testing.assert_array_equal(source.numpy(), original)

    def test_depth_occlusion_and_visibility(self):
        self.setup_model(particle_model(occluder=True))
        behind = self.frame()
        blue = np.array([29, 119, 231], dtype=np.uint8)
        self.assertFalse(np.array_equal(behind[96, 96], blue), "Rigid must hide the rear particle")
        self.state.particle_q.assign(np.array([[0., -1., 0.]], dtype=np.float32))
        front = self.frame()
        np.testing.assert_array_equal(front[96, 96], blue)
        self.viewer.show_particles = False
        hidden = self.frame()
        self.assertFalse(np.array_equal(hidden[96, 96], blue))
        self.viewer.show_particles = True
        self.viewer.layer.visible = False
        self.assertFalse(np.array_equal(self.frame()[96, 96], blue))
        self.viewer.layer.visible = True
        np.testing.assert_array_equal(self.frame()[96, 96], blue)

    def test_camera_rotation_without_particle_update(self):
        self.setup_model(particle_model())
        first = self.frame()
        batch = self.viewer.objects["/model/particles"].batch
        self.viewer.camera.pos = np.array((5., 0., 0.), dtype=np.float32)
        self.viewer.camera.look_at((0., 0., 0.))
        with patch.object(batch, "update", side_effect=AssertionError("Paused redraw updated particles")):
            rotated = self.frame(update=False)
        for pixels in (first, rotated):
            rows, columns = np.nonzero(np.all(pixels == [29, 119, 231], axis=2))
            self.assertGreater(len(rows), 100)
            self.assertLessEqual(abs(np.ptp(rows) - np.ptp(columns)), 1)
        np.testing.assert_array_equal(first[96, 96], rotated[96, 96])

    def test_layer_transform_keeps_simulation_positions_unchanged(self):
        self.setup_model(particle_model())
        first = self.frame()
        self.viewer.layer.xform = wp.transform((1., 0., 0.), wp.quat_identity())
        shifted = self.frame()
        self.assertFalse(np.array_equal(first[96, 96], shifted[96, 96]))
        self.assertGreater(np.count_nonzero(np.all(shifted == [29, 119, 231], axis=2)), 100)
        np.testing.assert_array_equal(self.state.particle_q.numpy(), [[0., 0., 0.]])

    def test_failed_update_unmaps_the_buffer_for_reuse(self):
        self.setup_model(particle_model())
        self.frame()
        batch = self.viewer.objects["/model/particles"].batch
        with patch.object(wp, "launch", side_effect=RuntimeError("Test launch failure")):
            with self.assertRaisesRegex(RuntimeError, "Test launch failure"):
                batch.update(self.state.particle_q, self.state.particle_qd, self.viewer.model.particle_radius)
        self.assertEqual(batch.count, 0)
        # A subsequent real map/update/draw must work after the failed launch.
        np.testing.assert_array_equal(self.frame()[96, 96], [29, 119, 231])

    def test_resize_empty_model_switch_and_close(self):
        model = particle_model()
        self.setup_model(model)
        self.frame()
        batch = self.viewer.objects["/model/particles"].batch
        for count in (3, 0, 2):
            points = wp.zeros(count, dtype=wp.vec3, device="cuda:0")
            radii = wp.full(count, .2, dtype=float, device="cuda:0")
            batch.update(points, points, radii)
            self.assertEqual(batch.count, count)
            self.frame(update=False)
        self.setup_model(particle_model())
        self.assertTrue(batch._closed)
        self.frame()
        replacement = self.viewer.objects["/model/particles"].batch
        self.assertIsNot(replacement, batch)
        self.viewer.set_model(None)
        self.assertTrue(replacement._closed)
        self.assertNotIn("/model/particles", self.viewer.objects)

    def test_single_opaque_pass_and_gl_state_restoration(self):
        self.setup_model(particle_model())
        self.frame()
        batch = self.viewer.objects["/model/particles"].batch
        self.viewer.renderer.draw_edges = True
        self.viewer.renderer.draw_shadows = True
        with patch.object(batch, "draw", wraps=batch.draw) as draw:
            self.viewer.end_frame()
            self.assertEqual(draw.call_count, 1)
        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDepthFunc(gl.GL_ALWAYS)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        integer_states = (gl.GL_CURRENT_PROGRAM, gl.GL_VERTEX_ARRAY_BINDING, gl.GL_DEPTH_WRITEMASK,
                          gl.GL_DEPTH_FUNC, gl.GL_DRAW_FRAMEBUFFER_BINDING)
        before = [_gl_integer(state) for state in integer_states]
        batch.draw(self.viewer.camera.get_view_matrix(), self.viewer.camera.get_projection_matrix())
        self.assertEqual(before, [_gl_integer(state) for state in integer_states])
        self.assertTrue(gl.glIsEnabled(gl.GL_BLEND))
        self.assertTrue(gl.glIsEnabled(gl.GL_CULL_FACE))
        self.assertFalse(gl.glIsEnabled(gl.GL_DEPTH_TEST))
        polygon_mode = (gl.GLint * 2)()
        gl.glGetIntegerv(gl.GL_POLYGON_MODE, polygon_mode)
        self.assertEqual(polygon_mode[0], gl.GL_LINE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)

    def test_invalid_inputs_do_not_replace_live_buffer(self):
        self.setup_model(particle_model())
        self.frame()
        batch = self.viewer.objects["/model/particles"].batch
        with self.assertRaises(ValueError):
            batch.update(self.state.particle_q, wp.zeros(2, dtype=wp.vec3, device="cuda:0"),
                         self.viewer.model.particle_radius)
        with self.assertRaises(ValueError):
            BillboardBatch("cuda:0", speed_color_mid=6.5, speed_color_max=6.5)
        self.assertEqual(batch.count, 1)
        batch.close()
        batch.close()
        with self.assertRaises(RuntimeError):
            batch.update(self.state.particle_q, self.state.particle_qd, self.viewer.model.particle_radius)


class RigidChecks(unittest.TestCase):
    def setUp(self):
        app.event_loop.has_exit = False
        self.viewer = NewtonBillboardViewer(width=192, height=192, headless=True, vsync=False)
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform((1., 2., 3.), wp.quat_identity()))
        builder.add_shape_capsule(body, radius=.2, half_height=.5)
        builder.add_shape_box(-1, xform=wp.transform((0., 0., 1.), wp.quat_identity()), hx=.2, hy=.3, hz=.4)
        self.model = builder.finalize(device="cuda:0")
        self.state = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.show_visual = True

    def tearDown(self):
        self.viewer.close()

    def packed(self):
        batch = self.viewer._rigid_batches[id(self.viewer.layer)][0]
        values = np.empty((batch.count, 24), dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, batch.buffer)
        gl.glGetBufferSubData(gl.GL_ARRAY_BUFFER, 0, values.nbytes, values.ctypes.data)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        return batch, values

    def test_capsules_and_dynamic_color_use_one_gpu_buffer(self):
        self.viewer.log_state(self.state)
        batch, values = self.packed()
        # One cylinder, two caps, and one static box.
        self.assertEqual(batch.count, 4)
        ids = batch.shape_ids.numpy()
        capsule = values[ids == 0]
        np.testing.assert_allclose(np.sort(capsule[:, 14]), [2.5, 3., 3.5])
        np.testing.assert_allclose(capsule[:, 12:14], [[1., 2.]] * 3)
        buffer_id, registration = batch.buffer.value, batch.registered
        self.model.shape_color.assign(np.array([[.1, .7, .3], [.8, .4, .2]], dtype=np.float32))
        self.state.body_q.assign(np.array([[4., 2., 3., 0., 0., 0., 1.]], dtype=np.float32))
        with patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected CPU readback")), \
                patch.object(wp, "synchronize", side_effect=AssertionError("Explicit device synchronization")), \
                patch.object(sys.modules["pyglet.gl"], "glBufferSubData", side_effect=AssertionError("Host upload")):
            self.viewer.log_state(self.state)
        batch, moved = self.packed()
        self.assertEqual(batch.buffer.value, buffer_id)
        self.assertIs(batch.registered, registration)
        np.testing.assert_allclose(moved[ids == 0, 12], [4.] * 3)
        np.testing.assert_allclose(moved[ids == 0, 16:19], [[.1, .7, .3]] * 3)
        np.testing.assert_allclose(moved[ids == 1, 12:15], [[0., 0., 1.]])
        self.viewer.begin_frame(0.)
        self.viewer.end_frame()
        self.assertTrue(np.isfinite(self.viewer.get_frame().numpy()).all())

    def test_appearance_invalidation_and_cleanup(self):
        self.viewer.log_state(self.state)
        old, _ = self.packed()
        self.model.shape_opacity.assign(np.array([.5, 1.], dtype=np.float32))
        self.viewer.invalidate_rigid_appearance()
        self.assertTrue(old.closed)
        self.assertEqual(old.buffer.value, 0)
        self.viewer.log_state(self.state)
        batch, _ = self.packed()
        self.assertTrue(any(drawable.has_transparency() for drawable in batch.drawables))
        self.viewer.set_visible_worlds(None)
        self.assertTrue(batch.closed)
        self.viewer.log_state(self.state)
        replacement, _ = self.packed()
        self.viewer.set_model(None)
        self.assertTrue(replacement.closed)

    def test_close_releases_shared_registration_before_window(self):
        self.viewer.log_state(self.state)
        batch, _ = self.packed()
        normal_close = self.viewer.renderer.close

        def close_renderer():
            self.assertTrue(batch.closed)
            self.assertIsNone(batch.registered)
            self.assertEqual(batch.buffer.value, 0)
            normal_close()

        with patch.object(self.viewer.renderer, "close", side_effect=close_renderer):
            self.viewer.close()
        self.viewer.close()

    def test_picking_and_com_overlay_still_work(self):
        self.viewer.log_state(self.state)
        self.viewer.picking.pick(self.state, wp.vec3(1., -3., 3.), wp.vec3(0., 1., 0.))
        self.assertTrue(self.viewer.picking.is_picking())
        self.assertEqual(self.viewer.picking.pick_body.numpy()[0], 0)
        self.viewer.log_state(self.state)
        self.viewer.picking.release()
        self.viewer.show_com = True
        self.viewer.log_state(self.state)
        self.assertFalse(self.viewer.objects["/model/com"].hidden)
        self.viewer.show_com = False
        self.viewer.log_state(self.state)
        self.assertTrue(self.viewer.objects["/model/com"].hidden)


class ViewerUIChecks(unittest.TestCase):
    def test_import_does_not_initialize_opengl(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import RSP.rendering.newton_gl; "
             "assert 'pyglet.gl' not in sys.modules, 'Viewer import initialized OpenGL'"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ui_keyboard_and_window_close_keep_resource_order(self):
        app.event_loop.has_exit = False
        viewer = NewtonBillboardViewer(width=640, height=480, headless=False, vsync=False)
        viewer.renderer.window.set_visible(False)
        try:
            viewer.set_model(particle_model())
            viewer.show_particles = True
            self.assertTrue(viewer.ui.is_available)
            calls = []

            def panel(imgui):
                calls.append("panel")
                imgui.text("Billboard UI compatibility")

            viewer.register_ui_callback(panel)
            viewer.renderer.register_key_press(lambda symbol, modifiers: calls.append("key"))
            viewer.begin_frame(0.)
            viewer.log_state(viewer.model.state())
            viewer.end_frame()
            self.assertIn("panel", calls)
            from pyglet.window import key
            viewer.renderer._on_key_press(key.R, 0)
            self.assertIn("key", calls)
            from PIL import Image
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(OUTPUT / "ui.png")
            batch = viewer.objects["/model/particles"].batch
            viewer.renderer.window.dispatch_event("on_close")
            viewer.renderer.window.dispatch_events()
            self.assertTrue(batch._closed)
            self.assertIsNone(batch._registered)
            self.assertIsNone(viewer.renderer.window.context)
        finally:
            viewer.close()
            viewer.close()


if __name__ == "__main__":
    unittest.main()
