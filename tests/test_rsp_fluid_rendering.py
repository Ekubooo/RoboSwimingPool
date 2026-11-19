"""Small real-GL mode checks; no recording or full-scene regression."""
from unittest.mock import patch
import numpy as np
from tests.test_rsp_rendering import BillboardChecks, particle_model, wp, OUTPUT, gl


class FluidModeChecks(BillboardChecks):
    def test_transparent_nearest_layer(self):
        points = [(0.,-1.,0.), (0.,1.,0.)]
        velocities = [(0.,0.,0.), (10.,0.,0.)]
        self.setup_model(particle_model(points, velocities))
        pipeline = self.viewer.fluid_pipeline
        pipeline.request('Transparent')
        pipeline.request_transparent(depth_enabled=False)
        layered = self.frame()
        self.assertTrue(pipeline.config.transparent_layer_blending)
        self.assertNotIn('nearest', pipeline.targets)
        self.viewer._paused = True
        pipeline.request_transparent(layer_blending=False)
        single = self.frame()
        np.testing.assert_allclose(single[96,96], np.array([29,119,231])*.8, atol=1)
        self.assertFalse(np.array_equal(single[96,96], layered[96,96]))
        np.testing.assert_array_equal(single[0,0], layered[0,0])
        # A closer slow particle wins regardless of draw order or rear color.
        self.state.particle_q.assign(np.array(points[::-1], dtype=np.float32))
        self.state.particle_qd.assign(np.array(velocities[::-1], dtype=np.float32))
        np.testing.assert_array_equal(self.frame(), single)
        pipeline.request_transparent(minimum=60)
        np.testing.assert_allclose(self.frame()[96,96], np.array([29,119,231])*.6, atol=1)
        pipeline.request_transparent(minimum=80, layer_blending=True)
        np.testing.assert_allclose(self.frame(), layered, atol=1)
        pipeline.request_transparent(layer_blending=False, depth_enabled=True)
        self.frame()
        with patch.object(wp.array, 'numpy', side_effect=AssertionError('readback')):
            self.viewer.log_state(self.state)
            self.viewer.end_frame()
        self.assertTrue(self.viewer._paused)
        # Opaque scene geometry still wins, even against a white particle.
        self.setup_model(particle_model(velocities=[(10.,0.,0.)], occluder=True))
        occluded = self.frame()
        pipeline.visible = False
        background = self.frame(update=False)
        np.testing.assert_array_equal(occluded[96,96], background[96,96])
        old = pipeline.targets['nearest']
        self.viewer.renderer.window.set_size(241,161)
        self.viewer.renderer.window.dispatch_events()
        self.viewer.renderer.resize(241,161)
        # Exercise resource resize without Newton's cached screenshot buffer.
        self.viewer.begin_frame(0.)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        self.assertEqual(old.depth.value, 0)
        self.assertNotEqual(pipeline.targets['nearest'].depth.value, 0)
        self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)

    def test_transparent_depth_shading_and_controls(self):
        points = [(-.6,-1.,0.), (.6,1.,0.)]
        self.setup_model(particle_model(points))
        pipeline = self.viewer.fluid_pipeline
        pipeline.request('Transparent')
        pipeline.request_transparent(depth_enabled=False)
        plain = self.frame()
        self.assertFalse(any(k.startswith('depth_range_') for k in pipeline.targets))
        r = self.viewer.renderer
        view = np.asarray(r._view_matrix).reshape(4,4,order='F')
        proj = np.asarray(r._projection_matrix).reshape(4,4,order='F')
        centers = []
        for point in points:
            p = proj @ view @ np.array([*point,1.])
            x,y = ((p[:2]/p[3]+1)*96).astype(int)
            centers.append((191-y,x))
        self.viewer._paused = True
        pipeline.request_transparent(depth_enabled=True, strength=1.)
        shaded = self.frame()
        self.assertTrue(np.all(shaded[centers[0]] >= plain[centers[0]]))
        self.assertGreater(int(shaded[centers[0]][0]), int(plain[centers[0]][0]))
        self.assertTrue(np.all(shaded[centers[1]] < plain[centers[1]]))
        # Uncovered background and all opacity/scene-depth logic stay intact.
        np.testing.assert_array_equal(shaded[0,0], plain[0,0])
        pipeline.request_transparent(strength=0.)
        np.testing.assert_array_equal(self.frame(), plain)
        pipeline.request_transparent(minimum=61.2, maximum=92.8)
        self.frame()
        self.assertEqual(pipeline.config.transparent_alpha_low, .61)
        self.assertEqual(pipeline.config.transparent_alpha_high, .93)
        pipeline.request_transparent(minimum=99, maximum=2, strength=1.)
        self.frame()
        self.assertEqual(pipeline.config.transparent_alpha_low, .8)
        self.assertEqual(pipeline.config.transparent_alpha_high, .8)
        self.assertEqual(pipeline.config.alpha_high, .9)  # SSFR untouched.
        with patch.object(wp.array, 'numpy', side_effect=AssertionError('readback')):
            self.viewer.log_state(self.state)
            self.viewer.end_frame()
        self.viewer.renderer.window.set_size(241,161)
        self.frame()
        self.assertEqual(list(pipeline.depth_range_levels())[-1][1:], (1,1))
        self.assertTrue(self.viewer._paused)
        self.assertEqual(gl.glGetError(),gl.GL_NO_ERROR)

    def test_water_color_switch_preserves_resources_and_current_look(self):
        self.setup_model(particle_model())
        pipeline = self.viewer.fluid_pipeline
        pipeline.request('SSFR')
        current = self.frame()
        self.assertEqual(pipeline.water_color, 'Current')
        targets = dict(pipeline.targets)
        registration = pipeline.batch.instances.buffer.registration
        positions = self.state.particle_q.numpy().copy()
        self.viewer._paused = True
        pipeline.request_water_color('Billboard blue')
        blue = self.frame()
        self.assertGreater(int(blue[96,96,2]), int(current[96,96,2]))
        self.assertEqual(pipeline.targets, targets)
        self.assertIs(pipeline.batch.instances.buffer.registration, registration)
        self.assertTrue(self.viewer._paused)
        np.testing.assert_array_equal(positions, self.state.particle_q.numpy())
        with patch.object(wp.array, 'numpy', side_effect=AssertionError('readback')):
            self.viewer.log_state(self.state)
            self.viewer.end_frame()
        pipeline.request('Original')
        self.frame()
        pipeline.request('SSFR')
        np.testing.assert_array_equal(blue, self.frame())
        self.assertEqual(pipeline.water_color, 'Billboard blue')
        pipeline.request_water_color('Current')
        np.testing.assert_array_equal(current, self.frame())
        self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)

    def material_sample(self):
        """Diagnostic-only attachment readback, never used during timing."""
        target = self.viewer.fluid_pipeline.targets['thickness_velocity']
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, target.fbo)
        color = np.zeros(4, dtype=np.float32)
        coverage = np.zeros(1, dtype=np.float32)
        try:
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
            gl.glReadPixels(96, 96, 1, 1, gl.GL_RGBA, gl.GL_FLOAT, color.ctypes.data)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT2)
            gl.glReadPixels(96, 96, 1, 1, gl.GL_RED, gl.GL_FLOAT, coverage.ctypes.data)
        finally:
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        return color, coverage[0]

    def test_ssfr_materials_and_surface_attribute_occlusion(self):
        self.setup_model(particle_model())
        pipeline = self.viewer.fluid_pipeline
        pipeline.request('SSFR')
        self.frame()
        self.assertEqual(pipeline.ssfr_preset, 'Water')
        self.assertNotIn('thickness_velocity', pipeline.targets)
        pipeline.request_ssfr_preset('Velocity')
        self.viewer._paused = True
        original_positions = self.state.particle_q.numpy().copy()
        expected = [((29,119,231),0.), ((160,221,255),0.), ((207.5,238,255),.5), ((255,255,255),1.)]
        for speed, (rgb, amount) in zip((0.,2.,4.25,6.5), expected):
            self.state.particle_qd.assign(np.array([(speed,0.,0.)], dtype=np.float32))
            pixels = self.frame()
            self.assertIsNone(pipeline.error)
            self.assertTrue(self.viewer._paused)
            color, coverage = self.material_sample()
            self.assertGreater(color[3], 0.)
            np.testing.assert_allclose(color[:3]/color[3], np.array(rgb)/255., atol=.002)
            self.assertAlmostEqual(float(coverage/color[3]), amount, places=3)
            if speed == 0.:
                # Neutral absorption preserves the palette hue over a black
                # background; Water intentionally keeps selective absorption.
                self.assertGreater(pixels[96,96,2], pixels[96,96,1])
                self.assertGreater(pixels[96,96,1], pixels[96,96,0])
                self.assertTrue(np.all(pixels[96,96] < 254))
            if speed == 2.:
                self.assertGreater(pixels[96,96,2], pixels[96,96,0])
            if speed == 6.5:
                # White over the black background must transmit 10%, even
                # when the underlying Water material contributes opacity.
                np.testing.assert_allclose(pixels[96,96], [230]*3, atol=1)
        np.testing.assert_array_equal(original_positions, self.state.particle_q.numpy())
        with patch.object(wp.array, 'numpy', side_effect=AssertionError('readback')):
            self.viewer.log_state(self.state)
            self.viewer.end_frame()
        target = pipeline.targets['thickness_velocity']
        pipeline.request_ssfr_preset('Water')
        with patch.object(pipeline, 'particle_pass', wraps=pipeline.particle_pass) as draws:
            self.frame()
            self.assertEqual([call.args[0] for call in draws.call_args_list], ['depth','thickness'])
        with patch.object(pipeline, 'prepare', side_effect=RuntimeError('allocation failed')):
            pipeline.request_ssfr_preset('Velocity')
            pipeline.apply_pending()
            self.assertEqual(pipeline.ssfr_preset, 'Water')
        pipeline.request_ssfr_preset('Velocity')
        self.frame()
        self.assertIs(pipeline.targets['thickness_velocity'], target)
        pipeline.request('Original')
        self.frame()
        pipeline.request('SSFR')
        self.frame()
        self.assertEqual(pipeline.ssfr_preset, 'Velocity')

        # The rear fast particle must not whiten a separate slow front surface.
        self.setup_model(particle_model([(0.,-.8,0.), (0.,1.,0.)], [(0.,0.,0.), (10.,0.,0.)]))
        self.frame()
        color, coverage = self.material_sample()
        np.testing.assert_allclose(color[:3]/color[3], np.array((29,119,231))/255., atol=.002)
        self.assertEqual(coverage, 0.)
        # Opaque geometry rejects material attributes as well as fluid depth.
        self.setup_model(particle_model(velocities=[(10.,0.,0.)], occluder=True))
        self.frame()
        color, coverage = self.material_sample()
        self.assertEqual(color[3], 0.)
        self.assertEqual(coverage, 0.)
        self.viewer.renderer.window.set_size(240,160)
        self.frame()
        self.assertEqual(pipeline.size, (self.viewer.renderer._screen_width,self.viewer.renderer._screen_height))
        self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)

    def test_transparent_opacity_and_hard_depth_intersection(self):
        self.setup_model(particle_model(velocities=[(6.5,0.,0.)]))
        pipeline = self.viewer.fluid_pipeline
        pipeline.request('Transparent')
        pixels = self.frame()
        np.testing.assert_allclose(pixels[96,96], [255]*3, atol=1)
        # Five m/s is no longer the white endpoint in either optional mode.
        self.state.particle_qd.assign(np.array([(5.,0.,0.)],dtype=np.float32))
        slower = self.frame()
        self.assertLess(int(slower[96,96,0]), int(pixels[96,96,0]))
        self.state.particle_qd.assign(np.zeros((1,3),dtype=np.float32))
        np.testing.assert_allclose(self.frame()[96,96], np.array([29,119,231])*.8, atol=1)
        # A particle just in front of the box remains fully opaque; behind
        # the box it is still occluded rather than blended over the geometry.
        self.setup_model(particle_model([(0.,-.71,0.)], [(6.5,0.,0.)], occluder=True))
        np.testing.assert_allclose(self.frame()[96,96], [255]*3, atol=1)
        self.state.particle_q.assign(np.array([(0.,0.,0.)],dtype=np.float32))
        self.assertFalse(np.all(self.frame()[96,96] >= 254))
        self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)

    def test_gpu_opacity_thresholds(self):
        from RSP.rendering.particles import opacity
        velocity = wp.array([(v,0.,0.) for v in (0.,2.,4.25,6.5,20.)],dtype=wp.vec3,device='cuda:0')
        packed = wp.zeros(10,dtype=wp.vec4,device='cuda:0')
        wp.launch(opacity,5,inputs=[velocity,packed,2.,6.5,.15,.9],device='cuda:0')
        np.testing.assert_allclose(packed.numpy()[1::2,3],[.15,.15,.525,.9,.9],atol=1e-6)

    def test_modes_preserve_state_and_original_image(self):
        positions = [(x*.25, y*.25, z*.25) for x in range(-2,3) for y in range(-1,2) for z in range(-2,3)]
        self.setup_model(particle_model(positions, [(4.,0.,0.)]*len(positions), occluder=True))
        pipeline = self.viewer.fluid_pipeline
        original = self.frame()
        self.assertEqual(pipeline.targets, {})
        self.assertIsNone(pipeline.batch)
        state = self.state.particle_q.numpy().copy()
        from PIL import Image
        Image.fromarray(original).save(OUTPUT/'fluid_original.png')
        for mode in ('Transparent','SSFR','Original'):
            pipeline.request(mode)
            result = self.frame()
            self.assertIsNone(pipeline.error)
            self.assertEqual(pipeline.mode, mode)
            Image.fromarray(result).save(OUTPUT/f'fluid_{mode}.png')
            with patch.object(wp.array,'numpy',side_effect=AssertionError('readback')):
                self.viewer.log_state(self.state)
                self.viewer.end_frame()
            np.testing.assert_array_equal(state,self.state.particle_q.numpy())
            if mode == 'Original':
                np.testing.assert_array_equal(original,result)
            else:
                self.assertGreater(np.count_nonzero(original != result),100)
            self.assertEqual(gl.glGetError(),gl.GL_NO_ERROR)
        pipeline.request('SSFR')
        self.viewer._paused = True
        self.frame()
        self.assertTrue(self.viewer._paused)
        self.viewer.camera.pos = np.array((5., 0., 0.), dtype=np.float32)
        self.viewer.camera.look_at((0., 0., 0.))
        with patch.object(pipeline.batch, 'update', side_effect=AssertionError('paused camera upload')):
            self.frame(update=False)
        self.viewer.renderer.window.set_size(240,160)
        self.frame()
        self.assertEqual(pipeline.size, (self.viewer.renderer._screen_width,self.viewer.renderer._screen_height))
        with patch.object(pipeline,'prepare',side_effect=RuntimeError('allocation failed')):
            pipeline.request('Transparent')
            pipeline.apply_pending()
            self.assertEqual(pipeline.mode,'SSFR')
            self.assertIn('allocation failed',pipeline.error)
        previous = pipeline.batch
        self.setup_model(particle_model())
        self.assertTrue(previous.instances._closed)
        self.assertIsNone(previous.instances.buffer.registration)
        self.frame()
        self.assertEqual(pipeline.mode, 'SSFR')
        self.assertIsNone(pipeline.error)
        self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)
