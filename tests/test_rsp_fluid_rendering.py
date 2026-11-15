"""Small real-GL mode checks; no recording or full-scene regression."""
from unittest.mock import patch
import numpy as np
from tests.test_rsp_rendering import BillboardChecks, particle_model, wp, OUTPUT, gl


class FluidModeChecks(BillboardChecks):
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
                self.assertTrue(np.all(pixels[96,96] >= 254), pixels[96,96])
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

    def test_gpu_opacity_thresholds(self):
        from RSP.rendering.fluid.resources import opacity
        velocity = wp.array([(v,0.,0.) for v in (0.,2.,4.25,6.5,20.)],dtype=wp.vec3,device='cuda:0')
        packed = wp.zeros(10,dtype=wp.vec4,device='cuda:0')
        wp.launch(opacity,5,inputs=[velocity,packed,2.,6.5,.15,.8],device='cuda:0')
        np.testing.assert_allclose(packed.numpy()[1::2,3],[.15,.15,.475,.8,.8],atol=1e-6)

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
        self.assertTrue(previous._closed)
        self.assertIsNone(previous._registered)
        self.frame()
        self.assertEqual(pipeline.mode, 'SSFR')
        self.assertIsNone(pipeline.error)
        self.assertEqual(gl.glGetError(), gl.GL_NO_ERROR)
