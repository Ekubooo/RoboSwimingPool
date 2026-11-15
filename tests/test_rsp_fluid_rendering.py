"""Small real-GL mode checks; no recording or full-scene regression."""
from unittest.mock import patch
import numpy as np
from tests.test_rsp_rendering import BillboardChecks, particle_model, wp, OUTPUT, gl


class FluidModeChecks(BillboardChecks):
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
