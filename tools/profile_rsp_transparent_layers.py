"""One fixed-scene comparison of transparent layers; no physics or asset baking."""
import ctypes
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import newton
import warp as wp
from PIL import Image
from pyglet import gl
from RSP.rendering.newton_gl import NewtonBillboardViewer


def summary(values):
    return dict(mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)))


def main():
    output = ROOT / 'output/rsp_render_profile/transparent_layers'
    output.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(ROOT / 'output/newton_pbf_refactor_checks/warp_cache')
    grid = np.indices((80, 60, 104), dtype=np.float32).reshape(3, -1).T
    points = grid * .02 + np.array([-.79, 0., .02], dtype=np.float32)
    velocities = np.zeros_like(points)
    velocities[:, 0] = grid[:, 1] / 59 * 8.
    builder = newton.ModelBuilder()
    builder.add_particles(pos=points, vel=velocities, mass=np.ones(len(points)),
                          radius=np.full(len(points), .01))
    builder.add_shape_box(-1, xform=wp.transform((0., -.25, .8), wp.quat_identity()),
                          hx=.2, hy=.15, hz=.4)
    model = builder.finalize(device='cuda:0')
    state = model.state()
    viewer = NewtonBillboardViewer(width=1280, height=720, headless=True, vsync=False)
    try:
        viewer.set_model(model)
        viewer.show_particles = viewer.show_visual = True
        viewer.camera.pos = np.array((3., -5., 3.), dtype=np.float32)
        viewer.camera.look_at((0., .6, 1.))
        pipeline = viewer.fluid_pipeline
        pipeline.request('Transparent')

        def frame():
            viewer.begin_frame(0.)
            viewer.log_state(state)
            viewer.end_frame()

        report = dict(particles=len(points), resolution=[1280, 720], warmup=30, frames=180,
                      scope='Fixed synthetic scene; render/update only, no physics, policy or UI', modes={})
        for layered in (True, False):
            name = 'layered' if layered else 'nearest'
            pipeline.request_transparent(layer_blending=layered)
            start = time.perf_counter()
            frame()
            wp.synchronize()
            gl.glFinish()
            assert pipeline.error is None, pipeline.error
            first = (time.perf_counter() - start) * 1000
            for _ in range(30):
                frame()
            wp.synchronize()
            gl.glFinish()
            queries = (gl.GLuint * 360)()
            gl.glGenQueries(360, queries)
            cpu, index = [], [0]
            original = pipeline.transparent

            def timed_pass():
                i = index[0]
                gl.glQueryCounter(queries[2*i], gl.GL_TIMESTAMP)
                original()
                gl.glQueryCounter(queries[2*i+1], gl.GL_TIMESTAMP)
                index[0] += 1

            try:
                with patch.object(pipeline, 'transparent', timed_pass), patch.object(
                        wp.array, 'numpy', side_effect=AssertionError('ordinary frame readback')):
                    for _ in range(180):
                        start = time.perf_counter()
                        frame()
                        cpu.append((time.perf_counter()-start)*1000)
                gl.glFinish()
                gpu = []
                for i in range(180):
                    begin, end = gl.GLuint64(), gl.GLuint64()
                    gl.glGetQueryObjectui64v(queries[2*i], gl.GL_QUERY_RESULT, ctypes.byref(begin))
                    gl.glGetQueryObjectui64v(queries[2*i+1], gl.GL_QUERY_RESULT, ctypes.byref(end))
                    gpu.append((end.value-begin.value)/1e6)
            finally:
                gl.glDeleteQueries(360, queries)
            Image.fromarray(viewer.get_frame().numpy()).save(output / f'{name}.png')
            report['modes'][name] = dict(render_frame_cpu_ms=summary(cpu), fluid_gpu_ms=summary(gpu),
                                         first_completed_frame_ms=first, readback_guard=True)
        (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
