"""Matched humanoid/G1 steady-state sampling for Nsight Systems, without recording.

Run under nsys with --capture-range=cudaProfilerApi --capture-range-end=stop
--cuda-graph-trace=node --trace=cuda,nvtx,opengl. The first pass is untraced;
the second resets to the same RNG state before warming up and capturing.
No simulation or renderer implementation is changed by this diagnostic.
"""
import argparse
from contextlib import ExitStack
import ctypes
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'mimickit')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('humanoid', 'g1'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--nsys', action='store_true')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    import warp as wp
    import newton
    from util import resource_util, mp_util, util
    from RSP.app.application import open_playback
    from RSP.integrations.mimickit.profiles import load_profile
    from RSP.rendering.newton_gl import NewtonBillboardViewer
    from RSP.app.playback import bind_playback
    from profile_rsp_render import statistics

    profile = load_profile(args.profile)
    viewer = None
    report = dict(profile=args.profile, warmup=args.warmup, frames=args.frames,
                  seed=profile.seed, passes={}, versions=dict(newton=newton.__version__, warp=wp.__version__,
                                                           torch=torch.__version__))
    mp_util.init(0, 1, profile.device, None)
    started = time.perf_counter()
    try:
        with wp.ScopedDevice(profile.device), torch.no_grad():
            viewer = resource_util.construct(NewtonBillboardViewer, width=args.width, height=args.height, vsync=False)
            viewer.renderer.window.set_visible(False)
            with open_playback(profile, profile.scene_preset, viewer) as session, bind_playback(session) as runner:
                env, agent = session.env, session.agent
                engine = env._engine
                model = engine.get_sim_model()
                pipeline = viewer.fluid_pipeline
                gl = viewer.renderer.gl
                report['initialization_s'] = time.perf_counter() - started
                shape_types = model.shape_type.numpy()
                report.update(particles=model.particle_count, bodies=model.body_count,
                    joints=model.joint_count, shapes=model.shape_count,
                    shape_types={newton.GeoType(int(k)).name: int(v) for k, v in zip(*np.unique(shape_types, return_counts=True))},
                    mesh_triangles=sum(len(source.indices)//3 for source in model.shape_source if source is not None and hasattr(source, 'indices')),
                    coupled_bodies=len(engine.get_play_scene().body_meshes),
                    schedule=asdict(engine.get_play_scene().fluid_schedule),
                    resolution=list(viewer.renderer.window.get_framebuffer_size()), vsync=False,
                    fluid_mode=pipeline.mode, shadows=viewer.renderer.draw_shadows,
                    actor_parameters=sum(p.numel() for p in agent._model._actor_layers.parameters()),
                    policy_interval=runner.policy_interval)

                def drain():
                    wp.synchronize()
                    gl.glFinish()

                row = None
                tracing = False
                queries = []

                def hook(stack, obj, name, label, gpu=False):
                    original = getattr(obj, name)
                    def measured(*a, **kw):
                        if row is None:
                            return original(*a, **kw)
                        if tracing:
                            torch.cuda.nvtx.range_push(label)
                        if gpu:
                            q = (gl.GLuint * 2)()
                            gl.glGenQueries(2, q)
                            gl.glQueryCounter(q[0], gl.GL_TIMESTAMP)
                        start = time.perf_counter()
                        try:
                            return original(*a, **kw)
                        finally:
                            row[label] = row.get(label, 0.) + (time.perf_counter()-start)*1000
                            if label == 'policy':
                                row['policy_calls'] = row.get('policy_calls', 0) + 1
                            if gpu:
                                gl.glQueryCounter(q[1], gl.GL_TIMESTAMP)
                                queries.append((q, row, label+'_gpu'))
                            if tracing:
                                torch.cuda.nvtx.range_pop()
                    stack.enter_context(patch.object(obj, name, measured))

                with ExitStack() as stack:
                    for obj, method, label, gpu in (
                        (runner.policy, 'act', 'policy', False),
                        (env, '_pre_physics_step', 'action', False),
                        (env, '_physics_step', 'physics', False),
                        (env, '_post_physics_step', 'observations', False),
                        (engine._presentation, 'render', 'render', False),
                        (viewer, 'log_state', 'state_upload', False),
                        (pipeline, 'original_render', 'scene_draw', True),
                        (viewer.renderer, 'present', 'present', False),
                        (viewer.renderer, 'update', 'events', False),
                        (viewer.gui, 'prepare_frame', 'ui_build', False),
                        (viewer.gui, 'render_prepared_frame', 'ui_draw', True),
                        (wp.RegisteredGLBuffer, 'map', 'interop_map', False),
                        (wp.RegisteredGLBuffer, 'unmap', 'interop_unmap', False)):

                        hook(stack, obj, method, label, gpu)
                    initial_root = None
                    # GSI samples use a pre-shuffled cursor, not just the global RNG.
                    # Restore the diagnostic's sampler state so both passes replay
                    # the same initial motion; production reset behavior is untouched.
                    gsi = env._gsi_buffer
                    sample_head, sample_order = gsi._sample_buf_head, gsi._sample_buf.clone()
                    for capture in (False, True) if args.nsys else (False,):
                        util.set_rand_seed(profile.seed)
                        gsi._sample_buf_head = sample_head
                        gsi._sample_buf.copy_(sample_order)
                        runner.reset()
                        viewer.set_paused(False)
                        root = env._engine.get_root_pos(env._get_char_id()).clone()
                        if initial_root is None:
                            initial_root = root
                        else:
                            torch.testing.assert_close(root, initial_root, rtol=0, atol=0)
                        for _ in range(args.warmup):
                            assert runner.tick()
                        drain()
                        queries.clear()
                        samples = []
                        if capture:
                            torch.cuda.profiler.start()
                        tracing = capture
                        start = time.perf_counter()
                        for index in range(args.frames):
                            row = {}
                            if tracing:
                                torch.cuda.nvtx.range_push(f'frame_{index}')
                            frame_start = time.perf_counter()
                            assert runner.tick()
                            row['frame'] = (time.perf_counter()-frame_start)*1000
                            if tracing:
                                torch.cuda.nvtx.range_pop()
                            samples.append(row)
                            row = None
                        drain()
                        elapsed = time.perf_counter()-start
                        tracing = False
                        if capture:
                            torch.cuda.profiler.stop()
                        for q, sample, label in queries:
                            a, b = gl.GLuint64(), gl.GLuint64()
                            gl.glGetQueryObjectui64v(q[0], gl.GL_QUERY_RESULT, ctypes.byref(a))
                            gl.glGetQueryObjectui64v(q[1], gl.GL_QUERY_RESULT, ctypes.byref(b))
                            sample[label] = (b.value-a.value)/1e6
                            gl.glDeleteQueries(2, q)
                        summary = {key: statistics([sample.get(key, 0.) for sample in samples])
                                   for key in set().union(*samples) if key != 'policy_calls'}
                        kind = 'nsys' if capture else 'untraced'
                        report['passes'][kind] = dict(fps=args.frames/elapsed, total_s=elapsed,
                            completed_frame_ms=elapsed*1000/args.frames,
                            policy_calls=sum(sample.get('policy_calls', 0) for sample in samples),
                            summary=summary, samples=samples)
                        data = engine._runtime._solver.rigid.mjw_data
                        counters = {name: getattr(data, name).numpy().tolist() for name in
                                    ('ncon', 'nacon', 'nefc', 'solver_niter') if hasattr(data, name)
                                    and hasattr(getattr(data, name), 'numpy')}
                        report['passes'][kind]['final_rigid_counters'] = counters
                        print('RESULT', args.profile, kind, 'fps', args.frames/elapsed, flush=True)
                        (args.output/'timings.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        resource_util.cleanup(lambda: resource_util.close_viewer(viewer), mp_util.shutdown)


if __name__ == '__main__':
    # Windows Nsight may not forward a target's stderr to its CLI log.
    try:
        main()
    except BaseException:
        import traceback
        if '--output' in sys.argv:
            folder = Path(sys.argv[sys.argv.index('--output') + 1])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
