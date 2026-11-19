"""One independent-process sample per fluid mode; no recording."""
import argparse
import ctypes
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'mimickit')]
from profile_rsp_render import statistics


def main():
    from contextlib import ExitStack
    playback_stack = ExitStack()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('Original','Transparent','SSFR'))
    parser.add_argument('--ssfr-preset', choices=('Water', 'Velocity'))
    parser.add_argument('--transparent-depth', choices=('off','on'))
    parser.add_argument('--check-materials', action='store_true', help='Check paused material switches/reset after sampling')
    parser.add_argument('--check-switches', action='store_true', help='Short paused-switch/reset check instead of sampling')
    args = parser.parse_args()
    if args.transparent_depth and args.mode != 'Transparent':
        parser.error('--transparent-depth requires Transparent mode')
    if (args.ssfr_preset or args.check_materials) and args.mode != 'SSFR':
        parser.error('SSFR material options require SSFR mode')
    output = ROOT/'output/rsp_render_profile/fluid_modes'/args.mode
    if args.ssfr_preset:
        output = ROOT/'output/rsp_render_profile/ssfr_presets'/args.ssfr_preset
    if args.transparent_depth:
        output = ROOT/'output/rsp_render_profile/transparent_depth'/args.transparent_depth
    output.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(ROOT/'output/newton_pbf_refactor_checks/tmp')
    import torch
    import warp as wp
    import yaml
    from PIL import Image
    wp.config.kernel_cache_dir = str(ROOT/'output/newton_pbf_refactor_checks/warp_cache')
    from RSP.integrations.mimickit import recorder as newton_pbf_engine
    from engines.newton_pbf_viewer import PBFViewer
    from envs.pbf_play_env import PBFPlayEnv
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import resource_util, mp_util
    from util.util import set_rand_seed
    from RSP.app.playback import bind_playback
    from RSP.integrations.mimickit.session import MimicKitSession
    class Viewer(PBFViewer):
        def __init__(self, **kw):
            kw.update(width=1280,height=720,vsync=False)
            super().__init__(**kw)
            self.renderer.window.set_visible(False)
    env = agent = None
    set_rand_seed(1234)
    mp_util.init(0,1,'cuda:0',None)
    try:
        with patch.object(newton_pbf_engine,'PBFViewer',Viewer):
            env = resource_util.construct(PBFPlayEnv,
                yaml.safe_load((ROOT/'data/envs/pbf_play_env.yaml').read_text()),
                yaml.safe_load((ROOT/'data/engines/newton_pbf_engine.yaml').read_text()),1,'cuda:0',True)
        checkpoint = ROOT/'output/a5000_dodgeball_2048_full_r01'
        agent = build_agent(str(checkpoint/'agent_config.yaml'),env,'cuda:0')
        agent.load(str(checkpoint/'model.pt'))
        agent.eval(); agent.set_mode(AgentMode.TEST)
        runner = playback_stack.enter_context(bind_playback(MimicKitSession(env, agent)))
        viewer, eng = runner.viewer, env._engine
        renderer, pipeline = viewer.renderer, viewer.fluid_pipeline
        gl = renderer.gl
        def drain():
            wp.synchronize(); gl.glFinish()
        def check_materials():
            viewer._paused = True
            sim_time = eng.get_sim_time()
            fluid_time = eng._runtime._solver.clock.time
            graph = eng._runtime._graph
            switches = []
            for preset in ('Velocity', 'Water', 'Velocity'):
                pipeline.request_ssfr_preset(preset)
                start = time.perf_counter()
                # Presentation ends the previous frame before logging the next
                # state. First tick commits the request; second displays it.
                assert runner.tick() is False
                assert runner.tick() is False
                drain()
                elapsed = (time.perf_counter()-start)*1000
                assert pipeline.error is None, pipeline.error
                assert pipeline.ssfr_preset == preset and viewer._paused
                assert eng.get_sim_time() == sim_time
                assert eng._runtime._solver.clock.time == fluid_time
                assert eng._runtime._graph is graph
                switches.append(dict(preset=preset, completed_frame_ms=elapsed,
                                     prepare_ms=pipeline.last_switch_ms))
                Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(output/f'paused_{preset}.png')
            viewer.gui.request_reset()
            assert runner.tick() is False
            assert runner.tick() is False
            assert pipeline.ssfr_preset == 'Velocity' and viewer._paused
            drain()
            return switches
        with torch.no_grad():
            runner.reset()
            pipeline.request(args.mode)
            if args.transparent_depth:
                pipeline.request_transparent(depth_enabled=args.transparent_depth == 'on')
            if args.ssfr_preset:
                pipeline.request_ssfr_preset(args.ssfr_preset)
            start = time.perf_counter()
            env.render_only()
            # Presentation logs the new mode after ending the previous frame.
            # Complete one more render-only frame before measuring readiness.
            env.render_only()
            drain()
            switch_ms = (time.perf_counter()-start)*1000
            assert pipeline.error is None, pipeline.error
            if args.check_switches:
                # Reach the same visual state as 30 warmup + 180 sample + two
                # readback-guard frames, without repeating performance sampling.
                for _ in range(212 if args.check_materials else 2):
                    runner.tick()
                drain()
                if args.check_materials:
                    checks = check_materials()
                    (output/'material_switch_checks.json').write_text(json.dumps(checks, indent=2))
                    print('MATERIAL_SWITCH_CHECKS', json.dumps(checks), flush=True)
                    return
                viewer._paused = True
                sim_time = eng.get_sim_time()
                fluid_time = eng._runtime._solver.clock.time
                graph = eng._runtime._graph
                switches = []
                for mode in ('Transparent', 'SSFR', 'Original', 'SSFR'):
                    pipeline.request(mode)
                    start = time.perf_counter()
                    assert runner.tick() is False
                    drain()
                    elapsed = (time.perf_counter()-start)*1000
                    assert pipeline.error is None, pipeline.error
                    assert pipeline.mode == mode and viewer._paused
                    assert eng.get_sim_time() == sim_time
                    assert eng._runtime._solver.clock.time == fluid_time
                    assert eng._runtime._graph is graph
                    switches.append(dict(mode=mode, completed_frame_ms=elapsed,
                                         prepare_ms=pipeline.last_switch_ms))
                viewer.gui.request_reset()
                assert runner.tick() is False
                assert pipeline.mode == 'SSFR' and viewer._paused
                drain()
                (output.parent/'switch_checks.json').write_text(json.dumps(switches, indent=2))
                print('SWITCH_CHECKS', json.dumps(switches), flush=True)
                return
            rows, queries, current = [], [], [None]
            def hook(stack, obj, name, label, gpu=False):
                original = getattr(obj,name)
                def measured(*a,**kw):
                    row = current[0]
                    if row is None:
                        return original(*a,**kw)
                    if gpu:
                        q=(gl.GLuint*2)(); gl.glGenQueries(2,q); gl.glQueryCounter(q[0],gl.GL_TIMESTAMP)
                    start=time.perf_counter()
                    try:
                        return original(*a,**kw)
                    finally:
                        row[label]=(time.perf_counter()-start)*1000
                        if gpu:
                            gl.glQueryCounter(q[1],gl.GL_TIMESTAMP);queries.append((q,row,label+'_gpu'))
                stack.enter_context(patch.object(obj,name,measured))
            with ExitStack() as stack:
                for obj,name,label,gpu in (
                    (runner.policy,'act','policy',False),(env,'_pre_physics_step','action',False),
                    (env,'_physics_step','simulation',False),(env,'_post_physics_step','observations',False),
                    (eng._presentation,'render','render',False),
                    (pipeline,'original_render','scene',True),(pipeline,'transparent','fluid',True),
                    (pipeline,'ssfr','fluid',True),(renderer,'render','scene_total',True)):
                    hook(stack,obj,name,label,gpu)
                for _ in range(30): runner.tick()
                drain()
                for _ in range(180):
                    row={};current[0]=row;start=time.perf_counter();runner.tick()
                    row['frame']=(time.perf_counter()-start)*1000
                    row['other']=row['frame']-sum(row.get(k,0) for k in ('policy','action','simulation','observations','render'))
                    rows.append(row);current[0]=None
                drain()
            for q,row,label in queries:
                a,b=gl.GLuint64(),gl.GLuint64()
                gl.glGetQueryObjectui64v(q[0],gl.GL_QUERY_RESULT,ctypes.byref(a))
                gl.glGetQueryObjectui64v(q[1],gl.GL_QUERY_RESULT,ctypes.byref(b))
                row[label]=(b.value-a.value)/1e6;gl.glDeleteQueries(2,q)
            with patch.object(wp.array,'numpy',side_effect=AssertionError('readback')):
                for _ in range(2): runner.tick()
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(output/'frame.png')
            material_switches = []
            if args.check_materials:
                material_switches = check_materials()
            report=dict(mode=args.mode,ssfr_preset=args.ssfr_preset,first_render_ms=switch_ms,
                transparent_depth=args.transparent_depth,
                material_switches=material_switches,readback_guard=True,
                particles=eng.get_sim_model().particle_count,seed=1234,warmup=30,frames=180,
                summary={k:statistics([row.get(k,0) for row in rows]) for k in set().union(*rows)},samples=rows)
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('RESULT',args.mode,json.dumps(report['summary']),flush=True)
    finally:
        resource_util.cleanup(playback_stack.close, env.close if env else None,agent.close if agent else None,mp_util.shutdown)


if __name__ == '__main__': main()
