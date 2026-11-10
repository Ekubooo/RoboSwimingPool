"""Compare PBF backend revisions in separate CUDA processes.

Run from the repository root with the Newton 1.7 environment. For example:
  python tools/check_newton_pbf_refactor.py snapshot --revision before --graph --substeps 8
  python tools/check_newton_pbf_refactor.py snapshot --revision after --graph --substeps 8
  python tools/check_newton_pbf_refactor.py compare before_g1_s8 after_g1_s8

Artifacts go to output/newton_pbf_refactor_checks. The saved original backend is
selected only inside this process; no engine registration or config is changed.
"""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
OUTPUT = ROOT / "output/newton_pbf_refactor_checks"


def backend(revision):
    suffix = "_before_refactor" if revision == "before" else ""
    return importlib.import_module("engines.newton_pbf_engine" + suffix)


def snapshot(args):
    import torch
    import warp as wp
    import yaml

    # Keep JIT artifacts inside the workspace, including in restricted shells.
    wp.config.kernel_cache_dir = str(OUTPUT / "warp_cache")
    (OUTPUT / "tmp").mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(OUTPUT / "tmp")
    module = backend(args.revision)
    sys.modules["engines.newton_pbf_engine"] = module
    from envs.pbf_play_env import PBFPlayEnv
    from envs.base_env import EnvMode
    from util.util import set_rand_seed
    import util.resource_util as resources

    set_rand_seed(1234)
    with (ROOT / "data/engines/newton_pbf_engine.yaml").open() as f:
        config = yaml.safe_load(f)
    with (ROOT / "data/envs/pbf_play_env.yaml").open() as f:
        env_config = yaml.safe_load(f)
    config["enable_cuda_graph"] = args.graph
    config["sim_freq"] = config["control_freq"] * args.substeps
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{args.revision}_g{int(args.graph)}_s{args.substeps}"
    env = resources.construct(PBFPlayEnv, env_config, config, 1, "cuda:0", False)
    arrays = {}
    try:
        env.set_mode(EnvMode.TEST)
        eng = env._engine
        state = eng.get_sim_state()
        model = eng.get_sim_model()
        metadata = {
            "api": {key: str(inspect.signature(value))
                    for key, value in inspect.getmembers(module.NewtonEngine, inspect.isfunction)
                    if not key.startswith("_")},
            "model_counts": [model.body_count, model.joint_count, model.joint_coord_count,
                             model.joint_dof_count, model.particle_count],
            "body_names": [eng.get_obj_body_names(i) for i in range(eng.get_objs_per_env())],
            "object_types": [eng.get_obj_type(i).name for i in range(eng.get_objs_per_env())],
            "graph": args.graph, "substeps": args.substeps,
        }
        getters = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
                   "dof_forces", "body_pos", "body_rot", "body_vel", "body_ang_vel",
                   "contact_forces", "ground_contact_forces")
        pointers = {attr: getattr(state.raw_state, attr).ptr
                    for attr in ("joint_q", "joint_qd", "body_q", "body_qd", "particle_q")}
        tensor_pointers = {(attr, i): getattr(eng, "get_" + attr)(i).data_ptr()
                           for attr in getters for i in range(eng.get_objs_per_env())}

        def capture(label, obs):
            wp.synchronize()
            for attr in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f", "particle_q", "particle_qd"):
                arrays[f"{label}/raw/{attr}"] = getattr(state.raw_state, attr).numpy().copy()
            for attr in getters:
                for i in range(eng.get_objs_per_env()):
                    value = getattr(eng, "get_" + attr)(i)
                    assert value.data_ptr() == tensor_pointers[attr, i], (label, attr, "view replaced")
                    arrays[f"{label}/obj{i}/{attr}"] = value.cpu().numpy().copy()
            arrays[f"{label}/obs"] = obs.cpu().numpy().copy()
            arrays[f"{label}/time"] = np.asarray(eng.get_sim_time())
            arrays[f"{label}/external_force"] = state._wp_body_force.numpy().copy()
            arrays[f"{label}/solver_step"] = np.asarray(eng._runtime._solver.rigid._step)
            for attr, ptr in pointers.items():
                assert getattr(state.raw_state, attr).ptr == ptr, (label, attr, "buffer replaced")

        obs, _ = env.reset()
        capture("initial", obs)
        # Fixed commands, independent of trajectory differences between backends.
        action = eng.get_dof_pos(0).clone()
        for step in range(6):
            if step == 2:
                eng.set_body_forces(None, 0, 0, torch.tensor([[3., -2., 1.]], device="cuda:0"))
            obs, *_ = env.step(action + (0.005 * step))
            capture(f"step{step}", obs)
        obs, _ = env.reset()
        capture("reset", obs)
        assert eng.get_sim_time() == 0
        for step in range(2):
            obs, *_ = env.step(action)
            capture(f"resumed{step}", obs)
        for key, array in arrays.items():
            assert np.isfinite(array).all(), f"Non-finite data: {key}"
        np.savez_compressed(OUTPUT / f"{name}.npz", **arrays)
        (OUTPUT / f"{name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"SNAPSHOT PASS: {name}, {len(arrays)} arrays, stable storage, finite state", flush=True)
    finally:
        env.close()
        env.close()


def compare(args):
    first = np.load(OUTPUT / f"{args.first}.npz")
    second = np.load(OUTPUT / f"{args.second}.npz")
    a_meta = json.loads((OUTPUT / f"{args.first}.json").read_text(encoding="utf-8"))
    b_meta = json.loads((OUTPUT / f"{args.second}.json").read_text(encoding="utf-8"))
    assert a_meta == b_meta, "API or model metadata differs"
    assert set(first.files) == set(second.files), "Snapshot fields differ"
    failures = []
    exact = 0
    for key in first.files:
        a, b = first[key], second[key]
        assert a.shape == b.shape and a.dtype == b.dtype, (key, a.shape, b.shape, a.dtype, b.dtype)
        if np.array_equal(a, b):
            exact += 1
        elif key.endswith(("/time", "/solver_step")) or not np.allclose(a, b, rtol=1e-5, atol=1e-6):
            failures.append({"field": key, "max_abs": float(np.max(np.abs(a - b)))})
    report = {"first": args.first, "second": args.second, "arrays": len(first.files),
              "exact": exact, "failures": failures, "rtol": 1e-5, "atol": 1e-6}
    (OUTPUT / f"compare_{args.first}_{args.second}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "failures": failures[:12]}, indent=2))
    if failures:
        raise SystemExit(1)


def integration(args):
    """Exercise the existing SMP checkpoint, GSI resets and headless GL recorder."""
    import torch
    import warp as wp
    import yaml
    from PIL import Image

    wp.config.kernel_cache_dir = str(OUTPUT / "warp_cache")
    (OUTPUT / "tmp").mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(OUTPUT / "tmp")
    module = backend(args.revision)
    sys.modules["engines.newton_pbf_engine"] = module
    from envs.pbf_play_env import PBFPlayEnv
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util.util import set_rand_seed
    from util import mp_util, resource_util

    set_rand_seed(1234)
    mp_util.init(0, 1, "cuda:0", None)
    with (ROOT / "data/engines/newton_pbf_engine.yaml").open() as f:
        config = yaml.safe_load(f)
    with (ROOT / "data/envs/pbf_play_env.yaml").open() as f:
        env_config = yaml.safe_load(f)
    env = agent = None
    try:
        env = resource_util.construct(PBFPlayEnv, env_config, config, 1, "cuda:0", False, record_video=True)
        agent = build_agent("output/a5000_dodgeball_2048_full_r01/agent_config.yaml", env, "cuda:0")
        agent.load("output/a5000_dodgeball_2048_full_r01/model.pt")
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        observations = []
        with torch.no_grad():
            obs, info = env.reset()
            for step in range(8):
                if step == 6:
                    obs, info = env.reset()
                action, _ = agent._decide_action(obs, info)
                assert torch.isfinite(action).all()
                obs, _, _, info = env.step(action)
                assert torch.isfinite(obs).all()
                observations.append(obs.cpu().numpy().copy())
        video = env._engine.get_video_recording()
        assert video.get_num_frames() == 8
        frames = video.get_frames()
        assert all(frame.shape == (480, 854, 3) and frame.dtype == np.uint8 for frame in frames)
        assert all(float(frame.std()) > 1 for frame in frames), "Recorder returned blank frames"
        prefix = OUTPUT / f"policy_{args.revision}"
        video.save(str(prefix.with_suffix(".mp4")))
        Image.fromarray(frames[-1]).save(prefix.with_suffix(".png"))
        np.save(prefix.with_suffix(".npy"), np.asarray(observations))
        print(f"INTEGRATION PASS: checkpoint + GSI reset + 8 policy steps + GL frames/video: {prefix}", flush=True)
    finally:
        resource_util.cleanup(env.close if env is not None else None,
                              agent.close if agent is not None else None, mp_util.shutdown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("--revision", choices=("before", "after"), required=True)
    capture.add_argument("--graph", action="store_true")
    capture.add_argument("--substeps", type=int, default=8)
    capture.add_argument("--name")
    capture.set_defaults(run=snapshot)
    comparison = commands.add_parser("compare")
    comparison.add_argument("first")
    comparison.add_argument("second")
    comparison.set_defaults(run=compare)
    smoke = commands.add_parser("integration")
    smoke.add_argument("--revision", choices=("before", "after"), default="after")
    smoke.set_defaults(run=integration)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
