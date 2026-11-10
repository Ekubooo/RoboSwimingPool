"""Shared preflight and provenance for the standalone Getup experiment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]
ENV_CONFIG = "data/envs/smp_getup_humanoid_env.yaml"
AGENT_CONFIG = "data/agents/smp_getup_humanoid_agent.yaml"
ENGINE_CONFIG = "data/engines/newton_engine.yaml"
FULL_SAMPLES = 1_000_013_824
FULL_OUTPUT = "output/a5000_getup_2048_full_r01"
SMOKE_OUTPUT = "output/a5000_getup_8_smoke_r01"


def read_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(env_file, agent_file, engine_file):
    env, agent, engine = [read_yaml(p) for p in (env_file, agent_file, engine_file)]
    if env["env_name"] != "task_getup" or agent["agent_name"] != "SMPGetup":
        raise ValueError("Use the dedicated Getup environment and SMPGetup agent.")
    if engine["engine_name"] != "newton":
        raise ValueError("This experiment targets the Newton backend.")
    if agent["enable_gsi"] or env["enable_early_termination"]:
        raise ValueError("Getup requires GSI off and fall termination off.")
    prior_file = Path(agent["smp_prior_cfg"])
    model = Path(agent["smp_prior_model"])
    if not prior_file.is_file() or not model.is_file():
        raise FileNotFoundError(
            f"Keep the trained prior on this machine: {prior_file} and {model}. "
            "output/ is Git-ignored; git pull does not download weights.")
    prior = read_yaml(prior_file)
    prior_env = read_yaml(prior["env_config"])
    for key in ("char_file", "global_obs", "root_height_obs", "enable_tar_obs",
                "num_disc_obs_steps", "disc_dof_vel_obs", "key_bodies"):
        if env[key] != prior_env[key]:
            raise ValueError(f"Prior/runtime feature mismatch: {key}")
    if engine["control_freq"] != prior["control_freq"]:
        raise ValueError("Prior/runtime control frequency mismatch.")
    if env["motion_file"] != prior["motion_file"]:
        raise ValueError("Getup reference dataset differs from the trained prior dataset.")
    dataset = read_yaml(env["motion_file"])
    required = [env["char_file"], prior["env_config"], env["motion_file"]]
    required += [item["file"] for item in dataset["motions"]]
    for name in required:
        if not Path(name).is_file():
            raise FileNotFoundError(f"Missing training asset: {name}")
    return env, agent, engine, {
        "prior_config": str(prior_file), "prior_model": str(model),
        "prior_model_sha256": sha256(model),
        "prior_requested_iterations": prior.get("num_iterations"),
        "asset_sha256": {p: sha256(p) for p in required},
    }


def provenance(extra):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    versions = {}
    for name in ("torch", "newton", "warp-lang", "mujoco", "mujoco-warp", "diffusers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--short"),
            "dependencies": versions, **extra}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
