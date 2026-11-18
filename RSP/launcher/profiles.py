"""Read the existing demo arguments, without a second runtime configuration."""
from dataclasses import dataclass
from pathlib import Path
import yaml

from RSP.sim_ready.paths import PROJECT_ROOT

CONFIG_ROOT = PROJECT_ROOT / "data/playbacks"


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def project_file(value):
    if not isinstance(value, str) or not value:
        raise ValueError("配置路径必须是非空字符串")
    path = PROJECT_ROOT / value
    if not path.is_file():
        raise ValueError(f"缺少文件：{path}")
    return path.resolve()


@dataclass(frozen=True)
class PlaybackProfile:
    scene_preset: Path
    env_config: Path
    engine_config: Path
    agent_config: Path
    model_file: Path
    device: str
    seed: int
    out_dir: Path


def load_profile():
    from util.arg_parser import ArgParser

    args = ArgParser()
    args.load_file(str(PROJECT_ROOT / "args/rsp_pbf_demo_args.txt"))
    devices = args.parse_strings("devices")
    if (args.parse_string("mode") != "test" or args.parse_int("num_envs") != 1
            or len(devices) != 1 or not args.parse_bool("visualize") or args.parse_bool("video")):
        raise ValueError("场景选择入口需要单环境、可视化、无录像的 test demo 配置")
    files = {key: project_file(args.parse_string(key))
             for key in ("env_config", "engine_config", "agent_config", "model_file")}
    scene = project_file(read_yaml(files["env_config"])["scene_preset"])
    return PlaybackProfile(scene_preset=scene, **files, device=devices[0],
                           seed=args.parse_int("rand_seed"), out_dir=PROJECT_ROOT / args.parse_string("out_dir"))


def load_presets():
    record = read_yaml(CONFIG_ROOT / "presets.yaml")
    if record["schema"] != 1:
        raise ValueError("预设目录版本错误")
    return {name: project_file(path) for name, path in record["presets"].items()}
