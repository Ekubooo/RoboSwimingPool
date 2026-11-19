"""Read the existing demo arguments, without a second runtime configuration."""
from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import ParseError

import yaml

from RSP.sim_ready.paths import PROJECT_ROOT
from RSP.sim_ready.catalog import AssetRef, scan_sources

CONFIG_ROOT = PROJECT_ROOT / "data/playbacks"


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def project_file(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Configuration path must be a non-empty string")
    path = PROJECT_ROOT / value
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
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
    profile_id: str = "humanoid"


def character_entries():
    record = read_yaml(CONFIG_ROOT / "characters.yaml")
    if record["schema"] != 1:
        raise ValueError("Unsupported humanoid profile catalog version")
    return record["characters"]


def profile_arguments(profile_id):
    from util.arg_parser import ArgParser

    entries = character_entries()
    if profile_id not in entries:
        raise ValueError(f"Unknown humanoid profile: {profile_id}")
    args = ArgParser()
    args.load_file(str(project_file(entries[profile_id]["args"])))
    return args


def profile_scene(args):
    scene = args.parse_string("scene_preset")
    if not scene:
        scene = read_yaml(project_file(args.parse_string("env_config"))).get("scene_preset")
    return project_file(scene)


def load_profile(profile_id="humanoid"):
    args = profile_arguments(profile_id)
    devices = args.parse_strings("devices")
    if (args.parse_string("mode") != "test" or args.parse_int("num_envs") != 1
            or len(devices) != 1 or not args.parse_bool("visualize") or args.parse_bool("video")):
        raise ValueError("Scene selection requires a single-environment test demo with visualization enabled and video disabled")
    files = {key: project_file(args.parse_string(key))
             for key in ("env_config", "engine_config", "agent_config", "model_file")}
    scene = profile_scene(args)
    return PlaybackProfile(scene_preset=scene, **files, device=devices[0],
                           seed=args.parse_int("rand_seed"), out_dir=PROJECT_ROOT / args.parse_string("out_dir"),
                           profile_id=profile_id)


@dataclass(frozen=True)
class CharacterChoice:
    profile_id: str
    label: str
    template: dict | None
    profile: PlaybackProfile | None
    error: str | None = None

    @property
    def asset(self):
        return AssetRef.from_mapping(self.template["character"]) if self.template is not None else None

    def require_available(self):
        if self.error or self.profile is None:
            raise ValueError(self.error or "Humanoid profile unavailable")
        return self.profile


@dataclass(frozen=True)
class LaunchRequest:
    profile_id: str
    scene: Path | dict


def load_characters(*, sources=None):
    """Check authored assets and policy inputs without preparing physics data."""
    sources = scan_sources() if sources is None else sources
    choices = {}
    for profile_id, entry in character_entries().items():
        template = profile = None
        error = None
        try:
            record = read_yaml(profile_scene(profile_arguments(profile_id)))
            if not isinstance(record, dict):
                raise ValueError("Expected a scene mapping for the playback profile")
            ref = AssetRef.from_mapping(record["character"])
            template = record
            profile = load_profile(profile_id)
            source = sources.get(ref)
            if source is None or source.model_path is None:
                raise ValueError(f"Missing humanoid model source: {ref.relative}")
            env, agent = read_yaml(profile.env_config), read_yaml(profile.agent_config)
            motions = read_yaml(project_file(env["motion_file"]))
            for motion in motions["motions"]:
                project_file(motion["file"])
            prior = read_yaml(project_file(agent["smp_prior_cfg"]))
            project_file(agent["smp_prior_model"])
            project_file(prior["env_config"])
        except (OSError, ValueError, KeyError, ParseError, yaml.YAMLError) as exc:
            error = str(exc)
        choices[profile_id] = CharacterChoice(profile_id, entry["name"], template, profile, error)
    return choices


def character_for_scene(scene, characters):
    ref = AssetRef.from_mapping(scene["character"])
    matches = [choice for choice in characters.values() if choice.asset == ref]
    if len(matches) != 1:
        raise ValueError(f"Scene humanoid must have exactly one playback profile: {ref.relative}")
    matches[0].require_available()
    return matches[0]


def load_presets():
    record = read_yaml(CONFIG_ROOT / "presets.yaml")
    if record["schema"] != 1:
        raise ValueError("Unsupported preset catalog version")
    return {name: project_file(path) for name, path in record["presets"].items()}
