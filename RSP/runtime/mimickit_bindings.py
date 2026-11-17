"""Construct borrowed MimicKit playback components without advancing simulation."""
from dataclasses import dataclass
from functools import partial
from typing import Any

from .inference import PolicyInference
from .interaction import FluidInteraction
from ..rendering.playback_ui import draw_policy_controls


@dataclass(frozen=True, eq=False)
class PlaybackComponents:
    policy: Any
    viewer: Any | None
    fluid_controls: FluidInteraction | None


def build_playback_components(env, agent, *, policy=None):
    """Bind one playback session; loading, reset and cleanup stay with callers."""
    if policy is None:
        policy = PolicyInference(**agent.get_playback_policy_components())
    viewer = env._engine.get_viewer()
    apply_settings = getattr(env._engine, "set_fluid_controls", None)
    controls = FluidInteraction(apply_settings) if apply_settings is not None else None
    if viewer is not None:
        viewer.bind_playback_controls(controls, env.request_reset)
        viewer.register_ui_callback(partial(
            draw_policy_controls,
            get_scene_sphere=env.get_policy_use_scene_sphere,
            request_scene_sphere=env.request_policy_use_scene_sphere), position="side")
    return PlaybackComponents(policy, viewer, controls)
