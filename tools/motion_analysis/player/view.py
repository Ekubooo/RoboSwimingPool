"""Display-only layouts and camera poses for motion PCA comparisons."""

from dataclasses import dataclass
from typing import Any

import numpy as np


VIEW_MODES = ("comparison", "local", "classroom")
VIEW_LABELS = ("Original", "Local poses", "Classroom")
ACTOR_COLORS = ((0.15, 0.45, 0.9), (0.95, 0.45, 0.1),
                (0.2, 0.7, 0.4), (0.65, 0.35, 0.85))


def cumulative_selections(rank: int) -> tuple[tuple[int, ...], ...]:
    """Small-rank data uses all available components in the remaining slots."""
    return tuple(tuple(range(min(k, rank))) for k in (1, 2, 3))


@dataclass(frozen=True)
class ViewLayout:
    offsets: np.ndarray
    visible: tuple[bool, ...]
    camera_position: np.ndarray
    camera_target: np.ndarray


def build_layout(view: str, root_height: float, spacing: float = 2.5) -> ViewLayout:
    """Keep classroom slots in the image plane, at identical camera depth."""
    if view not in VIEW_MODES:
        raise ValueError(f"Unknown playback view: {view}")
    back = np.array([0.6, -0.8, 0.55])
    back /= np.linalg.norm(back)
    right = np.array([0.8, 0.6, 0.0])
    screen_up = np.cross(back, right)
    offsets = np.zeros((4, 3), dtype=np.float32)
    if view == "classroom":
        lower_center = -3.0 * screen_up
        offsets[1:] = np.stack([lower_center + x * right for x in (-spacing, 0.0, spacing)])
        visible = (True, True, True, True)
    else:
        offsets[1] = ([0.0, spacing, 0.0] if view == "comparison" else spacing * right)
        visible = (True, True, False, False)
    target = np.array([0.0, 0.0, root_height + 0.25]) + offsets[list(visible)].mean(axis=0)
    # Leave room for Newton's native sidebar on the left of the GL viewport.
    target -= right * (1.0 if view == "classroom" else 0.6)
    distance = 9.5 if view == "classroom" else 7.5
    return ViewLayout(offsets, visible, target + back * distance, target)


class FixedCamera:
    """A repeatable camera; root motion and viewer input cannot move it."""

    def __init__(self, engine: Any, layout: ViewLayout):
        self.engine = engine
        self.pos = layout.camera_position.copy()
        self.target = layout.camera_target.copy()
        self.update()

    def update(self) -> None:
        self.engine.set_camera_pose(self.pos, self.target)
