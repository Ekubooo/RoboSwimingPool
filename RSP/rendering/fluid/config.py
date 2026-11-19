from dataclasses import dataclass
from ..particles import LOW_SPEED_COLOR

MODES = ("Original", "Transparent", "SSFR")
SSFR_PRESETS = ("Water", "Velocity")
WATER_COLORS = ("Current", "Billboard blue")


@dataclass(frozen=True)
class FluidRenderConfig:
    alpha_low: float = .15
    alpha_high: float = .9
    transparent_alpha_low: float = .8
    transparent_alpha_high: float = 1.
    transparent_layer_blending: bool = True
    transparent_depth_enabled: bool = True
    transparent_depth_strength: float = .5
    speed_low: float = 2.
    speed_high: float = 6.5
    surface_radius: float = 1.5
    filter_radius: int = 12
    ior: float = 1.333
    absorption: tuple = (2., .5, .15)
    # Preserve the original SSFR water material as the default appearance.
    base_color: tuple = (.02, .12, .2)
    billboard_color: tuple = LOW_SPEED_COLOR
    # Only particles within this many display radii of the nearest surface
    # contribute material attributes. This never changes depth or thickness.
    attribute_depth_range: float = 2.
