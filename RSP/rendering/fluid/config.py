from dataclasses import dataclass
from ..billboard import LOW_SPEED_COLOR

MODES = ("Original", "Transparent", "SSFR")
SSFR_PRESETS = ("Water", "Velocity")


@dataclass(frozen=True)
class FluidRenderConfig:
    alpha_low: float = .15
    alpha_high: float = .8
    speed_low: float = 2.
    speed_high: float = 6.5
    soft_intersection: float = 1.
    surface_radius: float = 1.5
    filter_radius: int = 12
    ior: float = 1.333
    absorption: tuple = (2., .5, .15)
    base_color: tuple = LOW_SPEED_COLOR
    # Only particles within this many display radii of the nearest surface
    # contribute material attributes. This never changes depth or thickness.
    attribute_depth_range: float = 2.
