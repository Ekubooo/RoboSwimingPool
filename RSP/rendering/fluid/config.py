from dataclasses import dataclass

MODES = ("Original", "Transparent", "SSFR")


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
