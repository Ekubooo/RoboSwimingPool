"""Settings shared by asset preparation and the playback entry point."""
from types import SimpleNamespace


POLICY_HZ = 30


def get_fluid_settings():
    from RSP.physics import fluid_clock

    return fluid_clock.DEFAULT_SETTINGS


def boundary_support_radius(settings=None):
    from RSP.physics.pbf.pbf_helper import init_scene_parameters

    settings = get_fluid_settings() if settings is None else settings
    particles = SimpleNamespace(config=settings.pbf)
    init_scene_parameters(particles)
    return particles.smoothing_length
