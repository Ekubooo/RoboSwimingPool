"""Derive the boundary support radius used by the current fluid settings."""

from types import SimpleNamespace

from RSP.physics.pbf.pbf_helper import init_scene_parameters


def boundary_support_radius(settings):
    particles = SimpleNamespace(config=settings.pbf)
    init_scene_parameters(particles)
    return particles.smoothing_length
