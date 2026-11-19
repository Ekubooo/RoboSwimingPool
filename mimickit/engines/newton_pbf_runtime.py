"""Compatibility entry for the Newton PBF scene used by legacy callers."""

from RSP.integrations.mimickit.scene import MimicKitScene as PBFScene
from RSP.simulation.scenes.build import runtime_config, sample_sdf

__all__ = ["PBFScene", "runtime_config", "sample_sdf"]
