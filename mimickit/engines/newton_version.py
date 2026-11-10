"""Fail early when an engine configuration uses the other Newton environment."""
from importlib.metadata import version


def check_newton_version(backend):
    expected, environment = {"newton": ("1.0", "mimickit-newton"),
                             "newton_1_7": ("1.7", "WarpSim")}[backend]
    installed = version("newton")
    if installed.split(".")[:2] != expected.split("."):
        raise RuntimeError(
            f"Backend {backend} requires Newton {expected}.x, found {installed}. "
            f"Use Conda environment {environment} and its matching engine config.")
