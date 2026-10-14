from .heat_diffusion_numba import HeatDiffusionNumba
from .heat_diffusion_scipy import HeatDiffusionScipy

__all__ = ["HAS_NUMBA", "HeatDiffusionNumba", "HeatDiffusionScipy"]

# the optional `numba` dependency is probed here and nowhere else
try:
    import numba  # noqa: F401
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
