__all__ = ["HAS_NUMBA"]

# the optional `numba` dependency is probed here and nowhere else
try:
    import numba  # noqa: F401
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
