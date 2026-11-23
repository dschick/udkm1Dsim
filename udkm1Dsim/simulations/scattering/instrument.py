#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Daniel Schick
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

r"""Profiles, quadrature nodes, convolutions, and footprints of the
instrument function of the scattering simulations, see
:class:`~udkm1Dsim.simulations.parameter_groups.InstrumentParameters` and
:meth:`~udkm1Dsim.simulations.scattering.scattering.Scattering.apply_instrument`.

All profiles :math:`p(x)` are area-normalized and given in units of their
full width at half maximum (FWHM), which is the full width of a box, see
:func:`profile`. The measured intensity is the ideal one averaged over the
profiles of the angle and the photon energy with the FWHM
:math:`\Delta\vartheta` and :math:`\Delta E`,

.. math::

    \langle R\rangle(E, \vartheta) = \iint p_E(x_e)\, p_\vartheta(x_t)\,
    R(E + x_e \Delta E, \vartheta + x_t \Delta\vartheta)\, \mathrm{d}x_e\,
    \mathrm{d}x_t,

which is approximated by a quadrature with the nodes :math:`x_i` and their
weights :math:`w_i` of :func:`profile_nodes`, which sum to 1. Every node is a
shifted copy of the experiment, at which the simulation is calculated once.
With the components :math:`m` of the ``spectrum`` of the photon energies
:math:`E_m = f_m E + \delta_m`, the normalized weights :math:`W_m`, and the
widths :math:`\Delta E_m`, the intensity with the instrument is

.. math::

    I(E, \vartheta) = I_0 \sum_m W_m \sum_e w_e \sum_t w_t\,
    F(\vartheta_t)\, R(E_m + x_e \Delta E_m, \vartheta_t) + I_\mathrm{bkg},
    \quad \vartheta_t = \vartheta + x_t \Delta\vartheta,

where the :func:`footprint` :math:`F` scales the reflected intensities only.
Hence, the simulation is calculated for components :math:`\times` energy
nodes :math:`\times` angle nodes. An axis in the ``'convolution'`` mode has
the single node :math:`x = 0` with :math:`w = 1`, and the sum is convolved
along the grid of the experiment instead, see :func:`convolve_on_grid`. The
``'qz'`` mode of the photon energy applies its nodes as shifts of the angle
with the same scattering vector,
:math:`\sin\vartheta' = \sin\vartheta_t\,(1 + x_e \Delta E_m / E_m)`.

"""

__all__ = [
    "profile",
    "PROFILES",
    "NODES",
    "MAX_NODE_SPACING",
    "profile_nodes",
    "check_nodes",
    "MIN_PROFILE_AREA",
    "profile_area",
    "check_range",
    "convolution_matrix",
    "is_resolved",
    "convolve",
    "convolve_on_grid",
    "convolve_delays",
    "footprint",
]

__docformat__ = "restructuredtext"

import warnings

import numpy as np
from scipy import sparse
from scipy.signal import oaconvolve
from scipy.special import erf

#: tuple[str]: profiles of :func:`profile`, besides ``'none'`` for no profile
PROFILES = ("box", "gauss", "lorentz", "pseudo_voigt")
#: tuple[str]: nodes of :func:`profile_nodes`
NODES = ("uniform", "gaussian_quadrature")
#: float: largest spacing of uniform nodes (FWHM), i.e. the default 7 nodes
#: within :math:`\pm 2` FWHM, see :func:`check_nodes`
MAX_NODE_SPACING = 2 / 3
#: float: smallest area of a profile within its ``range``, below which the
#: truncation of its wings is warned for, see :func:`check_range`
MIN_PROFILE_AREA = 0.95

# sigma of a Gaussian of unit FWHM
_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def profile(shape, x, eta=0.5):
    r"""profile

    Area-normalized profile of unit FWHM at ``x`` in units of the FWHM:

    * ``'box'``: :math:`1` for :math:`|x| < 1/2` and :math:`1/2` at its edges
    * ``'gauss'``: :math:`2\sqrt{\ln 2/\pi}\,\exp(-4 \ln 2\, x^2)`
    * ``'lorentz'``: :math:`2 / (\pi (1 + 4 x^2))`
    * ``'pseudo_voigt'``: :math:`\eta` times the Lorentzian plus
      :math:`1 - \eta` times the Gaussian

    Args:
        shape (str): profile.
        x (ndarray[float]): arguments in units of the FWHM.
        eta (float, optional): Lorentzian fraction of the pseudo-Voigt
            profile. Defaults to 0.5.

    Returns:
        p (ndarray[float]): profile.

    """
    x = np.asarray(x, dtype=float)
    if shape == "box":
        # the edges have half the weight, as by the trapezoidal rule
        inside = (np.abs(x) < 0.5 - 1e-9).astype(float)
        return inside + 0.5 * (np.abs(np.abs(x) - 0.5) <= 1e-9)
    gauss = 2 * np.sqrt(np.log(2) / np.pi) * np.exp(-4 * np.log(2) * x**2)
    if shape == "gauss":
        return gauss
    lorentz = 2 / (np.pi * (1 + 4 * x**2))
    if shape == "lorentz":
        return lorentz
    if shape == "pseudo_voigt":
        if not 0 <= eta <= 1:
            raise ValueError(
                f"The mixing eta of a pseudo-Voigt profile must be in [0, 1], not {eta}!"
            )
        return eta * lorentz + (1 - eta) * gauss
    raise ValueError(f"Unknown profile {shape!r}!")


def profile_nodes(shape, nodes="uniform", points=7, range=2.0, eta=0.5):
    r"""profile_nodes

    Nodes :math:`x_i` in units of the FWHM and normalized weights
    :math:`w_i` of the quadrature of a profile :math:`p(x)`, see
    :func:`profile` and the module :mod:`.instrument`, for :math:`n`
    ``points`` and the ``range`` :math:`r`:

    * ``'uniform'``: equidistant nodes
      :math:`x_i = -r + 2 r\, i / (n - 1)` with the weights of the
      trapezoidal rule :math:`w_i \propto p(x_i)`, halved at both ends, as
      the full convolution of GenX, which are robust for narrow features,
      e.g. Bragg peaks or resonances. A box has the centers of :math:`n`
      equal parts :math:`x_i = (i + 1/2) / n - 1/2` with :math:`w_i = 1/n`.
    * ``'gaussian_quadrature'``: the nodes of the Gaussian quadrature, i.e. the
      Gauss-Hermite nodes of a Gaussian of the standard deviation
      :math:`\sigma = 1 / (2\sqrt{2 \ln 2})` and the Gauss-Legendre nodes
      of a box, which integrate smooth
      functions exactly with few nodes, but are not defined for the other
      profiles.
    * ``'none'``: the single node :math:`x = 0` with :math:`w = 1`.

    An odd number of nodes contains the unshifted node :math:`x = 0`.

    Args:
        shape (str): profile.
        nodes (str, optional): ``'uniform'`` or ``'gaussian_quadrature'``.
            Defaults to ``'uniform'``.
        points (int, optional): number of nodes. Defaults to 7.
        range (float, optional): range of the uniform nodes in units of the
            FWHM. Defaults to 2.
        eta (float, optional): Lorentzian fraction of the pseudo-Voigt
            profile. Defaults to 0.5.

    Returns:
        (tuple):
        - *x (ndarray[float])* - nodes in units of the FWHM.
        - *w (ndarray[float])* - normalized weights.

    """
    if shape == "none":
        return np.zeros(1), np.ones(1)
    if shape not in PROFILES:
        raise ValueError(f"The profile must be one of {', '.join(PROFILES)}, not {shape!r}!")
    if nodes not in NODES:
        raise ValueError(f"The nodes must be 'uniform' or 'gaussian_quadrature', not {nodes!r}!")
    if isinstance(points, bool) or int(points) != points or points < 1:
        raise ValueError(f"The number of nodes must be a positive integer, not {points!r}!")
    points = int(points)
    if not range > 0:
        raise ValueError(f"The range of the nodes must be positive, not {range}!")
    if not 0 <= eta <= 1:
        raise ValueError(f"The mixing eta of a pseudo-Voigt profile must be in [0, 1], not {eta}!")
    if nodes == "gaussian_quadrature":
        if shape == "gauss":
            x, w = np.polynomial.hermite_e.hermegauss(points)
            return x * _SIGMA, w / np.sum(w)
        if shape == "box":
            x, w = np.polynomial.legendre.leggauss(points)
            return x / 2, w / np.sum(w)
        raise ValueError(
            "The Gaussian quadrature is defined for a 'gauss' and a 'box' profile only, "
            f"but not for {shape!r}, which needs 'uniform' nodes!"
        )
    if shape == "box":
        x = (np.arange(points) + 0.5) / points - 0.5
        return x, np.full(points, 1 / points)
    if points == 1:
        return np.zeros(1), np.ones(1)
    x = np.linspace(-range, range, points)
    # trapezoidal rule, as for the full convolution of GenX
    w = profile(shape, x, eta)
    w[[0, -1]] /= 2
    return x, w / np.sum(w)


def check_nodes(name, label, shape, nodes, points, range, stacklevel=2):
    """check_nodes

    Warns for too few nodes of :func:`profile_nodes`, i.e. a single node,
    which has no effect, or ``'uniform'`` nodes, which are more than
    :data:`MAX_NODE_SPACING` FWHM apart, so that their weights do not follow
    the profile, e.g. three nodes within :math:`\\pm 2` FWHM, whose outer
    weights vanish, or two nodes at :math:`\\pm 2` FWHM.

    Args:
        name (str): prefix of the ``*_points`` and ``*_range`` of the
            nodes in the warnings, e.g. ``'theta'``.
        label (str): description of the profile in the warnings, e.g.
            ``'theta resolution'``.
        shape (str): profile.
        nodes (str): ``'uniform'`` or ``'gaussian_quadrature'``.
        points (int): number of nodes.
        range (float): range of the uniform nodes in units of the FWHM.
        stacklevel (int, optional): stack level of the warnings, e.g. of the
            caller of the caller. Defaults to 2.

    """
    if shape == "none":
        return
    points = int(points)
    range = float(range)
    if points == 1:
        warnings.warn(
            f"The {name}_points is 1, so that the single node of the quadrature is the "
            f"unshifted point and the {label} has no effect!",
            stacklevel=stacklevel + 1,
        )
    elif nodes == "uniform" and shape != "box":
        spacing = 2 * range / (points - 1)
        if spacing > MAX_NODE_SPACING * (1 + 1e-9):
            needed = int(np.ceil(2 * range / MAX_NODE_SPACING)) + 1
            warnings.warn(
                f"The {points} uniform nodes of the {label} are {spacing:.2g} FWHM apart "
                f"within the {name}_range of {range:g} FWHM, which is too coarse for the "
                f"profile, so that the {label} is wrong, e.g. too small or too large! Use at "
                f"least {needed} {name}_points for this range.",
                stacklevel=stacklevel + 1,
            )


def profile_area(shape, range=2.0, eta=0.5):
    r"""profile_area

    Area of the profile of :func:`profile` within :math:`\pm` ``range`` in
    units of its FWHM, i.e. the part of the profile, which is kept by the
    truncation of the uniform nodes and of the convolution, as the weights
    are normalized to this part, e.g. 0.844 of a Lorentzian within
    :math:`\pm 2` FWHM, but :math:`1 - 2\times10^{-5}` of a Gaussian:

    * ``'box'``: :math:`1` for a range of at least :math:`1/2`
    * ``'gauss'``: :math:`\mathrm{erf}(2 \sqrt{\ln 2}\, r)`
    * ``'lorentz'``: :math:`2 \arctan(2 r) / \pi`
    * ``'pseudo_voigt'``: :math:`\eta` times the Lorentzian plus
      :math:`1 - \eta` times the Gaussian

    Args:
        shape (str): profile.
        range (float, optional): range in units of the FWHM. Defaults to 2.
        eta (float, optional): Lorentzian fraction of the pseudo-Voigt
            profile. Defaults to 0.5.

    Returns:
        area (float): area within the range.

    """
    if shape == "box":
        return float(min(range / 0.5, 1.0))
    gauss = float(erf(2 * np.sqrt(np.log(2)) * range))
    lorentz = float(2 * np.arctan(2 * range) / np.pi)
    if shape == "gauss":
        return gauss
    if shape == "lorentz":
        return lorentz
    if shape == "pseudo_voigt":
        return eta * lorentz + (1 - eta) * gauss
    raise ValueError(f"Unknown profile {shape!r}!")


def check_range(name, label, shape, range, eta=0.5, stacklevel=2):
    """check_range

    Warns for a ``range`` of a profile, within which less than
    :data:`MIN_PROFILE_AREA` of its area lies, see :func:`profile_area`,
    e.g. of a Lorentzian within the default range of 2 FWHM, so that its
    wings are missing.

    Args:
        name (str): prefix of the ``*_range`` in the warnings, e.g.
            ``'theta'``.
        label (str): description of the profile in the warnings, e.g.
            ``'theta resolution'``.
        shape (str): profile.
        range (float): range in units of the FWHM.
        eta (float, optional): Lorentzian fraction of the pseudo-Voigt
            profile. Defaults to 0.5.
        stacklevel (int, optional): stack level of the warnings, e.g. of the
            caller of the caller. Defaults to 2.

    """
    if shape in ("none", "box", "gauss"):
        return
    area = profile_area(shape, float(range), float(eta))
    if area < MIN_PROFILE_AREA:
        needed = np.tan(np.pi * MIN_PROFILE_AREA / 2) / 2
        if shape == "pseudo_voigt":
            needed = np.tan(np.pi / 2 * (1 - (1 - MIN_PROFILE_AREA) / max(eta, 1e-12))) / 2
        warnings.warn(
            f"Only {area:.1%} of the {shape} profile of the {label} lie within the "
            f"{name}_range of {float(range):g} FWHM, so that its wings are missing! Use a "
            f"{name}_range of at least {needed:.2g} FWHM with enough {name}_points.",
            stacklevel=stacklevel + 1,
        )


def convolution_matrix(x, width, shape, range=2.0, eta=0.5):
    r"""convolution_matrix

    Sparse matrix :math:`K` of the convolution :math:`y' = K y` of values
    :math:`y` on the grid :math:`x`, which can be non-equidistant, with a
    profile of the FWHM ``width``, which can differ for each point, see
    :func:`profile`. The rows

    .. math:: K_{ij} \propto p\left(\frac{x_j - x_i}{w_i}\right) \Delta x_j

    with the trapezoidal weights :math:`\Delta x_j` of the grid are
    normalized and truncated at :math:`\pm` ``range`` times the width, or
    at the box, as well as at the ends of the grid. A width below the
    spacing of the grid keeps the values.

    Each row is the trapezoidal rule of the integral over the profile at
    :math:`x_i` with the neighboring points of the grid as nodes. For an
    equidistant grid and a constant width, all rows are the same profile
    shifted by one point, i.e. :math:`K` is a Toeplitz matrix and
    :math:`K y` is the discrete convolution, which is calculated by the FFT
    in :func:`convolve_on_grid` instead. For a varying width, every row has
    the profile of its own width :math:`w_i`, i.e. of the point of the
    experiment, which is no convolution in the strict sense anymore and
    has no FFT, while the truncation keeps :math:`K` sparse.

    Args:
        x (ndarray[float]): grid of shape (N,).
        width (ndarray[float]): FWHM of the profile of shape (N,) or a scalar.
        shape (str): profile.
        range (float, optional): range in units of the FWHM. Defaults to 2.
        eta (float, optional): Lorentzian fraction of the pseudo-Voigt
            profile. Defaults to 0.5.

    Returns:
        K (scipy.sparse.csr_array): convolution matrix of shape (N, N).

    """
    x = np.asarray(x, dtype=float)
    N = np.size(x)
    width = np.broadcast_to(np.abs(np.asarray(width, dtype=float)), (N,))
    if shape == "none" or N < 2:
        return sparse.csr_array(sparse.identity(N))
    order = np.argsort(x)
    xs = x[order]
    # trapezoidal weights of the sorted grid
    dx = np.gradient(xs)
    reach = 0.5 if shape == "box" else range
    rows, cols, values = [], [], []
    for i in np.arange(N):
        center, w = x[i], width[i]
        # a tolerance of the window for its edges on the grid
        lo = np.searchsorted(xs, center - reach * w * (1 + 1e-9), side="left")
        hi = np.searchsorted(xs, center + reach * w * (1 + 1e-9), side="right")
        j = np.arange(lo, hi)
        k = profile(shape, (xs[j] - center) / w, eta) * dx[j] if w > 0 else np.zeros(0)
        if np.sum(k) <= 0:
            # a width below the spacing of the grid keeps the value
            j, k = np.array([np.argmin(np.abs(xs - center))]), np.ones(1)
        rows.append(np.full(j.size, i))
        cols.append(order[j])
        values.append(k / np.sum(k))
    return sparse.csr_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    )


def is_resolved(x, width, shape, range=2.0):
    """is_resolved

    Whether a convolution on the grid :math:`x` with a profile of the FWHM
    ``width`` changes the values at any point of the grid, i.e. whether at
    least one point has a neighbor within :math:`\\pm` ``range`` times its
    width, or within the box, see :func:`convolution_matrix`. Otherwise, a
    single point or a width below the spacing of the grid keeps the values.

    Args:
        x (ndarray[float]): grid of shape (N,).
        width (ndarray[float]): FWHM of the profile of shape (N,) or a scalar.
        shape (str): profile.
        range (float, optional): range in units of the FWHM. Defaults to 2.

    Returns:
        resolved (bool): the convolution changes the values.

    """
    x = np.asarray(x, dtype=float).ravel()
    N = np.size(x)
    if shape == "none" or N < 2:
        return False
    width = np.broadcast_to(np.abs(np.asarray(width, dtype=float)), (N,))
    reach = (0.5 if shape == "box" else range) * width * (1 + 1e-9)
    xs = np.sort(x)
    lo = np.searchsorted(xs, x - reach, side="left")
    hi = np.searchsorted(xs, x + reach, side="right")
    return bool(np.any((hi - lo > 1) & (width > 0)))


def convolve(values, axis, matrix):
    """convolve

    Convolution of ``values`` along their ``axis`` by a
    :func:`convolution_matrix`.

    Args:
        values (ndarray[float]): values.
        axis (int): axis of the grid.
        matrix (scipy.sparse.csr_array): convolution matrix.

    Returns:
        values (ndarray[float]): convolved values of the same shape.

    """
    moved = np.moveaxis(np.asarray(values), axis, 0)
    shape = moved.shape
    result = matrix @ moved.reshape(shape[0], -1)
    return np.moveaxis(np.asarray(result).reshape(shape), 0, axis)


def _interp_last(xp, fp, x):
    """linear interpolation of ``fp`` of the sorted grid ``xp`` along its
    last axis at ``x``"""
    i = np.clip(np.searchsorted(xp, x) - 1, 0, np.size(xp) - 2)
    t = (x - xp[i]) / (xp[i + 1] - xp[i])
    return fp[..., i] * (1 - t) + fp[..., i + 1] * t


def convolve_on_grid(values, axis, x, width, shape, range=2.0, eta=0.5):
    """convolve_on_grid

    Convolution of ``values`` along their ``axis`` on the grid :math:`x`
    with a profile of the FWHM ``width``, see :func:`convolution_matrix`.
    For a constant width, the values are convolved by the FFT, i.e. by the
    overlap-add method of :func:`scipy.signal.oaconvolve`, on an equidistant
    grid of the smallest spacing of :math:`x`, to which they
    are interpolated linearly for a non-equidistant grid, as for
    :func:`~udkm1Dsim.helpers.conv_with_function`, and normalized at the
    ends of the grid. A varying width is convolved exactly by the
    :func:`convolution_matrix`.

    Args:
        values (ndarray[float]): values.
        axis (int): axis of the grid.
        x (ndarray[float]): grid of shape (N,).
        width (ndarray[float]): FWHM of the profile of shape (N,) or a scalar.
        shape (str): profile.
        range (float, optional): range in units of the FWHM. Defaults to 2.
        eta (float, optional): Lorentzian fraction of the pseudo-Voigt
            profile. Defaults to 0.5.

    Returns:
        values (ndarray[float]): convolved values of the same shape.

    """
    x = np.asarray(x, dtype=float)
    N = np.size(x)
    width = np.broadcast_to(np.abs(np.asarray(width, dtype=float)), (N,))
    values = np.asarray(values, dtype=float)
    if shape == "none" or N < 2:
        return values
    order = np.argsort(x)
    xs = x[order]
    dx = np.min(np.diff(xs))
    w = width[0]
    reach = 0.5 if shape == "box" else range
    points = int(round((xs[-1] - xs[0]) / dx)) + 1 if dx > 0 else 0
    if not np.allclose(width, w) or dx <= 0 or points > 20 * N:
        return convolve(values, axis, convolution_matrix(x, width, shape, range, eta))
    half = int(np.floor(reach * w / dx * (1 + 1e-9)))
    if w == 0 or half == 0:
        # a width below the spacing of the grid keeps the values
        return values
    v = np.moveaxis(values, axis, -1)[..., order]
    uniform = np.allclose(np.diff(xs), dx, rtol=1e-6, atol=0)
    grid = xs if uniform else np.linspace(xs[0], xs[-1], points)
    if not uniform:
        v = _interp_last(xs, v, grid)
    kernel = profile(shape, np.arange(-half, half + 1) * dx / w, eta)
    kernel_nd = kernel.reshape((1,) * (v.ndim - 1) + (-1,))
    result = oaconvolve(v, kernel_nd, mode="same", axes=-1)
    result = result / oaconvolve(np.ones(np.size(grid)), kernel, mode="same")
    if not uniform:
        result = _interp_last(grid, result, xs)
    unsorted = np.empty_like(result)
    unsorted[..., order] = result
    return np.moveaxis(unsorted, -1, axis)


def convolve_delays(values, axis, delays, width, shape, range=2.0, eta=0.5):
    """convolve_delays

    Convolution of ``values`` along their ``axis`` of the ``delays``, which
    can be non-equidistant, with a profile of the FWHM ``width``, e.g. the
    temporal resolution of a pump-probe experiment, see
    :func:`convolution_matrix`. In contrast to :func:`convolve_on_grid`,
    the values are held at their first and last value beyond the ends of
    the delays, e.g. the unexcited sample before the excitation, instead of
    normalizing the truncated profile.

    Args:
        values (ndarray[float]): values.
        axis (int): axis of the delays.
        delays (ndarray[float]): delays of shape (N,).
        width (ndarray[float]): FWHM of the profile in the unit of the
            delays of shape (N,) or a scalar.
        shape (str): profile.
        range (float, optional): range in units of the FWHM. Defaults to 2.
        eta (float, optional): Lorentzian fraction of the pseudo-Voigt
            profile. Defaults to 0.5.

    Returns:
        values (ndarray[float]): convolved values of the same shape.

    """
    delays = np.asarray(delays, dtype=float)
    N = np.size(delays)
    width = np.broadcast_to(np.abs(np.asarray(width, dtype=float)), (N,))
    values = np.asarray(values, dtype=float)
    if shape == "none" or N < 2 or not np.any(width > 0):
        return values
    order = np.argsort(delays)
    ds, ws = delays[order], width[order]
    if np.any(np.diff(ds) <= 0):
        raise ValueError("The delays of a temporal convolution must be distinct!")
    reach = (0.5 if shape == "box" else range) * np.max(ws)

    def padding(step):
        # at most about 64 points beyond each end, which hold the end value
        step = max(step, reach / 64)
        return step * np.arange(1, int(np.ceil(reach * (1 + 1e-9) / step)) + 1)

    left = ds[0] - padding(ds[1] - ds[0])[::-1]
    right = ds[-1] + padding(ds[-1] - ds[-2])
    nl, nr = left.size, right.size
    x = np.concatenate((left, ds, right))
    w = np.concatenate((np.full(nl, ws[0]), ws, np.full(nr, ws[-1])))
    K = convolution_matrix(x, w, shape, range, eta)[nl : nl + N]
    # the columns of the padding are folded onto the first and last delay
    columns = np.concatenate((np.zeros(nl, dtype=int), np.arange(N), np.full(nr, N - 1)))
    fold = sparse.csr_array((np.ones(x.size), (np.arange(x.size), columns)), shape=(x.size, N))
    K = sparse.csr_array(K @ fold)
    sorted_values = np.take(values, order, axis=axis)
    result = convolve(sorted_values, axis, K)
    unsorted = np.empty_like(result)
    index = [slice(None)] * result.ndim
    index[axis] = order
    unsorted[tuple(index)] = result
    return unsorted


def footprint(shape, theta, beam_width, sample_length):
    r"""footprint

    Fraction of a beam of the FWHM or full ``beam_width``, which hits a
    sample of the ``sample_length`` at the grazing incidence angle
    :math:`\vartheta`

    * ``'box'``: :math:`\min(L \sin\vartheta / w, 1)`
    * ``'gauss'``: :math:`\mathrm{erf}(L \sin\vartheta / (2 \sqrt{2} \sigma))`
      with :math:`\sigma = w / (2 \sqrt{2 \ln 2})`

    which is zero for negative angles.

    Args:
        shape (str): ``'none'``, ``'box'``, or ``'gauss'`` beam.
        theta (ndarray[float]): grazing incidence angles (rad).
        beam_width (float): FWHM or full width of the beam.
        sample_length (float): length of the sample in the unit of the
            ``beam_width``.

    Returns:
        f (ndarray[float]): footprint fraction.

    """
    theta = np.asarray(theta, dtype=float)
    if shape == "none":
        return np.ones_like(theta)
    if beam_width <= 0 or sample_length <= 0:
        raise ValueError(
            "The beam_width and sample_length of a footprint must be positive, but are "
            f"{beam_width} and {sample_length}!"
        )
    projected = sample_length * np.clip(np.sin(theta), 0, None)
    if shape == "box":
        return np.minimum(projected / beam_width, 1.0)
    if shape == "gauss":
        return erf(projected / (2 * np.sqrt(2) * beam_width * _SIGMA))
    raise ValueError(f"Unknown footprint {shape!r}!")
