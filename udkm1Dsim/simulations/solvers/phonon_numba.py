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

"""Numba-accelerated solver for the coherent phonon dynamics of a linear chain.

The solver solves the same equations of motion as
:mod:`~udkm1Dsim.simulations.solvers.phonon_scipy`, which documents the
physics. The right-hand side :func:`rhs` is a plain loop over the layers and
the time integration runs in ``numba``'s nopython mode as well: an adaptive
Dormand-Prince 5(4) Runge-Kutta scheme with the ``rtol`` and ``atol``
semantics of ``scipy.integrate.solve_ivp``, which steps exactly onto every
requested delay. As an explicit scheme its time step is bounded by the
highest eigenfrequency of the chain, see
:func:`~udkm1Dsim.simulations.solvers.phonon_scipy.stability_time_step`. All
compiled functions are cached on disk.

"""

__all__ = ["integrate_rk45", "rhs", "solve"]

__docformat__ = "restructuredtext"

import numpy as np
from tqdm.auto import tqdm

from . import HAS_NUMBA
from .phonon_scipy import stability_time_step

if HAS_NUMBA:
    from numba import njit
else:  # pragma: no cover - the kernel is unusable without numba anyway

    def njit(*args, **kwargs):
        def wrap(func):
            return func

        return wrap


# Dormand-Prince 5(4) coefficients
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1 / 5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3 / 40, 9 / 40, 0.0, 0.0, 0.0, 0.0],
        [44 / 45, -56 / 15, 32 / 9, 0.0, 0.0, 0.0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0.0, 0.0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0.0],
        [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
)
# error coefficients: difference of the 5th and the embedded 4th order weights
_E = np.array([71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])


def solve(
    x0, delays, force_from_heat, damping, spring_consts, masses, ode_options, progress_bar=False
):
    r"""solve

    Solves the equations of motion of the linear chain for all layers with a
    ``numba``-compiled adaptive Runge-Kutta integrator. Same interface as
    :func:`~udkm1Dsim.simulations.solvers.phonon_scipy.solve`.

    The time step is limited to the
    :func:`~udkm1Dsim.simulations.solvers.phonon_scipy.stability_time_step`
    of the chain, which the explicit scheme must not exceed - the error control alone does
    not catch the instability, since the shifts of the layers are of the
    order of picometers and therefore far below the absolute tolerance
    ``atol`` that is set in meters.

    Args:
        x0 (ndarray[float]): initial shifts and velocities of all :math:`L`
            layers of shape (2L) [m, m/s].
        delays (ndarray[float]): delays range of simulation [s].
        force_from_heat (ndarray[float]): force due to thermal expansion of
            shape (L, M) for M delays [N].
        damping (ndarray[float]): phonon damping of all layers.
        spring_consts (ndarray[float]): spring constants of all layers.
        masses (ndarray[float]): masses of all layers [kg].
        ode_options (dict): options of scipy's ``solve_ivp``, of which
            ``rtol``, ``atol``, ``max_step`` and ``first_step`` are used by the
            adaptive integrator.
        progress_bar (bool, optional): enable tqdm progress bar. Defaults
            to False.

    Returns:
        X (ndarray[float]): shifts and velocities of all layers of shape
        (M, 2L) [m, m/s].

    """
    if not HAS_NUMBA:
        raise RuntimeError("The numba backend requires the optional numba package!")

    delays = np.ascontiguousarray(np.asarray(delays, dtype=float))
    M = len(delays)
    L = len(masses)
    force_from_heat = np.ascontiguousarray(np.asarray(force_from_heat, dtype=float))
    damping = np.ascontiguousarray(np.asarray(damping, dtype=float))
    spring_consts = np.ascontiguousarray(
        np.reshape(np.asarray(spring_consts, dtype=float), (L, -1))
    )
    masses = np.ascontiguousarray(np.asarray(masses, dtype=float))
    y = np.ascontiguousarray(np.asarray(x0, dtype=float).copy())
    X = np.empty((M, y.size))
    X[0] = y

    rtol = float(ode_options.get("rtol", 1e-3))
    atol = float(ode_options.get("atol", 1e-6))
    max_step = ode_options.get("max_step", np.inf)
    max_step = np.inf if max_step is None else float(max_step)
    max_step = min(max_step, stability_time_step(spring_consts, masses))
    first_step = ode_options.get("first_step", None)
    h = 0.0 if first_step is None else float(first_step)

    pbar = tqdm(total=M - 1) if progress_bar else None

    for m in range(1, M):
        t0, t1 = delays[m - 1], delays[m]
        h = integrate_rk45(
            y,
            t0,
            t1,
            h,
            rtol,
            atol,
            max_step,
            delays,
            force_from_heat,
            damping,
            spring_consts,
            masses,
        )
        X[m] = y
        if pbar is not None:
            pbar.update(1)
            pbar.set_description(f"Delay = {t1 * 1e12:.3f} ps")

    if pbar is not None:
        pbar.close()

    return X


@njit(cache=True)
def _delay_index(t, delays):
    """index of the last delay that is not larger than t, as helpers.finderb"""
    M = len(delays)
    if t <= delays[0]:
        return 0
    if t >= delays[M - 1]:
        return M - 1
    lo = 0
    hi = M - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t < delays[mid]:
            hi = mid
        else:
            lo = mid
    return lo


@njit(cache=True)
def rhs(t, y, dydt, delays, force_from_heat, damping, spring_consts, masses):
    r"""rhs

    Right-hand side of the equations of motion of the linear chain, written
    into ``dydt``, see :func:`~udkm1Dsim.simulations.solvers.phonon_scipy.odefunc`:
    the acceleration of layer :math:`i` is the sum of the spring forces of
    its two neighbours with all orders of the inter-atomic potential, the
    damping force and the force of the thermal expansion of the current
    delay, divided by its mass.

    Args:
        t (float): time [s].
        y (ndarray[float]): shifts followed by velocities :math:`2L`.
        dydt (ndarray[float]): output, velocities followed by accelerations.
        delays (ndarray[float]): delays range of simulation [s].
        force_from_heat (ndarray[float]): forces of the thermal expansion
            :math:`L \times M` [N].
        damping (ndarray[float]): phonon damping of the layers.
        spring_consts (ndarray[float]): spring constants :math:`L \times J`.
        masses (ndarray[float]): masses of the layers [kg].

    """
    L = len(masses)
    order = spring_consts.shape[1]
    m_idx = _delay_index(t, delays)
    for i in range(L):
        force = masses[i] * damping[i] * y[L + i] + force_from_heat[i, m_idx]
        if i > 0:
            # spring k_{i-1} between the left neighbour and this layer
            d = y[i] - y[i - 1]
            power = d
            for j in range(order):
                force -= spring_consts[i - 1, j] * power
                power *= d
        if i < L - 1:
            # spring k_i between this layer and the right neighbour
            d = y[i + 1] - y[i]
            power = d
            for j in range(order):
                force += spring_consts[i, j] * power
                power *= d
        dydt[i] = y[L + i]
        dydt[L + i] = force / masses[i]


@njit(cache=True)
def integrate_rk45(
    y, t0, t1, h, rtol, atol, max_step, delays, force_from_heat, damping, spring_consts, masses
):
    """integrate_rk45

    Advances the state ``y`` in place from ``t0`` to ``t1`` with the adaptive
    Dormand-Prince 5(4) scheme. The error control follows
    ``scipy.integrate.RK45``: the RMS norm of the local error, scaled by
    ``atol + rtol*|y|``, must stay below one.

    Args:
        y (ndarray[float]): state, advanced in place.
        t0 (float): start time.
        t1 (float): end time.
        h (float): initial step size, zero to estimate it.
        rtol (float): relative tolerance.
        atol (float): absolute tolerance.
        max_step (float): maximum step size.
        delays (ndarray[float]): delays range of simulation [s].
        force_from_heat (ndarray[float]): forces of the thermal expansion.
        damping (ndarray[float]): phonon damping of the layers.
        spring_consts (ndarray[float]): spring constants.
        masses (ndarray[float]): masses of the layers.

    Returns:
        h (float): step size the error control asks for next, to be reused
        for the next interval.

    """
    n = y.size
    k = np.empty((7, n))
    y_new = np.empty(n)
    t = t0
    span = t1 - t0
    if span <= 0.0:
        return h

    rhs(t, y, k[0], delays, force_from_heat, damping, spring_consts, masses)

    if h <= 0.0:
        # initial step size estimate as in scipy's select_initial_step
        d0 = 0.0
        d1 = 0.0
        for j in range(n):
            sc = atol + rtol * abs(y[j])
            d0 += (y[j] / sc) ** 2
            d1 += (k[0, j] / sc) ** 2
        d0 = np.sqrt(d0 / n)
        d1 = np.sqrt(d1 / n)
        if d0 < 1e-5 or d1 < 1e-5:
            h = 1e-6 * span
        else:
            h = 0.01 * d0 / d1
        h = min(h, span)

    n_steps = 0
    while t < t1:
        # h is the step size the error control asks for, h_step the one that
        # is actually taken, clipped to max_step and to the end of the interval
        h_step = min(h, max_step)
        last = False
        if t + h_step >= t1:
            h_step = t1 - t
            last = True

        # stages 2..7
        for s in range(1, 7):
            for j in range(n):
                acc = 0.0
                for q in range(s):
                    acc += _A[s, q] * k[q, j]
                y_new[j] = y[j] + h_step * acc
            rhs(
                t + _C[s] * h_step,
                y_new,
                k[s],
                delays,
                force_from_heat,
                damping,
                spring_consts,
                masses,
            )
        # y_new now holds the 5th order solution (stage 7 was evaluated at it)

        # error estimate
        err = 0.0
        for j in range(n):
            e = 0.0
            for q in range(7):
                e += _E[q] * k[q, j]
            e *= h_step
            sc = atol + rtol * max(abs(y[j]), abs(y_new[j]))
            err += (e / sc) ** 2
        err = np.sqrt(err / n)

        if err <= 1.0:
            # accept the step
            t = t1 if last else t + h_step
            for j in range(n):
                y[j] = y_new[j]
            # first-same-as-last: stage 7 is the derivative at the new point
            for j in range(n):
                k[0, j] = k[6, j]
            if err == 0.0:
                factor = 10.0
            else:
                factor = min(10.0, max(0.2, 0.9 * err ** (-0.2)))
            if h_step < h:
                # the step was clipped, keep the larger requested step size
                h = max(h, h_step * factor)
            else:
                h = h_step * factor
        else:
            factor = max(0.2, 0.9 * err ** (-0.2))
            h = h_step * factor

        n_steps += 1
        if n_steps > 10000000:
            raise RuntimeError("The adaptive integrator exceeded 1e7 steps in one interval!")
        if h < abs(span) * 2.220446049250313e-16:
            raise RuntimeError("The adaptive integrator step size underflowed!")

    return h
