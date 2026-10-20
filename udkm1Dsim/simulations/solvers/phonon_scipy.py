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

"""Scipy-based solver for the coherent phonon dynamics of a linear chain.

This is the default backend of the :class:`PhononNum` simulation. The
equations of motion of the linear chain of masses and springs are solved as a
first-order system for the shifts and velocities of all layers with
``scipy.integrate.solve_ivp`` and :meth:`odefunc` as right-hand side.

"""

__all__ = [
    "force_from_damping",
    "force_from_heat",
    "force_from_spring",
    "odefunc",
    "solve",
    "stability_time_step",
]

__docformat__ = "restructuredtext"

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from ...helpers import finderb

# explicit solvers of solve_ivp, whose time step is bounded by the stability
_EXPLICIT_METHODS = ("RK23", "RK45", "DOP853")


def solve(
    x0, delays, force_from_heat, damping, spring_consts, masses, ode_options, progress_bar=False
):
    r"""solve

    Solves the equations of motion of the linear chain for all layers using
    scipy's ``solve_ivp`` with :meth:`odefunc` as right-hand side.

    For the explicit solvers ``RK23``, ``RK45`` and ``DOP853`` the
    ``max_step`` is limited to the :meth:`stability_time_step` of the chain,
    which they must not exceed - the error control alone does not catch the
    instability, since the shifts of the layers are of the order of
    picometers and therefore far below the absolute tolerance ``atol`` that
    is set in meters.

    Args:
        x0 (ndarray[float]): initial shifts and velocities of all :math:`L`
            layers of shape (2L) [m, m/s].
        delays (ndarray[float]): delays range of simulation [s].
        force_from_heat (ndarray[float]): force due to thermal expansion of
            shape (L, M) for M delays [N].
        damping (ndarray[float]): phonon damping of all layers.
        spring_consts (ndarray[float]): spring constants of all layers.
        masses (ndarray[float]): masses of all layers [kg].
        ode_options (dict): options passed to scipy's ``solve_ivp``. A
            given ``max_step`` is only decreased by the stability time step.
        progress_bar (bool, optional): enable tqdm progress bar. Defaults
            to False.

    Returns:
        X (ndarray[float]): shifts and velocities of all layers of shape
        (M, 2L) [m, m/s].

    """
    L = len(masses)

    if progress_bar:  # with tqdm progressbar
        pbar = tqdm()
        pbar.set_description(f"Delay = {delays[0] * 1e12:.3f} ps")
        state = [delays[0], abs(delays[-1] - delays[0]) / 100]
    else:  # without progressbar
        pbar = None
        state = None

    options = dict(ode_options)
    if options.get("method", "RK45") in _EXPLICIT_METHODS:
        max_step = options.get("max_step")
        max_step = np.inf if max_step is None else max_step
        options["max_step"] = min(max_step, stability_time_step(spring_consts, masses))

    sol = solve_ivp(
        odefunc,
        [delays[0], delays[-1]],
        x0,
        args=(delays, force_from_heat, damping, spring_consts, masses, L, pbar, state),
        t_eval=delays,
        **options,
    )

    if pbar is not None:  # close tqdm progressbar if used
        pbar.close()

    return sol.y.T


def odefunc(
    t, X, delays, force_from_heat, damping, spring_consts, masses, L, pbar=None, state=None
):
    r"""odefunc

    Provides the according ode function for the ode solver which has to be
    solved. The ode function has the input :math:`t` and :math:`X(t)` and
    calculates the temporal derivative :math:`\dot{X}(t)` where the vector

    .. math::

        X(t) = [x(t) \; \dot{x}(t)] \quad \mbox{and } \quad
        \dot{X}(t) = [\dot{x}(t) \; \ddot{x}(t)] .

    :math:`x(t)` is the actual shift of each layer.

    Args:
        t (float): internal time step of the ode solver [s].
        X (ndarray[float]): internal variable of the ode solver.
        delays (ndarray[float]): delays range of simulation [s].
        force_from_heat (ndarray[float]): force due to thermal expansion.
        damping (ndarray[float]): phonon damping.
        spring_consts (ndarray[float]): spring constants of masses.
        masses (ndarray[float]): masses of layers.
        L (int): number of layers.
        pbar (tqdm, None): tqdm progressbar or None if disabled.
        state (list[float], None): last updated delay and update interval
            of the progress bar or None if disabled.

    Returns:
        X_prime (ndarray[float]): velocities and accelerations of masses.

    """
    if pbar is not None:
        # set everything for the tqdm progressbar
        last_t, dt = state
        n = (t - last_t) / dt
        if n >= 1:
            pbar.update(1)
            pbar.set_description(f"Delay = {t * 1e12:.3f} ps")
            state[0] = t
        elif n < 0:
            state[0] = t

    # start with the actual ode function
    x = X[0:L]
    v = X[L:]

    # the output must be a column vector
    X_prime = np.zeros([2 * L])

    # accelerations = derivative of velocities
    X_prime[L:] = (
        force_from_damping(v, damping, masses)
        + force_from_spring(np.r_[np.diff(x), 0], np.r_[0, np.diff(x)], spring_consts)
        + force_from_heat[:, finderb(t, delays)].squeeze()
    ) / masses

    # velocities = derivative of positions
    X_prime[0:L] = v

    return X_prime


def force_from_spring(d_X1, d_X2, spring_consts):
    r"""force_from_spring

    Calculates the force :math:`F_i^{spring}` acting on each mass due to
    the displacement between the left and right site of that mass.

    .. math::

        F_i^{spring} = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1})

    We introduce-higher order inter-atomic potentials by

    .. math::

        k_i(x_i-x_{i-1}) = \sum_{j=1}^{M} k_i^j (x_i-x_{i-1})^j

    where :math:`M` is the order of the spring constants.

    Args:
        d_X1 (ndarray[float]): left displacements.
        d_X2 (ndarray[float]): right displacements.
        spring_consts (ndarray[float]): spring constants of masses.

    Returns:
        F (ndarray[float]): force from springs.

    """
    try:
        spring_order = np.size(spring_consts, 1)
    except IndexError:
        spring_order = 1

    spring_consts = np.reshape(spring_consts, [np.size(spring_consts, 0), spring_order])

    coeff1 = np.vstack((-spring_consts[0:-1, :], np.zeros([1, spring_order])))
    coeff2 = np.vstack((np.zeros([1, spring_order]), -spring_consts[0:-1, :]))

    temp1 = np.zeros([len(d_X1), spring_order])
    temp2 = np.zeros([len(d_X1), spring_order])

    for i in range(spring_order):
        temp1[:, i] = d_X1 ** (i + 1)
        temp2[:, i] = d_X2 ** (i + 1)

    F = np.sum(coeff2 * temp2, 1) - np.sum(coeff1 * temp1, 1)

    return F


def force_from_heat(sticks, spring_consts):
    """force_from_heat

    Calculates the force acting on each mass due to the heat expansion,
    which is modeled by spacer sticks.

    Args:
        sticks (ndarray[float]): spacer sticks.
        spring_consts (ndarray[float]): spring constants of masses.

    Returns:
        F (ndarray[float]): force from thermal expansion.

    """
    M, L = np.shape(sticks)
    F = np.zeros([L, M])
    # traverse time
    for i in range(M):
        F[:, i] = -force_from_spring(
            np.hstack((sticks[i, 0 : L - 1], 0)),
            np.hstack((0, sticks[i, 0 : L - 1])),
            spring_consts,
        )

    return F


def force_from_damping(v, damping, masses):
    r"""force_from_damping

    Calculates the force acting on each mass in a linear spring due to
    damping (:math:`\gamma_i`) according to the shift velocity difference
    :math:`v_{i}-v_{i-1}` with :math:`v_i(t) = \dot{x}_i(t)`:

    .. math::

        F_i^{damp} = \gamma_i(\dot{x}_i-\dot{x}_{i-1})

    Args:
        v (ndarray[float]): velocity of masses.
        damping (ndarray[float]): phonon damping.
        masses (ndarray[float]): masses.

    Returns:
        F (ndarray[float]): force from damping.

    """
    F = masses * damping * np.diff(v, 0)

    return F


def stability_time_step(spring_consts, masses, fraction=1.0):
    r"""stability_time_step

    Largest time step the explicit Runge-Kutta schemes may take on the
    linear chain. Its highest eigenfrequency is bounded by Gershgorin's
    theorem applied to the dynamical matrix of the harmonic springs

    .. math::

        \omega_\mathrm{max}^2 \leq \max_i \frac{2\,(k_{i-1} + k_i)}{m_i}

    and the stability regions of the explicit Runge-Kutta schemes on the
    imaginary axis end at :math:`h\,\omega \approx 1.7` for ``RK23`` and
    :math:`h\,\omega \approx 3.3` for ``RK45``, so the step
    :math:`h = f/\omega_\mathrm{max}` with :math:`f = 1` keeps every mode of
    the chain stable. The step is a bound on the stability, the accuracy is
    left to the error control.

    Args:
        spring_consts (ndarray[float]): spring constants :math:`L \times J`,
            only the harmonic order is used.
        masses (ndarray[float]): masses of the layers [kg].
        fraction (float, optional): fraction :math:`f` of the stability
            limit. Defaults to 1.0.

    Returns:
        max_step (float): largest stable time step [s].

    """
    k = np.reshape(np.asarray(spring_consts, dtype=float), (len(masses), -1))[:, 0]
    # k_i connects the layers i and i+1, the last spring is unused
    k_left = np.r_[0.0, k[:-1]]
    k_right = np.r_[k[:-1], 0.0]
    omega_max = np.sqrt(np.max(2 * (k_left + k_right) / np.asarray(masses, dtype=float)))
    if not np.isfinite(omega_max) or omega_max <= 0:
        return np.inf
    return fraction / omega_max
