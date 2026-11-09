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

r"""Mean-field equilibrium magnetization.

The quantum Landau-Lifshitz-Bloch equation and the temperature-driven variant
of the Landau-Lifshitz-Gilbert equation need the equilibrium magnetization
:math:`m_\mathrm{eq}(T)` of every layer, which is the non-trivial solution of
the self-consistent mean-field equation

.. math::

    m_\mathrm{eq} = B_S\left(\frac{J\, m_\mathrm{eq}}{k_\mathrm{B} T}\right)
    \quad\mathrm{with}\quad J = \frac{3 S}{S+1}\, k_\mathrm{B} T_\mathrm{C},

where :math:`B_S` is the Brillouin function of the effective spin :math:`S`,
which is the Langevin function for :math:`S \rightarrow \infty`. Layers with
several sublattices solve the coupled equations with the mean-field exchange
matrix :math:`J_{ab}`. On top of the equilibrium magnetization, the LLB needs
the longitudinal susceptibilities and the damping parameters of the
sublattices, which depend on the temperature only as well. All of them are
tabulated per layer by :func:`equilibrium_tables` and interpolated by the
solvers. All functions here are vectorized ``numpy`` functions and are shared
by all magnetization backends.

"""

__all__ = [
    "brillouin",
    "dbrillouin_dx",
    "langevin",
    "dlangevin_dx",
    "mean_field_magnetization",
    "ordering_temperature",
    "sublattice_signs",
    "mean_field_magnetization_coupled",
    "TABLE",
    "CHI_MAX",
    "CHI_MIN",
    "temperature_grid",
    "equilibrium_tables",
    "longitudinal_susceptibility",
    "damping_parameters",
    "interpolate_tables",
    "interpolate_tables_map",
]

__docformat__ = "restructuredtext"

import numpy as np
import scipy.constants as constants


def brillouin(x, eff_spin):
    r"""brillouin

    Brillouin function

    .. math::

        B_S(x) = \frac{2 S+1}{2S} \coth{\left(\frac{2S+1}{2S} x\right)}
        - \frac{1}{2S}\coth{\left(\frac{x}{2S}\right)}

    with the small-argument limit :math:`B_S(x) \approx \frac{S+1}{3S} x`.

    Args:
        x (ndarray[float]): argument of the Brillouin function.
        eff_spin (float): effective spin :math:`S`.

    Returns:
        brillouin (ndarray[float]): Brillouin function.

    """
    x = np.asarray(x, dtype=float)
    eff_spin = np.asarray(eff_spin, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c1 = (2 * eff_spin + 1) / (2 * eff_spin)
        c2 = 1 / (2 * eff_spin)
        result = c1 / np.tanh(c1 * x) - c2 / np.tanh(c2 * x)
        small = np.abs(x) < 1e-6
        result = np.where(small, (eff_spin + 1) / (3 * eff_spin) * x, result)
    # an infinite spin is the classical limit, the Langevin function
    return np.where(np.isinf(eff_spin), langevin(x), result)


def langevin(x):
    r"""langevin

    Langevin function :math:`L(x) = \coth(x) - 1/x`, the classical limit
    :math:`S \rightarrow \infty` of the Brillouin function, with the
    small-argument expansion :math:`x/3 - x^3/45`.

    Args:
        x (ndarray[float]): argument.

    Returns:
        langevin (ndarray[float]): Langevin function.

    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = 1 / np.tanh(x) - 1 / x
    small = np.abs(x) < 1e-3
    return np.where(small, x / 3 - x**3 / 45, result)


def dlangevin_dx(x):
    r"""dlangevin_dx

    Derivative of the Langevin function
    :math:`L'(x) = 1/x^2 - 1/\sinh^2(x)` with the small-argument expansion
    :math:`1/3 - x^2/15`.

    Args:
        x (ndarray[float]): argument.

    Returns:
        dLdx (ndarray[float]): derivative of the Langevin function.

    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = 1 / x**2 - 1 / np.sinh(x) ** 2
    small = np.abs(x) < 1e-3
    return np.where(small, 1 / 3 - x**2 / 15, result)


def dbrillouin_dx(x, eff_spin):
    r"""dbrillouin_dx

    Derivative of the Brillouin function

    .. math::

        \frac{dB_S}{dx} = \frac{1}{4S^2\sinh^2(x/2S)}
            -\frac{(2S+1)^2}{4S^2\sinh^2\left(\frac{(2S+1)x}{2S}\right)}

    with the small-argument limit :math:`\frac{S+1}{3S}`.

    Args:
        x (ndarray[float]): argument of the Brillouin function.
        eff_spin (float): effective spin :math:`S`.

    Returns:
        dBdx (ndarray[float]): derivative of Brillouin function.

    """
    x = np.asarray(x, dtype=float)
    eff_spin = np.asarray(eff_spin, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        c1 = (2 * eff_spin + 1) / (2 * eff_spin)
        c2 = 1 / (2 * eff_spin)
        result = c2**2 / np.sinh(c2 * x) ** 2 - c1**2 / np.sinh(c1 * x) ** 2
        small = np.abs(x) < 1e-4
        result = np.where(small, (eff_spin + 1) / (3 * eff_spin), result)
    return np.where(np.isinf(eff_spin), dlangevin_dx(x), result)


def mean_field_magnetization(reduced_temps, eff_spin, iterations=64):
    r"""mean_field_magnetization

    Solves the self-consistent mean-field equation

    .. math::

        m = B_S\left(\frac{3S}{S+1}\frac{m}{T/T_\mathrm{C}}\right)

    for the non-trivial root :math:`m_\mathrm{eq} \in (0, 1]` by a vectorized
    bisection. The root is bracketed exactly, since the right-hand side lies
    below the left-hand side for :math:`m \rightarrow 0^+` and above it for
    :math:`m = 1` whenever :math:`T < T_\mathrm{C}`. Above and at the Curie
    temperature the only root is :math:`m_\mathrm{eq} = 0`.

    Args:
        reduced_temps (ndarray[float]): reduced temperatures
            :math:`T/T_\mathrm{C}`.
        eff_spin (float): effective spin :math:`S`.
        iterations (int, optional): number of bisection steps. Defaults to 64,
            which reaches double precision.

    Returns:
        mf_mags (ndarray[float]): mean-field magnetization.

    """
    t = np.atleast_1d(np.asarray(reduced_temps, dtype=float))
    # the classical limit S -> inf of 3S/(S+1) is 3
    a = 3.0 if np.isinf(eff_spin) else 3 * eff_spin / (eff_spin + 1)

    mf_mags = np.zeros_like(t)
    below = (t > 0) & (t < 1)
    mf_mags[t <= 0] = 1.0

    if np.any(below):
        tb = t[below]
        lo = np.full_like(tb, 1e-12)
        hi = np.ones_like(tb)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            f_mid = mid - brillouin(a * mid / tb, eff_spin)
            # f < 0 left of the non-trivial root, f > 0 right of it
            negative = f_mid < 0
            lo = np.where(negative, mid, lo)
            hi = np.where(negative, hi, mid)
        mf_mags[below] = 0.5 * (lo + hi)

    if np.ndim(reduced_temps) == 0:
        return mf_mags[0]
    return mf_mags


def _linearized_coupling(J, eff_spins):
    r"""_linearized_coupling

    Matrix of the self-consistent equations linearized at zero magnetization,
    :math:`M_{ab} = \frac{S_a+1}{3 S_a} J_{ab}`, whose largest eigenvalue
    gives the ordering temperature and whose eigenvector the relative
    orientation of the sublattices.

    Args:
        J (ndarray[float]): mean-field exchange matrix :math:`P \times P` (J).
        eff_spins (ndarray[float]): effective spins of the sublattices.

    Returns:
        matrix (ndarray[float]): linearized coupling matrix.

    """
    J = np.atleast_2d(np.asarray(J, dtype=float))
    eff_spins = np.broadcast_to(np.asarray(eff_spins, dtype=float), (J.shape[0],))
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = (eff_spins + 1) / (3 * eff_spins)
    # the classical limit of an infinite spin has the slope 1/3
    factors = np.where(np.isinf(eff_spins), 1 / 3, factors)
    factors = np.where(np.isfinite(factors), factors, 0.0)
    return factors[:, np.newaxis] * J


def ordering_temperature(J, eff_spins):
    r"""ordering_temperature

    Mean-field ordering (Curie) temperature of coupled sublattices

    .. math:: k_B T_C = \max \mathrm{eig}\left(\frac{S_a+1}{3 S_a} J_{ab}\right),

    which reduces to :math:`T_C = \frac{S+1}{3 S k_B} J` for one sublattice.

    Args:
        J (ndarray[float]): mean-field exchange matrix :math:`P \times P` (J).
        eff_spins (ndarray[float]): effective spins of the sublattices.

    Returns:
        curie_temp (float): ordering temperature (K).

    """
    eigenvalues = np.linalg.eigvals(_linearized_coupling(J, eff_spins))
    return float(max(np.max(np.real(eigenvalues)) / constants.k, 0.0))


def sublattice_signs(J, eff_spins):
    r"""sublattice_signs

    Relative orientation of the sublattices in the mean-field ground state,
    given by the signs of the eigenvector of the largest eigenvalue of the
    linearized self-consistent equations. Antiferromagnetically coupled
    sublattices (negative off-diagonal exchange) are antiparallel. The first
    sublattice defines the positive direction.

    Args:
        J (ndarray[float]): mean-field exchange matrix :math:`P \times P` (J).
        eff_spins (ndarray[float]): effective spins of the sublattices.

    Returns:
        signs (ndarray[float]): +1 or -1 per sublattice.

    """
    matrix = _linearized_coupling(J, eff_spins)
    if matrix.shape[0] == 1:
        return np.ones(1)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    vector = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    signs = np.sign(vector)
    signs[signs == 0] = 1.0
    return signs * signs[0]


def mean_field_magnetization_coupled(temps, J, eff_spins, iterations=300):
    r"""mean_field_magnetization_coupled

    Solves the coupled self-consistent mean-field equations of :math:`P`
    sublattices

    .. math::

        m_a = B_{S_a}\left(\frac{\sum_b J_{ab} m_b}{k_B T}\right)

    for all temperatures at once. The magnetizations are signed reduced
    magnetizations: antiferromagnetically coupled sublattices come out
    antiparallel, i.e. with opposite signs, and :math:`|m_a| \rightarrow 1` for
    :math:`T \rightarrow 0`. Above the :func:`ordering_temperature` all
    magnetizations vanish. A damped fixed-point iteration from saturation is
    polished by Newton steps on the coupled system.

    Args:
        temps (ndarray[float]): temperatures (K).
        J (ndarray[float]): mean-field exchange matrix :math:`P \times P` (J),
            row :math:`a` holding the couplings of the field on sublattice
            :math:`a`.
        eff_spins (ndarray[float]): effective spins of the sublattices.
        iterations (int, optional): number of fixed-point iterations.
            Defaults to 300.

    Returns:
        mf_mags (ndarray[float]): signed mean-field magnetizations
        :math:`(\ldots, P)`.

    """
    temps = np.atleast_1d(np.asarray(temps, dtype=float))
    J = np.atleast_2d(np.asarray(J, dtype=float))
    P = J.shape[0]
    eff_spins = np.broadcast_to(np.asarray(eff_spins, dtype=float), (P,))
    signs = sublattice_signs(J, eff_spins)
    curie_temp = ordering_temperature(J, eff_spins)

    shape = temps.shape
    temps = temps.reshape(-1)
    mf_mags = np.zeros((temps.size, P))
    mf_mags[temps <= 0, :] = signs
    below = (temps > 0) & (temps < curie_temp)

    if np.any(below):
        beta = 1 / (constants.k * temps[below])[:, np.newaxis]
        x = np.tile(signs, (np.count_nonzero(below), 1))
        for _ in range(iterations):
            x = 0.5 * x + 0.5 * brillouin(beta * (x @ J.T), eff_spins)
        x_fixed = x.copy()
        # Newton polish of the coupled system
        identity = np.eye(P)
        with np.errstate(all="ignore"):
            for _ in range(30):
                xi = beta * (x @ J.T)
                residual = x - brillouin(xi, eff_spins)
                jacobian = (
                    identity[np.newaxis, :, :]
                    - (beta * dbrillouin_dx(xi, eff_spins))[:, :, np.newaxis] * J[np.newaxis, :, :]
                )
                x = x - np.linalg.solve(jacobian, residual[:, :, np.newaxis])[:, :, 0]
        # points where Newton left the non-trivial branch keep the fixed point
        bad = (
            ~np.all(np.isfinite(x), axis=1)
            | (np.max(np.abs(x), axis=1) < 1e-10)
            | np.any(np.sign(x) * signs < 0, axis=1)
        )
        x[bad] = x_fixed[bad]
        mf_mags[below] = x

    return mf_mags.reshape(shape + (P,))


# rows of the equilibrium tables of a layer along their second axis
TABLE = {"mag": 0, "alpha_par": 1, "alpha_perp": 2, "chi": 3}

# cap of the longitudinal susceptibility (1/T), which diverges at the ordering
# temperature - its inverse, the longitudinal field per unit deviation, is
# then negligible - and its floor, where the quantum susceptibility
# underflows at very low temperatures
CHI_MAX = 1e12
CHI_MIN = 1e-300


def temperature_grid(temp_max, curie_temps, n_grid=4000, n_dense=200):
    r"""temperature_grid

    Temperature grid of the :func:`equilibrium_tables`: uniform from 1 K to
    ``temp_max``, refined around every ordering temperature with points
    that are uniform in :math:`s = \sqrt{|1 - T/T_C|}`, since the
    equilibrium magnetization vanishes as :math:`\sqrt{T_C - T}` and the
    susceptibility diverges at :math:`T_C`, and containing :math:`T_C` itself.

    Args:
        temp_max (float): largest temperature of the grid (K).
        curie_temps (ndarray[float]): ordering temperatures of the layers
            (K), zeros are ignored.
        n_grid (int, optional): number of uniform points. Defaults to 4000.
        n_dense (int, optional): number of refinement points on either side
            of every ordering temperature. Defaults to 200.

    Returns:
        temp_grid (ndarray[float]): ascending temperature grid (K).

    """
    temp_max = max(float(temp_max), 2.0)
    grid = [np.linspace(1.0, temp_max, int(n_grid))]
    s = np.linspace(0.0, 0.3, int(n_dense)) ** 2
    for curie_temp in np.unique(np.asarray(curie_temps, dtype=float)):
        if curie_temp > 0:
            grid.append(curie_temp * (1 - s))
            grid.append(curie_temp * (1 + s))
    grid = np.unique(np.concatenate(grid))
    return grid[(grid >= 1.0) & (grid <= temp_max)]


def equilibrium_tables(temp_grid, mf_exch_coupling, eff_spins, mag_moments, lambdas, active):
    r"""equilibrium_tables

    Temperature tables of all equilibrium quantities of the LLB of one
    layer with :math:`P` coupled sublattices: the mean-field magnetizations
    :math:`m_{e,a}(T)` from :func:`mean_field_magnetization_coupled`, the
    :func:`damping_parameters` :math:`\alpha_a^\parallel(T)`,
    :math:`\alpha_a^\perp(T)` and the :func:`longitudinal_susceptibility`
    :math:`\chi_a(T)`. They only depend on the temperature and the material,
    so they are tabulated once per layer and interpolated by the solvers, see
    :func:`interpolate_tables`. The rows along the second axis are given by
    :data:`TABLE`. Finite effective spins give the quantum LLB with Brillouin
    functions, infinite spins the classical one with Langevin functions.
    Padded (inactive) sublattices carry zeros and a susceptibility of one.

    Args:
        temp_grid (ndarray[float]): temperature grid :math:`N_T` (K).
        mf_exch_coupling (ndarray[float]): mean-field exchange matrix
            :math:`P \times P` (J).
        eff_spins (ndarray[float]): effective spins :math:`P`.
        mag_moments (ndarray[float]): atomic magnetic moments :math:`P` (A m²).
        lambdas (ndarray[float]): coupling-to-bath parameters :math:`P`.
        active (ndarray[float]): 1 for existing sublattices, 0 for padding.

    Returns:
        tables (ndarray[float]): tables :math:`P \times 4 \times N_T`.

    """
    temp_grid = np.asarray(temp_grid, dtype=float)
    active = np.asarray(active)
    P = len(active)
    p = int(np.count_nonzero(active))
    tables = np.zeros((P, 4, len(temp_grid)))
    tables[:, TABLE["chi"], :] = 1.0
    if p == 0:
        return tables
    J = np.asarray(mf_exch_coupling, dtype=float)[:p, :p]
    spins = np.asarray(eff_spins, dtype=float)[:p]
    mu = np.asarray(mag_moments, dtype=float)[:p]
    lam = np.asarray(lambdas, dtype=float)[:p]
    curie_temp = ordering_temperature(J, spins)
    if curie_temp <= 0:
        return tables

    me = np.abs(mean_field_magnetization_coupled(temp_grid, J, spins))
    J_abs = np.abs(J)
    zeta = (me @ J_abs.T) / (constants.k * temp_grid[:, np.newaxis])
    below = (temp_grid[:, np.newaxis] < curie_temp) & (me > 1e-8)
    alpha_par, alpha_perp = damping_parameters(temp_grid, me, zeta, spins, lam, curie_temp, below)
    chi = longitudinal_susceptibility(temp_grid, zeta, spins, J_abs, mu)
    tables[:p, TABLE["mag"]] = me.T
    tables[:p, TABLE["alpha_par"]] = alpha_par.T
    tables[:p, TABLE["alpha_perp"]] = alpha_perp.T
    tables[:p, TABLE["chi"]] = chi.T
    return tables


def longitudinal_susceptibility(temps, zeta, eff_spins, mf_exch_coupling_abs, mag_moments):
    r"""longitudinal_susceptibility

    Longitudinal susceptibilities :math:`\chi_a = \partial m_a/\partial B`
    (1/T) of :math:`P` coupled sublattices in the mean-field approximation,
    the solution of the linear system

    .. math::

        \tilde\chi_{a} - \beta B'_a \sum_{b} |J_{ab}|\, \tilde\chi_{b}
        = \beta B'_a \mu_a

    with the derivatives :math:`B'_a` of the Brillouin functions at the
    equilibrium arguments :math:`\zeta_{e,a} = \beta\sum_b |J_{ab}|
    m_{e,b}`, which are :math:`(S_a+1)/(3S_a)` above the ordering
    temperature. For one sublattice this is
    :math:`\tilde\chi = \mu B'/(k_\mathrm{B} T - J B')`
    (:cite:t:`nieves2014`), for two classical sublattices Eqs. (eS16) and
    (eS17) of :cite:t:`raposo2022`. The susceptibilities diverge at the
    ordering temperature.

    Args:
        temps (ndarray[float]): temperatures :math:`N_T` (K).
        zeta (ndarray[float]): equilibrium mean-field arguments
            :math:`N_T \times P`.
        eff_spins (ndarray[float]): effective spins :math:`P`.
        mf_exch_coupling_abs (ndarray[float]): moduli of the mean-field
            exchange matrix :math:`P \times P` (J).
        mag_moments (ndarray[float]): atomic magnetic moments :math:`P`
            (A m²).

    Returns:
        chi (ndarray[float]): longitudinal susceptibilities
        :math:`N_T \times P` (1/T).

    """
    P = np.shape(mf_exch_coupling_abs)[0]
    beta = 1 / (constants.k * np.asarray(temps, dtype=float))[:, np.newaxis]
    dB = dbrillouin_dx(zeta, eff_spins)
    matrix = np.eye(P)[np.newaxis, :, :] - (beta * dB)[:, :, np.newaxis] * mf_exch_coupling_abs
    rhs = beta * dB * mag_moments
    with np.errstate(all="ignore"):
        # the system is singular exactly at the ordering temperature, where
        # the susceptibility diverges - the affected temperatures get the cap
        determinant = np.linalg.det(matrix)
        singular = ~(np.abs(determinant) > 1e-12)
        matrix[singular] = np.eye(P)
        chi = np.linalg.solve(matrix, rhs[:, :, np.newaxis])[:, :, 0]
        chi[singular] = CHI_MAX
    # a finite cap keeps ratios of susceptibilities defined in the solvers;
    # at very low temperatures the quantum susceptibility underflows to zero
    # and gets a tiny positive floor instead
    chi = np.where(np.isfinite(chi), chi, CHI_MAX)
    return np.clip(chi, CHI_MIN, CHI_MAX)


def damping_parameters(temps, me, zeta, eff_spins, lambdas, curie_temp, below):
    r"""damping_parameters

    Longitudinal and transverse damping parameters of the quantum LLB,
    :cite:t:`nieves2014` Eqs. (20) and (21) for isotropic scattering,
    generalized to coupled sublattices:

    .. math::

        \alpha_a^\parallel = 2\lambda_a \frac{S_a}{S_a+1}
            \frac{m_{e,a}}{\zeta_{e,a}} \frac{2q_a}{\sinh(2q_a)}, \quad
        \alpha_a^\perp = \lambda_a\left(\frac{\tanh q_a}{q_a}
            - \frac{S_a}{S_a+1}\frac{m_{e,a}}{\zeta_{e,a}}\right), \quad
        q_a = \frac{\zeta_{e,a}}{2S_a}

    below the ordering temperature, where for a single sublattice
    :math:`\frac{S}{S+1}\frac{m_e}{\zeta_e} = \frac{T}{3T_C}` and
    :math:`q = \frac{3T_C m_e}{2(S+1)T}`, and
    :math:`\alpha^\parallel = \alpha^\perp = 2\lambda T/(3T_C)` at and above
    it. For :math:`S \rightarrow \infty` (:math:`q = 0`) these are the
    classical parameters :math:`2\lambda m_e/\zeta_e` and
    :math:`\lambda(1 - m_e/\zeta_e)` of :cite:t:`raposo2022` Eqs. (eS8) and
    (eS9), i.e. Garanin's :math:`2\lambda T/(3T_C)` and
    :math:`\lambda(1 - T/(3T_C))` for one sublattice. Both branches meet at
    :math:`T_C`.

    Args:
        temps (ndarray[float]): temperatures :math:`N_T` (K).
        me (ndarray[float]): equilibrium magnetizations :math:`N_T \times P`.
        zeta (ndarray[float]): equilibrium mean-field arguments
            :math:`N_T \times P`.
        eff_spins (ndarray[float]): effective spins :math:`P`.
        lambdas (ndarray[float]): coupling-to-bath parameters :math:`P`.
        curie_temp (float): ordering temperature (K).
        below (ndarray[boolean]): mask of the ordered phase
            :math:`N_T \times P`.

    Returns:
        (tuple):
        - *alpha_par (ndarray[float])* - longitudinal damping
          :math:`N_T \times P`.
        - *alpha_perp (ndarray[float])* - transverse damping
          :math:`N_T \times P`.

    """
    spins = np.asarray(eff_spins, dtype=float)
    temps = np.asarray(temps, dtype=float)
    with np.errstate(all="ignore"):
        factor = np.where(np.isinf(spins), 1.0, spins / (spins + 1))
        reduced = factor * me / np.maximum(zeta, 1e-300)
        q = np.where(np.isinf(spins), 0.0, zeta / (2 * spins))
        two_q_over_sinh = np.where(q > 1e-6, 2 * q / np.sinh(np.minimum(2 * q, 700.0)), 1.0)
        tanh_over_q = np.where(q > 1e-6, np.tanh(q) / np.maximum(q, 1e-300), 1.0)
    para = 2 * lambdas * temps[:, np.newaxis] / (3 * curie_temp)
    alpha_par = np.where(below, 2 * lambdas * reduced * two_q_over_sinh, para)
    alpha_perp = np.where(below, lambdas * (tanh_over_q - reduced), para)
    return alpha_par, alpha_perp


def interpolate_tables(temp_grid, tables, temps):
    r"""interpolate_tables

    Linear interpolation of the :func:`equilibrium_tables` of all layers at
    the current temperatures of the layers, holding the boundary values
    outside of the grid.

    Args:
        temp_grid (ndarray[float]): temperature grid :math:`N_T` (K).
        tables (ndarray[float]): tables :math:`N \times P \times 4 \times N_T`.
        temps (ndarray[float]): temperatures of the layers :math:`N` (K).

    Returns:
        values (ndarray[float]): interpolated tables :math:`N \times P \times 4`.

    """
    Nt = len(temp_grid)
    if Nt < 2:
        return tables[..., 0]
    idx = np.clip(np.searchsorted(temp_grid, temps, side="right") - 1, 0, Nt - 2)
    w = np.clip((temps - temp_grid[idx]) / (temp_grid[idx + 1] - temp_grid[idx]), 0.0, 1.0)
    lo = np.take_along_axis(tables, idx[:, np.newaxis, np.newaxis, np.newaxis], axis=3)[..., 0]
    hi = np.take_along_axis(tables, (idx + 1)[:, np.newaxis, np.newaxis, np.newaxis], axis=3)[
        ..., 0
    ]
    return (1 - w)[:, np.newaxis, np.newaxis] * lo + w[:, np.newaxis, np.newaxis] * hi


def interpolate_tables_map(temp_grid, tables, temp_map):
    r"""interpolate_tables_map

    Interpolates the :func:`equilibrium_tables` of all layers for a whole
    temperature map.

    Args:
        temp_grid (ndarray[float]): temperature grid :math:`N_T` (K).
        tables (ndarray[float]): tables :math:`N \times P \times 4 \times N_T`.
        temp_map (ndarray[float]): temperature map :math:`M \times N` (K).

    Returns:
        values (ndarray[float]): interpolated tables
        :math:`M \times N \times P \times 4`.

    """
    M, N = temp_map.shape
    P, Q = tables.shape[1:3]
    values = np.empty((M, N, P, Q))
    for n in range(N):
        for a in range(P):
            for q in range(Q):
                values[:, n, a, q] = np.interp(temp_map[:, n], temp_grid, tables[n, a, q])
    return values
