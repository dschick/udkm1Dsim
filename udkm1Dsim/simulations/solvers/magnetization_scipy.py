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

r"""Scipy-based solver for the magnetization dynamics.

This is the default backend of the magnetization simulations. The right-hand
sides of the Landau-Lifshitz-Gilbert (LLG) equation :func:`llg_rhs`, its
inertial extension (iLLG) :func:`illg_rhs` and the Landau-Lifshitz-Bloch
(LLB) equation :func:`llb_rhs` are assembled from small functions, one per
physical term, which are vectorized with ``numpy`` over all magnetic layers
and their sublattices. The time integration is done by
``scipy.integrate.solve_ivp``, see :func:`solve`.

Every magnetic layer carries :math:`P \geq 1` magnetic sublattices, which
are coupled by the mean-field exchange matrix :math:`J_{ab}` of the layer.
:math:`P = 1` is a ferromagnet. All arrays carry the sublattice axis, layers
with fewer sublattices are padded with inactive ones.

The static effective field :func:`effective_field` is the sum of the external
field, the anisotropy field, the exchange field between the same sublattice
of neighbouring layers and the magnetoelastic field. The LLG and iLLG add the
exchange field of the other sublattices of a layer, the LLB its transverse
part and the longitudinal field. All fields are in tesla. The time-dependent
inputs - the external field, the magnetization amplitude of the LLG and iLLG, the
temperature of the LLB and the strain - are given on the delay grid as
:class:`Inputs` and are interpolated linearly in time. The material parameters are the
:class:`~udkm1Dsim.structures.parameter_groups.MagneticParameters` of the
magnetic layers, which are stacked once per simulation into arrays by
:func:`stack_magnetic_parameters`. The equilibrium quantities of the LLB are
tabulated in temperature by :func:`tabulate_equilibrium`.

"""

__all__ = [
    "GAMMA_E",
    "MIN_MAG",
    "Inputs",
    "RHS",
    "stack_magnetic_parameters",
    "tabulate_equilibrium",
    "prepare_inputs",
    "time_weights",
    "interpolate_inputs",
    "external_field",
    "anisotropy_field",
    "interlayer_exchange_field",
    "magnetoelastic_field",
    "effective_field",
    "sublattice_exchange_field",
    "transverse_sublattice_exchange_field",
    "longitudinal_field",
    "llg_rhs",
    "illg_rhs",
    "llb_rhs",
    "solve",
]

__docformat__ = "restructuredtext"

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from .mean_field import TABLE, equilibrium_tables, interpolate_tables, temperature_grid

# gyromagnetic ratio of an electron [rad s^-1 T^-1], including its sign
GAMMA_E = -1.761e11

# floor for magnetization norms that appear in denominators
MIN_MAG = 1e-12

Inputs = namedtuple("Inputs", ["delays", "H_ext", "amplitude_map", "temp_map", "strain_map"])
Inputs.__doc__ = r"""Time-dependent inputs of the solver on the delay grid.

Attributes:
    delays (ndarray[float]): delays :math:`M` [s].
    H_ext (ndarray[float]): external magnetic field :math:`M \times 3` [T].
    amplitude_map (ndarray[float]): magnetization amplitude :math:`M \times N \times P`
        of the LLG and iLLG, ones for the LLB.
    temp_map (ndarray[float]): temperature :math:`M \times N` [K] of the
        LLB, zeros for the LLG and iLLG.
    strain_map (ndarray[float]): strain :math:`M \times N`.

"""


def stack_magnetic_parameters(layers):
    r"""stack_magnetic_parameters

    Stacks the parameters of the
    :class:`~udkm1Dsim.structures.parameter_groups.MagneticParameters` of
    the :math:`N` magnetic layers into arrays with a sublattice axis of
    length :math:`P = 1`, which keep the names of the parameters: scalars as
    :math:`N \times P`, vectors as :math:`N \times P \times 3` and the
    mean-field exchange coupling as matrix :math:`N \times P \times P`. The
    magnetization is the state of the solver and is left out. The mask
    ``active`` :math:`N \times P` is 1 for existing sublattices and 0 for
    padding.

    Args:
        layers (list[MagneticParameters]): magnetic parameters of the
            magnetic layers.

    Returns:
        mag (SimpleNamespace): stacked parameters [SI units].

    """
    N = len(layers)
    mag = SimpleNamespace(active=np.ones((N, 1)))
    if N == 0:
        return mag
    group = type(layers[0])
    for name in [*group._input_names(), *group._derived]:
        if name == "magnetization":
            continue
        values = np.array([getattr(layer, name).magnitude for layer in layers], dtype=float)
        setattr(mag, name, values[:, np.newaxis, ...])
    mag.mf_exch_coupling = mag.mf_exch_coupling[..., np.newaxis]
    return mag


def tabulate_equilibrium(layers, temp_max=None):
    r"""tabulate_equilibrium

    Tabulates the equilibrium quantities of the LLB of the magnetic layers,
    see :func:`~udkm1Dsim.simulations.solvers.mean_field.equilibrium_tables`,
    on a temperature grid that is refined around the Curie temperatures, see
    :func:`~udkm1Dsim.simulations.solvers.mean_field.temperature_grid`.
    Layers that share the same magnetic parameters are tabulated only once.

    Args:
        layers (list[MagneticParameters]): magnetic parameters of the
            magnetic layers.
        temp_max (float, optional): largest temperature of the simulation
            [K], which sets the range of the tables. Defaults to None.

    Returns:
        (tuple):
        - *temp_grid (ndarray[float])* - temperature grid :math:`N_T` [K].
        - *tables (ndarray[float])* - tables :math:`N \times P \times 4 \times N_T`,
          see :data:`~udkm1Dsim.simulations.solvers.mean_field.TABLE`.

    """
    mag = stack_magnetic_parameters(layers)
    N, P = mag.active.shape
    curie_temps = getattr(mag, "curie_temp", np.zeros((0, 1)))[:, 0]
    temp_max = 0.0 if temp_max is None else float(temp_max)
    T_max = max(1.2 * temp_max, 2 * float(np.max(curie_temps, initial=0.0)), 1000.0)
    temp_grid = temperature_grid(T_max, curie_temps)
    tables = np.zeros((N, P, 4, len(temp_grid)))
    cache = {}
    for k, layer in enumerate(layers):
        if id(layer) not in cache:
            cache[id(layer)] = equilibrium_tables(
                temp_grid,
                mag.mf_exch_coupling[k],
                mag.eff_spin[k],
                mag.mag_moment[k],
                mag.lamda[k],
                mag.active[k],
            )
        tables[k] = cache[id(layer)]
    return temp_grid, tables


def prepare_inputs(delays, N, H_ext, amplitude_map=None, temp_map=None, strain_map=None):
    r"""prepare_inputs

    Brings all time-dependent inputs onto their canonical shapes on the
    delay grid, filling in defaults for missing ones. Per-layer amplitude maps
    :math:`M \times N` are broadcast to all sublattices.

    Args:
        delays (ndarray[float]): delays range of simulation [s].
        N (int): number of magnetic layers.
        H_ext (ndarray[float]): external magnetic field [T], a single vector
            or one vector per delay :math:`M \times 3`.
        amplitude_map (ndarray[float], optional): magnetization amplitude map
            :math:`M \times N` or :math:`M \times N \times P` or None for
            ones.
        temp_map (ndarray[float], optional): temperature map
            :math:`M \times N` [K] or None for zeros.
        strain_map (ndarray[float], optional): strain map :math:`M \times N`
            or None for zeros.

    Returns:
        inputs (Inputs): inputs.

    """
    delays = np.ascontiguousarray(np.asarray(delays, dtype=float))
    M = len(delays)
    H_ext = np.asarray(H_ext, dtype=float)
    if H_ext.shape == (3,):
        H_ext = np.tile(H_ext, (M, 1))
    elif H_ext.shape != (M, 3):
        raise ValueError("H_ext must have the shape (3,) or (M, 3)!")

    def as_map(values, default):
        if values is None:
            return np.full((M, N), default, dtype=float)
        return np.ascontiguousarray(np.asarray(values, dtype=float).reshape(M, N))

    if amplitude_map is None:
        amplitude_map = np.ones((M, N, 1))
    amplitude_map = np.asarray(amplitude_map, dtype=float)
    if amplitude_map.ndim == 2:
        amplitude_map = amplitude_map[:, :, np.newaxis]

    return Inputs(
        delays,
        np.ascontiguousarray(H_ext),
        np.ascontiguousarray(amplitude_map.reshape(M, N, -1)),
        as_map(temp_map, 0.0),
        as_map(strain_map, 0.0),
    )


def time_weights(t, delays):
    """time_weights

    Index and weight for the linear interpolation of the inputs at time ``t``
    on the delay grid. Outside of the grid the boundary values are held.

    Args:
        t (float): time.
        delays (ndarray[float]): delays range of simulation [s].

    Returns:
        (tuple):
        - *idx (int)* - index of the lower grid point.
        - *w (float)* - weight of the upper grid point.

    """
    M = len(delays)
    if M < 2:
        return 0, 0.0
    idx = int(np.searchsorted(delays, t, side="right")) - 1
    idx = min(max(idx, 0), M - 2)
    w = (t - delays[idx]) / (delays[idx + 1] - delays[idx])
    return idx, min(max(w, 0.0), 1.0)


def interpolate_inputs(t, inputs):
    """interpolate_inputs

    Linear interpolation of all time-dependent inputs at time ``t``.

    Args:
        t (float): time.
        inputs (Inputs): inputs on the delay grid.

    Returns:
        (tuple):
        - *H_ext (ndarray[float])* - external field [T].
        - *amplitudes (ndarray[float])* - magnetization amplitudes
          :math:`N \\times P`.
        - *temps (ndarray[float])* - temperatures :math:`N` [K].
        - *strains (ndarray[float])* - strains :math:`N`.

    """
    idx, w = time_weights(t, inputs.delays)

    def interpolate(values):
        if w > 0:
            return (1 - w) * values[idx] + w * values[idx + 1]
        return values[idx]

    return (
        interpolate(inputs.H_ext),
        interpolate(inputs.amplitude_map),
        interpolate(inputs.temp_map),
        interpolate(inputs.strain_map),
    )


# effective field, term by term


def external_field(H_ext, N, P):
    """external_field

    The external magnetic field, the same for all layers and sublattices.

    Args:
        H_ext (ndarray[float]): external field [T].
        N (int): number of magnetic layers.
        P (int): number of sublattices.

    Returns:
        B (ndarray[float]): field :math:`N \\times P \\times 3` [T].

    """
    return np.tile(np.asarray(H_ext, dtype=float), (N, P, 1))


def anisotropy_field(m, ref_mags, mag):
    r"""anisotropy_field

    Anisotropy field of easy axes along the coordinate axes,
    :cite:t:`raposo2022` Eq. (eS12),

    .. math::

        \mathbf{B}_\mathrm{A} = \frac{2}{M_s}\, m_\mathrm{ref}^{\beta-2}
            \begin{bmatrix}K_x m_x\\K_y m_y\\K_z m_z\end{bmatrix}

    with the anisotropy constants :math:`K = (K_x, K_y, K_z)` and the exponent
    :math:`\beta` of their temperature dependence, which is expressed through
    the reference magnetization :math:`m_\mathrm{ref}`, the prescribed amplitude
    for the LLG and the equilibrium magnetization for the LLB. The field has a component along
    :math:`\mathbf{m}`, which enters the longitudinal dynamics of the LLB.

    Args:
        m (ndarray[float]): magnetization vectors :math:`N \times P \times 3`.
        ref_mags (ndarray[float]): reference magnetization :math:`N \times P`.
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.

    Returns:
        B (ndarray[float]): field :math:`N \times P \times 3` [T].

    """
    factor = 2 / mag.mag_saturation * _scaling(ref_mags, mag.aniso_exponent)
    return factor[..., np.newaxis] * mag.anisotropy * m


def interlayer_exchange_field(m, mag, exch_stiffnesses, thicknesses):
    r"""interlayer_exchange_field

    Exchange field between the same sublattice of neighbouring layers,
    for the :math:`i^\mathrm{th}` layer

    .. math::

        \mathbf{B}_{\mathrm{ex}, i} = \frac{2}{M_{s,i} \Delta z_i^2}
            \left(A_{i}^{i-1}\left(\mathbf{m}_{i-1}-\mathbf{m}_{i}\right)
            + A_i^{i+1}\left(\mathbf{m}_{i+1}-\mathbf{m}_{i}\right) \right)

    with the thicknesses :math:`\Delta z` and the directional exchange
    stiffnesses :math:`A` of the layers, see
    :meth:`~udkm1Dsim.simulations.magnetization.Magnetization.get_directional_exchange_stiffnesses`.
    Missing neighbours at the top and bottom do not contribute.

    Args:
        m (ndarray[float]): magnetization vectors :math:`N \times P \times 3`.
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.
        exch_stiffnesses (ndarray[float]): exchange stiffnesses towards the
            upper and lower magnetic layer :math:`N \times 2` [J/m].
        thicknesses (ndarray[float]): thicknesses :math:`N` [m].

    Returns:
        B (ndarray[float]): field :math:`N \times P \times 3` [T].

    """
    B = np.zeros_like(m)
    if len(m) < 2:
        return B
    es = 2 / (mag.mag_saturation * thicknesses[:, np.newaxis] ** 2)
    A = exch_stiffnesses
    B[1:] += (es[1:] * A[1:, 0, np.newaxis])[..., np.newaxis] * (m[:-1] - m[1:])
    B[:-1] += (es[:-1] * A[:-1, 1, np.newaxis])[..., np.newaxis] * (m[1:] - m[:-1])
    return B


def magnetoelastic_field(m, ref_mags, strains, mag):
    r"""magnetoelastic_field

    Magnetoelastic field of the out-of-plane strain :math:`\epsilon_{zz}` of
    the one-dimensional structure. The energy density
    :math:`B_1 \epsilon_{zz} m_z^2` with the magnetoelastic coupling
    :math:`B_1` acts as an additional anisotropy along :math:`z` with the same
    temperature scaling as the :func:`anisotropy_field`:

    .. math::

        \mathbf{B}_\mathrm{me} = -\frac{2 B_1 \epsilon_{zz}}{M_s}\,
            m_\mathrm{ref}^{\beta-2} \begin{bmatrix}0\\0\\m_z\end{bmatrix}

    Args:
        m (ndarray[float]): magnetization vectors :math:`N \times P \times 3`.
        ref_mags (ndarray[float]): reference magnetization :math:`N \times P`.
        strains (ndarray[float]): strains of the layers :math:`N`.
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.

    Returns:
        B (ndarray[float]): field :math:`N \times P \times 3` [T].

    """
    B = np.zeros_like(m)
    B[..., 2] = (
        -2
        * mag.magnetoelastic_coupling
        * strains[:, np.newaxis]
        / mag.mag_saturation
        * _scaling(ref_mags, mag.aniso_exponent)
        * m[..., 2]
    )
    return B


def effective_field(m, ref_mags, H_ext, strains, mag, exch_stiffnesses, thicknesses):
    r"""effective_field

    Static effective field common to all models

    .. math::

        \mathbf{B} = \mathbf{B}_\mathrm{ext} + \mathbf{B}_\mathrm{A}
            + \mathbf{B}_\mathrm{ex} + \mathbf{B}_\mathrm{me}

    the sum of the :func:`external_field`, the :func:`anisotropy_field`, the
    :func:`interlayer_exchange_field` and the :func:`magnetoelastic_field`.
    The exchange between the sublattices of a layer is model specific and
    added by the right-hand sides.

    Args:
        m (ndarray[float]): magnetization vectors :math:`N \times P \times 3`.
        ref_mags (ndarray[float]): reference magnetization :math:`N \times P`.
        H_ext (ndarray[float]): external field [T].
        strains (ndarray[float]): strains of the layers :math:`N`.
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.
        exch_stiffnesses (ndarray[float]): exchange stiffnesses towards the
            upper and lower magnetic layer :math:`N \times 2` [J/m].
        thicknesses (ndarray[float]): thicknesses :math:`N` [m].

    Returns:
        B (ndarray[float]): field :math:`N \times P \times 3` [T].

    """
    N, P = m.shape[0:2]
    return (
        external_field(H_ext, N, P)
        + anisotropy_field(m, ref_mags, mag)
        + interlayer_exchange_field(m, mag, exch_stiffnesses, thicknesses)
        + magnetoelastic_field(m, ref_mags, strains, mag)
    )


def sublattice_exchange_field(m, mag):
    r"""sublattice_exchange_field

    Full mean-field exchange field on sublattice :math:`a` due to the other
    sublattices :math:`b` of the same layer

    .. math::

        \mathbf{B}_\mathrm{ex}^{a} = \frac{1}{\mu_a} \sum_{b \neq a}
            J_{ab}\, \mathbf{m}_b

    used by the LLG and iLLG, where the amplitudes are prescribed. The
    intra-sublattice term :math:`J_{aa} \mathbf{m}_a` exerts no torque and is
    left out. Negative :math:`J_{ab}` couple antiferromagnetically. It
    vanishes for a single sublattice.

    Args:
        m (ndarray[float]): magnetization vectors :math:`N \times P \times 3`.
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.

    Returns:
        B (ndarray[float]): field :math:`N \times P \times 3` [T].

    """
    if m.shape[1] == 1:
        return np.zeros_like(m)
    J_off = _off_diagonal(mag.mf_exch_coupling)
    return np.einsum("nab,nbc->nac", J_off, m) / _safe_moments(mag.mag_moment)[..., None]


def transverse_sublattice_exchange_field(m, m_squared, mag):
    r"""transverse_sublattice_exchange_field

    Transverse part of the mean-field exchange field of the other sublattices
    for the LLB, :cite:t:`atxitia2012` Eq. (11) and :cite:t:`raposo2022`
    Eq. (eS13),

    .. math::

        \mathbf{B}_{\mathrm{ex},\perp}^{a} = \frac{1}{\mu_a} \sum_{b \neq a}
            J_{ab}\left(\mathbf{m}_b - \frac{(\mathbf{m}_a \cdot
            \mathbf{m}_b)}{m_a^2} \mathbf{m}_a\right)

    It vanishes for a single sublattice. The longitudinal part of the
    inter-sublattice exchange is contained in the :func:`longitudinal_field`.

    Args:
        m (ndarray[float]): magnetization vectors :math:`N \times P \times 3`.
        m_squared (ndarray[float]): squared norms :math:`N \times P`.
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.

    Returns:
        B (ndarray[float]): field :math:`N \times P \times 3` [T].

    """
    if m.shape[1] == 1:
        return np.zeros_like(m)
    J_off = _off_diagonal(mag.mf_exch_coupling)
    # dots[n, a, b] = m_a . m_b, Pi[n, a, b, :] = m_b - (m_a . m_b) m_a / m_a^2
    dots = np.einsum("nac,nbc->nab", m, m)
    Pi = (
        m[:, np.newaxis, :, :]
        - (dots / m_squared[:, :, np.newaxis])[..., np.newaxis] * m[:, :, np.newaxis, :]
    )
    return np.einsum("nab,nabc->nac", J_off, Pi) / _safe_moments(mag.mag_moment)[..., None]


def longitudinal_field(m, m_norm, me, chi, mag):
    r"""longitudinal_field

    Longitudinal field of the LLB, :cite:t:`raposo2022` Eqs. (eS14) and
    (eS15), parallel to :math:`\mathbf{m}_a`:

    .. math::

        \mathbf{B}_a^\parallel = \left[-\left(\frac{1}{\chi_a}
            + \sum_{b \neq a}\frac{|J_{ab}|}{\mu_a}\frac{\chi_b}{\chi_a}\right)
            \frac{m_a - m_{e,a}}{m_{e,a}}
            + \sum_{b \neq a}\frac{|J_{ab}|}{\mu_a}\frac{\tau_{ab} - m_{e,b}}
            {m_{e,a}}\right] \mathbf{m}_a

    below the ordering temperature, with the projection
    :math:`\tau_{ab} = |\mathbf{m}_a \cdot \mathbf{m}_b|/m_a` of the other
    sublattice onto :math:`\mathbf{m}_a`, and

    .. math::

        \mathbf{B}_a^\parallel = \left[-\left(\frac{1}{\chi_a}
            + \sum_{b \neq a}\frac{|J_{ab}|}{\mu_a}\frac{\chi_b}{\chi_a}\right)
            + \sum_{b \neq a}\frac{|J_{ab}|}{\mu_a}\frac{\tau_{ab}}{m_a}\right]
            \mathbf{m}_a

    at and above it, where :math:`m_{e,a} = 0`. It pulls the norm of every
    sublattice towards its equilibrium :math:`m_{e,a}(T)` with the
    longitudinal susceptibility :math:`\chi_a(T)` [1/T]. For a single
    sublattice it reduces to the thermal field
    :math:`-\frac{1}{\chi}\frac{m - m_e}{m_e}\mathbf{m}` of
    :cite:t:`nieves2014` below the Curie temperature.

    A sublattice without exchange coupling to other sublattices additionally
    gets the cubic term of :cite:t:`atxitia2012` at and above the ordering
    temperature:

    .. math::

        \mathbf{B}_a^\parallel = -\frac{1}{\chi_a}\left(1
            + c_S \frac{T_C}{T - T_C} m_a^2\right)\mathbf{m}_a
        = -\frac{1}{\chi_a}\mathbf{m}_a - c_S\frac{J_{aa}}{\mu_a}
            m_a^2\mathbf{m}_a

    using the mean-field susceptibility
    :math:`\chi_a = \mu_a T_C / (J_{aa}(T - T_C))`, which keeps it finite at
    :math:`T_C`. Expanding the Brillouin function up to third order yields
    :math:`c_S = \frac{3}{10}\frac{(S+1)^2 + S^2}{(S+1)^2}`, which is the
    classical :math:`c_\infty = 3/5` for an infinite effective spin.

    Args:
        m (ndarray[float]): magnetization vectors :math:`N \times P \times 3`.
        m_norm (ndarray[float]): norms :math:`N \times P`.
        me (ndarray[float]): equilibrium magnetizations :math:`N \times P`.
        chi (ndarray[float]): longitudinal susceptibilities
            :math:`N \times P` [1/T].
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.

    Returns:
        B (ndarray[float]): field :math:`N \times P \times 3` [T].

    """
    N, P = m_norm.shape
    mu = _safe_moments(mag.mag_moment)
    m_safe = np.maximum(m_norm, MIN_MAG)
    J_off = np.abs(_off_diagonal(mag.mf_exch_coupling))
    chi_safe = np.where(chi != 0, chi, np.inf)
    cubic = _cubic_coefficients(mag.eff_spin) * np.diagonal(mag.mf_exch_coupling, axis1=1, axis2=2)
    coef = np.zeros((N, P))
    for a in range(P):
        below = me[:, a] > 1e-6
        me_safe = np.where(below, me[:, a], 1.0)
        pref = 1 / chi_safe[:, a]
        cross_term = np.zeros(N)
        for b in range(P):
            if b == a:
                continue
            Jab = mag.active[:, b] * J_off[:, a, b] / mu[:, a]
            pref = pref + Jab * chi[:, b] / chi_safe[:, a]
            tau = np.abs(np.einsum("ni,ni->n", m[:, a], m[:, b])) / m_safe[:, a]
            cross_term += np.where(
                below, Jab * (tau - me[:, b]) / me_safe, Jab * tau / m_safe[:, a]
            )
        uncoupled = np.sum(mag.active * J_off[:, a, :], axis=1) == 0
        cubic_term = np.where(uncoupled, cubic[:, a] / mu[:, a] * m_norm[:, a] ** 2, 0.0)
        coef[:, a] = (
            np.where(below, -pref * (m_norm[:, a] - me[:, a]) / me_safe, -pref - cubic_term)
            + cross_term
        )
    return coef[..., np.newaxis] * m


# right-hand sides


def llg_rhs(t, y, inputs, mag, exch_stiffnesses, thicknesses):
    r"""llg_rhs

    Right-hand side of the Landau-Lifshitz-Gilbert equation in its
    Landau-Lifshitz form for the unit vector :math:`\mathbf{n}_a` of each
    sublattice of each layer

    .. math::

        \frac{d\mathbf{n}_a}{dt} = \frac{\gamma_e}{1+\alpha_a^2}\left(
            \mathbf{n}_a \times \mathbf{B}_a
            + \alpha_a\, \mathbf{n}_a \times (\mathbf{n}_a \times
            \mathbf{B}_a)\right)

    with the Gilbert damping :math:`\alpha_a` and :math:`\gamma_e = -1.761
    \times 10^{11}` rad/(s T). The field is the :func:`effective_field` plus
    the :func:`sublattice_exchange_field`, evaluated for the magnetization
    :math:`\mathbf{m}_a = m_a(t)\,\mathbf{n}_a` with the amplitude :math:`m_a(t)`
    from the inputs.

    Args:
        t (float): time [s].
        y (ndarray[float]): flattened unit vectors :math:`N \times P \times 3`.
        inputs (Inputs): time-dependent inputs.
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.
        exch_stiffnesses (ndarray[float]): exchange stiffnesses towards the
            upper and lower magnetic layer :math:`N \times 2` [J/m].
        thicknesses (ndarray[float]): thicknesses :math:`N` [m].

    Returns:
        dydt (ndarray[float]): flattened temporal derivative.

    """
    N, P = mag.active.shape
    H_ext, amplitudes, _, strains = interpolate_inputs(t, inputs)
    n = _normalize(y.reshape(N, P, 3))
    m = amplitudes[..., np.newaxis] * n
    B = effective_field(m, amplitudes, H_ext, strains, mag, exch_stiffnesses, thicknesses)
    B += sublattice_exchange_field(m, mag)
    alpha = mag.gilbert_damping
    precession = np.cross(n, B)
    damping = alpha[..., np.newaxis] * np.cross(n, precession)
    dndt = (GAMMA_E * mag.active / (1 + alpha**2))[..., np.newaxis] * (precession + damping)
    return dndt.ravel()


def illg_rhs(t, y, inputs, mag, exch_stiffnesses, thicknesses):
    r"""illg_rhs

    Right-hand side of the inertial Landau-Lifshitz-Gilbert equation

    .. math::

        \frac{d\mathbf{n}}{dt} = \gamma_e\, \mathbf{n} \times \mathbf{B}
            + \alpha\, \mathbf{n} \times \left(\frac{d\mathbf{n}}{dt}
            + \tau \frac{d^2\mathbf{n}}{dt^2}\right)

    of :cite:t:`ciornei2011` with the inertial relaxation time :math:`\tau`,
    split into two first-order equations for the unit vector
    :math:`\mathbf{n}` and its velocity :math:`\mathbf{v} = d\mathbf{n}/dt`
    of each sublattice:

    .. math::

        \frac{d\mathbf{n}}{dt} & = \mathbf{v} \\
        \frac{d\mathbf{v}}{dt} & = \frac{1}{\alpha \tau}\left(
            \gamma_e\, \mathbf{n} \times (\mathbf{n} \times \mathbf{B})
            - \alpha \mathbf{v} - \mathbf{n} \times \mathbf{v}\right)
            - \mathbf{n}\, |\mathbf{v}|^2

    The last term keeps :math:`|\mathbf{n}| = 1`. For :math:`\tau \rightarrow
    0` the velocity relaxes to the Landau-Lifshitz-Gilbert value, the free
    nutation frequency is :math:`1/(\alpha\tau)`. The field is the same as in
    :func:`llg_rhs`.

    Args:
        t (float): time [s].
        y (ndarray[float]): flattened unit vectors and velocities
            :math:`2 \times N \times P \times 3`.
        inputs (Inputs): time-dependent inputs.
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.
        exch_stiffnesses (ndarray[float]): exchange stiffnesses towards the
            upper and lower magnetic layer :math:`N \times 2` [J/m].
        thicknesses (ndarray[float]): thicknesses :math:`N` [m].

    Returns:
        dydt (ndarray[float]): flattened temporal derivative.

    """
    N, P = mag.active.shape
    H_ext, amplitudes, _, strains = interpolate_inputs(t, inputs)
    state = y.reshape(2, N, P, 3)
    n = _normalize(state[0])
    v = state[1]
    m = amplitudes[..., np.newaxis] * n
    B = effective_field(m, amplitudes, H_ext, strains, mag, exch_stiffnesses, thicknesses)
    B += sublattice_exchange_field(m, mag)
    alpha = mag.gilbert_damping[..., np.newaxis]
    # inactive sublattices have no damping or inertia, guard the division
    inertia = np.where(mag.active > 0, mag.gilbert_damping * mag.inertial_time, 1.0)
    torque = GAMMA_E * np.cross(n, np.cross(n, B))
    dvdt = (torque - alpha * v - np.cross(n, v)) / inertia[..., np.newaxis] - n * np.sum(
        v**2, axis=2
    )[..., np.newaxis]
    return (np.stack((v, dvdt)) * mag.active[np.newaxis, :, :, np.newaxis]).ravel()


def llb_rhs(t, y, inputs, mag, exch_stiffnesses, thicknesses, temp_grid, tables):
    r"""llb_rhs

    Right-hand side of the Landau-Lifshitz-Bloch equation in the form of
    :cite:t:`raposo2022` Eq. (eS7) for the magnetization :math:`\mathbf{m}_a`
    of each sublattice of each layer

    .. math::

        \frac{d\mathbf{m}_a}{dt} = \gamma_e \left(\mathbf{m}_a \times
            \mathbf{B}_a + \frac{\alpha_a^\perp}{m_a^2}\, \mathbf{m}_a \times
            (\mathbf{m}_a \times \mathbf{B}_a)
            - \frac{\alpha_a^\parallel}{m_a^2}\, (\mathbf{m}_a \cdot
            \mathbf{B}_a)\, \mathbf{m}_a\right)

    with the precession, the transverse and the longitudinal relaxation. The
    field is the :func:`effective_field` plus the
    :func:`transverse_sublattice_exchange_field` and the
    :func:`longitudinal_field`. The equilibrium magnetizations, the damping
    parameters :math:`\alpha^\parallel`, :math:`\alpha^\perp` and the
    susceptibilities depend on the temperature only and are interpolated from
    the :func:`~udkm1Dsim.simulations.solvers.mean_field.equilibrium_tables` of
    the layers: quantum with Brillouin functions for finite effective spins
    (:cite:t:`nieves2014`), classical with Langevin functions for infinite ones.

    Args:
        t (float): time [s].
        y (ndarray[float]): flattened magnetization vectors
            :math:`N \times P \times 3`.
        inputs (Inputs): time-dependent inputs.
        mag (SimpleNamespace): stacked magnetic parameters, see
            :func:`stack_magnetic_parameters`.
        exch_stiffnesses (ndarray[float]): exchange stiffnesses towards the
            upper and lower magnetic layer :math:`N \times 2` [J/m].
        thicknesses (ndarray[float]): thicknesses :math:`N` [m].
        temp_grid (ndarray[float]): temperature grid of the tables
            :math:`N_T` [K].
        tables (ndarray[float]): equilibrium tables, see
            :func:`tabulate_equilibrium`.

    Returns:
        dydt (ndarray[float]): flattened temporal derivative.

    """
    N, P = mag.active.shape
    H_ext, _, temps, strains = interpolate_inputs(t, inputs)
    m = y.reshape(N, P, 3)
    eq = interpolate_tables(temp_grid, tables, temps)
    me = eq[..., TABLE["mag"]]
    alpha_par = eq[..., TABLE["alpha_par"]]
    alpha_perp = eq[..., TABLE["alpha_perp"]]
    chi = eq[..., TABLE["chi"]]
    m_norm = np.sqrt(np.sum(m**2, axis=2))
    m_squared = np.maximum(m_norm, MIN_MAG) ** 2

    B = effective_field(m, me, H_ext, strains, mag, exch_stiffnesses, thicknesses)
    B += transverse_sublattice_exchange_field(m, m_squared, mag)
    B += longitudinal_field(m, m_norm, me, chi, mag)

    precession = np.cross(m, B)
    transverse = (alpha_perp / m_squared)[..., np.newaxis] * np.cross(m, precession)
    longitudinal = (alpha_par / m_squared * np.einsum("npi,npi->np", m, B))[..., np.newaxis] * m
    dmdt = GAMMA_E * (precession + transverse - longitudinal)
    return (mag.active[..., np.newaxis] * dmdt).ravel()


RHS = {"llg": llg_rhs, "illg": illg_rhs, "llb": llb_rhs}
"""Right-hand sides of the models by their names."""


# integration


def solve(
    model,
    init_state,
    inputs,
    layers,
    exch_stiffnesses,
    thicknesses,
    ode_options,
    progress_bar=False,
):
    r"""solve

    Solves the magnetization dynamics of the given ``model`` for all magnetic
    layers and their sublattices by ``scipy.integrate.solve_ivp``. The LLG and
    iLLG integrate the unit vectors of the magnetization, which are scaled by
    the prescribed amplitude of the ``inputs`` afterwards.

    Args:
        model (str): one of ``llg``, ``illg`` or ``llb``, see :data:`RHS`.
        init_state (ndarray[float]): initial state - cartesian
            :math:`N \times P \times 3` magnetization for ``llg`` and ``llb``
            and :math:`2 \times N \times P \times 3` for ``illg`` holding the
            unit vectors and their velocities.
        inputs (Inputs): time-dependent inputs on the delay grid.
        layers (list[MagneticParameters]): magnetic parameters of the
            magnetic layers.
        exch_stiffnesses (ndarray[float]): exchange stiffnesses towards the
            upper and lower magnetic layer :math:`N \times 2` [J/m].
        thicknesses (ndarray[float]): thicknesses :math:`N` [m].
        ode_options (dict): options for scipy solve_ivp ode solver.
        progress_bar (boolean, optional): enable tqdm progress bar. Defaults
            to False.

    Returns:
        magnetization_map (ndarray[float]): cartesian spatio-temporal
        magnetization map :math:`M \times N \times P \times 3` of the magnetic
        layers.

    """
    delays = inputs.delays
    M = len(delays)
    mag = stack_magnetic_parameters(layers)
    N, P = mag.active.shape
    if N == 0:
        return np.zeros((M, 0, P, 3))
    rhs = RHS[model]
    args = (inputs, mag, exch_stiffnesses, thicknesses)
    if model == "llb":
        temp_grid, tables = tabulate_equilibrium(layers, temp_max=np.max(inputs.temp_map))
        args += (temp_grid, tables)
    init_state = np.asarray(init_state, dtype=float)

    if progress_bar:
        pbar = tqdm(total=100)
        step = abs(delays[-1] - delays[0]) / 100
        state = {"done": 0}

        def fun(t, y, *args):
            done = int((t - delays[0]) / step) if step > 0 else 0
            if done > state["done"]:
                pbar.update(done - state["done"])
                pbar.set_description(f"Delay = {t * 1e12:.3f} ps")
                state["done"] = done
            return rhs(t, y, *args)

    else:
        pbar = None
        fun = rhs

    sol = solve_ivp(
        fun,
        [delays[0], delays[-1]],
        init_state.ravel(),
        args=args,
        t_eval=delays,
        **ode_options,
    )
    if pbar is not None:
        pbar.close()
    if not sol.success:
        raise RuntimeError("solve_ivp failed: " + sol.message)

    y = sol.y.T
    if model == "llb":
        magnetization_map = y.reshape([M, N, P, 3])
    else:
        # the unit vectors of the llg and illg scaled by the prescribed amplitude
        if model == "illg":
            y = y.reshape([M, 2, N, P, 3])[:, 0]
        magnetization_map = (
            _normalize(y.reshape([M, N, P, 3])) * inputs.amplitude_map[..., np.newaxis]
        )
    return magnetization_map * mag.active[np.newaxis, :, :, np.newaxis]


# helpers


def _safe_moments(mag_moments):
    """atomic moments as divisors: a vanishing moment yields a vanishing
    field instead of a division by zero. The moments are of the order of
    1e-23 J/T, far below the norm floor MIN_MAG, so they must not be floored."""
    mag_moments = np.asarray(mag_moments, dtype=float)
    return np.where(mag_moments > 0, mag_moments, np.inf)


def _cubic_coefficients(eff_spins):
    """coefficient c_S of the cubic longitudinal field above the Curie
    temperature from the third-order expansion of the Brillouin function,
    3/5 in the classical limit of an infinite spin"""
    eff_spins = np.asarray(eff_spins, dtype=float)
    with np.errstate(invalid="ignore"):
        c = 0.3 * ((eff_spins + 1) ** 2 + eff_spins**2) / (eff_spins + 1) ** 2
    return np.where(np.isinf(eff_spins), 0.6, c)


def _off_diagonal(matrices):
    """copies of the matrices with zeroed diagonals"""
    result = np.array(matrices, dtype=float, copy=True)
    P = result.shape[-1]
    result[..., np.arange(P), np.arange(P)] = 0.0
    return result


def _scaling(ref_mags, exponents):
    """temperature scaling m_ref^(beta-2) of the anisotropy and magnetoelastic
    fields, with the reference magnetization floored to avoid a division by
    zero for beta < 2"""
    return np.power(np.maximum(ref_mags, MIN_MAG), exponents - 2)


def _normalize(n):
    """unit vectors of the given vectors, leaving zero vectors untouched"""
    norms = np.sqrt(np.sum(n**2, axis=-1))
    norms[norms < MIN_MAG] = 1.0
    return n / norms[..., np.newaxis]
