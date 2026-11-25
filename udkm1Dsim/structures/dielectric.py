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

"""Energy-dependent dielectric functions of layers.

A :class:`Dispersion` is a single energy-dependent quantity, e.g. a tabulated
refractive index, which is evaluated as dielectric function
:math:`\\varepsilon`, i.e. the relative permittivity. A :class:`Dielectric`
combines the dispersions of a layer to its dielectric tensor, i.e. the
relative permittivity tensor,

.. math::

    \\varepsilon_{ij}(E, \\eta, \\mathbf{m}) = \\varepsilon^0_{ij}(E, \\eta)
    - i \\varepsilon^m(E, \\eta) \\epsilon_{ijk} m_k
    + \\frac{\\mathrm{d}\\varepsilon_{ij}}{\\mathrm{d}\\eta} \\eta

with the non-magnetic dielectric function :math:`\\varepsilon^0`, the
gyrotropic dielectric function :math:`\\varepsilon^m`, which is linear in the
magnetization :math:`\\mathbf{m}`, and the photoelastic change per strain
:math:`\\eta`, see :class:`Dielectric` for the strain models and the rotation
of the crystal by Euler angles.

A dispersion does not know, whether it is used for the diagonal, non-magnetic
part or the off-diagonal, gyrotropic part of the dielectric tensor. Hence, it
is evaluated with the vacuum contribution of its part, which is :math:`1` for
the non-magnetic and :math:`0` for the gyrotropic part, e.g. for the
conversion of a susceptibility :math:`\\chi = \\varepsilon - 1` and for the sum
of dispersions.

The imaginary part of the dielectric function is positive for absorption, i.e.
the refractive index is :math:`n = n' + i\\kappa` with :math:`\\kappa > 0` and
:math:`n = 1 - \\delta + i\\beta` for X-rays.

"""

__all__ = ["Dispersion", "Dielectric", "gyrotropic_tensor", "euler_matrix"]

__docformat__ = "restructuredtext"

import os
import warnings
from urllib.parse import parse_qs, urlparse

import numpy as np
import pint
from scipy import constants

from ..helpers import (
    SpectralTable,
    _format_energy,
    _md5,
    _range_text,
    convert_polar_to_cartesian,
    format_number,
    interpolate_table,
    read_columns,
    to_angle,
    to_energy,
    undefined_error,
)
from .parameters import Parameter

u = pint.get_application_registry()

# factor r_e (hc)² N_A / pi of the susceptibility from the atomic form
# factors, see :meth:`.Atom.get_susceptibility`, for the photon energy in eV
# and the molar density in mol/cm³, i.e. 830.358 by the CODATA constants
_ATOMIC_SUSCEPTIBILITY_FACTOR = (
    constants.physical_constants["classical electron radius"][0]
    * (constants.h * constants.c / constants.e) ** 2
    * constants.N_A
    / constants.pi
    * 1e6
)
# invalid values in the tables of Henke et al.
_INVALID_FORM_FACTOR = -9000
# vacuum contributions of a dispersion: 1 for the non-magnetic part and
# 0 for the magnetic part and the change per strain
_VACUUM_UNITY = 1.0
_VACUUM_ZERO = 0.0


def _to_tensor(eps):
    """Promote scalar values of shape (M,) to isotropic tensors of shape
    (M, 3, 3), while tensors are returned as they are."""
    if eps.ndim == 1:
        return eps[:, np.newaxis, np.newaxis] * np.eye(3)
    return eps


def _vacuum(vacuum, scalar):
    """Vacuum contribution as scalar or as tensor."""
    return vacuum if scalar else vacuum * np.eye(3)


def _invalid(eps):
    """Mask of the energies for which a dispersion is not defined."""
    invalid = np.isnan(eps.real) | np.isnan(eps.imag)
    return invalid if eps.ndim == 1 else invalid.any(axis=(1, 2))


def _vector_to_tensor(values, shape):
    """Tensors from uniaxial (⊥, ∥), diagonal, or full values with the given
    shape per photon energy. The leading axes are kept."""
    values = np.asarray(values, dtype=complex)
    if shape == (2,):
        values = values[..., [0, 0, 1]]
        shape = (3,)
    if shape == (3,):
        tensor = np.zeros(values.shape[:-1] + (3, 3), dtype=complex)
        for i in range(3):
            tensor[..., i, i] = values[..., i]
        return tensor
    return values


def euler_matrix(theta, phi, psi):
    r"""euler_matrix

    Rotation matrix of the proper z-x-z Euler angles
    :math:`(\theta, \phi, \psi)`

    .. math:: R = R_z(\phi) R_x(\theta) R_z(\psi)

    which rotates a tensor actively from the frame of the crystal to the
    frame of the sample, i.e. :math:`\varepsilon = R \varepsilon' R^T`.

    The frame of the sample is the one of :cite:t:`elzo2012` and Dyna:
    :math:`z` is the surface normal pointing out of the sample towards the
    superstrate, :math:`yz` is the plane of incidence and :math:`x` is along
    :math:`\sigma`, as for the magnetization, for the default ``azimuth``
    :math:`\Phi = 0` of the multilayer formalisms. The Euler angles orient
    the crystal within the fixed sample, while the ``azimuth`` rotates the
    plane of incidence relative to the sample counterclockwise about
    :math:`z`, see :attr:`.MultilayerScattering.azimuth`. Hence, the
    :math:`\sigma` polarization is along :math:`(\cos\Phi, \sin\Phi, 0)`,
    and for the tensor :math:`\varepsilon^0` alone, an azimuth :math:`\Phi`
    is equivalent to the Euler angle :math:`\phi - \Phi`.

    Without rotation, the crystal axes :math:`x'`, :math:`y'`, :math:`z'`
    coincide with :math:`x`, :math:`y`, :math:`z`. As for the
    `Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_
    :math:`(\alpha, \beta, \gamma)`,

    * :math:`\phi` is the angle of the line of nodes :math:`N`, i.e. the
      intersection of the :math:`xy` and :math:`x'y'` planes, from :math:`x`,
    * :math:`\theta` is the tilt of :math:`z'` from the surface normal
      :math:`z` about :math:`N`, and
    * :math:`\psi` is the angle of :math:`x'` from :math:`N` about
      :math:`z'`.

    Hence, :math:`z'` points along
    :math:`(\sin\theta\sin\phi, -\sin\theta\cos\phi, \cos\theta)`, i.e.
    it is tilted in the plane of incidence for :math:`\phi = 0`, and for
    :math:`\theta = 0`, :math:`\phi + \psi` is the azimuth of :math:`x'` from
    :math:`x`, i.e. from the :math:`\sigma` polarization, counterclockwise
    about :math:`z`.

    In contrast, `PyGTM <https://pygtm.readthedocs.io>`_ of
    :cite:t:`passler2017,passler2020` rotates the frame instead of the crystal,
    i.e. :math:`\varepsilon = R^T \varepsilon' R`, in its frame with the
    plane of incidence :math:`xz` and :math:`z` pointing into the sample.
    For its diagonal tensors of the principal values, its Euler angles
    :math:`(\theta_\mathrm{G}, \phi_\mathrm{G}, \psi_\mathrm{G})` correspond
    to :math:`(\theta, \phi, \psi) = (\theta_\mathrm{G}, 90° - \psi_\mathrm{G},
    -\phi_\mathrm{G})`.

    Args:
        theta (float): tilt of :math:`z'` from :math:`z` about the line of
            nodes (rad).
        phi (float): angle of the line of nodes from :math:`x` about
            :math:`z` (rad).
        psi (float): angle of :math:`x'` from the line of nodes about
            :math:`z'` (rad).

    Returns:
        R (ndarray[float]): rotation matrix.

    """

    def rot_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    c, s = np.cos(theta), np.sin(theta)
    rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return rot_z(phi) @ rot_x @ rot_z(psi)


def gyrotropic_tensor(eps0, eps_mag, magnetization, polar=False):
    r"""gyrotropic_tensor

    Complete dielectric tensor of a gyrotropic medium, i.e. the sum of the
    non-magnetic part :math:`\varepsilon^0` and the antisymmetric magnetic part
    of the strength :math:`\varepsilon^m`, which is linear in the magnetization
    :math:`\mathbf{m}`, following :cite:t:`elzo2012`:

    .. math:: \varepsilon_{ij} = \varepsilon^0_{ij}
        - i \varepsilon^m \epsilon_{ijk} m_k

    Args:
        eps0 (ndarray[complex]): non-magnetic dielectric function of shape
            (M,) or (..., M, 3, 3).
        eps_mag (ndarray[complex], None): gyrotropic dielectric function
            :math:`\varepsilon^m` per unit magnetization, i.e. the scalar
            strength of the magnetic part, of shape (M,) or (..., M).
        magnetization (tuple[float, Quantity], VectorParameter): cartesian
            magnetization :math:`(m_x, m_y, m_z)`, or for ``polar=True`` its
            amplitude and the angles :math:`\phi` and :math:`\gamma` as
            angle Quantities, e.g. ``(1, 90 * u.deg, 0 * u.deg)``.
        polar (bool, optional): the magnetization is given in polar
            coordinates. Defaults to False.

    Returns:
        eps (ndarray[complex]): dielectric tensor of shape (..., M, 3, 3).

    """
    eps = _to_tensor(np.asarray(eps0, dtype=complex)).copy()
    U = _cartesian_magnetization(magnetization, polar)
    if eps_mag is None or not np.any(U):
        return eps
    mag = np.asarray(eps_mag, dtype=complex)
    eps = np.broadcast_to(eps, np.broadcast_shapes(eps.shape, mag.shape + (3, 3))).copy()
    eps[..., 0, 1] += -1j * U[2] * mag
    eps[..., 0, 2] += 1j * U[1] * mag
    eps[..., 1, 0] += 1j * U[2] * mag
    eps[..., 1, 2] += -1j * U[0] * mag
    eps[..., 2, 0] += -1j * U[1] * mag
    eps[..., 2, 1] += 1j * U[0] * mag
    return eps


def atoms_susceptibility(atoms, energy, density, magnetic=False, select=None):
    r"""atoms_susceptibility

    Non-magnetic or gyrotropic susceptibility of atoms with the given number
    fractions and mass density :math:`\rho`, see
    :meth:`.Atom.get_susceptibility`. For several atoms, their mass numbers
    :math:`A` and form factors are averaged by their fractions as for an
    :class:`.AtomMixed`.

    Args:
        atoms (list[tuple[Atom, float]]): atoms and their number fractions.
        energy (ndarray[float]): photon energies (eV).
        density (float): mass density (kg/m³).
        magnetic (bool, optional): gyrotropic instead of the non-magnetic
            susceptibility. Defaults to False.
        select (tuple[int], optional): indices of the atoms, whose form
            factors contribute, e.g. of a magnetic sublattice, while the mass
            numbers of all atoms give their number density. Defaults to all.

    Returns:
        chi (ndarray[complex]): susceptibility.

    """
    energy = np.asarray(energy, dtype=float)
    mass_number = sum(fraction * atom.mass_number_a for atom, fraction in atoms)
    if mass_number == 0:
        return np.zeros_like(energy, dtype=complex)
    selected = atoms if select is None else [atoms[i] for i in select]
    if magnetic:
        f = sum(fraction * atom.get_magnetic_form_factor(energy) for atom, fraction in selected)
        sign = 1
    else:
        f = sum(fraction * atom.get_atomic_form_factor(energy) for atom, fraction in selected)
        sign = -1
    molar_density = density / 1000 / mass_number
    return sign * _ATOMIC_SUSCEPTIBILITY_FACTOR / energy**2 * molar_density * f


class Dispersion:
    r"""Dispersion

    An energy-dependent dielectric quantity, which is evaluated as dielectric
    function :math:`\varepsilon`. It is either a scalar, i.e. isotropic, or
    a 3x3 tensor per photon energy and is created by :meth:`from_array`,
    :meth:`from_ref_index`, :meth:`from_file`, :meth:`from_refractiveindex`,
    :meth:`from_opt_pen_depth`, :meth:`from_mass_attenuation`,
    :meth:`from_atoms`, or by :meth:`.Layer.dispersion_from_atoms`.

    Dispersions are combined by

    * ``a | b``: *fallback* - at every energy, ``a`` is used where it is
      defined, else ``b``, e.g. for an optical table and the atomic form
      factors in the X-ray range. A constant is defined at all energies and
      must hence be the last one.
    * ``a + b`` and ``a - b``: *sum* and *difference* of the contributions,
      i.e. of the susceptibilities :math:`\varepsilon_a \pm (\varepsilon_b - 1)`
      for the non-magnetic part, e.g. for several oscillators or to subtract
      a background.
    * ``c * a``: *scaling* of the contribution by the number ``c``, i.e. of
      the susceptibility :math:`1 + c (\varepsilon_a - 1)` for the
      non-magnetic part, e.g. for the fraction of a phase.

    Energies outside the range of a dispersion raise a ``ValueError``
    instead of an extrapolation.

    """

    scalar = True
    # how the dispersion was created, e.g. from a file, for its YAML
    # representation instead of the tabulated values, see Structure.to_yaml()
    _recipe = None

    @property
    def kinds(self):
        """set[str]: kinds of the input values, e.g. ``eps`` or ``n``."""
        return set()

    @property
    def energy_range(self):
        """tuple[float]: lowest and highest photon energy (eV) of the dispersion."""
        return -np.inf, np.inf

    @property
    def energy_ranges(self):
        """list[tuple[float]]: intervals of photon energies (eV), in which the
        dispersion is defined, e.g. with a gap between an optical table and
        the atomic form factors."""
        return [self.energy_range]

    @property
    def needs_density(self):
        """bool: true if the dispersion is evaluated for the density of a layer."""
        return False

    def _eps(self, energy, density, vacuum, eps_xx):
        """Dielectric function for the photon energies (eV) and the vacuum
        contribution of its part, which is NaN for energies outside the
        dispersion."""
        raise NotImplementedError

    def evaluate(self, energy, density=None, vacuum=_VACUUM_UNITY, eps_xx=None):
        """evaluate

        Dielectric function of the dispersion for the vacuum contribution of
        its part of the dielectric tensor, see :meth:`eps`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            density (float, optional): mass density of the layer (kg/m³),
                which is required by dispersions from atoms.
            vacuum (float, optional): vacuum contribution, which is 1 for the
                non-magnetic and 0 for the gyrotropic part. Defaults to 1.
            eps_xx (ndarray[complex], optional): diagonal dielectric function,
                which is required by a Voigt constant.

        Returns:
            eps (ndarray[complex]): dielectric function of shape (M,) or
            (M, 3, 3).

        """
        spectral = energy
        energy = np.atleast_1d(to_energy(energy))
        eps = self._eps(energy, density, vacuum, eps_xx)
        invalid = _invalid(eps)
        if np.any(invalid):
            raise undefined_error(
                repr(self),
                energy,
                invalid,
                self.energy_ranges,
                spectral,
                hint="Combine it with another Dispersion by '|' for the missing range.",
            )
        return eps

    def eps(self, energy, density=None, eps_xx=None):
        """eps

        Dielectric function :math:`\\varepsilon` of the dispersion as
        non-magnetic part of a dielectric tensor.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            density (float, optional): mass density of the layer (kg/m³),
                which is required by dispersions from atoms.
            eps_xx (ndarray[complex], optional): diagonal dielectric function,
                which is required by a Voigt constant.

        Returns:
            eps (ndarray[complex]): dielectric function of shape (M,) or
            (M, 3, 3).

        """
        return self.evaluate(energy, density, _VACUUM_UNITY, eps_xx)

    def chi(self, energy, density=None, eps_xx=None):
        """chi

        Susceptibility :math:`\\chi = \\varepsilon - 1` of the dispersion as
        non-magnetic part of a dielectric tensor.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            density (float, optional): mass density of the layer (kg/m³),
                which is required by dispersions from atoms.
            eps_xx (ndarray[complex], optional): diagonal dielectric function,
                which is required by a Voigt constant.

        Returns:
            chi (ndarray[complex]): susceptibility of shape (M,) or (M, 3, 3).

        """
        eps = self.eps(energy, density, eps_xx)
        return eps - _vacuum(_VACUUM_UNITY, eps.ndim == 1)

    def _label(self):
        """Short text of the dispersion for its display."""
        return repr(self)

    def __or__(self, other):
        return _Fallback(self, _check_dispersion(other))

    def __add__(self, other):
        return _Sum(self, _check_dispersion(other))

    def __sub__(self, other):
        return _Sum(self, _check_dispersion(other), sign=-1)

    def __mul__(self, factor):
        if isinstance(factor, Dispersion) or not np.isscalar(factor):
            return NotImplemented
        return _Scaled(self, factor)

    __rmul__ = __mul__

    def __neg__(self):
        return _Scaled(self, -1)

    # ------------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------------

    @classmethod
    def from_array(cls, x, values, kind="eps", molar_mass=None):
        """from_array

        Dispersion from tabulated values or from a constant. The shape of the
        values gives the symmetry of the dispersion:

        * ``(N,)``: isotropic
        * ``(N, 2)``: uniaxial with the components perpendicular and parallel
          to the c-axis of the crystal, which is along the surface normal
          without ``euler_angles`` of the :class:`Dielectric`
        * ``(N, 3)``: diagonal
        * ``(N, 3, 3)``: full tensor

        and the same without the first axis for a constant.

        Args:
            x (None, float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or wavelengths, frequencies, or
                wavenumbers as pint Quantity. ``None`` for a constant.
            values (complex, ndarray[complex]): tabulated values.
            kind (str, optional): ``eps`` for the dielectric function, ``n``
                for the refractive index, ``delta_beta`` for
                :math:`\\delta + i\\beta` of the refractive index
                :math:`n = 1 - \\delta + i\\beta`, e.g. of X-ray tables, or
                for the magnetic part for :math:`\\Delta\\delta + i\\Delta\\beta`,
                which are defined by
                :math:`\\varepsilon^m = 2 (\\Delta\\delta - i\\Delta\\beta)`, see
                :meth:`Dielectric.get_delta_beta_mag`,
                ``chi`` for the susceptibility
                :math:`\\chi = \\varepsilon - 1`, or ``voigt`` for the Voigt
                constant :math:`Q` of the gyrotropic part with
                :math:`\\varepsilon^m = \\varepsilon_{xx} Q`. For the
                gyrotropic part, ``eps`` and ``chi`` are the same. Moreover,
                ``f`` and ``m`` for the atomic and magnetic form factors
                :math:`f_1 - i f_2` and :math:`m_1 - i m_2` per formula unit
                with the ``molar_mass``, see :meth:`from_form_factors`.
                Defaults to ``eps``.
            molar_mass (float, Quantity, optional): molar mass of a formula
                unit for the kinds ``f`` and ``m``, where floats are taken in
                (g/mol).

        Returns:
            dispersion (Dispersion): dispersion.

        """
        if kind in ("f", "m"):
            if x is None or np.ndim(x) == 0:
                raise ValueError("Form factors must be tabulated for the photon energies!")
            return _FormFactors(SpectralTable(x, values), molar_mass, magnetic=kind == "m")
        if kind not in ("eps", "n", "delta_beta", "chi", "voigt"):
            raise ValueError(
                "kind must be one of 'eps', 'n', 'delta_beta', 'chi', 'voigt', 'f', or 'm'!"
            )
        values = np.asarray(values, dtype=complex)
        constant = x is None or np.ndim(x) == 0
        if constant:
            energy = None
            shape = values.shape
        else:
            energy = np.atleast_1d(to_energy(x))
            if energy.ndim != 1 or len(energy) < 2:
                raise ValueError("A table requires at least two photon energies!")
            if values.shape[:1] != energy.shape:
                raise ValueError(
                    f"The first axis of the values {values.shape} must match the "
                    f"{len(energy)} photon energies!"
                )
            shape = values.shape[1:]
        if shape not in ((), (2,), (3,), (3, 3)):
            raise ValueError(
                "The values must be scalars, uniaxial (2), diagonal (3), or full "
                f"tensors (3, 3) per photon energy, not of shape {shape}!"
            )
        scalar = shape == ()
        if kind == "voigt" and not scalar:
            raise ValueError("A Voigt constant must be a scalar!")
        if kind in ("n", "delta_beta") and shape == (3, 3):
            raise ValueError("A refractive index can only be given for the principal axes!")
        input_values = values
        stored_kind = kind
        if kind == "delta_beta" and not scalar:
            # n = 1 - delta + i beta of the principal axes of the non-magnetic part
            values = 1 - values.real + 1j * values.imag
        if not scalar:
            values = _vector_to_tensor(values, shape)
        if kind == "n" or (kind == "delta_beta" and not scalar):
            # the refractive indices of the principal axes are on the diagonal
            values = values**2
            stored_kind = "eps"
        if constant:
            return _Constant(values, scalar, stored_kind, input_values, kind)
        return _Table(energy, values, scalar, stored_kind, kind)

    @classmethod
    def from_ref_index(cls, n):
        """from_ref_index

        Constant, isotropic dispersion from a complex refractive index
        :math:`n = n' + i\\kappa`, or uniaxial (⊥, ∥) and diagonal for two or
        three indices.

        Args:
            n (complex, ndarray[complex]): refractive index.

        Returns:
            dispersion (Dispersion): dispersion.

        """
        return cls.from_array(None, n, kind="n")

    @classmethod
    def from_file(cls, filename, x_unit=None, kind="n", delimiter=None, molar_mass=None):
        """from_file

        Dispersion from a text file with the three columns of the spectral
        axis and the real and imaginary part, e.g. an ``.nk`` file or a
        table of `refractiveindex.info <https://refractiveindex.info>`_.
        Lines starting with ``#`` and lines without numbers are skipped.

        Args:
            filename (str): path of the file.
            x_unit (str, optional): unit of the spectral axis, e.g. ``um``,
                ``nm``, ``eV``, ``THz``, or ``1/cm``. Defaults to ``eV`` for
                the kinds ``delta_beta``, ``f``, and ``m`` of X-ray tables and
                to ``um`` else.
            kind (str, optional): ``n`` for the refractive index, ``eps`` for
                the dielectric function, ``delta_beta`` for the columns
                :math:`\\delta` and :math:`\\beta` of the refractive index
                :math:`n = 1 - \\delta + i\\beta`, e.g. of the tables of
                `CXRO <https://henke.lbl.gov/optical_constants/>`_, ``chi``
                for the susceptibility, or ``f`` and ``m`` for the columns
                :math:`f_1, f_2` or :math:`m_1, m_2` of atomic or magnetic
                form factors, e.g. of ``.cf``, ``.nff``, or ``.mf`` files, see
                :meth:`from_form_factors`. Defaults to ``n``.
            delimiter (str, optional): column delimiter, by default white
                space or a comma.
            molar_mass (float, Quantity, optional): molar mass of a formula
                unit for the kinds ``f`` and ``m``, where floats are taken in
                (g/mol).

        Returns:
            dispersion (Dispersion): dispersion.

        """
        recipe = {
            "file": str(filename),
            "x_unit": x_unit,
            "kind": kind,
            "delimiter": delimiter,
            "molar_mass": molar_mass,
        }
        if x_unit is None:
            x_unit = "eV" if kind in ("delta_beta", "f", "m") else "um"
        data = read_columns(filename, delimiter)
        if kind in ("f", "m"):
            table = SpectralTable.from_columns(
                data,
                x_unit=x_unit,
                imaginary_sign=-1,
                invalid_below=_INVALID_FORM_FACTOR,
                source=os.path.basename(str(filename)),
            )
            dispersion = _FormFactors(table, molar_mass, magnetic=kind == "m")
        else:
            data = data[~np.isnan(data[:, 1:3]).any(axis=1)]
            values = data[:, 1] + 1j * data[:, 2]
            dispersion = cls.from_array(data[:, 0] * u(x_unit), values, kind=kind)
            dispersion._origin = os.path.basename(str(filename))
        dispersion._recipe = recipe
        return dispersion

    @classmethod
    def from_form_factors(cls, filename, molar_mass, magnetic=False, x_unit="eV"):
        r"""from_form_factors

        Dispersion from tabulated atomic or magnetic form factors per formula
        unit, e.g. of ``.cf``, ``.nff``, or ``.mf`` files of an element or a
        compound, which are evaluated for the density :math:`\rho` of the
        layer with the number density :math:`n_a = \rho / M` of the formula
        units for their molar mass :math:`M`:

        .. math::

            \varepsilon^0 = 1 - \frac{r_e \lambda^2}{\pi} n_a f, \quad
            \varepsilon^m = \frac{r_e \lambda^2}{\pi} n_a m

        The same files are read by :class:`.Atom`, see
        :meth:`.Layer.dispersion_from_atoms` for the dispersion from the atoms
        of a layer.

        Args:
            filename (str): path of the file with the columns of the photon
                energy, :math:`f_1`, and :math:`f_2` or :math:`m_1` and
                :math:`m_2`.
            molar_mass (float, Quantity): molar mass of a formula unit, where
                floats are taken in g/mol.
            magnetic (bool, optional): magnetic instead of atomic form factors.
                Defaults to False.
            x_unit (str, optional): unit of the spectral axis. Defaults to eV.

        Returns:
            dispersion (Dispersion): dispersion.

        """
        return cls.from_file(
            filename, x_unit=x_unit, kind="m" if magnetic else "f", molar_mass=molar_mass
        )

    @classmethod
    def from_refractiveindex(cls, shelf=None, book=None, page=None, url=None, num=1000):
        """from_refractiveindex

        Dispersion from the `refractiveindex.info <https://refractiveindex.info>`_
        database of optical constants :cite:`polyanskiy2024`, which is
        accessed by the optional package `refractiveindex
        <https://github.com/toftul/refractiveindex>`_, e.g. installed by
        ``pip install udkm1Dsim[refractiveindex]``. The database is part of
        this package, so that no internet connection is needed.

        A page of the database is given by its ``shelf``, ``book``, and
        ``page``, or by the ``url`` of the page, e.g.
        ``https://refractiveindex.info/?shelf=main&book=Cr&page=Johnson``.
        Pages can be found by ``refractiveindex.search()``, e.g.
        ``refractiveindex.search("SiO2", quantity="k", wavelength=800)``.

        The dielectric function :math:`\\varepsilon = (n + i\\kappa)^2` of the
        page is tabulated for ``num`` wavelengths, which are distributed
        logarithmically in the range of the page, in which both :math:`n` and
        :math:`\\kappa` are given. Pages without :math:`\\kappa` raise a warning,
        as :math:`\\kappa = 0` is assumed.

        Args:
            shelf (str, optional): shelf of the page, e.g. ``main``.
            book (str, optional): book of the page, e.g. the material ``SiO2``.
            page (str, optional): page, e.g. the reference ``Malitson``.
            url (str, optional): url of the page instead of its ``shelf``,
                ``book``, and ``page``.
            num (int, optional): number of tabulated wavelengths. Defaults to
                1000.

        Returns:
            dispersion (Dispersion): dispersion.

        """
        try:
            import refractiveindex
        except ImportError:
            raise ImportError(
                "Dispersion.from_refractiveindex() requires the optional package "
                "refractiveindex, e.g. by pip install udkm1Dsim[refractiveindex]!"
            ) from None
        if url is not None:
            if any(value is not None for value in (shelf, book, page)):
                raise ValueError("Give either the url or the shelf, book, and page, not both!")
            material = refractiveindex.RefractiveIndexMaterial.from_url(url)
        elif None in (shelf, book, page):
            raise ValueError("The shelf, book, and page or the url of the page must be given!")
        else:
            material = refractiveindex.RefractiveIndexMaterial(shelf=shelf, book=book, page=page)
        try:
            material.get_wl_range(unit="um", quantity="k")
        except refractiveindex.NoExtinctionCoefficient:
            name = url if url is not None else f"{shelf}/{book}/{page}"
            warnings.warn(f"kappa = 0 for {name} without extinction coefficient", stacklevel=2)
        lo, hi = material.get_wl_range(unit="um")
        wavelength = np.geomspace(lo, hi, int(num))
        eps = np.asarray(material.get_epsilon(wavelength, unit="um"), dtype=complex)
        dispersion = cls.from_array(wavelength * u.um, eps, kind="eps")
        if url is not None:
            query = parse_qs(urlparse(url).query)
            shelf, book, page = (query.get(key, ["?"])[0] for key in ("shelf", "book", "page"))
        dispersion._origin = f"refractiveindex.info: {shelf}/{book}/{page}"
        dispersion._recipe = {
            "refractiveindex": {"shelf": shelf, "book": book, "page": page, "num": int(num)},
            "version": getattr(refractiveindex, "DATABASE_VERSION", "unknown"),
        }
        return dispersion

    @classmethod
    def from_opt_pen_depth(cls, opt_pen_depth, n=1.0):
        r"""from_opt_pen_depth

        Isotropic dispersion with a constant optical penetration depth of the
        intensity :math:`\zeta`, i.e. with the extinction coefficient

        .. math:: \kappa(\lambda) = \frac{\lambda}{4 \pi \zeta}

        at all wavelengths :math:`\lambda` and the constant real part
        :math:`n'` of the refractive index. This is mostly useful for the
        Lambert-Beer absorption, as :math:`n'` is usually not known, see
        :meth:`.Layer.set_opt_pen_depth` to keep the real part of a given
        dispersion.

        Args:
            opt_pen_depth (float, Quantity): optical penetration depth
                :math:`\zeta`, where floats are taken in m.
            n (float, optional): real part :math:`n'` of the refractive
                index. Defaults to 1.

        Returns:
            dispersion (Dispersion): dispersion.

        """
        return _OptPenDepth(opt_pen_depth, n=n)

    @classmethod
    def from_mass_attenuation(cls, x, mass_attenuation):
        r"""from_mass_attenuation

        Isotropic dispersion from tabulated mass attenuation coefficients
        :math:`\mu/\rho`, e.g. of the `NIST tables
        <https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients>`_, which
        give the imaginary part of the refractive index for the density
        :math:`\rho` of the layer:

        .. math:: \beta = \frac{\lambda}{4 \pi} \frac{\mu}{\rho} \rho

        The real part is not given by the mass attenuation and is set to
        :math:`n' = 1`, i.e. :math:`\delta = 0`, with a warning.

        Args:
            x (ndarray[float], Quantity): photon energies, where floats are
                taken in eV, or wavelengths, frequencies, or wavenumbers as
                pint Quantity.
            mass_attenuation (ndarray[float], Quantity): mass attenuation
                coefficients, where floats are taken in cm²/g.

        Returns:
            dispersion (Dispersion): dispersion.

        """
        warnings.warn(
            "The mass attenuation only gives the imaginary part of the refractive "
            "index, so that its real part is set to 1, i.e. delta = 0.",
            stacklevel=2,
        )
        return _MassAttenuation(x, mass_attenuation)

    @classmethod
    def from_atoms(cls, atoms, magnetic=False):
        r"""from_atoms

        Dispersion from the atomic or magnetic form factors of atoms, which
        are evaluated for the density :math:`\rho` of the layer, see
        :meth:`.Atom.get_susceptibility`. For several atoms, their mass
        numbers and form factors are averaged by their number fractions as
        for an :class:`.AtomMixed`, e.g. the fractions of a formula unit:

        .. code-block:: python

            Dispersion.from_atoms(Fe)
            Dispersion.from_atoms([(Fe, 2), (O, 3)])
            Dispersion.from_atoms(Fe, magnetic=True)

        Args:
            atoms (Atom, AtomMixed, list[tuple[Atom, float]]): atom or atoms
                and their number fractions.
            magnetic (bool, optional): gyrotropic dispersion from the
                magnetic form factors instead of the non-magnetic one from the
                atomic form factors. Defaults to False.

        Returns:
            dispersion (Dispersion): dispersion.

        """
        return _Atoms(_atoms_and_fractions(atoms), magnetic=magnetic)


def _atoms_and_fractions(atoms):
    """atoms and their number fractions from an atom or a list of atoms and
    fractions, see :meth:`Dispersion.from_atoms`"""
    if hasattr(atoms, "atomic"):  # a single Atom or AtomMixed
        return [(atoms, 1.0)]
    try:
        pairs = [(atom, float(fraction)) for atom, fraction in atoms]
    except (TypeError, ValueError):
        raise TypeError(
            "The atoms must be an Atom, an AtomMixed, or a list of atoms and their number "
            "fractions, e.g. [(Fe, 2), (O, 3)]!"
        ) from None
    if not pairs or any(not hasattr(atom, "atomic") for atom, _ in pairs):
        raise TypeError("The atoms must be a non-empty list of Atom or AtomMixed and fractions!")
    if any(fraction < 0 for _, fraction in pairs) or sum(f for _, f in pairs) <= 0:
        raise ValueError("The number fractions of the atoms must be positive!")
    return pairs


_KIND_SYMBOLS = {
    "eps": "ε",
    "n": "n",
    "delta_beta": "δ + iβ",
    "chi": "χ",
    "voigt": "Q",
    "f": "f",
    "m": "m",
}


def _format_value(value, mode="text", near_one=False):
    """Format a complex number, which is shown as real number without an
    imaginary part. Values close to 1, e.g. of X-rays, are shown as deviation
    from 1, e.g. ``1 + (−2.6×10⁻⁴ + 2.1×10⁻⁴i)``."""
    value = complex(value)
    if near_one and value != 1 and abs(value - 1) < 0.1:
        return "1 + (" + _format_value(value - 1, mode) + ")"
    return format_number(value.real if value.imag == 0 else value, mode)


def _format_values(values):
    """Text of a scalar or of the components of a vector."""
    values = np.asarray(values)
    if values.ndim == 0:
        return _format_value(values.item())
    return "(" + ", ".join(_format_value(v) for v in values.ravel()) + ")"


def _check_dispersion(value):
    if not isinstance(value, Dispersion):
        raise TypeError(f"Only a Dispersion can be combined, not {type(value).__name__}!")
    return value


def _convert(values, kind, scalar, vacuum, eps_xx):
    """Dielectric function from stored values of the kind ``eps``, ``chi``,
    ``delta_beta``, or ``voigt``."""
    if kind == "chi":
        return values + _vacuum(vacuum, scalar)
    if kind == "delta_beta":
        # the vacuum contribution tells the part of the dielectric tensor
        if vacuum == _VACUUM_UNITY:
            # non-magnetic part: n = 1 - delta + i beta
            return (1 - values.real + 1j * values.imag) ** 2
        # gyrotropic part: definition of Delta delta and Delta beta as scaled
        # magnetic form factors, which holds for any n
        return 2 * (values.real - 1j * values.imag)
    if kind == "voigt":
        if eps_xx is None:
            raise ValueError("A Voigt constant requires the diagonal dielectric function eps_xx!")
        return values * eps_xx
    return values


class _Constant(Dispersion):
    """Constant dispersion."""

    def __init__(self, values, scalar, kind, input_values, input_kind):
        self._values = np.asarray(values, dtype=complex)
        self.scalar = scalar
        self._kind = kind
        self._input_values = input_values
        self._input_kind = input_kind

    @property
    def kinds(self):
        return {self._input_kind}

    def _eps(self, energy, density, vacuum, eps_xx):
        values = np.broadcast_to(self._values, energy.shape + self._values.shape).copy()
        return _convert(values, self._kind, self.scalar, vacuum, eps_xx)

    @property
    def value(self):
        """complex, ndarray[complex]: the constant as given on creation."""
        return self._input_values

    def __repr__(self):
        value = self._input_values
        value = value.item() if value.ndim == 0 else value.tolist()
        return f"Dispersion({self._input_kind}={value})"

    def _label(self):
        return f"{_KIND_SYMBOLS[self._input_kind]} = {_format_values(self._input_values)}"


class _Table(Dispersion):
    """Tabulated dispersion, which is linearly interpolated."""

    def __init__(self, energy, values, scalar, kind, input_kind):
        self._table = SpectralTable(energy, values)
        self._energy = self._table.energy
        self._values = self._table.values
        self.scalar = scalar
        self._kind = kind
        self._input_kind = input_kind
        self._origin = ""

    @property
    def kinds(self):
        return {self._input_kind}

    @property
    def energy_range(self):
        return self._energy[0], self._energy[-1]

    def _eps(self, energy, density, vacuum, eps_xx):
        return _convert(self._table(energy), self._kind, self.scalar, vacuum, eps_xx)

    def _label(self):
        lo, hi = self.energy_range
        shape = "" if self.scalar else " tensors"
        source = f" ({self._origin})" if self._origin else ""
        return (
            f"table of {_KIND_SYMBOLS[self._input_kind]}{shape}: {len(self._energy)} values "
            f"from {_format_energy(lo)} to {_format_energy(hi)}{source}"
        )

    def __repr__(self):
        # the repr is part of the hash of the simulation results, which must
        # not depend on the location of the file of the table, but only on
        # its values given by their md5 digest
        lo, hi = self.energy_range
        return (
            f"Dispersion({self._input_kind}: {len(self._energy)} values from {lo:.4g} eV "
            f"to {hi:.4g} eV, md5={self._table.md5})"
        )


class _FormFactors(Dispersion):
    """Dispersion from tabulated atomic or magnetic form factors per formula
    unit with the given molar mass, which are the susceptibilities of the
    non-magnetic or gyrotropic part for the density of the layer."""

    def __init__(self, table, molar_mass, magnetic=False):
        if molar_mass is None:
            raise ValueError("Form factors require the molar_mass of their formula unit!")
        if isinstance(molar_mass, u.Quantity):
            molar_mass = molar_mass.to("g/mol").magnitude
        self._table = table
        self._molar_mass = float(molar_mass)
        self._magnetic = magnetic

    @property
    def kinds(self):
        return {"m" if self._magnetic else "f"}

    @property
    def needs_density(self):
        return True

    @property
    def energy_range(self):
        return self._table.energy_range

    def _eps(self, energy, density, vacuum, eps_xx):
        if density is None:
            raise ValueError(
                "A Dispersion from form factors requires the density of its layer, "
                "e.g. by Layer.get_dielectric_tensor()!"
            )
        sign = 1 if self._magnetic else -1
        molar_density = density / 1000 / self._molar_mass
        chi = (
            sign * _ATOMIC_SUSCEPTIBILITY_FACTOR / energy**2 * molar_density * self._table(energy)
        )
        return chi + vacuum

    def __repr__(self):
        kind = "magnetic form factors" if self._magnetic else "atomic form factors"
        lo, hi = self.energy_range
        source = f", source={self._table.source}" if self._table.source else ""
        return (
            f"Dispersion({kind}: {len(self._table)} values from {lo:.4g} eV to {hi:.4g} eV"
            f"{source}, molar_mass={self._molar_mass} g/mol, md5={self._table.md5})"
        )

    def _label(self):
        kind = "magnetic form factors" if self._magnetic else "atomic form factors"
        lo, hi = self.energy_range
        source = f" ({self._table.source})" if self._table.source else ""
        return (
            f"table of {kind}: {len(self._table)} values from {_format_energy(lo)} to "
            f"{_format_energy(hi)}{source}, M = {self._molar_mass:.4g} g/mol"
        )


class _Atoms(Dispersion):
    """Dispersion from the atomic or magnetic form factors of atoms, which
    are the susceptibilities of the non-magnetic or gyrotropic part."""

    def __init__(self, atoms, magnetic=False, select=None):
        self._atoms = [(atom, float(fraction)) for atom, fraction in atoms]
        self._magnetic = magnetic
        # indices of the contributing atoms, e.g. of a sublattice
        self._select = None if select is None else tuple(int(i) for i in select)

    def _selected(self):
        """contributing atoms and their fractions"""
        if self._select is None:
            return self._atoms
        return [self._atoms[i] for i in self._select]

    @property
    def kinds(self):
        return {"chi"}

    @property
    def needs_density(self):
        return True

    @property
    def energy_range(self):
        lo, hi = -np.inf, np.inf
        for atom, _ in self._selected():
            if self._magnetic:
                a_lo, a_hi = atom.magnetic_form_factor_range
            else:
                a_lo, a_hi = atom.atomic_form_factor_range
            lo, hi = max(lo, a_lo), min(hi, a_hi)
        return lo, hi

    def _eps(self, energy, density, vacuum, eps_xx):
        if density is None:
            raise ValueError(
                "A Dispersion from atoms requires the density of its layer, "
                "e.g. by Layer.get_dielectric_tensor()!"
            )
        # the form factors of the atoms raise an error outside their tables
        lo, hi = self.energy_range
        defined = (energy >= lo) & (energy <= hi)
        chi = np.full(energy.shape, np.nan, dtype=complex)
        if np.any(defined):
            chi[defined] = atoms_susceptibility(
                self._atoms, energy[defined], density, self._magnetic, self._select
            )
        return chi + vacuum

    def __repr__(self):
        atoms = ", ".join(f"{atom.id}: {fraction:.4g}" for atom, fraction in self._atoms)
        tables = []
        for atom, _ in self._atoms:
            coeff = (
                atom.magnetic_form_factor_coeff
                if self._magnetic
                else (atom.atomic_form_factor_coeff)
            )
            if coeff is not None:
                tables.append(np.asarray(coeff))
        kind = "magnetic form factors" if self._magnetic else "atomic form factors"
        select = "" if self._select is None else f", select={list(self._select)}"
        return f"Dispersion({kind} of {atoms}{select}, md5={_md5(*tables)})"

    def _label(self):
        # atoms of the same id are counted, e.g. 3×O of a unit cell
        fractions = {}
        for atom, fraction in self._atoms:
            fractions[atom.id] = fractions.get(atom.id, 0) + fraction
        smallest = min(fractions.values())
        names = []
        for name, fraction in fractions.items():
            count = fraction / smallest
            if len(fractions) == 1 or np.isclose(count, 1):
                names.append(name)
            elif np.isclose(count, round(count)):
                names.append(f"{round(count)}×{name}")
            else:
                names.append(f"{name} ({fraction:.3g})")
        kind = "magnetic form factors" if self._magnetic else "atomic form factors"
        lo, hi = self.energy_range
        energies = (
            "" if not np.isfinite(lo) else (f" from {_format_energy(lo)} to {_format_energy(hi)}")
        )
        if self._select is not None:
            selected = ", ".join(dict.fromkeys(atom.id for atom, _ in self._selected()))
            return f"{kind} of {selected} in {', '.join(names)}{energies}"
        return f"{kind} of {', '.join(names)}{energies}"


def _format_length(length):
    """Short text of a length (m)."""
    return f"{(length * u.m).to_compact():.4g~P}"


class _OptPenDepth(Dispersion):
    """Isotropic dispersion with a constant optical penetration depth, i.e. an
    extinction coefficient proportional to the wavelength, and the real part
    of the refractive index of a constant or of another dispersion."""

    def __init__(self, opt_pen_depth, n=1.0, base=None):
        if isinstance(opt_pen_depth, u.Quantity):
            opt_pen_depth = opt_pen_depth.to("m").magnitude
        self._opt_pen_depth = float(opt_pen_depth)
        if not self._opt_pen_depth > 0:
            raise ValueError("The optical penetration depth must be positive!")
        self._n = float(np.real(n))
        if base is not None and not base.scalar:
            raise ValueError("The optical penetration depth requires an isotropic dispersion!")
        self._base = base

    @property
    def opt_pen_depth(self):
        """Quantity: optical penetration depth of the intensity."""
        return (self._opt_pen_depth * u.m).to_compact()

    @property
    def kinds(self):
        return {"opt_pen_depth"} | (self._base.kinds if self._base is not None else set())

    @property
    def needs_density(self):
        return self._base is not None and self._base.needs_density

    def _eps(self, energy, density, vacuum, eps_xx):
        if vacuum != _VACUUM_UNITY:
            raise ValueError("An optical penetration depth is only defined for eps0!")
        n_real = np.full(energy.shape, self._n)
        if self._base is not None:
            # the real part of the base, which is n' at energies without base
            base = np.sqrt(self._base._eps(energy, density, vacuum, eps_xx)).real
            n_real = np.where(np.isnan(base), n_real, base)
        with np.errstate(divide="ignore"):
            wavelength = (energy * u.eV).to("m", "sp").magnitude
        kappa = wavelength / (4 * np.pi * self._opt_pen_depth)
        return (n_real + 1j * kappa) ** 2

    def __repr__(self):
        base = f"n={self._n}" if self._base is None else f"n={self._base!r}"
        return f"Dispersion(opt_pen_depth={self._opt_pen_depth} m, {base})"

    def _label(self):
        zeta = f"ζ = {_format_length(self._opt_pen_depth)} (κ = λ/4πζ)"
        if self._base is None:
            return f"{zeta}, n' = {format_number(self._n)}"
        text = f"{zeta}, n' from ({self._base._label()})"
        if np.isfinite(self._base.energy_range).all():
            text += f", else n' = {format_number(self._n)}"
        return text


class _MassAttenuation(Dispersion):
    """Isotropic dispersion from tabulated mass attenuation coefficients,
    which give the imaginary part of the refractive index for the density of
    the layer, while its real part is 1."""

    def __init__(self, x, mass_attenuation):
        energy = np.atleast_1d(to_energy(x))
        if isinstance(mass_attenuation, u.Quantity):
            mass_attenuation = mass_attenuation.to("m**2/kg").magnitude
        else:
            mass_attenuation = (
                (np.asarray(mass_attenuation) * u.cm**2 / u.g).to("m**2/kg").magnitude
            )
        mass_attenuation = np.atleast_1d(np.asarray(mass_attenuation, dtype=float))
        if mass_attenuation.shape != energy.shape or len(energy) < 2:
            raise ValueError("The mass attenuation must be tabulated for at least two energies!")
        order = np.argsort(energy)
        self._energy = energy[order]
        self._values = mass_attenuation[order]

    @property
    def kinds(self):
        return {"mass_attenuation"}

    @property
    def needs_density(self):
        return True

    @property
    def energy_range(self):
        return self._energy[0], self._energy[-1]

    def _eps(self, energy, density, vacuum, eps_xx):
        if density is None:
            raise ValueError(
                "A Dispersion from the mass attenuation requires the density of its "
                "layer, e.g. by Layer.get_dielectric_tensor()!"
            )
        mass_attenuation = interpolate_table(self._energy, self._values, energy).real
        wavelength = (energy * u.eV).to("m", "sp").magnitude
        beta = wavelength / (4 * np.pi) * mass_attenuation * density
        return (1 + 1j * beta) ** 2

    def __repr__(self):
        lo, hi = self.energy_range
        return (
            f"Dispersion(mass_attenuation: {len(self._energy)} values from {lo:.4g} eV to "
            f"{hi:.4g} eV, md5={_md5(self._energy, self._values)})"
        )

    def _label(self):
        lo, hi = self.energy_range
        return (
            f"table of μ/ρ: {len(self._energy)} values from {_format_energy(lo)} "
            f"to {_format_energy(hi)}, δ = 0"
        )


class _Fallback(Dispersion):
    """Dispersion a, else b at the energies not defined by a."""

    def __init__(self, a, b):
        self._a, self._b = a, b
        self.scalar = a.scalar and b.scalar

    @property
    def kinds(self):
        return self._a.kinds | self._b.kinds

    @property
    def needs_density(self):
        return self._a.needs_density or self._b.needs_density

    @property
    def energy_range(self):
        a_lo, a_hi = self._a.energy_range
        b_lo, b_hi = self._b.energy_range
        return min(a_lo, b_lo), max(a_hi, b_hi)

    @property
    def energy_ranges(self):
        intervals = sorted(self._a.energy_ranges + self._b.energy_ranges)
        merged = [list(intervals[0])]
        for lo, hi in intervals[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return [tuple(interval) for interval in merged]

    def _eps(self, energy, density, vacuum, eps_xx):
        a = self._a._eps(energy, density, vacuum, eps_xx)
        missing = _invalid(a)
        if not self.scalar:
            a = _to_tensor(a)
        if np.any(missing):
            xx = None if eps_xx is None else eps_xx[missing]
            b = self._b._eps(energy[missing], density, vacuum, xx)
            a[missing] = b if self.scalar else _to_tensor(b)
        return a

    def __repr__(self):
        return f"({self._a!r} | {self._b!r})"

    def _label(self):
        return f"{self._a._label()} | {self._b._label()}"


class _Sum(Dispersion):
    """Sum or difference of the contributions of a and b, i.e. a + sign *
    (b - vacuum)."""

    def __init__(self, a, b, sign=1):
        self._a, self._b = a, b
        self._sign = sign
        self.scalar = a.scalar and b.scalar

    @property
    def kinds(self):
        return self._a.kinds | self._b.kinds

    @property
    def needs_density(self):
        return self._a.needs_density or self._b.needs_density

    @property
    def energy_range(self):
        a_lo, a_hi = self._a.energy_range
        b_lo, b_hi = self._b.energy_range
        return max(a_lo, b_lo), min(a_hi, b_hi)

    def _eps(self, energy, density, vacuum, eps_xx):
        a = self._a._eps(energy, density, vacuum, eps_xx)
        b = self._b._eps(energy, density, vacuum, eps_xx)
        if self.scalar:
            return a + self._sign * (b - vacuum)
        return _to_tensor(a) + self._sign * (_to_tensor(b) - vacuum * np.eye(3))

    @property
    def _operator(self):
        return "+" if self._sign > 0 else "-"

    def __repr__(self):
        return f"({self._a!r} {self._operator} {self._b!r})"

    def _label(self):
        a, b = self._a._label(), self._b._label()
        a = f"({a})" if isinstance(self._a, _Fallback) else a
        b = f"({b})" if isinstance(self._b, (_Fallback, _Sum)) else b
        return f"{a} {self._operator} {b}"


class _Scaled(Dispersion):
    """Contribution of a scaled by a number, i.e. vacuum + factor * (a -
    vacuum)."""

    def __init__(self, a, factor):
        self._a = a
        self._factor = complex(factor) if np.iscomplexobj(factor) else float(factor)
        self.scalar = a.scalar

    @property
    def kinds(self):
        return self._a.kinds

    @property
    def needs_density(self):
        return self._a.needs_density

    @property
    def energy_range(self):
        return self._a.energy_range

    @property
    def energy_ranges(self):
        return self._a.energy_ranges

    def _eps(self, energy, density, vacuum, eps_xx):
        a = self._a._eps(energy, density, vacuum, eps_xx)
        offset = _vacuum(vacuum, self.scalar)
        return offset + self._factor * (a - offset)

    def __repr__(self):
        return f"({self._factor!r} * {self._a!r})"

    def _label(self):
        return f"{format_number(self._factor)} × ({self._a._label()})"


def _cartesian_magnetization(magnetization, polar=False):
    """Cartesian magnetization (m_x, m_y, m_z) of a magnetization, given as
    VectorParameter, as cartesian components, or for ``polar=True`` as
    (amplitude, phi, gamma) with the angles as angle Quantities, see
    :func:`~udkm1Dsim.helpers.to_angle`."""
    if magnetization is None:
        return np.zeros(3)
    if hasattr(magnetization, "polar_rad"):
        # a VectorParameter defines its coordinates itself
        return magnetization.magnitude
    if isinstance(magnetization, u.Quantity):
        if polar:
            raise TypeError(
                "The polar magnetization must be given by its amplitude and its angles "
                "separately, e.g. (1, 90 * u.deg, 0 * u.deg)!"
            )
        magnetization = magnetization.to("").magnitude
    values = []
    for i, value in enumerate(magnetization):
        if polar and i > 0:
            value = to_angle(value, name=f"angle {('phi', 'gamma')[i - 1]} of the magnetization")
        elif isinstance(value, u.Quantity):
            value = value.to("").magnitude
        values.append(float(value))
    if len(values) != 3:
        raise ValueError(
            "The magnetization must be given by its amplitude, phi, and gamma!"
            if polar
            else "The magnetization must be given by its cartesian components x, y, and z!"
        )
    if polar:
        return convert_polar_to_cartesian(np.array(values))
    return np.array(values)


def _is_magnetization_list(magnetization):
    """A list of magnetizations of several sublattices, i.e. of VectorParameters
    or of vectors, in contrast to the components of a single vector."""
    if magnetization is None or hasattr(magnetization, "polar_rad"):
        return False
    if isinstance(magnetization, (u.Quantity, np.ndarray)):
        return np.ndim(magnetization) == 2
    return (
        isinstance(magnetization, (list, tuple))
        and len(magnetization) > 0
        and all(np.ndim(m) == 1 or hasattr(m, "polar_rad") for m in magnetization)
    )


def _cartesian_magnetizations(magnetization, polar=False):
    """Cartesian magnetization of shape (3,), or (P, 3) for a list of the
    magnetizations of P sublattices, see :func:`_cartesian_magnetization`."""
    if _is_magnetization_list(magnetization):
        return np.array([_cartesian_magnetization(m, polar) for m in magnetization])
    return _cartesian_magnetization(magnetization, polar)


class Dielectric(Parameter):
    r"""Dielectric

    Dielectric function, i.e. relative permittivity, of a layer, which is
    given by a :class:`Dispersion` for the non-magnetic dielectric function
    :math:`\varepsilon^0` in ``eps0``, an optional scalar :class:`Dispersion`
    for the gyrotropic dielectric function :math:`\varepsilon^m` in
    ``eps_mag``, and the dependence on the strain :math:`\eta`:

    .. math::

        \varepsilon_{ij}(E, \eta, \mathbf{m}) = \varepsilon^0_{ij}(E, \eta)
        - i \varepsilon^m(E, \eta) \epsilon_{ijk} m_k
        + \frac{\mathrm{d}\varepsilon_{ij}}{\mathrm{d}\eta} \eta

    i.e. the non-magnetic, the magnetic, and the photoelastic term. The strain
    model gives the dependence on the strain :math:`\eta` with the scaling
    :math:`s(\eta) = 1/(1+\eta)` of the susceptibilities with the density of
    the layer:

    * ``none``: no dependence, i.e.
      :math:`\varepsilon^0_{ij}(E, \eta) = \varepsilon^0_{ij}(E)`,
      :math:`\varepsilon^m(E, \eta) = \varepsilon^m(E)`, and no photoelastic
      term.
    * ``density``: both susceptibilities scale with the density, i.e.
      :math:`\varepsilon^0_{ij}(E, \eta) = \delta_{ij}
      + s(\eta) (\varepsilon^0_{ij}(E) - \delta_{ij})` and
      :math:`\varepsilon^m(E, \eta) = s(\eta) \varepsilon^m(E)`, without
      photoelastic term, e.g. for the dielectric tensor from the atomic form factors.
    * ``photoelastic``: the photoelastic term is added with the measured,
      total change per strain ``deps_dstrain``, e.g. given by the
      photoelastic coefficients, which already includes the change of the
      density. Hence, the non-magnetic part is not scaled, i.e.
      :math:`\varepsilon^0_{ij}(E, \eta) = \varepsilon^0_{ij}(E)`, while the
      gyrotropic part still scales with the density,
      :math:`\varepsilon^m(E, \eta) = s(\eta) \varepsilon^m(E)`.

    The non-magnetic part is given in the frame of the crystal, which is
    rotated into the frame of the sample, i.e. with :math:`z` along the
    surface normal, by the z-x-z ``euler_angles``
    :math:`(\theta, \phi, \psi)`, see :func:`euler_matrix`:
    :math:`\varepsilon^0 = R \varepsilon^{0\prime} R^T`. The magnetization of
    the layer and the photoelastic change ``deps_dstrain`` are given in the
    frame of the sample and are not rotated. The ``azimuth`` of the
    multilayer formalisms, i.e. of the plane of incidence relative to the
    fixed sample, rotates the whole tensor of all layers, including the
    magnetization, see :attr:`.MultilayerScattering.azimuth`.

    The dielectric tensor is calculated by :meth:`get_tensor` or
    :meth:`.Layer.get_dielectric_tensor` for the actual photon energies,
    strain, and magnetization of a simulation.

    Args:
        eps0 (Dispersion, optional): non-magnetic dispersion. Without, the
            dielectric is not set.
        eps_mag (Dispersion, list[Dispersion], optional): scalar gyrotropic
            dispersion, i.e. the off-diagonal element of the dielectric
            tensor, of kind ``eps``, ``chi``, ``delta_beta``, ``voigt``, or
            magnetic form factors, or a list of them with one entry per
            magnetic sublattice of the layer, where None marks a sublattice
            without gyrotropic part, see ``gyrotropy``.
        strain_model (str, optional): ``none``, ``density``, or
            ``photoelastic``. The default ``density`` scales the
            susceptibilities with the density of the layer by
            :math:`1/(1+\eta)`, which holds for X-rays and is an estimate
            in the optical range, where the strain also shifts the bands.
            Defaults to ``density``.
        deps_dstrain (complex, ndarray[complex], Dispersion, optional):
            measured, total change of the dielectric function per strain for
            the strain model ``photoelastic`` in the frame of the sample.
            Defaults to 0.
        euler_angles (Quantity, tuple[Quantity], optional): z-x-z Euler
            angles :math:`(\theta, \phi, \psi)` of the crystal as angle
            Quantities, e.g. ``(30, 0, 0) * u.deg``, see
            :func:`~udkm1Dsim.helpers.to_angle`. Defaults to no rotation.
        gyrotropy (str, optional): coupling of the gyrotropic part to the
            magnetization of a layer with several magnetic sublattices:

            * ``"net"``: the sum of the gyrotropic dispersions
              :math:`\sum_a \varepsilon^m_a` couples to the net magnetization,
              i.e. the average of the magnetizations of the sublattices
              weighted by their saturation magnetization, as the net
              ``magnetization_map`` of the magnetization simulations.
            * ``"sublattices"``: the gyrotropic dispersion of every sublattice
              couples to its own magnetization,
              :math:`-i \sum_a \varepsilon^m_a \epsilon_{ijk} m_{a,k}`, e.g. for
              the element-specific magnetic dichroism at the absorption edges
              of the elements of a ferrimagnet. It requires one dispersion per
              sublattice.

            Both are approximations, and a layer with several sublattices
            raises an error until one of them is chosen. For a single
            sublattice, it has no effect. Defaults to None.

    """

    _strain_models = ("none", "density", "photoelastic")

    def __init__(
        self,
        eps0=None,
        eps_mag=None,
        strain_model="density",
        deps_dstrain=0.0,
        euler_angles=None,
        gyrotropy=None,
        name="",
        symbol=r"\varepsilon",
    ):
        self._eps0 = None
        self._eps_mag = None
        self._gyrotropy = None
        self._strain_model = "density"
        self._deps_dstrain = 0.0
        self._euler_angles = None
        self._display_energy = None
        super().__init__("", magnitude=None, name=name, symbol=symbol)
        self._set_fields(eps0, eps_mag, strain_model, deps_dstrain, euler_angles, gyrotropy)

    @classmethod
    def from_atoms(cls, atoms, fallback=None, sublattices=None):
        """from_atoms

        Dielectric from atoms with the non-magnetic dispersion from their
        atomic form factors, the gyrotropic dispersion from their magnetic
        form factors, if any is tabulated, and the strain model ``density``,
        see :meth:`Dispersion.from_atoms`:

        .. code-block:: python

            Dielectric.from_atoms(Fe)
            Dielectric.from_atoms([(Fe, 2), (O, 3)])
            Dielectric.from_atoms(Fe, fallback=Dispersion.from_ref_index(2.9 + 3.4j))
            # one gyrotropic dispersion per magnetic sublattice
            Dielectric.from_atoms([(Fe, 0.75), (Gd, 0.25)], sublattices=[0, 1])

        With the ``sublattices`` of the atoms, the gyrotropic dispersion of
        every sublattice follows from the magnetic form factors of its atoms
        for the number density of all atoms, so that their sum equals the
        gyrotropic dispersion without sublattices, and the ``gyrotropy`` is
        ``"sublattices"``, see :class:`Dielectric`.

        Args:
            atoms (Atom, AtomMixed, list[tuple[Atom, float]]): atom or atoms
                and their number fractions.
            fallback (Dispersion, optional): non-magnetic dispersion, which
                is used outside of the energy range of the atomic form factors,
                e.g. a refractive index in the optical range.
            sublattices (list[int, None], optional): index of the magnetic
                sublattice of every atom, or None for a non-magnetic atom.
                Defaults to a single gyrotropic dispersion of all atoms.

        Returns:
            dielectric (Dielectric): dielectric from the atoms.

        """
        atoms = _atoms_and_fractions(atoms)
        eps0 = _Atoms(atoms)
        if fallback is not None:
            eps0 = eps0 | fallback
        if sublattices is None:
            magnetic = any(atom.has_magnetic_form_factor for atom, _ in atoms)
            return cls(
                eps0=eps0,
                eps_mag=_Atoms(atoms, magnetic=True) if magnetic else None,
                strain_model="density",
            )
        sublattices = list(sublattices)
        if len(sublattices) != len(atoms):
            raise ValueError(
                f"The {len(atoms):d} atoms require one sublattice each, not {len(sublattices):d}!"
            )
        indices = [index for index in sublattices if index is not None]
        if not indices or any(int(index) != index or index < 0 for index in indices):
            raise ValueError("The sublattices must be indices from 0 or None!")
        eps_mag = []
        for index in range(int(max(indices)) + 1):
            select = [
                i
                for i, (sublattice, (atom, _)) in enumerate(zip(sublattices, atoms))
                if sublattice == index and atom.has_magnetic_form_factor
            ]
            eps_mag.append(_Atoms(atoms, magnetic=True, select=select) if select else None)
        if all(part is None for part in eps_mag):
            raise ValueError("None of the atoms of the sublattices has magnetic form factors!")
        return cls(eps0=eps0, eps_mag=eps_mag, strain_model="density", gyrotropy="sublattices")

    def _set_fields(
        self, eps0, eps_mag, strain_model, deps_dstrain, euler_angles=None, gyrotropy=None
    ):
        self._eps0 = self._check_eps0(eps0)
        self._eps_mag = self._check_eps_mag(eps_mag)
        self._strain_model = self._check_strain_model(strain_model)
        self._deps_dstrain = self._check_deps_dstrain(deps_dstrain)
        self._euler_angles = self._check_euler_angles(euler_angles)
        self._gyrotropy = self._check_gyrotropy(gyrotropy)

    @staticmethod
    def _check_euler_angles(value):
        """Euler angles in rad of angle Quantities."""
        if value is None:
            return None
        angles = np.atleast_1d(to_angle(value, name="Euler angles")).ravel()
        angles = tuple(float(angle) for angle in angles)
        if len(angles) != 3:
            raise ValueError("The Euler angles must be given as (theta, phi, psi)!")
        return angles

    @staticmethod
    def _check_eps0(value):
        if value is not None and not isinstance(value, Dispersion):
            raise TypeError(f"eps0 must be a Dispersion or None, not {type(value).__name__}!")
        if value is not None and value.kinds & {"voigt", "m"}:
            raise ValueError(
                "eps0 is the non-magnetic part of the dielectric tensor and cannot be "
                "given as Voigt constant or magnetic form factors!"
            )
        return value

    _gyrotropies = (None, "net", "sublattices")

    @classmethod
    def _check_gyrotropy(cls, value):
        if value not in cls._gyrotropies:
            raise ValueError(f"gyrotropy must be one of {cls._gyrotropies}, not {value!r}!")
        return value

    @classmethod
    def _check_eps_mag(cls, value):
        """a single gyrotropic dispersion or a list of them, one per sublattice"""
        if isinstance(value, (list, tuple)):
            if len(value) == 0 or all(v is None for v in value):
                raise ValueError("eps_mag must have a Dispersion for at least one sublattice!")
            return [None if v is None else cls._check_eps_mag(v) for v in value]
        if value is None:
            return None
        if not isinstance(value, Dispersion):
            raise TypeError(f"eps_mag must be a Dispersion or None, not {type(value).__name__}!")
        if not value.scalar:
            raise ValueError("eps_mag must be a scalar Dispersion!")
        if value.kinds & {"n", "opt_pen_depth", "mass_attenuation", "f"}:
            raise ValueError(
                "eps_mag is the off-diagonal, gyrotropic element of the dielectric "
                "tensor and cannot be given as refractive index, penetration depth, mass "
                "attenuation, or atomic form factors. Use kind='eps', 'delta_beta', 'voigt', "
                "or 'm'!"
            )
        return value

    @classmethod
    def _check_strain_model(cls, value):
        if value not in cls._strain_models:
            raise ValueError(f"strain_model must be one of {cls._strain_models}!")
        return value

    @staticmethod
    def _check_deps_dstrain(value):
        if isinstance(value, Dispersion):
            return value
        value = np.asarray(value, dtype=complex)
        if value.shape not in ((), (2,), (3,), (3, 3)):
            raise ValueError("deps_dstrain must be a scalar, uniaxial, diagonal, or full tensor!")
        return value if value.ndim == 0 else _vector_to_tensor(value, value.shape)

    def _notify(self):
        if self._caller is not None:
            self._caller._update_depending()

    # ------------------------------------------------------------------------
    # value of the parameter
    # ------------------------------------------------------------------------

    @property
    def magnitude(self):
        """The dielectric itself, which is set by another :class:`Dielectric` or by the inputs of a
        dispersion, as the value of a parameter.
        """
        return self

    @magnitude.setter
    def magnitude(self, value):
        if value is None:
            fields = (None, None, "none", 0.0, None, None)
        elif isinstance(value, Dielectric):
            fields = (
                value.eps0,
                value.eps_mag,
                value.strain_model,
                value.deps_dstrain,
                value._euler_angles,
                value.gyrotropy,
            )
        elif isinstance(value, Dispersion):
            fields = (value, None, "none", 0.0, None, None)
        else:
            raise TypeError(
                f"Cannot set the Dielectric from type {type(value).__name__}. "
                "Use a Dispersion, e.g. Dispersion.from_ref_index(n), or a Dielectric!"
            )
        self._set_fields(*fields)
        self._notify()

    @property
    def quantity(self):
        """The dielectric itself, as :attr:`magnitude`."""
        return self

    @quantity.setter
    def quantity(self, value):
        self.magnitude = value

    @property
    def is_set(self):
        """bool: true if the non-magnetic dispersion is set."""
        return self._eps0 is not None

    @property
    def eps0(self):
        """Dispersion: non-magnetic dielectric function :math:`\\varepsilon^0`."""
        return self._eps0

    @eps0.setter
    def eps0(self, value):
        self._eps0 = self._check_eps0(value)
        self._notify()

    @property
    def eps_mag(self):
        """Dispersion, list[Dispersion]: gyrotropic dielectric function
        :math:`\\varepsilon^m`, or one per magnetic sublattice."""
        return self._eps_mag

    @eps_mag.setter
    def eps_mag(self, value):
        self._eps_mag = self._check_eps_mag(value)
        self._notify()

    @property
    def gyrotropy(self):
        """str: coupling of the gyrotropic part to the magnetizations of
        several sublattices, ``"net"`` or ``"sublattices"``, or None."""
        return self._gyrotropy

    @gyrotropy.setter
    def gyrotropy(self, value):
        self._gyrotropy = self._check_gyrotropy(value)
        self._notify()

    def _eps_mag_parts(self):
        """gyrotropic dispersions of all sublattices, a single one, or none"""
        if self._eps_mag is None:
            return []
        return self._eps_mag if isinstance(self._eps_mag, list) else [self._eps_mag]

    @property
    def strain_model(self):
        """str: dependence on the strain, ``none``, ``density``, or
        ``photoelastic``."""
        return self._strain_model

    @strain_model.setter
    def strain_model(self, value):
        self._strain_model = self._check_strain_model(value)
        self._notify()

    @property
    def deps_dstrain(self):
        """complex, ndarray[complex], Dispersion: measured, total change of the
        dielectric function per strain for the strain model
        ``photoelastic``."""
        return self._deps_dstrain

    @deps_dstrain.setter
    def deps_dstrain(self, value):
        self._deps_dstrain = self._check_deps_dstrain(value)
        self._notify()

    @property
    def euler_angles(self):
        """Quantity: z-x-z Euler angles :math:`(\\theta, \\phi, \\psi)` of the
        crystal in deg, or None without rotation."""
        if self._euler_angles is None:
            return None
        return (np.rad2deg(self._euler_angles) * u.deg).to("deg")

    @euler_angles.setter
    def euler_angles(self, value):
        self._euler_angles = self._check_euler_angles(value)
        self._notify()

    @property
    def is_isotropic(self):
        """bool: true if the non-magnetic dispersion is isotropic, as well as
        its photoelastic change ``deps_dstrain`` for the strain model
        ``photoelastic``."""
        if self._eps0 is None or not self._eps0.scalar:
            return False
        if self._strain_model != "photoelastic":
            return True
        deps = self._deps_dstrain
        if isinstance(deps, Dispersion):
            return deps.scalar
        return deps.ndim == 0 or np.array_equal(deps, deps[0, 0] * np.eye(3))

    # ------------------------------------------------------------------------
    # calculations
    # ------------------------------------------------------------------------

    def get_tensor(self, energy, strain=0.0, magnetization=None, density=None, polar=False):
        """get_tensor

        Dielectric tensor for the photon energies, strain, and magnetization.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, ndarray[float], optional): strain :math:`\\eta`,
                or an array of strains of shape S, e.g. of a layer for all
                delays. Defaults to 0.
            magnetization (tuple[float, Quantity], VectorParameter, optional):
                cartesian magnetization :math:`(m_x, m_y, m_z)`, or for
                ``polar=True`` its amplitude and the angles :math:`\\phi` and
                :math:`\\gamma` as angle Quantities, e.g.
                ``(1, 90 * u.deg, 0 * u.deg)``, which couples to the sum of all
                gyrotropic dispersions. A list of the magnetizations of all
                sublattices couples to the gyrotropic dispersion of every
                sublattice. Defaults to none.
            density (float, optional): mass density of the layer (kg/m³),
                which is required by dispersions from atoms.
            polar (bool, optional): the magnetization is given in polar
                coordinates. Defaults to False.

        Returns:
            eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3), or
            (S, M, 3, 3) for an array of strains.

        """
        if self._eps0 is None:
            raise ValueError(
                "The dielectric tensor is not set. Set it by a Dispersion, e.g. "
                "layer.optical.dielectric = Dispersion.from_ref_index(n), or from "
                "atoms by layer.optical.dielectric = Dielectric.from_atoms(atom)!"
            )
        # the dispersions get the input of the user for their error messages
        spectral = energy if isinstance(energy, u.Quantity) else None
        energy = np.atleast_1d(to_energy(energy))
        if spectral is None:
            spectral = energy
        strain = np.asarray(strain, dtype=float)
        if strain.ndim == 0:
            strain = float(strain)
            strain_t = strain
        else:
            # strains of shape S for the tensors of shape (S, M, 3, 3)
            strain_t = strain[..., np.newaxis, np.newaxis, np.newaxis]
        # the susceptibilities scale with the density, while a measured, total
        # change per strain of the non-magnetic part already includes it
        scale = 1 / (1 + strain_t) if self._strain_model != "none" else 1.0
        identity = np.eye(3)

        eps0 = _to_tensor(self._eps0.evaluate(spectral, density, _VACUUM_UNITY))
        if self._euler_angles is not None:
            # from the frame of the crystal to the frame of the sample
            R = euler_matrix(*self._euler_angles)
            eps0 = np.einsum("ij,mjk,lk->mil", R, eps0, R)
        # the unstrained eps_xx of e.g. a Voigt constant, whose gyrotropic
        # part scales with the density below only once
        eps_xx = eps0[..., 0, 0]
        if self._strain_model == "density":
            eps0 = identity + scale * (eps0 - identity)
        elif self._strain_model == "photoelastic":
            deps = self._deps_dstrain
            if isinstance(deps, Dispersion):
                deps = _to_tensor(deps.evaluate(spectral, density, _VACUUM_ZERO))
            else:
                deps = _to_tensor(np.broadcast_to(deps, energy.shape + deps.shape).copy())
            eps0 = eps0 + deps * strain_t
        if np.ndim(strain) > 0:
            eps0 = np.broadcast_to(eps0, np.shape(strain) + eps0.shape[-3:])

        magnetization = _cartesian_magnetizations(magnetization, polar)
        parts = self._eps_mag_parts()
        if not parts or not np.any(magnetization):
            return gyrotropic_tensor(eps0, None, np.zeros(3))
        scale_mag = scale if np.ndim(scale) == 0 else scale[..., 0, 0]

        def evaluate(dispersion):
            return dispersion.evaluate(spectral, density, _VACUUM_ZERO, eps_xx=eps_xx) * scale_mag

        if magnetization.ndim == 1:
            # the sum of all gyrotropic dispersions couples to a single magnetization
            eps_mag = sum(evaluate(part) for part in parts if part is not None)
            return gyrotropic_tensor(eps0, eps_mag, magnetization)
        if len(parts) != len(magnetization):
            raise ValueError(
                f"The magnetizations of {len(magnetization):d} sublattices require one "
                f"gyrotropic dispersion per sublattice in eps_mag, not {len(parts):d}!"
            )
        eps = gyrotropic_tensor(eps0, None, np.zeros(3))
        for part, m in zip(parts, magnetization):
            if part is not None and np.any(m):
                eps = gyrotropic_tensor(eps, evaluate(part), m)
        return eps

    def get_ref_index(self, energy, strain=0.0, density=None):
        """get_ref_index

        Complex refractive index :math:`n = \\sqrt{\\varepsilon_{xx}}` of an
        isotropic dielectric without its magnetic part. It is calculated from
        the scalar dielectric function only, i.e. without the dielectric
        tensor, e.g. for many layers and strains of :class:`.Parratt`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, ndarray[float], optional): strain :math:`\\eta`,
                or an array of strains of shape S, e.g. of a layer for all
                delays. Defaults to 0.
            density (float, optional): mass density of the layer (kg/m³).

        Returns:
            n (ndarray[complex]): refractive index of shape (M,), or (S, M)
            for an array of strains.

        """
        if self.is_set and not self.is_isotropic:
            raise ValueError(
                "The dielectric tensor is anisotropic and has no single "
                "refractive index. Use get_tensor() instead!"
            )
        if self._eps0 is None:
            # raises the error of the unset dielectric tensor
            self.get_tensor(energy)
        spectral = energy if isinstance(energy, u.Quantity) else None
        energy = np.atleast_1d(to_energy(energy))
        if spectral is None:
            spectral = energy
        strain = np.asarray(strain, dtype=float)[..., np.newaxis]
        # the scalar eps0 is invariant under the rotation by the Euler angles
        eps = self._eps0.evaluate(spectral, density, _VACUUM_UNITY)
        if self._strain_model == "density":
            eps = 1 + (eps - 1) / (1 + strain)
        elif self._strain_model == "photoelastic":
            deps = self._deps_dstrain
            if isinstance(deps, Dispersion):
                deps = deps.evaluate(spectral, density, _VACUUM_ZERO)
            else:
                deps = np.broadcast_to(deps, energy.shape + np.shape(deps))
            if deps.ndim > 1:
                deps = deps[..., 0, 0]
            eps = eps + deps * strain
        else:
            eps = np.broadcast_to(eps, strain.shape[:-1] + eps.shape)
        return np.sqrt(eps)

    def get_opt_pen_depth(self, energy, strain=0.0, density=None):
        r"""get_opt_pen_depth

        Optical penetration depth of the intensity of an isotropic dielectric function

        .. math:: \zeta = \frac{\lambda}{4 \pi |\kappa|}

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain :math:`\eta`. Defaults to 0.
            density (float, optional): mass density of the layer (kg/m³).

        Returns:
            zeta (Quantity): penetration depth.

        """
        kappa = np.abs(np.imag(self.get_ref_index(energy, strain, density)))
        energy = np.atleast_1d(to_energy(energy))
        wavelength = (energy * u.eV).to("m", "sp").magnitude
        with np.errstate(divide="ignore"):
            zeta = np.where(kappa == 0, np.inf, wavelength / (4 * np.pi * kappa))
        return zeta * u.m

    def get_delta_beta(self, energy, strain=0.0, density=None):
        r"""get_delta_beta

        Real and imaginary part of the refractive index
        :math:`n = 1 - \delta + i\beta` of an isotropic dielectric function, as usual
        for X-rays.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain :math:`\eta`. Defaults to 0.
            density (float, optional): mass density of the layer (kg/m³).

        Returns:
            (tuple):
            - *delta (ndarray[float])* - decrement of the real part.
            - *beta (ndarray[float])* - imaginary part.

        """
        n = self.get_ref_index(energy, strain, density)
        return 1 - n.real, n.imag

    def get_delta_beta_mag(
        self, energy, strain=0.0, density=None, magnetization=None, polar=False
    ):
        r"""get_delta_beta_mag

        Magnetic optical constants :math:`\Delta\delta` and :math:`\Delta\beta`
        of the gyrotropic part per unit magnetization, as usual for X-rays,
        which are defined by

        .. math:: \varepsilon^m = 2 (\Delta\delta - i\Delta\beta)

        and thus proportional to the magnetic form factors :math:`m_1` and
        :math:`m_2` for any refractive index. The refractive indices of the two
        circular modes along the magnetization follow from
        :math:`n_\pm^2 = \varepsilon^0 \mp \varepsilon^m`, which is only for
        :math:`n \approx 1` approximately
        :math:`n_\pm = 1 - (\delta \pm \Delta\delta) + i (\beta \pm \Delta\beta)`.
        Without magnetic part, both are zero.

        For a given ``magnetization``, they are the magnetic optical constants
        of the circular modes along :math:`z` of this magnetization, i.e.
        :math:`\varepsilon_{xy} = -2i (\Delta\delta - i\Delta\beta)`, e.g. the
        sum :math:`\sum_a (\Delta\delta_a - i\Delta\beta_a) m_{a,z}` over the
        sublattices for the ``gyrotropy`` ``"sublattices"``. Several gyrotropic
        dispersions with this gyrotropy require the magnetizations of the
        sublattices, as the sum of the dispersions per unit magnetization
        assumes parallel sublattices.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain :math:`\eta`. Defaults to 0.
            density (float, optional): mass density of the layer (kg/m³).
            magnetization (tuple[float, Quantity], VectorParameter, optional):
                cartesian magnetization or a list of the magnetizations of all
                sublattices, see :meth:`get_tensor`. Defaults to the unit
                magnetization along :math:`z`.
            polar (bool, optional): the magnetization is given in polar
                coordinates. Defaults to False.

        Returns:
            (tuple):
            - *delta_mag (ndarray[float])* - magnetic decrement of the real part.
            - *beta_mag (ndarray[float])* - magnetic imaginary part.

        """
        if self._eps_mag is None:
            zeros = np.zeros(np.shape(np.atleast_1d(to_energy(energy))))
            return zeros, zeros.copy()
        if magnetization is None:
            if len(self._eps_mag_parts()) > 1 and self._gyrotropy != "net":
                raise ValueError(
                    f"The {len(self._eps_mag_parts()):d} gyrotropic dispersions of the "
                    "sublattices require their magnetizations, e.g. "
                    "magnetization=[(0, 0, 1), (0, 0, -1)], or the gyrotropy 'net' for the "
                    "magnetic optical constants per unit net magnetization!"
                )
            magnetization, polar = (0, 0, 1), False
        # the magnetic part of eps_xy = -i eps_mag for the magnetization along z
        eps = self.get_tensor(
            energy, strain, magnetization=magnetization, density=density, polar=polar
        )
        eps -= self.get_tensor(energy, strain, magnetization=(0, 0, 0), density=density)
        eps_mag = 1j * eps[:, 0, 1]
        return eps_mag.real / 2, -eps_mag.imag / 2

    def get_absorption_coefficient(self, energy, strain=0.0, density=None):
        r"""get_absorption_coefficient

        Attenuation coefficient of the intensity of an isotropic dielectric function,
        which is the inverse of the optical penetration depth or absorption
        length :math:`\zeta`:

        .. math:: \mu = \frac{1}{\zeta} = \frac{4 \pi \beta}{\lambda}

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain :math:`\eta`. Defaults to 0.
            density (float, optional): mass density of the layer (kg/m³).

        Returns:
            mu (Quantity): attenuation coefficient.

        """
        return (1 / self.get_opt_pen_depth(energy, strain, density)).to("1/um")

    def get_mass_attenuation_coefficient(self, energy, density, strain=0.0):
        r"""get_mass_attenuation_coefficient

        Mass attenuation coefficient :math:`\mu/\rho` of an isotropic
        dielectric for the density :math:`\rho` of its layer, as tabulated
        e.g. by `NIST <https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients>`_.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            density (float): mass density of the layer (kg/m³).
            strain (float, optional): strain :math:`\eta`. Defaults to 0.

        Returns:
            mu_rho (Quantity): mass attenuation coefficient.

        """
        mu = self.get_absorption_coefficient(energy, strain, density)
        return (mu / (density * u.kg / u.m**3)).to("cm**2/g")

    def get_critical_angle(self, energy, strain=0.0, density=None):
        r"""get_critical_angle

        Critical angle of the total external reflection of an isotropic
        dielectric with :math:`n' < 1` from vacuum for grazing incidence,

        .. math:: \cos(\theta_c) = n' \quad \Rightarrow \quad
            \theta_c \approx \sqrt{2 \delta}

        which is zero for :math:`n' \geq 1`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain :math:`\eta`. Defaults to 0.
            density (float, optional): mass density of the layer (kg/m³).

        Returns:
            theta_c (Quantity): critical angle.

        """
        n_real = self.get_ref_index(energy, strain, density).real
        return (np.arccos(np.clip(n_real, None, 1)) * u.rad).to("deg")

    # ------------------------------------------------------------------------
    # representation
    # ------------------------------------------------------------------------

    def _describe(self):
        if self._eps0 is None:
            return "not set"
        text = f"eps0={self._eps0!r}"
        if self._eps_mag is not None:
            text += f", eps_mag={self._eps_mag!r}"
        if self._gyrotropy is not None:
            text += f", gyrotropy={self._gyrotropy}"
        if self._euler_angles is not None:
            text += f", euler_angles={list(self._euler_angles)}"
        if self._strain_model != "none":
            text += f", strain_model={self._strain_model}"
        if self._strain_model == "photoelastic":
            deps = self._deps_dstrain
            deps = deps if isinstance(deps, Dispersion) else np.asarray(deps).tolist()
            text += f", deps_dstrain={deps}"
        return text

    def __repr__(self):
        return f"Dielectric({self.name}: {self._describe()})"

    @property
    def display_energy(self):
        """float, Quantity: photon energy, where floats are taken in eV, or
        other spectral Quantity, e.g. the wavelength, for which the values of
        the dielectric tensor are displayed, e.g. by ``print()`` or ``display()``.
        Without, the dispersions are described. It does not change any
        calculation."""
        return self._display_energy

    @display_energy.setter
    def display_energy(self, value):
        if value is not None:
            to_energy(value)  # raises for wrong units
        self._display_energy = value

    def _display_density(self):
        """Density of the layer of the dielectric tensor (kg/m³) for its display."""
        owner = getattr(self._caller, "_owner", None)
        try:
            return owner.structural.density.magnitude
        except AttributeError:
            return None

    @staticmethod
    def _format_text(text, mode):
        """Format a description in the mode `text`, `html`, or `latex`."""
        if mode == "html":
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if mode == "latex":
            for char in "\\{}_^%&#$":
                text = text.replace(char, "\\" + char)
            return r"\text{" + text + "}"
        return text

    def _strain_entries(self):
        """Symbols and descriptions of the strain dependence for the display."""
        scaled = "ε⁰ and εᵐ" if self._eps_mag is not None else "ε⁰"
        if self._strain_model == "density":
            return [(r"s(\eta)", f"1/(1+η) for {scaled}")]
        deps = self._deps_dstrain
        if isinstance(deps, Dispersion):
            deps = deps._label()
        else:
            deps = np.asarray(deps)
            if deps.ndim == 2 and np.allclose(deps, np.diag(np.diagonal(deps))):
                deps = np.diagonal(deps)
            deps = _format_values(deps)
        entries = [(r"\mathrm{d}\varepsilon/\mathrm{d}\eta", f"{deps} for ε⁰")]
        if self._eps_mag is not None:
            entries.append((r"s(\eta)", "1/(1+η) for εᵐ"))
        return entries

    def _not_defined(self, dispersion, error):
        """Short text of an error of a dispersion at the display energy."""
        if "density" in str(error):
            return "requires the density of a layer"
        return f"not defined, only {_range_text(dispersion.energy_ranges, self._display_energy)}"

    def _value_entries(self):
        """Symbols and values of the dielectric tensor at the display energy."""
        energy = self._display_energy
        density = self._display_density()
        try:
            eps0 = self.get_tensor(energy, magnetization=(0, 0, 0), density=density)[0]
        except ValueError as e:
            return [(r"\varepsilon^0", self._not_defined(self._eps0, e))]
        diagonal = np.diagonal(eps0)
        entries = []
        if np.allclose(eps0, diagonal[0] * np.eye(3)):
            entries.append((r"\varepsilon^0", diagonal[0]))
            entries.append(("n", np.sqrt(diagonal[0])))
        elif np.allclose(eps0, np.diag(diagonal)):
            for axis, value in zip(("xx", "yy", "zz"), diagonal):
                entries.append((r"\varepsilon^0_{" + axis + "}", value))
        else:
            for i, row in enumerate("xyz"):
                for j, column in enumerate("xyz"):
                    if eps0[i, j] != 0:
                        entries.append((r"\varepsilon^0_{" + row + column + "}", eps0[i, j]))
        for symbol, part in self._eps_mag_symbols():
            try:
                eps_mag = part.evaluate(
                    energy, density, _VACUUM_ZERO, eps_xx=np.array([eps0[0, 0]])
                )[0]
                entries.append((symbol, eps_mag))
            except ValueError as e:
                entries.append((symbol, self._not_defined(part, e)))
        return entries

    def _eps_mag_symbols(self):
        """symbols and gyrotropic dispersions, numbered by their sublattice"""
        if not isinstance(self._eps_mag, list):
            parts = self._eps_mag_parts()
            return [(r"\varepsilon^\mathrm{m}", part) for part in parts]
        return [
            (r"\varepsilon^\mathrm{m}_{" + f"{a + 1:d}" + "}", part)
            for a, part in enumerate(self._eps_mag)
            if part is not None
        ]

    def _format_symbol(self, symbol, mode):
        """Subscripts without unicode characters, e.g. of yy, are kept as
        plain text."""
        text = super()._format_symbol(symbol, mode)
        if mode == "text":
            text = text.replace("^m", "ᵐ").replace("_{", "_").replace("}", "")
        return text

    def _relation(self, index):
        """The dispersions are only described without a display energy."""
        if self._eps0 is None or self._display_energy is not None:
            return "="
        return ":" if index < 1 + len(self._eps_mag_symbols()) else "="

    def _entries(self, mode):
        if self._eps0 is None:
            entries = [(self._latex_symbol(), "", "not set")]
        elif self._display_energy is None:
            entries = [(r"\varepsilon^0", "", self._eps0._label())]
            for symbol, part in self._eps_mag_symbols():
                entries.append((symbol, "", part._label()))
        else:
            energy = self._display_energy
            energy = f"{energy:~P}" if isinstance(energy, u.Quantity) else f"{energy:g} eV"
            if mode == "latex":
                energy = r"\text{" + energy + "}"
            entries = [(symbol, f"({energy})", value) for symbol, value in self._value_entries()]
        if self._eps0 is not None and self._gyrotropy is not None:
            coupling = {
                "net": "net magnetization of all sublattices",
                "sublattices": "magnetization of every sublattice",
            }[self._gyrotropy]
            entries.append((r"\mathbf{m}", "", f"{coupling} (gyrotropy {self._gyrotropy})"))
        if self._eps0 is not None and self._euler_angles is not None:
            angles = ", ".join(f"{np.rad2deg(a):.4g}°" for a in self._euler_angles)
            entries.append(("R", "", f"Euler (z-x-z): ({angles})"))
        if self._eps0 is not None and self._strain_model != "none":
            entries += [(symbol, "", value) for symbol, value in self._strain_entries()]
        return [
            (
                self._format_symbol(symbol, mode),
                arguments,
                self._format_text(value, mode)
                if isinstance(value, str)
                else _format_value(
                    value, mode, near_one=not symbol.startswith(r"\varepsilon^\mathrm{m}")
                ),
                "",
            )
            for symbol, arguments, value in entries
        ]

    def _table_rows(self, mode):
        """Rows of the table view, where the display energy is given with the
        symbols."""
        return [
            (symbol + arguments, value, unit)
            for symbol, arguments, value, unit in self._entries(mode)
        ]
