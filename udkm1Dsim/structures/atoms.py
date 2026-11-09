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

__all__ = ["Atom", "AtomMixed"]

__docformat__ = "restructuredtext"

import os
import warnings

import numpy as np
import pint
from tabulate import tabulate

from ..helpers import SpectralTable, _format_energy, read_columns, to_energy, undefined_error
from .dielectric import atoms_susceptibility
from .parameter_groups import AtomMixedParameters, AtomParameters

u = pint.get_application_registry()
Q_ = u.Quantity

# invalid values in the tables of Henke et al.
_INVALID_FORM_FACTOR = -9000


def form_factor_table(coeff, source=""):
    """form_factor_table

    Table of the complex form factors :math:`f_1 - i f_2` from coefficients
    with the columns of the photon energy (eV), :math:`f_1`, and :math:`f_2`,
    e.g. of ``.cf``, ``.nff``, or ``.mf`` files, where invalid entries, e.g.
    the -9999 of the tables of Henke et al., are NaN.

    Args:
        coeff (ndarray[float]): coefficients of shape (N, 3).
        source (str, optional): source of the table.

    Returns:
        table (SpectralTable): table of the form factors.

    """
    return SpectralTable.from_columns(
        coeff, imaginary_sign=-1, invalid_below=_INVALID_FORM_FACTOR, source=source
    )


class Atom:
    """Atom

    Smallest structural unit of which larger structures can be build.

    It holds real physical properties of on the atomic level. Its numeric
    properties are organized as :class:`.AtomParameters` in the `atomic`
    parameter group, whose attributes are the :class:`.Parameter` objects
    with their name, symbol, value, and unit, e.g. `atom.atomic.mass`. In
    contrast, the shortcuts of the atom, e.g. `atom.mass`, return only their
    values for calculations, i.e. a pint Quantity or a number.

    Note:
        The properties of an atom are read by a layer when the atom is added
        to it, e.g. its mass for the density of a unit cell. Hence, changing
        an atom afterwards does not update the layers it was already added to.

    .. deprecated:: 3.0.0
        The polar shortcuts ``mag_amplitude``, ``mag_phi``, and ``mag_gamma``
        were removed. Use ``magnetization={"r": ..., "phi": ..., "gamma": ...}``
        on initialization and ``atom.atomic.magnetization`` afterwards.

    Args:
        symbol (str): symbol of the atom.

    Keyword Args:
        id (str): id of the atom, may differ from symbol and/or name.
        ionicity (int): ionicity of the atom.
        atomic_form_factor_path (str): path to atomic form factor coeffs.
        atomic_form_factor_source (str): either _henke_ or default _chantler_
        magnetic_form_factor_path (str): path to magnetic form factor coeffs.
        magnetization (tuple, dict, VectorParameter): magnetization as
            cartesian components, e.g. ``(0, 0, 1)``, or as dict of polar or
            cartesian components, e.g. ``{"r": 1, "phi": 90 * u.deg,
            "gamma": 0 * u.deg}`` or ``{"x": 0, "y": 0, "z": 1}``, see
            :class:`.VectorParameter`.

    Attributes:
        symbol (str): symbol of the element.
        id (str): id of the atom, may differ from symbol and/or name.
        name (str): name of the element (generic).
        atomic (AtomParameters): parameters of the atom.
        magnetization (Quantity): cartesian magnetization as for a layer,
            which can be set as on initialization.
        atomic_number_z (int): Z atomic number.
        mass_number_a (float): A atomic mass number.
        ionicity (int): ionicity of the atom.
        mass (float): mass of the atom (kg).
        atomic_form_factor_coeff (ndarray[float]): atomic form factor.
           coefficients for energy-dependent atomic form factor.
        cromer_mann_coeff (ndarray[float]): cromer-mann coefficients for
           angular-dependent atomic form factor.
        magnetic_form_factor_coeff (ndarray[float]): magnetic form factor
           coefficients for energy-dependent magnetic form factor.

    """

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.id = kwargs.get("id", symbol)

        try:
            filename = os.path.join(os.path.dirname(__file__), "../parameters/elements.dat")
            symbols = np.genfromtxt(filename, dtype="U2", usecols=(0))
            elements = np.genfromtxt(filename, dtype="U15, i8, f8", usecols=(1, 2, 3))
            [rowidx] = np.where(symbols == self.symbol)
            element = elements[rowidx[0]]
        except Exception as e:
            print("Cannot load element specific data from elements data file!")
            print(e)

        self.name = element[0]
        self.atomic = AtomParameters(
            atomic_number=int(element[1]),
            mass_number=element[2],
            ionicity=kwargs.get("ionicity", 0),
        )
        self._init_magnetization(kwargs)
        atomic_path = kwargs.get("atomic_form_factor_path", "")
        source = kwargs.get("atomic_form_factor_source", "chantler")
        self.atomic_form_factor_coeff = self.read_atomic_form_factor_coeff(
            filename=atomic_path, source=source
        )
        self._atomic_form_factor_source = (
            os.path.basename(atomic_path) if atomic_path else source.capitalize()
        )
        magnetic_path = kwargs.get("magnetic_form_factor_path", "")
        self.magnetic_form_factor_coeff = self.read_magnetic_form_factor_coeff(
            filename=magnetic_path
        )
        self._magnetic_form_factor_source = (
            os.path.basename(magnetic_path) if magnetic_path else f"{self.symbol}.mf"
        )
        self._read_cromer_mann_coeff()
        # changes of the parameters, e.g. of the ionicity, are notified to the atom
        self.atomic._owner = self

    def _update_depending(self):
        """Called on changes of the atomic parameters to re-read the
        Cromer-Mann coefficients on a change of the ionicity."""
        if self.ionicity != getattr(self, "_cromer_mann_ionicity", None):
            self._read_cromer_mann_coeff()

    def _read_cromer_mann_coeff(self):
        """Read the Cromer-Mann coefficients for the current ionicity."""
        self._cromer_mann_ionicity = self.ionicity
        self.cromer_mann_coeff = self.read_cromer_mann_coeff()

    @property
    def cromer_mann_coeff(self):
        """ndarray[float]: Cromer-Mann coefficients as read from the parameter
        file, including the atomic number and ionicity, see
        :meth:`read_cromer_mann_coeff`."""
        p = self.atomic
        return np.concatenate(
            [
                self._cromer_mann_keys,
                p.cromer_mann_a.magnitude,
                p.cromer_mann_b.magnitude,
                [p.cromer_mann_c.magnitude],
            ]
        )

    @cromer_mann_coeff.setter
    def cromer_mann_coeff(self, cm):
        cm = np.asarray(cm, dtype=float)
        # the atomic number and ionicity, for which the coefficients were read
        self._cromer_mann_keys = cm[0:2]
        self.atomic.cromer_mann_a.magnitude = cm[2:6]
        self.atomic.cromer_mann_b.magnitude = cm[6:10]
        self.atomic.cromer_mann_c.magnitude = float(cm[10])

    def _header(self):
        """Header with the class, name, id, and symbol of the atom"""
        return (
            f"{type(self).__name__}: {self.name}\nID: {self.id}\nSymbol: {self.symbol}\n"
            + "=" * 30
            + "\n"
        )

    def _form_factor_lines(self):
        """Sources and energy ranges of the atomic and magnetic form factors."""
        lines = []
        for kind, magnetic in (("Atomic", False), ("Magnetic", True)):
            source = self.form_factor_source(magnetic)
            if source is None:
                lines.append(f"{kind} form factors: not tabulated")
                continue
            lo, hi = self.magnetic_form_factor_range if magnetic else self.atomic_form_factor_range
            lines.append(
                f"{kind} form factors: {source} from {_format_energy(lo)} to {_format_energy(hi)}"
            )
        return lines

    def __repr__(self):
        """String representation of this class"""
        return (
            self._header()
            + "\n".join(self._form_factor_lines())
            + "\n"
            + self.atomic._table_representation()
        )

    def _repr_html_(self):
        """HTML representation of this class"""
        return (
            f"<h3>{type(self).__name__}: {self.name}</h3>"
            f"<b>ID:</b> <i>{self.id}</i>, <b>Symbol:</b> <i>{self.symbol}</i><br>"
            + "".join(f"{line}<br>" for line in self._form_factor_lines())
            + self.atomic._html_table()
        )

    # ============================================================================
    # form factors
    # ============================================================================

    @property
    def atomic_form_factor_coeff(self):
        """ndarray[float]: tabulated atomic form factors with the columns of
        the photon energy (eV), :math:`f_1`, and :math:`f_2`. Setting it,
        e.g. to experimental data, changes their source to ``custom``."""
        return self._atomic_form_factor_coeff

    @atomic_form_factor_coeff.setter
    def atomic_form_factor_coeff(self, coeff):
        self._atomic_form_factor_coeff = coeff
        self._atomic_form_factor_source = "custom"

    @property
    def magnetic_form_factor_coeff(self):
        """ndarray[float]: tabulated magnetic form factors with the columns of
        the photon energy (eV), :math:`m_1`, and :math:`m_2`, or None if not
        tabulated. Setting it, e.g. to experimental data, changes their source
        to ``custom``."""
        return self._magnetic_form_factor_coeff

    @magnetic_form_factor_coeff.setter
    def magnetic_form_factor_coeff(self, coeff):
        self._magnetic_form_factor_coeff = coeff
        self._magnetic_form_factor_source = "custom"

    def form_factor_source(self, magnetic=False):
        """form_factor_source

        Source of the atomic or magnetic form factors, e.g. ``Chantler``,
        ``Henke``, the name of the file, or ``custom``.

        Args:
            magnetic (bool, optional): magnetic instead of atomic form factors.
                Defaults to False.

        Returns:
            source (str, None): source, or None if not tabulated.

        """
        if magnetic:
            if self.magnetic_form_factor_coeff is None:
                return None
            return self._magnetic_form_factor_source
        return self._atomic_form_factor_source

    @property
    def atomic_form_factor_range(self):
        """tuple[float]: lowest and highest photon energy (eV) of the tabulated
        atomic form factors."""
        return self._cached_form_factor_table(False).energy_range

    @property
    def magnetic_form_factor_range(self):
        """tuple[float]: lowest and highest photon energy (eV) of the tabulated
        magnetic form factors, which are zero at all energies if not
        tabulated."""
        if self.magnetic_form_factor_coeff is None:
            return -np.inf, np.inf
        return self._cached_form_factor_table(True).energy_range

    @property
    def has_magnetic_form_factor(self):
        """bool: true if magnetic form factors are tabulated."""
        coeff = self.magnetic_form_factor_coeff
        return coeff is not None and bool(np.any(np.atleast_2d(coeff)[:, 1:]))

    def _cached_form_factor_table(self, magnetic):
        """Table of the atomic or magnetic form factors, which is only built
        again if their coefficients have changed."""
        coeff = self.magnetic_form_factor_coeff if magnetic else self.atomic_form_factor_coeff
        tables = self.__dict__.setdefault("_form_factor_tables", {})
        cached = tables.get(magnetic)
        if cached is None or not np.array_equal(cached[0], coeff, equal_nan=True):
            cached = (np.array(coeff, dtype=float), form_factor_table(coeff))
            tables[magnetic] = cached
        return cached[1]

    def _get_form_factor(self, energy, magnetic):
        """Interpolated atomic or magnetic form factors for photon energies,
        wavelengths, frequencies, or wavenumbers, which raise an error outside
        of the tabulated energy range in the quantity and unit of the input."""
        spectral = energy
        energy = np.asarray(to_energy(energy), dtype=float)
        shape = energy.shape
        energy = energy.reshape(-1)
        coeff = self.magnetic_form_factor_coeff if magnetic else self.atomic_form_factor_coeff
        if coeff is None:
            # no magnetic form factors are tabulated
            f = np.zeros(energy.shape, dtype=complex)
        else:
            f = self._cached_form_factor_table(magnetic)(energy)
            invalid = np.isnan(f.real) | np.isnan(f.imag)
            if np.any(invalid):
                kind = "magnetic" if magnetic else "atomic"
                energy_range = (
                    self.magnetic_form_factor_range if magnetic else self.atomic_form_factor_range
                )
                raise undefined_error(
                    f"The {kind} form factor of {self.id} ({self.form_factor_source(magnetic)})",
                    energy,
                    invalid,
                    [energy_range],
                    spectral,
                )
        return f.reshape(shape) if shape else f[0]

    def read_atomic_form_factor_coeff(self, source="chantler", filename=""):
        """read_atomic_form_factor_coeff

        The coefficients for the atomic form factor :math:`f` in dependence of
        the photon energy :math:`E` is read from a parameter file given by :cite:t:`henke1993`
        or by :cite:t:`chantler2003` as default.

        Args:
            source (str, optional): source of atmoic form factors can be either
                _henke_ or _chantler_. Defaults to _chantler_.
            filename (str, optional): full path and filename to the atomic form
                factor coefficients.

        Returns:
            f (ndarray[float]): atomic form factor coefficients.

        """
        if not filename:
            if source not in ["chantler", "henke"]:
                raise ValueError(
                    "The source of the atomic form factors must be either chantler or henke!"
                )

            if source == "chantler":
                sub_path = f"chantler/{self.symbol.lower():s}.cf"
            elif source == "henke":
                sub_path = f"henke/{self.symbol.lower():s}.nff"

            filename = os.path.join(
                os.path.dirname(__file__), f"../parameters/atomic_form_factors/{sub_path:s}"
            )
        return read_columns(filename)

    def get_atomic_form_factor(self, energy):
        """get_atomic_form_factor

        The complex atomic form factor for the photon energy :math:`E` is
        linearly interpolated from its table:

        .. math:: f(E)=f_1 - i f_2

        Convention of Ref. :cite:t:`alsnielsen2011` (p. 11, footnote) is a negative :math:`f_2`.

        Outside of the tabulated energy range, a ``ValueError`` is raised
        instead of an extrapolation.

        Args:
            energy (float, ndarray[float], Quantity): photon energy, where
                floats are taken in eV, or other spectral Quantities, e.g. the
                wavelength.

        Returns:
            f (complex, ndarray[complex]): energy-dependent atomic form factors.

        """
        return self._get_form_factor(energy, magnetic=False)

    def read_cromer_mann_coeff(self):
        r"""read_cromer_mann_coeff

        The Cromer-Mann coefficients (see :cite:t:`cromermann1968`) of the
        atomic number :math:`Z` and ionicity :math:`I` of the atom are read
        from the parameter file, see its README for the source, which is
        Table 6.1.1.4 of the International Tables for Crystallography
        :cite:`brown2006`. They are returned in the following order:

        .. math:: Z\; I\; a_1\; a_2\; a_3\; a_4\; b_1\; b_2\; b_3\; b_4\; c

        where the :math:`b_i` are given in Å². For several entries of the same
        :math:`Z` and :math:`I`, the first one is returned. If the ionicity
        is not tabulated for the atomic number, the entry of the closest
        tabulated ionicity is returned with a warning, which is the one
        closer to the neutral atom for two equally close ionicities.

        Returns:
            cm (ndarray[float]): Cromer-Mann coefficients.

        """
        filename = os.path.join(
            os.path.dirname(__file__), "../parameters/atomic_form_factors/cromermann.txt"
        )
        try:
            cm = np.genfromtxt(
                filename, skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
            )
        except FileNotFoundError:
            print(f"Cromer Mann coefficient file {filename:s} not found!")
            raise

        entries = cm[cm[:, 0] == self.atomic_number_z]
        if len(entries) == 0:
            raise ValueError(
                f"No Cromer-Mann coefficients tabulated for {self.symbol} "
                f"(Z = {self.atomic_number_z})!"
            )
        exact = entries[entries[:, 1] == self.ionicity]
        if len(exact) > 0:
            return exact[0]
        # the closest ionicity, but not the special valence entry of I = 99
        ionicities = np.unique(entries[entries[:, 1] != 99, 1])
        closest = min(ionicities, key=lambda i: (abs(i - self.ionicity), abs(i)))
        warnings.warn(
            f"No Cromer-Mann coefficients tabulated for {self.symbol} with an ionicity "
            f"of {self.ionicity}, the ones of the closest ionicity {closest:.0f} are used.",
            stacklevel=2,
        )
        return entries[entries[:, 1] == closest][0]

    @u.wraps(None, (None, "eV", "m**-1"), strict=False)
    def get_cm_atomic_form_factor(self, energy, qz):
        r"""get_cm_atomic_form_factor

        The atomic form factor :math:`f` is calculated in dependence of the
        photon energy :math:`E` and the :math:`z`-component of the scattering
        vector :math:`q_z` (see :cite:t:`cromermann1968`), which are given as
        Quantities or as floats in eV and (1/m), respectively.
        Note that the Cromer-Mann coefficients are fitted for :math:`q_z` in
        [Å :math:`^{-1}`], to which :math:`q_z` is converted internally!

        See Ref. :cite:t:`alsnielsen2011` (p. 235).

        .. math:: f(q_z,E) = f_{CM}(q_z) + \delta f_1(E) -i f_2(E)

        :math:`f_{CM}(q_z)` is given in Ref. :cite:t:`cromermann1968` and in
        Table 6.1.1.4 of Ref. :cite:t:`brown2006` with
        :math:`\sin(\theta)/\lambda = q_z/4\pi` and :math:`b_i` in Å² for
        :math:`\sin(\theta)/\lambda < 2` Å :math:`^{-1}`:

        .. math::

            f_{CM}(q_z) = \sum^4_{i=1} a_i \, \exp\left(-b_i \, (q_z/4\pi)^2\right) + c

        :math:`\delta f_1(E)` is the dispersion correction with respect to
        :math:`f_{CM}(0)`:

        .. math::

            \delta f_1(E) = f_1(E) - f_{CM}(0) = f_1(E) - \left(\sum^4_{i=1} a_i + c\right)

        Thus:

        .. math:: f(q_z,E) = \sum^4_{i=1} a_i \, \exp\left(-b_i \, (q_z/4\pi)^2\right)
           + c + f_1(E) - i f_2(E) - \left(\sum^4_{i=1} a_i + c\right)

        Args:
            energy (Quantity, ndarray[float]): photon energy, where floats
                are taken in eV.
            qz (Quantity, ndarray[float]): scattering vector, where floats
                are taken in 1/m.

        Returns:
            f (ndarray[complex]): energy- and qz-dependent Cromer-Mann atomic form
            factors.

        """
        # convert from 1/m to 1/Å and to a real column vector
        qz = np.array(qz * 1e-10, ndmin=2)
        energy = np.array(energy, ndmin=1)
        if np.size(qz, 0) != len(energy):
            raise TypeError("qz need to have as many rows as energies!")

        f = np.zeros_like(qz, dtype=complex)
        a = self.atomic.cromer_mann_a.magnitude
        b = self.atomic.cromer_mann_b.magnitude
        c = self.atomic.cromer_mann_c.magnitude

        for i, en in enumerate(energy):
            _qz = qz[i, :].reshape(-1, 1)
            f_cm = np.dot(a, np.exp(np.outer(-b, (_qz / (4 * np.pi)) ** 2))) + c
            # f_1(E) - i f_2(E) with the dispersion correction with respect to f_CM(0)
            f[i, :] = f_cm + self.get_atomic_form_factor(en) - (np.sum(a) + c)
        return f

    def read_magnetic_form_factor_coeff(self, filename=""):
        """read_magnetic_form_factor_coeff

        The coefficients for the magnetic form factor :math:`m` in dependence
        of the photon energy :math:`E` is read from a parameter file.

        Args:
            filename (str): optional full path and filename to the magnetic
                form factor coefficients.

        Returns:
            m (ndarray[float], None): magnetic form factor coefficients, or None
            if no file is given and none is tabulated for the element.

        """
        if not filename:
            filename = os.path.join(
                os.path.dirname(__file__),
                f"../parameters/magnetic_form_factors/{self.symbol:s}.mf",
            )
            if not os.path.isfile(filename):
                # no magnetic form factors are tabulated for this element
                return None
        return read_columns(filename)

    def get_magnetic_form_factor(self, energy):
        """get_magnetic_form_factor

        The complex magnetic form factor for the photon energy :math:`E` is
        linearly interpolated from its table:

        .. math:: m(E) = m_1 - i m_2

        Convention of Ref. :cite:t:`alsnielsen2011` (p. 11, footnote) is a negative :math:`m_2`.

        It is zero at all energies, if no magnetic form factors are tabulated,
        and raises a ``ValueError`` outside of the tabulated energy range.

        Args:
            energy (float, ndarray[float], Quantity): photon energy, where
                floats are taken in eV, or other spectral Quantities, e.g. the
                wavelength.

        Returns:
            m (complex, ndarray[complex]): energy-dependent magnetic form factors.

        """
        return self._get_form_factor(energy, magnetic=True)

    @u.wraps(None, (None, None, "kg/m**3"), strict=False)
    def get_susceptibility(self, energy, density):
        r"""get_susceptibility

        Non-magnetic and gyrotropic susceptibility :math:`\chi^0` and
        :math:`\chi^m` of the atom for the photon energy :math:`E` (eV) and
        the mass density :math:`\rho` around it following :cite:t:`elzo2012`:

        .. math::

            \chi^0 = -\frac{r_e \lambda^2}{\pi} n_a f(E), \quad
            \chi^m = \frac{r_e \lambda^2}{\pi} n_a m(E)

        with the number density :math:`n_a = \rho / m_a` of the atoms and
        their atomic and magnetic form factors :math:`f` and :math:`m`, see
        :class:`.Dielectric`.

        Args:
            energy (float, ndarray[float], Quantity): photon energy, where
                floats are taken in eV, or other spectral Quantities.
            density (Quantity, float): mass density, where floats are taken
                in kg/m³.

        Returns:
            (tuple):
            - *chi0 (ndarray[complex])* - non-magnetic susceptibility.
            - *chim (ndarray[complex])* - gyrotropic susceptibility.

        """
        energy = np.atleast_1d(to_energy(energy))
        chi0 = atoms_susceptibility([(self, 1)], energy, density)
        chim = atoms_susceptibility([(self, 1)], energy, density, magnetic=True)
        return chi0, chim

    # ============================================================================
    # properties of the atomic parameters
    # ============================================================================

    @property
    def atomic_number_z(self):
        return self.atomic.atomic_number.magnitude

    @atomic_number_z.setter
    def atomic_number_z(self, value):
        raise AttributeError("The atomic number of an Atom is given by its symbol.")

    @property
    def mass_number_a(self):
        return self.atomic.mass_number.magnitude

    @mass_number_a.setter
    def mass_number_a(self, value):
        raise AttributeError("The mass number of an Atom is given by its symbol.")

    @property
    def mass(self):
        return self.atomic.mass.quantity

    @mass.setter
    def mass(self, value):
        raise AttributeError("'mass' is derived from the mass number and cannot be set directly.")

    @property
    def _mass(self):
        return self.atomic.mass.magnitude

    @property
    def ionicity(self):
        return self.atomic.ionicity.magnitude

    @ionicity.setter
    def ionicity(self, value):
        self.atomic.ionicity.magnitude = value

    def _init_magnetization(self, kwargs):
        """Set the magnetization from the keyword arguments."""
        removed = [key for key in ("mag_amplitude", "mag_phi", "mag_gamma") if key in kwargs]
        if removed:
            raise TypeError(
                f"{', '.join(repr(key) for key in removed)} were removed in version 3.0.0, "
                'use magnetization={"r": ..., "phi": ..., "gamma": ...} instead!'
            )
        self.atomic.magnetization.set(kwargs.get("magnetization", (0.0, 0.0, 0.0)))

    @property
    def magnetization(self):
        return self.atomic.magnetization.quantity

    @magnetization.setter
    def magnetization(self, value):
        self.atomic.magnetization.set(value)


class AtomMixed(Atom):
    """AtomMixed

    Representation of mixed atoms in alloys and stochiometric mixtures.

    All properties of the included sub-atoms of class Atom are averaged
    and weighted with their stochiometric ratio. They are organized as
    :class:`.AtomMixedParameters` in the `atomic` parameter group, where the
    atomic number, mass number, mass, and ionicity are derived from the
    sub-atoms and cannot be set.

    Note:
        The properties of the sub-atoms are read when they are added to the
        mixed atom, and those of the mixed atom when it is added to a layer.
        Hence, changing an atom afterwards does not update the mixed atoms
        and layers it was already added to.

    .. deprecated:: 3.0.0
        The polar shortcuts ``mag_amplitude``, ``mag_phi``, and ``mag_gamma``
        were removed. Use ``magnetization={"r": ..., "phi": ..., "gamma": ...}``
        on initialization and ``atom.atomic.magnetization`` afterwards.

    Args:
        symbol (str): symbol of the atom.

    Keyword Args:
        id (str): id of the atom, may differ from symbol and/or name.
        name (str): name of the mixed atom, default is symbol.
        atomic_form_factor_path (str): path to atomic form factor coeffs.
        magnetic_form_factor_path (str): path to magnetic form factor coeffs.
        magnetization (tuple, dict, VectorParameter): magnetization as
            cartesian components, e.g. ``(0, 0, 1)``, or as dict of polar or
            cartesian components, e.g. ``{"r": 1, "phi": 90 * u.deg,
            "gamma": 0 * u.deg}`` or ``{"x": 0, "y": 0, "z": 1}``, see
            :class:`.VectorParameter`.

    Attributes:
        symbol (str): symbol of the element.
        id (str): id of the atom, may differ from symbol and/or name.
        name (str): name of the mixed atom, default is symbol.
        atomic (AtomMixedParameters): parameters of the mixed atom.
        magnetization (Quantity): cartesian magnetization as for a layer,
            which can be set as on initialization.
        atomic_number_z (int): Z atomic number.
        mass_number_a (float): A atomic mass number.
        ionicity (int): ionicity of the atom.
        mass (float): mass of the atom (kg).
        atomic_form_factor_coeff (ndarray[float]): atomic form factor.
           coefficients for energy-dependent atomic form factor.
        magnetic_form_factor_coeff (ndarray[float]): magnetic form factor
           coefficients for energy-dependent magnetic form factor.

    """

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.id = kwargs.get("id", symbol)
        self.name = kwargs.get("name", symbol)
        self.atomic = AtomMixedParameters()
        self._init_magnetization(kwargs)
        self.atoms = []
        self.num_atoms = 0
        atomic_path = kwargs.get("atomic_form_factor_path", "")
        self.atomic_form_factor_coeff = self.read_atomic_form_factor_coeff(filename=atomic_path)
        self._atomic_form_factor_source = os.path.basename(atomic_path)
        magnetic_path = kwargs.get("magnetic_form_factor_path", "")
        self.magnetic_form_factor_coeff = self.read_magnetic_form_factor_coeff(
            filename=magnetic_path
        )
        self._magnetic_form_factor_source = os.path.basename(magnetic_path)

    def form_factor_source(self, magnetic=False):
        """form_factor_source

        Source of the atomic or magnetic form factors, which are averaged from
        the constituents without own table.

        Args:
            magnetic (bool, optional): magnetic instead of atomic form factors.
                Defaults to False.

        Returns:
            source (str, None): source, or None if not tabulated.

        """
        coeff = self.magnetic_form_factor_coeff if magnetic else self.atomic_form_factor_coeff
        if coeff is not None:
            return (
                self._magnetic_form_factor_source
                if magnetic
                else (self._atomic_form_factor_source)
            )
        if magnetic and not self.has_magnetic_form_factor:
            return None
        return "averaged from the constituents"

    def _constituent_range(self, magnetic):
        """Common energy range (eV) of the form factors of the constituents."""
        lo, hi = -np.inf, np.inf
        for atom, _ in self.atoms:
            a_lo, a_hi = (
                atom.magnetic_form_factor_range if magnetic else (atom.atomic_form_factor_range)
            )
            lo, hi = max(lo, a_lo), min(hi, a_hi)
        return lo, hi

    @property
    def atomic_form_factor_range(self):
        """tuple[float]: lowest and highest photon energy (eV) of the atomic
        form factors, which is the common range of the constituents without
        own table."""
        if self.atomic_form_factor_coeff is None:
            return self._constituent_range(magnetic=False)
        return super().atomic_form_factor_range

    @property
    def magnetic_form_factor_range(self):
        """tuple[float]: lowest and highest photon energy (eV) of the magnetic
        form factors, which is the common range of the constituents without
        own table."""
        if self.magnetic_form_factor_coeff is None:
            return self._constituent_range(magnetic=True)
        return super().magnetic_form_factor_range

    @property
    def has_magnetic_form_factor(self):
        """bool: true if magnetic form factors are tabulated for the mixed atom
        or for any of its constituents."""
        if self.magnetic_form_factor_coeff is None:
            return any(atom.has_magnetic_form_factor for atom, _ in self.atoms)
        return super().has_magnetic_form_factor

    def _constituents_table(self):
        """Name/fraction table of the constituent atoms"""
        return [[atom.name, f"{fraction * 100:.1f} %"] for atom, fraction in self.atoms]

    def __repr__(self):
        """String representation of this class"""
        return (
            self._header()
            + "\n".join(self._form_factor_lines())
            + "\n"
            + self.atomic._table_representation()
            + f"\n{self.num_atoms:d} Constituents\n"
            + tabulate(
                self._constituents_table(), headers=["atom", "fraction"], colalign=("left",)
            )
        )

    def _repr_html_(self):
        """HTML representation of this class"""
        return (
            f"<h3>{type(self).__name__}: {self.name}</h3>"
            f"<b>ID:</b> <i>{self.id}</i>, <b>Symbol:</b> <i>{self.symbol}</i><br>"
            + "".join(f"{line}<br>" for line in self._form_factor_lines())
            + self.atomic._html_table()
            + f"<h4>{self.num_atoms:d} Constituents</h4>"
            + tabulate(
                self._constituents_table(),
                headers=["atom", "fraction"],
                colalign=("left",),
                tablefmt="html",
            )
        )

    def add_atom(self, atom, fraction):
        """add_atom

        Add an Atom instance with its stochiometric fraction and recalculate
        averaged properties.

        Args:
            atom (Atom): atom to add.
            fraction (float): fraction of the atom - sum of all fractions must
                be 1.

        """
        if isinstance(atom, Atom):
            self.atoms.append([atom, fraction])
            self.num_atoms = self.num_atoms + 1
            # calculate the mixed atomic properties of the atomMixed instance
            p = self.atomic
            p.atomic_number.magnitude += fraction * atom.atomic_number_z
            p.mass_number.magnitude += fraction * atom.mass_number_a
            p.mass.magnitude += fraction * atom._mass
            p.ionicity.magnitude += fraction * atom.ionicity
        else:
            warnings.warn("Only Atom objects can be added to a MixedAtom!")

    def read_atomic_form_factor_coeff(self, filename=""):
        """read_atomic_form_factor_coeff

        The coefficients for the atomic form factor :math:`f` in dependence of
        the photon energy :math:`E` must be read from an external file given
        by ``filename``.

        Args:
            filename (str, optional): full path and filename to the atomic form
                factor coefficients.

        Returns:
            f (ndarray[float]): atomic form factor coefficients.

        """
        if not filename:
            return None
        return read_columns(filename)

    def get_atomic_form_factor(self, energy):
        """get_atomic_form_factor

        Averaged energy dependent atomic form factor.
        If ``atomic_form_factor_path`` was given on initialization this file
        will be used instead.

        Args:
            energy (float, ndarray[float], Quantity): photon energy, where
                floats are taken in eV, or other spectral Quantities.

        Returns:
            f (ndarray[complex]): energy-dependent atomic form factors.

        """
        if self.atomic_form_factor_coeff is None:
            # no external file is given
            # calculate average from added atoms
            f = 0
            for i in range(self.num_atoms):
                f += self.atoms[i][0].get_atomic_form_factor(energy) * self.atoms[i][1]
        else:
            f = super().get_atomic_form_factor(energy)

        return f

    @u.wraps(None, (None, "eV", "m**-1"), strict=False)
    def get_cm_atomic_form_factor(self, energy, qz):
        """get_cm_atomic_form_factor

        Averaged energy and qz-dependent atomic form factors.

        Args:
            energy (Quantity, ndarray[float]): photon energy, where floats
                are taken in eV.
            qz (Quantity, ndarray[float]): scattering vector, where floats
                are taken in 1/m.

        Returns:
            f (ndarray[complex]): energy- and qz-dependent Cromer-Mann atomic
            form factors.

        """
        if self.atomic_form_factor_coeff is None:
            # no external file is given
            # calculate average from added atoms
            f = 0
            for i in range(self.num_atoms):
                f += self.atoms[i][0].get_cm_atomic_form_factor(energy, qz) * self.atoms[i][1]
        else:
            warnings.warn(
                "Cromer-Mann correction cannot be applied to "
                "atomic form factors from external files. "
                "Returning uncorrected values instead!"
            )
            f = self.get_atomic_form_factor(energy)

        return f

    def read_magnetic_form_factor_coeff(self, filename=""):
        """read_magnetic_form_factor_coeff

        The coefficients for the magnetic form factor :math:`m` in dependence
        of the photon energy :math:`E` must be read from an external file given
        by ``filename``.

        Args:
            filename (str): optional full path and filename to the magnetic
                form factor coefficients.

        Returns:
            m (ndarray[float]): magnetic form factor coefficients.

        """
        if not filename:
            return None
        return read_columns(filename)

    def get_magnetic_form_factor(self, energy):
        """get_magnetic_form_factor

        Mixed energy dependent magnetic form factors.

        Args:
            energy (float, ndarray[float], Quantity): photon energy, where
                floats are taken in eV, or other spectral Quantities.

        Returns:
            f (ndarray[complex]): energy-dependent magnetic form factors.

        """
        if self.magnetic_form_factor_coeff is None:
            # no external file is given
            # calculate average from added atoms
            m = 0
            for i in range(self.num_atoms):
                m += self.atoms[i][0].get_magnetic_form_factor(energy) * self.atoms[i][1]
        else:
            m = super().get_magnetic_form_factor(energy)

        return m

    @Atom.atomic_number_z.setter
    def atomic_number_z(self, value):
        raise AttributeError("The atomic number of an AtomMixed is given by its atoms.")

    @Atom.mass_number_a.setter
    def mass_number_a(self, value):
        raise AttributeError("The mass number of an AtomMixed is given by its atoms.")

    @Atom.ionicity.setter
    def ionicity(self, value):
        raise AttributeError("The ionicity of an AtomMixed is given by its atoms.")
