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

__all__ = [
    "ParameterGroup",
    "AtomParameters",
    "AtomMixedParameters",
    "StructuralParameters",
    "LatticeParameters",
    "ThermalParameters",
    "ElasticParameters",
    "OpticalParameters",
    "MagneticParameters",
]

__docformat__ = "restructuredtext"

import copy
import inspect
import re

import numpy as np
import pint
import scipy.constants as constants
import tabulate as tabulate_module
from tabulate import tabulate

from .parameters import Parameter, TemperatureParameter, VectorParameter

u = pint.get_application_registry()


class ParameterGroup:
    """A group of Parameters with units.

    Changes of the independent parameters of a group update its depending
    parameters and notify the owner of the group, e.g. a :class:`.Layer`,
    which updates the parameters depending on several groups.

    """

    # names of the parameters, which are derived from the others of the group
    _derived = ()
    # the owner, e.g. a Layer, which is notified on changes of the group
    _owner = None
    _updating = False

    @classmethod
    def _input_names(cls):
        """Names of the parameters, which can be given on initialization."""
        return [name for name in inspect.signature(cls.__init__).parameters if name != "self"]

    def _parameters(self):
        """All parameters of the group by their names."""
        return {name: p for name, p in vars(self).items() if isinstance(p, Parameter)}

    def _connect_parameters(self):
        """Name all parameters and connect the independent ones to the group,
        so that changes of them update the depending parameters."""
        for name, p in self._parameters().items():
            p.name = name
            if name not in self._derived:
                p._caller = self

    def _update_depending(self):
        """Called on changes of the independent parameters of the group."""
        # the derived parameters are not connected, but the guard prevents any
        # recursion, e.g. by an owner setting parameters of this group
        if self._updating:
            return
        self._updating = True
        try:
            self._update_group()
        finally:
            self._updating = False
        if self._owner is not None:
            self._owner._update_depending()

    def _update_group(self):
        """Update the depending parameters within the group."""

    def __setattr__(self, name, value):
        """Assignments to a parameter of the group set its value, so that the
        parameter keeps its unit, symbol and connection to the group and the
        depending parameters are updated, e.g. ``group.curie_temp = 500 * u.K``.
        A :class:`.Parameter` of the same type replaces the parameter, e.g. to
        give a new symbol, after its value is converted to the unit of the
        group. Derived parameters cannot be set."""
        current = self.__dict__.get(name)
        if not isinstance(current, Parameter):
            # parameters on initialization and all other attributes
            super().__setattr__(name, value)
            return
        if name in self._derived:
            raise AttributeError(
                f"'{name}' is derived from the other parameters of "
                f"{type(self).__name__} and cannot be set!"
            )
        if isinstance(value, Parameter):
            self._replace_parameter(name, current, value)
        elif isinstance(current, VectorParameter):
            current.set(value)
        else:
            current.magnitude = value

    def _replace_parameter(self, name, current, parameter):
        """Replace the parameter by a copy of the given one in the unit of the
        group and connect it to the group."""
        if type(parameter) is not type(current):
            raise TypeError(
                f"'{name}' must be a {type(current).__name__}, not {type(parameter).__name__}!"
            )
        if not parameter.unit.is_compatible_with(current.unit):
            raise pint.DimensionalityError(parameter.unit, current.unit)
        # do not copy the group the parameter is connected to
        new = copy.deepcopy(parameter, {id(parameter._caller): None})
        if new.unit != current.unit:
            if isinstance(new, TemperatureParameter):
                raise ValueError(
                    f"The temperature-dependent parameter '{name}' must be given in "
                    f"{current.unit:~P}!"
                )
            if isinstance(new, VectorParameter):
                new.unit = current.unit
                new.set(parameter)
            else:
                new.unit = current.unit
                new.magnitude = parameter.quantity
        new.name = name
        new._caller = self
        super().__setattr__(name, new)
        self._update_depending()

    def __deepcopy__(self, memo):
        # A group copied on its own, e.g. on assignment to a layer, must not copy
        # its owner along. When copying the owner, it is already in the memo.
        if self._owner is not None and id(self._owner) not in memo:
            memo[id(self._owner)] = None
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in vars(self).items():
            setattr(new, key, copy.deepcopy(value, memo))
        return new

    _table_headers = ("Parameter", "Symbol", "Value", "Unit")
    _table_align = ("left", "center", "right", "left")

    def _table_representation(self, style="double_grid"):
        """Table of the name, symbol, value, and unit of all parameters in the
        given tabulate style, where the sub-systems of a parameter are given
        in separate lines of a single row."""
        rows = []
        for name, p in self._parameters().items():
            symbols, values, units = zip(*p._table_rows("text"))
            rows.append([name, "\n".join(symbols), "\n".join(values), "\n".join(units)])
        kwargs = dict(
            headers=self._table_headers,
            tablefmt=style,
            colalign=self._table_align,
            disable_numparse=True,
        )
        # The whitespace must be preserved, as the lines of a cell can be empty,
        # e.g. the dimensionless amplitude of a vector before its angles.
        try:
            return tabulate(rows, preserve_whitespace=True, **kwargs)
        except TypeError:  # tabulate < 0.10 has a module-wide option only
            preserve_whitespace = tabulate_module.PRESERVE_WHITESPACE
            tabulate_module.PRESERVE_WHITESPACE = True
            try:
                return tabulate(rows, **kwargs)
            finally:
                tabulate_module.PRESERVE_WHITESPACE = preserve_whitespace

    def _html_table(self):
        """HTML table of the name, symbol, value, and unit of all parameters,
        where the name and unit of a parameter span the rows of its sub-systems.
        The table is not made by tabulate, which pads all cells to the length of
        the longest HTML string and does not support cells spanning rows."""

        def cell(tag, content, align, rowspan=1):
            span = f' rowspan="{rowspan}"' if rowspan > 1 else ""
            style = f' style="text-align: {align};"' if align != "left" else ""
            return f"<{tag}{span}{style}>{content}</{tag}>"

        header = "".join(
            cell("th", name, align) for name, align in zip(self._table_headers, self._table_align)
        )
        body = ""
        for name, p in self._parameters().items():
            rows = p._table_rows("html")
            # a unit, which is the same for all rows, spans them
            same_unit = all(unit == "" for _, _, unit in rows[1:])
            for i, (symbol, value, unit) in enumerate(rows):
                cells = [cell("td", symbol, "center"), cell("td", value, "right")]
                if i == 0:
                    cells.insert(0, cell("td", name, "left", len(rows)))
                if i == 0 and same_unit:
                    cells.append(cell("td", unit, "left", len(rows)))
                elif not same_unit:
                    cells.append(cell("td", unit, "left"))
                body += "<tr>" + "".join(cells) + "</tr>\n"
        return f"<table>\n<thead>\n<tr>{header}</tr>\n</thead>\n<tbody>\n{body}</tbody>\n</table>"

    def _pretty_class_name(self):
        return " ".join(
            re.sub(
                "([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", self.__class__.__name__)
            ).split()
        )

    def __repr__(self):
        class_str = self._pretty_class_name() + "\n"
        class_str += self._table_representation()
        return class_str

    def _repr_html_(self):
        return f"<h3>{self._pretty_class_name()}</h3>" + self._html_table()


class _AtomParametersBase(ParameterGroup):
    """Parameters common to :class:`AtomParameters` and :class:`AtomMixedParameters`."""

    _derived = ("atomic_number", "mass_number", "mass")

    def __init__(
        self,
        atomic_number=0,
        mass_number=0.0,
        ionicity=0,
        magnetization=(0.0, 0.0, 0.0),
    ):
        self.atomic_number = Parameter("", atomic_number, symbol="Z")
        self.mass_number = Parameter("", mass_number, symbol="A")
        self.mass = Parameter("kg", 0.0, symbol="m")
        self.ionicity = Parameter("", ionicity, symbol="I")
        self.magnetization = VectorParameter("", magnetization, symbol=r"\mathbf{m}")

        self._connect_parameters()
        self._update_group()

    def _update_group(self):
        self.mass.magnitude = self.mass_number.magnitude * constants.atomic_mass


class AtomParameters(_AtomParametersBase):
    r"""Parameters of an atom.

    The parameters of an atom are read by a layer when the atom is added to
    it, e.g. its mass for the density of a unit cell. Hence, changing the
    parameters of an atom afterwards does not update the layers it was
    already added to.

    The atomic number and mass number are given by the element of the atom.
    A change of the ionicity updates the Cromer-Mann coefficients, see
    :meth:`.Atom.read_cromer_mann_coeff`.

    Attributes:
        atomic_number (int): atomic number :math:`Z`.
        mass_number (float): mass number :math:`A`.
        mass (float): mass of the atom :math:`m` [kg], derived from the mass number.
        ionicity (int): ionicity :math:`I`.
        magnetization (VectorParameter): magnetization :math:`\mathbf{m}` as
            for the :class:`.MagneticParameters` of a layer, given by its
            amplitude :math:`r`, which is negative for the reversed direction,
            and the angles :math:`\phi` from the z-axis and :math:`\gamma` in
            the xy-plane from the x-axis, and given in the same forms.
        cromer_mann_a (ndarray[float]): Cromer-Mann coefficients :math:`a_1 .. a_4`.
        cromer_mann_b (ndarray[float]): Cromer-Mann coefficients :math:`b_1 .. b_4`
            [Å²].
        cromer_mann_c (float): Cromer-Mann coefficient :math:`c`.

    """

    # the Cromer-Mann coefficients are given by the atomic number and ionicity
    _derived = _AtomParametersBase._derived + ("cromer_mann_a", "cromer_mann_b", "cromer_mann_c")

    def __init__(
        self,
        atomic_number=0,
        mass_number=0.0,
        ionicity=0,
        magnetization=(0.0, 0.0, 0.0),
        cromer_mann_a=(0.0, 0.0, 0.0, 0.0),
        cromer_mann_b=(0.0, 0.0, 0.0, 0.0),
        cromer_mann_c=0.0,
    ):
        super().__init__(atomic_number, mass_number, ionicity, magnetization)
        self.cromer_mann_a = Parameter("", np.asarray(cromer_mann_a, dtype=float), symbol="a")
        self.cromer_mann_b = Parameter(
            "angstrom**2", np.asarray(cromer_mann_b, dtype=float), symbol="b"
        )
        self.cromer_mann_c = Parameter("", float(cromer_mann_c), symbol="c")

        self._connect_parameters()


class AtomMixedParameters(_AtomParametersBase):
    r"""Parameters of a mixed atom.

    The atomic number, mass number, mass, and ionicity are the averages of
    the constituent atoms weighted by their fractions, see
    :meth:`.AtomMixed.add_atom`. A mixed atom has no Cromer-Mann
    coefficients, as its form factors are averaged from its constituents.

    Attributes:
        atomic_number (float): averaged atomic number :math:`Z`.
        mass_number (float): averaged mass number :math:`A`.
        mass (float): averaged mass of the atom :math:`m` [kg].
        ionicity (float): averaged ionicity :math:`I`.
        magnetization (VectorParameter): magnetization :math:`\mathbf{m}`, see
            :class:`AtomParameters`.

    """

    _derived = ("atomic_number", "mass_number", "mass", "ionicity")

    def _update_group(self):
        """The derived parameters are given by the constituents."""


class StructuralParameters(ParameterGroup):
    r"""Structural parameters of a layer.

    Attributes:
        thickness (float): thickness of the layer :math:`d` [m].
        density (float): density of the layer :math:`\rho` [kg/m³].
        area (float): area of layer :math:`A` [m²].
        roughness (float): gaussian width of the top roughness of a layer :math:`\sigma` [m].
        mass (float): mass of the layer :math:`m` [kg].
        mass_unit_area (float): mass of layer normalized to unit area of 1 Å² :math:`m_A` [kg].
        volume (float): volume of layer :math:`V` [m³].

    """

    _derived = ("mass", "mass_unit_area", "volume")

    def __init__(self, thickness=0.0, density=0.0, area=1.0 * u.angstrom**2, roughness=0.0):
        self.thickness = Parameter("m", thickness, symbol="d")
        self.density = Parameter("kg/m**3", density, symbol=r"\rho")
        self.area = Parameter("m**2", area, symbol="A")
        self.roughness = Parameter("m", roughness, symbol=r"\sigma")
        self.mass = Parameter("kg", 0.0, symbol="m")
        self.mass_unit_area = Parameter("kg", 0.0, symbol="m_A")
        self.volume = Parameter("m**3", 0.0, symbol="V")

        self._connect_parameters()
        self._update_group()

    def _update_group(self):
        self.volume.magnitude = self.thickness.magnitude * self.area.magnitude
        self.mass.magnitude = self.density.magnitude * self.volume.magnitude
        try:
            self.mass_unit_area.quantity = (
                self.mass.quantity / self.area.quantity * 1.0 * u.angstrom**2
            )
        except ZeroDivisionError:
            self.mass_unit_area.quantity = 0 * u.kg


class LatticeParameters(ParameterGroup):
    """Lattice parameters of a unit cell.

    Attributes:
        a_axis (float): lattice parameter :math:`a` [m], defaults to c_axis.
        b_axis (float): lattice parameter :math:`b` [m], defaults to c_axis.
        c_axis (float): lattice parameter :math:`c` [m].

    """

    def __init__(self, a_axis=None, b_axis=None, c_axis=0.0):
        self.a_axis = Parameter("m", c_axis if a_axis is None else a_axis, symbol="a")
        self.b_axis = Parameter("m", c_axis if b_axis is None else b_axis, symbol="b")
        self.c_axis = Parameter("m", c_axis, symbol="c")

        self._connect_parameters()


class ThermalParameters(ParameterGroup):
    r"""Thermal parameters of a layer.

    Attributes:
        therm_cond (list[@lambda]): list of T-dependent thermal conductivity
            :math:`\kappa` [W/(m K)].
        lin_therm_exp (list[@lambda]): list of T-dependent linear thermal
            expansion coefficient :math:`\alpha` (relative).
        heat_capacity (list[@lambda]): list of T-dependent heat capacity
            function :math:`c` [J/(kg K)].
        sub_system_coupling (list[@lambda]): list of coupling functions of
            different subsystems :math:`G` [W/m³].
        deb_wal_fac (list[@lambda]): list of T-dependent Debye-Waller factors
            :math:`\langle u^2\rangle` [m²].
        num_sub_systems (int): number of subsystems :math:`N` for heat and
            phonons (electrons, lattice, spins, ...).
        sub_system_labels (list[str]): labels of the subsystems, e.g.
            ``["e", "p"]``, which are displayed as subscripts of the symbols of
            the temperature-dependent parameters and of the temperatures.
            Missing labels default to the index of the subsystem.

    """

    _derived = ("num_sub_systems",)

    def __init__(
        self,
        therm_cond=0.0,
        heat_capacity=0.0,
        lin_therm_exp=0.0,
        sub_system_coupling=0.0,
        deb_wal_fac=0.0,
        sub_system_labels=None,
    ):
        self.therm_cond = TemperatureParameter("W/(m K)", therm_cond, symbol=r"\kappa")
        self.heat_capacity = TemperatureParameter("J/(kg K)", heat_capacity, symbol="c")
        self.lin_therm_exp = TemperatureParameter("", lin_therm_exp, symbol=r"\alpha")
        self.sub_system_coupling = TemperatureParameter("W/m**3", sub_system_coupling, symbol="G")
        self.deb_wal_fac = TemperatureParameter("m**2", deb_wal_fac, symbol=r"\langle u^2 \rangle")
        self.num_sub_systems = Parameter("", 0, symbol="N")
        self.sub_system_labels = sub_system_labels

        self._connect_parameters()
        self._update_group()

    @property
    def sub_system_labels(self):
        """list[str]: labels of the sub-systems, which default to their index."""
        n = self.num_sub_systems.magnitude
        labels = (self._sub_system_labels or [])[:n]
        return labels + [str(i) for i in range(len(labels), n)]

    @sub_system_labels.setter
    def sub_system_labels(self, labels):
        self._sub_system_labels = None if labels is None else [str(label) for label in labels]

    def _display_sub_system_labels(self):
        """Labels of the sub-systems for the display of the temperature-dependent
        parameters, or None for a single sub-system without label."""
        if self._sub_system_labels is None and self.num_sub_systems.magnitude <= 1:
            return None
        return self.sub_system_labels

    def _update_group(self):
        # The number of sub-systems is given by the temperature-dependent
        # parameter with the most sub-systems. Their consistency is not checked
        # here, as they are typically set one after another, but by the
        # simulations for the parameters they actually use, see
        # Structure.check_num_sub_systems().
        self.num_sub_systems.magnitude = max(
            p.num_sub_systems for p in vars(self).values() if isinstance(p, TemperatureParameter)
        )


class ElasticParameters(ParameterGroup):
    r"""Elastic and phonon parameters of a layer.

    Attributes:
        sound_vel (float): longitudinal sound velocity in the layer :math:`v` [m/s].
        phonon_damping (float): damping constant of phonon propagation :math:`\gamma` [kg/s].
        spring_const (ndarray[float]): spring constant :math:`k` of the layer
            [kg/s²] and higher orders.
        acoustic_impedance (float): acoustic impedance of a layer :math:`Z` [kg/(m² s)].

    """

    _derived = ("spring_const", "acoustic_impedance")

    def __init__(self, sound_vel=0.0, phonon_damping=0.0):
        self.sound_vel = Parameter("m/s", sound_vel, symbol="v")
        self.phonon_damping = Parameter("kg/s", phonon_damping, symbol=r"\gamma")
        self.spring_const = Parameter("kg/s**2", np.array([0.0]), symbol="k")
        self.acoustic_impedance = Parameter("kg/(m**2*s)", 0.0, symbol="Z")

        self._connect_parameters()

    def calc_spring_const(self, mass_unit_area, thickness):
        r"""calc_spring_const

        Calculates the spring constant of the layer from the mass per unit area,
        sound velocity and thickness

        .. math:: k = m \, \left(\frac{v}{d}\right)^2

        """
        try:
            self.spring_const.magnitude[0] = (
                mass_unit_area * (self.sound_vel.magnitude / thickness) ** 2
            )
        except (ZeroDivisionError, AttributeError):
            # no mass set, yet
            self.spring_const.magnitude[0] = 0

    def calc_acoustic_impedance(self, mass, area):
        """calc_acoustic_impedance

        Calculates the acoustic impedance.

        Returns:
            Z (float): acoustic impedance.

        """
        self.acoustic_impedance.magnitude = np.sqrt(
            self.spring_const.magnitude[0] * mass / area**2
        )

    def set_ho_spring_constants(self, HO):
        """set_ho_spring_constants

        Set the higher orders of the spring constant for anharmonic
        phonon simulations.

        Args:
            HO (ndarray[float]): higher order spring constants.

        """
        # reset old higher order spring constants
        self.spring_const.magnitude = np.delete(
            self.spring_const.magnitude, np.r_[1 : len(self.spring_const.magnitude)]
        )
        self.spring_const.magnitude = np.hstack((self.spring_const.magnitude, HO))


class OpticalParameters(ParameterGroup):
    r"""Optical and X-ray parameters of a layer.

    Attributes:
        opt_pen_depth (float): optical penetration depth of the layer :math:`\zeta` [m].
        opt_ref_index (complex): complex optical refractive index :math:`n`.
        opt_ref_index_per_strain (complex): change of the complex optical
            refractive index per strain :math:`\mathrm{d}n/\mathrm{d}\epsilon`.

    """

    def __init__(
        self, opt_pen_depth=0.0, opt_ref_index=0.0 + 0.0j, opt_ref_index_per_strain=0.0 + 0.0j
    ):
        self.opt_pen_depth = Parameter("m", opt_pen_depth, symbol=r"\zeta")
        self.opt_ref_index = Parameter("", opt_ref_index, symbol="n")
        self.opt_ref_index_per_strain = Parameter(
            "", opt_ref_index_per_strain, symbol=r"\mathrm{d}n/\mathrm{d}\epsilon"
        )

        self._connect_parameters()

    def set_opt_pen_depth_from_ref_index(self, wavelength):
        r"""set_opt_pen_depth_from_ref_index

        Set the optical penetration depth from the optical refractive index
        for a given wavelength:

        .. math:: \zeta = \frac{\lambda}{4 \pi |\mathrm{Im}(n)|}

        Args:
            wavelength (Quantity): wavelength as Pint Quantity.

        """
        kappa = np.abs(np.imag(self.opt_ref_index.magnitude))
        if kappa == 0:
            self.opt_pen_depth.magnitude = np.inf
        else:
            self.opt_pen_depth.quantity = wavelength / (4 * np.pi * kappa)


class MagneticParameters(ParameterGroup):
    r"""Magnetic parameters of a layer.

    A layer can consist of :math:`P` magnetic sublattices, e.g. the two
    antiparallel sublattices of a ferrimagnet. The per-sublattice parameters
    are given either as scalar, which holds for all sublattices, or as one
    value per sublattice, and their numbers must be consistent, see
    :meth:`check_sublattices`. The number of sublattices is given by the
    per-sublattice parameter with the most values, see
    :attr:`num_sublattices`.

    The sublattices are coupled by the mean-field exchange matrix
    :math:`J_{ab}` [J], whose row :math:`a` holds the coupling constants of
    the field acting on sublattice :math:`a` due to sublattice :math:`b`, and
    whose negative off-diagonal elements couple antiferromagnetically. It is
    derived from the Curie temperature for a single sublattice, see
    :meth:`calc_mf_exchange_coupling`, and must be given as
    ``mf_exch_coupling`` for several sublattices, in which case the Curie
    temperature is derived from it as the ordering temperature, see
    :meth:`calc_curie_temp`. Setting one of both makes the other one the
    derived parameter, where the Curie temperature can only be set for a
    single sublattice.

    Attributes:
        eff_spin (float, ndarray[float]): effective spin :math:`S` per
            sublattice, which is infinite for the classical limit.
        curie_temp (float): Curie temperature :math:`T_\mathrm{C}` [K].
        mf_exch_coupling (float, ndarray[float]): mean-field exchange coupling
            constant :math:`J` [m²kg/s²], a :math:`P \times P` matrix for
            several sublattices.
        lamda (float, ndarray[float]): intrinsic coupling to bath parameter
            :math:`\lambda` per sublattice.
        mag_moment (float, ndarray[float]): atomic magnetic moment
            :math:`\mu` per sublattice [A m²].
        aniso_exponent (float, ndarray[float]): exponent :math:`\beta` of the
            temperature dependence of the anisotropy per sublattice.
        anisotropy (ndarray[float]): anisotropy :math:`K` at T=0 K [J/m³] as
            x,y,z component vector, or one vector per sublattice as
            :math:`P \times 3` array.
        exch_stiffness (float): exchange stiffness at T=0 K
            :math:`A_\mathrm{ex}` [J/m] towards the neighbouring layers.
        mag_saturation (float, ndarray[float]): saturation magnetization at
            0 K :math:`M_s` per sublattice [J/T/m³].
        magnetization (VectorParameter, list[VectorParameter]): magnetization
            :math:`\mathbf{m}` given by its amplitude :math:`r`, which is
            negative for the reversed direction, and the angles :math:`\phi`
            from the z-axis and :math:`\gamma` in the xy-plane from the
            x-axis. It is given as cartesian components, e.g. ``(0, 0, 1)``,
            or as dict of polar or cartesian components, e.g. ``{"r": 1,
            "phi": 90 * u.deg, "gamma": 0 * u.deg}`` or ``{"x": 0, "y": 0,
            "z": 1}``, see :class:`.VectorParameter`, and as list of such
            inputs with one magnetization per sublattice. For an
            :class:`.AmorphousLayer` it is inherited from its atom, i.e. from
            the constituents of an :class:`.AtomMixed` with one constituent
            per sublattice.
        gilbert_damping (float, ndarray[float]): Gilbert damping
            :math:`\alpha` of the LLG per sublattice.
        gyromagnetic_ratio (float, ndarray[float]): gyromagnetic ratio
            :math:`\gamma = -g \mu_B / \hbar` of the magnetization dynamics
            per sublattice, including its negative sign [rad/(s T)]. Defaults
            to the one of a free electron
            :math:`\gamma_e = -1.761\times10^{11}` rad/(s T).
        inertial_time (float, ndarray[float]): inertial relaxation time
            :math:`\tau` of the inertial LLG per sublattice [s].
        magnetoelastic_coupling (float, ndarray[float]): magnetoelastic
            coupling constant :math:`B_1` of the out-of-plane strain per
            sublattice [J/m³].
        exch_relax_rate (float): exchange-relaxation rate
            :math:`\lambda_\mathrm{ex}` between two sublattices of the
            extended LLB, see
            :func:`~udkm1Dsim.simulations.solvers.magnetization_scipy.exchange_relaxation_torque`.
        concentration (float, ndarray[float]): concentration :math:`x` per
            sublattice for the exchange relaxation. It is derived as equal
            concentrations :math:`1/P` unless it is given, e.g. as the
            fractions of the constituents of an :class:`.AtomMixed`.

    """

    _derived = ("mf_exch_coupling",)

    # parameters with one scalar value per sublattice
    _sublattice_scalars = (
        "eff_spin",
        "lamda",
        "mag_moment",
        "aniso_exponent",
        "mag_saturation",
        "gilbert_damping",
        "gyromagnetic_ratio",
        "inertial_time",
        "magnetoelastic_coupling",
        "concentration",
    )

    def __init__(
        self,
        eff_spin=0.0,
        curie_temp=0.0,
        lamda=0.0,
        mag_moment=0.0,
        aniso_exponent=0.0,
        anisotropy=np.array([0.0, 0.0, 0.0]),
        exch_stiffness=np.array([0.0, 0.0, 0.0]),
        mag_saturation=0.0,
        magnetization=np.array([0.0, 0.0, 0.0]),
        gilbert_damping=0.0,
        gyromagnetic_ratio=-constants.physical_constants["electron gyromag. ratio"][0],
        inertial_time=0.0,
        magnetoelastic_coupling=0.0,
        mf_exch_coupling=None,
        exch_relax_rate=0.0,
        concentration=None,
    ):
        # the exchange matrix is given explicitly, the Curie temperature derived
        self._exchange_derived = "mf_exch_coupling" if mf_exch_coupling is None else "curie_temp"
        # without explicit concentrations they are equal
        self._concentration_explicit = concentration is not None
        self._update_derived_names()
        self.eff_spin = Parameter("", eff_spin, symbol="S")
        self.curie_temp = Parameter("K", curie_temp, symbol=r"T_\mathrm{C}")
        self.mf_exch_coupling = Parameter(
            "m**2kg/s**2", 0.0 if mf_exch_coupling is None else mf_exch_coupling, symbol="J"
        )
        self.lamda = Parameter("", lamda, symbol=r"\lambda")
        self.mag_moment = Parameter("A*m**2", mag_moment, symbol=r"\mu")
        self.aniso_exponent = Parameter("", aniso_exponent, symbol=r"\beta")
        self.anisotropy = Parameter("J/m**3", anisotropy, symbol="K")
        self.exch_stiffness = Parameter("J/m", exch_stiffness, symbol=r"A_\mathrm{ex}")
        self.mag_saturation = Parameter("J/T/m**3", mag_saturation, symbol="M_s")
        self.magnetization = self._make_magnetization(magnetization)
        self.gilbert_damping = Parameter("", gilbert_damping, symbol=r"\alpha")
        self.gyromagnetic_ratio = Parameter("rad/(s*T)", gyromagnetic_ratio, symbol=r"\gamma")
        self.inertial_time = Parameter("s", inertial_time, symbol=r"\tau")
        self.magnetoelastic_coupling = Parameter("J/m**3", magnetoelastic_coupling, symbol="B_1")
        self.exch_relax_rate = Parameter("", exch_relax_rate, symbol=r"\lambda_\mathrm{ex}")
        self.concentration = Parameter(
            "", 1.0 if concentration is None else concentration, symbol="x"
        )

        self._connect_parameters()
        self._update_group()
        self._initialized = True

    def _update_derived_names(self):
        """the names of the derived parameters"""
        self._derived = (self._exchange_derived,) + (
            () if self._concentration_explicit else ("concentration",)
        )

    # the magnetization of several sublattices is a list of VectorParameters

    @staticmethod
    def _is_magnetization_list(value):
        """A list of magnetizations is a list, tuple, or array whose elements
        are vectors, dicts, or VectorParameters, in contrast to the
        components of a single vector."""
        if isinstance(value, np.ndarray):
            return value.ndim == 2
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return all(
                isinstance(v, (list, tuple, dict, np.ndarray, VectorParameter)) for v in value
            )
        return False

    def _make_magnetization(self, value):
        """A VectorParameter or a list of VectorParameters, one per
        sublattice, from the given input."""
        if not self._is_magnetization_list(value):
            return VectorParameter("", value, symbol=r"\mathbf{m}")
        vectors = []
        for a, v in enumerate(value):
            vector = VectorParameter("", (0.0, 0.0, 0.0), symbol=rf"\mathbf{{m}}_{{{a + 1:d}}}")
            vector.set(v)
            vectors.append(vector)
        return vectors

    def _parameters(self):
        """All parameters of the group by their names, where the
        magnetizations of several sublattices are numbered from 1."""
        parameters = {}
        for name, p in vars(self).items():
            if isinstance(p, Parameter):
                parameters[name] = p
            elif name == "magnetization" and isinstance(p, list):
                for a, vector in enumerate(p):
                    parameters[f"magnetization_{a + 1:d}"] = vector
        return parameters

    def __setattr__(self, name, value):
        """Besides the assignments of :class:`ParameterGroup`, the
        magnetization is set from a single or several vectors, and the Curie
        temperature and the exchange coupling switch the derived one."""
        initialized = self.__dict__.get("_initialized", False)
        if not initialized or name.startswith("_"):
            super().__setattr__(name, value)
        elif name == "magnetization":
            self._set_magnetization(value)
        elif name == "mf_exch_coupling" and "mf_exch_coupling" in self._derived:
            # an explicit exchange coupling makes the Curie temperature derived
            self._switch_derived("curie_temp")
            super().__setattr__(name, value)
        elif name == "curie_temp" and "curie_temp" in self._derived:
            if self.num_sublattices > 1:
                raise AttributeError(
                    "The Curie temperature of several sublattices is derived from the "
                    "mean-field exchange matrix mf_exch_coupling and cannot be set!"
                )
            self._switch_derived("mf_exch_coupling")
            super().__setattr__(name, value)
        elif name == "concentration" and not self._concentration_explicit:
            # explicit concentrations are not derived anymore
            self._concentration_explicit = True
            self._update_derived_names()
            self.concentration._caller = self
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def _switch_derived(self, name):
        """Make the given parameter of the exchange the derived one and
        connect the other."""
        self._exchange_derived = name
        self._update_derived_names()
        self.curie_temp._caller = None if name == "curie_temp" else self
        self.mf_exch_coupling._caller = None if name == "mf_exch_coupling" else self

    def _set_magnetization(self, value):
        """Set the magnetization from a single or several vectors, keeping
        the existing VectorParameters for the same number of sublattices."""
        current = self.magnetization
        if isinstance(value, VectorParameter) and isinstance(current, VectorParameter):
            # a VectorParameter replaces the current one, e.g. for its symbol
            ParameterGroup.__setattr__(self, "magnetization", value)
            return
        if self._is_magnetization_list(value):
            if isinstance(current, list) and len(current) == len(value):
                for vector, v in zip(current, value):
                    vector._caller = None
                    vector.set(v)
                    vector._caller = self
            else:
                object.__setattr__(self, "magnetization", self._make_magnetization(value))
        elif isinstance(current, VectorParameter):
            current.set(value)
            return
        else:
            object.__setattr__(self, "magnetization", self._make_magnetization(value))
        self._connect_parameters()
        self._update_depending()

    def _connect_parameters(self):
        super()._connect_parameters()
        # all magnetizations of the sublattices carry the name of the parameter
        if isinstance(self.__dict__.get("magnetization"), list):
            for vector in self.magnetization:
                vector.name = "magnetization"

    # number and consistency of the sublattices

    @property
    def num_sublattices(self):
        """int: number of magnetic sublattices, the largest number of values of
        any per-sublattice parameter."""
        sizes = [np.size(getattr(self, name).magnitude) for name in self._sublattice_scalars]
        anisotropy = np.asarray(self.anisotropy.magnitude)
        sizes.append(anisotropy.shape[0] if anisotropy.ndim == 2 else 1)
        if isinstance(self.magnetization, list):
            sizes.append(len(self.magnetization))
        exchange = np.asarray(self.mf_exch_coupling.magnitude)
        if "curie_temp" in self._derived and exchange.ndim == 2:
            sizes.append(exchange.shape[0])
        return max(sizes)

    def check_sublattices(self):
        """check_sublattices

        Checks that all per-sublattice parameters have either a single value,
        which holds for all sublattices, or one value per sublattice, that
        several sublattices have an explicit exchange matrix
        ``mf_exch_coupling`` and a magnetization per sublattice, unless it is
        zero. The parameters are not checked on every change, as they are
        typically set one after another, but by the simulations.

        Returns:
            P (int): number of sublattices.

        """
        P = self.num_sublattices
        for name in self._sublattice_scalars:
            size = np.size(getattr(self, name).magnitude)
            if size not in (1, P):
                raise ValueError(
                    f"The magnetic parameter {name} has {size:d} values, but there are "
                    f"{P:d} sublattices!"
                )
        anisotropy = np.asarray(self.anisotropy.magnitude)
        if anisotropy.shape not in ((3,), (P, 3)):
            raise ValueError(
                f"The anisotropy must be a vector or one vector for each of the {P:d} sublattices!"
            )
        if isinstance(self.magnetization, list):
            if len(self.magnetization) != P:
                raise ValueError(
                    f"There are {len(self.magnetization):d} magnetizations, but {P:d} sublattices!"
                )
        elif P > 1 and np.any(self.magnetization.magnitude != 0):
            raise ValueError(f"A magnetization must be given for each of the {P:d} sublattices!")
        if P > 1:
            exchange = np.asarray(self.mf_exch_coupling.magnitude)
            if "curie_temp" not in self._derived or exchange.shape != (P, P):
                raise ValueError(
                    f"Several sublattices require an explicit {P:d}x{P:d} mean-field "
                    "exchange matrix mf_exch_coupling!"
                )
        return P

    def sublattice_values(self, name):
        """sublattice_values

        Returns the magnitudes of a per-sublattice parameter for all
        sublattices, i.e. as array :math:`P` or :math:`P \times 3` for the
        anisotropy, or the exchange matrix :math:`P \times P`.

        Args:
            name (str): name of the parameter.

        Returns:
            values (ndarray[float]): magnitudes for all sublattices.

        """
        P = self.num_sublattices
        value = np.asarray(getattr(self, name).magnitude, dtype=float)
        if name == "anisotropy":
            return np.broadcast_to(value, (P, 3)).copy()
        if name == "mf_exch_coupling":
            return np.broadcast_to(value.reshape((1, 1)) if value.size == 1 else value, (P, P))
        return np.broadcast_to(value.reshape(-1) if value.size == 1 else value, (P,)).copy()

    def sublattice_magnetizations(self):
        """sublattice_magnetizations

        Returns the magnetizations of all sublattices as list of
        :class:`.VectorParameter`, where a single zero magnetization holds for
        all sublattices.

        Returns:
            magnetizations (list[VectorParameter]): magnetizations.

        """
        if isinstance(self.magnetization, list):
            return list(self.magnetization)
        return [self.magnetization] * self.num_sublattices

    # update of the derived parameters

    def _update_group(self):
        if "curie_temp" in self._derived:
            self.calc_curie_temp()
        else:
            self.calc_mf_exchange_coupling()
        if not self._concentration_explicit:
            # equal concentrations, a scalar for all sublattices
            self.concentration.magnitude = 1.0 / self.num_sublattices
        self.check_array(self.exch_stiffness, "Exchange stiffness")
        self.check_anisotropy()

    def calc_mf_exchange_coupling(self):
        r"""calc_mf_exchange_coupling

        Calculate the mean-field exchange coupling constant

        .. math:: J = \frac{3 S_{eff}}{S_{eff}+1} k_B T_C

        which is :math:`J = 3 k_B T_C` in the classical limit of an infinite
        effective spin.

        """
        eff_spin = np.asarray(self.eff_spin.magnitude, dtype=float)
        with np.errstate(invalid="ignore"):
            factor = np.where(np.isinf(eff_spin), 3.0, 3 * eff_spin / (eff_spin + 1))
        J = factor * constants.k * self.curie_temp.magnitude
        self.mf_exch_coupling.magnitude = float(J) if J.ndim == 0 else J

    def calc_curie_temp(self):
        r"""calc_curie_temp

        Calculate the Curie temperature from the explicit mean-field exchange
        matrix as the ordering temperature of the coupled sublattices, see
        :func:`~udkm1Dsim.simulations.solvers.mean_field.ordering_temperature`

        .. math:: k_B T_C = \max \mathrm{eig}\left(\frac{S_a+1}{3 S_a} J_{ab}\right)

        which reduces to :math:`T_C = \frac{S+1}{3 S k_B} J` for one sublattice.

        """
        from ..simulations.solvers.mean_field import ordering_temperature

        J = np.asarray(self.mf_exch_coupling.magnitude, dtype=float)
        J = J.reshape((1, 1)) if J.size == 1 else J
        eff_spins = np.asarray(self.eff_spin.magnitude, dtype=float).reshape(-1)
        try:
            eff_spins = np.broadcast_to(eff_spins, (J.shape[0],))
            curie_temp = ordering_temperature(J, eff_spins) if J.ndim == 2 else 0.0
        except ValueError:
            # inconsistent numbers of sublattices, see check_sublattices
            curie_temp = 0.0
        self.curie_temp.magnitude = curie_temp

    def check_anisotropy(self):
        """check_anisotropy

        A scalar anisotropy holds for all three axes, a vector of length 3 or
        one vector per sublattice as :math:`P \times 3` array are kept.

        """
        value = np.asarray(self.anisotropy.magnitude)
        if value.ndim == 2 and value.shape[1] == 3:
            return
        self.check_array(self.anisotropy, "Anisotropy")

    def check_array(self, parameter, name):
        value = np.asarray(parameter.magnitude)

        if value.ndim == 0:
            value = np.full(3, value)
        elif value.shape == (1,):
            value = np.full(3, value[0])
        elif value.shape != (3,):
            raise ValueError(f"{name} must be a scalar or a vector of length 3!")

        # set directly, as the setter would notify the group again
        parameter._magnitude = value
