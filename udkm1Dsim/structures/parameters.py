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
    "Parameter",
    "ParameterGroup",
    "StructuralParameters",
    "LatticeParameters",
    "ThermalParameters",
    "ElasticParameters",
    "OpticalParameters",
    "MagneticParameters",
]

__docformat__ = "restructuredtext"

import re
from dataclasses import dataclass, field, fields
from inspect import isfunction
import warnings

import numpy as np
import pint
import scipy.constants as constants
from scipy.integrate import quad
from sympy import integrate, lambdify, symarray, symbols, sympify
from sympy.printing.numpy import NumPyPrinter
from tabulate import tabulate

u = pint.get_application_registry()


class Parameter:
    """Parameter with a unit and a magnitude."""

    def __init__(self, unit, magnitude=0.0, name=""):
        self.unit = u.Unit(unit)
        self.name = name
        self._caller = None
        self.magnitude = magnitude

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, value):
        if isinstance(value, u.Quantity):
            self._magnitude = value.to(self.unit).magnitude
        elif isinstance(value, (int, float, complex, np.ndarray)):
            self._magnitude = value
        else:
            raise TypeError(f"Cannot set Parameter '{self.name}' from type {type(value)}")

        if self._caller is not None:
            self._caller._update_depending()

    @property
    def quantity(self):
        return self._magnitude * self.unit

    @quantity.setter
    def quantity(self, value):
        self.magnitude = value

    def __repr__(self):
        return f"Parameter({self.name}={self.magnitude} {self.unit})"


class TemperatureParameter(Parameter):
    """Parameter with a unit and a magnitude, which depends on temperature."""

    def __init__(self, unit, magnitude=0.0, name=""):
        super().__init__(unit, magnitude=0.0, name="")
        self._functional = []
        self._integral = []
        self._integral_expr = []

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, value):
        self._magnitude = self.parse_input(value)
        # reset of dependent parameters on change
        self._functional = []
        self._integral = []
        self._integral_expr = []
        if self._caller is not None:
            self._caller._update_depending()

    @property
    def functional(self):
        if self._functional == []:
            for expression in self._magnitude:
                syms = sorted(expression.free_symbols, key=lambda s: s.name)
                if len(syms) == 0:
                    syms = ['T']

                if len(syms) == 1:
                    is_vector = False
                else:
                    is_vector = True

                if is_vector:
                    body = NumPyPrinter().doprint(expression)
                    unpack = "".join(f"    T_{i} = T[{i}]\n" for i in range(len(syms)))
                    src = f"def _f(T):\n{unpack}    return {body}\n"
                    ns = {'numpy': np}
                    exec(src, ns)
                    f = ns["_f"]
                    self._functional.append(f)
                else:
                    self._functional.append(lambdify(syms, expression, modules="numpy"))

        return self._functional

    @property
    def integral(self):
        if self._integral == []:
            self._integral_expr = []
            T = symbols("T")
            for hc, hcs in zip(self.functional, self.magnitude):
                try:
                    integral = integrate(hcs, T)
                    self._integral.append(lambdify(T, integral, modules='numpy'))
                    self._integral_expr.append(integral)
                except Exception:
                    warnings.warn('\nSympy\'s analytical integration of the heat capacity '
                                    'did not work.\n'
                                    'Just do it numerically with scipy.integrate.quad')
                    self._integral.append(lambda T: quad(hc, 0, T, limit=10000)[0])
                    self._integral_expr.append(f'scipy.integrate.quad({hcs:s}, 0, T)[0]')

        return self._integral

    @property
    def integral_expr(self):
        if self._integral_expr == []:
            self.integral
        return self._integral_expr

    @property
    def quantity(self):
        return self.magnitude

    @quantity.setter
    def quantity(self, value):
        self.magnitude = value

    def __repr__(self):
        return f"Parameter({self.name}={self.magnitude} {self.unit})"

    def parse_input(self, inputs):
        """parse_input

        Parses the input and create a list of function handle strings with T as
        argument. Inputs can be strings, floats, ints, or pint quantities.

        Args:
            inputs (list[str, int, float, Quantity]): list of strings, int, floats,
                or Pint quantities.

        Returns:
            (tuple):
            - *expressions (list[@expres])* - list of sympy expressions from sympify.

        """
        expressions = []
        # if the input is not a list, we convert it to one
        if not isinstance(inputs, list):
            inputs = [inputs]
        k = len(inputs)

        # traverse each list element and convert it to a function handle
        for input in inputs:
            # first create a string
            if isfunction(input):
                raise ValueError("Please use string representation of function!")
            elif isinstance(input, str):
                # backwards compatibility for direct lambda definition
                if ":" in input:
                    # strip lambda prefix
                    input = input.split(":")[1]
                # backwards compatibility for []-indexing
                input = input.replace("[", "_").replace("]", "")
            elif isinstance(input, (int, float)):
                input = str(input)
            elif isinstance(input, u.Quantity):
                input = str(input.to_base_units().magnitude)

            if '_' in input:
                # the temperature is input as a vector
                T = symarray("T", k)
            else:
                # the temperature is input as a scalar
                T = symbols("T")
            try:
                expressions.append(sympify(input))
            except Exception as e:
                print(
                    "String input for layer property "
                    + input
                    + " \
                    cannot be converted to function handle!"
                )
                print(e)

        return expressions


@dataclass
class ParameterGroup:
    """A group of Parameters with units."""

    def _table_representation(self, style="double_grid"):
        rows = [[f.name, getattr(self, f.name).quantity] for f in fields(self)]
        return tabulate(
            rows, headers=["Parameter", "Value"], tablefmt=style, colalign=("right", "right")
        )

    def _pretty_class_name(self):
        return "".join(
            f"{word} "
            for word in re.sub(
                "([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", self.__class__.__name__)
            ).split()
        )

    def __repr__(self):
        class_str = self._pretty_class_name() + "\n"
        class_str += self._table_representation()
        return class_str

    def _repr_html_(self):
        return f"<h3>{self._pretty_class_name()}</h3>" + self._table_representation(style="html")


@dataclass(repr=False)
class StructuralParameters(ParameterGroup):
    """Structural parameters of a layer.

    thickness (float): thickness of the layer [m].
    density (float): density of the layer [kg/m³].
    area (float): area of layer [m²].
    roughness (float): gaussian width of the top roughness of a layer [m].
    mass (float): mass of the layer [kg].
    mass_unit_area (float): mass of layer normalized to unit area of 1 Å² [kg].
    volume (float): volume of layer [m³].

    """

    thickness: Parameter = field(default_factory=lambda: Parameter("m", 0.0))
    density: Parameter = field(default_factory=lambda: Parameter("kg/m**3", 0.0))
    area: Parameter = field(default_factory=lambda: Parameter("m**2", 1.0 * u.angstrom**2))
    roughness: Parameter = field(default_factory=lambda: Parameter("m", 0.0))
    mass: Parameter = field(default_factory=lambda: Parameter("kg", 0.0))
    mass_unit_area: Parameter = field(default_factory=lambda: Parameter("kg", 0.0))
    volume: Parameter = field(default_factory=lambda: Parameter("m**3", 0.0))

    def __post_init__(self):
        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name
            if name in ["thickness", "density", "area"]:
                p._caller = self

    def _update_depending(self):
        self.volume.magnitude = self.thickness.magnitude * self.area.magnitude
        self.mass.magnitude = self.density.magnitude * self.volume.magnitude
        self.mass_unit_area.quantity = (
            self.mass.quantity / self.area.quantity * 1.0 * u.angstrom**2
        )


@dataclass(repr=False)
class LatticeParameters(ParameterGroup):
    """Lattice parameters of a unit cell.

    a_axis (float): lattice parameter a [m].
    b_axis (float): lattice parameter b [m].
    c_axis (float): lattice parameter c [m].
    """

    a_axis: Parameter = field(default_factory=lambda: Parameter("m", 0.0))
    b_axis: Parameter = field(default_factory=lambda: Parameter("m", 0.0))
    c_axis: Parameter = field(default_factory=lambda: Parameter("m", 0.0))

    def __post_init__(self):
        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name


@dataclass(repr=False)
class ThermalParameters(ParameterGroup):
    """Thermal parameters of a layer.

    therm_cond (list[@lambda]): list of T-dependent thermal conductivity
            [W/(m K)].
    lin_therm_exp (list[@lambda]): list of T-dependent linear thermal
        expansion coefficient (relative).
    heat_capacity (list[@lambda]): list of T-dependent heat capacity
        function [J/(kg K)].
    sub_system_coupling (list[@lambda]): list of coupling functions of
        different subsystems [W/m³].    
    deb_wal_fac (list[@lambda]): list of T-dependent Debye-Waller factors
                    `\langle u^2\rangle` [m²].
    num_sub_systems (int): number of subsystems for heat and phonons
        (electrons, lattice, spins, ...).

    """

    therm_cond: TemperatureParameter = field(
        default_factory=lambda: TemperatureParameter("W/(m K)", 0.0)
    )
    heat_capacity: TemperatureParameter = field(
        default_factory=lambda: TemperatureParameter("J/(kg K)", 0.0)
    )
    lin_therm_exp: TemperatureParameter = field(
        default_factory=lambda: TemperatureParameter("", 0.0)
    )
    sub_system_coupling: TemperatureParameter = field(
        default_factory=lambda: TemperatureParameter("W/m**3", 0.0)
    )

    deb_wal_fac: TemperatureParameter = field(
        default_factory=lambda: TemperatureParameter("m*3", 0.0)
    )
    num_sub_systems: Parameter = field(default_factory=lambda: Parameter("", 1))

    def __post_init__(self):
        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name

    # update number of subsystems
    # K = self.num_sub_systems
    # k = len(inputs)
    # if k != K and change_num_sub_systems:
    #     print(f'Number of subsystems changed from {K:d} to {k:d}.')
    #     self.num_sub_systems = k

@dataclass(repr=False)
class ElasticParameters(ParameterGroup):
    """Elastic and phonon parameters of a layer.

    sound_vel (float): longitudinal sound velocity in the layer [m/s].
    phonon_damping (float): damping constant of phonon propagation [kg/s].
    spring_const (ndarray[float]): spring constant of the layer [kg/s²]
        and higher orders.
    acoustic_impedance (float): acoustic impedence of a layer [kg/m/s].

    """

    sound_vel: Parameter = field(default_factory=lambda: Parameter("m/s", 0.0))
    phonon_damping: Parameter = field(default_factory=lambda: Parameter("kg/s", 0.0))
    spring_const: Parameter = field(default_factory=lambda: Parameter("kg/s**2", np.array([0.0])))
    acoustic_impedance: Parameter = field(default_factory=lambda: Parameter("kg/m/s", 0.0))

    def __post_init__(self):
        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name

    def calc_spring_const(self, mass_unit_area, thickness):
        r"""calc_spring_const

        Calculates the spring constant of the layer from the mass per unit area,
        sound velocity and thickness

        .. math:: k = m \, \left(\frac{v}{c}\right)^2

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


@dataclass(repr=False)
class OpticalParameters(ParameterGroup):
    r"""Optical adn X-ray parameters of a layer.

    opt_pen_depth (float): optical penetration depth of the layer [m].
    opt_ref_index (ndarray[float]): optical refractive index - real
        and imagenary part :math:`n + i\kappa`.
    opt_ref_index_per_strain (ndarray[float]): optical refractive
        index change per strain - real and imagenary part
        :math:`\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}`.

    """

    opt_pen_depth: Parameter = field(default_factory=lambda: Parameter("m", 0.0))
    opt_ref_index: Parameter = field(default_factory=lambda: Parameter("", 0.0))
    opt_ref_index_per_strain: Parameter = field(default_factory=lambda: Parameter("", 0.0))

    def __post_init__(self):
        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name

    # def set_opt_pen_depth_from_ref_index(self, wavelength):
    #     """set_opt_pen_depth_from_ref_index

    #     Set the optical penetration depth from the optical referactive index
    #     for a given wavelength.

    #     Args:
    #         wavelength (Quantity): wavelength as Pint Quantitiy.

    #     """
    #     if np.imag(self.opt_ref_index) == 0:
    #         self.opt_pen_depth = Q_(np.inf, u.m)
    #     else:
    #         self.opt_pen_depth = wavelength/(4*np.pi*np.abs(np.imag(self.opt_ref_index)))


@dataclass(repr=False)
class MagneticParameters(ParameterGroup):
    """Magnetic parameters of a layer.

    eff_spin (float): effective spin.
    curie_temp (float): Curie temperature [K].
    mf_exch_coupling (float): mean field exchange coupling constant [m²kg/s²].
    lamda (float): intrinsic coupling to bath parameter.
    mag_moment (float): atomic magnetic moment [mu_Bohr].
    aniso_exponent(ndarray[float]): exponent of T-dependence uniaxial
        anisotropy.
    anisotropy (ndarray[float]): anisotropy at T=0 K [J/m³] as x,y,z component vector.
    exch_stiffness (float): exchange stiffness at T=0 K [J/m].
    mag_saturation (float): saturation magnetization at 0 K [J/T/m³].
    magnetization (dict[float]): magnetization amplitude, phi and
        gamma angle inherited from the atom.

    """

    eff_spin: Parameter = field(default_factory=lambda: Parameter("", 0.0))
    curie_temp: Parameter = field(default_factory=lambda: Parameter("K", 0.0))
    mf_exch_coupling: Parameter = field(default_factory=lambda: Parameter("m**2kg/s**2", 0.0))
    lamda: Parameter = field(default_factory=lambda: Parameter("", 0.0))
    mag_moment: Parameter = field(default_factory=lambda: Parameter("bohr_magneton", 0.0))
    aniso_exponent: Parameter = field(default_factory=lambda: Parameter("", 0.0))
    anisotropy: Parameter = field(default_factory=lambda: Parameter("J/m**3", 0.0))
    exch_stiffness: Parameter = field(default_factory=lambda: Parameter("J/m", 0.0))
    mag_saturation: Parameter = field(default_factory=lambda: Parameter("J/T/m**3", 0.0))
    magnetization: Parameter = field(
        default_factory=lambda: Parameter("", np.array([0.0, 0.0, 0.0]))
    )

    def __post_init__(self):
        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name
            if name in ["eff_spin", "curie_temp"]:
                p._caller = self

    def _update_depending(self):
        self.calc_mf_exchange_coupling()

    def calc_mf_exchange_coupling(self):
        r"""calc_mf_exchange_coupling

        Calculate the mean-field exchange coupling constant

        .. math:: J = \frac{3}{S_{eff}+1} k_B T_C

        """
        try:
            self.mf_exch_coupling.magnitude = (
                3
                * self.eff_spin.magnitude
                / (self.eff_spin.magnitude + 1)
                * constants.k
                * self.curie_temp.magnitude
            )
        except AttributeError:
            # on initialization self.curie_temp
            self.mf_exch_coupling.magnitude = 0


#     @property
#     def heat_capacity(self):
#         return self._heat_capacity

#     @heat_capacity.setter
#     def heat_capacity(self, heat_capacity):
#         # (re)calculate the integrated heat capacity
#         self._heat_capacity, self.heat_capacity_str = self.check_input(heat_capacity)
#         # delete last anti-derivative
#         self._int_heat_capacity = None
#         # recalculate the anti-derivative
#         self.int_heat_capacity


#     @int_heat_capacity.setter
#     def int_heat_capacity(self, int_heat_capacity):
#         self._int_heat_capacity, self.int_heat_capacity_str = self.check_input(
#                 int_heat_capacity)

#     @lin_therm_exp.setter
#     def lin_therm_exp(self, lin_therm_exp):
#         # (re)calculate the integrated linear thermal expansion coefficient
#         self._lin_therm_exp, self.lin_therm_exp_str = self.check_input(lin_therm_exp)
#         # delete last anti-derivative
#         self._int_lin_therm_exp = None
#         # recalculate the anti-derivative
#         self.int_lin_therm_exp

#     @property
#     def sub_system_coupling(self):
#         return self._sub_system_coupling

#     @sub_system_coupling.setter
#     def sub_system_coupling(self, sub_system_coupling):
#         self._sub_system_coupling, self.sub_system_coupling_str = \
#             self.check_input(sub_system_coupling)
