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
    'Parameter',
    'ParameterGroup',
    'StructuralParameters',
    'LatticeParameters',
    'ThermalParameters',
    'ElasticParameters',
    'OpticalParameters',
    'MagneticParameters',
]

__docformat__ = 'restructuredtext'

import re
from dataclasses import dataclass, field, fields

import numpy as np
import pint
import scipy.constants as constants
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


@dataclass
class ParameterGroup:
    """A group of Parameters with units."""

    def _table_representation(self, style="double_grid"):
        rows = [[f.name, getattr(self, f.name).quantity] for f in fields(self)]
        return tabulate(rows, headers=["Parameter", "Value"], tablefmt=style,
                        colalign=("right", "right"))

    def _pretty_class_name(self):
        return "".join(f"{word} "
                       for word in re.sub('([A-Z][a-z]+)', r' \1',
                                          re.sub('([A-Z]+)', r' \1',
                                                 self.__class__.__name__)).split())

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
    area: Parameter = field(default_factory=lambda: Parameter("m**2", 1.0*u.angstrom**2))
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
        self.mass_unit_area.quantity = self.mass.quantity / self.area.quantity * 1.0*u.angstrom**2


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
    int_lin_therm_exp (list[@lambda]): list of T-dependent integrated
        linear thermal expansion coefficient.
    heat_capacity (list[@lambda]): list of T-dependent heat capacity
        function [J/(kg K)].
    int_heat_capacity (list[@lambda]): list of T-dependent integrated heat
        capacity function.
    sub_system_coupling (list[@lambda]): list of coupling functions of
        different subsystems [W/m³].
    num_sub_systems (int): number of subsystems for heat and phonons
        (electrons, lattice, spins, ...).

    """

    therm_cond: Parameter = field(default_factory=lambda: Parameter("W/(m K)", 0.0))
    heat_capacity: Parameter = field(default_factory=lambda: Parameter("J/(kg K)", 0.0))
    lin_therm_exp: Parameter = field(default_factory=lambda: Parameter("", 0.0))
    int_lin_therm_exp: Parameter = field(default_factory=lambda: Parameter("", 0.0))
    int_heat_capacity: Parameter = field(default_factory=lambda: Parameter("J/(kg K)", 0.0))
    sub_system_coupling: Parameter = field(default_factory=lambda: Parameter("W/m**3", 0.0))
    num_sub_systems: Parameter = field(default_factory=lambda: Parameter("", 1))

    def __post_init__(self):
        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name


@dataclass(repr=False)
class ElasticParameters(ParameterGroup):
    """Elastic and phonon parameters of a layer.

    sound_vel (float): longitudinal sound velocity in the layer [m/s].
    spring_const (ndarray[float]): spring constant of the layer [kg/s²]
        and higher orders.
    phonon_damping (float): damping constant of phonon propagation [kg/s].

    """

    sound_vel: Parameter = field(default_factory=lambda: Parameter("m/s", 0.0))
    phonon_damping: Parameter = field(default_factory=lambda: Parameter("kg/s", 0.0))
    spring_const: Parameter = field(default_factory=lambda: Parameter("kg/s**2",
                                                                      np.array([0.0])))

    def __post_init__(self):
        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name
            if name in ["thickness", "density", "area"]:
                p._caller = self

    def calc_spring_const(self, mass_unit_area, thickness):
        r"""calc_spring_const

        Calculates the spring constant of the layer from the mass per unit area,
        sound velocity and thickness

        .. math:: k = m \, \left(\frac{v}{c}\right)^2

        """
        try:
            self.spring_const.magnitude[0] = (mass_unit_area
                                              * (self.sound_vel.magnitude/thickness)**2)
        except (ZeroDivisionError, AttributeError):
            # no mass set, yet
            self.spring_const.magnitude[0] = 0


@dataclass(repr=False)
class OpticalParameters(ParameterGroup):
    r"""Optical adn X-ray parameters of a layer.

    opt_pen_depth (float): optical penetration depth of the layer [m].
    opt_ref_index (ndarray[float]): optical refractive index - real
        and imagenary part :math:`n + i\kappa`.
    opt_ref_index_per_strain (ndarray[float]): optical refractive
        index change per strain - real and imagenary part
        :math:`\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}`.
    deb_wal_fac (list[@lambda]): list of T-dependent Debye-Waller factors
                `\langle u^2\rangle` [m²].

    """

    opt_pen_depth: Parameter = field(default_factory=lambda: Parameter("m", 0.0))
    opt_ref_index: Parameter = field(default_factory=lambda: Parameter("", 0.0))
    opt_ref_index_per_strain: Parameter = field(
        default_factory=lambda: Parameter("", 0.0))
    deb_wal_fac: Parameter = field(default_factory=lambda: Parameter("m²", 0.0))

    def __post_init__(self):
        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name


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
        default_factory=lambda: Parameter("", np.array([0.0, 0.0, 0.0])))

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
            self.mf_exch_coupling.magnitude = 3*self.eff_spin.magnitude \
                / (self.eff_spin.magnitude+1)*constants.k*self.curie_temp.magnitude
        except AttributeError:
            # on initialization self.curie_temp
            self.mf_exch_coupling.magnitude = 0



# @property
#     def thickness(self):
#         return Q_(self._thickness, u.meter).to('nm')

#     @thickness.setter
#     def thickness(self, thickness):
#         self._thickness = thickness.to_base_units().magnitude

#     @property
#     def mass(self):
#         return Q_(self._mass, u.kg)

#     @mass.setter
#     def mass(self, mass):
#         self._mass = mass.to_base_units().magnitude

#     @property
#     def mass_unit_area(self):
#         return Q_(self._mass_unit_area, u.kg)

#     @mass_unit_area.setter
#     def mass_unit_area(self, mass_unit_area):
#         self._mass_unit_area = mass_unit_area.to_base_units().magnitude

#     @property
#     def density(self):
#         return Q_(self._density, u.kg/u.m**3)

#     @density.setter
#     def density(self, density):
#         self._density = density.to_base_units().magnitude

#     @property
#     def area(self):
#         return Q_(self._area, u.m**2)

#     @area.setter
#     def area(self, area):
#         self._area = area.to_base_units().magnitude

#     @property
#     def volume(self):
#         return Q_(self._volume, u.m**3)

#     @volume.setter
#     def volume(self, volume):
#         self._volume = volume.to_base_units().magnitude

#     @property
#     def deb_wal_fac(self):
#         return self._deb_wal_fac

#     @deb_wal_fac.setter
#     def deb_wal_fac(self, deb_wal_fac):
#         self._deb_wal_fac, self.deb_wal_fac_str = self.check_input(deb_wal_fac, False)

#     @property
#     def sound_vel(self):
#         return Q_(self._sound_vel, u.m/u.s)

#     @sound_vel.setter
#     def sound_vel(self, sound_vel):
#         # spring constants are (re)calculated on setting the sound velocity
#         self._sound_vel = sound_vel.to_base_units().magnitude
#         self.calc_spring_const()

#     @property
#     def phonon_damping(self):
#         return Q_(self._phonon_damping, u.kg/u.s)

#     @phonon_damping.setter
#     def phonon_damping(self, phonon_damping):
#         self._phonon_damping = phonon_damping.to_base_units().magnitude

#     @property
#     def opt_pen_depth(self):
#         return Q_(self._opt_pen_depth, u.meter).to('nanometer')

#     @opt_pen_depth.setter
#     def opt_pen_depth(self, opt_pen_depth):
#         self._opt_pen_depth = opt_pen_depth.to_base_units().magnitude

#     @property
#     def roughness(self):
#         return Q_(self._roughness, u.meter).to('nm')

#     @roughness.setter
#     def roughness(self, roughness):
#         self._roughness = roughness.to_base_units().magnitude

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

#     @property
#     def therm_cond(self):
#         return self._therm_cond

#     @therm_cond.setter
#     def therm_cond(self, therm_cond):
#         self._therm_cond, self.therm_cond_str = self.check_input(therm_cond)

#     @property
#     def int_heat_capacity(self):
#         if hasattr(self, '_int_heat_capacity') and isinstance(self._int_heat_capacity, list):
#             return self._int_heat_capacity
#         else:
#             self._int_heat_capacity = []
#             self.int_heat_capacity_str = []
#             T = symbols('T')
#             for hc, hcs in zip(self.heat_capacity, self.heat_capacity_str):
#                 try:
#                     integral = integrate(hcs, T)
#                     self._int_heat_capacity.append(lambdify(T, integral, modules='numpy'))
#                     self.int_heat_capacity_str.append(str(integral))
#                 except Exception:
#                     warnings.warn('\nSympy\'s analytical integration of the heat capacity '
#                                   'did not work.\n'
#                                   'Just do it numerically with scipy.integrate.quad')
#                     self._int_heat_capacity.append(lambda T: quad(hc, 0, T, limit=10000)[0])
#                     self.int_heat_capacity_str.append(f'scipy.integrate.quad({hcs:s}, 0, T)[0]')

#         return self._int_heat_capacity

#     @int_heat_capacity.setter
#     def int_heat_capacity(self, int_heat_capacity):
#         self._int_heat_capacity, self.int_heat_capacity_str = self.check_input(
#                 int_heat_capacity)

#     @property
#     def lin_therm_exp(self):
#         return self._lin_therm_exp

#     @lin_therm_exp.setter
#     def lin_therm_exp(self, lin_therm_exp):
#         # (re)calculate the integrated linear thermal expansion coefficient
#         self._lin_therm_exp, self.lin_therm_exp_str = self.check_input(lin_therm_exp)
#         # delete last anti-derivative
#         self._int_lin_therm_exp = None
#         # recalculate the anti-derivative
#         self.int_lin_therm_exp

#     @property
#     def int_lin_therm_exp(self):
#         if hasattr(self, '_int_lin_therm_exp') and isinstance(self._int_lin_therm_exp, list):
#             return self._int_lin_therm_exp
#         else:
#             self._int_lin_therm_exp = []
#             self.int_lin_therm_exp_str = []
#             T = symbols('T')
#             for lte, ltes in zip(self.lin_therm_exp, self.lin_therm_exp_str):
#                 try:
#                     integral = integrate(ltes, T)
#                     self._int_lin_therm_exp.append(lambdify(T, integral, modules='numpy'))
#                     self.int_lin_therm_exp_str.append(str(integral))
#                 except Exception:
#                     warnings.warn('\nSympy\'s analytical integration of the linear thermal '
#                                   'expansion did not work.\n'
#                                   'Just do it numerically with scipy.integrate.quad')
#                     self._int_lin_therm_exp.append(lambda T: quad(lte, 0, T, limit=10000)[0])
#                     self.int_lin_therm_exp_str.append(f'scipy.integrate.quad({ltes:s}, 0, T)[0]')

#         return self._int_lin_therm_exp

#     @int_lin_therm_exp.setter
#     def int_lin_therm_exp(self, int_lin_therm_exp):
#         self._int_lin_therm_exp, self.int_lin_therm_exp_str = self.check_input(
#                 int_lin_therm_exp)

#     @property
#     def sub_system_coupling(self):
#         return self._sub_system_coupling

#     @sub_system_coupling.setter
#     def sub_system_coupling(self, sub_system_coupling):
#         self._sub_system_coupling, self.sub_system_coupling_str = \
#             self.check_input(sub_system_coupling)

#     @property
#     def eff_spin(self):
#         return self._eff_spin

#     @eff_spin.setter
#     def eff_spin(self, eff_spin):
#         self._eff_spin = float(eff_spin)
#         self.calc_mf_exchange_coupling()

#     @property
#     def curie_temp(self):
#         return Q_(self._curie_temp, u.K)

#     @property
#     def mf_exch_coupling(self):
#         return Q_(self._mf_exch_coupling, u.m**2*u.kg/(u.s**2))

#     @curie_temp.setter
#     def curie_temp(self, curie_temp):
#         self._curie_temp = float(curie_temp.to_base_units().magnitude)
#         self.calc_mf_exchange_coupling()

#     @property
#     def mag_moment(self):
#         return Q_(self._mag_moment, u.A*u.m**2).to('bohr_magneton')

#     @mag_moment.setter
#     def mag_moment(self, mag_moment):
#         self._mag_moment = float(mag_moment.to_base_units().magnitude)

#     @property
#     def anisotropy(self):
#         return Q_(self._anisotropy, u.J/u.m**3)

#     @anisotropy.setter
#     def anisotropy(self, anisotropy):
#         self._anisotropy = np.zeros(3)
#         try:
#             if len(anisotropy) == 3:
#                 self._anisotropy = anisotropy.to_base_units().magnitude
#             else:
#                 warnings.warn('Anisotropy must be a scalar or vector of length 3!')
#         except TypeError:
#             self._anisotropy[0] = anisotropy.to_base_units().magnitude

#     @property
#     def exch_stiffness(self):
#         return Q_(self._exch_stiffness, u.J/u.m)

#     @exch_stiffness.setter
#     def exch_stiffness(self, exch_stiffness):
#         self._exch_stiffness = np.zeros(3)
#         try:
#             if len(exch_stiffness) == 3:
#                 self._exch_stiffness = exch_stiffness.to_base_units().magnitude
#             else:
#                 warnings.warn('Exchange stiffness must be a scalar or vector of length 3!')
#         except TypeError:
#             self._exch_stiffness[:] = exch_stiffness.to_base_units().magnitude

#     @property
#     def mag_saturation(self):
#         return Q_(self._mag_saturation, u.J/u.T/u.m**3)

#     @mag_saturation.setter
#     def mag_saturation(self, mag_saturation):
#         self._mag_saturation = float(mag_saturation.to_base_units().magnitude)

    # @property
    # def magnetization(self):
    #     return {'amplitude': self._magnetization['amplitude'],
    #             'phi': Q_(self._magnetization['phi'], u.rad).to('deg'),
    #             'gamma': Q_(self._magnetization['gamma'], u.rad).to('deg')
    #             }

    # @magnetization.setter
    # def magnetization(self, magnetization):
    #     self._magnetization = {'amplitude': magnetization['amplitude'],
    #                            'phi': magnetization['phi'].to_base_units().magnitude,
    #                            'gamma': magnetization['gamma'].to_base_units().magnitude
    #                            }

    # def check_input(self, inputs, change_num_sub_systems=True):
    #         """check_input

    #         Checks the input and create a list of function handle strings with T as
    #         argument. Inputs can be strings, floats, ints, or pint quantities.

    #         Args:
    #             inputs (list[str, int, float, Quantity]): list of strings, int, floats,
    #                 or Pint quantities.
    #             change_num_sub_systems (boolean, optional): wheather the number of
    #                 sub-systems should be changed. Defaults to True.

    #         Returns:
    #             (tuple):
    #             - *output (list[@lambda])* - list of lambda functions.
    #             - *output_strs (list[str])* - list of string-representations.

    #         """
    #         output = []
    #         output_strs = []
    #         # if the input is not a list, we convert it to one
    #         if not isinstance(inputs, list):
    #             inputs = [inputs]
    #         # update number of subsystems
    #         K = self.num_sub_systems
    #         k = len(inputs)
    #         if k != K and change_num_sub_systems:
    #             print(f'Number of subsystems changed from {K:d} to {k:d}.')
    #             self.num_sub_systems = k

    #         # traverse each list element and convert it to a function handle
    #         for input in inputs:
    #             T = symbols('T')
    #             if isfunction(input):
    #                 raise ValueError('Please use string representation of function!')
    #             elif isinstance(input, str):
    #                 try:
    #                     # backwards compatibility for direct lambda definition
    #                     if ':' in input:
    #                         # strip lambda prefix
    #                         input = input.split(':')[1]
    #                     # backwards compatibility for []-indexing
    #                     input = input.replace('[', '_').replace(']', '')
    #                     # check for presence of indexing and use symarray as argument
    #                     if '_' in input:
    #                         T = symarray('T', k)
    #                         output.append(lambdify([T], input, modules='numpy'))
    #                     else:
    #                         output.append(lambdify(T, input, modules='numpy'))
    #                     output_strs.append(input.strip())
    #                 except Exception as e:
    #                     print('String input for layer property ' + input + ' \
    #                         cannot be converted to function handle!')
    #                     print(e)
    #             elif isinstance(input, (int, float)):
    #                 output.append(lambdify(T, input, modules='numpy'))
    #                 output_strs.append(str(float(input)))
    #             elif isinstance(input, object):
    #                 output.append(lambdify(T, input.to_base_units().magnitude, modules='numpy'))
    #                 output_strs.append(str(float(input.to_base_units().magnitude)))
    #             else:
    #                 raise ValueError('Layer property input has to be a single or '
    #                                  'list of numerics, Quantities, or function handle strings '
    #                                  'which can be converted into a lambda function!')

    #         return output, output_strs
