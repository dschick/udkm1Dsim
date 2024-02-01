#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

__all__ = ['Layer', 'AmorphousLayer', 'UnitCell']

__docformat__ = 'restructuredtext'

from .atoms import Atom, AtomMixed
from .. import u, Q_
import numpy as np
from inspect import isfunction
from sympy import integrate, lambdify, symbols, symarray
from tabulate import tabulate
import scipy.constants as constants
import warnings


class Layer:
    r"""Layer

    Base class of real physical layers, such as amorphous layers and unit cells.

    It holds different structural, thermal, and elastic properties that are
    relevant for simulations.

    Args:
        id (str): id of the layer.
        name (str): name of the layer.

    Keyword Args:
        roughness (float): gaussian width of the top roughness of a layer.
        deb_wal_fac (float): Debye Waller factor.
        sound_vel (float): sound velocity.
        phonon_damping (float): phonon damping.
        opt_pen_depth (float): optical penetration depth.
        opt_ref_index (float): refractive index.
        opt_ref_index_per_strain (float): change of refractive index per strain.
        heat_capacity (float): heat capacity.
        therm_cond (float): thermal conductivity.
        lin_therm_exp (float): linear thermal expansion.
        sub_system_coupling (float): sub-system coupling.

    Attributes:
        id (str): id of the layer.
        name (str): name of the layer.
        thickness (float): thickness of the layer [m].
        mass (float): mass of the layer [kg].
        mass_unit_area (float): mass of layer normalized to unit area of 1 Å² [kg].
        density (float): density of the layer [kg/m³].
        area (float): area of layer [m²].
        volume (float): volume of layer [m³].
        roughness (float): gaussian width of the top roughness of a layer [m].
        deb_wal_fac (float): Debye-Waller factor [m²].
        sound_vel (float): longitudinal sound velocity in the layer [m/s].
        spring_const (ndarray[float]): spring constant of the layer [kg/s²]
            and higher orders.
        phonon_damping (float): damping constant of phonon propagation [kg/s].
        opt_pen_depth (float): optical penetration depth of the layer [m].
        opt_ref_index (ndarray[float]): optical refractive index - real
           and imagenary part :math:`n + i\kappa`.
        opt_ref_index_per_strain (ndarray[float]): optical refractive
           index change per strain - real and imagenary part
           :math:`\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}`.
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
        eff_spin (float): effective spin.
        curie_temp (float): Curie temperature [K].
        mf_exch_coupling (float): mean field exchange coupling constant [m²kg/s²].
        lamda (float): intrinsic coupling to bath parameter.
        mag_moment (float): atomic magnetic moment [mu_Bohr].
        aniso_exponent(ndarray[float]): exponent of T-dependence uniaxial
            anisotropy.
        anisotropy (float): anisotropy at T=0 K [J/m³] as x,y,z component vector.
        exch_stiffness (float): exchange stiffness at T=0 K [J/m].
        mag_saturation (float): saturation magnetization at 0 K [J/T/m³].

    """

    def __init__(self, id, name, **kwargs):
        self.id = id
        self.name = name
        self.num_sub_systems = 1
        self.roughness = kwargs.get('roughness', 0*u.nm)
        self.spring_const = np.array([0.0])
        self.deb_wal_fac = kwargs.get('deb_wal_fac', 0*u.m**2)
        self.sound_vel = kwargs.get('sound_vel', 0*u.m/u.s)
        self.phonon_damping = kwargs.get('phonon_damping', 0*u.kg/u.s)
        self.opt_pen_depth = kwargs.get('opt_pen_depth', 0*u.nm)
        self.opt_ref_index = kwargs.get('opt_ref_index', 0)
        self.opt_ref_index_per_strain = kwargs.get('opt_ref_index_per_strain', 0)
        self.heat_capacity = kwargs.get('heat_capacity', 0)
        self.therm_cond = kwargs.get('therm_cond', 0)
        self.lin_therm_exp = kwargs.get('lin_therm_exp', 0)
        self.sub_system_coupling = kwargs.get('sub_system_coupling', 0)

        if len(self.heat_capacity) == len(self.therm_cond) \
                == len(self.lin_therm_exp) == len(self.sub_system_coupling):
            self.num_sub_systems = len(self.heat_capacity)
        else:
            raise ValueError('Heat capacity, thermal conductivity, linear '
                             'thermal expansion and subsystem coupling have not '
                             'the same number of elements!')

        self.eff_spin = kwargs.get('eff_spin', 0)
        self.curie_temp = kwargs.get('curie_temp', 0.0*u.K)
        self.lamda = kwargs.get('lamda', 0)
        self.mag_moment = kwargs.get('mag_moment', 0*u.bohr_magneton)
        self.aniso_exponent = kwargs.get('aniso_exponent', 0)
        self.anisotropy = kwargs.get('anisotropy', [0, 0, 0]*u.J/u.m**3)
        self.exch_stiffness = kwargs.get('exch_stiffness', 0*u.J/u.m)
        self.mag_saturation = kwargs.get('mag_saturation', 0*u.J/u.T/u.m**3)

    def __str__(self):
        """String representation of this class"""
        output = [
                  ['area', '{:.4~P}'.format(self.area.to('nm**2'))],
                  ['volume', '{:.4~P}'.format(self.volume.to('nm**3'))],
                  ['mass', '{:4~P}'.format(self.mass)],
                  ['mass per unit area', '{:4~P}'.format(self.mass_unit_area)],
                  ['density', '{:.4~P}'.format(self.density.to('kg/meter**3'))],
                  ['roughness', '{:.4~P}'.format(self.roughness.to('nm'))],
                  ['Debye Waller Factor', self.deb_wal_fac.to('meter**2')],
                  ['sound velocity', '{:.4~P}'.format(self.sound_vel.to('meter/s'))],
                  ['spring constant', self.spring_const * u.kg/u.s**2],
                  ['phonon damping', self.phonon_damping.to('kg/s')],
                  ['opt. pen. depth', self.opt_pen_depth.to('nm')],
                  ['opt. refractive index', self.opt_ref_index],
                  ['opt. ref. index/strain', self.opt_ref_index_per_strain],
                  ['thermal conduct.', ' W/(m K)\n'.join(self.therm_cond_str) + ' W/(m K)'],
                  ['linear thermal expansion', '\n'.join(self.lin_therm_exp_str)],
                  ['heat capacity', ' J/(kg K)\n'.join(self.heat_capacity_str) + ' J/(kg K)'],
                  ['subsystem coupling', ' W/m³\n'.join(self.sub_system_coupling_str) + ' W/m³'],
                  ['effective spin', self.eff_spin],
                  ['Curie temperature', '{:.4~P}'.format(self.curie_temp.to('K'))],
                  ['mean-field exch. coupling', self.mf_exch_coupling*u.m**2*u.kg/u.s**2],
                  ['coupling to bath parameter', self.lamda],
                  ['atomic magnetic moment', '{:.4~P}'.format(self.mag_moment.to(
                      'bohr_magneton'))],
                  ['uniaxial anisotropy exponent', self.aniso_exponent],
                  ['anisotropy', self.anisotropy],
                  ['exchange stiffness', self.exch_stiffness],
                  ['saturation magnetization', self.mag_saturation],
                ]

        return output

    def check_input(self, inputs):
        """check_input

        Checks the input and create a list of function handle strings with T as
        argument. Inputs can be strings, floats, ints, or pint quantaties.

        Args:
            inputs (list[str, int, float, Quantity]): list of strings, int, floats,
                or Pint quantities.

        Returns:
            (tuple):
            - *output (list[@lambda])* - list of lambda functions.
            - *output_strs (list[str])* - list of string-representations.

        """
        output = []
        output_strs = []
        # if the input is not a list, we convert it to one
        if not isinstance(inputs, list):
            inputs = [inputs]
        # update number of subsystems
        K = self.num_sub_systems
        k = len(inputs)
        if k != K:
            print('Number of subsystems changed from {:d} to {:d}.'.format(K, k))
            self.num_sub_systems = k

        # traverse each list element and convert it to a function handle
        for input in inputs:
            T = symbols('T')
            if isfunction(input):
                raise ValueError('Please use string representation of function!')
            elif isinstance(input, str):
                try:
                    # backwards compatibility for direct lambda definition
                    if ':' in input:
                        # strip lambda prefix
                        input = input.split(':')[1]
                    # backwards compatibility for []-indexing
                    input = input.replace('[', '_').replace(']', '')
                    # check for presence of indexing and use symarray as argument
                    if '_' in input:
                        T = symarray('T', k)
                        output.append(lambdify([T], input, modules='numpy'))
                    else:
                        output.append(lambdify(T, input, modules='numpy'))
                    output_strs.append(input.strip())
                except Exception as e:
                    print('String input for layer property ' + input + ' \
                        cannot be converted to function handle!')
                    print(e)
            elif isinstance(input, (int, float)):
                output.append(lambdify(T, input, modules='numpy'))
                output_strs.append(str(input))
            elif isinstance(input, object):
                output.append(lambdify(T, input.to_base_units().magnitude, modules='numpy'))
                output_strs.append(str(input.to_base_units().magnitude))
            else:
                raise ValueError('Layer property input has to be a single or '
                                 'list of numerics, Quantities, or function handle strings '
                                 'which can be converted into a lambda function!')

        return output, output_strs

    def get_property_dict(self, **kwargs):
        """get_property_dict

        Returns a dictionary with all parameters. objects or dicts and
        objects are converted to strings. if a type is given, only these
        properties are returned.

        Args:
            **kwargs (list[str]): types of requested properties.

        Returns:
            R (dict): dictionary with requested properties.

        """
        # initialize input parser and define defaults and validators
        properties_by_types = {'heat': ['_thickness', '_mass_unit_area', '_density',
                                        '_opt_pen_depth', 'opt_ref_index',
                                        'therm_cond_str', 'heat_capacity_str',
                                        'int_heat_capacity_str', 'sub_system_coupling_str',
                                        'num_sub_systems'],
                               'phonon': ['num_sub_systems', 'int_lin_therm_exp_str', '_thickness',
                                          '_mass_unit_area', 'spring_const', '_phonon_damping'],
                               'xray': ['num_atoms', '_area', '_mass', '_deb_wal_fac',
                                        '_thickness'],
                               'optical': ['_c_axis', '_opt_pen_depth', 'opt_ref_index',
                                           'opt_ref_index_per_strain'],
                               'magnetic': ['_thickness', 'magnetization', 'eff_spin',
                                            '_curie_temp', '_aniso_exponents', '_anisotropy',
                                            '_exch_stiffness', '_mag_saturation', 'lamda'],
                               }

        types = (kwargs.get('types', 'all'))
        if not type(types) is list:
            types = [types]
        attrs = vars(self)
        R = {}
        for t in types:
            # define the property names by the given type
            if t == 'all':
                return attrs
            else:
                S = dict((key, value) for key, value in attrs.items()
                         if key in properties_by_types[t])
                R.update(S)

        return R

    def get_acoustic_impedance(self):
        """get_acoustic_impedance

        Calculates the acoustic impedance.

        Returns:
            Z (float): acoustic impedance.

        """
        Z = np.sqrt(self.spring_const[0] * self.mass/self.area**2)
        return Z

    def set_ho_spring_constants(self, HO):
        """set_ho_spring_constants

        Set the higher orders of the spring constant for anharmonic
        phonon simulations.

        Args:
            HO (ndarray[float]): higher order spring constants.

        """
        # reset old higher order spring constants
        self.spring_const = np.delete(self.spring_const, np.r_[1:len(self.spring_const)])
        self.spring_const = np.hstack((self.spring_const, HO))

    def set_opt_pen_depth_from_ref_index(self, wavelength):
        """set_opt_pen_depth_from_ref_index

        Set the optical penetration depth from the optical referactive index
        for a given wavelength.

        Args:
            wavelength (Quantity): wavelength as Pint Quantitiy.

        """
        if np.imag(self.opt_ref_index) == 0:
            self.opt_pen_depth = Q_(np.inf, u.m)
        else:
            self.opt_pen_depth = wavelength/(4*np.pi*np.abs(np.imag(self.opt_ref_index)))

    def calc_spring_const(self):
        r"""calc_spring_const

        Calculates the spring constant of the layer from the mass per unit area,
        sound velocity and thickness

        .. math:: k = m \, \left(\frac{v}{c}\right)^2

        """
        self.spring_const[0] = (self._mass_unit_area * (self._sound_vel/self._thickness)**2)

    def calc_mf_exchange_coupling(self):
        r"""calc_mf_exchange_coupling

        Calculate the mean-field exchange coupling constant

        .. math:: J = \frac{3}{S_{eff}+1} k_B T_C

        """
        try:
            self.mf_exch_coupling = 3*self.eff_spin/(self.eff_spin+1)*constants.k*self._curie_temp
        except AttributeError:
            # on initialization self._curie_temp
            self.mf_exch_coupling = 0

    @property
    def thickness(self):
        return Q_(self._thickness, u.meter).to('nm')

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness.to_base_units().magnitude

    @property
    def mass(self):
        return Q_(self._mass, u.kg)

    @mass.setter
    def mass(self, mass):
        self._mass = mass.to_base_units().magnitude

    @property
    def mass_unit_area(self):
        return Q_(self._mass_unit_area, u.kg)

    @mass_unit_area.setter
    def mass_unit_area(self, mass_unit_area):
        self._mass_unit_area = mass_unit_area.to_base_units().magnitude

    @property
    def density(self):
        return Q_(self._density, u.kg/u.m**3)

    @density.setter
    def density(self, density):
        self._density = density.to_base_units().magnitude

    @property
    def area(self):
        return Q_(self._area, u.m**2)

    @area.setter
    def area(self, area):
        self._area = area.to_base_units().magnitude

    @property
    def volume(self):
        return Q_(self._volume, u.m**3)

    @volume.setter
    def volume(self, volume):
        self._volume = volume.to_base_units().magnitude

    @property
    def deb_wal_fac(self):
        return Q_(self._deb_wal_fac, u.m**2)

    @deb_wal_fac.setter
    def deb_wal_fac(self, deb_wal_fac):
        self._deb_wal_fac = deb_wal_fac.to_base_units().magnitude

    @property
    def sound_vel(self):
        return Q_(self._sound_vel, u.m/u.s)

    @sound_vel.setter
    def sound_vel(self, sound_vel):
        # spring constants are (re)calculated on setting the sound velocity
        self._sound_vel = sound_vel.to_base_units().magnitude
        self.calc_spring_const()

    @property
    def phonon_damping(self):
        return Q_(self._phonon_damping, u.kg/u.s)

    @phonon_damping.setter
    def phonon_damping(self, phonon_damping):
        self._phonon_damping = phonon_damping.to_base_units().magnitude

    @property
    def opt_pen_depth(self):
        return Q_(self._opt_pen_depth, u.meter).to('nanometer')

    @opt_pen_depth.setter
    def opt_pen_depth(self, opt_pen_depth):
        self._opt_pen_depth = opt_pen_depth.to_base_units().magnitude

    @property
    def roughness(self):
        return Q_(self._roughness, u.meter).to('nm')

    @roughness.setter
    def roughness(self, roughness):
        self._roughness = roughness.to_base_units().magnitude

    @property
    def heat_capacity(self):
        return self._heat_capacity

    @heat_capacity.setter
    def heat_capacity(self, heat_capacity):
        # (re)calculate the integrated heat capacity
        self._heat_capacity, self.heat_capacity_str = self.check_input(heat_capacity)
        # delete last anti-derivative
        self._int_heat_capacity = None
        # recalculate the anti-derivative
        self.int_heat_capacity

    @property
    def therm_cond(self):
        return self._therm_cond

    @therm_cond.setter
    def therm_cond(self, therm_cond):
        self._therm_cond, self.therm_cond_str = self.check_input(therm_cond)

    @property
    def int_heat_capacity(self):
        if hasattr(self, '_int_heat_capacity') and isinstance(self._int_heat_capacity, list):
            return self._int_heat_capacity
        else:
            self._int_heat_capacity = []
            self.int_heat_capacity_str = []
            T = symbols('T')
            try:
                for hcs in self.heat_capacity_str:
                    integral = integrate(hcs, T)
                    self._int_heat_capacity.append(lambdify(T, integral, modules='numpy'))
                    self.int_heat_capacity_str.append(str(integral))
            except Exception as e:
                print('The sympy integration did not work. You can set the '
                      'analytical anti-derivative of the heat capacity '
                      'of your layer as function str of the temperature '
                      'T by typing layer.int_heat_capacity = \'c(T)\' '
                      'where layer is the name of the layer object.')
                print(e)

        return self._int_heat_capacity

    @int_heat_capacity.setter
    def int_heat_capacity(self, int_heat_capacity):
        self._int_heat_capacity, self.int_heat_capacity_str = self.check_input(
                int_heat_capacity)

    @property
    def lin_therm_exp(self):
        return self._lin_therm_exp

    @lin_therm_exp.setter
    def lin_therm_exp(self, lin_therm_exp):
        # (re)calculate the integrated linear thermal expansion coefficient
        self._lin_therm_exp, self.lin_therm_exp_str = self.check_input(lin_therm_exp)
        # delete last anti-derivative
        self._int_lin_therm_exp = None
        # recalculate the anti-derivative
        self.int_lin_therm_exp

    @property
    def int_lin_therm_exp(self):
        if hasattr(self, '_int_lin_therm_exp') and isinstance(self._int_lin_therm_exp, list):
            return self._int_lin_therm_exp
        else:
            self._int_lin_therm_exp = []
            self.int_lin_therm_exp_str = []
            T = symbols('T')
            try:
                for ltes in self.lin_therm_exp_str:
                    integral = integrate(ltes, T)
                    self._int_lin_therm_exp.append(lambdify(T, integral, modules='numpy'))
                    self.int_lin_therm_exp_str.append(str(integral))
            except Exception as e:
                print('The sympy integration did not work. You can set the '
                      'analytical anti-derivative of the linear thermal expansion '
                      'of your unit cells as lambda function of the temperature '
                      'T by typing layer.int_lin_therm_exp = \'c(T)\' '
                      'where layer is the name of the layer object.')
                print(e)

        return self._int_lin_therm_exp

    @int_lin_therm_exp.setter
    def int_lin_therm_exp(self, int_lin_therm_exp):
        self._int_lin_therm_exp, self.int_lin_therm_exp_str = self.check_input(
                int_lin_therm_exp)

    @property
    def sub_system_coupling(self):
        return self._sub_system_coupling

    @sub_system_coupling.setter
    def sub_system_coupling(self, sub_system_coupling):
        self._sub_system_coupling, self.sub_system_coupling_str = \
            self.check_input(sub_system_coupling)

    @property
    def eff_spin(self):
        return self._eff_spin

    @eff_spin.setter
    def eff_spin(self, eff_spin):
        self._eff_spin = float(eff_spin)
        self.calc_mf_exchange_coupling()

    @property
    def curie_temp(self):
        return Q_(self._curie_temp, u.K)

    @curie_temp.setter
    def curie_temp(self, curie_temp):
        self._curie_temp = float(curie_temp.to_base_units().magnitude)
        self.calc_mf_exchange_coupling()

    @property
    def mag_moment(self):
        return Q_(self._mag_moment, u.A*u.m**2).to('bohr_magneton')

    @mag_moment.setter
    def mag_moment(self, mag_moment):
        self._mag_moment = float(mag_moment.to_base_units().magnitude)

    @property
    def anisotropy(self):
        return Q_(self._anisotropy, u.J/u.m**3)

    @anisotropy.setter
    def anisotropy(self, anisotropy):
        self._anisotropy = np.zeros(3)
        try:
            if len(anisotropy) == 3:
                self._anisotropy = anisotropy.to_base_units().magnitude
            else:
                warnings.warn('Anisotropy must be a scalar or vector of length 3!')
        except TypeError:
            self._anisotropy[0] = anisotropy.to_base_units().magnitude

    @property
    def exch_stiffness(self):
        return Q_(self._exch_stiffness, u.J/u.m)

    @exch_stiffness.setter
    def exch_stiffness(self, exch_stiffness):
        self._exch_stiffness = np.zeros(3)
        try:
            if len(exch_stiffness) == 3:
                self._exch_stiffness = exch_stiffness.to_base_units().magnitude
            else:
                warnings.warn('Exchange stiffness must be a scalar or vector of length 3!')
        except TypeError:
            self._exch_stiffness[:] = exch_stiffness.to_base_units().magnitude

    @property
    def mag_saturation(self):
        return Q_(self._mag_saturation, u.J/u.T/u.m**3)

    @mag_saturation.setter
    def mag_saturation(self, mag_saturation):
        self._mag_saturation = float(mag_saturation.to_base_units().magnitude)


class AmorphousLayer(Layer):
    r"""AmorphousLayer

    Representation of amorphous layers containing an Atom or AtomMixed.

    Args:
        id (str): id of the layer.
        name (str): name of layer.
        thickness (float): thickness of the layer.
        density (float): density of the layer.

    Keyword Args:
        atom (object): Atom or AtomMixed in the layer.
        roughness (float): gaussian width of the top roughness of a layer.
        deb_wal_fac (float): Debye Waller factor.
        sound_vel (float): sound velocity.
        phonon_damping (float): phonon damping.
        roughness (float): gaussian width of the top roughness of a layer.
        opt_pen_depth (float): optical penetration depth.
        opt_ref_index (float): refractive index.
        opt_ref_index_per_strain (float): change of refractive index per
           strain.
        heat_capacity (float): heat capacity.
        therm_cond (float): thermal conductivity.
        lin_therm_exp (float): linear thermal expansion.
        sub_system_coupling (float): sub-system coupling.

    Attributes:
        id (str): id of the layer.
        name (str): name of the layer.
        thickness (float): thickness of the layer [m].
        mass (float): mass of the layer [kg].
        mass_unit_area (float): mass of layer normalized to unit area of 1 Å² [kg].
        density (float): density of the layer [kg/m³].
        area (float): area of layer [m²].
        volume (float): volume of layer [m³].
        roughness (float): gaussian width of the top roughness of a layer [m].
        deb_wal_fac (float): Debye-Waller factor [m²].
        sound_vel (float): longitudinal sound velocity in the layer [m/s].
        spring_const (ndarray[float]): spring constant of the layer [kg/s²]
            and higher orders.
        phonon_damping (float): damping constant of phonon propagation [kg/s].
        opt_pen_depth (float): optical penetration depth of the layer [m].
        opt_ref_index (ndarray[float]): optical refractive index - real
           and imagenary part :math:`n + i\kappa`.
        opt_ref_index_per_strain (ndarray[float]): optical refractive
           index change per strain - real and imagenary part
           :math:`\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}`.
        therm_cond (list[@lambda]): list of HANDLES T-dependent thermal
           conductivity [W/(m K)].
        lin_therm_exp (list[@lambda]): list of T-dependent linear thermal
           expansion coefficient (relative).
        int_lin_therm_exp (list[@lambda]): list of T-dependent integrated
           linear thermal expansion coefficient.
        heat_capacity (list[@lambda]): list of T-dependent heat capacity
           function [J/(kg K)].
        int_heat_capacity (list[@lambda]): list of T-dependent integrated heat
           capacity function.
        sub_system_coupling (list[@lambda]): list of of coupling functions of
           different subsystems [W/m³].
        num_sub_systems (int): number of subsystems for heat and phonons
           (electrons, lattice, spins, ...).
        eff_spin (float): effective spin.
        curie_temp (float): Curie temperature [K].
        mf_exch_coupling (float): mean field exchange coupling constant [m²kg/s²].
        lamda (float): intrinsic coupling to bath parameter.
        mag_moment (float): atomic magnetic moment [mu_Bohr].
        aniso_exponent(ndarray[float]): exponent of T-dependence uniaxial
            anisotropy.
        anisotropy (float): anisotropy at T=0 K [J/m³] as x,y,z component vector.
        exch_stiffness (float): exchange stiffness at T=0 K [J/m].
        mag_saturation (float): saturation magnetization at 0 K [J/T/m³].
        magnetization (dict[float]): magnetization amplitude, phi and
           gamma angle inherited from the atom.
        atom (object): Atom or AtomMixed in the layer.

    """

    def __init__(self, id, name, thickness, density, **kwargs):
        self.thickness = thickness
        self.density = density
        self.area = 1*u.angstrom**2  # set as unit area
        self.volume = self.area*self.thickness
        self.mass = self.density*self.volume
        self.mass_unit_area = self.mass
        self.atom = kwargs.get('atom', [])
        super().__init__(id, name, **kwargs)

    def __str__(self):
        """String representation of this class"""
        output = [['id', self.id],
                  ['name', self.name],
                  ['thickness', '{:.4~P}'.format(self.thickness)],
                  ]
        output += super().__str__()

        try:
            output += [['atom', self.atom.name],
                       ['magnetization', ''],
                       ['amplitude', self.magnetization['amplitude']],
                       ['phi [°]', self.magnetization['phi']],
                       ['gamma [°]', self.magnetization['gamma']], ]
        except AttributeError:
            output += [['no atom set', '']]

        class_str = 'Amorphous layer with the following properties\n\n'
        class_str += tabulate(output, headers=['parameter', 'value'], tablefmt='rst',
                              colalign=('right',), floatfmt=('.2f', '.2f'))
        return class_str

    @property
    def atom(self):
        return self._atom

    @atom.setter
    def atom(self, atom):
        if atom == []:  # no atom is set
            self.magnetization = {'amplitude': 0,
                                  'phi': 0*u.deg,
                                  'gamma': 0*u.deg,
                                  }
            return

        if not isinstance(atom, (Atom, AtomMixed)):
            raise ValueError('Class '
                             + type(atom).__name__
                             + ' is no possible atom of an amorphous layer. '
                             + 'Only Atom and AtomMixed are allowed!')
        self._atom = atom
        self.magnetization = {'amplitude': atom.mag_amplitude,
                              'phi': atom.mag_phi,
                              'gamma': atom.mag_gamma,
                              }

    @property
    def magnetization(self):
        return {'amplitude': self._magnetization['amplitude'],
                'phi': Q_(self._magnetization['phi'], u.rad).to('deg'),
                'gamma': Q_(self._magnetization['gamma'], u.rad).to('deg')
                }

    @magnetization.setter
    def magnetization(self, magnetization):
        self._magnetization = {'amplitude': magnetization['amplitude'],
                               'phi': magnetization['phi'].to_base_units().magnitude,
                               'gamma': magnetization['gamma'].to_base_units().magnitude
                               }


class UnitCell(Layer):
    r"""UnitCell

    Representation of unit cells made of one or multiple Atom or AtomMixed
    instances at defined positions.

    Args:
        id (str): id of the UnitCell.
        name (str): name of the UnitCell.
        c_axis (float): c-axis of the UnitCell.

    Keyword Args:
        a_axis (float): a-axis of the UnitCell.
        b_axis (float): b-axis of the UnitCell.
        deb_wal_fac (float): Debye Waller factor.
        sound_vel (float): sound velocity.
        phonon_damping (float): phonon damping.
        roughness (float): gaussian width of the top roughness of a layer.
        opt_pen_depth (float): optical penetration depth.
        opt_ref_index (float): refractive index.
        opt_ref_index_per_strain (float): change of refractive index per
           strain.
        heat_capacity (float): heat capacity.
        therm_cond (float): thermal conductivity.
        lin_therm_exp (float): linear thermal expansion.
        sub_system_coupling (float): sub-system coupling.

    Attributes:
        id (str): id of the layer.
        name (str): name of the layer.
        c_axis (float): out-of-plane c-axis [m].
        a_axis (float): in-plane a-axis [m].
        b_axis (float): in-plane b-axis [m].
        thickness (float): thickness of the layer [m].
        mass (float): mass of the layer [kg].
        mass_unit_area (float): mass of layer normalized to unit area of 1 Å² [kg].
        density (float): density of the layer [kg/m³].
        area (float): area of layer [m²].
        volume (float): volume of layer [m³].
        roughness (float): gaussian width of the top roughness of a layer [m].
        deb_wal_fac (float): Debye-Waller factor [m²].
        sound_vel (float): longitudinal sound velocity in the layer [m/s].
        spring_const (ndarray[float]): spring constant of the layer [kg/s²]
            and higher orders.
        phonon_damping (float): damping constant of phonon propagation [kg/s].
        opt_pen_depth (float): optical penetration depth of the layer [m].
        opt_ref_index (ndarray[float]): optical refractive index - real
           and imagenary part :math:`n + i\kappa`.
        opt_ref_index_per_strain (ndarray[float]): optical refractive
           index change per strain - real and imagenary part
           :math:`\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}`.
        therm_cond (list[@lambda]): list of HANDLES T-dependent thermal
           conductivity [W/(m K)].
        lin_therm_exp (list[@lambda]): list of T-dependent linear thermal
           expansion coefficient (relative).
        int_lin_therm_exp (list[@lambda]): list of T-dependent integrated
           linear thermal expansion coefficient.
        heat_capacity (list[@lambda]): list of T-dependent heat capacity
           function [J/(kg K)].
        int_heat_capacity (list[@lambda]): list of T-dependent integrated heat
           capacity function.
        sub_system_coupling (list[@lambda]): list of of coupling functions of
           different subsystems [W/m³].
        num_sub_systems (int): number of subsystems for heat and phonons
           (electrons, lattice, spins, ...).
        atoms (list[atom, @lambda]): list of atoms and function handle
           for strain dependent displacement.
        num_atoms (int): number of atoms in unit cell.
        eff_spin (float): effective spin.
        curie_temp (float): Curie temperature [K].
        mf_exch_coupling (float): mean field exchange coupling constant [m²kg/s²].
        lamda (float): intrinsic coupling to bath parameter.
        mag_moment (float): atomic magnetic moment [mu_Bohr].
        aniso_exponent(ndarray[float]): exponent of T-dependence uniaxial
            anisotropy.
        anisotropy (float): anisotropy at T=0 K [J/m³] as x,y,z component vector.
        exch_stiffness (float): exchange stiffness at T=0 K [J/m].
        mag_saturation (float): saturation magnetization at 0 K [J/T/m³].
        magnetization (list[float]): magnetization amplitudes, phi, and
           gamma angle of each atom in the unit cell.

    """

    def __init__(self, id, name, c_axis, **kwargs):
        self.c_axis = c_axis
        self.thickness = c_axis
        self.a_axis = kwargs.get('a_axis', self.c_axis)
        self.b_axis = kwargs.get('b_axis', self.a_axis)
        self.mass = 0*u.kg
        self.mass_unit_area = 0*u.kg
        self.density = 0*u.kg/u.m**2

        super().__init__(id, name, **kwargs)

        self.area = self.a_axis * self.b_axis
        self.volume = self.area * self.c_axis
        self.atoms = []
        self.num_atoms = 0
        self.magnetization = []

    def __str__(self):
        """String representation of this class"""
        output = [['id', self.id],
                  ['name', self.name],
                  ['a-axis', '{:.4~P}'.format(self.a_axis)],
                  ['b-axis', '{:.4~P}'.format(self.b_axis)],
                  ['c-axis', '{:.4~P}'.format(self.c_axis)],
                  ['area', '{:.4~P}'.format(self.area.to('nm**2'))],
                  ['volume', '{:.4~P}'.format(self.volume.to('nm**3'))],
                  ['mass', '{:.4~P}'.format(self.mass)],
                  ['mass per unit area', '{:.4~P}'.format(self.mass_unit_area)],
                  ]
        output += super().__str__()

        class_str = 'Unit Cell with the following properties\n\n'
        class_str += tabulate(output, headers=['parameter', 'value'], tablefmt='rst',
                              colalign=('right',), floatfmt=('.2f', '.2f'))
        class_str += '\n\n' + str(self.num_atoms) + ' Constituents:\n'

        atoms_str = []
        for i in range(self.num_atoms):
            atoms_str.append([self.atoms[i][0].name,
                              '{:0.2f}'.format(self.atoms[i][1](0)),
                              self.atoms[i][2],
                              '',
                              self.atoms[i][0].mag_amplitude,
                              self.atoms[i][0].mag_phi.magnitude,
                              self.atoms[i][0].mag_gamma.magnitude,
                              ])
        class_str += tabulate(atoms_str, headers=['atom', 'position', 'position function',
                                                  'magn.', 'amplitude', 'phi [°]', 'gamma [°]'],
                              tablefmt='rst')
        return class_str

    def visualize(self, **kwargs):
        """visualize

        Allows for 3D presentation of unit cell by allow for a & b
        coordinate of atoms.
        Also add magnetization per atom.

        Todo:
            use the avogadro project as plugin
        Todo:
            create unit cell from CIF file e.g. by xrayutilities plugin.

        Args:
            **kwargs (str): strain or magnetization for manipulating unit cell
                visualization.

        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        strains = kwargs.get('strains', 0)
        if not isinstance(strains, np.ndarray):
            strains = np.array([strains])

        colors = [cm.get_cmap('Dark2')(x) for x in np.linspace(0, 1, self.num_atoms)]
        atom_ids = self.get_atom_ids()

        for strain in strains:
            plt.figure()
            atoms_plotted = np.zeros_like(atom_ids)
            for j in range(self.num_atoms):
                if not atoms_plotted[atom_ids.index(self.atoms[j][0].id)]:
                    label = self.atoms[j][0].id
                    atoms_plotted[atom_ids.index(self.atoms[j][0].id)] = True
                    plt.plot(1+j, self.atoms[j][1](strain), 'o',
                             markersize=10,
                             markeredgecolor=[0, 0, 0],
                             markerfacecolor=colors[atom_ids.index(self.atoms[j][0].id)],
                             label=label)
                else:
                    label = '_nolegend_'
                    plt.plot(1+j, self.atoms[j][1](strain), 'o',
                             markersize=10,
                             markeredgecolor=[0, 0, 0],
                             markerfacecolor=colors[atom_ids.index(self.atoms[j][0].id)],
                             label=label)

            plt.axis([0.1, self.num_atoms+0.9, -0.1, (1.1+np.max(strains))])
            plt.grid(True)

            plt.title('Strain: {:0.2f}%'.format(strain))
            plt.ylabel('relative Position')
            plt.xlabel('# Atoms')
            plt.legend()
            plt.show()

    def add_atom(self, atom, position):
        r"""add_atom

        Adds an AtomBase/AtomMixed at a relative position of the unit cell.

        Sort the list of atoms by the position at zero strain.

        Update the mass, density and spring constant of the unit cell
        automatically:

        .. math:: \kappa = m \cdot (v_s / c)^2

        Args:
            atom (Atom, AtomMixed): Atom or AtomMixed added to unit cell.
            position (float): relative position within unit cel [0 .. 1].

        """
        s = symbols('s')
        position_str = ''
        # test the input type of the position
        if isfunction(position):
            raise ValueError('Please use string representation of function!')
        elif isinstance(position, str):
            try:
                # backwards compatibility for direct lambda definition
                if ':' in position:
                    # strip lambda prefix
                    position = position.split(':')[1]
                position_str = position.strip()
                position = lambdify(s, position, modules='numpy')
            except Exception as e:
                print('String input for unit cell property ' + position + ' \
                    cannot be converted to function handle!')
                print(e)
        elif isinstance(position, (int, float)):
            position_str = str(position)
            position = lambdify(s, position, modules='numpy')
        else:
            raise ValueError('Atom position input has to be a scalar, or string'
                             'which can be converted into a lambda function!')

        # add the atom at the end of the array
        self.atoms.append([atom, position, position_str])
        # sort list of atoms by position at zero strain
        self.atoms.sort(key=lambda x: x[1](0))
        # increase the number of atoms
        self.num_atoms = self.num_atoms + 1

        self.magnetization.append([atom.mag_amplitude, atom.mag_phi, atom.mag_gamma])

        self.mass = 0*u.kg
        for i in range(self.num_atoms):
            self.mass = self.mass + self.atoms[i][0].mass

        self.density = self.mass / self.volume
        # set mass per unit area
        self.mass_unit_area = self.mass * 1*u.angstrom**2 / self.area
        self.calc_spring_const()

    def add_multiple_atoms(self, atom, position, Nb):
        """add_multiple_atoms

        Adds multiple AtomBase/AtomMixed at a relative position of the
        unit cell.

        Args:
            atom (Atom, AtomMixed): Atom or AtomMixed added to unit cell.
            position (float): relative position within unit cel [0 .. 1].
            Nb (int): repetition of atoms.

        """
        for _ in range(int(Nb)):
            self.add_atom(atom, position)

    def get_atom_ids(self):
        """get_atom_ids

        Provides a list of atom ids within the unit cell.

        Returns:
            ids (list[str]): list of atom ids within unit cell

        """
        ids = []
        for i in range(self.num_atoms):
            if not self.atoms[i][0].id in ids:
                ids.append(self.atoms[i][0].id)

        return ids

    def get_atom_positions(self, *args):
        """get_atom_positions

        Calculates the relative positions of the atoms in the unit cell

        Returns:
            res (ndarray[float]): relative postion of the atoms within the unit
            cell.

        """
        if args:
            strain = args[0]
        else:
            strain = 0

        res = np.zeros([self.num_atoms])
        for i, atom in enumerate(self.atoms):
            res[i] = atom[1](strain)

        return res

    @property
    def a_axis(self):
        return Q_(self._a_axis, u.meter).to('nm')

    @a_axis.setter
    def a_axis(self, a_axis):
        self._a_axis = a_axis.to_base_units().magnitude

    @property
    def b_axis(self):
        return Q_(self._b_axis, u.meter).to('nm')

    @b_axis.setter
    def b_axis(self, b_axis):
        self._b_axis = b_axis.to_base_units().magnitude

    @property
    def c_axis(self):
        return Q_(self._c_axis, u.meter).to('nm')

    @c_axis.setter
    def c_axis(self, c_axis):
        self._c_axis = c_axis.to_base_units().magnitude
