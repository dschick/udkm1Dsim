#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the udkm1Dsimpy module.
#
# udkm1Dsimpy is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2019 Daniel Schick

__all__ = ["Layer", "AmorphousLayer", "UnitCell"]

__docformat__ = "restructuredtext"

import numpy as np
from .atoms import Atom, AtomMixed
from inspect import isfunction
from sympy import integrate, Symbol
from sympy.utilities.lambdify import lambdify
from tabulate import tabulate
from . import u, Q_


class Layer:
    """Layer

    The layer class hold different structural properties of real
    physical layers, such as amorphous layers and unit cells.

    Args:
        id (str): id of the layer
        name (str): name of the layer

    Keyword Args:
        deb_wal_fac (float): Debye Waller factor
        sound_vel (float): sound velocity
        phonon_damping (float): phonon damping
        roughness (float): gaussian width of the top roughness of a layer
        opt_pen_depth (float): optical penetration depth
        opt_ref_index (float): refractive index
        opt_ref_index_per_strain (float): change of refractive index per
           strain
        heat_capacity (float): heat capacity
        therm_cond (float): thermal conductivity
        lin_therm_exp (float): linear thermal expansion
        sub_system_coupling (float): sub-system coupling

    Attributes:
        id (str): id of the layer
        name (str): name of the layer
        thickness (float): thickness of the layer
        roughness (float): gaussian width of the top roughness of a layer
        spring_const (ndarray[float]): spring constant of the layer
           [kg/s²] and higher orders
        opt_ref_index (ndarray[float]): optical refractive index - real
           and imagenary part :math:`n + i\kappa`
        opt_ref_index_per_strain (ndarray[float]): optical refractive
           index change per strain - real and imagenary part
           :math:`\\frac{d n}{d \eta} + i\\frac{d \kappa}{d \eta}`
        therm_cond (list[@lambda]): list of HANDLES T-dependent thermal
           conductivity [W/(m K)]
        lin_therm_exp (list[@lambda]): list of HANDLES T-dependent
           linear thermal expansion coefficient (relative)
        int_lin_therm_exp (list[@lambda]): list of HANDLES T-dependent
           integrated linear thermal expansion coefficient
        heat_capacity (list[@lambda]): list of HANDLES T-dependent heat
           capacity function [J/(kg K)]
        int_heat_capacity (list[@lambda]): list of HANDLES T-dependent
           integrated heat capacity function
        sub_system_coupling (list[@lambda]): list of HANDLES of coupling
           functions of different subsystems [W/m³]
        num_sub_systems (int): number of subsystems for heat and phonons
           (electrons, lattice, spins, ...)

    """

    def __init__(self, id, name, **kwargs):
        self.id = id
        self.name = name
        self.roughness = kwargs.get('roughness', 0*u.nm)
        self.spring_const = np.array([0])
        self.deb_wal_fac = kwargs.get('deb_wal_fac', 0*u.m**2)
        self.sound_vel = kwargs.get('sound_vel', 0*u.m/u.s)
        self.phonon_damping = kwargs.get('phonon_damping', 0*u.kg/u.s)
        self.opt_pen_depth = kwargs.get('opt_pen_depth', 0*u.nm)
        self.opt_ref_index = kwargs.get('opt_ref_index', 0)
        self.opt_ref_index_per_strain = kwargs.get('opt_ref_index_per_strain', 0)
        self.heat_capacity = kwargs.get('heat_capacity', 0)
        self.therm_cond, self.therm_cond_str = self.check_cell_array_input(
                kwargs.get('therm_cond', 0))
        self.lin_therm_exp = kwargs.get('lin_therm_exp', 0)
        self.sub_system_coupling, self.sub_system_coupling_str = self.check_cell_array_input(
                kwargs.get('sub_system_coupling', 0))

        if (len(self.heat_capacity) == len(self.therm_cond)
            and len(self.heat_capacity) == len(self.lin_therm_exp)
                and len(self.heat_capacity) == len(self.sub_system_coupling)):
            self.num_sub_systems = len(self.heat_capacity)
        else:
            raise ValueError('Heat capacity, thermal conductivity, linear'
                             'thermal expansion and subsystem coupling have not'
                             'the same number of elements!')

    def __str__(self):
        """String representation of this class"""
        output = [
                  ['area', '{:.4~P}'.format(self.area.to('nm**2'))],
                  ['volume', '{:.4~P}'.format(self.volume.to('nm**3'))],
                  ['mass', '{:4~P}'.format(self.mass)],
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
                  ['subsystem coupling', ' W/m³\n'.join(self.sub_system_coupling_str) + ' W/m³']]

        return output

    def check_cell_array_input(self, inputs):
        """ check_cell_array_input

        Checks the input for inputs which are cell arrays of function
        handles, such as the heat capacity which is a cell array of N
        function handles.

        """
        output = []
        outputStrs = []
        # if the input is not a list, we convert it to one
        if not isinstance(inputs, list):
            inputs = [inputs]
        # traverse each list element and convert it to a function handle
        for input in inputs:
            if isfunction(input):
                raise ValueError('Please use string representation of function!')
                output.append(input)
                outputStrs.append('no str representation available')
            elif isinstance(input, str):
                try:
                    output.append(eval(input))
                    outputStrs.append(input)
                except Exception as e:
                    print('String input for unit cell property ' + input + ' \
                        cannot be converted to function handle!')
                    print(e)
            elif isinstance(input, (int, float)):
                output.append(eval('lambda T: {:f}'.format(input)))
                outputStrs.append('lambda T: {:f}'.format(input))
            elif isinstance(input, object):
                output.append(eval('lambda T: {:f}'.format(input.to_base_units().magnitude)))
                outputStrs.append('lambda T: {:f}'.format(input.to_base_units().magnitude))
            else:
                raise ValueError('Unit cell property input has to be a single or '
                                 'cell array of numerics, function handles or strings which can be'
                                 'converted into a function handle!')

        return(output, outputStrs)

    def get_property_dict(self, **kwargs):
        """get_property_dict

        Returns a dictionary with all parameters. objects or dicts and
        objects are converted to strings. if a type is given, only these
        properties are returned.

        """
        # initialize input parser and define defaults and validators
        properties_by_types = {'heat': ['_thickness', '_area', '_volume', '_opt_pen_depth',
                                        'therm_cond_str', 'heat_capacity_str',
                                        'int_heat_capacity_str', 'sub_system_coupling_str',
                                        'num_sub_systems'],
                               'phonon': ['num_sub_systems', 'int_lin_therm_exp_str', '_thickness',
                                          '_mass', 'spring_const', '_phonon_damping'],
                               'xray': ['num_atoms', '_area', '_deb_wal_fac', '_thickness'],
                               'optical': ['_c_axis', '_opt_pen_depth', 'opt_ref_index',
                                           'opt_ref_index_per_strain'],
                               'magnetic': ['magnetization'],
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

    @property
    def thickness(self):
        """float: out-of-plane thickness [m]"""
        return Q_(self._thickness, u.meter).to('nm')

    @thickness.setter
    def thickness(self, thickness):
        """set.thickness"""
        self._thickness = thickness.to_base_units().magnitude

    @property
    def heat_capacity(self):
        """get heat_capacity

        Returns the temperature-dependent heat capactity function.

        """

        return self._heat_capacity

    @heat_capacity.setter
    def heat_capacity(self, heat_capacity):
        """set heat_capacity

        Set the heat_capacity (and its string representation)
        and calls setting its anti-derivative.

        """

        self._heat_capacity, self.heat_capacity_str = self.check_cell_array_input(heat_capacity)
        self.int_heat_capacity  # recalculate the anti-derivative

    @property
    def int_heat_capacity(self):
        """get int_heat_capacity

        Returns the anti-derrivative of the temperature-dependent heat
        :math:`c(T)` capacity function. If the int_heat_capacity
        property is not set, the symbolic integration is performed.

        """
        if hasattr(self, '_int_heat_capacity') and isinstance(self._int_heat_capacity, list):
            return self._int_heat_capacity
        else:
            self._int_heat_capacity = []
            self.int_heat_capacity_str = []
            try:
                T = Symbol('T')
                for i, hcs in enumerate(self.heat_capacity_str):
                    integral = integrate(hcs.split(':')[1], T)
                    self._int_heat_capacity.append(lambdify(T, integral))
                    self.int_heat_capacity_str.append('lambda T : ' + str(integral))

            except Exception as e:
                print('The sympy integration did not work. You can set the'
                      'analytical anti-derivative of the heat capacity'
                      'of your unit cells as lambda function of the temperature'
                      'T by typing UC.int_heat_capacity = lambda T: c(T)'
                      'where UC is the name of the unit cell object.')
                print(e)

        return self._int_heat_capacity

    @int_heat_capacity.setter
    def int_heat_capacity(self, int_heat_capacity):
        """set int_heat_capacity

        Set the integrated heat capacity manually when no sympy is
        installed.

        """
        self._int_heat_capacity, self.int_heat_capacity_str = self.check_cell_array_input(
                int_heat_capacity)

    @property
    def lin_therm_exp(self):
        """get lin_therm_exp

        Returns the temperature-dependent linear thermal expansion function.

        """

        return self._lin_therm_exp

    @lin_therm_exp.setter
    def lin_therm_exp(self, lin_therm_exp):
        """set lin_therm_exp

        Set the linear thermal expansion coefficient (and string representation)
        and calls setting its anti-derivative.

        """

        self._lin_therm_exp, self.lin_therm_exp_str = self.check_cell_array_input(lin_therm_exp)
        self.int_lin_therm_exp  # recalculate the anti-derivative

    @property
    def int_lin_therm_exp(self):
        """get int_lin_therm_exp

        Returns the anti-derrivative of the integrated
        temperature-dependent linear thermal expansion function. If the
        int_lin_therm_exp property is not set, the symbolic integration
        is performed.

        """

        self._int_lin_therm_exp = []
        self.int_lin_therm_exp_str = []
        try:
            T = Symbol('T')
            for i, ltes in enumerate(self.lin_therm_exp_str):
                integral = integrate(ltes.split(':')[1], T)
                self._int_lin_therm_exp.append(lambdify(T, integral))
                self.int_lin_therm_exp_str.append('lambda T : ' + str(integral))

        except Exception as e:
            print('The sympy integration did not work. You can set the '
                  'analytical anti-derivative of the linear thermal expansion '
                  'of your unit cells as lambda function of the temperature '
                  'T by typing UC.int_lin_therm_exp = lambda T: c(T) '
                  'where UC is the name of the unit cell object.')
            print(e)

        return self._int_lin_therm_exp

    @int_lin_therm_exp.setter
    def int_lin_therm_exp(self, int_lin_therm_exp):
        """set int_lin_therm_exp

        Set the integrated linear thermal expansion coefficient manually
        when no sympy installed.

        """
        self._int_lin_therm_exp, self.int_lin_therm_exp_str = self.check_cell_array_input(
                int_lin_therm_exp)

    def get_acoustic_impedance(self):
        """get_acoustic_impedance"""
        Z = np.sqrt(self.spring_const[0] * self.mass/self.area**2)
        return Z

    def set_ho_spring_constants(self, HO):
        """set_ho_spring_constants

        Set the higher orders of the spring constant for anharmonic
        phonon simulations.

        """
        # reset old higher order spring constants
        self.spring_const = np.delete(self.spring_const, np.r_[1:len(self.spring_const)])
        self.spring_const = np.hstack((self.spring_const, HO))

    @property
    def mass(self):
        """float: mass of unit cell normalized to area of 1 Å² [kg]"""
        return Q_(self._mass, u.kg)

    @mass.setter
    def mass(self, mass):
        """set.mass"""
        self._mass = mass.to_base_units().magnitude

    @property
    def density(self):
        """float: density of the unitCell [kg/m³]"""
        return Q_(self._density, u.kg/u.m**3)

    @density.setter
    def density(self, density):
        """set.density"""
        self._density = density.to_base_units().magnitude

    @property
    def area(self):
        """
        float: area of epitaxial unit cells need for normation for
        correct intensities) [m²]

        """
        return Q_(self._area, u.m**2)

    @area.setter
    def area(self, area):
        """set.area"""
        self._area = area.to_base_units().magnitude

    @property
    def volume(self):
        """float: volume of unit cell [m³]"""
        return Q_(self._volume, u.m**3)

    @volume.setter
    def volume(self, volume):
        """set.volume"""
        self._volume = volume.to_base_units().magnitude

    @property
    def deb_wal_fac(self):
        """float: Debye-Waller factor [m²]"""
        return Q_(self._deb_wal_fac, u.m**2)

    @deb_wal_fac.setter
    def deb_wal_fac(self, deb_wal_fac):
        """set.deb_wal_fac"""
        self._deb_wal_fac = deb_wal_fac.to_base_units().magnitude

    @property
    def sound_vel(self):
        """float: sound velocity in the unit cell [m/s]"""
        return Q_(self._sound_vel, u.m/u.s)

    @sound_vel.setter
    def sound_vel(self, sound_vel):
        """set.sound_vel
        If the sound velocity is set, the spring constant is
        (re)calculated.
        """
        self._sound_vel = sound_vel.to_base_units().magnitude
        self.calc_spring_const()

    @property
    def phonon_damping(self):
        """float: damping constant of phonon propagation [kg/s]"""
        return Q_(self._phonon_damping, u.kg/u.s)

    @phonon_damping.setter
    def phonon_damping(self, phonon_damping):
        """set.phonon_damping"""
        self._phonon_damping = phonon_damping.to_base_units().magnitude

    @property
    def opt_pen_depth(self):
        """
        float: penetration depth for pump always for 1st subsystem
        light in the unit cell [m]

        """
        return Q_(self._opt_pen_depth, u.meter).to('nanometer')

    @opt_pen_depth.setter
    def opt_pen_depth(self, opt_pen_depth):
        """set.opt_pen_depth"""
        self._opt_pen_depth = opt_pen_depth.to_base_units().magnitude

    @property
    def roughness(self):
        """float: roughness of the top of layer [m]"""
        return Q_(self._roughness, u.meter).to('nm')

    @roughness.setter
    def roughness(self, roughness):
        """set.roughness"""
        self._roughness = roughness.to_base_units().magnitude


class AmorphousLayer(Layer):
    """AmorphousLayer

    The AmorphousLayer class hold different structural properties of real
    physical amorphous layers and also an array of atoms in the layer.

    Args:
        id (str): id of the AmorphousLayer
        name (str): name of AmorphousLayer
        thickness (float): thickness of the layer
        density (float): density of the layer

    Keyword Args:
        atom (object): Atom or AtomMixed in the layer
        deb_wal_fac (float): Debye Waller factor
        sound_vel (float): sound velocity
        phonon_damping (float): phonon damping
        roughness (float): gaussian width of the top roughness of a layer
        opt_pen_depth (float): optical penetration depth
        opt_ref_index (float): refractive index
        opt_ref_index_per_strain (float): change of refractive index per
           strain
        heat_capacity (float): heat capacity
        therm_cond (float): thermal conductivity
        lin_therm_exp (float): linear thermal expansion
        sub_system_coupling (float): sub-system coupling

    Attributes:
        atom (object): Atom or AtomMixed in the layer
        magnetization (dict[float]): magnetization amplitude, phi and
           gamma angle inherited from the atom

    """

    def __init__(self, id, name, thickness, density, **kwargs):
        self.thickness = thickness
        self.density = density
        self.area = 1*u.angstrom**2  # set as unit area
        self.volume = self.area*self.thickness
        self.mass = self.density*self.volume
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

    def calc_spring_const(self):
        """calc_spring_const

        Calculates the spring constant of the layer from the mass,
        sound velocity and thickness

        .. math:: k = m \, \left(\\frac{v}{c}\\right)^2

        """
        self.spring_const[0] = (self._mass * (self._sound_vel/self._thickness)**2)

    @property
    def atom(self):
        """get atom

        Returns the atom of the layer.

        """

        return self._atom

    @atom.setter
    def atom(self, atom):
        """set atom

        Set the atom of the layer and check if its of type Atom or AtomMixed.

        """
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


class UnitCell(Layer):
    """Unit Cell

    The unitCell class hold different structural properties of real
    physical unit cells and also an array of atoms at different postions
    in the unit cell.

    Args:
        id (str): id of the UnitCell
        name (str): name of the UnitCell
        c_axis (float): c-axis of the UnitCell

    Keyword Args:
        a_axis (float): a-axis of the UnitCell
        b_axis (float): b-axis of the UnitCell
        deb_wal_fac (float): Debye Waller factor
        sound_vel (float): sound velocity
        phonon_damping (float): phonon damping
        roughness (float): gaussian width of the top roughness of a layer
        opt_pen_depth (float): optical penetration depth
        opt_ref_index (float): refractive index
        opt_ref_index_per_strain (float): change of refractive index per
           strain
        heat_capacity (float): heat capacity
        therm_cond (float): thermal conductivity
        lin_therm_exp (float): linear thermal expansion
        sub_system_coupling (float): sub-system coupling

    Attributes:
        atoms (list[atom, @lambda]): list of atoms and funtion handle
           for strain dependent displacement
        num_atoms (int): number of atoms in unit cell
        magnetization (list[foat]): magnetization amplitutes, phi, and
           gamma angle of each atom in the unit cell

    """

    def __init__(self, id, name, c_axis, **kwargs):
        self.c_axis = c_axis
        self.thickness = c_axis
        self.a_axis = kwargs.get('a_axis', self.c_axis)
        self.b_axis = kwargs.get('b_axis', self.a_axis)
        self.mass = 0*u.kg
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

        Todo: use the avogadro project as plugin
        Todo: create unit cell from CIF file e.g. by xrayutilities
        plugin.

        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cmx

        strains = kwargs.get('strains', 0)
        if not isinstance(strains, np.ndarray):
            strains = np.array([strains])

        colors = [cmx.Dark2(x) for x in np.linspace(0, 1, self.num_atoms)]
        atom_ids = self.get_atom_ids()

        for strain in strains:
            plt.figure()
            atoms_plotted = np.zeros_like(atom_ids)
            for j in range(self.num_atoms):
                if not atoms_plotted[atom_ids.index(self.atoms[j][0].id)]:
                    label = self.atoms[j][0].id
                    atoms_plotted[atom_ids.index(self.atoms[j][0].id)] = True
                    plt.plot(1+j, self.atoms[j][1](strain), 'o',
                             MarkerSize=10,
                             markeredgecolor=[0, 0, 0],
                             markerfaceColor=colors[atom_ids.index(self.atoms[j][0].id)],
                             label=label)
                else:
                    label = '_nolegend_'
                    plt.plot(1+j, self.atoms[j][1](strain), 'o',
                             MarkerSize=10,
                             markeredgecolor=[0, 0, 0],
                             markerfaceColor=colors[atom_ids.index(self.atoms[j][0].id)],
                             label=label)

            plt.axis([0.1, self.num_atoms+0.9, -0.1, (1.1+np.max(strains))])
            plt.grid(True)

            plt.title('Strain: {:0.2f}%'.format(strain))
            plt.ylabel('relative Position')
            plt.xlabel('# Atoms')
            plt.legend()
            plt.show()

    def add_atom(self, atom, position):
        """ add_atom

        Adds an atomBase/atomMixed at a relative position of the unit
        cell.

        Sort the list of atoms by the position at zero strain.

        Update the mass, density and spring constant of the unit cell
        automatically:

        .. math:: \kappa = m \cdot (v_s / c)^2

        """
        position_str = ''
        # test the input type of the position
        if isfunction(position):
            raise ValueError('Please use string representation of function!')
            pass
        elif isinstance(position, str):
            try:
                position_str = position
                position = eval(position)
            except Exception as e:
                print('String input for unit cell property ' + position + ' \
                    cannot be converted to function handle!')
                print(e)
        elif isinstance(position, (int, float)):
            position_str = 'lambda strain: {:e}*(strain+1)'.format(position)
            position = eval(position_str)
        else:
            raise ValueError('Atom position input has to be a scalar, or string'
                             'which can be converted into a lambda function!')

        # add the atom at the end of the array
        self.atoms.append([atom, position, position_str])
        # sort list of atoms by position at zero strain
        self.atoms.sort(key=lambda x: x[1](0))
        # increase the number of atoms
        self.num_atoms = self.num_atoms + 1

        self.magnetization.append([atom.mag_amplitude, atom._mag_phi, atom._mag_gamma])

        self.mass = 0*u.kg
        for i in range(self.num_atoms):
            self.mass = self.mass + self.atoms[i][0].mass

        self.density = self.mass / self.volume
        # set mass per unit area (do not know if necessary)
        self.mass = self.mass * 1*u.angstrom**2 / self.area
        self.calc_spring_const()

    def add_multiple_atoms(self, atom, position, Nb):
        """add_multiple_atoms

        Adds multiple atomBase/atomMixed at a relative position of the
        unit cell.

        """
        for i in range(Nb):
            self.addAtom(atom, position)

    def get_atom_ids(self):
        """get_atom_ids

        Returns a cell array of all atom ids in the unit cell.

        """
        ids = []
        for i in range(self.num_atoms):
            if not self.atoms[i][0].id in ids:
                ids.append(self.atoms[i][0].id)

        return ids

    def get_atom_positions(self, *args):
        """get_atom_positions

        Returns a vector of all relative postion of the atoms in the
        unit cell.

        """
        if args:
            strain = args[0]
        else:
            strain = 0

        res = np.zeros([self.num_atoms])
        for i, atom in enumerate(self.atoms):
            res[i] = atom[1](strain)

        return res

    def calc_spring_const(self):
        """calc_spring_const

        Calculates the spring constant of the unit cell from the mass,
        sound velocity and c-axis

        .. math:: k = m \, \left(\\frac{v}{c}\\right)^2

        """
        self.spring_const[0] = (self._mass * (self._sound_vel/self._c_axis)**2)

    @property
    def a_axis(self):
        """float: in-plane a-axis [m]"""
        return Q_(self._a_axis, u.meter).to('nm')

    @a_axis.setter
    def a_axis(self, a_axis):
        """set.a_axis"""
        self._a_axis = a_axis.to_base_units().magnitude

    @property
    def b_axis(self):
        """float: in-plane b-axis [m]"""
        return Q_(self._b_axis, u.meter).to('nm')

    @b_axis.setter
    def b_axis(self, b_axis):
        """set.a_axis"""
        self._b_axis = b_axis.to_base_units().magnitude

    @property
    def c_axis(self):
        """float: out-of-plane c-axis [m]"""
        return Q_(self._c_axis, u.meter).to('nm')

    @c_axis.setter
    def c_axis(self, c_axis):
        """set.c_axis"""
        self._c_axis = c_axis.to_base_units().magnitude
