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
# Copyright (C) 2017 Daniel Schick

"""A :mod:`UnitCell` module """

__all__ = ["UnitCell"]

__docformat__ = "restructuredtext"

import numpy as np
from inspect import isfunction
from sympy import integrate, Symbol
from sympy.utilities.lambdify import lambdify
import numericalunits as u
u.reset_units('SI')


class UnitCell:
    """UnitCell

    The unitCell class hold different structural properties of real physical
    unit cells and also an array of atoms at different postions in the unit
    cell.

    id (str)                        : id of the unit cell
    name (str)                      : name of the unit cell
    atoms (list[atom, @lambda])     : list of atoms and funtion handle for
                                    strain dependent displacement
    num_atoms (int)                  : number of atoms in unit cell
    a_axis (float)                   : in-plane a-axis [m]
    b_axis (float)                   : in-plane b-axis [m]
    c_axis (float)                   : out-of-plane c-axis [m]
    area  (float)                   : area of epitaxial unit cells
                                      need for normation for correct intensities) [m^2]
    volume (float)                  : volume of unit cell [m^3]
    mass (float)                    : mass of unit cell normalized to an area of 1 Ang^2 [kg]
    density (float)                 : density of the unitCell [kg/m^3]
    deb_wal_fac (float)               : Debye Waller factor <u>^2 [m^2]
    sound_vel (float)                : sound velocity in the unit cell [m/s]
    spring_const (ndarray[float])    : spring constant of the unit cell [kg/s^2] and higher orders
    phonon_damping (float)           : damping constant of phonon propagation [kg/s]
    opt_pen_depth (float)             : penetration depth for pump always for 1st subsystem
                                    light in the unit cell [m]
    opt_ref_index (ndarray[float])    : optical refractive index - real and imagenary part
                                    $n + i\kappa$
    opt_ref_index_per_strain (ndarray[float])   :
            optical refractive index change per strain -
            real and imagenary part %\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}$
    therm_cond (list[@lambda])               :
            list of HANDLES T-dependent thermal conductivity [W/(m K)]
    lin_therm_exp (list[@lambda])             :
            list of HANDLES T-dependent linear thermal expansion coefficient (relative)
    int_lin_therm_exp (list[@lambda])          :
            list of HANDLES T-dependent integrated linear thermal expansion coefficient
    heat_capacity (list[@lambda])            :
            list of HANDLES T-dependent heat capacity function [J/(kg K)]
    int_heat_capacity (list[@lambda])         :
            list of HANDLES T-dependent integrated heat capacity function
    sub_system_coupling (list[@lambda])       :
            list of HANDLES of coupling functions of different subsystems [W/m^3]
    num_sub_systems (int)                     :
            number of subsystems for heat and phonons (electrons, lattice, spins, ...)
    """

    def __init__(self, id, name, c_axis, **kwargs):
        # % initialize input parser and define defaults and validators
        # p = inputParser
        # p.addRequired('id'                      , @ischar)
        # p.addRequired('name'                    , @ischar)
        # p.addRequired('c_axis'                   , @isnumeric)
        # p.addParamValue('a_axis'                 , c_axis , @isnumeric)
        # p.addParamValue('b_axis'                 , c_axis , @isnumeric)
        # p.addParamValue('deb_wal_fac'             , 0     , @isnumeric)
        # p.addParamValue('sound_vel'              , 0     , @isnumeric)
        # p.addParamValue('phonon_damping'         , 0     , @isnumeric)
        # p.addParamValue('opt_pen_depth'           , 0     , @isnumeric)
        # p.addParamValue('opt_ref_index'           , [0,0] , @(x) (isnumeric(x) && numel(x) == 2))
        # p.addParamValue('opt_ref_index_per_strain', [0,0] , @(x) (isnumeric(x) && numel(x) == 2))
        # p.addParamValue('therm_cond'             , 0     , @(x)(isnumeric(x) ||
        # isa(x,'function_handle') || ischar(x) || iscell(x)))
        # p.addParamValue('lin_therm_exp'           , 0     , @(x)(isnumeric(x) ||
        # isa(x,'function_handle') || ischar(x) || iscell(x)))
        # p.addParamValue('heat_capacity'          , 0     , @(x)(isnumeric(x) ||
        # isa(x,'function_handle') || ischar(x) || iscell(x)))
        # p.addParamValue('sub_system_coupling'     , 0     , @(x)(isnumeric(x) ||
        # isa(x,'function_handle') || ischar(x) || iscell(x)))
        # % parse the input
        # p.parse(id,name,c_axis,varargin{:})
        # % assign parser results to object properties
        self.id = id
        self.name = name
        self.c_axis = c_axis
        self.a_axis = kwargs.get('a_axis', self.c_axis)
        self.b_axis = kwargs.get('b_axis', self.a_axis)
        self.atoms = []
        self.num_atoms = 0
        self.mass = 0
        self.density = 0
        self.spring_const = np.array([0])
        self.deb_wal_fac = kwargs.get('deb_wal_fac', 0)
        self.sound_vel = kwargs.get('sound_vel', 0)
        self.phonon_damping = kwargs.get('phonon_damping', 0)
        self.opt_pen_depth = kwargs.get('opt_pen_depth', 0)
        self.opt_ref_index = kwargs.get('opt_ref_index', 0)
        self.opt_ref_index_per_strain = kwargs.get('opt_ref_index_per_strain', 0)
        self.heat_capacity, self.heat_capacity_str = self.checkCellArrayInput(
                kwargs.get('heat_capacity', 0))
        self.therm_cond, self.therm_cond_str = self.checkCellArrayInput(kwargs.get('therm_cond', 0))
        self.lin_therm_exp, self.lin_therm_exp_str = self.checkCellArrayInput(
                kwargs.get('lin_therm_exp', 0))
        self.sub_system_coupling, self.sub_system_coupling_str = self.checkCellArrayInput(
                kwargs.get('sub_system_coupling', 0))

        if len(self.heat_capacity) == len(self.therm_cond) \
                and len(self.heat_capacity) == len(self.lin_therm_exp) \
                and len(self.heat_capacity) == len(self.sub_system_coupling):
            self.num_sub_systems = len(self.heat_capacity)
        else:
            raise ValueError('Heat capacity, thermal conductivity, linear'
                             'thermal expansion and subsystem coupling have not'
                             'the same number of elements!')

        self.area = self.a_axis * self.b_axis
        self.volume = self.area * self.c_axis

    def __str__(self):
        """String representation of this class

        """
        class_str = 'Unit Cell with the following properties\n'
        class_str += 'id                     : {:s}\n'.format(self.id)
        class_str += 'name                   : {:s}\n'.format(self.name)
        class_str += 'a-axis                 : {:3.2f} Å\n'.format(self.a_axis/u.angstrom)
        class_str += 'b-axis                 : {:3.2f} Å\n'.format(self.b_axis/u.angstrom)
        class_str += 'c-axis                 : {:3.2f} Å\n'.format(self.c_axis/u.angstrom)
        class_str += 'area                   : {:3.2f} Å²\n'.format(self.area/u.angstrom**2)
        class_str += 'volume                 : {:3.2f} Å³\n'.format(self.volume/u.angstrom**3)
        class_str += 'mass                   : {:3.2e} kg\n'.format(self.mass/u.kg)
        class_str += 'density                : {:3.2e} kg/m³\n'.format(self.density/(u.kg/u.m**3))
        class_str += 'Debye Waller Factor    : {:3.2f} m²\n'.format(self.deb_wal_fac/u.m**2)
        class_str += 'sound velocity         : {:3.2f} nm/ps\n'.format(self.sound_vel/(u.nm/u.ps))
        class_str += 'spring constant        : {:s} kg/s²\n'.format(np.array_str(
                self.spring_const/(u.kg/u.s**2)))
        class_str += 'phonon damping         : {:3.2f} kg/s\n'.format(
                self.phonon_damping/(u.kg/u.s))
        class_str += 'opt. pen. depth        : {:3.2f} nm\n'.format(self.opt_pen_depth/u.nm)
        class_str += 'opt. refractive index  : {:3.2f}\n'.format(self.opt_ref_index)
        class_str += 'opt. ref. index/strain : {:3.2f}\n'.format(self.opt_ref_index_per_strain)
        class_str += 'thermal conduct. [W/m K]       :\n'
        for func in self.therm_cond_str:
            class_str += '\t\t\t {:s}\n'.format(func)
        class_str += 'linear thermal expansion [1/K] :\n'
        for func in self.lin_therm_exp_str:
            class_str += '\t\t\t {:s}\n'.format(func)
        class_str += 'heat capacity [J/kg K]         :\n'
        for func in self.heat_capacity_str:
            class_str += '\t\t\t {:s}\n'.format(func)
        class_str += 'subsystem coupling [W/m^3]     :\n'
        for func in self.sub_system_coupling_str:
            class_str += '\t\t\t {:s}\n'.format(func)
        # display the constituents
        class_str += str(self.num_atoms) + ' Constituents:\n'
        for i in range(self.num_atoms):
            class_str += '{:s} \t {:0.2f} \t {:s}\n'.format(self.atoms[i][0].name,
                                                            self.atoms[i][1](0), self.atoms[i][2])

        return class_str

    def visualize(self, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cmx

        strains = kwargs.get('strains', 0)
        if not isinstance(strains, np.ndarray):
            strains = np.array([strains])

        colors = [cmx.Dark2(x) for x in np.linspace(0, 1, self.num_atoms)]
        atom_ids = self.getAtomIDs()

        for strain in strains:
            plt.figure()
            atoms_plotted = np.zeros_like(atom_ids)
            for j in range(self.num_atoms):
                if not atoms_plotted[atom_ids.index(self.atoms[j][0].id)]:
                    label = self.atoms[j][0].id
                    atoms_plotted[atom_ids.index(self.atoms[j][0].id)] = True
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

    def getPropertyStruct(self, **kwargs):
        """getParameterStruct

        Returns a struct with all parameters. objects or cell arrays and
        objects are converted to strings. if a type is given, only these
        properties are returned.
        """
        # initialize input parser and define defaults and validators
        types = ['all', 'heat', 'phonon', 'XRD', 'optical']
        properties_by_types = {'heat': ['c_axis', 'area', 'volume', 'opt_pen_depth',
                                        'therm_cond_str', 'heat_capacity_str',
                                        'int_heat_capacity_str', 'sub_system_coupling_str',
                                        'num_sub_systems'],
                               'phonon': ['num_sub_systems', 'int_lin_therm_exp_str', 'c_axis',
                                          'mass', 'spring_const', 'phonon_damping'],
                               'XRD': ['num_atoms', 'atoms', 'area', 'deb_wal_fac', 'c_axis'],
                               'optical': ['c_axis', 'opt_pen_depth', 'opt_ref_index',
                                           'opt_ref_index_per_strain'],
                               }

        types = kwargs.get('types', 'all')
        attrs = vars(self)
        # define the property names by the given type
        if types == 'all':
            S = attrs
        else:
            S = dict((key, value) for key, value in attrs.items()
                     if key in properties_by_types[types])

        return S

    def checkCellArrayInput(self, inputs):
        """ checkCellArrayInput

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
            else:
                raise ValueError('Unit cell property input has to be a single or'
                                 'cell array of numerics, function handles or strings which can be'
                                 'converted into a function handle!')

        return(output, outputStrs)

    @property
    def int_heat_capacity(self):
        """get int_heat_capacity

        Returns the anti-derrivative of the temperature-dependent heat
        $c(T)$ capacity function. If the _int_heat_capacity_ property is
        not set, the symbolic integration is performed.
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

        Set the integrated heat capacity manually when no Smybolic Math
        Toolbox is installed.
        """
        self._int_heat_capacity, self.int_heat_capacity_str = self.checkCellArrayInput(
                int_heat_capacity)

    @property
    def int_lin_therm_exp(self):
        """get int_lin_therm_exp

        Returns the anti-derrivative of theintegrated temperature-dependent
        linear thermal expansion function. If the __int_lin_therm_exp__
        property is not set, the symbolic integration is performed.
        """

        if hasattr(self, '_int_lin_therm_exp') and isinstance(self._int_lin_therm_exp, list):
            return self._int_lin_therm_exp
        else:
            self._int_lin_therm_exp = []
            self.int_lin_therm_exp_str = []
            try:
                T = Symbol('T')
                for i, ltes in enumerate(self.lin_therm_exp_str):
                    integral = integrate(ltes.split(':')[1], T)
                    self._int_lin_therm_exp.append(lambdify(T, integral))
                    self.int_lin_therm_exp_str.append('lambda T : ' + str(integral))

            except Exception as e:
                print('The sympy integration did not work. You can set the'
                      'the analytical anti-derivative of the heat capacity'
                      'of your unit cells as lambda function of the temperature'
                      'T by typing UC.int_heat_capacity = lambda T: c(T)'
                      'where UC is the name of the unit cell object.')
                print(e)

        return self._int_lin_therm_exp

    @int_lin_therm_exp.setter
    def int_lin_therm_exp(self, int_lin_therm_exp):
        """set int_lin_therm_exp

        Set the integrated linear thermal expansion coefficient manually
        when no Smybolic Math Toolbox is installed.
        """
        self._int_lin_therm_exp, self.int_lin_therm_exp_str = self.checkCellArrayInput(
                int_lin_therm_exp)

    def addAtom(self, atom, position):
        """ addAtom
        Adds an atomBase/atomMixed at a relative position of the unit
        cell.
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
        # increase the number of atoms
        self.num_atoms = self.num_atoms + 1
        # Update the mass, density and spring constant of the unit cell
        # automatically:
        #
        # $$ \kappa = m \cdot (v_s / c)^2 $$

        self.mass = 0
        for i in range(self.num_atoms):
            self.mass = self.mass + self.atoms[i][0].mass

        self.density = self.mass / self.volume
        # set mass per unit area (do not know if necessary)
        self.mass = self.mass * 1*u.angstrom**2 / self.area
        self.calcspring_const()

    def addMultipleAtoms(self, atom, position, Nb):
        """addMultipleAtoms

        Adds multiple atomBase/atomMixed at a relative position of the unit
        cell.
        """
        for i in range(Nb):
            self.addAtom(atom, position)

    def calcspring_const(self):
        """ calcspring_const

        Calculates the spring constant of the unit cell from the mass,
        sound velocity and c-axis

        $$ k = m \, \left(\frac{v}{c}\right)^2 $$
        """
        self.spring_const[0] = self.mass * (self.sound_vel/self.c_axis)**2

    def getAcousticImpedance(self):
        """getAcousticImpedance
        """
        Z = np.sqrt(self.spring_const[0] * self.mass/self.area**2)
        return Z

    @property
    def sound_vel(self):
        return self._sound_vel

    @sound_vel.setter
    def sound_vel(self, sound_vel):
        """set.sound_vel
        If the sound velocity is set, the spring constant is
        (re)calculated.
        """
        self._sound_vel = sound_vel
        self.calcspring_const()

    def setHOspringConstants(self, HO):
        """setHOspringConstants

        Set the higher orders of the spring constant for anharmonic
        phonon simulations.
        """
        # check if HO is column vector and transpose it in this case
        if HO.shape[0] > 1:
            HO = HO.T

        # reset old higher order spring constants
        self.spring_const = np.delete(self.spring_const, np.r_[1:len(self.spring_const)])
        self.spring_const = np.hstack((self.spring_const, HO))

    def getAtomIDs(self):
        """getAtomIDs

        Returns a cell array of all atom ids in the unit cell.
        """

        ids = []
        for i in range(self.num_atoms):
            if not self.atoms[i][0].id in ids:
                ids.append(self.atoms[i][0].id)

        return ids

    def getAtomPositions(self, *args):
        """getAtomPositions

        Returns a vector of all relative postion of the atoms in the unit
        cell.
        """

        if args:
            strain = args
        else:
            strain = 0

        res = np.zeros([self.num_atoms])
        for i, atom in enumerate(self.atoms):
            res[i] = atom[1](strain)

        return res
