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

__all__ = ["Atom", "AtomMixed"]

__docformat__ = "restructuredtext"

import os
import numpy as np
import scipy.constants as constants
from tabulate import tabulate
from . import u, Q_


class Atom:
    """Atom

    The atom class is the smallest structural unit of which one can
    build larger structures. It holds real physical properties of atoms
    defined in the attrubutes section and can return parameters and data
    necessary for different simulation types.

    Args:
        symbol (str): symbol of the atom

    Keyword Args:
        id (str): id of the atom
        ionicity (int): ionicity of the atom

    Attributes:
        symbol (str): symbol of the element
        id (str): identifier of the atom, may differ from symbol and/or
           name
        name (str): name of the element (generic)
        atomic_number_z (int): Z atomic number
        mass_number_a (float): A atomic mass number
        ionicity (int): ionicity of the atom
        mass (float): mass of the atom [kg]
        atomic_form_factor_coeff (ndarray[float]): atomic form factor
           coefficients for energy-dependent atomic form factor
        cromer_mann_coeff (ndarray[float]): cromer-mann coefficients for
           angular-dependent atomic form factor
        mag_amplitude (float): magnetization amplitude 0 .. 1
        mag_phi (float): phi angle of the magnetization [deg]
        mag_gamma (float): gamma angle of the magnetization [deg]

    References:

        .. [1] D. T. Cromer & J. B. Mann (1968). X-ray scattering
           factors computed from numerical Hartree–Fock wave functions.
           `Acta Crystallographica Section A, 24(2), 321–324.
           <http://www.doi.org/10.1107/S0567739468000550>`_
        .. [2] J. Als-Nielson, & D. McMorrow (2001).
           `Elements of Modern X-Ray Physics. New York: John Wiley &
           Sons, Ltd. <http://www.doi.org/10.1002/9781119998365>`_
        .. [3] B. L. Henke, E. M. Gullikson & J. C. Davis (1993).
           X-Ray Interactions: Photoabsorption, Scattering,
           Transmission, and Reflection at E = 50-30,000 eV, Z = 1-92.
           `Atomic Data and Nuclear Data Tables, 54(2), 181–342.
           <http://www.doi.org/10.1006/adnd.1993.1013>`_

    """

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.id = kwargs.get('id', symbol)
        self.ionicity = kwargs.get('ionicity', 0)
        self.mag_amplitude = kwargs.get('mag_amplitude', 0)
        self.mag_phi = kwargs.get('mag_phi', 0*u.deg)
        self.mag_gamma = kwargs.get('mag_gamma', 0*u.deg)

        try:
            filename = os.path.join(os.path.dirname(__file__),
                                    'parameters/elements/elements.dat')
            symbols = np.genfromtxt(filename, dtype='U2', usecols=(0))
            elements = np.genfromtxt(filename, dtype='U15, i8, f8', usecols=(1, 2, 3))
            [rowidx] = np.where(symbols == self.symbol)
            element = elements[rowidx[0]]
        except Exception as e:
            print('Cannot load element specific data from elements data file!')
            print(e)

        self.name = element[0]
        self.atomic_number_z = element[1]
        self.mass_number_a = element[2]
        self._mass = self.mass_number_a*constants.atomic_mass
        self.mass = self._mass*u.kg
        self.atomic_form_factor_coeff = self.read_atomic_form_factor_coeff()
        self.magnetic_form_factor_coeff = self.read_magnetic_form_factor_coeff()
        self.cromer_mann_coeff = self.read_cromer_mann_coeff()

    def __str__(self):
        """String representation of this class"""
        output = {'parameter': ['id', 'symbol', 'name', 'atomic number Z', 'mass number A', 'mass',
                                'ionicity', 'Cromer Mann coeff', '', '',
                                'magn. amplitude', 'magn. phi', 'magn. gamma'],
                  'value': [self.id, self.symbol, self.name, self.atomic_number_z,
                            self.mass_number_a, '{:.4~P}'.format(self.mass), self.ionicity,
                            np.array_str(self.cromer_mann_coeff[0:4]),
                            np.array_str(self.cromer_mann_coeff[4:8]),
                            np.array_str(self.cromer_mann_coeff[8:]),
                            self.mag_amplitude, self.mag_phi, self.mag_gamma]}

        return 'Atom with the following properties\n' + \
               tabulate(output, colalign=('right',), tablefmt="rst", floatfmt=('.2f', '.2f'))

    def read_atomic_form_factor_coeff(self):
        """read_atomic_form_factor_coeff

        The atomic form factor :math:`f` in dependence from the energy
        :math:`E` is read from a parameter file given by [3]_.

        """
        filename = os.path.join(os.path.dirname(__file__),
                                'parameters/atomicFormFactors/{:s}.nff'.format(
                                        self.symbol.lower()))
        try:
            f = np.genfromtxt(filename, skip_header=1)
        except Exception as e:
            print('File {:s} not found!\nMake sure the path /parameters/atomicFormFactors/ is in'
                  'your search path!',
                  filename)
            print(e)

        return f

    @u.wraps(None, (None, 'eV'), strict=False)
    def get_atomic_form_factor(self, energy):
        """get_atomic_form_factor

        Returns the complex atomic form factor

        .. math:: f(E)=f_1-i f_2

        for the energy :math:`E` [eV].

        Convention of Ref. [2]_ (p. 11, footnote) is a negative :math:`f_2`

        """
        # interpolate the real and imaginary part in dependence of E
        f1 = np.interp(energy, self.atomic_form_factor_coeff[:, 0],
                       self.atomic_form_factor_coeff[:, 1])
        f2 = np.interp(energy, self.atomic_form_factor_coeff[:, 0],
                       self.atomic_form_factor_coeff[:, 2])

        return f1 - f2*1j

    def read_cromer_mann_coeff(self):
        """read_cromer_mann_coeff

        The Cromer-Mann coefficients (Ref. [1]_) are read from a
        parameter file and are returned in the following order:

        .. math:: a_1\; a_2\; a_3\; a_4\; b_1\; b_2\; b_3\; b_4\; c

        """
        filename = os.path.join(os.path.dirname(__file__),
                                'parameters/atomicFormFactors/cromermann.txt')
        try:
            cm = np.genfromtxt(filename, skip_header=1,
                               usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
        except Exception as e:
            print('File {:s} not found!\nMake sure the path'
                  '/parameters/atomicFormFactors/ is in your search path!',
                  filename)
            print(e)

        return cm[(cm[:, 0] == self.atomic_number_z) & (cm[:, 1] == self.ionicity)][0]

    @u.wraps(None, (None, 'eV', 'm**-1'), strict=False)
    def get_cm_atomic_form_factor(self, energy, qz):
        """get_cm_atomic_form_factor

        Returns the atomic form factor :math:`f` in dependence of the
        energy :math:`E` [J] and the :math:`z`-component of the
        scattering vector :math:`q_z` [Å :math:`^{-1}`] (Ref. [1]_).
        Since the CM coefficients are fitted for :math:`q_z` in
        [Å :math:`^{-1}`] we have to convert it before!

        See Ref. [2]_ (p. 235).

        .. math:: f(q_z,E) = f_{CM}(q_z) + \delta f_1(E) -i f_2(E)

        :math:`f_{CM}(q_z)` is given in Ref. 1:

        .. math::

            f_{CM}(q_z) = \sum(a_i \, \exp(-b_i \, (q_z/4\pi)^2))+ c

        :math:`\delta f_1(E)` is the dispersion correction:

        .. math::

            \delta f_1(E) = f_1(E) - \left(\sum^4_i(a_i) + c\\right)

        Thus:

        .. math:: f(q_z,E) = \sum(a_i \, \exp(-b_i \, q_z/2\pi))
           + c + f_1(E)-i f_2(E) - \left(\sum(a_i) + c\\right)

        .. math:: f(q_z,E) = \sum(a_i \, \exp(-b_i \, q_z/2\pi))
           + f_1(E) -i f_2(E) - \sum(a_i)

        """
        # convert from 1/nm to 1/Å and to a real column vector
        qz = np.array(qz*1e10, ndmin=2)
        energy = np.array(energy, ndmin=1)
        if np.size(qz, 0) != len(energy):
            raise TypeError('qz need to have as many rows as energies!')

        f = np.zeros_like(qz, dtype=complex)

        for i, en in enumerate(energy):
            _qz = qz[i, :].reshape(-1, 1)

            f_cm = np.dot(self.cromer_mann_coeff[0:3],
                          np.exp(np.outer(-self.cromer_mann_coeff[4:7],
                                          (_qz/(4*np.pi))**2)))
            f[i, :] = f_cm + self.get_atomic_form_factor(en) -\
                np.sum(self.cromer_mann_coeff[0:3])
        return f

    def read_magnetic_form_factor_coeff(self):
        """read_magnetic_form_factor_coeff

        The magnetic form factor :math:`m` in dependence from the energy
        :math:`E` is read from a parameter file.

        """
        filename = os.path.join(os.path.dirname(__file__),
                                'parameters/magneticFormFactors/{:s}.mf'.format(
                                        self.symbol))
        try:
            f = np.genfromtxt(filename, skip_header=1)
        except Exception as e:
            print('File {:s} not found!\nMake sure the path /parameters/magneticFormFactors/ is in'
                  'your search path!',
                  filename)
            print(e)
            # return zero array
            f = np.zeros([1, 3])

        return f

    @u.wraps(None, (None, 'eV'), strict=False)
    def get_magnetic_form_factor(self, energy):
        """get_magnetic_form_factor

        Returns the complex magnetic form factor

        .. math:: m(E)=m_1-i m_2

        for the energy :math:`E` [eV].

        Convention of Ref. [2]_ (p. 11, footnote) is a negative :math:`f_2`

        """
        # interpolate the real and imaginary part in dependence of E
        m1 = np.interp(energy, self.magnetic_form_factor_coeff[:, 0],
                       self.magnetic_form_factor_coeff[:, 1])
        m2 = np.interp(energy, self.magnetic_form_factor_coeff[:, 0],
                       self.magnetic_form_factor_coeff[:, 2])

        return m1 + m2*1j

    @property
    def mag_phi(self):
        """float: phi angle of magnetization [deg]"""
        return Q_(self._mag_phi, u.rad).to('deg')

    @mag_phi.setter
    def mag_phi(self, mag_phi):
        """set.mag_phi"""
        self._mag_phi = mag_phi.to_base_units().magnitude

    @property
    def mag_gamma(self):
        """float: gamma angle of magnetization [deg]"""
        return Q_(self._mag_gamma, u.rad).to('deg')

    @mag_gamma.setter
    def mag_gamma(self, mag_gamma):
        """set.mag_gamma"""
        self._mag_gamma = mag_gamma.to_base_units().magnitude


class AtomMixed(Atom):
    """AtomMixed

    The AtomMixed class is sub class of Atom and enables mixed atoms for
    certain alloys and stochiometric mixtures. All properties of the
    included sub-atoms of class atomBase are averaged and weighted with
    their stochiometric ratio

    Args:
        symbol (str): symbol of the atom

    Keyword Args:
        id (str): id of the atom
        ionicity (int): ionicity of the atom

    Attributes:
        symbol (str): symbol of the element
        id (str): identifier of the atom, may differ from symbol and/or
           name
        name (str): name of the element (generic)
        atomic_number_z (int): Z atomic number
        mass_number_a (float): A atomic mass number
        ionicity (int): ionicity of the atom
        mass (float): mass of the atom [kg]
        atomic_form_factor_coeff (ndarray[float]): atomic form factor
           coefficients for energy-dependent atomic form factor
        cromer_mann_coeff (ndarray[float]): cromer-mann coefficients for
           angular-dependent atomic form factor

    """

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.id = kwargs.get('id', symbol)
        self.name = kwargs.get('name', symbol)
        self.ionicity = 0
        self.atomic_number_z = 0
        self.mass_number_a = 0
        self.mass = 0
        self.atoms = []
        self.num_atoms = 0
        self.cromer_mann_coeff = np.array([])

    def __str__(self):
        """String representation of this class"""
        output = []
        for i in range(self.num_atoms):
            output.append([self.atoms[i][0].name, '{:.1f} %'.format(self.atoms[i][1]*100)])

        return (super().__str__()
                + '\n{:d} Constituents:\n'.format(self.num_atoms)
                + tabulate(output, colalign=('right',), floatfmt=('.2f', '.2f')))

    def add_atom(self, atom, fraction):
        """add_atom

        Add an Atom instance with its stochiometric fraction to the AtomMixed instance.
        """
        self.atoms.append([atom, fraction])
        self.num_atoms = self.num_atoms + 1
        # calculate the mixed atomic properties of the atomMixed instance
        self.atomic_number_z = self.atomic_number_z + fraction * atom.atomic_number_z
        self.mass_number_a = self.mass_number_a + fraction * atom.mass_number_a
        self.mass = self.mass + fraction * atom.mass
        self.ionicity = self.ionicity + fraction * atom.ionicity

    def get_atomic_form_factor(self, energy):
        """get_atomic_form_factor

        Returns the mixed energy dependent atomic form factor.
        """
        f = 0
        for i in range(self.num_atoms):
            f += self.atoms[i][0].get_atomic_form_factor(energy) * self.atoms[i][1]

        return f

    def get_cm_atomic_form_factor(self, energy, qz):
        """get_cm_atomic_form_factor

        Returns the mixed energy and angle dependent atomic form factor.
        """
        f = 0
        for i in range(self.num_atoms):
            f += self.atoms[i][0].get_cm_atomic_form_factor(energy, qz) * self.atoms[i][1]

        return f
