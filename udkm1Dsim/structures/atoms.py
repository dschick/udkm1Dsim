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

__all__ = ['Atom', 'AtomMixed']

__docformat__ = 'restructuredtext'

from .. import u, Q_
import os
import numpy as np
import scipy.constants as constants
import warnings
from tabulate import tabulate


class Atom:
    """Atom

    Smallest structural unit of which larger structures can be build.

    It holds real physical properties of on the atomic level.

    Args:
        symbol (str): symbol of the atom.

    Keyword Args:
        id (str): id of the atom, may differ from symbol and/or name.
        ionicity (int): ionicity of the atom.
        atomic_form_factor_path (str): path to atomic form factor coeffs.
        atomic_form_factor_source (str): either _henke_ or default _chantler_
        magnetic_form_factor_path (str): path to magnetic form factor coeffs.

    Attributes:
        symbol (str): symbol of the element.
        id (str): id of the atom, may differ from symbol and/or name.
        name (str): name of the element (generic).
        atomic_number_z (int): Z atomic number.
        mass_number_a (float): A atomic mass number.
        ionicity (int): ionicity of the atom.
        mass (float): mass of the atom [kg].
        atomic_form_factor_coeff (ndarray[float]): atomic form factor.
           coefficients for energy-dependent atomic form factor.
        cromer_mann_coeff (ndarray[float]): cromer-mann coefficients for
           angular-dependent atomic form factor.
        magnetic_form_factor_coeff (ndarray[float]): magnetic form factor
           coefficients for energy-dependent magnetic form factor.
        mag_amplitude (float): magnetization amplitude -1 .. 1.
        mag_phi (float): phi angle of magnetization [rad].
        mag_gamma (float): gamma angle of magnetization [rad].

    References:

        .. [1] B. L. Henke, E. M. Gullikson & J. C. Davis,
           *X-Ray Interactions: Photoabsorption, Scattering,
           Transmission, and Reflection at E = 50-30,000 eV, Z = 1-92*,
           `Atomic Data and Nuclear Data Tables, 54(2), 181–342, (1993).
           <http://www.doi.org/10.1006/adnd.1993.1013>`_
        .. [2] C.T. Chantler, K. Olsen, R.A. Dragoset, J. Chang, A.R. Kishore,
           S.A. Kotochigova, & D.S. Zucker,
           *Detailed Tabulation of Atomic Form Factors, Photoelectric
           Absorption and Scattering Cross Section, and Mass Attenuation
           Coefficients for Z = 1-92 from E = 1-10 eV to E = 0.4-1.0 MeV*,
           `NIST Standard Reference Database 66.
           <https://dx.doi.org/10.18434/T4HS32>`_
        .. [3] J. Als-Nielson, & D. McMorrow,
           `Elements of Modern X-Ray Physics. New York: John Wiley &
           Sons, Ltd. (2001) <http://www.doi.org/10.1002/9781119998365>`_
        .. [4] D. T. Cromer & J. B. Mann, *X-ray scattering
           factors computed from numerical Hartree–Fock wave functions*,
           `Acta Crystallographica Section A, 24(2), 321–324 (1968).
           <http://www.doi.org/10.1107/S0567739468000550>`_

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
                                    '../parameters/elements.dat')
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
        self.atomic_form_factor_coeff = self.read_atomic_form_factor_coeff(
            filename=kwargs.get('atomic_form_factor_path', ''),
            source=kwargs.get('atomic_form_factor_source', 'chantler'))
        self.magnetic_form_factor_coeff = self.read_magnetic_form_factor_coeff(
            filename=kwargs.get('magnetic_form_factor_path', ''))
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

    def read_atomic_form_factor_coeff(self, source='chantler', filename=''):
        """read_atomic_form_factor_coeff

        The coefficients for the atomic form factor :math:`f` in dependence of
        the photon energy :math:`E` is read from a parameter file given by [1]_
        or by [2]_ as default.

        Args:
            source (str, optional): source of atmoic form factors can be either
                _henke_ or _chantler_. Defaults to _chantler_.
            filename (str, optional): full path and filename to the atomic form
                factor coefficients.

        Returns:
            f (ndarray[float]): atomic form factor coefficients.

        """
        if not filename:
            if source not in ['chantler', 'henke']:
                raise ValueError('The source of the atomic form factors must be '
                                 'either chantler or henke!')

            if source == 'chantler':
                sub_path = 'chantler/{:s}.cf'.format(self.symbol.lower())
            elif source == 'henke':
                sub_path = 'henke/{:s}.nff'.format(self.symbol.lower())

            filename = os.path.join(os.path.dirname(__file__),
                                    '../parameters/atomic_form_factors/{:s}'.format(sub_path))
        try:
            f = np.genfromtxt(filename, skip_header=0)
        except OSError:
            print('Atomic form factor file {:s} not found!'.format(filename))
            raise

        return f

    @u.wraps(None, (None, 'eV'), strict=False)
    def get_atomic_form_factor(self, energy):
        """get_atomic_form_factor

        The complex atomic form factor for the photon energy :math:`E` [eV] is
        calculated by:

        .. math:: f(E)=f_1 - i f_2

        Convention of Ref. [3]_ (p. 11, footnote) is a negative :math:`f_2`.

        Args:
            energy (ndarray[float]): photon energy [eV].

        Returns:
            f (ndarray[complex]): energy-dependent atomic form factors.

        """
        # interpolate the real and imaginary part in dependence of E
        f1 = np.interp(energy, self.atomic_form_factor_coeff[:, 0],
                       self.atomic_form_factor_coeff[:, 1])
        f2 = np.interp(energy, self.atomic_form_factor_coeff[:, 0],
                       self.atomic_form_factor_coeff[:, 2])

        return f1 - f2*1j

    def read_cromer_mann_coeff(self):
        r"""read_cromer_mann_coeff

        The Cromer-Mann coefficients (Ref. [4]_) are read from a parameter file
        and are returned in the following order:

        .. math:: a_1\; a_2\; a_3\; a_4\; b_1\; b_2\; b_3\; b_4\; c

        Returns:
            cm (ndarray[float]): Cromer-Mann coefficients.

        """
        filename = os.path.join(os.path.dirname(__file__),
                                '../parameters/atomic_form_factors/cromermann.txt')
        try:
            cm = np.genfromtxt(filename, skip_header=1,
                               usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
        except Exception as e:
            print('File {:s} not found!'.format(filename))
            print(e)

        return cm[(cm[:, 0] == self.atomic_number_z) & (cm[:, 1] == self.ionicity)][0]

    @u.wraps(None, (None, 'eV', 'm**-1'), strict=False)
    def get_cm_atomic_form_factor(self, energy, qz):
        r"""get_cm_atomic_form_factor

        The atomic form factor :math:`f` is calculated in dependence of the
        photon energy :math:`E` [eV] and the :math:`z`-component of the
        scattering vector :math:`q_z` [Å :math:`^{-1}`] (Ref. [4]_).
        Note that the Cromer-Mann coefficients are fitted for :math:`q_z` in
        [Å :math:`^{-1}`]!

        See Ref. [3]_ (p. 235).

        .. math:: f(q_z,E) = f_{CM}(q_z) + \delta f_1(E) -i f_2(E)

        :math:`f_{CM}(q_z)` is given in Ref. [4]_:

        .. math::

            f_{CM}(q_z) = \sum(a_i \, \exp(-b_i \, (q_z/4\pi)^2))+ c

        :math:`\delta f_1(E)` is the dispersion correction:

        .. math::

            \delta f_1(E) = f_1(E) - \left(\sum^4_i(a_i) + c\right)

        Thus:

        .. math:: f(q_z,E) = \sum(a_i \, \exp(-b_i \, q_z/2\pi))
           + c + f_1(E)-i f_2(E) - \left(\sum(a_i) + c\right)

        .. math:: f(q_z,E) = \sum(a_i \, \exp(-b_i \, q_z/2\pi))
           + f_1(E) -i f_2(E) - \sum(a_i)

        Args:
            energy (ndarray[float]): photon energy [eV].
            qz (ndarray[float]): scattering vector [1/m].

        Returns:
            f (ndarray[complex]): energy- and qz-dependent Cromer-Mann atomic form
            factors.

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

    def read_magnetic_form_factor_coeff(self, filename=''):
        """read_magnetic_form_factor_coeff

        The coefficients for the magnetic form factor :math:`m` in dependence
        of the photon energy :math:`E` is read from a parameter file.

        Args:
            filename (str): optional full path and filename to the magnetic
                form factor coefficients.

        Returns:
            m (ndarray[float]): magnetic form factor coefficients.

        """
        if not filename:
            filename = os.path.join(os.path.dirname(__file__),
                                    '../parameters/magnetic_form_factors/{:s}.mf'.format(
                                            self.symbol))
        try:
            m = np.genfromtxt(filename)
        except Exception as e:
            print('File {:s} not found!'.format(filename))
            print(e)
            # return zero array
            m = np.zeros([1, 3])

        return m

    @u.wraps(None, (None, 'eV'), strict=False)
    def get_magnetic_form_factor(self, energy):
        """get_magnetic_form_factor

        The complex magnetic form factor is claculated by:

        .. math:: m(E) = m_1 - i m_2

        for the photon energy :math:`E` [eV].

        Convention of Ref. [3]_ (p. 11, footnote) is a negative :math:`m_2`

        Args:
            energy (ndarray[float]): photon energy [eV].

        Returns:
            m (ndarray[complex]): energy-dependent magnetic form factors.

        """
        # interpolate the real and imaginary part in dependence of E
        m1 = np.interp(energy, self.magnetic_form_factor_coeff[:, 0],
                       self.magnetic_form_factor_coeff[:, 1])
        m2 = np.interp(energy, self.magnetic_form_factor_coeff[:, 0],
                       self.magnetic_form_factor_coeff[:, 2])

        return m1 - m2*1j

    @property
    def mag_phi(self):
        return Q_(self._mag_phi, u.rad).to('deg')

    @mag_phi.setter
    def mag_phi(self, mag_phi):
        self._mag_phi = mag_phi.to_base_units().magnitude

    @property
    def mag_gamma(self):
        return Q_(self._mag_gamma, u.rad).to('deg')

    @mag_gamma.setter
    def mag_gamma(self, mag_gamma):
        self._mag_gamma = mag_gamma.to_base_units().magnitude


class AtomMixed(Atom):
    """AtomMixed

    Representation of mixed atoms in alloys and stochiometric mixtures.

    All properties of the included sub-atoms of class Atom are averaged
    and weighted with their stochiometric ratio.

    Args:
        symbol (str): symbol of the atom.

    Keyword Args:
        id (str): id of the atom, may differ from symbol and/or name.
        name (str): name of the mixed atom, default is symbol.
        atomic_form_factor_path (str): path to atomic form factor coeffs.
        magnetic_form_factor_path (str): path to magnetic form factor coeffs.

    Attributes:
        symbol (str): symbol of the element.
        id (str): id of the atom, may differ from symbol and/or name.
        name (str): name of the mixed atom, default is symbol.
        atomic_number_z (int): Z atomic number.
        mass_number_a (float): A atomic mass number.
        ionicity (int): ionicity of the atom.
        mass (float): mass of the atom [kg].
        atomic_form_factor_coeff (ndarray[float]): atomic form factor.
           coefficients for energy-dependent atomic form factor.
        magnetic_form_factor_coeff (ndarray[float]): magnetic form factor
           coefficients for energy-dependent magnetic form factor.
        mag_amplitude (float): magnetization amplitude -1 .. 1.
        mag_phi (float): phi angle of magnetization [rad].
        mag_gamma (float): gamma angle of magnetization [rad].
        atoms (list[Atoms]): list of Atoms.
        num_atoms (int): number of atoms.

    """

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        self.id = kwargs.get('id', symbol)
        self.name = kwargs.get('name', symbol)
        self.mag_amplitude = kwargs.get('mag_amplitude', 0)
        self.mag_phi = kwargs.get('mag_phi', 0*u.deg)
        self.mag_gamma = kwargs.get('mag_gamma', 0*u.deg)
        self.ionicity = 0
        self.atomic_number_z = 0
        self.mass_number_a = 0
        self.mass = 0
        self.atoms = []
        self.num_atoms = 0
        self.atomic_form_factor_coeff = self.read_atomic_form_factor_coeff(
            filename=kwargs.get('atomic_form_factor_path', ''))
        self.magnetic_form_factor_coeff = self.read_magnetic_form_factor_coeff(
            filename=kwargs.get('magnetic_form_factor_path', ''))

    def __str__(self):
        """String representation of this class"""

        output = {'parameter': ['id', 'symbol', 'name', 'atomic number Z', 'mass number A', 'mass',
                                'ionicity', 'magn. amplitude', 'magn. phi', 'magn. gamma'],
                  'value': [self.id, self.symbol, self.name, self.atomic_number_z,
                            self.mass_number_a, '{:.4~P}'.format(self.mass), self.ionicity,
                            self.mag_amplitude, self.mag_phi, self.mag_gamma]}

        output_atom = []
        for i in range(self.num_atoms):
            output_atom.append([self.atoms[i][0].name, '{:.1f} %'.format(self.atoms[i][1]*100)])

        return ('AtomMixed with the following properties\n'
                + tabulate(output, colalign=('right',), tablefmt="rst", floatfmt=('.2f', '.2f'))
                + '\n{:d} Constituents:\n'.format(self.num_atoms)
                + tabulate(output_atom, colalign=('right',), floatfmt=('.2f', '.2f')))

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
            self.atomic_number_z = self.atomic_number_z + fraction * atom.atomic_number_z
            self.mass_number_a = self.mass_number_a + fraction * atom.mass_number_a
            self.mass = self.mass + fraction * atom.mass
            self.ionicity = self.ionicity + fraction * atom.ionicity
        else:
            warnings.warn('Only Atom objects can be added to a MixedAtom!')

    def read_atomic_form_factor_coeff(self, filename=''):
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
        try:
            f = np.genfromtxt(filename, skip_header=0)
        except Exception as e:
            print('File {:s} not found!'.format(filename))
            print(e)

        return f

    @u.wraps(None, (None, 'eV'), strict=False)
    def get_atomic_form_factor(self, energy):
        """get_atomic_form_factor

        Averaged energy dependent atomic form factor.
        If ``atomic_form_factor_path`` was given on initialization this file
        will be used instead.

        Args:
            energy (ndarray[float]): photon energy [eV].

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

    @u.wraps(None, (None, 'eV', 'm**-1'), strict=False)
    def get_cm_atomic_form_factor(self, energy, qz):
        """get_cm_atomic_form_factor

        Averaged energy and qz-dependent atomic form factors.

        Args:
            energy (ndarray[float]): photon energy [eV].
            qz (ndarray[float]): scattering vector [1/m].

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
            warnings.warn('Cromer-Mann correction cannot be applied to '
                          'atomic form factors from external files. '
                          'Returning uncorrected values instead!')
            f = self.get_atomic_form_factor(energy)

        return f

    def read_magnetic_form_factor_coeff(self, filename=''):
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
        try:
            m = np.genfromtxt(filename)
        except Exception as e:
            print('File {:s} not found!'.format(filename))
            print(e)
            # return zero array
            m = np.zeros([1, 3])

        return m

    @u.wraps(None, (None, 'eV'), strict=False)
    def get_magnetic_form_factor(self, energy):
        """get_magnetic_form_factor

        Mixed energy dependent magnetic form factors.

        Args:
            energy (ndarray[float]): photon energy [eV].

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

    @property
    def mag_phi(self):
        return Q_(self._mag_phi, u.rad).to('deg')

    @mag_phi.setter
    def mag_phi(self, mag_phi):
        self._mag_phi = mag_phi.to_base_units().magnitude

    @property
    def mag_gamma(self):
        return Q_(self._mag_gamma, u.rad).to('deg')

    @mag_gamma.setter
    def mag_gamma(self, mag_gamma):
        self._mag_gamma = mag_gamma.to_base_units().magnitude
