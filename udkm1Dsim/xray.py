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

"""A :mod:`Xray` module """

__all__ = ["Xray"]

__docformat__ = "restructuredtext"

import numpy as np
from .simulation import Simulation
from . import u, Q_
from tabulate import tabulate


class Xray(Simulation):
    """Xray

    Base class for all xray simulatuons.

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Attributes:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results
        polarization (float): polarization state

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self._energy = np.array([])
        self._wl = np.array([])
        self._k = np.array([])
        self._theta = np.array([])
        self._qz = np.array([])
        self.polarization = 0

    def __str__(self):
        """String representation of this class"""
        output = [['force recalc', self.force_recalc],
                  ['cache directory', self.cache_dir]]

        class_str = 'This is the current structure for the simulations:\n\n'
        class_str += self.S.__str__()
        class_str += '\n\nDisplay properties:\n\n'
        class_str += tabulate(output, headers=['parameter', 'value'], tablefmt="rst",
                              colalign=('right',), floatfmt=('.2f', '.2f'))
        return class_str

    def get_polarization_factor(self):
        """get_polarization_factor

        Returns the polarization factor :math:`P(\\vartheta)` for a
        given incident angle :math:`\\vartheta` for the case of
        s-polarization (pol = 0), or p-polarization (pol = 1), or
        unpolarized X-rays (pol = 0.5):

        .. math::

            P(\\vartheta) = \sqrt{(1-\mbox{pol}) + \mbox{pol} \cdot \cos(2\\vartheta)}

        """

        return np.sqrt((1-self.polarization) + self.polarization*np.cos(2*self._theta)**2)

    def update_experiment(self, caller):
        """update experimental parameters

        Recalculate energy, wavelength, and wavevector as well as theta
        and the scattering vector in case any of these has changed.

        .. math::

            \lambda = \\frac {hc} {E}

            E = \\frac {hc} {\lambda}

            k = \\frac {2\pi} {\lambda}

            \\vartheta = \\arcsin{ \\frac{\lambda q_z}{4\pi}}

            q_z = 2k\\sin{\\vartheta}

        """
        from scipy import constants
        if caller != 'energy':
            if caller == 'wl':  # calc energy from wavelength
                self._energy = Q_((constants.h*constants.c)/self._wl, 'J').to('eV').magnitude
            elif caller == 'k':  # calc energy von wavevector
                self._energy = \
                    Q_((constants.h*constants.c)/(2*np.pi/self._k), 'J').to('eV').magnitude
        if caller != 'wl':
            if caller == 'energy':  # calc wavelength from energy
                self._wl = (constants.h*constants.c)/self.energy.to('J').magnitude
            elif caller == 'k':  # calc wavelength from wavevector
                self._wl = 2*np.pi/self._k
        if caller != 'k':
            if caller == 'energy':  # calc wavevector from energy
                self._k = 2*np.pi/self._wl
            elif caller == 'wl':  # calc wavevector from wavelength
                self._k = 2*np.pi/self._wl

        if caller != 'theta':
            self._theta = np.arcsin(np.outer(self._wl, self._qz)/np.pi/4)
        if caller != 'qz':
            self._qz = np.outer(2*self._k, np.sin(self._theta))

    @property
    def energy(self):
        """ndarray[float]: photon energy(s) [eV]"""
        return Q_(self._energy, u.eV)

    @energy.setter
    def energy(self, energy):
        """set.energy"""
        self._energy = np.array(energy.to('eV').magnitude)
        self.update_experiment('energy')

    @property
    def wl(self):
        """ndarray[float]: photon wavelength(s) [nm]"""
        return Q_(self._wl, u.m).to('nm')

    @wl.setter
    def wl(self, wl):
        """set.wl"""
        self._wl = np.array(wl.to_base_units().magnitude)
        self.update_experiment('wl')

    @property
    def k(self):
        """ndarray[float]: photon wavevector(s) [1/nm]"""
        return Q_(self._k, 1/u.m).to('1/nm')

    @k.setter
    def k(self, k):
        """set.k"""
        self._k = np.array(k.to_base_units().magnitude)
        self.update_experiment('k')

    @property
    def theta(self):
        """ndarray[float]: incoming xray angle(s) :math:`\theta` [deg]"""
        return Q_(self._theta, u.rad).to('deg')

    @theta.setter
    def theta(self, theta):
        """set.theta"""
        self._theta = np.array(theta.to_base_units().magnitude)
        self.update_experiment('theta')

    @property
    def qz(self):
        """ndarray[float]: scattering vector(s) :math:`q_z` [1/nm]"""
        return Q_(self._qz, 1/u.m).to('1/nm')

    @qz.setter
    def qz(self, qz):
        """set.qz"""
        self._qz = np.array(qz.to_base_units().magnitude)
        self.update_experiment('qz')
