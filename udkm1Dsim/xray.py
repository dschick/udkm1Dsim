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
from . import u


class Xray(Simulation):
    """Xray

    Base class for all xray simulatuons.

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Attributes:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results
        energy (ndarray[float]): photon energy(s) [eV]
        lambd (ndarray[float]): photon wavelength(s) [m]
        theta (ndarray[float]): incoming xray angle(s) :math:`\theta` [deg]
        qz (ndarray[float]): scattering vector(s) :math:`q_z` [1/m]
        polarization (int): index of different polarization states

    """

    def __init__(self, S, force_recalc, **kwargs):
        super(S, force_recalc).__init__(**kwargs)
        self.S = S
        self.force_recalc = force_recalc
        self.energy = np.array([])
        self.lambd = np.array([])
        self.theta = np.array([])
        self.qz = np.array([])
        self.polarization = 0

    def __str__(self):
        """String representation of this class"""
        output = [['force recalc', self.force_recalc],
                  ['cache directory', self.cache_dir],]

        class_str = 'This is the current structure for the simulations:\n\n'
        class_str += self.S.__str__()
        class_str += '\n\nDisplay properties:\n\n'
        class_str += tabulate(output, headers=['parameter', 'value'], tablefmt="rst",
                              colalign=('right',), floatfmt=('.2f', '.2f'))
        return class_str

    @property
    def cache_dir(self):
        """str: path to cached data"""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir):
        """set.cache_dir"""
        import os.path as path
        if path.exists(cache_dir):
            self._cache_dir = cache_dir
        else:
            print('Cache dir does not exist.\nPlease create the path first.')
