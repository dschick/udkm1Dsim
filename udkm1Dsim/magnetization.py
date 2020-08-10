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

"""A :mod:`Magnetization` module """

__all__ = ["Magnetization"]

__docformat__ = "restructuredtext"

import numpy as np
from time import time
from os import path
from .simulation import Simulation
from .helpers import make_hash_md5


class Magnetization(Simulation):
    """Magnetization

    Base class for all magnetization simulatuons.

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self, output=[]):
        """String representation of this class"""
        return super().__str__(output)

    def get_hash(self, delays, **kwargs):
        """get_hash

        Returns a unique hash given by the energy :math:`E`,
        :math:`q_z` range, polarization states and the strain vectors as
        well as the sample structure hash for relevant xray parameters.
        Optionally, part of the strain_map is used.

        """
        param = [delays]

        if 'strain_map' in kwargs:
            strain_map = kwargs.get('strain_map')
            if np.size(strain_map) > 1e6:
                strain_map = strain_map.flatten()[0:1000000]
            param.append(strain_map)
            kwargs.pop('strain_map')

        if 'temp_map' in kwargs:
            temp_map = kwargs.get('temp_map')
            if np.size(temp_map) > 1e6:
                temp_map = temp_map.flatten()[0:1000000]
            param.append(temp_map)
            kwargs.pop('temp_map')

        for key, value in kwargs.items():
            param.append(value)

        return self.S.get_hash(types='magnetic') + '_' + make_hash_md5(param)

    def get_magnetization_map(self, delays, **kwargs):
        """get_magnetization_map

        Returns a differential `magnetization_map` for the sample structure.
        The `magnetization_map` can depend on the `temp_map` and `strain_map`
        that can be also calculated for the sample structure.

        """
        # create a hash of all simulation parameters
        filename = 'magnetization_map_' \
                   + self.get_hash(delays, **kwargs) \
                   + '.npy'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            magnetization_map = np.load(full_filename)
            self.disp_message('_magnetization_map_ loaded from file:\n\t' + filename)
        else:
            t1 = time()
            self.disp_message('Calculating _magnetization_map_ ...')
            # parse the input arguments

            if ('strain_map' in kwargs):
                if not isinstance(kwargs['strain_map'], np.ndarray):
                    raise TypeError('strain_map must be a numpy ndarray!')
            if ('temp_map' in kwargs):
                if not isinstance(kwargs['temp_map'], np.ndarray):
                    raise TypeError('temp_map must be a numpy ndarray!')

            magnetization_map = self.calc_magnetization_map(delays, **kwargs)

            self.disp_message('Elapsed time for _magnetization_map_:'
                              ' {:f} s'.format(time()-t1))
            self.save(full_filename, magnetization_map, '_magnetization_map_')
        return magnetization_map

    def calc_magnetization_map(self, delays, **kwargs):
        """calc_magnetization_map

        Calculates a differential `magnetization_map` for the sample structure.
        The `magnetization_map` can depend on the `temp_map` and `strain_map`
        that can be also calculated for the sample structure.

        This method is just an interface and should be overwritten for the
        actual simulations.

        """

        pass
