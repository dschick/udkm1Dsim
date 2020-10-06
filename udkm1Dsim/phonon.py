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

"""A :mod:`Phonon` module """

__all__ = ["Phonon"]

__docformat__ = "restructuredtext"

import numpy as np
from .simulation import Simulation
from .helpers import make_hash_md5


class Phonon(Simulation):
    """Phonon

    Base class for phonon simulatuons.

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Keyword Args:
        only_heat (boolean): true when including only thermal expanison without
            coherent phonon dynamics

    Attributes:
        S (object): sample to do simulations with
        only_heat (boolean): force recalculation of results
        heat_diffusion (boolean): true when including only thermal expanison without
            coherent phonon dynamics
        matlab_engine (module): MATLAB to Python API engine required for
            calculating heat diffusion

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.only_heat = kwargs.get('only_heat', False)
        self.matlab_engine = []

    def __str__(self, output=[]):
        """String representation of this class"""

        output = [['only heat', self.heat_diffusion],
                  ] + output

        class_str = 'Phonon simulation properties:\n\n'
        class_str += super().__str__(output)

        return class_str

    def get_hash(self, delays, temp_map, delta_temp_map, **kwargs):
        """get_hash

        Returns an unique hash given by the delays, and temp- and delta_temp_map
        as well as the sample structure hash for relevant thermal parameters.

        """
        param = [delays, self.only_heat]

        if np.size(temp_map) > 1e6:
            temp_map = temp_map.flatten()[0:1000000]
            delta_temp_map = delta_temp_map.flatten()[0:1000000]
        param.append(temp_map)
        param.append(delta_temp_map)

        for key, value in kwargs.items():
            param.append(value)

        return self.S.get_hash(types='phonon') + '_' + make_hash_md5(param)

    def get_all_strains_per_unique_layer(self, strain_map):
        """get_all_strains_per_unique_layer

        Returns a dict with all strains per unique layer that
        are given by the input _strain_map_.

        """
        # get the position indices of all unique layers in the sample structure
        positions = self.S.get_all_positions_per_unique_layer()
        strains = {}

        for key, value in positions.items():
            strains[key] = np.sort(np.unique(strain_map[:, value].flatten()))

        return strains

    def get_reduced_strains_per_unique_layer(self, strain_map, N=100):
        """ get_reduced_strains_per_unique_layer

        Returns a dict with all strains per unique layer that are given by the
        input _strain_map_, BUT with a reduced number. The reduction is done
        by equally spacing the strains between the min and max strain with a
        given number :math:`N`, which can be also a vector of the
        :math:`len(N) = M`, where :math:`M` is the number of unique layers.

        """
        # initialize
        all_strains = self.get_all_strains_per_unique_layer(strain_map)
        M = len(all_strains)  # Nb. of unique layers
        strains = {}

        if np.size(N) == 1:
            N = N*np.ones([M, 1])
        elif np.size(N) != M:
            raise ValueError('The dimension of N must be either 1 or the number '
                             'of unique layers the structure!')

        for i, (key, value) in enumerate(all_strains.items()):
            min_strain = np.min(value)
            max_strain = np.max(value)
            strains[key] = np.sort(np.unique(
                np.r_[0, np.linspace(min_strain, max_strain, int(N[i]))]))

        return strains

    def check_temp_maps(self, temp_map, delta_temp_map, delays):
        """ check_temp_maps

        Returns the corrected _delta_temp_map_ for the _strain_map_
        calculation and checks _temp_map_ and _delta_temp_map_ for the
        correct dimensions.

        """
        M = len(delays)
        N = self.S.get_number_of_layers()
        K = self.S.num_sub_systems

        # check size of delta_temp_map
        if K == 1:
            if np.shape(delta_temp_map) == (1, N):
                temp = delta_temp_map
                delta_temp_map = np.zeros([M, N])
                delta_temp_map[0, :] = temp
            elif (np.size(delta_temp_map, 0) != M) or (np.size(delta_temp_map, 1) != N):
                raise ValueError('The given temperature difference map does not have the '
                                 'dimension M x N, where M is the number of time steps '
                                 'and N the number of layers!')
        else:
            if np.shape(delta_temp_map) == (1, N, K):
                temp = delta_temp_map
                delta_temp_map = np.zeros([M, N, K])
                delta_temp_map[0, :, :] = temp
            elif ((np.size(delta_temp_map, 0) != M)
                  or (np.size(delta_temp_map, 1) != N)
                  or (np.size(delta_temp_map, 2) != K)):
                raise ValueError('The given temperature difference map does not have the '
                                 'dimension M x N x K, where M is the number of time steps '
                                 'and N the number of layers and K is the number of subsystems!')

        if np.shape(temp_map) != np.shape(delta_temp_map):
            raise ValueError('The temperature map does not have the same size as the '
                             'temperature difference map!')

        return temp_map, delta_temp_map

    def calc_sticks_from_temp_map(self, temp_map, delta_temp_map):
        """calc_sticks_from_temp_map

        Calculates the sticks to insert into the layer springs which model the
        external force (thermal stress). The length of :math:`l_i` of the
        :math:`i`-th spacer stick is calculated from the temperature-dependent
        linear thermal expansion :math:`\alpha(T)` of the layer:

        .. math::

            \alpha(T) = \frac{1}{L} \frac{d L}{d T}

        which results after integration in

        .. math::

            l = \Delta L = L_1 \exp(A(T_2) - A(T_1)) - L_1 $$

        where :math:`A(T)` is the integrated lin. therm. expansion coefficient
        in respect to the temperature :math:`T`. The indices 1 and 2 indicate
        the initial and final state.

        """
        M = np.size(temp_map, 0)  # nb of delay steps
        N = self.S.get_number_of_layers()
        K = self.S.num_sub_systems

        thicknesses = self.S.get_layer_property_vector('_thickness')
        int_lin_therm_exps = self.S.get_layer_property_vector('_int_lin_therm_exp')

        # evaluated initial integrated linear thermal expansion from T1 to T2
        int_alpha_T0 = np.zeros([N, K])
        # evaluated integrated linear thermal expansion from T1 to T2
        int_alpha_T = np.zeros([N, K])
        sticks = np.zeros([M, N])  # the sticks inserted in the unit cells
        sticks_sub_systems = np.zeros([M, N, K])  # the sticks for each thermodynamic subsystem

        # calculate initial integrated linear thermal expansion from T1 to T2
        # traverse subsystems
        for j in range(K):
            for l in range(N):
                int_alpha_T0[l, j] = int_lin_therm_exps[l][j](np.squeeze(temp_map[0, l, j]) - np.squeeze(delta_temp_map[0, l, j]))

        # calculate sticks for all subsytsems for all delay steps
        # traverse time
        for i in range(M):
            if np.any(delta_temp_map[i, :]):  # there is a temperature change
                # Calculate new sticks from the integrated linear
                # thermal expansion from initial temperature to
                # current temperature for each subsystem
                # traverse subsystems
                for j in range(K):
                    for l in range(N):
                        int_alpha_T[l, j] = int_lin_therm_exps[l][j](np.squeeze(temp_map[i, l, j]))

                # calculate the length of the sticks of each subsystem and sum
                # them up
                sticks_sub_systems[i, :, :] = np.tile(thicknesses, (K, 1)).T * np.exp(int_alpha_T-int_alpha_T0) - np.tile(thicknesses, (K, 1)).T
                # sticks[i, :] = np.sum(sticks_sub_systems[i, :, :], 2)
            else:  # no temperature change, so keep the current sticks
                if i > 0:
                    sticks_sub_systems[i, :, :] = sticks_sub_systems[i-1, :, :]
                    sticks[i, :] = sticks[i-1, :]
        return sticks, sticks_sub_systems
