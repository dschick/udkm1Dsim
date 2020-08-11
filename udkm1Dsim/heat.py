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

"""A :mod:`Heat` module """

__all__ = ["Heat"]

__docformat__ = "restructuredtext"

import numpy as np
# from time import time
# from os import path
from .simulation import Simulation
from .helpers import make_hash_md5


class Heat(Simulation):
    """Heat

    Base class for heat simulatuons.

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Keyword Args:
        heat_diffusion (boolean): true when including heat diffusion in the
            calculations
        intp_at_interface (int): number of additional spacial points at the
            interface of each layer

    Attributes:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results
        heat_diffusion (boolean): true when including heat diffusion in the
            calculations
        intp_at_interface (int): number of additional spacial points at the
            interface of each layer
        distances (ndarray[float]): array of distances where to calc heat
            diffusion. If not set heat diffusion is calculated at each unit
            cell location or at every angstrom in amorphous layers
        ode_options (dict): dict with options for the MATLAB pdepe solver, see
            odeset, used for heat diffusion.
        boundary_types (list[str]): description of boundary types
        boundary_conditions (dict): dict of the left and right type of the
            boundary conditions for the MATLAB heat diffusion calculation
            1: isolator - 2: temperature - 3: flux
            For the last two cases the corresponding value has to be set as
            Kx1 array, where K is the number of sub-systems

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.heat_diffusion = kwargs.get('heat_diffusion', False)
        self.intp_at_interface = kwargs.get('intp_at_interface', 11)

        self.distances = np.array([])
        self.boundary_types = ['isolator', 'temperature', 'flux']
        self.boundary_conditions = {
            'left_type': 0,
            'left_value': np.array([]),
            'right_type': 0,
            'right_value': np.array([]),
            }
        self.ode_options = {'RelTol': 1e-3}

    def __str__(self, output=[]):
        """String representation of this class"""

        output = [['heat diffusion', self.heat_diffusion],
                  ['interpolate at interfaces', self.intp_at_interface],
                  ['distances', 'no distance mesh is set for heat diffusion calculations'
                   if self.distances.size == 0 else
                   'a distance mesh is set for heat diffusion calculations.'],
                  ['left boundary type',
                   self.boundary_types[self.boundary_conditions['left_type']]],
                  ] + output

        if self.boundary_conditions['left_type'] == 1:
            output += [['left boundary temperature',
                        str(self.boundary_conditions['left_value']) + ' K']]
        elif self.boundary_conditions['left_type'] == 2:
            output += [['left boundary flux',
                        str(self.boundary_conditions['left_value']) + ' W/m²']]

        output += [['right boundary type',
                   self.boundary_types[self.boundary_conditions['right_type']]]]

        if self.boundary_conditions['right_type'] == 1:
            output += [['right boundary temperature',
                        str(self.boundary_conditions['right_value']) + ' K']]
        elif self.boundary_conditions['right_type'] == 2:
            output += [['right boundary flux',
                        str(self.boundary_conditions['right_value']) + ' W/m²']]

        class_str = 'Heat simulation properties:\n\n'
        class_str += super().__str__(output)

        return class_str

    def get_hash(self, delays, excitation, init_temp, **kwargs):
        """get_hash

        Returns a unique hash given by the energy :math:`E`,
        :math:`q_z` range, polarization states and the strain vectors as
        well as the sample structure hash for relevant xray parameters.
        Optionally, part of the strain_map is used.

        """
        param = [delays, excitation, init_temp, self.heat_diffusion, self.intp_at_interface]

        for key, value in kwargs.items():
            param.append(value)

        return self.S.get_hash(types='heat') + '_' + make_hash_md5(param)

    def set_boundary_condition(self, boundary_side='left', boundary_type='isolator', value=0):
        """set_boundary_condition

        set the boundary conditions of the heat diffusion simulations

        """

        if boundary_type == 'temperature':
            btype = 1
        elif boundary_type == 'flux':
            btype = 2
        elif boundary_type == 'isolator':
            btype = 0
        else:
            btype = 0
            raise ValueError('boundary_type must be either _isolator_, '
                             '_temperature_ or _flux_!')

        K = self.S.num_sub_systems
        if (btype > 0) and (np.size(value) != K):
            raise ValueError('Non-isolating boundary conditions must have the '
                             'same dimensionality as the numer of sub-systems K!')

        if boundary_side == 'left':
            self.boundary_conditions['left_type'] = btype
            self.boundary_conditions['left_value'] = value
        elif boundary_side == 'right':
            self.boundary_conditions['right_type'] = btype
            self.boundary_conditions['right_value'] = value
        else:
            raise ValueError('boundary_side must be either _left_ or _right_!')

    def check_initial_temperature(self, init_temp):
        """check_initial_temperature

        An inital temperature for a heat simulation can be either a
        single temperature which is assumed to be valid for all layers
        in the structure or a temeprature profile is given with one
        temperature for each layer in the structure and for each subsystem.

        Args:
            init_temp (float, ndarray): initial temperature

        """
        N = self.S.get_number_of_layers()
        K = self.S.num_sub_systems

        # check size of initTemp
        if np.size(init_temp) == 1:
            # it is the same initial temperature for all layers
            init_temp = init_temp*np.ones([N, K])
        elif np.shape(init_temp) != [N, K]:
            # init_temp is a vector but has not as many elements as layers
            raise ValueError('The initial temperature vector must have 1 or '
                             'NxK elements, where N is the number of layers '
                             'in the structure and K the number of subsystems!')

        return init_temp
