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
from . import u, Q_
from .helpers import make_hash_md5, finderb


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
        excitation (dict): dictionary of excitation parameters: fluence,
            delay_pump, and pulse_width
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

        self._excitation = {'fluence': [], 'delay_pump': [], 'pulse_width': []}
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

    def get_hash(self, delays, init_temp, **kwargs):
        """get_hash

        Returns a unique hash given by the energy :math:`E`,
        :math:`q_z` range, polarization states and the strain vectors as
        well as the sample structure hash for relevant xray parameters.
        Optionally, part of the strain_map is used.

        """
        param = [delays, init_temp, self.heat_diffusion, self.intp_at_interface,
                 self.excitation, self.distances]

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

    def check_excitation(self, delays):
        """check_excitation

        The optical excitation is a dictionary with fluence
        :math:`F` [J/m²], delays :math:`t` [s] of the pump events, and pulse
        width :math:`\tau´ [s]. :math:`N` is the number of pump events.

        """
        delays = delays.to('s').magnitude
        fluence = self._excitation['fluence']
        delay_pump = self._excitation['delay_pump']
        pulse_width = self._excitation['pulse_width']

        # throw warnings if heat diffusion should be enabled
        if (self.S.num_sub_systems > 1) and not self.heat_diffusion:
            raise Warning('If you are introducing more than 1 subsystem you '
                          'should enable heat diffusion!')

        if np.sum(pulse_width) > 0 and not self.heat_diffusion:
            pulse_width = np.zeros_like(fluence)
            raise Warning('The effect of finite pulse duration of the excitation '
                          'is only considered if heat diffusion is enabled! '
                          'All pulse durations are set to 0!')

        """
        traverse excitation vector to update the `delay_pump` :math:`t_p`
        vector for finite pulse durations :math:`w(i)` as follows

        .. math:: t_p(i)-\mbox{window}\cdot w(i):w(i)/\mbox{intp}:t_p(i)+\mbox{window}\cdot w(i)

        and to combine excitations which have overlapping intervalls
        """
        n_excitation = []  # the result of the traversed excitation is a cell vector
        window = 1.5  # window factor for finite pulse duration
        intp = 1000  # interpolation factor for finite pulse duration

        i = 0  # start counter
        while i < len(delay_pump):
            k = i
            temp = []
            # check all upcoming excitations if they overlap with the current
            while k < len(delay_pump):
                temp.append([delay_pump[k], pulse_width[k], fluence[k]])
                if (k+1 < len(delay_pump)) and \
                    ((delay_pump[k] + window*pulse_width[k]) >=
                     (delay_pump[k+1] - window*pulse_width[k+1])):
                    # there is an overlap in time so add the next
                    # excitation to the current element
                    k += 1
                    if pulse_width[k] == 0:
                        # an overlapping pulse cannot have a pulseWidth
                        # of 0! Throw an error!
                        raise ValueError('Overlapping pulse must have duration > 0!')
                else:
                    # no overlap, so go to the next iteration of the outer while loop
                    break

            # caclulate the new time vector of the current excitation
            delta_delay = np.min(pulse_width[i:(k+1)])/intp
            if delta_delay == 0:
                # its pulse_width = 0 or no heat diffusion was enabled
                # so calculate just at a single delay step
                intervall = np.array([delay_pump[i]])
            else:
                intervall = np.r_[(delay_pump[i] - window*pulse_width[i]):
                                  (delay_pump[k] + window*pulse_width[k]):
                                  delta_delay]
            # update the new excitation list
            n_excitation.append([intervall,
                                 [t[0] for t in temp],
                                 [t[1] for t in temp],
                                 [t[2] for t in temp]])
            i = k+1  # increase counter

        """
        traverse the n_excitation list and add additional time vectors between
        the pump events for the later temperature calculation
        """
        res = []  # initialize the result list

        # check for delay < delay_pump[0]
        if np.size(delays[delays < np.min(n_excitation[0][0])]) > 0:
            res.append([delays[delays < np.min(n_excitation[0][0])], [], [], []])
        else:
            raise Warning('Please add more delay steps before the first excitation!')

        # traverse n_excitation
        for i, excitation in enumerate(n_excitation):
            res.append(excitation)
            if i+1 < len(n_excitation):
                # there is an upcoming pump event
                if np.size(delays[np.logical_and(delays > excitation[0][-1],
                                                 delays < n_excitation[i+1][0][0])]) > 0:
                    # there are times between the current and next excitation
                    temp = [delays[np.logical_and(delays > excitation[0][-1],
                                                  delays < n_excitation[i+1][0][0])], [], [], []]
                    res.append(temp)
            else:  # this is the last pump event
                if np.size(delays[delays > excitation[0][-1]]) > 0:
                    # there are times after the current last excitation
                    temp = [delays[delays > excitation[0][-1]], [], [], []]
                    res.append(temp)

        return res, fluence, delay_pump, pulse_width

    def get_absorption_profile(self, distances=[]):
        """get_absorption_profile

        Returns a vector of the absorption profile derived from Lambert-Beer's
        law. The transmission is given by:

        .. math:: \tau = \frac{I}{I_0} =  \exp(-z/\zeta)

        and the absorption by:

        .. math:: \alpha = 1 - \tau =  1 - \exp(-z/\zeta)

        The absorption profile can be derived from the spatial derivative:

        .. math:: \frac{d \alpha(z)}{dz} = \frac{1}{\zeta} \exp(-z/\zeta)

        """
        if distances == []:
            # if no distances are set, calculate the extinction on
            # the middle of each unit cell
            d_start, _, distances = self.S.get_distances_of_layers()
        else:
            d_start, _, _ = self.S.get_distances_of_layers()

        interfaces = self.S.get_distances_of_interfaces()
        # convert to [m] and get rid of quantities for faster calculations
        d_start = d_start.to('m').magnitude
        distances = distances.to('m').magnitude
        interfaces = interfaces.to('m').magnitude

        N = len(distances)
        dalpha_dz = np.zeros(N)  # initialize relative absorbed energies
        I0 = 1  # initial intensity
        k = 0  # counter for first layer
        for i in range(len(interfaces)-1):
            # find the first layer and get properties
            index = finderb(interfaces[i], d_start)
            layer = self.S.get_layer_handle(index[0])
            opt_pen_depth = layer.opt_pen_depth.to('m').magnitude

            # get all distances in the current layer we have to
            # calculate the absorption profile for
            if i >= len(interfaces)-2:  # last layer
                z = distances[np.logical_and(distances >= interfaces[i],
                                             distances <= interfaces[i+1])]
            else:
                z = distances[np.logical_and(distances >= interfaces[i],
                                             distances < interfaces[i+1])]
            m = len(z)
            if not np.isinf(opt_pen_depth):
                # the layer is absorbing
                dalpha_dz[k:k+m] = I0/opt_pen_depth*np.exp(-(z-interfaces[i])/opt_pen_depth)
                # calculate the remaining intensity for the next layer
                I0 = I0*np.exp(-(interfaces[i+1]-interfaces[i])/opt_pen_depth)
            k = k+m  # set the counter
        return dalpha_dz

    @property
    def excitation(self):
        """dict: excitation parameters

        Convert to from default SI units to real quantities

        """
        excitation = {'fluence': Q_(self._excitation['fluence'], u.J/u.m**2).to('mJ/cm**2'),
                      'delay_pump': Q_(self._excitation['delay_pump'], u.s).to('ps'),
                      'pulse_width': Q_(self._excitation['pulse_width'], u.s).to('ps')}

        return excitation

    @excitation.setter
    def excitation(self, excitation):
        """set.excitation"""

        # check the size of excitation, if we have a multipulse excitation
        if isinstance(excitation, Q_):
            # just a fluence is given
            self._excitation['fluence'] = [excitation.to('J/m**2').magnitude]
            self._excitation['delay_pump'] = [0]  # we define that the exciation is at t=0
            self._excitation['pulse_width'] = [0]  # pulse width is 0 by default
        elif isinstance(excitation, dict):
            try:
                self._excitation['fluence'] = excitation['fluence'].to('J/m**2').magnitude
                self._excitation['delay_pump'] = excitation['delay_pump'].to('s').magnitude
                self._excitation['pulse_width'] = excitation['pulse_width'].to('s').magnitude
            except KeyError:
                print('The excitation dictionary must include the tree keys '
                      '_fluence_, _delay_pump_, _pulse_width_. Each must be'
                      'a single or array of pint quantities.')
        else:
            raise ValueError('_excitation_ must be either a float/int or dict '
                             'of pint quantities!')

        if not (len(self._excitation['fluence'])
                == len(self._excitation['delay_pump'])
                == len(self._excitation['pulse_width'])):
            raise ValueError('Elements of excitation dict must have '
                             'the same number of elements!')

        # check the elements of the delay_pump vector
        if len(self._excitation['delay_pump']) != len(np.unique(self._excitation['delay_pump'])):
            raise ValueError('The excitations have to be unique in delays!')
        else:
            self._excitation['delay_pump'] = np.sort(self._excitation['delay_pump'])
