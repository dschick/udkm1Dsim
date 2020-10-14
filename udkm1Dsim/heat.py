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
from scipy.optimize import brentq
from scipy.interpolate import interp2d
from scipy.integrate import solve_ivp
from time import time
from os import path
from .simulation import Simulation
from . import u, Q_
from .helpers import make_hash_md5, finderb, multi_gauss
import warnings
from tqdm.notebook import tqdm


class Heat(Simulation):
    """Heat

    Base class for heat simulatuons.

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Keyword Args:
        progress_bar (boolean): enable tqdm progress bar
        heat_diffusion (boolean): true when including heat diffusion in the
            calculations
        intp_at_interface (int): number of additional spacial points at the
            interface of each layer
        backend (str): pde solver backend - either default scipy or matlab

    Attributes:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results
        progress_bar (boolean): enable tqdm progress bar
        heat_diffusion (boolean): true when including heat diffusion in the
            calculations
        intp_at_interface (int): number of additional spacial points at the
            interface of each layer
        backend (str): pde solver backend - either default scipy or matlab
        boundary_conditions (dict): dictionary of boundary conditions:
            boundary type top/bottom: isolator/temperature/flux
            boundary value top/bottom
        excitation (dict): dictionary of excitation parameters: fluence,
            delay_pump, pulse_width, wavelength, theta, polarization,
            multilayer_absorption
        distances (ndarray[float]): array of distances where to calc heat
            diffusion. If not set heat diffusion is calculated at each unit
            cell location or at every angstrom in amorphous layers
        ode_options (dict): options for scipy solve_ivp ode solver, see
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>
        ode_options_matlab (dict): dict with options for the MATLAB pdepe solver,
            see odeset, used for heat diffusion.
        boundary_types (list[str]): description of boundary types
        boundary_conditions (dict): dict of the top and bottom type of the
            boundary conditions for the MATLAB heat diffusion calculation
            1: isolator - 2: temperature - 3: flux
            For the last two cases the corresponding value has to be set as
            Kx1 array, where K is the number of sub-systems
        matlab_engine (module): MATLAB to Python API engine required for
            calculating heat diffusion

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.heat_diffusion = kwargs.get('heat_diffusion', False)
        self.intp_at_interface = kwargs.get('intp_at_interface', 11)
        self.backend = kwargs.get('backend', 'scipy')
        self._excitation = {'fluence': [], 'delay_pump': [0], 'pulse_width': [0],
                            'wavelength': 800e-9, 'theta': np.pi/2,
                            # 'polarization': 'p',
                            'multilayer_absorption': True}
        self._distances = np.array([])
        self.boundary_types = ['isolator', 'temperature', 'flux']
        self._boundary_conditions = {
            'top_type': 0,
            'top_value': np.array([]),
            'bottom_type': 0,
            'bottom_value': np.array([]),
            }
        self.ode_options = {
            'method': 'Radau',
            'first_step': None,
            'max_step': np.inf,
            'rtol': 1e-3,
            'atol': 1e-6,
            }
        self.ode_options_matlab = {'RelTol': 1e-3}
        self.matlab_engine = []

    def __str__(self, output=[]):
        """String representation of this class"""

        output = [['excitation fluence', self.excitation['fluence']],
                  ['excitation delay', self.excitation['delay_pump']],
                  ['excitation pulse length', self.excitation['pulse_width']],
                  ['excitation wavelength', self.excitation['wavelength']],
                  ['excitation theta', self.excitation['theta']],
                  # ['excitation polarization', self.excitation['polarization']],
                  ['excitation multilayer absorption', self.excitation['multilayer_absorption']],
                  ['heat diffusion', self.heat_diffusion],
                  ['interpolate at interfaces', self.intp_at_interface],
                  ['backend', self.backend],
                  ['distances', 'no distance mesh is set for heat diffusion calculations'
                   if self.distances.size == 0 else
                   'a distance mesh is set for heat diffusion calculations.'],
                  ['top boundary type', self.boundary_conditions['top_type']],
                  ] + output

        if self._boundary_conditions['top_type'] == 1:
            output += [['top boundary temperature',
                        str(self.boundary_conditions['top_value'])]]
        elif self._boundary_conditions['top_type'] == 2:
            output += [['top boundary flux',
                        str(self.boundary_conditions['top_value'])]]

        output += [['bottom boundary type', self.boundary_conditions['bottom_type']]]

        if self._boundary_conditions['bottom_type'] == 1:
            output += [['bottom boundary temperature',
                        str(self.boundary_conditions['bottom_value'])]]
        elif self._boundary_conditions['bottom_type'] == 2:
            output += [['bottom boundary flux',
                        str(self.boundary_conditions['bottom_value'])]]

        class_str = 'Heat simulation properties:\n\n'
        class_str += super().__str__(output)

        return class_str

    def get_hash(self, delays, init_temp, **kwargs):
        """get_hash

        Returns a unique hash given by the delays, and init_temp as
        well as the sample structure hash for relevant thermal parameters.

        """
        param = [delays, init_temp, self.heat_diffusion,
                 self.intp_at_interface, self.excitation, self.distances]

        for key, value in kwargs.items():
            param.append(value)

        return self.S.get_hash(types='heat') + '_' + make_hash_md5(param)

    def check_initial_temperature(self, init_temp, distances=[]):
        """check_initial_temperature

        An inital temperature for a heat simulation can be either a
        single temperature which is assumed to be valid for all layers
        in the structure or a temeprature profile is given with one
        temperature for each layer in the structure and for each subsystem.

        Args:
            init_temp (float, ndarray): initial temperature

        """

        try:
            init_temp = init_temp.to('K').magnitude
        except AttributeError:
            pass

        if distances == []:
            N = self.S.get_number_of_layers()
        else:
            N = len(distances)

        K = self.S.num_sub_systems
        # check size of initTemp
        if np.size(init_temp) == 1:
            # it is the same initial temperature for all layers
            init_temp = init_temp*np.ones([N, K])
        elif np.shape(init_temp) != (N, K):
            # init_temp is a vector but has not as many elements as layers
            raise ValueError('The initial temperature vector must have 1 or '
                             'NxK elements, where N is the number of layers '
                             'in the structure and K the number of subsystems!')

        return init_temp

    def check_excitation(self, delays):
        """check_excitation

        The optical excitation is a dictionary with fluence
        :math:`F` [J/m²], delays :math:`t` [s] of the pump events, and pulse
        width :math:`\tau` [s]. :math:`N` is the number of pump events.

        Traverse excitation vector to update the `delay_pump` :math:`t_p`
        vector for finite pulse durations :math:`w(i)` as follows

        .. math::

            t_p(i)-\mbox{window}\cdot w(i):t_p(i)+\mbox{window}\cdot w(i):
            w(i)/\mbox{intp}

        and to combine excitations which have overlapping intervalls.

        """
        delays = delays.to('s').magnitude
        fluence = self._excitation['fluence']
        delay_pump = self._excitation['delay_pump']
        pulse_width = self._excitation['pulse_width']

        # throw warnings if heat diffusion should be enabled
        if (self.S.num_sub_systems > 1) and not self.heat_diffusion:
            warnings.warn('If you are introducing more than 1 subsystem you '
                          'should enable heat diffusion!')

        if np.sum(pulse_width) > 0 and not self.heat_diffusion:
            pulse_width = np.zeros_like(fluence)
            warnings.warn('The effect of finite pulse duration of the excitation '
                          'is only considered if heat diffusion is enabled! '
                          'All pulse durations are set to 0!')

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
            # print(delay_pump[k] + window*pulse_width[k])
            # print(intervall)
            # update the new excitation list
            n_excitation.append([intervall,
                                 [t[0] for t in temp],
                                 [t[1] for t in temp],
                                 [t[2] for t in temp]])
            i = k+1  # increase counter

        # traverse the n_excitation list and add additional time vectors between
        # the pump events for the later temperature calculation
        res = []  # initialize the result list

        # check for delay < delay_pump[0]
        if np.size(delays[delays < np.min(n_excitation[0][0])]) > 0:
            res.append([delays[delays < n_excitation[0][0][0]], [], [], []])
        else:
            warnings.warn('Please add more delay steps before the first excitation!')

        # traverse n_excitation
        for i, excitation in enumerate(n_excitation):
            res.append(excitation)
            # print(excitation[0][-1])
            # print(n_excitation[i+1][0][0])
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

        Args:
            distances (ndarray[float]): spatial grid for calculation

        Returns a vector of the absorption profile calculated either by
        Lambert-Beers law or by a mulitlayer absorption formalism

        """
        if self._excitation['multilayer_absorption']:
            dAdz, _, _, _ = self.get_multilayers_absorption_profile(distances)
            return dAdz
        else:
            return self.get_Lambert_Beer_absorption_profile(distances)

    def get_Lambert_Beer_absorption_profile(self, distances=[]):
        r"""get_Lambert_Beer_absorption_profile

        Args:
            distances (ndarray[float]): spatial grid for calculation

        Returns a vector of the absorption profile derived from Lambert-Beer's
        law. The transmission is given by:

        .. math:: \tau = \frac{I}{I_0} =  \exp(-z/ \zeta)

        and the absorption by:

        .. math:: \alpha = 1 - \tau =  1 - \exp(-z/\zeta)

        The absorption profile can be derived from the spatial derivative:

        .. math::

            \frac{\mbox{d}\alpha(z)}{\mbox{d}z} = \frac{1}{\zeta}
            \exp(-z/\zeta)

        """
        if distances == []:
            # if no distances are set, calculate the extinction on
            # the middle of each unit cell
            d_start, _, distances = self.S.get_distances_of_layers(False)
        else:
            d_start, _, _ = self.S.get_distances_of_layers(False)

        interfaces = self.S.get_distances_of_interfaces(False)

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

    def get_multilayers_absorption_profile(self, distances=[]):
        """get_multilayers_absorption_profile

        Calculates the intensity, absorption and temperature increase profiles
        in each layer of a multilayers structure for p-polarized light.

        Calculation of intensity, absorption and temperature increase profiles
        in multilayers.

        Calculation based on the method by K. Ohta and H. Ishida, Appl. Opt. 29,
        2466 (1990).
        Code developed Matlab for L. Le Guyader & al., Phys. Rev. B 87, 054437 (2013).

        Copyright (2012-2014) Loïc Le Guyader <loic.le_guyader@helmholtz-berlin.de>

        Args:
            distances (ndarray[float]): spatial grid for calculation

        Returns:
            dAdz (ndarray[float]): differential absorption within each layer
            Ints (ndarray[float]): intensity profiles within each layer
            R_total (float): total amount of reflection from the multilayer
            T_total (float): total transmission in the last layer of the multilayer

        """

        if distances == []:
            # if no distances are set, calculate the extinction on
            # the middle of each unit cell
            d_start, _, distances = self.S.get_distances_of_layers(False)
        else:
            d_start, _, _ = self.S.get_distances_of_layers(False)

        interfaces = self.S.get_distances_of_interfaces(False)
        N = len(interfaces)
        # if a substrate is included add it at the end
        if self.S.substrate != []:
            M = N + 1
        else:
            M = N

        opt_ref_indices = np.empty(M, dtype=complex)
        thicknesses = np.empty(M, dtype=float)

        # first layer is vacuum/air
        opt_ref_indices[0] = 1+0.0j
        thicknesses[0] = 1e-9

        for i in range(N-1):
            index = finderb(interfaces[i], d_start)
            layer = self.S.get_layer_handle(index[0])
            opt_ref_indices[i+1] = layer.opt_ref_index
            thicknesses[i+1] = interfaces[i+1]-interfaces[i]

        if M != N:
            opt_ref_indices[N] = self.S.substrate.get_layer_handle(0).opt_ref_index
            thicknesses[N] = self.S.substrate.get_thickness(False)

        # Snell laws
        alpha = np.empty(M, dtype=complex)
        alpha[0] = np.pi/2 - self._excitation['theta']
        alpha[1:] = np.arcsin(opt_ref_indices[0]/opt_ref_indices[1:]*np.sin(alpha[0]))

        # fresnel coefficient
        rfresnel = np.empty(M-1, dtype=complex)
        tfresnel = np.empty(M-1, dtype=complex)

        # if self._excitation['polarization'] == 's':
        #     rfresnel[:] = (opt_ref_indices[0:-1]*np.cos(alpha[0:-1])
        #                    - opt_ref_indices[1:]*np.cos(alpha[1:])) \
        #         / (opt_ref_indices[0:-1]*np.cos(alpha[0:-1])
        #            + opt_ref_indices[1:]*np.cos(alpha[1:]))
        #     tfresnel[:] = 2.0*opt_ref_indices[0:-1]*np.cos(alpha[0:-1]) \
        #         / (opt_ref_indices[0:-1]*np.cos(alpha[0:-1])
        #            + opt_ref_indices[1:]*np.cos(alpha[1:]))
        # else:  # p-polarization
        rfresnel[:] = (opt_ref_indices[1:]*np.cos(alpha[0:-1])
                       - opt_ref_indices[0:-1]*np.cos(alpha[1:])) \
            / (opt_ref_indices[1:]*np.cos(alpha[0:-1])
               + opt_ref_indices[0:-1]*np.cos(alpha[1:]))
        tfresnel[:] = 2.0*opt_ref_indices[0:-1]*np.cos(alpha[0:-1]) \
            / (opt_ref_indices[1:]*np.cos(alpha[0:-1])
               + opt_ref_indices[0:-1]*np.cos(alpha[1:]))

        # interface change matrix
        Jnm = np.empty((2, 2, M-1), dtype=complex)
        Jnm[0, 0, :] = 1.0/tfresnel
        Jnm[0, 1, :] = rfresnel/tfresnel
        Jnm[1, 0, :] = rfresnel/tfresnel
        Jnm[1, 1, :] = 1.0/tfresnel

        # calculating z-component of the wave vector
        k_z = 2.0*np.pi/self._excitation['wavelength']*opt_ref_indices*np.cos(alpha)

        # phase changes
        beta = k_z*thicknesses
        Ln = np.empty((2, 2, M-1), dtype=complex)
        Ln[:, :, 0] = [[1, 0], [0, 1]]
        Ln[0, 0, 1:] = np.exp(-1.0j*beta[1:-1])
        Ln[0, 1, 1:] = 0
        Ln[1, 0, 1:] = 0
        Ln[1, 1, 1:] = np.exp(1.0j*beta[1:-1])

        # calculating propagation matrix
        S = Jnm[:, :, M-2]
        for i in range(M-3, -1, -1):
            S = np.dot(Jnm[:, :, i], np.dot(Ln[:, :, i+1], S))

        # Total transmission and reflection of the multilayer
        R_total = np.abs(S[1, 0]/S[0, 0])**2
        T_total = np.asscalar(np.real(np.conj(opt_ref_indices[M-1])*np.cos(alpha[M-1])
                                      / (opt_ref_indices[0]*np.cos(alpha[0])))
                              * np.abs(1/S[0, 0])**2)

        # calculating D matrix for intermediate field
        Dn = np.empty((2, 2, M), dtype=complex)
        Dn[0, 0, M-1] = np.asscalar(1.0/S[0, 0])
        Dn[0, 1, M-1] = 0.0
        Dn[1, 0, M-1] = 0.0
        Dn[1, 1, M-1] = np.asscalar(1.0/S[0, 0])
        for i in range(M-2, -1, -1):
            Temp = np.dot(Ln[:, :, i], np.dot(Jnm[:, :, i], Dn[:, :, i+1]))
            Dn[0, 0, i] = Temp[0, 0]
            Dn[0, 1, i] = Temp[0, 1]
            Dn[1, 0, i] = Temp[1, 0]
            Dn[1, 1, i] = Temp[1, 1]

        K = len(distances)
        Ints = np.empty(K, dtype=float)  # initialize relative intensities
        dAdz = np.empty(K, dtype=float)  # initialize relative absorbed energies
        k = 0  # counter for first layer
        for i in range(1, N):
            # get all distances in the current layer we have to
            # calculate the absorption profile for
            if i >= N-1:  # last layer
                z = distances[np.logical_and(distances >= interfaces[i-1],
                                             distances <= interfaces[i])]
            else:
                z = distances[np.logical_and(distances >= interfaces[i-1],
                                             distances < interfaces[i])]
            m = len(z)
            z -= interfaces[i-1]  # relative positon within the layer
            # For each layer except the first, compute Intensity, Absorption
            Ep = Dn[0, 0, i]*np.exp(1.0j*k_z[i]*z)
            Em = Dn[1, 0, i]*np.exp(-1.0j*k_z[i]*z)
            Etx = 0.0*Ep + 0.0*Em
            Ety = np.cos(alpha[i])*Ep - np.cos(alpha[i])*Em
            Etz = -np.sin(alpha[i])*Ep - np.sin(alpha[i])*Em
            Ints[k:k+m] = np.real(Etx * np.conj(Etx) + Ety*np.conj(Ety) + Etz*np.conj(Etz))
            dAdz[k:k+m] = np.real(opt_ref_indices[i]*np.cos(alpha[i])
                                  / (opt_ref_indices[0]*np.cos(alpha[0]))) \
                * 2.0*np.imag(k_z[i])*Ints[k:k+m]

            k = k+m  # set the counter

        return dAdz, Ints, R_total, T_total

    def get_temperature_after_delta_excitation(self, fluence, init_temp, distances=[]):
        r"""get_temperature_after_delta_excitation

        Args:
            fluence (float/pint quantity): incident fluence in J/m² as float
                or as pint quantity
            init_temp (float): initial temperature of the sample either
                homogeneous temperature across the whole sample or as array
                for every layer of the sample structure

        Returns a vector of the end temperature and temperature change
        for each layer of the sample structure after an optical
        exciation with a fluence :math:`F` [J/m^2] and an inital temperature
        :math:`T_1` [K]:

        .. math:: \Delta E = \int_{T_1}^{T_2} m \, c(T)\, \mbox{d}T

        where :math:`\Delta E` is the absorbed energy in each layer and
        :math:`c(T)` is the temperature-dependent heat capacity [J/kg K] and
        :math:`m` is the mass [kg].

        The absorbed energy per layer can be linearized from the
        absorption profile :math:`\mbox{d} \alpha / \mbox{d}z` as

        .. math:: \Delta E = \frac{\mbox{d} \alpha}{\mbox{d}z} E_0 \Delta z

        where :math:`E_0` is the initial energy impinging on the first layer
        given by the fluence :math:`F = E / A`.
        :math:`\Delta z` is equal to the thickness of each layer.

        Finally, one has to minimize the following modulus to obtain the
        final temperature :math:`T_2` of each layer:

        .. math::

             \left| \int_{T_1}^{T_2} m c(T)\, temp_mapT - \frac{\mbox{d}\alpha}
             {\mbox{d}z} E_0 \Delta z \right| \stackrel{!}{=} 0

        """
        # initialize
        t1 = time()
        if distances == []:
            # if no distances are set, calculate the extinction on
            # the middle of each unit cell
            d_start, _, distances = self.S.get_distances_of_layers(False)
        else:
            d_start, _, _ = self.S.get_distances_of_layers(False)

        # absorption profile from Lambert-Beer's law or multilayer absorption
        dalpha_dz = self.get_absorption_profile(distances)

        try:
            fluence = fluence.to('J/m**2').magnitude
        except AttributeError:
            pass

        int_heat_capacities = self.S.get_layer_property_vector('_int_heat_capacity')
        thicknesses = self.S.get_layer_property_vector('_thickness')
        masses = self.S.get_layer_property_vector('_mass')
        areas = self.S.get_layer_property_vector('_area')
        E0 = np.array(fluence)*areas[1]  # mass are normalized to 1Ang^2

        # check the intial temperature
        init_temp = self.check_initial_temperature(init_temp, distances)
        final_temp = init_temp
        # traverse layers
        for i, dist in enumerate(distances):
            idx = finderb(dist, d_start)[0]
            if dalpha_dz[i] > 0:
                # if there is absorption in the current layer
                del_E = dalpha_dz[i]*E0*thicknesses[idx]

                def fun(final_temp):
                    return (masses[idx]*(int_heat_capacities[idx][0](final_temp)
                                         - int_heat_capacities[idx][0](init_temp[i, 0]))
                            - del_E)
                final_temp[i, 0] = brentq(fun, init_temp[i, 0], 1e5)
        delta_T = final_temp - init_temp  # this is the temperature change
        self.disp_message('Elapsed time for _temperature_after_delta_excitation_:'
                          ' {:f} s'.format(time()-t1))
        return final_temp, delta_T

    def get_temp_map(self, delays, init_temp):
        """get_temp_map

        Returns a tempperature profile for the sample structure after optical
        excitation. The result can be saved using an unique hash of the sample
        and the simulation parameters in order to reuse it.

        """
        init_temp = self.check_initial_temperature(init_temp)  # check the intial temperature
        filename = 'temp_map_' \
                   + self.get_hash(delays, init_temp) \
                   + '.npy'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            temp_map, delta_temp_map, checked_excitations = np.load(full_filename)
            self.disp_message('_temp_map_ loaded from file:\n\t' + filename)
        else:
            # file does not exist so calculate and save
            temp_map, delta_temp_map, checked_excitations = \
                self.calc_temp_map(delays, init_temp)
            self.save(full_filename, [temp_map, delta_temp_map, checked_excitations], '_temp_map_')
        return temp_map, delta_temp_map, checked_excitations

    def calc_temp_map(self, delays, input_init_temp):
        """calc_temp_map

        Calculates the temp_map and temp_map difference for a given delay
        vector, exciation and initial temperature. Heat diffusion can be
        included if _heat_diffusion = true_.

        """
        t1 = time()
        # initialize
        L = self.S.get_number_of_layers()
        K = self.S.num_sub_systems

        # there is an initial time step for the init_temp - we will remove it later on
        # check the intial temperature
        input_init_temp = self.check_initial_temperature(input_init_temp)
        checked_excitation, _, _, _ = self.check_excitation(delays)  # check excitation
        _, _, d_mid = self.S.get_distances_of_layers(False)

        if self.heat_diffusion:
            # in case heat diffusion is enabled all calculation are first done
            # on an interpolated grid at the interfaces (or defined by the user)
            if len(self.distances) == 0:
                # no user-defined distaces are given, so calculate heat diffusion
                # by layer and also interpolate at the interfaces
                distances, _ = self.S.interp_distance_at_interfaces(self.intp_at_interface, False)
            else:
                # a user-defined distances vector is given, so determine the
                # indicies for final assignment per layer
                distances = self._distances
        else:
            distances = d_mid

        N = len(distances)
        temp_map = np.zeros([1, N, K])
        # interpolate the initial temperature onto distance grid
        init_temp = np.zeros([N, K])
        for iii in range(K):
            init_temp[:, iii] = np.interp(distances, d_mid, input_init_temp[:, iii])
        temp_map[0, :, :] = init_temp  # this is initial temperature before the simulation starts

        num_ex = 1  # excitation counter
        excitation_delays = np.array([])
        temp = []
        # traverse excitations
        for i, excitation in enumerate(checked_excitation):
            # reset inital temperature for delta excitation with heat diffusion enabled
            special_init_temp = []
            # extract excitation parameters for the current iteration
            sub_delays = excitation[0]
            excitation_delays = np.concatenate((excitation_delays, sub_delays))
            delay_pump = excitation[1]
            pulse_width = excitation[2]
            fluence = excitation[3]
            # determine if a temperature gradient is present and if
            # heat diffusion is required
            temp_gradient = np.sum(np.sum(np.abs(np.diff(temp_map[-1, :, :], 1, 0))))
            if self.heat_diffusion and (len(sub_delays) > 2) \
                    and ((np.sum(fluence) == 0 and temp_gradient != 0)
                         or (np.sum(fluence) > 0 and np.sum(pulse_width) > 0)
                         or (self._boundary_conditions['top_type']
                             + self._boundary_conditions['bottom_type']) > 0):
                # heat diffusion enabled and more than 2 time steps AND
                # either no excitation with temperature gradient or excitation with finite pulse
                # duration
                if np.sum(fluence) == 0:
                    self.disp_message('Calculating _heat_diffusion_ ...')
                else:
                    self.disp_message('Calculating _heat_diffusion_ for excitation ' +
                                      '{:d}:{:d} ...'.format(num_ex, num_ex+len(fluence)-1))

                start = 0
                stop = 0
                if i > 0:
                    # check if there was a intervall before and add
                    # last time of this intervall to the current
                    sub_delays = np.r_[checked_excitation[i-1][0][-1], sub_delays]
                    start = 1
                if i < len(checked_excitation)-1 and \
                        np.sum(checked_excitation[i+1][3]) > 0 and \
                        np.sum(checked_excitation[i+1][2]) == 0:
                    # there is a next intervall of delta excitation so
                    # we add this time at the end of the current
                    # intervall
                    sub_delays = np.r_[sub_delays, checked_excitation[i+1][0][0]]
                    stop = 1

                # calc heat diffusion
                temp = self.calc_heat_diffusion(init_temp, distances, sub_delays, delay_pump,
                                                pulse_width, fluence)

                if stop == 1:
                    # there is an upcomming delta excitation so we have
                    # to set the initial temperature for this next
                    # intervall seperately
                    special_init_temp = temp[-1, :, :]
                    temp = temp[start:-1, :, :]
                else:
                    temp = temp[start:, :, :].copy()
                # cut the before added time steps
            elif np.sum(fluence) > 0 and (not self.heat_diffusion or
                                          (self.heat_diffusion and
                                           np.sum(pulse_width) == 0)):
                # excitation with no heat diffusion -> only delta excitation
                # possible in this case OR excitation with heat diffusion and
                # pulse_width equal to 0
                temp, _ = self.get_temperature_after_delta_excitation(fluence,
                                                                      init_temp,
                                                                      distances)
                temp = np.reshape(temp, [1, np.size(temp, 0), np.size(temp, 1)])
            else:
                # no excitation and no heat diffusion or not enough time
                # step to calculate heat difusion, so just repeat the
                # initial temperature + every unhandled case
                temp = np.tile(np.reshape(init_temp, [1, N, K]), [len(sub_delays), 1, 1])

            # concat results
            temp_map = np.append(temp_map, temp, axis=0)
            # set the initial temperature for the next iteration
            if len(special_init_temp) > 0:
                init_temp = special_init_temp
            else:
                init_temp = temp_map[-1, :, :].copy()
            # increase excitation counder
            if np.sum(fluence) > 0:
                num_ex += len(fluence)

        if not np.all(excitation_delays == delays.to('s').magnitude) or self.heat_diffusion:
            # if the time grid for the calculation is not the same as
            # the grid to return the results on. Then extrapolate the
            # results on the original delay array but keep the first
            # element in time for the deltaTempMap calculation.
            temp = temp_map.copy()
            temp_map = np.zeros([len(delays)+1, L, K])
            for iii in range(K):
                init = np.interp(d_mid, distances, temp[0, :, iii]).reshape([1, L])
                f = interp2d(distances, excitation_delays, temp[1:, :, iii], kind='linear')
                temp_map[:, :, iii] = np.append(init, f(d_mid, delays.to('s').magnitude), axis=0)

        # calculate the difference temperature map
        delta_temp_map = np.diff(temp_map, axis=0)
        # delete the initial temperature that was added at the beginning
        temp_map = temp_map[1:, :, :]
        self.disp_message('Elapsed time for _temp_map_:'
                          ' {:f} s'.format(time()-t1))
        return np.squeeze(temp_map), np.squeeze(delta_temp_map), checked_excitation

    def calc_heat_diffusion(self, init_temp, distances, delays, delay_pump, pulse_width, fluence):
        r""" calc_heat_diffusion

        Returns a temp_map that is calculated by heat diffusion for a
        given delay array and initial temperature profile.
        Here we have to solve the 1D heat equation:

        .. math::

            c(T) \, \rho \, \frac{\partial T}{\partial t}
            = \frac{\partial}{\partial z}
            \left( k(T) \,\frac{\partial T}{\partial z} \right)
            + S(z,t)

        where :math:`T` is the temperature [K], :math:`z` the distance [m],
        :math:`t` the delay [s], :math:`c(T)` the temperature dependent heat
        capacity [J/kg K], :math:`\rho` the density [kg/m^3] and :math:`k(T)`
        is the temperature-dependent thermal conductivity [W/m K] and
        :math:`S(z,t)` is a source term [W/m^3].

        """
        t1 = time()
        M = len(delays)
        K = self.S.num_sub_systems
        init_temp = self.check_initial_temperature(init_temp, distances)
        d_start, _, d_mid = self.S.get_distances_of_layers(False)

        d_distances = np.diff(distances)
        N = len(distances)
        dalpha_dz = self.get_absorption_profile(distances)

        if self.backend == 'matlab':
            # use of matlab backend for heat diffusion calculation
            # first try to import required python-matlab bridge
            try:
                import matlab.engine
            except ImportError:
                raise Warning('You need to have a working MATLAB installation '
                              'on your machine with installed matlab.engine for '
                              'Python.\n'
                              'See '
                              'https://de.mathworks.com/help/matlab/matlab-engine-for-python.html '
                              'for details.')

            # start MATLAB engine if not already done
            if self.matlab_engine == []:
                self.matlab_engine = matlab.engine.start_matlab()

            # add path of matlab script to matlab's search path
            matlab_path = path.join(path.dirname(path.abspath(__file__)), 'matlab')
            self.matlab_engine.addpath(matlab_path)

            temp_map = self.matlab_engine.calc_heat_diffusion(
                K,
                matlab.double(init_temp.tolist()),
                matlab.double(d_start.tolist()),
                matlab.double(distances.tolist()),
                matlab.double(fluence),
                matlab.double(pulse_width),
                matlab.double(delay_pump),
                matlab.double(dalpha_dz.tolist()),
                matlab.double(delays.tolist()),
                self.S.get_layer_property_vector('therm_cond_str'),
                self.S.get_layer_property_vector('heat_capacity_str'),
                matlab.double(self.S.get_layer_property_vector('_density').tolist()),
                self.S.get_layer_property_vector('sub_system_coupling_str'),
                matlab.int32([self._boundary_conditions['top_type']+1]),
                matlab.double([self._boundary_conditions['top_value'].tolist()]),
                matlab.int32([self._boundary_conditions['bottom_type']+1]),
                matlab.double([self._boundary_conditions['bottom_value'].tolist()]),
                self.ode_options
            )
        else:
            # use python scipy backend
            if self.progress_bar:  # with tqdm progressbar
                pbar = tqdm()
                pbar.set_description('Delay = {:.3f} ps'.format(delays[0]*1e12))
                state = [delays[0], abs(delays[-1]-delays[0])/100]
            else:  # without progressbar
                pbar = None
                state = None

            indicies = finderb(distances, d_start)
            # solve pdepe with method-of-lines
            sol = solve_ivp(
                Heat.odefunc,
                [delays[0], delays[-1]],
                init_temp[:, 0],
                args=(d_distances,
                      d_start,
                      self.S.get_layer_property_vector('therm_cond'),
                      self.S.get_layer_property_vector('heat_capacity'),
                      self.S.get_layer_property_vector('_density'),
                      indicies,
                      N,
                      dalpha_dz,
                      fluence,
                      delay_pump,
                      pulse_width,
                      self._boundary_conditions['top_type'],
                      self._boundary_conditions['top_value'],
                      self._boundary_conditions['bottom_type'],
                      self._boundary_conditions['bottom_value'],
                      pbar, state),
                t_eval=delays,
                dense_output=True,
                **self.ode_options)

            if pbar is not None:  # close tqdm progressbar if used
                pbar.close()
            temp_map = sol.y.T

        temp_map = np.array(temp_map).reshape([M, N, K])
        if fluence == []:
            self.disp_message('Elapsed time for _heat_diffusion_: {:f} s'.format(time()-t1))
        else:
            self.disp_message('Elapsed time for _heat_diffusion_ with {:d} '
                              'excitation(s): {:f} s'.format(len(fluence), time()-t1))

        return temp_map

    @staticmethod
    def odefunc(t, u, d_x_grid, x, thermal_conds, heat_capacities, densities,
                indicies, N, dalpha_dz, fluence, delay_pump, pulse_length,
                bc_top_type, bc_top_value, bc_bottom_type, bc_bottom_value,
                pbar, state):
        # state is a list containing last updated time t:
        # state = [last_t, dt]
        # I used a list because its values can be carried between function
        # calls throughout the ODE integration
        last_t, dt = state
        n = int((t - last_t)/dt)

        if n >= 1:
            pbar.update(n)
            pbar.set_description('Delay = {:.3f} ps'.format(t*1e12))
            state[0] = t
        elif n < 0:
            state[0] = t

        dudt = np.zeros(N)
        ks = np.zeros(N)
        cs = np.zeros(N)
        rhos = np.zeros(N)
        if fluence != []:
            source = \
                dalpha_dz * multi_gauss(t, s=pulse_length, x0=delay_pump, A=fluence)
        else:
            source = np.zeros(N)

        for i in range(N):
            idx = indicies[i]
            ks[i] = thermal_conds[idx][0](u[i])
            cs[i] = heat_capacities[idx][0](u[i])
            rhos[i] = densities[idx]

        # boundary conditions
        if bc_top_type == 1:  # temperature
            u[0] = bc_top_value
        elif bc_top_type == 2:  # flux
            dudt[0] = ((ks[0]*(u[1] - u[0])/d_x_grid[0] + bc_top_value)/d_x_grid[0]
                       + source[0])/cs[0]/rhos[0]
        else:  # isolator
            dudt[0] = (ks[0]*(u[1] - u[0])/d_x_grid[0]**2 + source[0])/cs[0]/rhos[0]

        if bc_bottom_type == 1:  # temperature
            u[-1] = bc_bottom_value
        elif bc_bottom_type == 2:  # flux
            dudt[-1] = ((bc_bottom_value - ks[-1]*(u[-1] - u[-2])/d_x_grid[-1])/d_x_grid[-1]
                        + source[-1])/cs[-1]/rhos[-1]
        else:  # isolator
            dudt[-1] = (ks[-1]*(u[-1] - u[-2])/d_x_grid[-1]**2 + source[-1])/cs[-1]/rhos[-1]

        # calculate derivative
        for i in range(1, N-1):
            dudt[i] = ((
                 ks[i+1]*(u[i+1] - u[i])/(d_x_grid[i]) - ks[i]*(u[i] - u[i-1])/(d_x_grid[i-1]))
                / ((d_x_grid[i]+d_x_grid[i-1])/2) + source[i])/cs[i]/rhos[i]

        return dudt

    @property
    def backend(self):
        """str: backend"""

        return self._backend

    @backend.setter
    def backend(self, backend):
        """set.backend"""

        if backend in ['scipy', 'matlab']:
            self._backend = backend
        else:
            warnings.warn('Backend must be either _scipy_ or _matlab_. '
                          'Set to _scipy_ default!')
            self._backend = 'scipy'

    @property
    def excitation(self):
        """dict: excitation parameters

        Convert to from default SI units to real quantities

        """
        excitation = {'fluence': Q_(self._excitation['fluence'], u.J/u.m**2).to('mJ/cm**2'),
                      'delay_pump': Q_(self._excitation['delay_pump'], u.s).to('ps'),
                      'pulse_width': Q_(self._excitation['pulse_width'], u.s).to('ps'),
                      'wavelength': Q_(self._excitation['wavelength'], u.m).to('nm'),
                      'theta': Q_(self._excitation['theta'], u.rad).to('deg'),
                      # 'polarization': self._excitation['polarization'],
                      'multilayer_absorption': self._excitation['multilayer_absorption']}

        return excitation

    @excitation.setter
    def excitation(self, excitation):
        """set.excitation"""

        # check the size of excitation, if we have a multipulse excitation
        if isinstance(excitation, Q_):
            # just a fluence is given
            self._excitation['fluence'] = [excitation.to('J/m**2').magnitude]
        elif isinstance(excitation, dict):
            if 'fluence' in excitation:
                self._excitation['fluence'] = excitation['fluence'].to('J/m**2').magnitude
            if 'delay_pump' in excitation:
                self._excitation['delay_pump'] = excitation['delay_pump'].to('s').magnitude
            if 'pulse_width' in excitation:
                self._excitation['pulse_width'] = excitation['pulse_width'].to('s').magnitude
            if 'wavelength' in excitation:
                self._excitation['wavelength'] = excitation['wavelength'].to('m').magnitude
            if 'theta' in excitation:
                self._excitation['theta'] = excitation['theta'].to('rad').magnitude
            # if 'polarization' in excitation:
            #     if excitation['polarization'] in ['s', 'p']:
            #         self._excitation['polarization'] = excitation['polarization']
            #     else:
            #         self._excitation['polarization'] = 'p'
            #         raise Warning('Polarization musted be either _s_ or _p_!')
            if 'multilayer_absorption' in excitation:
                self._excitation['multilayer_absorption'] = \
                    bool(excitation['multilayer_absorption'])
        else:
            raise ValueError('_excitation_ must be either a float/int or dict!')

        if not (len(self._excitation['fluence'])
                == len(self._excitation['delay_pump'])
                == len(self._excitation['pulse_width'])):
            raise ValueError('Elements of excitation dict, fluence, delay_pump, '
                             'and pulse_width must have '
                             'the same number of elements!')

        # check the elements of the delay_pump vector
        if len(self._excitation['delay_pump']) != len(np.unique(self._excitation['delay_pump'])):
            raise ValueError('The excitations have to be unique in delays!')
        else:
            self._excitation['delay_pump'] = np.sort(self._excitation['delay_pump'])

    @property
    def boundary_conditions(self):
        """dict: boundary_conditions

        Convert to from default SI units to real quantities

        """
        boundary_conditions = {'top_type':
                               self.boundary_types[self._boundary_conditions['top_type']],
                               }

        if self._boundary_conditions['top_type'] == 1:
            boundary_conditions['top_value'] = Q_(self._boundary_conditions['top_value'],
                                                  'K')
        elif self._boundary_conditions['top_type'] == 2:
            boundary_conditions['top_value'] = Q_(self._boundary_conditions['top_value'],
                                                  'W/m**2')

        boundary_conditions['bottom_type'] = \
            self.boundary_types[self._boundary_conditions['bottom_type']]

        if self._boundary_conditions['bottom_type'] == 1:
            boundary_conditions['bottom_value'] = Q_(self._boundary_conditions['bottom_value'],
                                                     'K')
        elif self._boundary_conditions['bottom_type'] == 2:
            boundary_conditions['bottom_value'] = Q_(self._boundary_conditions['bottom_value'],
                                                     'W/m**2')
        return boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        """set.boundary_conditions"""

        if isinstance(boundary_conditions, dict):
            if 'top_type' in boundary_conditions:
                try:
                    btype = self.boundary_types.index(boundary_conditions['top_type'])
                except ValueError:
                    raise ValueError('boundary_type must be either _isolator_, '
                                     '_temperature_ or _flux_!')

                self._boundary_conditions['top_type'] = btype
            if 'bottom_type' in boundary_conditions:
                try:
                    btype = self.boundary_types.index(boundary_conditions['bottom_type'])
                except ValueError:
                    raise ValueError('boundary_type must be either _isolator_, '
                                     '_temperature_ or _flux_!')

                self._boundary_conditions['bottom_type'] = btype
            if 'top_value' in boundary_conditions:
                if self._boundary_conditions['top_type'] == 1:
                    self._boundary_conditions['top_value'] = \
                        np.array(boundary_conditions['top_value'].to('K').magnitude)
                elif self._boundary_conditions['top_type'] == 2:
                    self._boundary_conditions['top_value'] = \
                        np.array(boundary_conditions['top_value'].to('W/m**2').magnitude)
            if 'bottom_value' in boundary_conditions:
                if self._boundary_conditions['bottom_type'] == 1:
                    self._boundary_conditions['bottom_value'] = \
                        np.array(boundary_conditions['bottom_value'].to('K').magnitude)
                elif self._boundary_conditions['bottom_type'] == 2:
                    self._boundary_conditions['bottom_value'] = \
                        np.array(boundary_conditions['bottom_value'].to('W/m**2').magnitude)
        else:
            raise ValueError('_boundary_conditions_ must be a dict!')

        K = self.S.num_sub_systems
        if (self._boundary_conditions['top_type'] > 0) \
                and (np.size(self._boundary_conditions['top_value']) != K):
            raise ValueError('Non-isolating top boundary conditions must have the '
                             'same dimensionality as the numer of sub-systems K!')
        if (self._boundary_conditions['bottom_type'] > 0) \
                and (np.size(self._boundary_conditions['bottom_value']) != K):
            raise ValueError('Non-isolating bottom boundary conditions must have the '
                             'same dimensionality as the numer of sub-systems K!')

    @property
    def distances(self):
        """float: distances for heat diffusion [m]"""
        return Q_(self._distances, u.meter).to('nm')

    @distances.setter
    def distances(self, distances):
        """set.distances"""
        self._distances = distances.to_base_units().magnitude

    def __del__(self):
        # stop matlab engine if exists
        try:
            self.matlab_engine.quit()
        except AttributeError:
            pass
