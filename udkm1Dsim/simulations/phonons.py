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

__all__ = ['Phonon', 'PhononNum']

__docformat__ = 'restructuredtext'

from .simulation import Simulation
from ..helpers import make_hash_md5, finderb
import numpy as np
from os import path
from time import time
from scipy.integrate import solve_ivp
from tqdm.notebook import tqdm


class Phonon(Simulation):
    """Phonon

    Base class for phonon simulations.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expanison without
            coherent phonon dynamics.

    Attributes:
        S (Structure): sample structure to calculate simulations on.
        force_recalc (boolean): force recalculation of results.
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expanison without
            coherent phonon dynamics.

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.only_heat = kwargs.get('only_heat', False)

    def __str__(self, output=[]):
        """String representation of this class"""

        output = [['only heat', self.only_heat],
                  ] + output

        class_str = 'Phonon simulation properties:\n\n'
        class_str += super().__str__(output)

        return class_str

    def get_hash(self, delays, temp_map, delta_temp_map, **kwargs):
        """get_hash

        Calculates an unique hash given by the ``delays``, and ``temp_map``
        and ``delta_temp_map`` as well as the sample structure hash for
        relevant lattice parameters.

        Args:
            delays (ndarray[float]): delay grid for the simulation.
            temp_map (ndarray[float]): spatio-temporal temperature profile.
            delta_temp_map (ndarray[float]): spatio-temporal temperature
                difference profile.
            **kwargs (float, optional): optional parameters.

        Returns:
            hash (str): unique hash.

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

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.

        Returns:
            strains (dict{ndarray[float]}: all strains per unique layer.

        """
        # get the position indices of all unique layers in the sample structure
        positions = self.S.get_all_positions_per_unique_layer()
        strains = {}

        for key, value in positions.items():
            strains[key] = np.sort(np.unique(strain_map[:, value].flatten()))

        return strains

    def get_reduced_strains_per_unique_layer(self, strain_map, N=100):
        """get_reduced_strains_per_unique_layer

        Calculates all strains per unique layer that are given by the
        input ``strain_map``, but with a reduced number. The reduction is done
        by equally spacing the strains between the ``min`` and ``max`` strain
        with a given number :math:`N`, which can be also an array of the
        :math:`len(N) = L`, where :math:`L` is the number of unique layers.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            N (int, optional): number of reduced strains. Defaults to 100.

        Returns:
            strains (dict{ndarray[float]}: reduced strains per unique layer.

        """
        # initialize
        all_strains = self.get_all_strains_per_unique_layer(strain_map)
        L = len(all_strains)  # Nb. of unique layers
        strains = {}

        if np.size(N) == 1:
            N = N*np.ones([L, 1])
        elif np.size(N) != L:
            raise ValueError('The dimension of N must be either 1 or the number '
                             'of unique layers the structure!')

        for i, (key, value) in enumerate(all_strains.items()):
            min_strain = np.min(value)
            max_strain = np.max(value)
            strains[key] = np.sort(np.unique(
                np.r_[0, np.linspace(min_strain, max_strain, int(N[i]))]))

        return strains

    def check_temp_maps(self, temp_map, delta_temp_map, delays):
        """check_temp_maps

        Check temperature profiles for correct dimensions.

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature profile.
            delta_temp_map (ndarray[float]): spatio-temporal temperature
                difference profile.
            delays (ndarray[float]): delay grid for the simulation.

        Returns:
            (tuple):
            - *temp_map (ndarray[float])* - checked spatio-temporal temperature
              profile.
            - *delta_temp_map (ndarray[float])* - checked spatio-temporal
              differential temperature profile.

        """
        M = len(delays)
        L = self.S.get_number_of_layers()
        K = self.S.num_sub_systems

        # check size of delta_temp_map
        if K == 1:
            if np.shape(delta_temp_map) == (1, L):
                temp = delta_temp_map
                delta_temp_map = np.zeros([M, L])
                delta_temp_map[0, :] = temp
            elif (np.size(delta_temp_map, 0) != M) or (np.size(delta_temp_map, 1) != L):
                raise ValueError('The given temperature difference map does not have the '
                                 'dimension M x L, where K is the number of delay steps '
                                 'and L the number of layers!')
        else:
            if np.shape(delta_temp_map) == (1, L, K):
                temp = delta_temp_map
                delta_temp_map = np.zeros([K, L, K])
                delta_temp_map[0, :, :] = temp
            elif ((np.size(delta_temp_map, 0) != M)
                  or (np.size(delta_temp_map, 1) != L)
                  or (np.size(delta_temp_map, 2) != K)):
                raise ValueError('The given temperature difference map does not have the '
                                 'dimension M x L x K, where M is the number of delay steps '
                                 'and L the number of layers and K is the number of subsystems!')

        if np.shape(temp_map) != np.shape(delta_temp_map):
            raise ValueError('The temperature map does not have the same size as the '
                             'temperature difference map!')

        return temp_map, delta_temp_map

    def calc_sticks_from_temp_map(self, temp_map, delta_temp_map):
        r"""calc_sticks_from_temp_map

        Calculates the sticks to insert into the layer springs which model the
        external force (thermal stress). The length of :math:`l_i` of the
        :math:`i`-th spacer stick is calculated from the temperature-dependent
        linear thermal expansion :math:`\alpha(T)` of the layer:

        .. math::

            \alpha(T) = \frac{1}{L} \frac{d L}{d T}

        which results after integration in

        .. math::

            l = \Delta L = L_1 \exp(A(T_2) - A(T_1)) - L_1

        where :math:`A(T)` is the integrated lin. therm. expansion coefficient
        in respect to the temperature :math:`T`. The indices 1 and 2 indicate
        the initial and final state.

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature profile.
            delta_temp_map (ndarray[float]): spatio-temporal temperature
                difference profile.

        Returns:
            (tuple):
            - *sticks (ndarray[float])* - summed spacer sticks.
            - *sticks_sub_systems (ndarray[float])* - spacer sticks per
              sub-system.

        """
        M = np.size(temp_map, 0)  # nb of delay steps
        L = self.S.get_number_of_layers()
        K = self.S.num_sub_systems

        temp_map = np.reshape(temp_map, [M, L, K])
        delta_temp_map = np.reshape(delta_temp_map, [M, L, K])

        thicknesses = self.S.get_layer_property_vector('_thickness')
        int_lin_therm_exps = self.S.get_layer_property_vector('_int_lin_therm_exp')

        # evaluated initial integrated linear thermal expansion from T1 to T2
        int_alpha_T0 = np.zeros([L, K])
        # evaluated integrated linear thermal expansion from T1 to T2
        int_alpha_T = np.zeros([L, K])
        sticks = np.zeros([M, L])  # the sticks inserted in the unit cells
        sticks_sub_systems = np.zeros([M, L, K])  # the sticks for each thermodynamic subsystem

        # calculate initial integrated linear thermal expansion from T1 to T2
        # traverse subsystems
        for ii in range(L):
            for iii in range(K):
                int_alpha_T0[ii, iii] = int_lin_therm_exps[ii][iii](temp_map[0, ii, iii]
                                                                    - delta_temp_map[0, ii, iii])

        # calculate sticks for all subsytsems for all delay steps
        # traverse time
        for i in range(M):
            if np.any(delta_temp_map[i, :]):  # there is a temperature change
                # Calculate new sticks from the integrated linear thermal
                # expansion from initial temperature to current temperature for
                # each subsystem
                # traverse subsystems
                for ii in range(L):
                    for iii in range(K):
                        int_alpha_T[ii, iii] = int_lin_therm_exps[ii][iii](temp_map[i, ii, iii])

                # calculate the length of the sticks of each subsystem and sum
                # them up
                sticks_sub_systems[i, :, :] = np.tile(thicknesses, (K, 1)).T \
                    * np.exp(int_alpha_T-int_alpha_T0) - np.tile(thicknesses, (K, 1)).T
                sticks[i, :] = np.sum(sticks_sub_systems[i, :, :], 1)
            else:  # no temperature change, so keep the current sticks
                if i > 0:
                    sticks_sub_systems[i, :, :] = sticks_sub_systems[i-1, :, :]
                    sticks[i, :] = sticks[i-1, :]
        return sticks, sticks_sub_systems


class PhononNum(Phonon):
    """PhononNum

    Base class for numerical phonon simulations.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expanison without
            coherent phonon dynamics.

    Attributes:
        S (Structure): sample structure to calculate simulations on.
        force_recalc (boolean): force recalculation of results.
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expanison without
            coherent phonon dynamics.
        ode_options (dict): options for scipy solve_ivp ode solver.

    References:

        .. [6] A. Bojahr, M. Herzog, D. Schick, I. Vrejoiu, & M. Bargheer,
           *Calibrated real-time detection of nonlinearly propagating
           strain waves*, `Phys. Rev. B, 86(14), 144306 (2012).
           <http://www.doi.org/10.1103/PhysRevB.86.144306>`_

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.ode_options = {
            'method': 'RK23',
            'first_step': None,
            'max_step': np.inf,
            'rtol': 1e-3,
            'atol': 1e-6,
            }

    def __str__(self, output=[]):
        """String representation of this class"""

        class_str = 'Numerical Phonon simulation properties:\n\n'
        class_str += super().__str__()

        return class_str

    def get_strain_map(self, delays, temp_map, delta_temp_map):
        """get_strain_map

        Returns a strain profile for the sample structure for given temperature
        profile. The result can be saved using an unique hash of the sample
        and the simulation parameters in order to reuse it.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            temp_map (ndarray[float]): spatio-temporal temperature map.
            delta_temp_map (ndarray[float]): spatio-temporal differential
            temperature map.

        Returns:
            strain_map (ndarray[float]): spatio-temporal strain profile.

        """
        filename = 'strain_map_num_' \
                   + self.get_hash(delays, temp_map, delta_temp_map) \
                   + '.npz'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            strain_map = tmp['strain_map']
            self.disp_message('_strain_map_ loaded from file:\n\t' + filename)
        else:
            # file does not exist so calculate and save
            strain_map, sticks_sub_systems, velocities = \
                self.calc_strain_map(delays, temp_map, delta_temp_map)
            self.save(full_filename, {'strain_map': strain_map}, '_strain_map_num_')
        return strain_map

    def calc_strain_map(self, delays, temp_map, delta_temp_map):
        r"""calc_strain_map

        Calculates the ``strain_map`` of the sample structure for a given
        ``temp_map`` and ``delta_temp_map`` and ``delay`` array. Further
        details are given in Ref. [6]_. The coupled differential equations are
        solved for each oscillator in a linear chain of masses and springs:

        .. math::

            m_i\ddot{x}_i = & -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1}) \\
            & + m_i\gamma_i(\dot{x}_i-\dot{x}_{i-1}) + F_i^{heat}(t)

        where :math:`x_i(t) = z_{i}(t) - z_i^0` is the shift of each layer.
        :math:`m_i` is the mass and :math:`k_i = m_i\, v_i^2/c_i^2` is the
        spring constant of each layer. Furthermore an empirical damping term
        :math:`F_i^{damp} = \gamma_i(\dot{x}_i-\dot{x}_{i-1})` is introduced
        and the external force (thermal stress) :math:`F_i^{heat}(t)`.
        The thermal stresses are modelled as spacer sticks which are
        calculated from the linear thermal expansion coefficients. The
        equation of motion can be reformulated as:

        .. math::

            m_i\ddot{x}_i = F_i^{spring} + F_i^{damp} + F_i^{heat}(t)

        The numerical solution also allows for non-harmonic inter-atomic
        potentials of up to the order :math:`M`. Accordingly
        :math:`k_i = (k_i^1 \ldots k_i^{M-1})` can be an array accounting for
        higher orders of the potential which is in the harmonic case purely
        quadratic (:math:`k_i = k_i^1`). The resulting force from the
        displacement of the springs

        .. math::

            F_i^{spring} = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1})

        includes:

        .. math::

            k_i(x_i-x_{i-1}) = \sum_{j=1}^{M-1} k_i^j (x_i-x_{i-1})^j

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            temp_map (ndarray[float]): spatio-temporal temperature map.
            delta_temp_map (ndarray[float]): spatio-temporal differential
                temperature map.

        Returns:
            (tuple):
            - *strain_map (ndarray[float])* - spatio-temporal strain profile.
            - *sticks_sub_systems (ndarray[float])* - spacer sticks per
              sub-system.
            - *velocities (ndarray[float])* - spatio-temporal velocity profile.

        """
        t1 = time()

        # initialize
        L = self.S.get_number_of_layers()
        thicknesses = self.S.get_layer_property_vector('_thickness')
        x0 = np.zeros([2*L])  # initial condition for the shift of the layers

        try:
            delays = delays.to('s').magnitude
        except AttributeError:
            pass

        # check temp_maps
        [temp_map, delta_temp_map] = self.check_temp_maps(temp_map, delta_temp_map, delays)

        # calculate the sticks due to heat expansion first for all delay steps
        self.disp_message('Calculating linear thermal expansion ...')
        sticks, sticks_sub_systems = self.calc_sticks_from_temp_map(temp_map, delta_temp_map)

        if self.only_heat:
            # no coherent dynamics so calculate the strain directly
            strain_map = sticks/np.tile(thicknesses, [np.size(sticks, 0), 1])
            velocities = np.zeros_like(strain_map)  # this is quasi-static
        else:
            # include coherent dynamics
            self.disp_message('Calculating coherent dynamics with ODE solver ...')

            L = self.S.get_number_of_layers()
            masses = self.S.get_layer_property_vector('_mass_unit_area')
            thicknesses = self.S.get_layer_property_vector('_thickness')
            spring_consts = self.S.get_layer_property_vector('spring_const')
            damping = self.S.get_layer_property_vector('_phonon_damping')
            force_from_heat = PhononNum.calc_force_from_heat(sticks, spring_consts)

            # apply scipy's ode-solver together
            if self.progress_bar:  # with tqdm progressbar
                pbar = tqdm()
                pbar.set_description('Delay = {:.3f} ps'.format(delays[0]*1e12))
                state = [delays[0], abs(delays[-1]-delays[0])/100]
            else:  # without progressbar
                pbar = None
                state = None

            sol = solve_ivp(
                PhononNum.ode_func,
                [delays[0], delays[-1]],
                x0,
                args=(delays, force_from_heat, damping, spring_consts, masses, L,
                      pbar, state),
                t_eval=delays,
                **self.ode_options)

            if pbar is not None:  # close tqdm progressbar if used
                pbar.close()

            # calculate the strain_map as the second spacial derivative
            # of the layer shift x(t). The result of the ode solver
            # contains x(t) = X(:,1:N) and v(t) = X(:,N+1:end) the
            # positions and velocities of the layers, respectively.
            temp = np.diff(sol.y[0:L, :].T, 1, 1)
            strain_map = temp/np.tile(thicknesses[:-1], [np.size(temp, 0), 1])
            velocities = sol.y[L:, :].T
        self.disp_message('Elapsed time for _strain_map_:'
                          ' {:f} s'.format(time()-t1))
        return strain_map, sticks_sub_systems, velocities

    @staticmethod
    def ode_func(t, X, delays, force_from_heat, damping, spring_consts, masses, L,
                 pbar=None, state=None):
        """ode_func

        Provides the according ode function for the ode solver which has to be
        solved. The ode function has the input :math:`t` and :math:`X(t)` and
        calculates the temporal derivative :math:`\dot{X}(t)` where the vector

        .. math::

            X(t) = [x(t) \; \dot{x}(t)] \quad \mbox{and } \quad
            \dot{X}(t) = [\dot{x}(t) \; \ddot{x}(t)] .

        :math:`x(t)` is the actual shift of each layer.

        Args:
            t (ndarray[float]): internal time steps of the ode solver.
            X (ndarray[float]): internal variable of the ode solver.
            delays (ndarray[float]): delays range of simulation [s].
            force_from_heat (ndarray[float]): force due to thermal expansion.
            damping (ndarray[float]): phonon damping.
            spring_consts (ndarray[float]): spring constants of masses.
            masses (ndarray[float]): masses of layers.
            L (int): number of layers.
            pbar (tqdm): tqdm progressbar.
            state (list[float]): state variables for progress bar.

        Returns:
            X_prime (ndarray[float]): velocities and accelerations of masses.

        """
        if pbar is not None:
            # set everything for the tqdm progressbar
            last_t, dt = state
            n = (t - last_t)/dt
            if n >= 1:
                pbar.update(1)
                pbar.set_description('Delay = {:.3f} ps'.format(t*1e12))
                state[0] = t
            elif n < 0:
                state[0] = t

        # start with the actual ode function
        x = X[0:L]
        v = X[L:]

        # the output must be a column vector
        X_prime = np.zeros([2*L])

        # accelerations = derivative of velocities
        X_prime[L:] = (
            PhononNum.calc_force_from_damping(v, damping, masses)
            + PhononNum.calc_force_from_spring(
                np.r_[np.diff(x), 0],
                np.r_[0, np.diff(x)],
                spring_consts)
            + force_from_heat[:, finderb(t, delays)].squeeze()
            )/masses

        # velocities = derivative of positions
        X_prime[0:L] = v

        return X_prime

    @staticmethod
    def calc_force_from_spring(d_X1, d_X2, spring_consts):
        """calc_force_from_spring

        Calculates the force :math:`F_i^{spring}` acting on each mass due to
        the displacement between the left and right site of that mass.

        .. math::

            F_i^{spring} = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1})

        We introduce-higher order inter-atomic potentials by

        .. math::

            k_i(x_i-x_{i-1}) = \sum_{j=1}^{M} k_i^j (x_i-x_{i-1})^j

        where :math:`M` is the order of the spring constants.

        Args:
            d_X1 (ndarray[float]): left displacements.
            d_X2 (ndarray[float]): right displacements.
            spring_consts (ndarray[float]): spring constants of masses.

        Returns:
            F (ndarray[float]): force from springs.

        """
        try:
            spring_order = np.size(spring_consts, 1)
        except IndexError:
            spring_order = 1

        spring_consts = np.reshape(spring_consts, [np.size(spring_consts, 0), spring_order])

        coeff1 = np.vstack((-spring_consts[0:-1, :], np.zeros([1, spring_order])))
        coeff2 = np.vstack((np.zeros([1, spring_order]), -spring_consts[0:-1, :]))

        temp1 = np.zeros([len(d_X1), spring_order])
        temp2 = np.zeros([len(d_X1), spring_order])

        for i in range(spring_order):
            temp1[:, i] = d_X1**(i+1)
            temp2[:, i] = d_X2**(i+1)

        F = np.sum(coeff2*temp2, 1) - np.sum(coeff1*temp1, 1)

        return F

    @staticmethod
    def calc_force_from_heat(sticks, spring_consts):
        """calc_force_from_heat

        Calculates the force acting on each mass due to the heat expansion,
        which is modeled by spacer sticks.

        Args:
            stciks (ndarray[float]): spacer sticks.
            spring_consts (ndarray[float]): spring constants of masses.

        Returns:
            F (ndarray[float]): force from thermal expansion.

        """
        M, L = np.shape(sticks)
        F = np.zeros([L, M])
        # traverse time
        for i in range(M):
            F[:, i] = -PhononNum.calc_force_from_spring(
                np.hstack((sticks[i, 0:L-1], 0)),
                np.hstack((0, sticks[i, 0:L-1])),
                spring_consts
                )

        return F

    @staticmethod
    def calc_force_from_damping(v, damping, masses):
        """calc_force_from_damping

        Calculates the force acting on each mass in a linear spring due to
        damping (:math:`\gamma_i`) according to the shift velocity difference
        :math:`v_{i}-v_{i-1}` with :math:`v_i(t) = \dot{x}_i(t)`:

        .. math::

            F_i^{damp} = \gamma_i(\dot{x}_i-\dot{x}_{i-1})

        Args:
            v (ndarray[float]): velocity of masses.
            damping (ndarray[float]): phonon damping.
            masses (ndarray[float]): masses.

        Returns:
            F (ndarray[float]): force from damping.

        """
        F = masses*damping*np.diff(v, 0)

        return F
