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

        for value in kwargs.values():
            param.append(value)

        return self.S.get_hash(types='phonon') + '_' + make_hash_md5(param)

    def get_all_strains_per_unique_layer(self, strain_map):
        """get_all_strains_per_unique_layer

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.

        Returns:
            strains (list[ndarray[float]]: all strains per unique layer.

        """
        # get the position indices of all unique layers in the sample structure
        positions = self.S.get_all_positions_per_unique_layer()
        strains = []

        for value in positions.values():
            strains.append(np.sort(np.unique(strain_map[:, value].flatten())))

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
            strains (list[ndarray[float]]: reduced strains per unique layer.

        """
        # initialize
        all_strains = self.get_all_strains_per_unique_layer(strain_map)
        L = len(all_strains)  # Nb. of unique layers
        strains = []

        if np.size(N) == 1:
            N = N*np.ones([L, 1])
        elif np.size(N) != L:
            raise ValueError('The dimension of N must be either 1 or the number '
                             'of unique layers the structure!')

        for i, value in enumerate(all_strains):
            min_strain = np.min(value)
            max_strain = np.max(value)
            strains.append(np.sort(np.unique(
                np.r_[0, np.linspace(min_strain, max_strain, int(N[i]))])))

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

        .. [7] A. Bojahr, M. Herzog, D. Schick, I. Vrejoiu, & M. Bargheer,
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
            strain_map, _, _ = \
                self.calc_strain_map(delays, temp_map, delta_temp_map)
            self.save(full_filename, {'strain_map': strain_map}, '_strain_map_num_')
        return strain_map

    def calc_strain_map(self, delays, temp_map, delta_temp_map):
        r"""calc_strain_map

        Calculates the ``strain_map`` of the sample structure for a given
        ``temp_map`` and ``delta_temp_map`` and ``delay`` array. Further
        details are given in Ref. [7]_. The coupled differential equations are
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
            strain_map = np.zeros([temp.shape[0], temp.shape[1]+1])
            strain_map[:, :-1] = temp
            strain_map = strain_map/np.tile(thicknesses[:], [np.size(temp, 0), 1])
            velocities = sol.y[L:, :].T
        self.disp_message('Elapsed time for _strain_map_:'
                          ' {:f} s'.format(time()-t1))
        return strain_map, sticks_sub_systems, velocities

    @staticmethod
    def ode_func(t, X, delays, force_from_heat, damping, spring_consts, masses, L,
                 pbar=None, state=None):
        r"""ode_func

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
        r"""calc_force_from_spring

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
        r"""calc_force_from_damping

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


class PhononAna(Phonon):
    """PhononAna

    Base class for analytical phonon simulations.

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

    References:

        .. [8] M. Herzog, D. Schick, P. Gaal, R. Shayduk, C. von Korff Schmising
           & M. Bargheer, *Analysis of ultrafast X-ray diffraction data in a
           linear-chain model of the lattice dynamics*, `Applied Physics A,
           106(3), 489-499 (2011).
           <http://www.doi.org/doi:10.1007/s00339-011-6719-z>`_

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self, output=[]):
        """String representation of this class"""

        class_str = 'Analytical Phonon simulation properties:\n\n'
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
            X (ndarray[float]): position shift vector.
            V (ndarray[float]): velocity vector.
            A (ndarray[float]): coefficient vector A of general solution.
            B (ndarray[float]): coefficient vector B of general solution.

        """
        filename = 'strain_map_ana_' \
                   + self.get_hash(delays, temp_map, delta_temp_map) \
                   + '.npz'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            strain_map = tmp['strain_map']
            X = tmp['X']
            V = tmp['V']
            A = tmp['A']
            B = tmp['B']
            self.disp_message('_strain_map_ loaded from file:\n\t' + filename)
        else:
            # file does not exist so calculate and save
            strain_map, X, V, A, B = \
                self.calc_strain_map(delays, temp_map, delta_temp_map)
            self.save(full_filename, {'strain_map': strain_map,
                                      'X': X,
                                      'V': V,
                                      'A': A,
                                      'B': B},
                      '_strain_map_ana_')
        return strain_map, X, V, A, B

    # %% calcStrainMap
    # % Calculates the _strainMap_ of the sample structure for a given 
    # % _tempMap_ and _deltaTempMap_ and _time_ vector. Further details 
    # % are given in Ref. [1]. Within the linear chain of $N$ masses 
    # % ($m_i$) at position $z_i$ coupled with spring constants $k_i$ one 
    # % can formulate the differential equation of motion as follow:
    # %
    # % $$ m_i\ddot{x}_i = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1}) +
    # % F_i^{heat}(t) $$
    # %
    # % Since we only consider nearest-neighbor interaction one can
    # % write:
    # %
    # % $$ \ddot{x}_i = \sum_{n=1}^N \kappa_{i,n} x_n = \Delta_i(t) $$
    # %
    # % Here $x_i(t) = z_i(t)-z_i^0$ is the shift of each unit cell,
    # % $F_i^{heat}(t)$ is the external force (thermal stress) of each 
    # % unit cell and $\kappa_{i,i} = -(k_i + k_{i+1})/m_i$, and 
    # % $\kappa_{i,i+1} = \kappa_{i+1,i} = k_{i+1}/m_i$.
    # %
    # % $k_i = m_i\, v_i^2/c_i^2$ is the spring constant and $c_i$ and
    # % $v_i$ are the lattice $c$-axis and longitudinal sound velocity of
    # % each unit cell respectively.
    # % One can rewrite the homogeneous differential equation in matrix 
    # % form to obtain the general solution
    # %
    # % $$ \frac{d^2}{dt^2} X = K\, X $$
    # %
    # % Here $X = (x_1 \ldots x_N)$ and $K$ is the
    # % tri-diagonal matrix of $\kappa$ which is real and symmetric.
    # % The differential equation can be solved with the ansatz:
    # %
    # % $$ X(t) = \sum_j \Xi_j \, (A_j \cos(\omega_j \, t) + B_j \sin(\omega_j \, t)) $$
    # %
    # % where $\Xi_j = (\xi_1^j \ldots \xi_N^j)$ are the eigenvectors of
    # % the matrix $K$. Thus by solving the Eigenproblem for $K$ one 
    # % gets the eigenvecotrs $\Xi_j$ and the eigenfrequencies $\omega_j$.
    # % From the initial conditions
    # %
    # % $$ X(0) = \sum_j \Xi_j \, A_j = \Xi \, A \qquad V(0) = \dot{X}(0) 
    # %         = \sum_j \Xi_j \, \omega_j\, B_j = \Xi \, \omega \, B $$
    # %
    # % one can determine the real coefficient vecots $A$ and $B$ in 
    # % order to calculate $X(t)$ and $V(t)$ using the ansatz:
    # %
    # % $$ A = \Xi \setminus X(0) \qquad B = (\Xi \setminus V(0)) ./ \omega $$
    # %
    # % The external force is implemented as spacer sticks which are
    # % inserted into the springs and hence the unit cells have a new 
    # % equillibrium  positions $z_i(\infty) = z_i^\infty$. Thus we can 
    # % do a coordination transformation:
    # %
    # % $$ z_i(t) = z_i^0 + x_i(t) = z_i^\infty + x_i^\infty(t) $$
    # %
    # % and
    # %
    # % $$ x_i^\infty(t) = z_i^0 - z_i^\infty + x_i(t) $$
    # %
    # % with the initial condition $x_i(0) = 0$ the becomes
    # %
    # % $$ x_i^\infty(0) = z_i^0 - z_i^\infty = \sum_{j = i+1}^N l_j $$
    # %
    # % $x_i^\infty(0)$ is the new initial condition after the excitation
    # % where $l_i$ is the length of the i-th spacer stick. The spacer 
    # % sticks are calculated from the temperature change and the linear 
    # % thermal expansion coefficients. 
    # % The actual strain $\epsilon_i(t)$ of each unit cell is calculates 
    # % as follows:
    # % 
    # % $$ \epsilon_i(t) = [ \Delta x_i(t) + l_i) ] / c_i $$
    # %
    # % with $\Delta x_i = x_i - x_{i-1}$. The stick $l_i$ have to be
    # % added here, because $x_i$ has been transformed into the new
    # % coordinate system $x_i^\infty$.
    # function [strainMap X V A B sticksSubSystems] = calcStrainMap(obj,time,tempMap,deltaTempMap)
    #     tic
    #     % initialize
    #     N = obj.S.getNumberOfUnitCells; % nb of unit cells
    #     M = length(time);               % nb of time steps
        
    #     time0       = time(1); % initial time
    #     cAxises     = obj.S.getUnitCellPropertyVector('cAxis');
    #     X           = zeros(M,N); % shifts of the unitCells
    #     V           = zeros(M,N); % velocities of the unitCells
    #     A           = zeros(M,N); % coefficient vector for eigenwert solution
    #     B           = zeros(M,N); % coefficient vector for eigenwert solution
    #     strainMap   = zeros(M,N); % the restulting strain pattern of the unitCells
        
    #     % check tempMaps
    #     [tempMap, deltaTempMap] = obj.checkTempMaps(tempMap,deltaTempMap,time);
        
    #     % calculate the sticks due to heat expansion first for all time
    #     % steps
    #     obj.dispMessage('Calculating linear thermal expansion ...');
    #     [sticks, sticksSubSystems] = obj.calcSticksFromTempMap(tempMap,deltaTempMap);
                    
    #     if obj.onlyheat
    #         % no coherent dynamics so calculate the strain directly
    #         strainMap = sticks./repmat(cAxises',size(sticks,1),1);
    #     else
    #         % solve the eigenproblem for the structure to obtains the
    #         % eigenvectors Xi and eigenfreqeuencies omega for the N 
    #         % coupled differential equations
    #         [Xi, omega] = obj.solveEigenproblem();
        
    #         % calculate the actual strain pattern with the solution of the
    #         % eigenproblem and the external force (sticks, thermal stress) 
    #         obj.dispMessage('Calculating _strainMap_ ...');
    #         obj.progressBar('Please wait... ');
    #         % traverse time
    #         for i=1:M
    #             obj.progressBar(i/M*100); % plot the progress

    #             dt = time(i)-time0; % this is the time step

    #             % calculate the current shift X and velocity V of all 
    #             % unitCells using the ansatz
    #             X(i,:)  = Xi*(         A(i,:)'.*cos(omega*dt) + B(i,:)'.*sin(omega*dt));
    #             V(i,:)  = Xi*(omega.*(-A(i,:)'.*sin(omega*dt) + B(i,:)'.*cos(omega*dt)));
    #             % remember the velocities and shifts as ic for the next
    #             % time step
    #             X0      = X(i,:)';
    #             V0      = V(i,:)';

    #             % the strain can only be calculated for N-1 unitCells, so
    #             % we neglect the last one
    #             if i > 1
    #                 strainMap(i,1:N-1) = (diff(X(i,:),1,2)+sticks(i-1,1:N-1))./cAxises(1:N-1)';
    #             else
    #                 % initial sticks are zero
    #                 strainMap(i,1:N-1) =  diff(X(i,:),1,2)./cAxises(1:N-1)';
    #             end%if

    #             % calculate everything for the next step
    #             if i < M % check, if there is a next step
    #                 if find(deltaTempMap(i,:)) % there is a temperature change
    #                     time0 = time(i); % set new initial time                       
                        
    #                     % determining the shifts due to inserted sticks
    #                     % as new ininital conditions
    #                     if i > 1
    #                         temp = flipud(cumsum(flipud(sticks(i,:)'-sticks(i-1,:)')));
    #                     else
    #                         % initial sticks are zero
    #                         temp = flipud(cumsum(flipud(sticks(i,:)')));
    #                     end%if
    #                     X0 = X0 + vertcat(temp(2:end),0);

    #                     % determining the cofficient vectors A and B of
    #                     % the general solution of X(t) using the inital
    #                     % conditions X0 and V0
    #                     A(i+1,:) = ( Xi\X0);
    #                     B(i+1,:) = ((Xi\V0)./omega)';
    #                 else
    #                     % no temperature change, so keep the current As,
    #                     % Bs, and sticks
    #                     A(i+1,:) = A(i,:);
    #                     B(i+1,:) = B(i,:);
    #                 end%if
    #             end%if
    #         end%for
    #         obj.progressBar('');
    #     end
    #     obj.dispMessage('Elapsed time for _strainMap_:',toc);
    # end%function

    # %% solveEigenproblem
    # % Creates the real and symmetric $K$ matrix ($N \times N$) of 
    # % spring constants $k_i$ and masses $m_i$ and calculates the 
    # % eigenvectors $\Xi_j$ and eigenfrequencies $\omega_j$ for the 
    # % matrix which are used to calculate the _strainMap_ of the
    # % structure.
    # % If the result has been save to file, load it from there.
    # function [Xi,omega] = solveEigenproblem(obj)
    #     % create the file name to look for
    #     filename = fullfile(obj.cacheDir, ['eigenValues_' obj.S.getHash('phonon') '.mat']);
    #     if exist(filename,'file') && ~obj.forceRecalc
    #         % file exists so load it 
    #         load(filename);
    #         obj.dispMessage(['_eigenValues_ loaded from file ' filename]);
    #     else
    #         % no file - so lets calculate everything
    #         tic
    #         obj.dispMessage('Calculating _eigenValues_ ...');
    #         % initialize
    #         N       = obj.S.getNumberOfUnitCells; % nb of unit cells
    #         K       = zeros(N,N); %Initializing three-diagonal springs-masses matrix.
    #         omega   = zeros(N,1); %Initializing a vector for eigenfrequencies
            
    #         masses       = obj.S.getUnitCellPropertyVector('mass'); % get masses vector
    #         springConsts = obj.S.getUnitCellPropertyVector('springConst'); % get the first order springs vector
    #         springConsts = vertcat(0, springConsts(:,1)); % set the first spring free
            
    #         for i=1:N %Defining main diagonal.
    #             K(i,i)=-(springConsts(i)+springConsts(i+1))/masses(i);
    #         end%for

    #         for i=2:N %Defining the two other diagonals. Nearest neightbour interaction.
    #             K(i,i-1) = springConsts(i)/masses(i);
    #             K(i-1,i) = springConsts(i)/masses(i-1);
    #         end%for
            
    #         % Determining the eigenvectors and the eigenvalues
    #         [Xi,lambda] = eig(K);

    #         for i=1:N % calculate the eigenfrequencies from the eigenvalues
    #             omega(i)=sqrt(-lambda(i,i));
    #         end%for
            
    #         obj.dispMessage('Elapsed time for _eigenValues_:',toc);
    #         % save the result to file
    #         save(filename,'Xi', 'omega');
    #         obj.dispMessage(['_eigenValues_ saved to file ' filename]);
    #     end%if
    # end%function
            
    # %% getEnergyPerEigenmode
    # % Returns the energy per Eigenmode of the coherent phonons of
    # % the 1D sample sorted and unsorted.
    # %
    # % $$ E_j = \frac{1}{2} (A^2_j + B^2_j)\, \omega_j^2\, m_j \, \| \Xi_j\|^2 $$
    # %
    # % Frequencies are in [Hz] and energy per mode in [J].
    # function [omegaSort ESort omega E] = getEnergyPerEigenmode(obj,A,B)
    #     % initialize
    #     N       = obj.S.getNumberOfUnitCells; % nb of unit cells
    #     M       = size(A,1); % nb of time steps
    #     E       = zeros(M,N);
    #     ESort   = zeros(M,N);
    #     masses  = obj.S.getUnitCellPropertyVector('mass'); % mass vector of unitCells
        
    #     % get the eigenVectors and eigenFrequencies
    #     [Xi,omega] = obj.solveEigenproblem(); 
        
    #     % sort the frequencies and remeber the permutation of indicies
    #     [omegaSort sortIndex] = sort(omega);            
        
    #     % traverse time
    #     for i=1:M
    #         % calculate the energy for the jth mode
    #         E(i,:) = 0.5 * (A(i,:)'.^2 + B(i,:)'.^2).* omega(:).^2.*masses(:) .* sum(Xi.^2,1)';
    #         % sort the energies according to the frequencies
    #         ESort(i,:) = E(i,sortIndex);
    #     end%for
