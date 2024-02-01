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

__all__ = ['Magnetization', 'LLB']

__docformat__ = 'restructuredtext'

from .simulation import Simulation
from .. import u, Q_
from ..helpers import make_hash_md5, finderb
from ..helpers import convert_cartesian_to_polar, convert_polar_to_cartesian
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import scipy.constants as constants
from time import time
from os import path
from tqdm.notebook import tqdm


class Magnetization(Simulation):
    """Magnetization

    Base class for all magnetization simulations.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.

    Attributes:
        S (Structure): sample structure to calculate simulations on.
        force_recalc (boolean): force recalculation of results.
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        ode_options (dict): options for scipy solve_ivp ode solver

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.ode_options = {
            'method': 'RK45',
            'first_step': None,
            'max_step': np.inf,
            'rtol': 1e-3,
            'atol': 1e-6,
            }

    def __str__(self, output=[]):
        """String representation of this class"""

        class_str = 'Magnetization simulation properties:\n\n'
        class_str += super().__str__(output)
        return class_str

    def get_hash(self, **kwargs):
        """get_hash

        Calculates an unique hash given by the delays as well as the sample
        structure hash for relevant magnetic parameters.
        Optionally, part of the ``strain_map`` and ``temp_map`` are used.

        Args:
            delays (ndarray[float]): delay grid for the simulation.
            **kwargs (ndarray[float], optional): optional delays, strain and
                temperature profile as well as external magnetic field and
                initial magnetization.

        Returns:
            hash (str): unique hash.

        """
        param = []

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

        for value in kwargs.values():
            param.append(value)

        return self.S.get_hash(types='magnetic') + '_' + make_hash_md5(param)

    def check_initial_magnetization(self, init_mag, distances=[]):
        """check_initial_magnetization

        Check if a given initial magnetization profile is valid. The profile
        must be either a vector `[amplitude, phi, gamma]` describing a global
        initial magnetization or an array of shape `[Nx3]` with N being the
        number of layers or the length of the specific spatial grid. If no
        initial magnetization is given, the initial profile is determined from
        the magnetization of the layers on creation. In addition, a spatial
        grid can be provided.

        Args:
            init_mag (ndarray[float]): initial global or local magnetization
                profile.
            distances (ndarray[float, Quantity], optional): spatial grid of the
                initial magnetization.

        Returns:
            init_mag (ndarray[float]): checked initial magnetization as array on
                the according spatial grid.

        """
        try:
            distances = distances.to('m').magnitude
        except AttributeError:
            pass

        N = len(distances)
        if N == 0:
            # no spatial grid is provided
            N = self.S.get_number_of_layers()
            [distances, _, _] = self.S.get_distances_of_layers(False)

        if len(init_mag) == 0:
            self.disp_message('No explicit initial magnetization given '
                              '- use magnetization of layers instead.')
            init_mag = np.zeros([N, 3])
            # use finderb search to find the corresponding indices between the
            # internal and external spatial grids
            [d_start, _, _] = self.S.get_distances_of_layers(False)
            idx = finderb(distances, d_start)

            magnetizations = self.S.get_layer_property_vector('_magnetization')
            init_mag[:, 0] = np.array([mag['amplitude'] for mag in magnetizations])[idx]
            init_mag[:, 1] = np.array([mag['phi'] for mag in magnetizations])[idx]
            init_mag[:, 2] = np.array([mag['gamma'] for mag in magnetizations])[idx]
        else:
            if np.size(init_mag) == 3:
                # it is the same initial magnetization for all layers
                init_mag = np.tile(init_mag, (N, 1))
            elif np.shape(init_mag) != (N, 3):
                # init_temp is a vector but has not as many elements as layers
                raise ValueError('The initial magnetization array must have 3 or '
                                 'Nx3 elements, where N is the number of layers '
                                 'in the structure or the length of the spatial '
                                 'grid provided as distances vector!')

            # convert phi and gamma to rad and store only magnitudes
            try:
                init_mag[:, 1] = init_mag[:, 1].to('rad').magnitude
            except AttributeError:
                pass
            try:
                init_mag[:, 2] = init_mag[:, 2].to('rad').magnitude
            except AttributeError:
                pass
        return init_mag

    def get_magnetization_map(self, delays, **kwargs):
        r"""get_magnetization_map

        Returns an absolute ``magnetization_map`` for the sample structure with
        the dimension :math:`M \times N \times 3` with :math:`M` being the
        number of delays and :math:`N` the number of layers in the structure or
        the length of the given spatial grid. Each element of the map contains
        the three magnetization components ``[amplitude, phi, gamma]``.
        The angles ``phi`` and ``gamma`` must be returned in radians as pure
        numpy arrays.
        The ``magnetization_map`` can depend on the ``temp_map`` and
        ``strain_map`` that can be also calculated for the sample structure.
        More over an external magnetic field ``H_ext`` and initial magnetization
        profile ``init_mag`` can be provided.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            **kwargs (ndarray[float], optional): optional strain and
                temperature profile as well external magnetic field in [T] and
                initial magnetization.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
               magnetization profile.

        """
        # create a hash of all simulation parameters
        filename = 'magnetization_map_' \
                   + self.get_hash(delays=delays, **kwargs) \
                   + '.npz'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            magnetization_map = tmp['magnetization_map']
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
            if ('H_ext' in kwargs):
                if not isinstance(kwargs['H_ext'], np.ndarray):
                    raise TypeError('H_ext must be a numpy ndarray!')
                elif kwargs['H_ext'].shape != (3,):
                    raise ValueError('H_ext must be a vector with 3 components '
                                     '(H_x, H_y, H_z)!')
            if ('init_mag' in kwargs):
                if not isinstance(kwargs['init_mag'], np.ndarray):
                    raise TypeError('init_mag must be a numpy ndarray with '
                                    'all in radians without units!')
                elif kwargs['init_mag'].shape != (3,):
                    raise ValueError('init_mag must be a vector with Nx3 '
                                     'with N being the number of layers.')

            magnetization_map = self.calc_magnetization_map(delays, **kwargs)

            self.disp_message('Elapsed time for _magnetization_map_:'
                              ' {:f} s'.format(time()-t1))
            self.save(full_filename, {'magnetization_map': magnetization_map},
                      '_magnetization_map_')
        return magnetization_map

    def calc_magnetization_map(self, delays, **kwargs):
        """calc_magnetization_map

        Calculates an absolute ``magnetization_map`` - see
        :meth:`get_magnetization_map` for details.

        This method is just an interface and should be overwritten for the
        actual simulations.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            **kwargs (ndarray[float], optional): optional strain and
                temperature profile as well external magnetic field and initial
                magnetization.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
                magnetization profile.

        """
        raise NotImplementedError


class LLB(Magnetization):
    """LLB

    Mean-field quantum Landau-Lifshitz-Bloch simulations.

    Please find a detailed review on the Landau-Lifshitz-Bloch equation by
    Unai Atxitia et al. [11]_.

    In collaboration with Theodor Griepe
    (`@Nilodirf <https://github.com/Nilodirf>`_) from the group of
    Unai Atxitia at Instituto de Ciencia de Materiales de Madrid (ICMM-CSIC).

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.

    Attributes:
        S (Structure): sample structure to calculate simulations on.
        force_recalc (boolean): force recalculation of results.
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.

    References:

        .. [11] U. Atxitia, D. Hinzke, and U. Nowak,
           *Fundamentals and Applications of the Landau-Lifshitz-Bloch Equation*,
           `J. Phys. D. Appl. Phys. 50, (2017).
           <https://www.doi.org/10.1088/1361-6463/50/3/033003>`_

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self):
        """String representation of this class"""
        class_str = 'Landau-Lifshitz-Bloch Magnetization Dynamics simulation ' \
                    'properties:\n\n'
        class_str += super().__str__()
        return class_str

    def calc_magnetization_map(self, delays, temp_map, H_ext=np.array([0, 0, 0]), init_mag=[]):
        r"""calc_magnetization_map

        Calculates the magnetization map using the mean-field quantum
        Landau-Lifshitz-Bloch equation (LLB) for a given delay range and
        according temperature map:

        .. math::

            \frac{d\mathbf{m}}{dt}=\gamma_e \left(\mathbf{m} \times
              \mathbf{H}_\mathrm{eff} + \frac{\alpha_{\perp}}{m^2}\mathbf{m}
              \times (\mathbf{m} \times \mathbf{H}_\mathrm{eff}) -
              \frac{\alpha_{\parallel}}{m^2}(\mathbf{m} \cdot
              \mathbf{H}_\mathrm{eff}) \cdot \mathbf{m}\right)

        The three terms describe

        #. **precession** at Larmor frequency,
        #. **transversal damping** (conserving the macrospin length), and
        #. **longitudinal damping** (changing macrospin length due to incoherent
           atomistic spin excitations within the layer the macrospin is
           defined on).

        :math:`\alpha_{\parallel}` and :math:`\alpha_{\perp}` are the
        :meth:`longitudinal damping<calc_longitudinal_damping>` and
        :meth:`transverse damping<calc_transverse_damping>` parameters,
        respectively.
        :math:`\gamma_e = -1.761\times10^{11}\,\mathrm{rad\,s^{-1}\,T^{-1}}` is
        the gyromagnetic ratio of an electron.

        The effective magnetic field is the sum of all relevant magnetic
        interactions:

        .. math::

            \mathbf{H}_\mathrm{eff} = \mathbf{H}_\mathrm{ext}
              + \mathbf{H}_\mathrm{A}
              + \mathbf{H}_\mathrm{ex}
              + \mathbf{H}_\mathrm{th}

        where

        * :math:`\mathbf{H}_\mathrm{ext}` is the external magnetic field
        * :math:`\mathbf{H}_\mathrm{A}` is the :meth:`uniaxial anisotropy field
          <calc_uniaxial_anisotropy_field>`
        * :math:`\mathbf{H}_\mathrm{ex}` is the :meth:`exchange field
          <calc_exchange_field>`
        * :math:`\mathbf{H}_\mathrm{th}` is the :meth:`thermal field
          <calc_thermal_field>`

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            temp_map (ndarray[float]): spatio-temporal temperature map.
            H_ext (ndarray[float], optional): external magnetic field
                (H_x, H_y, H_z) [T].

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
            magnetization profile.

        """
        t1 = time()
        try:
            delays = delays.to('s').magnitude
        except AttributeError:
            pass
        M = len(delays)

        distances, _, _ = self.S.get_distances_of_layers(False)

        init_mag = self.check_initial_magnetization(init_mag, distances)
        # convert initial magnetization from polar to cartesian coordinates
        init_mag = convert_polar_to_cartesian(init_mag)
        # get layer properties
        curie_temps = self.S.get_layer_property_vector('_curie_temp')
        eff_spins = self.S.get_layer_property_vector('eff_spin')
        lambdas = self.S.get_layer_property_vector('lamda')
        mf_exch_couplings = self.S.get_layer_property_vector('mf_exch_coupling')
        mag_moments = self.S.get_layer_property_vector('_mag_moment')
        aniso_exponents = self.S.get_layer_property_vector('aniso_exponent')
        anisotropies = self.S.get_layer_property_vector('_anisotropy')
        mag_saturations = self.S.get_layer_property_vector('_mag_saturation')
        exch_stiffnesses = self.get_directional_exchange_stiffnesses()
        thicknesses = self.S.get_layer_property_vector('_thickness')
        # calculate the mean magnetization maps for each unique layer
        # and all relevant parameters
        mean_mag_map = self.get_mean_field_mag_map(temp_map[:, :, 0])
        # mask for magnetic layers only
        is_magnetic = curie_temps > 0
        N = np.count_nonzero(is_magnetic)

        if self.progress_bar:  # with tqdm progressbar
            pbar = tqdm()
            pbar.set_description('Delay = {:.3f} ps'.format(delays[0]*1e12))
            state = [delays[0], abs(delays[-1]-delays[0])/100]
        else:  # without progressbar
            pbar = None
            state = None
        # solve pdepe with method-of-lines
        sol = solve_ivp(
            LLB.odefunc,
            [delays[0], delays[-1]],
            np.reshape(init_mag[is_magnetic, :], N*3, order='F'),
            args=(delays,
                  N,
                  H_ext,
                  temp_map[:, is_magnetic, 0],  # provide only the electron temperature
                  mean_mag_map[:, is_magnetic],
                  curie_temps[is_magnetic],
                  eff_spins[is_magnetic],
                  lambdas[is_magnetic],
                  mf_exch_couplings[is_magnetic],
                  mag_moments[is_magnetic],
                  aniso_exponents[is_magnetic],
                  anisotropies[is_magnetic],
                  mag_saturations[is_magnetic],
                  exch_stiffnesses[is_magnetic],
                  thicknesses[is_magnetic],
                  pbar, state),
            t_eval=delays,
            **self.ode_options)

        if pbar is not None:  # close tqdm progressbar if used
            pbar.close()
        temp = sol.y.T
        # final magnetization map is zero for all non-magnetic layers
        magnetization_map = np.zeros([M, len(distances), 3])
        # reshape results and set only for magnetic layers
        magnetization_map[:, is_magnetic, :] = np.array(temp).reshape([M, N, 3], order='F')
        # convert to polar coordinates
        magnetization_map = convert_cartesian_to_polar(magnetization_map)
        self.disp_message('Elapsed time for _LLB_: {:f} s'.format(time()-t1))

        return magnetization_map

    def get_mean_field_mag_map(self, temp_map):
        r"""get_mean_field_mag_map

        Returns the mean-field magnetization map see
        :meth:`calc_mean_field_mag_map` for details. The dimension of the map
        are :math:`M \times N` with :math:`M` being the number of delays and
        :math:`N` the number of layers in the structure.

        Args:
            temp_map (ndarray[float]): spatio-temporal electron temperature map.

        Returns:
            mf_mag_map (ndarray[float]): spatio-temporal mean_field
            magnetization map.

        """
        # create a hash of all simulation parameters
        filename = 'mf_magnetization_map_' \
                   + self.get_hash(temp_map=temp_map) \
                   + '.npz'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            mf_mag_map = tmp['mf_mag_map']
            self.disp_message('_mean_field_magnetization_map_ loaded from file:\n\t' + filename)
        else:
            t1 = time()
            self.disp_message('Calculating _mean_field_magnetization_map_ ...')
            # parse the input arguments
            if not isinstance(temp_map, np.ndarray):
                raise TypeError('temp_map must be a numpy ndarray!')

            mf_mag_map = self.calc_mean_field_mag_map(temp_map)

            self.disp_message('Elapsed time for _mean_field_magnetization_map_:'
                              ' {:f} s'.format(time()-t1))
            self.save(full_filename, {'mf_mag_map': mf_mag_map},
                      '_mean_field_magnetization_map_')
        return mf_mag_map

    def calc_mean_field_mag_map(self, temp_map):
        r"""calc_mean_field_mag_map

        Calculate the mean-field magnetization map :math:`m_\mathrm{eq}` by
        solving the self consistent equation

        .. math::

            m_\mathrm{eq}(T) & = B_S(m_\mathrm{eq}, T) \\

        where :math:`B_S` is the Brillouin function.

        Args:
            temp_map (ndarray[float]): spatio-temporal electron temperature map.

        Returns:
            mf_mag_map (ndarray[float]): spatio-temporal mean_field
            magnetization map.

        """
        (M, N) = temp_map.shape  # M - number of delays, N - number of layers

        mf_mag_map = np.zeros_like(temp_map)

        unique_layers = self.S.get_unique_layers()
        relevant_temps = {}
        # iterate over all positions per unique layers
        for i, (k, v) in enumerate(self.S.get_all_positions_per_unique_layer().items()):
            relevant_temps[k] = []
            # unique layer properties
            curie_temp = unique_layers[1][i]._curie_temp
            # mean-field magnetization is only calculated for a non-zero Curie
            # temperature of magnetic layers
            if curie_temp > 0:
                eff_spin = unique_layers[1][i].eff_spin
                mf_exch_coupling = unique_layers[1][i].mf_exch_coupling

                # simple round for down-sampling
                unique_temps = np.unique(np.round(temp_map[:, v].flatten(), decimals=1))
                # only temperatures below T_C are relevant
                unique_temps = unique_temps[unique_temps <= curie_temp]
                #  are normalized by T_C
                reduced_temps = unique_temps/curie_temp
                mf_mags = np.zeros_like(reduced_temps)

                for j, T in enumerate(reduced_temps):
                    if T == 1:
                        mf_mags[j] = 0
                    else:
                        mf_mags[j] = fsolve(
                            lambda x: x - LLB.calc_Brillouin(x, T, eff_spin, mf_exch_coupling,
                                                             curie_temp), np.sqrt(1-T))

                relevant_temps[k] = np.stack((unique_temps, mf_mags))

                # for every temperature in temp_map search for best match in
                # relevant_temps and assign according mf_mag into mf_mag_map
                idx = finderb(np.round(temp_map[:, v].flatten(), decimals=1),
                              relevant_temps[k][0, :])
                mf_mag_map[:, v] = np.reshape(relevant_temps[k][1, idx], (M, len(v)))
            else:
                # non-magnetic layers with Curie temperature = 0
                mf_mag_map[:, v] = 0

        return mf_mag_map

    def get_directional_exchange_stiffnesses(self):
        r"""get_directional_exchange_stiffnesses

        Returns the directional exchange stiffnesses with size
        :math:`N \times 2`, with :math:`N` being the number of layers, between
        the :math:`(i-1)^\mathrm{th}` and :math:`i^\mathrm{th}` layers as well
        as between the :math:`i^\mathrm{th}` and :math:`(i+1)^\mathrm{th}`
        layers in the structure.
        In case two neighboring layers are identical the center entry
        of the 3-component `exchange_stiffness` is used. In case of an interface
        to the top :math:`(i-1)\rightarrow i` it will be the first entry and
        for an interface to the bottom :math:`i\rightarrow (i+1)` it will be
        the third entry.

        Returns:
            A (ndarray[float]): directional exchange stiffnesses.

        """
        exch_stiffnesses = self.S.get_layer_property_vector('_exch_stiffness')

        indices, _, _ = self.S.get_layer_vectors()

        A = np.zeros([len(indices), 2])
        interfaces = (np.r_[1, np.diff(indices), 1])
        interfaces[interfaces != 0] = -1
        select = (interfaces+1).astype(np.int16)

        A[:, 0] = exch_stiffnesses[np.arange(len(select[0:-1])), select[0:-1]]

        interfaces[interfaces != 0] = 1
        select = (interfaces+1).astype(np.int16)
        A[:, 1] = exch_stiffnesses[np.arange(len(select[1:])), select[1:]]

        return A

    @staticmethod
    def odefunc(t, m,
                delays, N, H_ext, temp_map, mean_mag_map, curie_temps, eff_spins, lambdas,
                mf_exch_couplings, mag_moments, aniso_exponents, anisotropies, mag_saturations,
                exch_stiffnesses, thicknesses, pbar, state):
        """odefunc

        Ordinary differential equation that is solved for 1D LLB.

        Args:
            t (ndarray[float]): internal time steps of the ode solver.
            m (ndarray[float]): internal variable of the ode solver.
            delays (ndarray[float]): delays range of simulation [s].
            N (int): number of spatial grid points.
            H_ext (ndarray[float]): external magnetic field
                (H_x, H_y, H_z) [T].
            temp_map (ndarray[float]): spatio-temporal electron temperature map.
            mean_mag_map (ndarray[float]): spatio-temporal
                mean-field magnetization map.
            curie_temps (ndarray[float]): Curie temperatures of layers.
            eff_spins (ndarray[float]): effective spins of layers.
            lambdas (ndarray[float]): coupling-to-bath parameter of layers.
            mf_exch_couplings (ndarray[float]): mean-field exchange couplings of
                 layers.
            mag_moments (ndarray[float]): atomic magnetic moments of layers.
            aniso_exponents (ndarray[float]): exponent of uniaxial anisotropy of
                layers.
            anisotropies (ndarray[float]): anisotropy vectors of layers.
            mag_saturations (ndarray[float]): saturation magnetization of
                layers.
            exch_stiffnesses (ndarray[float]): exchange stiffness of layers
                towards the upper and lower layer.
            thicknesses (ndarray[float]): thicknesses of layers.
            pbar (tqdm): tqdm progressbar.
            state (list[float]): state variables for progress bar.

        Returns:
            dmdt (ndarray[float]): temporal derivative of internal variable.

        """
        # state is a list containing last updated time t:
        # state = [last_t, dt]
        # I used a list because its values can be carried between function
        # calls throughout the ODE integration
        last_t, dt = state
        try:
            n = int((t - last_t)/dt)
        except ValueError:
            n = 0

        if n >= 1:
            pbar.update(n)
            pbar.set_description('Delay = {:.3f} ps'.format(t*1e12))
            state[0] = t
        elif n < 0:
            state[0] = t

        # initialize arrays
        # reshape input temperature
        m = np.array(m).reshape([N, 3], order='F')

        # nearest delay index for current time t
        idt = finderb(t, delays)[0]
        temps = temp_map[idt, :].flatten()
        # binary masks for layers being under or over its Curie temperature
        under_tc = (temps < curie_temps)
        over_tc = ~under_tc
        # get the current mean-field magnetization
        mf_magnetizations = mean_mag_map[idt, :]

        # actual calculations
        m_squared = np.sum(np.power(m, 2), axis=1)
        gamma_e = -1.761e11

        # external field H_ext is given as input
        # calculate uniaxial anisotropy field
        H_A = LLB.calc_uniaxial_anisotropy_field(m, mf_magnetizations, aniso_exponents,
                                                 anisotropies, mag_saturations)
        # calculate exchange field
        H_ex = LLB.calc_exchange_field(m, exch_stiffnesses, mag_saturations, thicknesses)
        # calculate thermal field
        H_th = LLB.calc_thermal_field(m, m_squared, temps, mf_magnetizations, eff_spins,
                                      curie_temps, mf_exch_couplings, mag_moments, under_tc,
                                      over_tc)

        # calculate the effective field
        H_eff = H_ext + H_A + H_ex + H_th

        # calculate components of LLB
        # precessional term:
        m_rot = np.cross(m, H_eff)

        # damping
        qs = LLB.calc_qs(temps, curie_temps, eff_spins, mf_magnetizations,
                         under_tc)
        # transversal damping
        alpha_trans = LLB.calc_transverse_damping(temps, curie_temps, lambdas,
                                                  qs, mf_magnetizations,
                                                  under_tc, over_tc)
        trans_damping = np.multiply(
            np.divide(alpha_trans, m_squared)[:, np.newaxis],
            np.cross(m, m_rot)
            )
        # longitudinal damping
        alpha_long = LLB.calc_longitudinal_damping(temps, curie_temps,
                                                   eff_spins, lambdas, qs,
                                                   under_tc, over_tc)
        long_damping = np.multiply(
            np.divide(alpha_long, m_squared)[:, np.newaxis],
            np.multiply(np.einsum('ij,ij->i', m, H_eff)[:, np.newaxis], m)
            )

        dmdt = gamma_e * (m_rot + trans_damping - long_damping)

        return np.reshape(dmdt, N*3, order='F')

    @staticmethod
    def calc_uniaxial_anisotropy_field(mag_map, mf_magnetizations, aniso_exponents, anisotropies,
                                       mag_saturations):
        r"""calc_uniaxial_anisotropy_field

        Calculate the uniaxial anisotropy component of the effective field.

        .. math::

            \mathbf{H}_\mathrm{A} = -
            \frac{2}{M_s}
            \left(
                K_x\,m_\mathrm{eq}(T)^{\kappa-2}
                    \begin{bmatrix}0\\m_y\\m_z\end{bmatrix}
                + K_y\,m_\mathrm{eq}(T)^{\kappa-2}
                    \begin{bmatrix}m_x\\0\\m_z\end{bmatrix}
                + K_z\,m_\mathrm{eq}(T)^{\kappa-2}
                    \begin{bmatrix}m_x\\m_y\\0\end{bmatrix}
            \right)

        with :math:`K = (K_x, K_y, K_z)` as the anisotropy and :math:`\kappa` as
        the uniaxial anisotropy exponent.

        Args:
            mag_map (ndarray[float]): spatio-temporal magnetization map
                - possibly for a single delay.
            mf_magnetizations (ndarray[float]): mean-field magnetization of
                layers.
            aniso_exponents (ndarray[float]): exponent of uniaxial
                anisotropy of layers.
            anisotropies (ndarray[float]): anisotropy vectors of layers.
            mag_saturations (ndarray[float]): saturation magnetization of
                layers.

        Returns:
            H_A (ndarray[float]): uniaxial anisotropy field.

        """
        H_A = np.zeros_like(mag_map)

        factor = -2/mag_saturations
        unit_vector = np.array([0, 1, 1])[np.newaxis, :]
        for i in range(3):
            H_A += factor[:, np.newaxis] * anisotropies[:, i, np.newaxis]\
                * np.power(mf_magnetizations,
                           aniso_exponents-2)[:, np.newaxis] \
                * mag_map*np.roll(unit_vector, i, axis=1)

        return H_A

    @staticmethod
    def calc_exchange_field(mag_map, exch_stiffnesses, mag_saturations, thicknesses):
        r"""calc_exchange_field

        Calculate the exchange component of the effective field, which is
        defined as for each :math:`i^\mathrm{th}` layer in the structure as

        .. math::

            H_{\mathrm{ex}, i}=\frac{2}{M_{s,i} \Delta z_i^2}
                \left(A_{i}^{i-1}\left(\mathbf{m}_{i-1}-\mathbf{m}_{i}\right)
                + A_i^{i+1}\left(\mathbf{m}_{i+1}-\mathbf{m}_{i}\right) \right),

        where :math:`\Delta z` is the thickness of the layers or magnetic grains
        and :math:`M_s` is the saturation magnetization. :math:`A_{i}^{i-1}` and
        :math:`A_i^{i+1}` describe the exchange stiffness between the nearest
        neighboring layers provided by
        :meth:`get_directional_exchange_stiffnesses`.

        Args:
            mag_map (ndarray[float]): spatio-temporal magnetization map
                - possibly for a single delay.
            exch_stiffnesses (ndarray[float]): exchange stiffness of layers
                towards the upper and lower layer.
            mag_saturations (ndarray[float]): saturation magnetization of
                layers.

        Returns:
            H_ex (ndarray[float]): exchange field.

        """
        H_ex = np.zeros_like(mag_map)

        m_diff_down = np.concatenate((np.diff(mag_map, axis=0), np.zeros((1, 3))), axis=0)
        m_diff_up = -np.roll(m_diff_down, 1)

        es = np.divide(2, np.multiply(mag_saturations, thicknesses**2))

        H_ex = es[:, np.newaxis]*exch_stiffnesses[:, 0, np.newaxis]*m_diff_up \
            + es[:, np.newaxis]*exch_stiffnesses[:, 1, np.newaxis]*m_diff_down

        return -H_ex

    @staticmethod
    def calc_thermal_field(mag_map, mag_map_squared, temp_map, mf_magnetizations, eff_spins,
                           curie_temps, mf_exch_couplings, mag_moments, under_tc, over_tc):
        r"""calc_thermal_field

        Calculate the thermal component of the effective field.

        .. math::

            \mathbf{H}_\mathrm{th} = \begin{cases}
                \frac{1}{2\chi_{\parallel}}\left(1-\frac{m^2}{m_\mathrm{eq}^2}
                    \right)\mathbf{m} & \mathrm{for}\ T < T_\mathrm{C} \\
                -\frac{1}{\chi_{\parallel}}\left(1+\frac{3}{5}
                    \frac{T_\mathrm{C}}{T-T_\mathrm{C}}m^2\right)\mathbf{m}
                    & \mathrm{for}\ T \geq T_\mathrm{C}
            \end{cases}

        with :math:`\chi_{\parallel}` being the
        :meth:`longitudinal susceptibility<calc_long_susceptibility>`.

        Args:
            mag_map (ndarray[float]): spatio-temporal magnetization map
                - possibly for a single delay.
            mag_map_squared (ndarray[float]): spatio-temporal magnetization map
                squared- possibly for a single delay.
            temp_map (ndarray[float]): spatio-temporal temperature map
                - possibly for a single delay.
            mf_magnetizations (ndarray[float]): mean-field magnetization of
                layers.
            eff_spins (ndarray[float]): effective spin of layers.
            curie_temps (ndarray[float]): Curie temperature of layers.
            mf_exch_couplings (ndarray[float]): mean-field exch. coupling of
                layers.
            mag_moments (ndarray[float]): atomic magnetic moments of layers.
            under_tc (ndarray[boolean]): mask temperatures under the Curie
                temperature.
            over_tc (ndarray[boolean]): mask temperatures over the Curie
                temperature.

        Returns:
            H_th (ndarray[float]): thermal field.

        """
        chi_long = LLB.calc_long_susceptibility(temp_map, mf_magnetizations, curie_temps,
                                                eff_spins, mf_exch_couplings, mag_moments,
                                                under_tc, over_tc)

        H_th = np.zeros_like(temp_map)
        H_th[under_tc] = 1/(2 * chi_long[under_tc]) * (
            1 - mag_map_squared[under_tc]/mf_magnetizations[under_tc]**2
            )
        H_th[over_tc] = -1/chi_long[over_tc] * (
            1 + 3/5 * curie_temps[over_tc]/(temp_map[over_tc]-curie_temps[over_tc])
            ) * mag_map_squared[over_tc]

        return np.multiply(H_th[:, np.newaxis], mag_map)

    @staticmethod
    def calc_Brillouin(mag, temp, eff_spin, mf_exch_coupling, curie_temp):
        r"""calc_Brillouin

        .. math::

            B_S(m, T) = \frac{2 S+1}{2S} \coth{\left(\frac{2S+1}{2S}
            \frac{J \, m}{k_\mathrm{B}\, T}\right)}
            - \frac{1}{2S}\coth{\left(\frac{1}{2S}
            \frac{J \, m}{k_\mathrm{B}\,T}\right)}

        where

        .. math::

            J = 3\frac{S}{S+1}k_\mathrm{B} \, T_\mathrm{C}

        is the mean field exchange coupling constant for effective spin
        :math:`S` and Curie temperature :math:`T_\mathrm{C}`.

        Args:
            mag (ndarray[float]): magnetization of layer.
            temp (ndarray[float]): electron temperature of layer.
            eff_spin (ndarray[float]): effective spin of layer.
            mf_exch_coupling (ndarray[float]): mean-field exch. coupling of
                layers.
            curie_temp (ndarray[float]): Curie temperature of layer.

        Returns:
            brillouin (ndarray[float]): brillouin function.

        """

        eta = mf_exch_coupling * mag / constants.k / temp / curie_temp
        c1 = (2 * eff_spin + 1) / (2 * eff_spin)
        c2 = 1 / (2 * eff_spin)
        brillouin = c1 / np.tanh(c1 * eta) - c2 / np.tanh(c2 * eta)
        return brillouin

    @staticmethod
    def calc_dBrillouin_dx(temp_map, mf_magnetizations, eff_spins, mf_exch_couplings):
        r"""calc_dBrillouin_dx

        Calculate the derivative of the Brillouin function :math:`B_x` at
        :math:`m = m_\mathrm{eq}`:

        .. math::

            B_x = \frac{dB}{dx} = \frac{1}{4S^2\sinh^2(x/2S)}
                -\frac{(2S+1)^2}{4S^2\sinh^2\left(\frac{(2S+1)x}{2S}\right)}

        with :math:`x=\frac{J\,m}{k_\mathrm{B}\,T}`.

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature map
                - possibly for a single delay.
            mf_magnetizations (ndarray[float]): mean-field magnetization of
                layers.
            eff_spins (ndarray[float]): effective spin of layers.
            mf_exch_couplings (ndarray[float]): mean-field exchange couplings of
                layers.

        Returns:
            dBdx (ndarray[float]): derivative of Brillouin function.

        """
        x = np.divide(mf_exch_couplings*mf_magnetizations,
                      constants.k*temp_map)

        two_eff_spins = 2*eff_spins
        dBdx = 1 / (two_eff_spins**2 * np.sinh(x / (two_eff_spins))**2) \
            - (two_eff_spins + 1)**2 / \
            (two_eff_spins**2 * np.sinh(((two_eff_spins + 1) * x) / (two_eff_spins))**2)

        return dBdx

    @staticmethod
    def calc_transverse_damping(temp_map, curie_temps, lambdas, qs,
                                mf_magnetizations, under_tc, over_tc):
        r"""calc_transverse_damping

        Calculate the transverse damping parameter:

        .. math::

            \alpha_{\perp} = \begin{cases}
                \frac{\lambda}{m_\mathrm{eq}(T)}\left(\frac{\tanh(q_s)}{q_s}-
                    \frac{T}{3T_\mathrm{C}}\right)
                    & \mathrm{for}\ T < T_\mathrm{C} \\
                \frac{2 \lambda}{3}\frac{T}{T_\mathrm{C}}
                    & \mathrm{for}\ T \geq T_\mathrm{C}
            \end{cases}

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature map
                - possibly for a single delay.
            curie_temps (ndarray[float]): Curie temperatures of layers.
            lambdas (ndarray[float]): coupling-to-bath parameter of layers.
            qs (ndarray[float]): qs parameter.
            mf_magnetizations (ndarray[float]): mean-field magnetization of
                layers.
            under_tc (ndarray[boolean]): mask temperatures under the Curie
                temperature.
            over_tc (ndarray[boolean]): mask temperatures over the Curie
                temperature.

        Returns:
            alpha_trans (ndarray[float]): transverse damping parameter.

        """
        alpha_trans = np.zeros_like(temp_map)
        alpha_trans[under_tc] = np.multiply(
            np.divide(lambdas[under_tc], mf_magnetizations[under_tc]), (
                np.divide(np.tanh(qs), qs)
                - np.divide(temp_map[under_tc], 3*curie_temps[under_tc])
                )
            )
        alpha_trans[over_tc] = lambdas[over_tc]*2/3*np.divide(
            temp_map[over_tc], curie_temps[over_tc]
            )
        return alpha_trans

    @staticmethod
    def calc_longitudinal_damping(temp_map, curie_temps, eff_spins, lambdas, qs,
                                  under_tc, over_tc):
        r"""calc_transverse_damping

        Calculate the transverse damping parameter:

        .. math::

            \alpha_{\parallel} = \begin{cases}
                \frac{2\lambda}{S+1}
                \frac{1}{\sinh(2q_s)} & \mathrm{for}\ T < T_\mathrm{C} \\
                \frac{2 \lambda}{3}\frac{T}{T_\mathrm{C}}
                    & \mathrm{for}\ T \geq T_\mathrm{C}
            \end{cases}

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature map
                - possibly for a single delay.
            curie_temps (ndarray[float]): Curie temperatures of layers.
            eff_spins (ndarray[float]): effective spins of layers.
            lambdas (ndarray[float]): coupling-to-bath parameter of layers.
            qs (ndarray[float]): qs parameter.
            under_tc (ndarray[boolean]): mask temperatures under the Curie
                temperature.
            over_tc (ndarray[boolean]): mask temperatures over the Curie
                temperature.

        Returns:
            alpha_long (ndarray[float]): transverse damping parameter.

        """
        alpha_long = np.zeros_like(temp_map)
        alpha_long[under_tc] = np.divide(2*np.divide(lambdas[under_tc],
                                                     (eff_spins[under_tc]+1)),
                                         np.sinh(2*qs)
                                         )
        alpha_long[over_tc] = lambdas[over_tc]*2/3*np.divide(
            temp_map[over_tc], curie_temps[over_tc]
            )

        return alpha_long

    @staticmethod
    def calc_qs(temp_map, mf_magnetizations, curie_temps, eff_spins, under_tc):
        r"""calc_qs

        Calculate the :math:`q_s` parameter:

        .. math::

            q_s=\frac{3 T_\mathrm{C} m_\mathrm{eq}(T)}{(2S+1)T}

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature map
                - possibly for a single delay.
            mf_magnetizations (ndarray[float]): mean-field magnetization of
                layers.
            curie_temps (ndarray[float]): Curie temperatures of layers.
            eff_spins (ndarray[float]): effective spins of layers.
            under_tc (ndarray[boolean]): mask temperatures below the Curie
                temperature.

        Returns:
            qs (ndarray[float]): qs parameter.

        """
        return np.divide(
            3*curie_temps[under_tc] * mf_magnetizations[under_tc],
            (2*eff_spins[under_tc] + 1)*temp_map[under_tc]
            )

    @staticmethod
    def calc_long_susceptibility(temp_map, mf_magnetizations, curie_temps, eff_spins,
                                 mf_exch_couplings, mag_moments, under_tc, over_tc):
        r"""calc_long_susceptibility

        Calculate the the longitudinal susceptibility

        .. math::

            \chi_{\parallel} = \begin{cases}
                \frac{\mu_{\rm{B}}\,B_x(m_{eq}, T)}{
                    T\,k_\mathrm{B}-J\,B_x(m_{eq}, T)}
                    & \mathrm{for}\ T < T_\mathrm{C} \\
                \frac{\mu_{\rm{B}}T_\mathrm{C}}{J(T-T_\mathrm{C})}
                    & \mathrm{for}\ T \geq T_\mathrm{C}
            \end{cases}

        with :math:`B_x(m_{eq},T)` being the :meth:`derivative of the Brillouin
        function<calc_dBrillouin_dx>`.

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature map
                - possibly for a single delay.
            mf_magnetizations (ndarray[float]): mean-field magnetization of
                layers.
            curie_temps (ndarray[float]): Curie temperatures of layers.
            eff_spins (ndarray[float]): effective spins of layers.
            mf_exch_couplings (ndarray[float]): mean-field exchange couplings of
                layers.
            mag_moments (ndarray[float]): atomic magnetic moments of layers.
            under_tc (ndarray[boolean]): mask temperatures below the Curie
                temperature.
            over_tc (ndarray[boolean]): mask temperatures over the Curie
                temperature.

        Returns:
            chi_long (ndarray[float]): longitudinal susceptibility.

        """

        dBdx = LLB.calc_dBrillouin_dx(temp_map[under_tc],
                                      mf_magnetizations[under_tc],
                                      eff_spins[under_tc],
                                      mf_exch_couplings[under_tc])

        chi_long = np.zeros_like(temp_map)
        chi_long[under_tc] = np.divide(
            mag_moments[under_tc]*dBdx,
            temp_map[under_tc]*constants.k - mf_exch_couplings[under_tc]*dBdx
            )
        chi_long[over_tc] = np.divide(
            mag_moments[over_tc]*curie_temps[over_tc],
            mf_exch_couplings[over_tc]*(temp_map[over_tc]-curie_temps[over_tc])
            )

        return chi_long

    @property
    def distances(self):
        return Q_(self._distances, u.meter).to('nm')

    @distances.setter
    def distances(self, distances):
        self._distances = distances.to_base_units().magnitude
