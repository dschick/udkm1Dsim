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

__all__ = ['Magnetization']

__docformat__ = 'restructuredtext'

from .simulation import Simulation
from .. import u, Q_
from ..helpers import make_hash_md5, finderb
import numpy as np
from time import time
from os import path


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

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self, output=[]):
        """String representation of this class"""
        class_str = 'Magnetization simulation properties:\n\n'
        class_str += super().__str__(output)
        return class_str

    def get_hash(self, delays, **kwargs):
        """get_hash

        Calculates an unique hash given by the delays as well as the sample
        structure hash for relevant magnetic parameters.
        Optionally, part of the ``strain_map`` and ``temp_map`` are used.

        Args:
            delays (ndarray[float]): delay grid for the simulation.
            **kwargs (ndarray[float], optional): optional strain and
                temperature profile.

        Returns:
            hash (str): unique hash.

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

        for value in kwargs.values():
            param.append(value)

        return self.S.get_hash(types='magnetic') + '_' + make_hash_md5(param)

    def get_magnetization_map(self, delays, **kwargs):
        """get_magnetization_map

        Returns an absolute `magnetization_map` for the sample structure.
        The angles for `gamma` and `phi` must be in radians as pure numpy
        arrays.
        The `magnetization_map` can depend on the ``temp_map`` and
        ``strain_map`` that can be also calculated for the sample structure.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            **kwargs (ndarray[float], optional): optional strain and
                temperature profile.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
            magnetization profile.

        """
        # create a hash of all simulation parameters
        filename = 'magnetization_map_' \
                   + self.get_hash(delays, **kwargs) \
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

            magnetization_map = self.calc_magnetization_map(delays, **kwargs)

            self.disp_message('Elapsed time for _magnetization_map_:'
                              ' {:f} s'.format(time()-t1))
            self.save(full_filename, {'magnetization_map': magnetization_map},
                      '_magnetization_map_')
        return magnetization_map

    def calc_magnetization_map(self, delays, **kwargs):
        """calc_magnetization_map

        Calculates an absolute ``magnetization_map`` for the sample structure.
        The angles for `gamma` and `phi` must be in radians as pure numpy
        arrays.
        The ``magnetization_map`` can depend on the ``temp_map`` and
        ``strain_map`` that can be also calculated for the sample structure.

        This method is just an interface and should be overwritten for the
        actual simulations.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            **kwargs (ndarray[float], optional): optional strain and
                temperature profile.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
            magnetization profile.

        """
        raise NotImplementedError


class LLB(Magnetization):
    """LLB

    Mean-Field Quantum Landau-Lifshitz-Bloch simulations.

    In collaboration with Theodor Griepe (@Nilodirf)

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

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self):
        """String representation of this class"""
        class_str = 'Landau-Lifshitz-Bloch Magnetization Dynamics simulation ' \
                    'properties:\n\n'
        class_str += super().__str__()
        return class_str

    def calc_mean_field_mag_map(self, temp_map):
        """calc_mean_field_mag_map

        Args:
            temp_map (_type_): _description_

        calculate the mean-field mean magnetization map
        """
        return None

    def calc_magnetization_map(self, delays, temp_map):
        """calc_magnetization_map

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            **kwargs (ndarray[float], optional): optional strain and
                temperature profile.

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
        d_distances = np.diff(distances)
        N = len(distances)

        if self.progress_bar:  # with tqdm progressbar
            pbar = tqdm()
            pbar.set_description('Delay = {:.3f} ps'.format(delays[0]*1e12))
            state = [delays[0], abs(delays[-1]-delays[0])/100]
        else:  # without progressbar
            pbar = None
            state = None

        init_mag = np.zeros([N, 3])
        # calculate the mean magnetization maps for each unique layer
        # and all relevant parameters
        mean_mag_map = self.calc_mean_field_mag_map(temp_map)
        # solve pdepe with method-of-lines
        sol = solve_ivp(
            LLB.odefunc,
            [delays[0], delays[-1]],
            np.reshape(init_mag, N*3, order='F'),
            args=(delays,
                  N,
                  d_distances,
                  distances,
                  temp_map,
                  mean_mag_map,
                  pbar, state),
            t_eval=delays,
            **self.ode_options)

        if pbar is not None:  # close tqdm progressbar if used
            pbar.close()
        magnetization_map = sol.y.T

        magnetization_map = np.array(magnetization_map).reshape([M, N, 3], order='F')
        self.disp_message('Elapsed time for _LLB_: {:f} s'.format(time()-t1))

        return magnetization_map

    @staticmethod
    def odefunc(t, m, delays, N, d_x_grid, x, temp_map, mean_mag_map,
                pbar, state):
        """odefunc

        Ordinary differential equation that is solved for 1D LLB.

        Args:
            t (ndarray[float]): internal time steps of the ode solver.
            m (ndarray[float]): internal variable of the ode solver.
            delays (ndarray[float]): delays range of simulation [s].
            N (int): number of spatial grid points.
            d_x_grid (ndarray[float]): derivative of spatial grid.
            x (ndarray[float]): start point of actual layers.
            temp_map (ndarray[float]): spatio-temporal temperature map.
            mean_mag_map (ndarray[float]): spatio-temporal
                mean-field magnetization map.
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
        dmdt = np.zeros([N, 3])

        # nearest delay index for current time t
        idt = finderb(t, delays)[0]

        # actual calculations
        m_squared = np.sum(np.power(m, 2), axis=1)
        ##########
        # utc = under_tc[t_index, :]
        # otc = ~utc

        # m = m_flat.reshape(len(sample), 3)
        # m_diff_down = np.concatenate((np.diff(m, axis=0), np.zeros((1, 3))), axis=0)
        # m_diff_up = -np.roll(m_diff_down, 1)

        # H_es = e_s(t).T[:, 0][:, np.newaxis] * m_diff_up
        #   + e_s(t).T[:, 1][:, np.newaxis] * m_diff_down
        # H_ani = ani(t)[:, np.newaxis] * (m * ani_perp_sam)

        # factor = 1 / x_p(t)
        # H_th_pref = np.zeros(len(sample))
        # H_th_pref[utc] = (1 - m_squared[utc] / mmag_sam_T(t)[utc] ** 2) * factor[utc] / 2
        # H_th_pref[otc] = -(1 + 3 / 5 * Tc_sam[t_index, otc] / (tes_T(t)[otc]
        #   - Tc_sam[t_index, otc] + 1e-1)) * \
        #                  m_squared[otc] * factor[otc]
        # H_th = np.multiply(H_th_pref[:, np.newaxis], m)

        # H_eff = H_ani + H_es + H_th

        # pref_trans = np.divide(a_t(t), m_squared)
        # pref_long = np.multiply(np.divide(a_p(t), m_squared), np.einsum('ij,ij->i', m, H_eff))

        # # and all the cross products:
        # m_rot = np.cross(m, H_eff)  # precessional term
        # m_trans = np.cross(m, m_rot)  # transverse damping term

        # trans_damp = np.multiply(pref_trans[:, np.newaxis], m_trans)
        # long_damp = np.multiply(pref_long[:, np.newaxis], m)

        # # Now compute the magnetization increment
        # dm_dt = gamma * (-m_rot - trans_damp + long_damp)
        # dmdt = np.zeros([N])

        return np.reshape(dmdt, N*3, order='F')

    @property
    def distances(self):
        return Q_(self._distances, u.meter).to('nm')

    @distances.setter
    def distances(self, distances):
        self._distances = distances.to_base_units().magnitude
