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

__all__ = ['XrayDyn']

__docformat__ = 'restructuredtext'

from . import Scattering
from ...structures.layers import AmorphousLayer, UnitCell
from ...helpers import make_hash_md5, m_power_x, m_times_n, finderb
import numpy as np
import scipy.constants as constants
from time import time
from os import path
from tqdm.auto import trange
import warnings

r_0 = constants.physical_constants['classical electron radius'][0]


class XrayDyn(Scattering):
    r"""XrayDyn

    Dynamical X-ray scattering simulations.

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
        energy (ndarray[float]): photon energies :math:`E` of scattering light
        wl (ndarray[float]): wavelengths :math:`\lambda` of scattering light
        k (ndarray[float]): wavenumber :math:`k` of scattering light
        theta (ndarray[float]): incidence angles :math:`\theta` of scattering
            light
        qz (ndarray[float]): scattering vector :math:`q_z` of scattering light
        polarizations (dict): polarization states and according names.
        pol_in_state (int): incoming polarization state as defined in
            polarizations dict.
        pol_out_state (int): outgoing polarization state as defined in
            polarizations dict.
        pol_in (float): incoming polarization factor (can be a complex ndarray).
        pol_out (float): outgoing polarization factor (can be a complex ndarray).
        last_atom_ref_trans_matrices (list): remember last result of
           atom ref_trans_matrices to speed up calculation.

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.last_atom_ref_trans_matrices = {'atom_ids': [],
                                             'hashes': [],
                                             'H': []}

    def __str__(self):
        """String representation of this class"""
        class_str = 'Dynamical X-Ray Diffraction simulation properties:\n\n'
        class_str += super().__str__()
        return class_str

    def set_incoming_polarization(self, pol_in_state):
        """set_incoming_polarization

        Sets the incoming polarization factor for sigma, pi, and unpolarized
        polarization.

        Args:
            pol_in_state (int): incoming polarization state id.

        """
        self.pol_in_state = pol_in_state
        if (self.pol_in_state == 1):  # circ +
            self.disp_message('incoming polarizations {:s} not implemented'.format(
                self.polarizations[self.pol_in_state]))
            self.set_incoming_polarization(3)
            return
        elif (self.pol_in_state == 2):  # circ-
            self.disp_message('incoming polarizations {:s} not implemented'.format(
                self.polarizations[self.pol_in_state]))
            self.set_incoming_polarization(3)
            return
        elif (self.pol_in_state == 3):  # sigma
            self.pol_in = 0
        elif (self.pol_in_state == 4):  # pi
            self.pol_in = 1
        else:  # unpolarized
            self.pol_in_state = 0
            self.pol_in = 0.5

        self.disp_message('incoming polarizations set to: {:s}'.format(
            self.polarizations[self.pol_in_state]))

    def set_outgoing_polarization(self, pol_out_state):
        """set_outgoing_polarization

        For dynamical X-ray simulation only "no analyzer polarization" is allowed.

        Args:
            pol_out_state (int): outgoing polarization state id.

        """
        self.pol_out_state = pol_out_state
        if self.pol_out_state == 0:
            self.disp_message('analyzer polarizations set to: {:s}'.format(
                self.polarizations[self.pol_out_state]))
        else:
            self.disp_message('XrayDyn does only allow for NO analyzer polarizations')
            self.set_outgoing_polarization(0)

    def homogeneous_reflectivity(self, strains=[], temps=[]):
        r"""homogeneous_reflectivity

        Calculates the reflectivity :math:`R` of the whole sample structure
        and the reflectivity-transmission matrices :math:`M_{RT}` for
        each substructure. The reflectivity of the :math:`2\times 2`
        matrices for each :math:`q_z` is calculates as follow:

        .. math:: R = \left|M_{RT}^t(0,1)/M_{RT}^t(1,1)\right|^2

        Args:
            strains (ndarray[float], optional): strains of each sub-structure
            temps (ndarray[float], optional): temperatures of each sub-structure

        Returns:
            (tuple):
            - *R (ndarray[float])* - homogeneous reflectivity.
            - *A (ndarray[complex])* - reflectivity-transmission matrices of
              sub-structures.

        """
        t1 = time()
        self.disp_message('Calculating _homogenous_reflectivity_ ...')
        # get the reflectivity-transmission matrix of the structure
        RT, A = self.homogeneous_ref_trans_matrix(self.S, strains, temps)
        # calculate the real reflectivity from the RT matrix
        R = self.calc_reflectivity_from_matrix(RT)
        self.disp_message('Elapsed time for _homogenous_reflectivity_: {:f} s'.format(time()-t1))
        return R, A

    def homogeneous_ref_trans_matrix(self, S, strains=[], temps=[]):
        r"""homogeneous_ref_trans_matrix

        Calculates the reflectivity-transmission matrices :math:`M_{RT}` of
        the whole sample structure as well as for each sub-structure.
        The reflectivity-transmission matrix of a single unit cell is
        calculated from the reflection-transmission matrices :math:`H_i`
        of each atom and the phase matrices between the atoms :math:`L_i`:

        .. math:: M_{RT} = \prod_i H_i \ L_i

        For :math:`N` similar layers of unit cells one can calculate the
        :math:`N`-th power of the unit cell :math:`\left(M_{RT}\right)^N`.
        The reflection-transmission matrix for the whole sample
        :math:`M_{RT}^t` consisting of :math:`j = 1\ldots M`
        sub-structures is then again:

        .. math::  M_{RT}^t = \prod_{j=1}^M \left(M_{RT^,j}\right)^{N_j}

        Args:
            S (Structure, UnitCell): structure or sub-structure to calculate on.
            strains (ndarray[float], optional): strains of each sub-structure
            temps (ndarray[float], optional): temperatures of each sub-structure

        Returns:
            (tuple):
            - *RT (ndarray[complex])* - reflectivity-transmission matrix.
            - *A (ndarray[complex])* - reflectivity-transmission matrices of
              sub-structures.

        """
        L = S.get_number_of_sub_structures()
        # if no strains are given we assume no strain (1)
        if len(strains) == 0:
            strains = np.zeros([L])
        else:
            strains = np.array(strains)
        if len(strains) != L:
            raise IndexError('Number of strains must match the number of '
                             'substructures: {:d}'.format(L))

        if len(temps) == 0:
            temps = np.zeros([L, 1])
        else:
            temps = np.array(temps)
            if temps.ndim == 1:
                # add second dimension for temperature
                temps = temps[:, np.newaxis]
            if temps.shape[0] != L:
                raise IndexError('First dimension of temperatures must match the number of '
                                 'substructures {:d}.'.format(L))

            # check length (number of sub-systems) of Debye-Waller factor, which is not checked
            # in setter method
            numel_deb_wal_fac = self.S.get_numel_of_layer_property('deb_wal_fac')
            if temps.shape[1] != numel_deb_wal_fac:
                raise IndexError('Second dimension of temperatures must match the number of '
                                 'subsystems for the Debye-Waller factor: {:d}'.format(
                                     numel_deb_wal_fac))
        # initialize
        RT = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :],
                     (np.size(self._qz, 0), np.size(self._qz, 1), 1, 1))  # ref_trans_matrix
        A = []  # list of ref_trans_matrices of substructures
        counter = 0

        # traverse substructures
        for sub_structure in S.sub_structures:
            if isinstance(sub_structure[0], UnitCell):
                # the sub_structure is an unitCell
                # calculate the ref-trans matrices for N unitCells
                tmp = m_power_x(self.get_uc_ref_trans_matrix(
                        sub_structure[0], strains[counter], temps[counter, :]),
                        sub_structure[1])
                counter += 1
                # remember the result
                A.append([tmp, '{:d}x {:s}'.format(sub_structure[1], sub_structure[0].name)])
            elif isinstance(sub_structure[0], AmorphousLayer):
                raise ValueError('The substructure cannot be an AmorphousLayer!')
            else:
                # its a structure
                # make a recursive call
                idx = np.r_[counter:(counter+sub_structure[0].get_number_of_sub_structures())]
                tmp, tmp2 = self.homogeneous_ref_trans_matrix(
                        sub_structure[0],
                        strains[idx],
                        temps[idx, :])
                A.append([tmp2, sub_structure[0].name + ' substructures'])
                counter = counter+sub_structure[0].get_number_of_sub_structures()
                A.append([tmp, '{:d}x {:s}'.format(sub_structure[1], sub_structure[0].name)])
                # calculate the ref-trans matrices for N sub structures
                tmp = m_power_x(tmp, sub_structure[1])
                A.append([tmp, '{:d}x {:s}'.format(sub_structure[1], sub_structure[0].name)])

            # multiply it to the output
            RT = m_times_n(RT, tmp)

        # if a substrate is included add it at the end
        if S.substrate != []:
            tmp, tmp2 = self.homogeneous_ref_trans_matrix(S.substrate)
            A.append([tmp2, 'static substrate'])
            RT = m_times_n(RT, tmp)

        return RT, A

    def inhomogeneous_reflectivity(self, strain_map, strain_vectors=[], temp_map=np.array([]),
                                   **kwargs):
        """inhomogeneous_reflectivity

        Returns the reflectivity of an inhomogeneously strained sample
        structure for a given ``strain_map`` in position and time, as well
        as for a given set of possible strains for each unit cell in the
        sample structure (``strain_vectors``).
        If no reflectivity is saved in the cache it is caluclated.
        Providing the ``calc_type`` for the calculation the corresponding
        sub-routines for the reflectivity computation are called:

        * ``parallel`` parallelization over the time steps utilizing
          `Dask <https://dask.org/>`_
        * ``distributed`` not implemented in Python, but should be possible
          with `Dask <https://dask.org/>`_ as well
        * ``sequential`` no parallelization at all

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            strain_vectors (list[ndarray[float]], optional): reduced strains per unique
                layer.
            temp_map (ndarray[float], optional): spatio-temporal temperature profile.
            **kwargs:
                - *calc_type (str)* - type of calculation.
                - *dask_client (Dask.Client)* - Dask client.
                - *job (Dask.job)* - Dask job.
                - *num_workers (int)* - Dask number of workers.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        # create a hash of all simulation parameters
        filename = 'inhomogeneous_reflectivity_dyn_' \
                   + self.get_hash(strain_vectors, strain_map=strain_map, temp_map=temp_map) \
                   + '.npz'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            R = tmp['R']
            self.disp_message('_inhomogeneous_reflectivity_ loaded from file:\n\t' + filename)
        else:
            t1 = time()
            self.disp_message('Calculating _inhomogeneousReflectivity_ ...')
            # parse the input arguments
            if not isinstance(strain_map, np.ndarray):
                raise TypeError('strain_map must be a numpy ndarray!')
            if not isinstance(strain_vectors, list):
                raise TypeError('strain_vectors must be a list!')
            if not isinstance(temp_map, np.ndarray):
                raise TypeError('temp_map must be a numpy ndarray!')

            (M, L) = strain_map.shape
            # check length (number of sub-systems) of Debye-Waller factor, which is not checked
            # in setter method
            numel_deb_wal_fac = self.S.get_numel_of_layer_property('deb_wal_fac')

            if len(temp_map) == 0:
                temp_map = np.zeros([M, L, numel_deb_wal_fac])
            else:
                try:
                    temp_map = np.reshape(temp_map, [M, L, numel_deb_wal_fac])
                except ValueError:
                    raise ValueError('Third dimension of temp_map must match the number of '
                                     'sub-systems for the Debye-Waller factor: {:d}'.format(
                                      numel_deb_wal_fac))

                if len(strain_vectors) > 0:
                    warnings.warn('strain_vectors and temp_map are not compatible '
                                  'with each other!\nstrain_vectors takes over.')

            dask_client = kwargs.get('dask_client', [])
            calc_type = kwargs.get('calc_type', 'sequential')
            if calc_type not in ['parallel', 'sequential', 'distributed']:
                raise TypeError('calc_type must be either _parallel_, '
                                '_sequential_, or _distributed_!')
            job = kwargs.get('job')
            num_workers = kwargs.get('num_workers', 1)

            # optinally calculate all ref-trans matrices for all unique unitCells
            # and for all possible strains in advance, if strain_vectors are given
            if len(strain_vectors) > 0:
                RTM = self.get_all_ref_trans_matrices(strain_vectors)
            else:
                RTM = []

            # select the type of computation
            if calc_type == 'parallel':
                R = self.parallel_inhomogeneous_reflectivity(strain_map,
                                                             strain_vectors,
                                                             RTM,
                                                             temp_map,
                                                             dask_client)
            elif calc_type == 'distributed':
                R = self.distributed_inhomogeneous_reflectivity(strain_map,
                                                                strain_vectors,
                                                                job,
                                                                num_workers,
                                                                RTM,
                                                                temp_map)
            else:  # sequential
                R = self.sequential_inhomogeneous_reflectivity(strain_map,
                                                               strain_vectors,
                                                               RTM,
                                                               temp_map)

            self.disp_message('Elapsed time for _inhomogeneous_reflectivity_:'
                              ' {:f} s'.format(time()-t1))
            self.save(full_filename, {'R': R}, '_inhomogeneous_reflectivity_')
        return R

    def sequential_inhomogeneous_reflectivity(self, strain_map, strain_vectors, RTM, temp_map):
        """sequential_inhomogeneous_reflectivity

        Returns the reflectivity of an inhomogeneously strained sample structure
        for a given ``strain_map`` in position and time, as well as for a given
        set of possible strains for each unit cell in the sample structure
        (``strain_vectors``). The function calculates the results sequentially
        without parallelization.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            strain_vectors (list[ndarray[float]]): reduced strains per unique
                layer.
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.
            temp_map (ndarray[float], optional): spatio-temporal temperature profile.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        # initialize
        M = np.size(strain_map, 0)  # delay steps
        R = np.zeros([M, np.size(self._qz, 0), np.size(self._qz, 1)])
        if self.progress_bar:
            iterator = trange(M, desc='Progress', leave=True)
        else:
            iterator = range(M)
        # get the inhomogeneous reflectivity of the sample
        # structure for each time step of the strain_map and temp_map
        for i in iterator:
            R[i, :, :] = self.calc_inhomogeneous_reflectivity(strain_map[i, :],
                                                              strain_vectors,
                                                              RTM,
                                                              temp_map[i, :, :],)
        return R

    def parallel_inhomogeneous_reflectivity(self, strain_map, strain_vectors,
                                            RTM, temp_map, dask_client):
        """parallel_inhomogeneous_reflectivity

        Returns the reflectivity of an inhomogeneously strained sample structure
        for a given ``strain_map`` in position and time, as well as for a given
        set of possible strains for each unit cell in the sample structure
        (``strain_vectors``). The function parallelizes the calculation over the
        time steps, since the results do not depend on each other.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            strain_vectors (list[ndarray[float]]): reduced strains per unique
                layer.
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.
            temp_map (ndarray[float], optional): spatio-temporal temperature profile.
            dask_client (Dask.Client): Dask client.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        if not dask_client:
            raise ValueError('no dask client set')
        from dask import delayed  # to allow parallel computation

        # initialize
        res = []
        M = np.size(strain_map, 0)  # delay steps
        N = np.size(self._qz, 0)  # energy steps
        K = np.size(self._qz, 1)  # qz steps

        R = np.zeros([M, N, K])
        # init unity matrix for matrix multiplication
        RTU = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (N, K, 1, 1))

        # precalculate the substrate ref_trans_matrix if present
        if self.S.substrate != []:
            RTS, _ = self.homogeneous_ref_trans_matrix(self.S.substrate)
        else:
            RTS = RTU

        if len(strain_vectors) > 0:
            uc_indices, _, _ = self.S.get_layer_vectors()
            # make RTM available for all works
            remote_RTM = dask_client.scatter(RTM)
            remote_RTU = dask_client.scatter(RTU)
            remote_uc_indices = dask_client.scatter(uc_indices)
            remote_strain_vectors = dask_client.scatter(strain_vectors)
            # create dask.delayed tasks for all delay steps
            for i in range(M):
                RT = delayed(XrayDyn.lookup_inhomogeneous_ref_trans_matrix)(
                        remote_uc_indices,
                        remote_RTU,
                        strain_map[i, :],
                        remote_strain_vectors,
                        remote_RTM
                        )
                RT = delayed(m_times_n)(RT, RTS)
                Ri = delayed(XrayDyn.calc_reflectivity_from_matrix)(RT)
                res.append(Ri)
        else:
            for i in range(M):
                RT = delayed(self.calc_inhomogeneous_ref_trans_matrix)(
                        strain_map[i, :],
                        temp_map[i, :, :]
                        )
                RT = delayed(m_times_n)(RT, RTS)
                Ri = delayed(XrayDyn.calc_reflectivity_from_matrix)(RT)
                res.append(Ri)

        # compute results
        res = dask_client.compute(res, sync=True)

        # reorder results to reflectivity matrix
        for i in range(M):
            R[i, :, :] = res[i]

        return R

    def distributed_inhomogeneous_reflectivity(self, strain_map, strain_vectors, RTM,
                                               temp_map, job, num_worker):
        """distributed_inhomogeneous_reflectivity

        This is a stub. Not yet implemented in python.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            strain_vectors (list[ndarray[float]]): reduced strains per unique
                layer.
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.
            temp_map (ndarray[float], optional): spatio-temporal temperature profile.
            job (Dask.job): Dask job.
            num_workers (int): Dask number of workers.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        raise NotImplementedError

    def calc_inhomogeneous_reflectivity(self, strains, strain_vectors, RTM, temps):
        r"""calc_inhomogeneous_reflectivity

        Calculates the reflectivity of a inhomogeneous sample structure for
        given ``strain_vectors`` for a single time step. Similar to the
        homogeneous sample structure, the reflectivity of an unit cell is
        calculated from the reflection-transmission matrices :math:`H_i` of
        each atom and the phase matrices between the atoms :math:`L_i` in the
        unit cell:

        .. math:: M_{RT} = \prod_i H_i \ L_i

        Since all layers are generally inhomogeneously strained we have to
        traverse all individual unit cells (:math:`j = 1\ldots M`) in the
        sample to calculate the total reflection-transmission matrix
        :math:`M_{RT}^t`:

        .. math:: M_{RT}^t = \prod_{j=1}^M M_{RT,j}

        The reflectivity of the :math:`2\times 2` matrices for each :math:`q_z`
        is calculates as follow:

        .. math:: R = \left|M_{RT}^t(1,2)/M_{RT}^t(2,2)\right|^2

        Args:
            strains (ndarray[float]): spatial strain profile for single time
                step.
            strain_vectors (list[ndarray[float]]): reduced strains per unique
                layer.
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.
            temps (ndarray[float]): spatial temperature profile for single time
                step.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        # initialize ref_trans_matrix
        N = np.shape(self._qz)[1]  # number of q_z
        M = np.shape(self._qz)[0]  # number of energies
        uc_indices, _, _ = self.S.get_layer_vectors()

        # initialize ref_trans_matrix
        RTU = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (M, N, 1, 1))

        if len(strain_vectors) > 0:
            RT = XrayDyn.lookup_inhomogeneous_ref_trans_matrix(uc_indices,
                                                               RTU,
                                                               strains,
                                                               strain_vectors,
                                                               RTM)
        else:
            RT = self.calc_inhomogeneous_ref_trans_matrix(strains, temps)

        # if a substrate is included add it at the end
        if self.S.substrate != []:
            RTS, _ = self.homogeneous_ref_trans_matrix(self.S.substrate)
            RT = m_times_n(RT, RTS)
        # calculate reflectivity from ref-trans matrix
        R = self.calc_reflectivity_from_matrix(RT)
        return R

    def calc_inhomogeneous_ref_trans_matrix(self, strains, temps):
        r"""calc_inhomogeneous_ref_trans_matrix

        Sub-function of :meth:`calc_inhomogeneous_reflectivity` and for
        parallel computing (needs to be static) only for calculating the
        total reflection-transmission matrix :math:`M_{RT}^t`:

        .. math:: M_{RT}^t = \prod_{j=1}^M M_{RT,j}

        Args:
            strains (ndarray[float]): spatial strain profile for single time
                step.
            temps (ndarray[float]): spatial temperature profile for single time
                step.

        Returns:
            RT (ndarray[complex]): reflection-transmission matrix.

        """
        _, _, uc_handles = self.S.get_layer_vectors()

        N = np.shape(self._qz)[1]  # number of q_z
        M = np.shape(self._qz)[0]  # number of energies
        RT = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (M, N, 1, 1))
        # traverse all unit cells in the sample structure
        for i, uc in enumerate(uc_handles):

            if not isinstance(uc, UnitCell):
                raise ValueError('All layers  must be UnitCells!')
            RT = m_times_n(RT, self.get_uc_ref_trans_matrix(uc, strains[i], temps[i, :]))

        return RT

    @staticmethod
    def lookup_inhomogeneous_ref_trans_matrix(uc_indices, RT, strains,
                                              strain_vectors, RTM):
        r"""lookup_inhomogeneous_ref_trans_matrix

        Sub-function of :meth:`calc_inhomogeneous_reflectivity` and for
        parallel computing (needs to be static) only for looking up the
        total reflection-transmission matrix :math:`M_{RT}^t`:

        .. math:: M_{RT}^t = \prod_{j=1}^M M_{RT,j}

        Args:
            uc_indices (ndarray[float]): unit cell indices.
            RT (ndarray[complex]): reflection-transmission matrix.
            strains (ndarray[float]): spatial strain profile for single time
                step.
            strain_vectors (list[ndarray[float]]): reduced strains per unique
                layer.
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.

        Returns:
            RT (ndarray[complex]): reflection-transmission matrix.

        """
        # traverse all unit cells in the sample structure
        for i, uc_index in enumerate(uc_indices):
            # Find the ref-trans matrix in the RTM cell array for the
            # current unit_cell ID and applied strain. Use the
            # ``knnsearch`` function to find the nearest strain value.
            strain_index = finderb(strains[i], strain_vectors[int(uc_index)])[0]
            tmp = RTM[int(uc_index)][strain_index]
            if tmp is not []:
                RT = m_times_n(RT, tmp)
            else:
                raise ValueError('RTM not found')

        return RT

    def get_all_ref_trans_matrices(self, *args):
        """get_all_ref_trans_matrices

        Returns a list of all reflection-transmission matrices for each
        unique unit cell in the sample structure for a given set of applied
        strains for each unique unit cell given by the ``strain_vectors``
        input. If this data was saved on disk before, it is loaded, otherwise
        it is calculated.

        Args:
            args (list[ndarray[float]], optional): reduced strains per unique
                layer.

        Returns:
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
            all given strains per unique layer.

        """
        if len(args) == 0:
            strain_vectors = [np.array([1])]*self.S.get_number_of_unique_layers()
        else:
            strain_vectors = args[0]
        # create a hash of all simulation parameters
        filename = 'all_ref_trans_matrices_dyn_' \
            + self.get_hash(strain_vectors) + '.npz'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            RTM = tmp['RTM']
            self.disp_message('_all_ref_trans_matrices_dyn_ loaded from file:\n\t' + filename)
        else:
            # nothing found so calculate it and save it
            RTM = self.calc_all_ref_trans_matrices(strain_vectors)
            self.save(full_filename, {'RTM': RTM}, '_all_ref_trans_matrices_dyn_')
        return RTM

    def calc_all_ref_trans_matrices(self, *args):
        """calc_all_ref_trans_matrices

        Calculates a list of all reflection-transmission matrices for each
        unique unit cell in the sample structure for a given set of applied
        strains to each unique unit cell given by the ``strain_vectors`` input.

        Args::
            args (list[ndarray[float]], optional): reduced strains per unique
                layer.

        Returns:
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
            all given strains per unique layer.

        """
        t1 = time()
        self.disp_message('Calculate all _ref_trans_matrices_ ...')
        # initialize
        uc_ids, uc_handles = self.S.get_unique_layers()
        # if no strain_vectors are given we just do it for no strain (1)
        if len(args) == 0:
            strain_vectors = [np.array([1])]*len(uc_ids)
        else:
            strain_vectors = args[0]
        # check if there are strains for each unique unitCell
        if len(strain_vectors) is not len(uc_ids):
            raise TypeError('The strain vector has not the same size '
                            'as number of unique unit cells')

        # initialize ref_trans_matrices
        RTM = []

        # traverse all unique unit_cells
        for i, uc in enumerate(uc_handles):
            # traverse all strains in the strain_vector for this unique
            # unit_cell
            if not isinstance(uc, UnitCell):
                raise ValueError('All layers  must be UnitCells!')
            temp = []
            for strain in strain_vectors[i]:
                temp.append(self.get_uc_ref_trans_matrix(uc, strain))
            RTM.append(temp)
        self.disp_message('Elapsed time for _ref_trans_matrices_: {:f} s'.format(time()-t1))
        return RTM

    def get_uc_ref_trans_matrix(self, uc, strain=0, temp=np.array([0])):
        r"""get_uc_ref_trans_matrix

        Returns the reflection-transmission matrix of a unit cell:

        .. math:: M_{RT} = \prod_i H_i \  L_i

        where :math:`H_i` and :math:`L_i` are the atomic reflection-
        transmission matrix and the phase matrix for the atomic distances,
        respectively.

        Args:
            uc (UnitCell): unit cell object.
            strain (float, optional): strain of unit cell.
            temp (ndarray[float], optional): temperature of unit cell.

        Returns:
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.

        """
        M = len(self._energy)  # number of energies
        N = np.shape(self._qz)[1]  # number of q_z
        K = uc.num_atoms  # number of atoms
        # initialize matrices
        RTM = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (M, N, 1, 1))
        # traverse all atoms of the unit cell
        for i in range(K):
            # Calculate the relative distance between the atoms.
            # The relative position is calculated by the function handle
            # stored in the atoms list as 3rd element. This
            # function returns a relative postion dependent on the
            # applied strain.
            if i == (K-1):  # its the last atom
                rel_dist = (strain+1)-uc.atoms[i][1](strain)
            else:
                rel_dist = uc.atoms[i+1][1](strain)-uc.atoms[i][1](strain)

            # sum Debye-Waller factors for all sub-systems
            deb_wal_fac = np.sum(np.array([dbf(T) for dbf, T in zip(uc.deb_wal_fac, temp)]))

            # get the reflection-transmission matrix and phase matrix
            # from all atoms in the unit cell and multiply them
            # together
            RTM = m_times_n(RTM,
                            self.get_atom_ref_trans_matrix(uc.atoms[i][0],
                                                           uc._area,
                                                           deb_wal_fac))
            RTM = m_times_n(RTM,
                            self.get_atom_phase_matrix(rel_dist*uc._c_axis))
        return RTM

    def get_atom_ref_trans_matrix(self, atom, area, deb_wal_fac):
        r"""get_atom_ref_trans_matrix

        Calculates the reflection-transmission matrix of an atom from dynamical
        x-ray theory:

        .. math::

            H = \frac{1}{\tau} \begin{bmatrix}
            \left(\tau^2 - \rho^2\right) & \rho \\
            -\rho & 1
            \end{bmatrix}

        Args:
            atom (Atom, AtomMixed): atom or mixed atom
            area (float): area of the unit cell [m²]
            deb_wal_fac (float): Debye-Waller factor for unit cell

        Returns:
            H (ndarray[complex]): reflection-transmission matrix

        """
        # check for already calculated data
        _hash = make_hash_md5([self._energy, self._qz, self.pol_in_state, self.pol_out_state,
                               area, deb_wal_fac])
        try:
            index = self.last_atom_ref_trans_matrices['atom_ids'].index(atom.id)
        except ValueError:
            index = -1

        if (index >= 0) and (_hash == self.last_atom_ref_trans_matrices['hashes'][index]):
            # These are the same X-ray parameters as last time so we
            # can use the same matrix again for this atom
            H = self.last_atom_ref_trans_matrices['H'][index]
        else:
            # These are new parameters so we have to calculate.
            # Get the reflection-transmission-factors
            rho = self.get_atom_reflection_factor(atom, area, deb_wal_fac)
            tau = self.get_atom_transmission_factor(atom, area, deb_wal_fac)
            # calculate the reflection-transmission matrix
            H = np.zeros([np.shape(self._qz)[0], np.shape(self._qz)[1], 2, 2], dtype=np.complex128)
            H[:, :, 0, 0] = (1/tau)*(tau**2-rho**2)
            H[:, :, 0, 1] = (1/tau)*(rho)
            H[:, :, 1, 0] = (1/tau)*(-rho)
            H[:, :, 1, 1] = (1/tau)
            # remember this matrix for next use with the same
            # parameters for this atom
            if index >= 0:
                self.last_atom_ref_trans_matrices['atom_ids'][index] = atom.id
                self.last_atom_ref_trans_matrices['hashes'][index] = _hash
                self.last_atom_ref_trans_matrices['H'][index] = H
            else:
                self.last_atom_ref_trans_matrices['atom_ids'].append(atom.id)
                self.last_atom_ref_trans_matrices['hashes'].append(_hash)
                self.last_atom_ref_trans_matrices['H'].append(H)
        return H

    def get_atom_reflection_factor(self, atom, area, deb_wal_fac):
        r"""get_atom_reflection_factor

        Calculates the reflection factor from dynamical x-ray theory:

        .. math::  \rho = \frac{-i 4 \pi \ r_e \ f(E,q_z) \ P(\theta)
                   \exp(-M)}{q_z \ A}

        - :math:`r_e` is the electron radius
        - :math:`f(E,q_z)` is the energy and angle dispersive atomic
          form factor
        - :math:`P(q_z)` is the polarization factor
        - :math:`A` is the area in :math:`x-y` plane on which the atom
          is placed
        - :math:`M = 0.5 \mbox{dbf} q_z^2` where
          :math:`\mbox{dbf} = \langle u^2\rangle` is the average
          thermal vibration of the atoms - Debye-Waller factor

        Args:
            atom (Atom, AtomMixed): atom or mixed atom
            area (float): area of the unit cell [m²]
            deb_wal_fac (float): Debye-Waller factor for unit cell

        Returns:
            rho (complex): reflection factor

        """
        rho = (-4j*np.pi*r_0
               * atom.get_cm_atomic_form_factor(self._energy, self._qz)
               * self.get_polarization_factor(self._theta)
               * np.exp(-0.5*deb_wal_fac*self._qz**2))/(self._qz*area)
        return rho

    def get_atom_transmission_factor(self, atom, area, deb_wal_fac):
        r"""get_atom_transmission_factor

        Calculates the transmission factor from dynamical x-ray theory:

        .. math:: \tau = 1 - \frac{i 4 \pi r_e f(E,0) \exp(-M)}{q_z A}

        - :math:`r_e` is the electron radius
        - :math:`f(E,0)` is the energy dispersive atomic form factor
          (no angle correction)
        - :math:`A` is the area in :math:`x-y` plane on which the atom
          is placed
        - :math:`M = 0.5 \mbox{dbf} q_z^2` where
          :math:`\mbox{dbf} = \langle u^2\rangle` is the average
          thermal vibration of the atoms - Debye-Waller factor

        Args:
            atom (Atom, AtomMixed): atom or mixed atom
            area (float): area of the unit cell [m²]
            deb_wal_fac (float): Debye-Waller factor for unit cell

        Returns:
            tau (complex): transmission factor

        """
        tau = 1 - (4j*np.pi*r_0
                   * atom.get_cm_atomic_form_factor(self._energy, np.zeros_like(self._qz))
                   * np.exp(-0.5*deb_wal_fac*self._qz**2))/(self._qz*area)
        return tau

    def get_atom_phase_matrix(self, distance):
        r"""get_atom_phase_matrix

        Calculates the phase matrix from dynamical x-ray theory:

        .. math::

            L = \begin{bmatrix}
            \exp(i \phi) & 0 \\
            0            & \exp(-i \phi)
            \end{bmatrix}

        Args:
            distance (float): distance between atomic planes

        Returns:
            L (ndarray[complex]): phase matrix

        """
        phi = self.get_atom_phase_factor(distance)
        L = np.zeros([np.shape(self._qz)[0], np.shape(self._qz)[1], 2, 2], dtype=np.complex128)
        L[:, :, 0, 0] = np.exp(1j*phi)
        L[:, :, 1, 1] = np.exp(-1j*phi)
        return L

    def get_atom_phase_factor(self, distance):
        r"""get_atom_phase_factor

        Calculates the phase factor :math:`\phi` for a distance :math:`d`
        from dynamical x-ray theory:

        .. math:: \phi = \frac{d \ q_z}{2}

        Args:
            distance (float): distance between atomic planes

        Returns:
            phi (float): phase factor

        """
        phi = distance * self._qz/2
        return phi

    @staticmethod
    def calc_reflectivity_from_matrix(M):
        r"""calc_reflectivity_from_matrix

        Calculates the reflectivity from an :math:`2\times2` matrix of
        transmission and reflectivity factors:

        .. math:: R = \left|M(0,1)/M(1,1)\right|^2

        Args:
            M (ndarray[complex]): reflection-transmission matrix

        Returns:
            R (ndarray[float]): reflectivity

        """
        return np.abs(M[:, :, 0, 1]/M[:, :, 1, 1])**2
