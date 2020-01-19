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

"""A :mod:`XrayDyn` module """

__all__ = ["XrayDyn"]

__docformat__ = "restructuredtext"

import numpy as np
import scipy.constants as constants
from .xray import Xray
from .unitCell import UnitCell
from time import time
from os import path
from tqdm import trange
from .helpers import make_hash_md5, m_power_x, m_times_n, finderb

r_0 = constants.physical_constants['classical electron radius'][0]


class XrayDyn(Xray):
    """XrayDyn

    Dynamical Xray simulations

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Attributes:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results
        polarization (float): polarization state
        last_atom_ref_trans_matrices (list): remember last result of
           atom ref_trans_matrices to speed up calculation

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

    def homogeneous_reflectivity(self, *args):
        """homogeneous_reflectivity

        Returns the reflectivity :math:`R` of the whole sample structure
        and the reflectivity-transmission matrices :math:`M_{RT}` for
        each substructure. The reflectivity of the :math:`2\\times 2`
        matrices for each :math:`q_z` is calculates as follow:

        .. math:: R = \left|M_{RT}^t(0,1)/M_{RT}^t(1,1)\\right|^2

        """
        # if no strains are given we assume no strain
        if len(args) == 0:
            strains = np.zeros([self.S.get_number_of_sub_structures(), 1])
        else:
            strains = args[0]
        t1 = time()
        self.disp_message('Calculating _homogenous_reflectivity_ ...')
        # get the reflectivity-transmisson matrix of the structure
        RT, A = self.homogeneous_ref_trans_matrix(self.S, strains)
        # calculate the real reflectivity from the RT matrix
        R = self.calc_reflectivity_from_matrix(RT)
        self.disp_message('Elapsed time for _homogenous_reflectivity_: {:f} s'.format(time()-t1))
        return R, A

    def homogeneous_ref_trans_matrix(self, S, *args):
        """homogeneous_ref_trans_matrix

        Returns the reflectivity-transmission matrices :math:`M_{RT}` of
        the whole sample structure as well as for each sub structure.
        The reflectivity-transmission matrix of a single unit cell is
        calculated from the reflection-transmission matrices :math:`H_i`
        of each atom and the phase matrices between the atoms :math:`L_i`:

        .. math:: M_{RT} = \prod_i H_i \ L_i

        For :math:`N` similar layers of unit cells one can calculate the
        N-th power of the unit cell :math:`\left(M_{RT}\\right)^N`. The
        reflection-transmission matrix for the whole sample
        :math:`M_{RT}^t` consisting of :math:`j = 1\ldots M`
        substructures is then again:

        .. math::  M_{RT}^t = \prod_{j=1}^M \left(M_{RT^,j}\\right)^{N_j}

        """
        # if no strains are given we assume no strain (1)
        if len(args) == 0:
            strains = np.zeros([S.get_number_of_sub_structures(), 1])
        else:
            strains = args[0]
        # initialize
        RT = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :],
                     (np.size(self._qz, 0), np.size(self._qz, 1), 1, 1))  # ref_trans_matrix
        A = []  # list of ref_trans_matrices of substructures
        strainCounter = 0

        # traverse substructures
        for i, sub_structure in enumerate(S.sub_structures):
            if isinstance(sub_structure[0], UnitCell):
                # the sub_structure is an unitCell
                # calculate the ref-trans matrices for N unitCells
                temp = m_power_x(self.get_uc_ref_trans_matrix(
                        sub_structure[0], strains[strainCounter]),
                        sub_structure[1])
                strainCounter += 1
                # remember the result
                A.append([temp, '{:d}x {:s}'.format(sub_structure[1], sub_structure[0].name)])
            else:
                # its a structure
                # make a recursive call
                temp, temp2 = self.homogeneous_ref_trans_matrix(
                        sub_structure[0],
                        strains[strainCounter:(strainCounter
                                               + sub_structure[0].get_number_of_sub_structures())])
                A.append([temp2, sub_structure[0].name + ' substructures'])
                strainCounter = strainCounter+sub_structure[0].get_number_of_sub_structures()
                A.append([temp, '{:d}x {:s}'.format(sub_structure[1], sub_structure[0].name)])
                # calculate the ref-trans matrices for N sub structures
                temp = m_power_x(temp, sub_structure[1])
                A.append([temp, '{:d}x {:s}'.format(sub_structure[1], sub_structure[0].name)])

            # multiply it to the output
            RT = m_times_n(RT, temp)

        # if a substrate is included add it at the end
        if S.substrate != []:
            temp, temp2 = self.homogeneous_ref_trans_matrix(S.substrate)
            A.append([temp2, 'static substrate'])
            RT = m_times_n(RT, temp)

        return RT, A

    """
    Inhomogenous Sample Structure
    All unit cells in the sample are inhomogeneously
    strained. This is generally the case when calculating the
    transient rocking curves for coherent phonon dynamics in the
    sample structure.
    """

    def inhomogeneous_reflectivity(self, strain_map, strain_vectors, **kwargs):
        """inhomogeneous_reflectivity

        Returns the reflectivity of an inhomogenously strained sample
        structure for a given _strainMap_ in position and time, as well
        as for a given set of possible strains for each unit cell in the
        sample structure (``strain_vectors``).
        If no reflectivity is saved in the cache it is caluclated.
        Providing the _type_ (parallel [default], sequential,
        distributed) for the calculation the corresponding subroutines
        for the reflectivity computation are called:

        * ``parallel`` parallelization over the time steps utilizing
          Dask
        * ``distributed`` not implemented in Python, yet
        * ``sequential`` no parallelization at all

        """
        # create a hash of all simulation parameters
        filename = 'inhomogeneous_reflectivity_dyn_' \
                   + self.get_hash(strain_vectors,
                                   qz=self.qz,
                                   energy=self.energy,
                                   strain_map=strain_map) \
                   + '.npy'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            R = np.load(full_filename)
            self.disp_message('_inhomogeneous_reflectivity_ loaded from file:\n\t' + filename)
        else:
            t1 = time()
            self.disp_message('Calculating _inhomogenousReflectivity_ ...')
            # parse the input arguments
            if not isinstance(strain_map, np.ndarray):
                raise TypeError('strain_map must be a numpy ndarray!')
            if not isinstance(strain_vectors, list):
                raise TypeError('strain_vectors must be a list!')

            dask_client = kwargs.get('dask_client', [])
            calc_type = kwargs.get('calc_type', 'sequential')
            if calc_type not in ['parallel', 'sequential', 'distributed']:
                raise TypeError('calc_type must be either _parallel_, '
                                '_sequential_, or _distributed_!')
            job = kwargs.get('job')
            num_workers = kwargs.get('num_workers', 1)

            # All ref-trans matrices for all unique unitCells and for all
            # possible strains, given by strainVectors, are calculated in
            # advance.
            RTM = self.get_all_ref_trans_matrices(strain_vectors)

            # select the type of computation
            if calc_type == 'parallel':
                R = self.parallel_inhomogeneous_reflectivity(strain_map,
                                                             strain_vectors,
                                                             RTM,
                                                             dask_client)
            elif calc_type == 'distributed':
                R = self.distributed_inhomogeneous_reflectivity(strain_map,
                                                                strain_vectors,
                                                                job,
                                                                num_workers,
                                                                RTM)
            else:  # sequential
                R = self.sequential_inhomogeneous_reflectivity(strain_map,
                                                               strain_vectors,
                                                               RTM)

            self.disp_message('Elapsed time for _inhomogenous_reflectivity_:'
                              ' {:f} s'.format(time()-t1))
            self.save(full_filename, R, '_inhomogeneous_reflectivity_')
        return R

    def sequential_inhomogeneous_reflectivity(self, strain_map, strain_vectors, RTM):
        """sequential_inhomogeneous_reflectivity

        Returns the reflectivity of an inhomogenously strained sample
        structure for a given ``strain_map`` in position and time, as
        well as for a given set of possible strains for each unit cell
        in the sample structure (``strain_vectors``).
        The function calculates the results sequentially without
        parallelization.

        """
        # initialize
        N = np.size(strain_map, 0)  # delay steps
        R = np.zeros([N, np.size(self._qz, 0), np.size(self._qz, 1)])
        for i in trange(N, desc='Progress', leave=True):
            # get the inhomogenous reflectivity of the sample
            # structure for each time step of the strain map
            R[i, :, :] = self.calc_inhomogeneous_reflectivity(strain_map[i, :],
                                                              strain_vectors,
                                                              RTM)
        return R

    def parallel_inhomogeneous_reflectivity(self, strain_map, strain_vectors,
                                            RTM, dask_client):
        """parallel_inhomogeneous_reflectivity

        Returns the reflectivity of an inhomogenously strained sample
        structure for a given ``strain_map`` in position and time, as
        well as for a given set of possible strains for each unit cell
        in the sample structure (``strain_vectors``).
        The function tries to parallize the calculation over the time
        steps, since the results do not depent on each other.

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
        uc_indicies, _, _ = self.S.get_unit_cell_vectors()
        # init unity matrix for matrix multiplication
        RTU = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (N, K, 1, 1))
        # make RTM available for all works
        remote_RTM = dask_client.scatter(RTM)
        remote_RTU = dask_client.scatter(RTU)
        remote_uc_indicies = dask_client.scatter(uc_indicies)
        remote_strain_vectors = dask_client.scatter(strain_vectors)

        # precalculate the substrate ref_trans_matrix if present
        if self.S.substrate != []:
            RTS, _ = self.homogeneous_ref_trans_matrix(self.S.substrate)
        else:
            RTS = RTU

        # create dask.delayed tasks for all delay steps
        for i in range(M):
            RT = delayed(XrayDyn.calc_inhomogeneous_ref_trans_matrix)(
                    remote_uc_indicies,
                    remote_RTU,
                    strain_map[i, :],
                    remote_strain_vectors,
                    remote_RTM)
            RT = delayed(m_times_n)(RT, RTS)
            Ri = delayed(XrayDyn.calc_reflectivity_from_matrix)(RT)
            res.append(Ri)

        # compute results
        res = dask_client.compute(res, sync=True)

        # reorder results to reflectivity matrix
        for i in range(M):
            R[i, :, :] = res[i]

        return R

    def distributed_inhomogeneous_reflectivity(self, job, num_worker,
                                               strain_map, strain_vectors, RTM):
        """distributed_inhomogeneous_reflectivity

        This is a stub. Not yet implemented in python.

        """
        return

    def calc_inhomogeneous_reflectivity(self, strains, strain_vectors, RTM):
        """calc_inhomogeneous_reflectivity

        Calculates the reflectivity of a inhomogenous sample structure
        for a given strain vector for a single time step. Similar to the
        homogeneous sample structure, the reflectivity of an unit cell
        is calculated from the reflection-transmission matrices
        :math:`H_i` of each atom and the phase matrices between the
        atoms :math:`L_i`:

        .. math: M_{RT} = \prod_i H_i \ L_i

        Since all layers are generally inhomogeneously strained we have
        to traverse all individual unit cells (:math:`j = 1\ldots M`) in
        the sample to calculate the total reflection-transmission matrix
        :math:`M_{RT}^t`:

        .. math: M_{RT}^t = \prod_{j=1}^M M_{RT,j}

        The reflectivity of the :math:`2\\times 2` matrices for each
        :math:`q_z` is calculates as follow:

        .. math: R = \left|M_{RT}^t(1,2)/M_{RT}^t(2,2)\\right|^2

        """
        # initialize ref_trans_matrix
        N = np.shape(self._qz)[1]  # number of q_z
        M = np.shape(self._qz)[0]  # number of energies
        uc_indicies, _, _ = self.S.get_unit_cell_vectors()

        # initialize ref_trans_matrix
        RTU = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (M, N, 1, 1))

        RT = XrayDyn.calc_inhomogeneous_ref_trans_matrix(uc_indicies,
                                                         RTU,
                                                         strains,
                                                         strain_vectors,
                                                         RTM)

        # if a substrate is included add it at the end
        if self.S.substrate != []:
            RTS, _ = self.homogeneous_ref_trans_matrix(self.S.substrate)
            RT = m_times_n(RT, RTS)
        # calculate reflectivity from ref-trans matrix
        R = self.calc_reflectivity_from_matrix(RT)
        return R

    @staticmethod
    def calc_inhomogeneous_ref_trans_matrix(uc_indicies, RT, strains,
                                            strain_vectors, RTM):
        """calc_inhomogeneous_ref_trans_matrix

        Sub-function of ``calc_inhomogeneous_reflectivity`` and for
        parallel computing (needs to be static) only for calculating the
        total reflection-transmission matrix :math:`M_{RT}^t`:

        .. math: M_{RT}^t = \prod_{j=1}^M M_{RT,j}

        """
        # traverse all unit cells in the sample structure
        for i, uc_index in enumerate(uc_indicies):
            # Find the ref-trans matrix in the RTM cell array for the
            # current unit_cell ID and applied strain. Use the
            # ``knnsearch`` funtion to find the nearest strain value.
            strain_index = finderb(strains[i], strain_vectors[uc_index])[0]
            temp = RTM[uc_index][strain_index]
            if temp is not []:
                RT = m_times_n(RT, temp)
            else:
                raise(ValueError, 'RTM not found')

        return RT

    def get_all_ref_trans_matrices(self, *args):
        """get_all_ref_trans_matrices

        Returns a list of all reflection-transmission matrices for
        each unique unit cell in the sample structure for a given set of
        applied strains for each unique unit cell given by the
        ``strain_vectors`` input. If this data was saved on disk before,
        it is loaded, otherwise it is calculated.

        """
        if len(args) == 0:
            strain_vectors = [np.array([1])]*self.S.get_number_of_unique_unit_cells()
        else:
            strain_vectors = args[0]
        # create a hash of all simulation parameters
        filename = 'all_ref_trans_matrices_dyn_' \
            + self.get_hash(strain_vectors, energy=self._energy, qz=self._qz) + '.npy'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            RTM = np.load(full_filename)
            self.disp_message('_all_ref_trans_matrices_dyn_ loaded from file:\n\t' + filename)
        else:
            # nothing found so calculate it and save it
            RTM = self.calc_all_ref_trans_matrices(strain_vectors)
            self.save(full_filename, RTM, '_all_ref_trans_matrices_dyn_')
        return RTM

    def calc_all_ref_trans_matrices(self, *args):
        """calc_all_ref_trans_matrices

        Calculates a list of all reflection-transmission matrices
        for each unique unit cell in the sample structure for a given
        set of applied strains to each unique unit cell given by the
        ``strainVectors`` input.

        """
        t1 = time()
        self.disp_message('Calculate all _ref_trans_matricies_ ...')
        # initalize
        uc_ids, uc_handles = self.S.get_unique_unit_cells()
        # if no strain_vecorts are given we just do it for no strain (1)
        if len(args) == 0:
            strain_vectors = [np.array([1])]*len(uc_ids)
        else:
            strain_vectors = args[0]
        # check if there are strains for each unique unitCell
        if len(strain_vectors) is not len(uc_ids):
            raise TypeError('The strain vecotr has not the same size '
                            'as number of unique unit cells')

        # initialize refTransMatrices
        RTM = []

        # traverse all unique unitCells
        for i, uc in enumerate(uc_handles):
            # traverse all strains in the strain_vector for this unique
            # unit_cell
            temp = []
            for strain in strain_vectors[i]:
                temp.append(self.get_uc_ref_trans_matrix(uc, strain))
            RTM.append(temp)
        self.disp_message('Elapsed time for _ref_trans_matricies_: {:f} s'.format(time()-t1))
        return RTM

    def get_uc_ref_trans_matrix(self, uc, *args):
        """get_uc_ref_trans_matrix

        Returns the reflection-transmission matrix of a unit cell:

        .. math:: M_{RT} = \prod_i H_i \  L_i

        where :math:`H_i` and :math:`L_i` are the atomic reflection-
        transmission matrix and the phase matrix for the atomic
        distances, respectively.

        """
        if len(args) == 0:
            strain = 0  # set the defalut strain to 0
        else:
            strain = args[0]

        M = len(self._energy)  # number of energies
        N = np.shape(self._qz)[1]  # number of q_z
        K = uc.num_atoms  # number of atoms
        # initialize matrices
        RTM = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (M, N, 1, 1))
        # traverse all atoms of the unit cell
        for i in range(K):
            # Calculate the relative distance between the atoms.
            # The raltive position is calculated by the function handle
            # stored in the atoms list as 3rd element. This
            # function returns a relative postion dependent on the
            # applied strain.
            if i == (K-1):  # its the last atom
                del_dist = (strain+1)-uc.atoms[i][1](strain)
            else:
                del_dist = uc.atoms[i+1][1](strain)-uc.atoms[i][1](strain)

            # get the reflection-transmission matrix and phase matrix
            # from all atoms in the unit cell and multiply them
            # together
            RTM = m_times_n(RTM,
                            self.get_atom_ref_trans_matrix(uc.atoms[i][0],
                                                           uc._area,
                                                           uc._deb_wal_fac))
            RTM = m_times_n(RTM,
                            self.get_atom_phase_matrix(del_dist*uc._c_axis))
        return RTM

    def get_atom_ref_trans_matrix(self, atom, area, deb_wal_fac):
        """get_atom_ref_trans_matrix

        Returns the reflection-transmission matrix of an atom from
        dynamical xray theory:

        .. math::

            H = \\frac{1}{\\tau} \\begin{bmatrix}
            \left(\\tau^2 - \\rho^2\\right) & \\rho \\\\
            -\\rho & 1
            \\end{bmatrix}

        """
        # check for already calculated data
        _hash = make_hash_md5([self._energy, self._qz, self.polarization, area, deb_wal_fac])
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
            H = np.zeros([np.shape(self._qz)[0], np.shape(self._qz)[1], 2, 2], dtype=np.cfloat)
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
        """get_atom_reflection_factor

        Returns the reflection factor from dynamical xray theory:

        .. math::  \\rho = \\frac{-i 4 \pi \ r_e \ f(E,q_z) \ P(\\theta)
                   \exp(-M)}{q_z \ A}

        - :math:`r_e` is the electron radius
        - :math:`f(E,q_z)` is the energy and angle dispersive atomic
          form factor
        - :math:`P(q_z)` is the polarization factor
        - :math:`A` is the area in :math:`x-y` plane on which the atom
          is placed
        - :math:`M = 0.5(\mbox{deb_wal_fac} \ q_z)^2)` where
          :math:`\mbox{dbf}^2 = \\langle u^2\\rangle` is the average
          thermal vibration of the atoms - Debye-Waller factor

        """
        rho = (-4j*np.pi*r_0
               * atom.get_cm_atomic_form_factor(self._energy, self._qz)
               * self.get_polarization_factor(self._theta)
               * np.exp(-0.5*(deb_wal_fac*self._qz)**2))/(self._qz*area)
        return rho

    def get_atom_transmission_factor(self, atom, area, deb_wal_fac):
        """get_atom_transmission_factor

        Returns the transmission factor from dynamical xray theory:

        .. math:: \\tau = 1 - \\frac{i 4 \pi r_e f(E,0) \exp(-M)}{q_z A}

        - :math:`r_e` is the electron radius
        - :math:`f(E,0)` is the energy dispersive atomic form factor
          (no angle correction)
        - :math:`A` is the area in :math:`x-y` plane on which the atom
          is placed
        - :math:`M = 0.5(\mbox{dbf} \ q_z)^2` where
          :math:`\mbox{dbf}^2 = \\langle u^2\\rangle` is the average
          thermal vibration of the atoms - Debye-Waller factor

        """
        tau = 1 - (4j*np.pi*r_0
                   * atom.get_cm_atomic_form_factor(self._energy, np.zeros_like(self._qz))
                   * np.exp(-0.5*(deb_wal_fac*self._qz)**2))/(self._qz*area)
        return tau

    def get_atom_phase_matrix(self, distance):
        """get_atom_phase_matrix

        Returns the phase matrix from dynamical xray theory:

        .. math::

            L = \\begin{bmatrix}
            \exp(i \phi) & 0 \\\\
            0            & \exp(-i \phi)
            \end{bmatrix}

        """
        phi = self.get_atom_phase_factor(distance)
        L = np.zeros([np.shape(self._qz)[0], np.shape(self._qz)[1], 2, 2], dtype=np.cfloat)
        L[:, :, 0, 0] = np.exp(1j*phi)
        L[:, :, 1, 1] = np.exp(-1j*phi)
        return L

    def get_atom_phase_factor(self, distance):
        """get_atom_phase_factor

        Returns the phase factor :math:`\phi` for a distance :math:`d`
        from dynamical xray theory:

        .. math:: \phi = \\frac{d \ q_z}{2}

        """
        phi = distance * self._qz/2
        return phi

    @staticmethod
    def calc_reflectivity_from_matrix(M):
        """calc_reflectivity_from_matrix

        Returns the physical reflectivity from an 2x2 matrix of
        transmission and reflectifity factors:

        .. math:: R = \\left|M(0,1)/M(1,1)\\right|^2

        """
        return np.abs(M[:, :, 0, 1]/M[:, :, 1, 1])**2
