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

"""A :mod:`XrayDynMag` module """

__all__ = ["XrayDynMag"]

__docformat__ = "restructuredtext"

import numpy as np
import scipy.constants as constants
from time import time
from os import path
from .xray import Xray
from .layer import AmorphousLayer, UnitCell
from tqdm.notebook import trange
from .helpers import make_hash_md5, m_power_x, m_times_n

r_0 = constants.physical_constants['classical electron radius'][0]


class XrayDynMag(Xray):
    """XrayDynMag

    Dynamical magnetic Xray simulations adapted from Elzo et.al. [4]_.
    Initially realized in `Project Dyna
    <http://neel.cnrs.fr/spip.php?rubrique1008>`_.

    Original copyright notice:

    *Copyright Institut Neel, CNRS, Grenoble, France*

    **Project Collaborators:**

    - Stéphane Grenier, stephane.grenier@neel.cnrs.fr
    - Marta Elzo (PhD, 2009-2012)
    - Nicolas Jaouen Sextants beamline, Synchrotron Soleil,
      nicolas.jaouen@synchrotron-soleil.fr
    - Emmanuelle Jal (PhD, 2010-2013) now at `LCPMR CNRS, Paris
      <https://lcpmr.cnrs.fr/content/emmanuelle-jal>`_
    - Jean-Marc Tonnerre, jean-marc.tonnerre@neel.cnrs.fr
    - Ingrid Hallsteinsen - Padraic Shaffer’s group - Berkeley Nat. Lab.

    **Questions to:**

    - Stéphane Grenier, stephane.grenier@neel.cnrs.fr

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Attributes:
        last_atom_ref_trans_matrices (list): remember last result of
           atom ref_trans_matrices to speed up calculation

    References:

        .. [4] M. Elzo, E. Jal, O. Bunau, S. Grenier, Y. Joly, A. Y.
           Ramos, H. C. N. Tolentino, J. M. Tonnerre, and N. Jaouen,
           `J. Magn. Magn. Mater. 324, 105 (2012).
           <http://www.doi.org/10.1016/j.jmmm.2011.07.019>`_

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.last_atom_ref_trans_matrices = {'atom_ids': [],
                                             'hashes': [],
                                             'A': [],
                                             'A_phi': [],
                                             'P': [],
                                             'P_phi': [],
                                             'A_inv': [],
                                             'A_inv_phi': [],
                                             'k_z': []}

    def __str__(self):
        """String representation of this class"""
        class_str = 'Dynamical Magnetic X-Ray Diffraction simulation properties:\n\n'
        class_str += super().__str__()
        return class_str

    def get_hash(self, **kwargs):
        """get_hash

        Returns a unique hash given by the energy :math:`E`,
        :math:`q_z` range, polarization states and the strain vectors as
        well as the sample structure hash for relevant xray and magnetic
        parameters. Optionally, part of the strain_map and magnetization_map
        are used.

        """
        param = [self.pol_in_state, self.pol_out_state, self._qz, self._energy]

        if 'strain_map' in kwargs:
            strain_map = kwargs.get('strain_map')
            if np.size(strain_map) > 1e6:
                strain_map = strain_map.flatten()[0:1000000]
            param.append(strain_map)
        if 'magnetization_map' in kwargs:
            magnetization_map = kwargs.get('magnetization_map')
            if np.size(magnetization_map) > 1e6:
                magnetization_map = magnetization_map.flatten()[0:1000000]
            param.append(magnetization_map)

        return self.S.get_hash(types=['xray', 'magnetic']) + '_' + make_hash_md5(param)

    def set_incoming_polarization(self, pol_in_state):
        """set_incoming_polarization

        Sets the incoming polarization factor for circular +, circular -, sigma, pi,
        and unpolarized polarization.

        """

        self.pol_in_state = pol_in_state
        if (self.pol_in_state == 1):  # circ +
            self.pol_in = np.array([-np.sqrt(.5), -1j*np.sqrt(.5)], dtype=np.cfloat)
        elif (self.pol_in_state == 2):  # circ -
            self.pol_in = np.array([np.sqrt(.5), -1j*np.sqrt(.5)], dtype=np.cfloat)
        elif (self.pol_in_state == 3):  # sigma
            self.pol_in = np.array([1, 0], dtype=np.cfloat)
        elif (self.pol_in_state == 4):  # pi
            self.pol_in = np.array([0, 1], dtype=np.cfloat)
        else:  # unpolarized
            self.pol_in_state = 0  # catch any number and set state to 0
            self.pol_in = np.array([np.sqrt(.5), np.sqrt(.5)], dtype=np.cfloat)

        self.disp_message('incoming polarizations set to: {:s}'.format(
            self.polarizations[self.pol_in_state]))

    def set_outgoing_polarization(self, pol_out_state):
        """set_outgoing_polarization

        Sets the outgoing polarization factor for circular +, circular -, sigma, pi,
        and unpolarized polarization.

        """

        self.pol_out_state = pol_out_state
        if (self.pol_out_state == 1):  # circ +
            self.pol_out = np.array([-np.sqrt(.5), 1j*np.sqrt(.5)], dtype=np.cfloat)
        elif (self.pol_out_state == 2):  # circ -
            self.pol_out = np.array([np.sqrt(.5), 1j*np.sqrt(.5)], dtype=np.cfloat)
        elif (self.pol_out_state == 3):  # sigma
            self.pol_out = np.array([1, 0], dtype=np.cfloat)
        elif (self.pol_out_state == 4):  # pi
            self.pol_out = np.array([0, 1], dtype=np.cfloat)
        else:  # no analyzer
            self.pol_out_state = 0  # catch any number and set state to 0
            self.pol_out = np.array([], dtype=np.cfloat)

        self.disp_message('analyzer polarizations set to: {:s}'.format(
            self.polarizations[self.pol_out_state]))

    def homogeneous_reflectivity(self, *args):
        """homogeneous_reflectivity

        Returns the reflectivity :math:`R` of the whole sample structure
        allowing only no or homogeneous strain and magnetization.

        The reflection-transmission matrices

        .. math:: RT = A_f^{-1} \\prod_m \\left( A_m P_m A_m^{-1} \\right) A_0

        are calculated for every substructure :math:`m` before post-processing
        the incoming and analyzer polarizations and calculating the actual
        reflectivities as function of energy and :math:`q_z`.

        """
        # if no strains are given we assume no strain
        if len(args) == 0:
            strains = np.zeros([self.S.get_number_of_sub_structures(), 1])
        else:
            strains = args[0]

        if len(args) < 2:
            magnetizations = np.zeros([self.S.get_number_of_sub_structures(), 3])
        else:
            magnetizations = args[1]

        t1 = time()
        self.disp_message('Calculating _homogenous_reflectivity_ ...')
        # vacuum boundary
        A0, A0_phi, _, _, _, _, k_z_0 = self.get_atom_boundary_phase_matrix([], 0, 0)
        # calc the reflectivity-transmisson matrix of the structure
        # and the inverse of the last boundary matrix
        RT, RT_phi, last_A, last_A_phi, last_A_inv, last_A_inv_phi, last_k_z = \
            self.calc_homogeneous_matrix(self.S, A0, A0_phi, k_z_0, strains, magnetizations)
        # if a substrate is included add it at the end
        if self.S.substrate != []:
            RT_sub, RT_sub_phi, last_A, last_A_phi, last_A_inv, last_A_inv_phi, _ = \
                self.calc_homogeneous_matrix(
                    self.S.substrate, last_A, last_A_phi, last_k_z)
            RT = m_times_n(RT_sub, RT)
            RT_phi = m_times_n(RT_sub_phi, RT_phi)
        # multiply the result of the structure with the boundary matrix
        # of vacuum (initial layer) and the final layer
        RT = m_times_n(last_A_inv, m_times_n(last_A, RT))
        RT_phi = m_times_n(last_A_inv_phi, m_times_n(last_A_phi, RT_phi))
        # calc the actual reflectivity from the matrix
        R = XrayDynMag.calc_reflectivity_from_matrix(RT, self.pol_in, self.pol_out)
        R_phi = XrayDynMag.calc_reflectivity_from_matrix(RT_phi, self.pol_in, self.pol_out)
        self.disp_message('Elapsed time for _homogenous_reflectivity_: {:f} s'.format(time()-t1))
        return R, R_phi

    def calc_homogeneous_matrix(self, S, last_A, last_A_phi, last_k_z, *args):
        """calc_homogeneous_matrix

        Calculates the product of all reflection-transmission matrices of the
        sample structure

        .. math:: RT = \\prod_m \\left(P_m A_m^{-1} A_{m-1} \\right)

        If the sub-structure :math:`m` consists of :math:`N` unit cells
        the matrix exponential rule is applied:

        .. math:: RT_m = \\left( P_{UC} A_{UC}^{-1} A_{UC} \\right)^N

        Roughness is also included by a gaussian width

        """
        # if no strains are given we assume no strain (1)
        if len(args) == 0:
            strains = np.zeros([S.get_number_of_sub_structures(), 1])
        else:
            strains = args[0]

        if len(args) < 2:
            magnetizations = np.zeros([S.get_number_of_sub_structures(), 3])
        else:
            magnetizations = args[1]

        strainCounter = 0
        # traverse substructures
        for i, sub_structure in enumerate(S.sub_structures):
            layer = sub_structure[0]
            repetitions = sub_structure[1]
            if isinstance(layer, UnitCell):
                # the sub_structure is an unitCell
                # calculate the ref-trans matrices for N unitCells
                RT_uc, RT_uc_phi, A, A_phi, A_inv, A_inv_phi, k_z = \
                    self.calc_uc_boundary_phase_matrix(
                        layer, last_A, last_A_phi, last_k_z, strains[strainCounter],
                        magnetizations[strainCounter])
                temp = RT_uc
                temp_phi = RT_uc_phi
                if repetitions > 1:
                    # use m_power_x for more than one repetition
                    temp2, temp2_phi, A, A_phi, A_inv, A_inv_phi, k_z = \
                        self.calc_uc_boundary_phase_matrix(
                            layer, A, A_phi, k_z, strains[strainCounter],
                            magnetizations[strainCounter])
                    temp2 = m_power_x(temp2, repetitions-1)
                    temp2_phi = m_power_x(temp2_phi, repetitions-1)
                    temp = m_times_n(temp2, temp)
                    temp_phi = m_times_n(temp2_phi, temp_phi)

                strainCounter += 1
            elif isinstance(layer, AmorphousLayer):
                # the sub_structure is an amorphous layer
                # calculate the ref-trans matrices for N layers

                A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = \
                    self.get_atom_boundary_phase_matrix(layer.atom,
                                                        layer._density,
                                                        layer._thickness*(
                                                            strains[strainCounter]+1),
                                                        magnetizations[strainCounter])

                roughness = layer._roughness
                F = m_times_n(A_inv, last_A)
                F_phi = m_times_n(A_inv_phi, last_A_phi)
                if roughness > 0:
                    W = XrayDynMag.calc_roughness_matrix(roughness, k_z, last_k_z)
                    F = F * W
                    F_phi = F_phi * W

                RT_amorph = m_times_n(P, F)
                RT_amorph_phi = m_times_n(P_phi, F_phi)
                temp = RT_amorph
                temp_phi = RT_amorph_phi
                if repetitions > 1:
                    # use m_power_x for more than one repetition
                    F = m_times_n(A_inv, A)
                    F_phi = m_times_n(A_inv_phi, A_phi)
                    RT_amorph = m_times_n(P, F)
                    RT_amorph_phi = m_times_n(P_phi, F_phi)
                    temp = m_times_n(m_power_x(RT_amorph, repetitions-1), temp)
                    temp_phi = m_times_n(m_power_x(RT_amorph_phi, repetitions-1), temp_phi)
                strainCounter += 1
            else:
                # its a structure
                # make a recursive call
                temp, temp_phi, A, A_phi, A_inv, A_inv_phi, k_z = self.calc_homogeneous_matrix(
                        layer, last_A, last_A_phi, last_k_z,
                        strains[strainCounter:(
                            strainCounter
                            + layer.get_number_of_sub_structures()
                            )],
                        magnetizations[strainCounter:(
                            strainCounter
                            + layer.get_number_of_sub_structures()
                            )])
                # calculate the ref-trans matrices for N sub structures
                if repetitions > 1:
                    # use m_power_x for more than one repetition
                    temp2, temp2_phi, A, A_phi, A_inv, A_inv_phi, k_z = \
                        self.calc_homogeneous_matrix(
                            layer, A, A_phi, k_z,
                            strains[strainCounter:(strainCounter
                                                   + layer.get_number_of_sub_structures())],
                            magnetizations[strainCounter:(strainCounter
                                                          + layer.get_number_of_sub_structures())])

                    temp = m_times_n(m_power_x(temp2, repetitions-1), temp)
                    temp_phi = m_times_n(m_power_x(temp2_phi, repetitions-1), temp_phi)

                strainCounter = strainCounter+layer.get_number_of_sub_structures()

            # multiply it to the output
            if i == 0:
                RT = temp
                RT_phi = temp_phi
            else:
                RT = m_times_n(temp, RT)
                RT_phi = m_times_n(temp_phi, RT_phi)

            # update the last A and k_z
            last_A = A
            last_A_phi = A_phi
            last_k_z = k_z

        return RT, RT_phi, A, A_phi, A_inv, A_inv_phi, k_z

    """
    Inhomogenous Sample Structure
    All unit cells in the sample are inhomogeneously
    strained. This is generally the case when calculating the
    transient rocking curves for coherent phonon dynamics in the
    sample structure.
    """

    def inhomogeneous_reflectivity(self, strain_map, magnetization_map, **kwargs):
        """inhomogeneous_reflectivity

        Returns the reflectivity of an inhomogenously strained and
        magnetized sample structure for a given _strain_map_ and
        _magnetization_map_ in position and time for each unit cell or
        amorphous layer in the sample structure.
        If no reflectivity is saved in the cache it is caluclated.
        Providing the _type_ (parallel [default], sequential,
        distributed) for the calculation the corresponding subroutines
        for the reflectivity computation are called:

        * ``parallel`` parallelization over the time steps utilizing
          Dask
        * ``distributed`` not yet implemented
        * ``sequential`` no parallelization at all

        """
        # create a hash of all simulation parameters
        filename = 'inhomogeneous_reflectivity_dynMag_' \
                   + self.get_hash(strain_map=strain_map, magnetization_map=magnetization_map) \
                   + '.npy'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            R, R_phi = np.load(full_filename)
            self.disp_message('_inhomogeneous_reflectivity_ loaded from file:\n\t' + filename)
        else:
            t1 = time()
            self.disp_message('Calculating _inhomogenousReflectivity_ ...')
            # parse the input arguments
            if not isinstance(strain_map, np.ndarray):
                raise TypeError('strain_map must be a numpy ndarray!')
            if not isinstance(magnetization_map, np.ndarray):
                raise TypeError('magnetization_map must be a numpy ndarray!')

            dask_client = kwargs.get('dask_client', [])
            calc_type = kwargs.get('calc_type', 'sequential')
            if calc_type not in ['parallel', 'sequential', 'distributed']:
                raise TypeError('calc_type must be either _parallel_, '
                                '_sequential_, or _distributed_!')
            job = kwargs.get('job')
            num_workers = kwargs.get('num_workers', 1)

            # select the type of computation
            if calc_type == 'parallel':
                R, R_phi = self.parallel_inhomogeneous_reflectivity(strain_map,
                                                                    magnetization_map,
                                                                    dask_client)
            elif calc_type == 'distributed':
                R, R_Phi = self.distributed_inhomogeneous_reflectivity(strain_map,
                                                                       magnetization_map,
                                                                       job,
                                                                       num_workers)
            else:  # sequential
                R, R_phi = self.sequential_inhomogeneous_reflectivity(strain_map,
                                                                      magnetization_map)

            self.disp_message('Elapsed time for _inhomogenous_reflectivity_:'
                              ' {:f} s'.format(time()-t1))
            self.save(full_filename, [R, R_phi], '_inhomogeneous_reflectivity_')
        return R, R_phi

    def sequential_inhomogeneous_reflectivity(self, strain_map, magnetization_map):
        """sequential_inhomogeneous_reflectivity

        Returns the reflectivity of an inhomogenously strained sample
        structure for a given ``strain_map`` in position and time, as
        well as for a given set of possible strains for each unit cell
        in the sample structure (``strain_vectors``).
        The function calculates the results sequentially without
        parallelization for every atomic layer.

        """
        # initialize
        N = np.size(strain_map, 0)  # delay steps
        R = np.zeros([N, np.size(self._qz, 0), np.size(self._qz, 1)])
        R_phi = np.zeros_like(R)

        for i in trange(N, desc='Progress', leave=True):
            # get the inhomogenous reflectivity of the sample
            # structure for each time step of the strain map

            # vacuum boundary
            A0, A0_phi, _, _, _, _, k_z_0 = self.get_atom_boundary_phase_matrix([], 0, 0)

            RT, RT_phi, last_A, last_A_phi, last_A_inv, last_A_inv_phi, last_k_z = \
                self.calc_inhomogeneous_matrix(
                    A0, A0_phi, k_z_0, strain_map[i, :], magnetization_map[i, :])
            # if a substrate is included add it at the end
            if self.S.substrate != []:
                RT_sub, RT_sub_phi, last_A, last_A_phi, last_A_inv, last_A_inv_phi, _ = \
                    self.calc_homogeneous_matrix(
                        self.S.substrate, last_A, last_A_phi, last_k_z)
                RT = m_times_n(RT_sub, RT)
                RT_phi = m_times_n(RT_sub_phi, RT_phi)
            # multiply vacuum and last layer
            RT = m_times_n(last_A_inv, m_times_n(last_A, RT))
            RT_phi = m_times_n(last_A_inv_phi, m_times_n(last_A_phi, RT_phi))

            R[i, :, :] = XrayDynMag.calc_reflectivity_from_matrix(
                RT, self.pol_in, self.pol_out)
            R_phi[i, :, :] = XrayDynMag.calc_reflectivity_from_matrix(
                RT_phi, self.pol_in, self.pol_out)

        return R, R_phi

    def parallel_inhomogeneous_reflectivity(self, strain_map, magnetization_map, dask_client):
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
        R_phi = np.zeros_like(R)
        # vacuum boundary
        A0, A0_phi, _, _,  _, _, k_z_0 = self.get_atom_boundary_phase_matrix([], 0, 0)
        remote_A0 = dask_client.scatter(A0)
        remote_A0_phi = dask_client.scatter(A0_phi)
        remote_k_z_0 = dask_client.scatter(k_z_0)
        remote_pol_in = dask_client.scatter(self.pol_in)
        remote_pol_out = dask_client.scatter(self.pol_out)
        remote_substrate = dask_client.scatter(self.S.substrate)

        # create dask.delayed tasks for all delay steps
        for i in range(M):
            t = delayed(self.calc_inhomogeneous_matrix)(remote_A0,
                                                        remote_A0_phi,
                                                        remote_k_z_0,
                                                        strain_map[i, :],
                                                        magnetization_map[i, :])

            RT = t[0]
            RT_phi = t[1]
            last_A = t[2]
            last_A_phi = t[3]
            last_A_inv = t[4]
            last_A_inv_phi = t[5]
            last_k_z = t[6]
            if remote_substrate != []:
                t2 = delayed(self.calc_homogeneous_matrix)(
                    remote_substrate, last_A, last_A_phi, last_k_z)
                RT_sub = t2[0]
                RT_sub_phi = t2[1]
                last_A = t2[2]
                last_A_phi = t2[3]
                last_A_inv = t2[4]
                last_A_inv_phi = t2[5]
                RT = delayed(m_times_n)(RT_sub, RT)
                RT_phi = delayed(m_times_n)(RT_sub_phi, RT_phi)
            # multiply vacuum and last layer
            temp = delayed(m_times_n)(last_A, RT)
            temp_phi = delayed(m_times_n)(last_A_phi, RT_phi)
            RT = delayed(m_times_n)(last_A_inv, temp)
            RT_phi = delayed(m_times_n)(last_A_inv_phi, temp_phi)
            Ri = delayed(XrayDynMag.calc_reflectivity_from_matrix)(RT,
                                                                   remote_pol_in,
                                                                   remote_pol_out)
            Ri_phi = delayed(XrayDynMag.calc_reflectivity_from_matrix)(RT_phi,
                                                                       remote_pol_in,
                                                                       remote_pol_out)
            res.append(Ri)
            res.append(Ri_phi)

        # compute results
        res = dask_client.compute(res, sync=True)

        # reorder results to reflectivity matrix
        for i in range(M):
            R[i, :, :] = res[2*i]
            R_phi[i, :, :] = res[2*i + 1]

        return R, R_phi

    def distributed_inhomogeneous_reflectivity(self, job, num_worker,
                                               strain_map, magnetization_map):
        """distributed_inhomogeneous_reflectivity

        This is a stub. Not yet implemented in python.

        """
        return

    def calc_inhomogeneous_matrix(self, last_A, last_A_phi, last_k_z, strains, magnetizations):
        """calc_inhomogeneous_matrix

        Calculates the product of all reflection-transmission matrices of the
        sample structure for every atomic layer.

        .. math:: RT = \\prod_m \\left( P_m A_m^{-1} A_{m-1}  \\right)

        """
        L = self.S.get_number_of_layers()  # number of unit cells
        _, _, layer_handles = self.S.get_layer_vectors()

        for i in range(L):
            layer = layer_handles[i]
            if isinstance(layer, UnitCell):
                RT_layer, RT_layer_phi, A, A_phi, A_inv, A_inv_phi, k_z = \
                    self.calc_uc_boundary_phase_matrix(
                        layer, last_A, last_A_phi, last_k_z, strains[i], magnetizations[i])
            elif isinstance(layer, AmorphousLayer):
                A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = \
                    self.get_atom_boundary_phase_matrix(
                        layer.atom, layer._density, layer._thickness*(strains[i]+1),
                        magnetizations[i])
                roughness = layer._roughness
                F = m_times_n(A_inv, last_A)
                F_phi = m_times_n(A_inv_phi, last_A_phi)
                if roughness > 0:
                    W = XrayDynMag.calc_roughness_matrix(roughness, k_z, last_k_z)
                    F = F * W
                    F_phi = F_phi * W
                RT_layer = m_times_n(P, F)
                RT_layer_phi = m_times_n(P_phi, F_phi)
            else:
                raise ValueError('All layers must be either AmorphousLayers or UnitCells!')
            if i == 0:
                RT = RT_layer
                RT_phi = RT_layer_phi
            else:
                RT = m_times_n(RT_layer, RT)
                RT_phi = m_times_n(RT_layer_phi, RT_phi)

            # update the last A and k_z
            last_A = A
            last_A_phi = A_phi
            last_k_z = k_z

        return RT, RT_phi, A, A_phi, A_inv, A_inv_phi, k_z

    def calc_uc_boundary_phase_matrix(self, uc, last_A, last_A_phi, last_k_z, strain,
                                      magnetization):
        """calc_uc_boundary_phase_matrix

        Calculates the product of all reflection-transmission matrices of
        a single unit cell for a given strain:

        .. math:: RT = \\prod_m \\left( P_m A_m^{-1} A_{m-1}\\right)

        and returns also the last matrices :math:`A, A^{-1}, k_z`.

        """
        K = uc.num_atoms  # number of atoms
        for j in range(K):
            if j == (K-1):  # its the last atom
                del_dist = (strain+1)-uc.atoms[j][1](strain)
            else:
                del_dist = uc.atoms[j+1][1](strain)-uc.atoms[j][1](strain)
            distance = del_dist*uc._c_axis

            try:
                # calculate density
                if distance == 0:
                    density = 0
                else:
                    density = uc.atoms[j][0]._mass/(uc._area*distance)
            except AttributeError:
                density = 0

            A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = \
                self.get_atom_boundary_phase_matrix(uc.atoms[j][0], density, distance,
                                                    magnetization)
            F = m_times_n(A_inv, last_A)
            F_phi = m_times_n(A_inv_phi, last_A_phi)
            if (j == 0) and (uc._roughness > 0):
                # it is the first layer so care for the roughness
                W = XrayDynMag.calc_roughness_matrix(uc._roughness, k_z, last_k_z)
                F = F * W
                F_phi = F_phi * W
            temp = m_times_n(P, F)
            temp_phi = m_times_n(P_phi, F_phi)
            if j == 0:
                RT = temp
                RT_phi = temp_phi
            else:
                RT = m_times_n(temp, RT)
                RT_phi = m_times_n(temp_phi, RT_phi)

            # update last A and k_z
            last_A = A
            last_A_phi = A_phi
            last_k_z = k_z

        return RT, RT_phi, A, A_phi, A_inv, A_inv_phi, k_z

    def get_atom_boundary_phase_matrix(self, atom, density, distance, *args):
        """get_atom_boundary_phase_matrix

        Returns the boundary and phase matrices of an atom from
        Elzo formalism. The results for a given atom, energy, :math:`q_z`,
        polarization, and magnetization are stored to RAM to avoid
        recalculation.

        """
        try:
            index = self.last_atom_ref_trans_matrices['atom_ids'].index(atom.id)
        except ValueError:
            index = -1
        except AttributeError:
            # its vacuum
            A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = \
                self.calc_atom_boundary_phase_matrix(atom, density, distance, *args)
            return A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z

        # check for already calculated data
        _hash = make_hash_md5([self._energy, self._qz, self.pol_in, self.pol_out,
                               density, distance,
                               atom.mag_amplitude,
                               atom.mag_gamma,
                               atom.mag_phi,
                               *args])

        if (index >= 0) and (_hash == self.last_atom_ref_trans_matrices['hashes'][index]):
            # These are the same X-ray parameters as last time so we
            # can use the same matrix again for this atom
            A = self.last_atom_ref_trans_matrices['A'][index]
            A_phi = self.last_atom_ref_trans_matrices['A_phi'][index]
            P = self.last_atom_ref_trans_matrices['P'][index]
            P_phi = self.last_atom_ref_trans_matrices['P_phi'][index]
            A_inv = self.last_atom_ref_trans_matrices['A_inv'][index]
            A_inv_phi = self.last_atom_ref_trans_matrices['A_inv_phi'][index]
            k_z = self.last_atom_ref_trans_matrices['k_z'][index]
        else:
            # These are new parameters so we have to calculate.
            # Get the reflection-transmission-factors
            A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = \
                self.calc_atom_boundary_phase_matrix(atom, density, distance, *args)
            # remember this matrix for next use with the same
            # parameters for this atom
            if index >= 0:
                self.last_atom_ref_trans_matrices['atom_ids'][index] = atom.id
                self.last_atom_ref_trans_matrices['hashes'][index] = _hash
                self.last_atom_ref_trans_matrices['A'][index] = A
                self.last_atom_ref_trans_matrices['A_phi'][index] = A_phi
                self.last_atom_ref_trans_matrices['P'][index] = P
                self.last_atom_ref_trans_matrices['P_phi'][index] = P_phi
                self.last_atom_ref_trans_matrices['A_inv'][index] = A_inv
                self.last_atom_ref_trans_matrices['A_inv_phi'][index] = A_inv_phi
                self.last_atom_ref_trans_matrices['k_z'][index] = k_z
            else:
                self.last_atom_ref_trans_matrices['atom_ids'].append(atom.id)
                self.last_atom_ref_trans_matrices['hashes'].append(_hash)
                self.last_atom_ref_trans_matrices['A'].append(A)
                self.last_atom_ref_trans_matrices['A_phi'].append(A_phi)
                self.last_atom_ref_trans_matrices['P'].append(P)
                self.last_atom_ref_trans_matrices['P_phi'].append(P_phi)
                self.last_atom_ref_trans_matrices['A_inv'].append(A_inv)
                self.last_atom_ref_trans_matrices['A_inv_phi'].append(A_inv_phi)
                self.last_atom_ref_trans_matrices['k_z'].append(k_z)
        return A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z

    def calc_atom_boundary_phase_matrix(self, atom, density, distance, *args):
        """calc_atom_boundary_phase_matrix

        Calculates the boundary and phase matrices of an atom from
        Elzo formalism.

        """

        if len(args) > 0:
            magnetization = args[0]
            mag_amplitude = magnetization[0]
            mag_phi = magnetization[1]
            mag_gamma = magnetization[2]
        else:
            try:
                mag_amplitude = atom.mag_amplitude
            except AttributeError:
                mag_amplitude = 0
            try:
                mag_phi = atom.mag_phi.to_base_units().magnitude
            except AttributeError:
                mag_phi = 0
            try:
                mag_gamma = atom.mag_gamma.to_base_units().magnitude
            except AttributeError:
                mag_gamma = 0
        
        M = len(self._energy)  # number of energies
        N = np.shape(self._qz)[1]  # number of q_z

        U = [np.sin(mag_phi) *
             np.cos(mag_gamma),
             np.sin(mag_phi) *
             np.sin(mag_gamma),
             np.cos(mag_phi)]

        eps = np.zeros([M, N, 3, 3], dtype=np.cfloat)
        A = np.zeros([M, N, 4, 4], dtype=np.cfloat)
        A_phi = np.zeros_like(A, dtype=np.cfloat)
        P = np.zeros_like(A, dtype=np.cfloat)
        P_phi = np.zeros_like(A, dtype=np.cfloat)

        try:
            molar_density = density/1000/atom.mass_number_a
        except AttributeError:
            molar_density = 0

        energy = self._energy
        factor = 830.9471/energy**2
        theta = self._theta

        try:
            cf = atom.get_atomic_form_factor(energy)
        except AttributeError:
            cf = np.zeros_like(energy, dtype=np.cfloat)
        try:
            mf = atom.get_magnetic_form_factor(energy)
        except AttributeError:
            mf = np.zeros_like(energy, dtype=np.cfloat)

        mag = factor * molar_density * mag_amplitude * mf
        mag = np.tile(mag[:, np.newaxis], [1, N])
        eps0 = 1 - factor*molar_density*cf
        eps0 = np.tile(eps0[:, np.newaxis], [1, N])

        eps[:, :, 0, 0] = eps0
        eps[:, :, 0, 1] = -1j * U[2] * mag
        eps[:, :, 0, 2] = 1j * U[1] * mag
        eps[:, :, 1, 0] = -eps[:, :, 0, 1]
        eps[:, :, 1, 1] = eps0
        eps[:, :, 1, 2] = -1j * U[0] * mag
        eps[:, :, 2, 0] = -eps[:, :, 0, 2]
        eps[:, :, 2, 1] = -eps[:, :, 1, 2]
        eps[:, :, 2, 2] = eps0

        alpha_y = np.divide(np.cos(theta), np.sqrt(eps[:, :, 0, 0]))
        alpha_z = np.sqrt(1 - alpha_y**2)
        # reshape self._k for elementwise multiplication
        k = np.reshape(np.repeat(self._k, N), (M, N))
        k_z = k * (np.sqrt(eps[:, :, 0, 0]) * alpha_z)

        n_right_down = np.sqrt(eps[:, :, 0, 0] - 1j * eps[:, :, 0, 2] * alpha_y
                               - 1j * eps[:, :, 0, 1] * alpha_z)
        n_left_down = np.sqrt(eps[:, :, 0, 0] + 1j * eps[:, :, 0, 2] * alpha_y
                              + 1j * eps[:, :, 0, 1] * alpha_z)
        n_right_up = np.sqrt(eps[:, :, 0, 0] - 1j * eps[:, :, 0, 2] * alpha_y
                             + 1j * eps[:, :, 0, 1] * alpha_z)
        n_left_up = np.sqrt(eps[:, :, 0, 0] + 1j * eps[:, :, 0, 2] * alpha_y
                            - 1j * eps[:, :, 0, 1] * alpha_z)

        alpha_y_right_down = np.cos(theta)/n_right_down
        alpha_z_right_down = np.sqrt(1-alpha_y_right_down**2)
        alpha_y_left_down = np.cos(theta)/n_left_down
        alpha_z_left_down = np.sqrt(1-alpha_y_left_down**2)
        alpha_y_right_up = np.cos(theta)/n_right_up
        alpha_z_right_up = np.sqrt(1-alpha_y_right_up**2)
        alpha_y_left_up = np.cos(theta)/n_left_up
        alpha_z_left_up = np.sqrt(1-alpha_y_left_up**2)

        A[:, :, 0, 0] = (-1 - 1j * eps[:, :, 0, 1] * alpha_z_right_down
                         - 1j * eps[:, :, 0, 2] * alpha_y_right_down)
        A[:, :, 0, 1] = (1 - 1j * eps[:, :, 0, 1] * alpha_z_left_down
                         - 1j * eps[:, :, 0, 2] * alpha_y_left_down)
        A[:, :, 0, 2] = (-1 + 1j * eps[:, :, 0, 1] * alpha_z_right_up
                         - 1j * eps[:, :, 0, 2] * alpha_y_right_up)
        A[:, :, 0, 3] = (1 + 1j * eps[:, :, 0, 1] * alpha_z_left_up
                         - 1j * eps[:, :, 0, 2] * alpha_y_left_up)

        A[:, :, 1, 0] = (1j * alpha_z_right_down - eps[:, :, 0, 1]
                         - 1j * eps[:, :, 1, 2] * alpha_y_right_down)
        A[:, :, 1, 1] = (1j * alpha_z_left_down + eps[:, :, 0, 1]
                         - 1j * eps[:, :, 1, 2] * alpha_y_left_down)
        A[:, :, 1, 2] = (-1j * alpha_z_right_up - eps[:, :, 0, 1]
                         - 1j * eps[:, :, 1, 2] * alpha_y_right_up)
        A[:, :, 1, 3] = (-1j * alpha_z_left_up + eps[:, :, 0, 1]
                         - 1j * eps[:, :, 1, 2] * alpha_y_left_up)

        A[:, :, 2, 0] = -1j * n_right_down * A[:, :, 0, 0]
        A[:, :, 2, 1] = 1j * n_left_down * A[:, :, 0, 1]
        A[:, :, 2, 2] = -1j * n_right_up * A[:, :, 0, 2]
        A[:, :, 2, 3] = 1j * n_left_up * A[:, :, 0, 3]

        A[:, :, 3, 0] = - alpha_z_right_down * n_right_down * A[:, :, 0, 0]
        A[:, :, 3, 1] = - alpha_z_left_down * n_left_down * A[:, :, 0, 1]
        A[:, :, 3, 2] = alpha_z_right_up * n_right_up * A[:, :, 0, 2]
        A[:, :, 3, 3] = alpha_z_left_up * n_left_up * A[:, :, 0, 3]

        A_phi[:, :, 0, 0] = (-1 + 1j * eps[:, :, 0, 1] * alpha_z_left_down
                             + 1j * eps[:, :, 0, 2] * alpha_y_left_down)
        A_phi[:, :, 0, 1] = (1 + 1j * eps[:, :, 0, 1] * alpha_z_right_down
                             + 1j * eps[:, :, 0, 2] * alpha_y_right_down)
        A_phi[:, :, 0, 2] = (-1 - 1j * eps[:, :, 0, 1] * alpha_z_left_up
                             + 1j * eps[:, :, 0, 2] * alpha_y_left_up)
        A_phi[:, :, 0, 3] = (1 - 1j * eps[:, :, 0, 1] * alpha_z_right_up
                             + 1j * eps[:, :, 0, 2] * alpha_y_right_up)

        A_phi[:, :, 1, 0] = (1j * alpha_z_left_down + eps[:, :, 0, 1]
                             + 1j * eps[:, :, 1, 2] * alpha_y_left_down)
        A_phi[:, :, 1, 1] = (1j * alpha_z_right_down - eps[:, :, 0, 1]
                             + 1j * eps[:, :, 1, 2] * alpha_y_right_down)
        A_phi[:, :, 1, 2] = (-1j * alpha_z_left_up + eps[:, :, 0, 1]
                             + 1j * eps[:, :, 1, 2] * alpha_y_left_up)
        A_phi[:, :, 1, 3] = (-1j * alpha_z_right_up - eps[:, :, 0, 1]
                             + 1j * eps[:, :, 1, 2] * alpha_y_right_up)

        A_phi[:, :, 2, 0] = 1j * n_left_down * A_phi[:, :, 0, 0]
        A_phi[:, :, 2, 1] = -1j * n_right_down * A_phi[:, :, 0, 1]
        A_phi[:, :, 2, 2] = 1j * n_left_up * A_phi[:, :, 0, 2]
        A_phi[:, :, 2, 3] = -1j * n_right_up * A_phi[:, :, 0, 3]

        A_phi[:, :, 3, 0] = - alpha_z_left_down * n_left_down * A_phi[:, :, 0, 0]
        A_phi[:, :, 3, 1] = - alpha_z_right_down * n_right_down * A_phi[:, :, 0, 1]
        A_phi[:, :, 3, 2] = alpha_z_left_up * n_right_up * A_phi[:, :, 0, 2]
        A_phi[:, :, 3, 3] = alpha_z_right_up * n_right_up * A_phi[:, :, 0, 3]

        A[:, :, :, :] = np.divide(
            A[:, :, :, :],
            np.sqrt(2) * eps[:, :, 0, 0][:, :, np.newaxis, np.newaxis])

        A_phi[:, :, :, :] = np.divide(
            A_phi[:, :, :, :],
            np.sqrt(2) * eps[:, :, 0, 0][:, :, np.newaxis, np.newaxis])

        A_inv = np.linalg.inv(A)
        A_inv_phi = np.linalg.inv(A_phi)

        phase = self._k * distance
        phase = phase[:, np.newaxis]

        P[:, :, 0, 0] = np.exp(1j * phase * n_right_down * alpha_z_right_down)
        P[:, :, 1, 1] = np.exp(1j * phase * n_left_down * alpha_z_left_down)
        P[:, :, 2, 2] = np.exp(-1j * phase * n_right_up * alpha_z_right_up)
        P[:, :, 3, 3] = np.exp(-1j * phase * n_left_up * alpha_z_left_up)

        P_phi[:, :, 0, 0] = P[:, :, 1, 1]
        P_phi[:, :, 1, 1] = P[:, :, 0, 0]
        P_phi[:, :, 2, 2] = P[:, :, 3, 3]
        P_phi[:, :, 3, 3] = P[:, :, 2, 2]

        return A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z

    @staticmethod
    def calc_reflectivity_from_matrix(RT, pol_in, pol_out):
        """calc_reflectivity_from_matrix

        Calculates the actual reflectivity from the reflectivity-transmission
        matrix for a given incoming and analyzer polarization.

        """

        Ref = np.tile(np.eye(2, 2, dtype=np.cfloat)[np.newaxis, np.newaxis, :, :],
                      (np.size(RT, 0), np.size(RT, 1), 1, 1))

        d = np.divide(1, RT[:, :, 3, 3] * RT[:, :, 2, 2] - RT[:, :, 3, 2] * RT[:, :, 2, 3])
        Ref[:, :, 0, 0] = (-RT[:, :, 3, 3] * RT[:, :, 2, 0] + RT[:, :, 2, 3] * RT[:, :, 3, 0]) * d
        Ref[:, :, 0, 1] = (-RT[:, :, 3, 3] * RT[:, :, 2, 1] + RT[:, :, 2, 3] * RT[:, :, 3, 1]) * d
        Ref[:, :, 1, 0] = (RT[:, :, 3, 2] * RT[:, :, 2, 0] - RT[:, :, 2, 2] * RT[:, :, 3, 0]) * d
        Ref[:, :, 1, 1] = (RT[:, :, 3, 2] * RT[:, :, 2, 1] - RT[:, :, 2, 2] * RT[:, :, 3, 1]) * d

        Ref = np.matmul(np.matmul(np.array([[-1, 1], [-1j, -1j]]), Ref),
                        np.array([[-1, 1j], [1, 1j]])*0.5)

        if pol_out.size == 0:
            # no analyzer polarization
            X = np.matmul(Ref, pol_in)
            R = np.real(np.matmul(np.square(np.absolute(X)), np.array([1, 1], dtype=np.cfloat)))
        else:
            X = np.matmul(np.matmul(Ref, pol_in), pol_out)
            R = np.real(np.square(np.absolute(X)))

        return R

    @staticmethod
    def calc_roughness_matrix(roughness, k_z, last_k_z):
        """calc_roughness_matrix

        Calculates the roughness matrix for an interface with a gaussian
        roughness for the Elzo formalism.

        """
        W = np.zeros([k_z.shape[0], k_z.shape[1], 4, 4], dtype=np.cfloat)
        rugosp = np.exp(-((k_z + last_k_z)**2) * roughness**2 / 2)
        rugosn = np.exp(-((-k_z + last_k_z)**2) * roughness**2 / 2)
        W[:, :, 0, 0] = rugosn
        W[:, :, 0, 1] = rugosn
        W[:, :, 0, 2] = rugosp
        W[:, :, 0, 3] = rugosp
        W[:, :, 1, 0] = rugosn
        W[:, :, 1, 1] = rugosn
        W[:, :, 1, 2] = rugosp
        W[:, :, 1, 3] = rugosp
        W[:, :, 2, 0] = rugosp
        W[:, :, 2, 1] = rugosp
        W[:, :, 2, 2] = rugosn
        W[:, :, 2, 3] = rugosn
        W[:, :, 3, 0] = rugosp
        W[:, :, 3, 1] = rugosp
        W[:, :, 3, 2] = rugosn
        W[:, :, 3, 3] = rugosn

        return W
