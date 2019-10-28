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
from .helpers import make_hash_md5, m_power_x, m_times_n

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
        self.last_atom_ref_trans_matrices = []

    def __str__(self):
        """String representation of this class"""
        class_str = 'Dynamical X-Ray Diffraction simulation properties:\n\n'
        class_str += super().__str__()
        return class_str

    def get_hash(self, strain_vectors, *args):
        """get_hash

        Returns a unique hash given by the energy :math:`E`,
        :math:`q_z` range, polarization factor and the strain vectors as
        well as the sample structure hash for relevant xray parameters.

        """
        param = [self.energy, self.qz, self.polarization, strain_vectors]
        if args:
            strain_map = args[0]
            # reduce size of strainMap when it has more than 1e6 elements
            if np.size(strain_map) > 1e6:
                strain_map = strain_map.flatten()[0:1000000]
            param.append(strain_map)

        return self.S.get_hash(types='xray') + '_' + make_hash_md5(param)

    def homogeneous_reflectivity(self, *args):
        """homogeneous_reflectivity

        Returns the reflectivity :math:`R` of the whole sample structure
        and the reflectivity-transmission matrices :math:`M_{RT}` for
        each substructure. The reflectivity of the :math:`2\times 2`
        matrices for each :math:`q_z` is calculates as follow:

        .. math:: R = \left|M_{RT}^t(1,2)/M_{RT}^t(2,2)\\right|^2

        """
        # if no strains are given we assume no strain
        if len(args) == 0:
            strains = np.zeros([self.S.get_number_of_sub_structures(), 1])
        else:
            strains = args[0]
        t1 = time()
        self.disp_message('Calculating _homogenous_reflectivity_ ...')
        R = np.zeros_like(self._qz)
        for i, energy in enumerate(self._energy):
            qz = self._qz[i, :]
            theta = self._theta[i, :]
            # get the reflectivity-transmisson matrix of the structure
            RT, A = self.homogeneous_ref_trans_matrix(self.S, energy, qz, theta, strains)
            # calculate the real reflectivity from the RT matrix
            R[i, :] = self.get_reflectivity_from_matrix(RT)
        self.disp_message('Elapsed time for _homogenous_reflectivity_: {:f} s'.format(time()-t1))
        return R, A

    def homogeneous_ref_trans_matrix(self, S, energy, qz, theta, *args):
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
        N = len(qz)
        RT = np.tile(np.eye(2, 2)[:, :, np.newaxis], (1, 1, N))  # ref_trans_matrix
        A = []  # list of ref_trans_matrices of substructures
        strainCounter = 0

        # traverse substructures
        for i, sub_structure in enumerate(S.sub_structures):
            if isinstance(sub_structure[0], UnitCell):
                # the sub_structure is an unitCell
                # calculate the ref-trans matrices for N unitCells
                temp = m_power_x(self.get_uc_ref_trans_matrix(
                        sub_structure[0], energy, qz, theta, strains[strainCounter]),
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
            RT = m_times_x(RT, temp)

        return RT, A

    def get_uc_ref_trans_matrix(self, uc, energy, qz, theta, *args):
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

        N = len(qz)  # number of q_z
        M = uc.num_atoms  # number of atoms
        # initialize matrices
        RTM = np.tile(np.eye(2, 2)[:, :, np.newaxis], (1, 1, N))
        # traverse all atoms of the unit cell
        for i in range(M):
            # Calculate the relative distance between the atoms.
            # The raltive position is calculated by the function handle
            # stored in the atoms list as 3rd element. This
            # function returns a relative postion dependent on the
            # applied strain.
            if i == (M-1):  # its the last atom
                del_dist = (strain+1)-uc.atoms[i][1](strain)
            else:
                del_dist = uc.atoms[i+1][1](strain)-uc.atoms[i][1](strain)

            # get the reflection-transmission matrix and phase matrix
            # from all atoms in the unit cell and multiply them
            # together
            RTM = m_times_n(RTM,
                            self.get_atom_ref_trans_matrix(uc.atoms[i][0],
                                                           energy,
                                                           qz,
                                                           theta,
                                                           uc._area,
                                                           uc._deb_wal_fac))
            RTM = m_times_n(RTM,
                            self.get_atom_phase_matrix(qz,
                                                       del_dist*uc._c_axis))
        return RTM

    def get_atom_ref_trans_matrix(self, atom, energy, qz, theta, area, deb_wal_fac):
        """get_atom_ref_trans_matrix

        Returns the reflection-transmission matrix of an atom from
        dynamicla xray theory:

        .. math::

            H = \\frac{1}{\\tau} \\begin{bmatrix}
            \left(\\tau^2 - \\rho^2\\right) & \\rho \\\\
            -\\rho & 1
            \\end{bmatrix}

        """
        # check for already calculated data
#        _hash = make_hash_md5([energy, qz, self.polarization, area, dbf])
#        index = find(strcmp(self.last_atom_ref_trans_matrices, atom.ID))
#        if not index and strcmp(_hash, self.last_atom_ref_trans_matrices[index][1]):
#            # These are the same X-ray parameters as last time so we
#            # can use the same matrix again for this atom
#            H = self.last_atom_ref_trans_matrices[index][2]
#        else:
        # These are new parameters so we have to calculate.
        # Get the reflection-transmission-factors
        rho = self.get_atom_reflection_factor(energy, qz, theta, atom, area, deb_wal_fac)
        tau = self.get_atom_transmission_factor(energy, qz, atom, area, deb_wal_fac)
        # calculate the reflection-transmission matrix
        H = np.ones([2, 2, len(qz)], dtype=complex)
        H[0, 0, :] = (1/tau)*(tau**2-rho**2)
        H[0, 1, :] = (1/tau)*(rho)
        H[1, 0, :] = (1/tau)*(-rho)
        H[1, 1, :] = (1/tau)
        # remember this matrix for next use with the same
        # parameters for this atom
#            if not index:
#                self.last_atom_ref_trans_matrices[index] = [atom.ID, _hash, H]
#            else:
#                self.last_atom_ref_trans_matrices.append([atom.ID, _hash, H])
        return H

    def get_atom_reflection_factor(self, energy, qz, theta, atom, area, deb_wal_fac):
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
               * atom.get_cm_atomic_form_factor(energy, qz)
               * self.get_polarization_factor(theta)
               * np.exp(-0.5*(deb_wal_fac*qz)**2))/(qz*area)
        return rho

    def get_atom_transmission_factor(self, energy, qz, atom, area, dbf):
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
                   * atom.get_atomic_form_factor(energy)
                   * np.exp(-0.5*(dbf*qz)**2))/(qz*area)
        return tau

    def get_atom_phase_matrix(self, qz, distance):
        """get_atom_phase_matrix

        Returns the phase matrix from dynamical xray theory:

        .. math::

            L = \\begin{bmatrix}
            \exp(i \phi) & 0 \\\\
            0            & \exp(-i \phi)
            \end{bmatrix}

        """
        phi = self.get_atom_phase_factor(qz, distance)
        L = np.zeros([2, 2, len(qz)], dtype=complex)
        L[0, 0, :] = np.exp(1j*phi)
        L[1, 1, :] = np.exp(-1j*phi)
        return L

    def get_atom_phase_factor(self, qz, distance):
        """get_atom_phase_factor

        Returns the phase factor :math:`\phi` for a distance :math:`d`
        from dynamical xray theory:

        .. math:: \phi = \\frac{d \ q_z}{2}

        """
        phi = distance * qz/2
        return phi

    @staticmethod
    def get_reflectivity_from_matrix(M):
        """get_reflectivity_from_matrix

        Returns the physical reflectivity from an 2x2 matrix of
        transmission and reflectifity factors:

        .. math:: R = \\left|M(0,1)/M(1,1)\\right|^2

        """
        return np.abs(M[0, 1, :]/M[1, 1, :])**2
