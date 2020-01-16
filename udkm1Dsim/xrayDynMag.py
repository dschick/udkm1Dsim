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
from .xray import Xray
from .unitCell import UnitCell
from tqdm import trange
from .helpers import make_hash_md5, m_power_x2

r_0 = constants.physical_constants['classical electron radius'][0]


class XrayDynMag(Xray):
    """XrayDynMag

    Dynamical magnetic Xray simulations adapted from Elzo et.al. [4]_.
    Initially realized in `Project Dyna
    <http://neel.cnrs.fr/spip.php?rubrique1008>`_

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
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results
        polarization (float): polarization state
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
                                             'P': []}

    def __str__(self):
        """String representation of this class"""
        class_str = 'Dynamical magnetic X-Ray Diffraction simulation ' \
                    'properties:\n\n'
        class_str += super().__str__()
        return class_str

    def get_atom_boundary_phase_matrix(self, atom, area, distance, *args):
        """get_atom_boundary_phase_matrix

        Returns the reflection-transmission matrix of an atom from
        Elzo formalism:

        """
        # check for already calculated data
        _hash = make_hash_md5([self._energy, self._qz, self.polarization, area, distance, args])
        try:
            index = self.last_atom_ref_trans_matrices['atom_ids'].index(atom.id)
        except ValueError:
            index = -1
        except AttributeError:
            # its vacuum
            A, P = self.calc_atom_boundary_phase_matrix(atom, area, distance, args)
            return A, P

        if (index >= 0) and (_hash == self.last_atom_ref_trans_matrices['hashes'][index]):
            # These are the same X-ray parameters as last time so we
            # can use the same matrix again for this atom
            A = self.last_atom_ref_trans_matrices['A'][index]
            P = self.last_atom_ref_trans_matrices['P'][index]
        else:
            # These are new parameters so we have to calculate.
            # Get the reflection-transmission-factors
            A, P = self.calc_atom_boundary_phase_matrix(atom, area, distance, args)
            # remember this matrix for next use with the same
            # parameters for this atom
            if index >= 0:
                self.last_atom_ref_trans_matrices['atom_ids'][index] = atom.id
                self.last_atom_ref_trans_matrices['hashes'][index] = _hash
                self.last_atom_ref_trans_matrices['A'][index] = A
                self.last_atom_ref_trans_matrices['P'][index] = P
            else:
                self.last_atom_ref_trans_matrices['atom_ids'].append(atom.id)
                self.last_atom_ref_trans_matrices['hashes'].append(_hash)
                self.last_atom_ref_trans_matrices['A'].append(A)
                self.last_atom_ref_trans_matrices['P'].append(P)
        return A, P

    def calc_atom_boundary_phase_matrix(self, atom, area, distance, *args):
        """calc_atom_boundary_phase_matrix

        Calculates the reflection-transmission matrix of an atom from
        Elzo formalism:

        """
        if len(args) > 0:
            mag_amplitude = args[0]
        else:
            try:
                mag_amplitude = atom.magnetization
            except AttributeError:
                mag_amplitude = 0

        if len(args) > 1:
            mag_phi = args[1]
        else:
            try:
                mag_phi = atom.mag_phi
            except AttributeError:
                mag_phi = 0

        if len(args) > 2:
            mag_gamma = args[2]
        else:
            try:
                mag_gamma = atom.mag_gamma
            except AttributeError:
                mag_gamma = 0

        M = len(self._energy)  # number of energies
        N = np.shape(self._qz)[1]  # number of q_z

        u = [np.sin(mag_phi) * np.cos(mag_gamma),
             np.sin(mag_phi) * np.sin(mag_gamma),
             np.cos(mag_phi)]

        eps = np.zeros([M, N, 3, 3], dtype=np.cfloat)
        A = np.zeros([M, N, 4, 4], dtype=np.cfloat)
        P = np.zeros([M, N, 4, 4], dtype=np.cfloat)

        try:
            # calculate molar density
            mass_density = atom._mass/(area*distance)/1000  # in g/cm³
            density = mass_density/atom.mass_number_a
        except AttributeError:
            density = 0
        energy = self._energy
        factor = 830.9471/energy**2
        theta = self._theta

        try:
            cf = atom.get_atomic_form_factor(energy)
        except AttributeError:
            cf = np.zeros_like(energy)
#            mag = mag_amplitude * atom.get_magnetic_scattering_factor(energy)
        mag = np.zeros_like(energy)  # * factor * mag_amplitude * 0.01 * cf
        mag = np.tile(mag[:, np.newaxis], [1, N])

        eps0 = 1 - factor*density*cf
        eps0 = np.tile(eps0[:, np.newaxis], [1, N])

        eps[:, :, 0, 0] = eps0
        eps[:, :, 0, 1] = -1j * u[2] * mag
        eps[:, :, 0, 2] = 1j * u[1] * mag
        eps[:, :, 1, 0] = -eps[:, :, 0, 1]
        eps[:, :, 1, 1] = eps0
        eps[:, :, 1, 2] = -1j * u[0] * mag
        eps[:, :, 2, 0] = -eps[:, :, 0, 2]
        eps[:, :, 2, 1] = -eps[:, :, 1, 2]
        eps[:, :, 2, 2] = eps0

        alpha_y = np.divide(np.cos(theta), np.sqrt(eps[:, :, 0, 0]))
        alpha_z = np.sqrt(1 - alpha_y**2)

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

        A[:, :, 0, 0] = -1 - 1j * eps[:, :, 0, 1] * alpha_z_right_down
        - 1j * eps[:, :, 0, 2] * alpha_y_right_down
        A[:, :, 0, 1] = 1 - 1j * eps[:, :, 0, 1] * alpha_z_left_down
        - 1j * eps[:, :, 0, 2] * alpha_y_left_down
        A[:, :, 0, 2] = -1 + 1j * eps[:, :, 0, 1] * alpha_z_right_up
        - 1j * eps[:, :, 0, 2] * alpha_y_right_up
        A[:, :, 0, 3] = 1 + 1j * eps[:, :, 0, 1] * alpha_z_left_up
        - 1j * eps[:, :, 0, 2] * alpha_y_left_up

        A[:, :, 1, 0] = 1j * alpha_z_right_down - eps[:, :, 0, 1]
        - 1j * eps[:, :, 1, 2] * alpha_y_right_down
        A[:, :, 1, 1] = 1j * alpha_z_left_down + eps[:, :, 0, 1]
        - 1j * eps[:, :, 1, 2] * alpha_y_left_down
        A[:, :, 1, 2] = -1j * alpha_z_right_up - eps[:, :, 0, 1]
        - 1j * eps[:, :, 1, 2] * alpha_y_right_up
        A[:, :, 1, 3] = -1j * alpha_z_left_up + eps[:, :, 0, 1]
        - 1j * eps[:, :, 1, 2] * alpha_y_left_up

        A[:, :, 2, 0] = -1j * n_right_down * A[:, :, 0, 0]
        A[:, :, 2, 1] = 1j * n_left_down * A[:, :, 0, 1]
        A[:, :, 2, 2] = -1j * n_right_up * A[:, :, 0, 2]
        A[:, :, 2, 3] = 1j * n_left_up * A[:, :, 0, 3]

        A[:, :, 3, 0] = - alpha_z_right_down * n_right_down * A[:, :, 0, 0]
        A[:, :, 3, 1] = - alpha_z_left_down * n_left_down * A[:, :, 0, 1]
        A[:, :, 3, 2] = alpha_z_right_up * n_right_up * A[:, :, 0, 2]
        A[:, :, 3, 3] = alpha_z_left_up * n_left_up * A[:, :, 0, 3]

        A[:, :, :, :] = np.divide(A[:, :, :, :],
                                  np.sqrt(2) * eps[:, :, 0, 0][:, :, np.newaxis, np.newaxis])

        phase = self._k * distance
        phase = phase[:, np.newaxis]

        P[:, :, 0, 0] = np.exp(1j * phase * n_right_down * alpha_z_right_down)
        P[:, :, 1, 1] = np.exp(1j * phase * n_left_down * alpha_z_left_down)
        P[:, :, 2, 2] = np.exp(-1j * phase * n_right_up * alpha_z_right_up)
        P[:, :, 3, 3] = np.exp(-1j * phase * n_left_up * alpha_z_left_up)

        return A, P

    def calc_uc_boundary_phase_matrix(self, uc, strain):
        K = uc.num_atoms  # number of atoms
        for j in range(K):
            if j == (K-1):  # its the last atom
                del_dist = (strain+1)-uc.atoms[j][1](strain)
            else:
                del_dist = uc.atoms[j+1][1](strain)-uc.atoms[j][1](strain)

            A, P = self.get_atom_boundary_phase_matrix(uc.atoms[j][0],
                                                       uc._area,
                                                       del_dist*uc._c_axis)
            A_inv = np.linalg.inv(A)
            if j == 0:
                RT = np.einsum("lmij,lmjk->lmik", A, np.einsum("lmij,lmjk->lmik", P, A_inv))
            else:
                RT = np.einsum("lmij,lmjk->lmik", A,
                               np.einsum("lmij,lmjk->lmik", P,
                                         np.einsum("lmij,lmjk->lmik", A_inv, RT)))
        return RT, A_inv

    def inhomogeneous_reflectivity(self, *args):
        """inhomogeneous_reflectivity"""
        RT = self.calc_inhomogeneous_matrix()
        Ref = XrayDynMag.calc_reflectivity_from_matrix(RT)
        Pol_in = np.array([1+0.j, 0.j], dtype=complex)
        Pol_out = np.array([1+0.j, 1+0.j], dtype=complex)

        X = np.matmul(Ref, Pol_in)
        R = np.real(np.matmul(np.square(np.absolute(X)), Pol_out))
        return R

    def calc_inhomogeneous_matrix(self):
        """calc_inhomogeneous_matrix"""
        strain = 0
        L = self.S.get_number_of_unit_cells()  # number of unit cells
        _, _, uc_handles = self.S.get_unit_cell_vectors()

        for i in trange(L):
            uc = uc_handles[i]
            RT_uc, A_inv = self.calc_uc_boundary_phase_matrix(uc, strain)

            if i == 0:
                RT = RT_uc
            else:
                RT = np.einsum("lmij,lmjk->lmik", RT_uc, RT)

        # vacuum
        A0, _ = self.get_atom_boundary_phase_matrix([], 0, 0)
        # multiply vacuum and last layer
        RT = np.einsum("lmij,lmjk->lmik", A_inv, np.einsum("lmij,lmjk->lmik", RT, A0))

        return RT

    def homogeneous_reflectivity(self, *args):
        """homogeneous_reflectivity"""

        RT, A_inv = self.calc_homogeneous_matrix(self.S)

        A0, _ = self.get_atom_boundary_phase_matrix([], 0, 0)
        RT = np.einsum("lmij,lmjk->lmik", A_inv, np.einsum("lmij,lmjk->lmik", RT, A0))

        Ref = self.calc_reflectivity_from_matrix(RT)
        Pol_in = np.array([1+0.j, 0.j], dtype=complex)
        Pol_out = np.array([1+0.j, 1+0.j], dtype=complex)

        X = np.matmul(Ref, Pol_in)
        R = np.real(np.matmul(np.square(np.absolute(X)), Pol_out))
        return R

    def calc_homogeneous_matrix(self, S, *args):
        """calc_homogeneous_matrix"""
        # if no strains are given we assume no strain (1)
        if len(args) == 0:
            strains = np.zeros([S.get_number_of_sub_structures(), 1])
        else:
            strains = args[0]

        strainCounter = 0

        # traverse substructures
        for i, sub_structure in enumerate(S.sub_structures):
            if isinstance(sub_structure[0], UnitCell):
                # the sub_structure is an unitCell
                # calculate the ref-trans matrices for N unitCells
                RT_uc, A_inv = self.calc_uc_boundary_phase_matrix(sub_structure[0],
                                                                  strains[strainCounter])
                
                temp = m_power_x2(RT_uc, sub_structure[1])
                strainCounter += 1
            else:
                # its a structure
                # make a recursive call
                temp, A_inv = self.calc_homogeneous_matrix(
                        sub_structure[0],
                        strains[strainCounter:(strainCounter
                                                + sub_structure[0].get_number_of_sub_structures())])
                strainCounter = strainCounter+sub_structure[0].get_number_of_sub_structures()
                # calculate the ref-trans matrices for N sub structures
                temp = m_power_x2(temp, sub_structure[1])

            # multiply it to the output
            if i == 0:
                RT = temp
            else:
                RT = np.einsum("lmij,lmjk->lmik", temp, RT)
        

        return RT, A_inv

    @staticmethod
    def calc_reflectivity_from_matrix(RT):
        """calc_reflectivity_from_matrix"""
        Ref = np.tile(np.eye(2, 2, dtype=np.cfloat)[np.newaxis, np.newaxis, :, :],
                      (np.size(RT, 0), np.size(RT, 1), 1, 1))
        d = np.divide(1, RT[:, :, 3, 3] * RT[:, :, 2, 2] - RT[:, :, 3, 2] * RT[:, :, 2, 3])
        Ref[:, :, 0, 0] = (-RT[:, :, 3, 3] * RT[:, :, 2, 0] + RT[:, :, 2, 3] * RT[:, :, 3, 0]) * d
        Ref[:, :, 0, 1] = (-RT[:, :, 3, 3] * RT[:, :, 2, 1] + RT[:, :, 2, 3] * RT[:, :, 3, 1]) * d
        Ref[:, :, 1, 0] = (RT[:, :, 3, 2] * RT[:, :, 2, 0] - RT[:, :, 2, 2] * RT[:, :, 3, 0]) * d
        Ref[:, :, 1, 1] = (RT[:, :, 3, 2] * RT[:, :, 2, 1] - RT[:, :, 2, 2] * RT[:, :, 3, 1]) * d

        temp = np.array([[-1, 1], [-1j, -1j]])
        Ref = np.matmul(np.matmul(temp, Ref), temp/2)
        return Ref
