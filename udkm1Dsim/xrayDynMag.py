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
# from .unitCell import UnitCell
# from time import time
# from os import path
# from tqdm import trange
# from .helpers import make_hash_md5, m_power_x, m_times_n, finderb

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

    def __str__(self):
        """String representation of this class"""
        class_str = 'Dynamical magnetic X-Ray Diffraction simulation ' \
                    'properties:\n\n'
        class_str += super().__str__()
        return class_str

    def get_atom_ref_trans_matrix(self, atom, area, distance, *args):
        """get_atom_ref_trans_matrix

        Returns the reflection-transmission matrix of an atom from
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

        eps = np.zeros([3, 3, M], dtype=np.cfloat)
        k_z = np.zeros([M, N], dtype=np.cfloat)

        A = np.zeros([4, 4, M, N], dtype=np.cfloat)
        P = np.zeros([4, 4, M, N], dtype=np.cfloat)
        factor = 1.38E-7
        for k in range(M):
            energy = self._energy[k]
#            qz = self._qz[k, :]
            theta = self._theta[k, :]

#            mag = mag_amplitude * atom.get_magnetic_scattering_factor(energy)
            mag = 0 * factor * mag_amplitude * 0.01 * atom.get_atomic_form_factor(energy)

            eps0 = 1 - factor*atom.get_atomic_form_factor(energy)
            eps[0, 0, k] = eps0
            eps[0, 1, k] = -1j * u[2] * mag
            eps[0, 2, k] = 1j * u[1] * mag
            eps[1, 0, k] = -eps[0, 1, k]
            eps[1, 1, k] = eps0
            eps[1, 2, k] = -1j * u[0] * mag
            eps[2, 0, k] = -eps[0, 2, k]
            eps[2, 1, k] = -eps[1, 2, k]
            eps[2, 2, k] = eps0

            alpha_y = np.cos(theta) / np.sqrt(eps[0, 0, k])
            alpha_z = np.sqrt(1 - alpha_y**2)

            k_z[k, :] = self._k[k] * np.sqrt(eps[0, 0, k]) * alpha_z

            n_right_down = np.sqrt(eps[0, 0, k] - 1j * eps[0, 2, k] * alpha_y
                                   - 1j * eps[0, 1, k] * alpha_z)
            n_left_down = np.sqrt(eps[0, 0, k] + 1j * eps[0, 2, k] * alpha_y
                                  + 1j * eps[0, 1, k] * alpha_z)
            n_right_up = np.sqrt(eps[0, 0, k] - 1j * eps[0, 2, k] * alpha_y
                                 + 1j * eps[0, 1, k] * alpha_z)
            n_left_up = np.sqrt(eps[0, 0, k] + 1j * eps[0, 2, k] * alpha_y
                                - 1j * eps[0, 1, k] * alpha_z)

            alpha_y_right_down = np.cos(theta)/n_right_down
            alpha_z_right_down = np.sqrt(1-alpha_y_right_down**2)
            alpha_y_left_down = np.cos(theta)/n_left_down
            alpha_z_left_down = np.sqrt(1-alpha_y_left_down**2)
            alpha_y_right_up = np.cos(theta)/n_right_up
            alpha_z_right_up = np.sqrt(1-alpha_y_right_up**2)
            alpha_y_left_up = np.cos(theta)/n_left_up
            alpha_z_left_up = np.sqrt(1-alpha_y_left_up**2)

            A[0, 0, k, :] = -1 - 1j * eps[0, 1, k] * alpha_z_right_down
            - 1j * eps[0, 2, k] * alpha_y_right_down
            A[0, 1, k, :] = 1 - 1j * eps[0, 1, k] * alpha_z_left_down
            - 1j * eps[0, 2, k] * alpha_y_left_down
            A[0, 2, k, :] = -1 + 1j * eps[0, 1, k] * alpha_z_right_up
            - 1j * eps[0, 2, k] * alpha_y_right_up
            A[0, 3, k, :] = 1 + 1j * eps[0, 1, k] * alpha_z_left_up
            - 1j * eps[0, 2, k] * alpha_y_left_up

            A[1, 0, k, :] = 1j * alpha_z_right_down - eps[0, 1, k]
            - 1j * eps[1, 2, k] * alpha_y_right_down
            A[1, 1, k, :] = 1j * alpha_z_left_down + eps[0, 1, k]
            - 1j * eps[1, 2, k] * alpha_y_left_down
            A[1, 2, k, :] = -1j * alpha_z_right_up - eps[0, 1, k]
            - 1j * eps[1, 2, k] * alpha_y_right_up
            A[1, 3, k, :] = -1j * alpha_z_left_up + eps[0, 1, k]
            - 1j * eps[1, 2, k] * alpha_y_left_up

            A[2, 0, k, :] = -1j * n_right_down * A[0, 0, k, :]
            A[2, 1, k, :] = 1j * n_left_down * A[0, 1, k, :]
            A[2, 2, k, :] = -1j * n_right_up * A[0, 2, k, :]
            A[2, 3, k, :] = 1j * n_left_up * A[0, 3, k, :]

            A[3, 0, k, :] = - alpha_z_right_down * n_right_down * A[0, 0, k, :]
            A[3, 1, k, :] = - alpha_z_left_down * n_left_down * A[0, 1, k, :]
            A[3, 2, k, :] = alpha_z_right_up * n_right_up * A[0, 2, k, :]
            A[3, 3, k, :] = alpha_z_left_up * n_left_up * A[0, 3, k, :]

            A[:, :, k, :] = A[:, :, k, :] / (np.sqrt(2) * eps[0, 0, k])

            phase = self._k[k] * distance

            P[0, 0, k, :] = np.exp(1j * phase * n_right_down * alpha_z_right_down)
            P[1, 1, k, :] = np.exp(1j * phase * n_left_down * alpha_z_left_down)
            P[2, 2, k, :] = np.exp(-1j * phase * n_right_up * alpha_z_right_up)
            P[3, 3, k, :] = np.exp(-1j * phase * n_left_up * alpha_z_left_up)

            return A, P

    def calc_reflectivity(self):
        """calc_reflectivity"""
        
        M = len(self._energy)  # number of energies
        N = np.shape(self._qz)[1]  # number of q_z
        _, _, uc_handles = self.S.get_unit_cell_vectors()

        S = np.tile(np.eye(4, 4, dtype=np.cfloat)[:, :, np.newaxis, np.newaxis], (1, 1, M, N))
        F = np.zeros_like(S)
        Ref = np.tile(np.eye(2, 2, dtype=np.cfloat)[:, :, np.newaxis, np.newaxis], (1, 1, M, N))
        
        # traverse all unit cells in the sample structure
        index = 0
        last_A = []
        for i, uc in enumerate(uc_handles):
            for atom, _, _ in uc.atoms:
                A, P = self.get_atom_ref_trans_matrix(atom, uc._area, 1e-10)
                
                if index > 0:
                    for k, energy in enumerate(self._energy):
                        for j in range(N):
                            F[:, :, k, j] = np.matmul(np.linalg.inv(A[:, :, k, j]), last_A[:, :, k, j])                            
                            # skip roughness for now
                            # how about debye waller?
                            S[:, :, k, j] = np.matmul(P[:, :, k, j], np.matmul(F[:, :, k, j], S[:, :, k, j]))

                index += 1
                last_A = A
        temp = np.divide(1, S[3, 3, :, :] * S[2, 2, :, :] - S[3, 2, :, :] * S[2, 3, :, :])

        Ref[0, 0, :, :] = (-S[3, 3, :, :] * S[2, 0, :, :] + S[2, 3, :, :] * S[3, 0, :, :]) * temp
        Ref[0, 1, :, :] = (-S[3, 3, :, :] * S[2, 1, :, :] + S[2, 3, :, :] * S[3, 1, :, :]) * temp
        Ref[1, 0, :, :] = ( S[3, 2, :, :] * S[2, 0, :, :] - S[2, 2, :, :] * S[3, 0, :, :]) * temp
        Ref[1, 1, :, :] = ( S[3, 2, :, :] * S[2, 1, :, :] - S[2, 2, :, :] * S[3, 1, :, :]) * temp
        
        temp = np.tile(np.array([[-1, 1], [-1j, -1j]])[:, :, np.newaxis, np.newaxis], (1, 1, M, N))
        #Ref = np.matmul(np.matmul(np.array([[-1, 1], [-1j, -1j]]), Ref), np.array([[-1, 1j], [1, 1j]]) * 0.5)   
        Ref = np.einsum("ijlm,jklm->iklm", np.einsum("ijlm,jklm->iklm", temp, Ref), temp/2)
        return Ref
