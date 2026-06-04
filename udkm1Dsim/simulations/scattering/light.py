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

__all__ = ['Light']

__docformat__ = 'restructuredtext'

from . import Scattering
import numpy as np
from time import time


class Light(Scattering):
    r"""Light

    Dynamical light scattering simulations.

    Calculation based on the method in Ref [5]_ and code developed Matlab
    by L. Le Guyader, see Ref [6]_.

    Copyright (2012-2014) Loïc Le Guyader
    <loic.le_guyader@helmholtz-berlin.de>

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

    References:

    .. [5] K. Ohta & H. Ishida, *Matrix formalism for calculation of the
        light beam intensity in stratified multilayered films, and its use
        in the analysis of emission spectra*, `Appl. Opt. 29, 2466 (1990).
        <https://doi.org/10.1364/AO.29.002466>`_
    .. [6] L. Le Guyader, A. Kleibert, F. Nolting, L. Joly, P.M. Derlet,
        R.V. Pisarev, A. Kirilyuk, Th. Rasing & A.V. Kimel, *Dynamics of
        laser-induced spin reorientation in Co/SmFeO_3 heterostructure*,
        `Phys. Rev. B 87, 054437 (2013).
           <https://doi.org/10.1103/PhysRevB.87.054437>`
    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self):
        """String representation of this class"""
        class_str = 'Dynamical light scattering simulation properties:\n\n'
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

        For light scattering simulation only "no analyzer polarization" is allowed.

        Args:
            pol_out_state (int): outgoing polarization state id.

        """
        self.pol_out_state = pol_out_state
        if self.pol_out_state == 0:
            self.disp_message('analyzer polarizations set to: {:s}'.format(
                self.polarizations[self.pol_out_state]))
        else:
            self.disp_message('Light scattering does only allow for NO analyzer polarizations')
            self.set_outgoing_polarization(0)

    def homogeneous_reflectivity(self):
        t1 = time()
        self.disp_message('Calculating _homogeneous_reflectivity_ ...')

        N = np.size(self._qz, 0)  # energy steps
        K = np.size(self._qz, 1)  # qz steps

        # interfaces = self.S.get_distances_of_interfaces(False)
        # N = len(interfaces)
        # # if a substrate is included add it at the end
        # if self.S.substrate != []:
        #     M = N + 1
        # else:
        #     M = N

        # opt_ref_indices = np.empty(M, dtype=complex)
        # thicknesses = np.empty(M, dtype=float)

        # # first layer is vacuum/air
        # opt_ref_indices[0] = 1+0.0j
        # thicknesses[0] = 1e-9

        # for i in range(N-1):
        #     index = finderb(interfaces[i], d_start)
        #     layer = structure.get_layer_handle(index[0])
        #     opt_ref_indices[i+1] = layer.opt_ref_index
        #     thicknesses[i+1] = interfaces[i+1]-interfaces[i]

        opt_ref_indices = self.S.get_layer_property_vector('opt_ref_index')
        thicknesses = self.S.get_layer_property_vector('_thickness')

        opt_ref_indices = np.concatenate((np.array([1+0.0j]), opt_ref_indices))
        thicknesses = np.concatenate((np.array([1]), thicknesses))

        L = len(thicknesses)

        if self.S.substrate != []:
            opt_ref_indices = np.concatenate(
                (opt_ref_indices, np.array([self.S.substrate.get_layer_handle(0).opt_ref_index])))
            thicknesses = np.concatenate(
                (thicknesses, np.array([self.S.substrate.get_thickness(False)])))
            L += 1

        R_total = np.zeros((N, K))
        T_total = np.zeros((N, K))

        for i in range(N):  # energies
            for j in range(K):  # angles
                # Snell laws
                alpha = np.empty(L, dtype=complex)
                alpha[0] = np.pi/2 - self._theta[i, j]
                alpha[1:] = np.arcsin(opt_ref_indices[0]/opt_ref_indices[1:]*np.sin(alpha[0]))

                # fresnel coefficient
                rfresnel = np.empty(L-1, dtype=complex)
                tfresnel = np.empty(L-1, dtype=complex)

                if False:  # self._excitation['polarization'] == 's':
                    rfresnel[:] = (opt_ref_indices[0:-1]*np.cos(alpha[0:-1])
                                   - opt_ref_indices[1:]*np.cos(alpha[1:])) \
                        / (opt_ref_indices[0:-1]*np.cos(alpha[0:-1])
                           + opt_ref_indices[1:]*np.cos(alpha[1:]))
                    tfresnel[:] = 2.0*opt_ref_indices[0:-1]*np.cos(alpha[0:-1]) \
                        / (opt_ref_indices[0:-1]*np.cos(alpha[0:-1])
                           + opt_ref_indices[1:]*np.cos(alpha[1:]))
                else:  # p-polarization
                    rfresnel[:] = (opt_ref_indices[1:]*np.cos(alpha[0:-1])
                                   - opt_ref_indices[0:-1]*np.cos(alpha[1:])) \
                        / (opt_ref_indices[1:]*np.cos(alpha[0:-1])
                           + opt_ref_indices[0:-1]*np.cos(alpha[1:]))
                    tfresnel[:] = 2.0*opt_ref_indices[0:-1]*np.cos(alpha[0:-1]) \
                        / (opt_ref_indices[1:]*np.cos(alpha[0:-1])
                           + opt_ref_indices[0:-1]*np.cos(alpha[1:]))

                # interface change matrix
                Jnm = np.empty((2, 2, L-1), dtype=complex)
                Jnm[0, 0, :] = 1.0/tfresnel
                Jnm[0, 1, :] = rfresnel/tfresnel
                Jnm[1, 0, :] = rfresnel/tfresnel
                Jnm[1, 1, :] = 1.0/tfresnel

                # calculating z-component of the wave vector
                k_z = 2.0*np.pi/self._wl[i]*opt_ref_indices*np.cos(alpha)

                # phase changes
                beta = k_z*thicknesses
                Ln = np.empty((2, 2, L-1), dtype=complex)
                Ln[:, :, 0] = [[1, 0], [0, 1]]
                Ln[0, 0, 1:] = np.exp(-1.0j*beta[1:-1])
                Ln[0, 1, 1:] = 0
                Ln[1, 0, 1:] = 0
                Ln[1, 1, 1:] = np.exp(1.0j*beta[1:-1])

                # calculating propagation matrix
                S = Jnm[:, :, L-2]
                for k in range(L-3, -1, -1):
                    S = np.dot(Jnm[:, :, k], np.dot(Ln[:, :, k+1], S))

                # Total transmission and reflection of the multilayer
                R_total[i, j] = np.abs(S[1, 0]/S[0, 0])**2
                if False:  # self._excitation['polarization'] == 's':
                    T_total[i, j] = (np.real(opt_ref_indices[L-1]*np.cos(alpha[L-1])
                                             / (opt_ref_indices[0]*np.cos(alpha[0])))
                                     * np.abs(1/S[0, 0])**2)
                else:
                    T_total[i, j] = (np.real(np.conj(opt_ref_indices[L-1])*np.cos(alpha[L-1])
                                             / (opt_ref_indices[0]*np.cos(alpha[0])))
                                     * np.abs(1/S[0, 0])**2)

        self.disp_message('Elapsed time for _homogeneous_reflectivity_: {:f} s'.format(time()-t1))
        return R_total, T_total
