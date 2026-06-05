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
from tqdm.auto import trange


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

    def homogeneous_reflectivity(self, strains=[]):
        """
        this must be very much simplified to be "homogeneous" and by being vectorized
        """
        t1 = time()
        # self.disp_message('Calculating _homogeneous_reflectivity_ ...')

        N = np.size(self._qz, 0)  # energy steps
        K = np.size(self._qz, 1)  # qz steps

        opt_ref_indices = self.S.get_layer_property_vector('opt_ref_index')
        opt_ref_indices_per_strain = self.S.get_layer_property_vector('opt_ref_index_per_strain')
        thicknesses = self.S.get_layer_property_vector('_thickness')

        if len(strains) == 0:
            strains = np.zeros_like(thicknesses)

        # both the superstrate and the substrate should be semiinfinite and static

        if self.S.substrate != []:
            opt_ref_index_substrate = self.S.substrate.get_layer_handle(0).opt_ref_index
        else:  # its vacuum
            opt_ref_index_substrate = 1+0j

        # adding a superstrate and substrate
        opt_ref_indices = np.concatenate((np.array([1+0.0j]), opt_ref_indices, np.array([opt_ref_index_substrate])))
        opt_ref_indices_per_strain = np.concatenate((np.array([0+0.0j]), opt_ref_indices_per_strain, np.array([0+0.0j])))
        thicknesses = np.concatenate((np.array([1]), thicknesses, np.array([1])))

        strains = np.concatenate((np.array([0]), strains, np.array([0])))
        # number of layers + super- and substrate
        L = len(thicknesses)

        opt_ref_indices += opt_ref_indices_per_strain*strains
        thicknesses *= (strains+1)

        # account for energy dependence of the refractive index
        # dirty hack so far

        for i, ref_index in enumerate(opt_ref_indices):
            layer = self.S.get_layer_handle(i)

            try:
                layer.opt_ref_index_nk
            except:
                pass

        R_total = np.zeros((N, K))
        T_total = np.zeros((N, K))
        
        # Snell laws
        alpha = np.empty((N, K, L), dtype=complex)
        alpha[:, :, 0] = np.pi/2 - self._theta[:, :]
        alpha[:, :, 1:] = np.arcsin(
            np.einsum('nk,l->nkl', np.sin(alpha[:, :, 0]), opt_ref_indices[0]/opt_ref_indices[1:])
            )
        
        # fresnel coefficient
        rfresnel = np.empty((N, K, L-1), dtype=complex)
        tfresnel = np.empty((N, K, L-1), dtype=complex)

        if self.pol_in_state == 3:  # self._excitation['polarization'] == 's':
            rfresnel[:, :, :] = \
                (np.einsum('l,nkl->nkl', opt_ref_indices[0:-1], np.cos(alpha[:, :, 0:-1])) - np.einsum('l,nkl->nkl', opt_ref_indices[1:], np.cos(alpha[:, :, 1:]))) \
                / (np.einsum('l,nkl->nkl', opt_ref_indices[0:-1], np.cos(alpha[:, :, 0:-1])) + np.einsum('l,nkl->nkl', opt_ref_indices[1:], np.cos(alpha[:, :, 1:])))
            tfresnel[:, :, :] = 2.0*opt_ref_indices[0:-1]*np.cos(alpha[:, :, 0:-1]) \
                / (opt_ref_indices[0:-1]*np.cos(alpha[:, :, 0:-1])
                    + opt_ref_indices[1:]*np.cos(alpha[:, :, 1:]))
        elif self.pol_in_state == 4:  # p-polarization
            rfresnel[:, :, :] = (opt_ref_indices[1:]*np.cos(alpha[:, :, 0:-1])
                            - opt_ref_indices[0:-1]*np.cos(alpha[:, :, 1:])) \
                / (opt_ref_indices[1:]*np.cos(alpha[:, :, 0:-1])
                    + opt_ref_indices[0:-1]*np.cos(alpha[:, :, 1:]))
            tfresnel[:, :, :] = 2.0*opt_ref_indices[0:-1]*np.cos(alpha[:, :, 0:-1]) \
                / (opt_ref_indices[1:]*np.cos(alpha[:, :, 0:-1])
                    + opt_ref_indices[0:-1]*np.cos(alpha[:, :, 1:]))

        # interface change matrix
        Jnm = np.empty((N, K, 2, 2, L-1), dtype=complex)
        Jnm[:, :, 0, 0, :] = 1.0/tfresnel
        Jnm[:, :, 0, 1, :] = rfresnel/tfresnel
        Jnm[:, :, 1, 0, :] = rfresnel/tfresnel
        Jnm[:, :, 1, 1, :] = 1.0/tfresnel

        # calculating z-component of the wave vector
        k_z = 2.0*np.einsum('nkl,n,l->nkl', np.cos(alpha), np.pi/self._wl, opt_ref_indices)

        # phase changes
        beta = np.einsum('nkl,l->nkl', k_z, thicknesses)
        Ln = np.empty((N, K, 2, 2, L-1), dtype=complex)
        Ln[:, :, :, :, 0] = [[1, 0], [0, 1]]
        Ln[:, :, 0, 0, 1:] = np.exp(-1.0j*beta[:, :, 1:-1])
        Ln[:, :, 0, 1, 1:] = 0
        Ln[:, :, 1, 0, 1:] = 0
        Ln[:, :, 1, 1, 1:] = np.exp(1.0j*beta[:, :, 1:-1])

        # calculating propagation matrix
        S = Jnm[:, :, :, :, L-2]
        for k in range(L-3, -1, -1):
            S = np.einsum('nkpo, nkpj -> nkoj', Jnm[:, :, :, :, k],
                          np.einsum('nkpo, nkpj -> nkoj', Ln[:, :, :, :, k+1], S))

        # Total transmission and reflection of the multilayer
        R_total = np.abs(S[:, :, 1, 0]/S[:, :, 0, 0])**2
        if self.pol_in_state == 3:  # self._excitation['polarization'] == 's':
            T_total = (np.real(opt_ref_indices[L-1]*np.cos(alpha[:, :, L-1])
                                        / (opt_ref_indices[0]*np.cos(alpha[:, :, 0])))
                                * np.abs(1/S[:, :, 0, 0])**2)
        elif self.pol_in_state == 4:
            T_total = (np.real(np.conj(opt_ref_indices[L-1])*np.cos(alpha[:, :, L-1])
                                        / (opt_ref_indices[0]*np.cos(alpha[:, :, 0])))
                                * np.abs(1/S[:, :, 0, 0])**2)

        # self.disp_message('Elapsed time for _homogeneous_reflectivity_: {:f} s'.format(time()-t1))
        return R_total, T_total

    def inhomogeneous_reflectivity(self, strain_map):
        M = np.size(strain_map, 0)  # delay steps
        R = np.zeros([M, np.size(self._qz, 0), np.size(self._qz, 1)])
        T = np.zeros([M, np.size(self._qz, 0), np.size(self._qz, 1)])

        for i in trange(M):
            R[i, :, :], T[i, :, :] = self.homogeneous_reflectivity(strain_map[i, :])

        return R, T
