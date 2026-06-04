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

__all__ = ['XrayKin']

__docformat__ = 'restructuredtext'

from . import Scattering
from ...structures.layers import AmorphousLayer, UnitCell
from ... import u
import numpy as np
import scipy.constants as constants
from time import time

r_0 = constants.physical_constants['classical electron radius'][0]


class XrayKin(Scattering):
    r"""XrayKin

    Kinetic X-ray scattering simulations.

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

    References:
        .. [9] B. E. Warren (1990). *X-ray diffraction*.
           New York: Dover Publications

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self):
        """String representation of this class"""
        class_str = 'Kinematical X-Ray Diffraction simulation properties:\n\n'
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

        For kinematical X-ray simulation only "no analyzer polarization" is allowed.

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

    @u.wraps(None, (None, 'eV', 'm**-1', None), strict=False)
    def get_uc_atomic_form_factors(self, energy, qz, uc):
        """ get_uc_atomic_form_factors

        Returns the energy- and angle-dependent atomic form factors
        :math: `f(q_z, E)` of all atoms in the unit cell as a vector.

        Args:
            energy (float, Quantity): photon energy.
            qz (ndarray[float, Quantity]): scattering vectors.
            uc (UnitCell): unit cell object.

        Returns:
            f (ndarray[complex]): unit cell atomic form factors.

        """
        if (not np.isscalar(energy)) and (not isinstance(energy, object)):
            raise TypeError('Only scalars or Quantities are allowed for the energy!')
        f = np.zeros([uc.num_atoms, len(qz)], dtype=complex)
        for i in range(uc.num_atoms):
            f[i, :] = uc.atoms[i][0].get_cm_atomic_form_factor(energy, qz)
        return f

    @u.wraps(None, (None, 'eV', 'm**-1', None, None), strict=False)
    def get_uc_structure_factor(self, energy, qz, uc, strain=0):
        r"""get_uc_structure_factor

        Calculates the energy-, angle-, and strain-dependent structure factor
        .. math: `S(E,q_z,\epsilon)` of the unit cell:

        .. math::

            S(E,q_z,\epsilon) = \sum_i^N f_i \, \exp(-i q_z z_i(\epsilon))

        Args:
            energy (float, Quantity): photon energy.
            qz (ndarray[float, Quantity]): scattering vectors.
            uc (UnitCell): unit cell object.
            strain (float, optional): strain of the unit cell 0 .. 1.
                Defaults to 0.

        Returns:
            S (ndarray[complex]): unit cell structure factor.

        """
        if (not np.isscalar(energy)) and (not isinstance(energy, object)):
            raise TypeError('Only scalars or Quantities for the energy are allowed!')

        if np.isscalar(qz):
            qz = np.array([qz])

        S = np.sum(self.get_uc_atomic_form_factors(energy, qz, uc)
                   * np.exp(1j * uc._c_axis
                   * np.outer(uc.get_atom_positions(strain), qz)), 0)
        return S

    def homogeneous_reflectivity(self, strains=0):
        r"""homogeneous_reflectivity

        Calculates the reflectivity :math:`R = E_p^t\,(E_p^t)^*` of a
        homogeneous sample structure as well as the reflected field
        :math:`E_p^N` of all substructures.

        Args:
            strains (ndarray[float], optional): strains of each sub-structure
                0 .. 1. Defaults to 0.

        Returns:
            (tuple):
            - *R (ndarray[complex])* - homogeneous reflectivity.
            - *A (ndarray[complex])* - reflected fields of sub-structures.

        """
        if strains == 0:
            strains = np.zeros([self.S.get_number_of_sub_structures(), 1])

        t1 = time()
        self.disp_message('Calculating _homogenous_reflectivity_ ...')
        # get the reflected field of the structure for each energy
        R = np.zeros_like(self._qz)
        for i, energy in enumerate(self._energy):
            qz = self._qz[i, :]
            theta = self._theta[i, :]
            Ept, A = self.homogeneous_reflected_field(self.S, energy, qz, theta, strains)
            # calculate the real reflectivity from Ef
            R[i, :] = np.real(Ept*np.conj(Ept))
        self.disp_message('Elapsed time for _homogenous_reflectivity_: {:f} s'.format(time()-t1))
        return R, A

    @u.wraps((None, None), (None, None, 'eV', 'm**-1', 'rad', None), strict=False)
    def homogeneous_reflected_field(self, S, energy, qz, theta, strains=0):
        r"""homogeneous_reflected_field

        Calculates the reflected field :math:`E_p^t` of the whole sample
        structure as well as for each sub-structure (:math:`E_p^N`). The
        reflected wave field :math:`E_p` from a single layer of unit cells at
        the detector is calculated according to  Ref. [9]_:

        .. math::

            E_p = \frac{i}{\varepsilon_0}\frac{e^2}{m_e c_0^2}
                  \frac{P(\vartheta)  S(E,q_z,\epsilon)}{A q_z}

        For the case of :math:`N` similar planes of unit cells one can write:

        .. math::

            E_p^N = \sum_{n=0}^{N-1} E_p \exp(i q_z z n )

        where :math:`z` is the distance between the planes (c-axis). The above
        equation can be simplified to:

        .. math::

            E_p^N = E_p \psi(q_z,z,N)

        introducing the interference function

        .. math::

            \psi(q_z,z,N) & = \sum_{n=0}^{N-1} \exp(i q_z z n) \\
              & = \frac{1- \exp(i q_z  z  N)}{1- \exp(i q_z z)}

        The total reflected wave field of all :math:`i = 1\ldots M` homogeneous
        layers (:math:`E_p^t`) is the phase-correct summation of all individual
        :math:`E_p^{N,i}`:

        .. math::

            E_p^t = \sum_{i=1}^M E_p^{N,i} \exp(i q_z Z_i)

        where :math:`Z_i = \sum_{j=1}^{i-1} N_j z_j` is the distance of the
        :math:`i`-th layer from the surface.

        Args:
            S (Structure, UnitCell): structure or sub-structure to calculate on.
            energy (float, Quantity): photon energy.
            qz (ndarray[float, Quantity]): scattering vectors.
            theta (ndarray[float, Quantity]): scattering incidence angle.
            strains (ndarray[float], optional): strains of each sub-structure
                0 .. 1. Defaults to 0.

        Returns:
            (tuple):
            - *Ept (ndarray[complex])* - reflected field.
            - *A (ndarray[complex])* - reflected fields of substructures.

        """
        # if no strains are given we assume no strain (1)
        if np.isscalar(strains) and strains == 0:
            strains = np.zeros([self.S.get_number_of_sub_structures(), 1])

        N = len(qz)  # nb of qz
        Ept = np.zeros([1, N])  # total reflected field
        Z = 0  # total length of the substructure from the surface
        A = list([0, 2])  # cell matrix of reflected fields EpN of substructures
        strainCounter = 0  # the is the index of the strain vector if applied

        # traverse substructures
        for sub_structures in S.sub_structures:
            if isinstance(sub_structures[0], UnitCell):
                # the substructure is an unit cell and we can calculate
                # Ep directly
                Ep = self.get_Ep(energy, qz, theta, sub_structures[0], strains[strainCounter])
                z = sub_structures[0]._c_axis
                strainCounter = strainCounter+1
            elif isinstance(sub_structures[0], AmorphousLayer):
                raise ValueError('The substructure cannot be an AmorphousLayer!')
            else:
                # the substructure is a structure, so we do a recursive
                # call of this method
                d = sub_structures[0].get_number_of_sub_structures()
                Ep, temp = self.homogeneous_reflected_field(
                        sub_structures[0], energy, qz, theta,
                        strains[strainCounter:(strainCounter + d)])
                z = sub_structures[0].get_length().magnitude
                strainCounter = strainCounter + d
                A.append([temp, [sub_structures[0].name + ' substructures']])
                A.append([Ep, '{:d}x {:s}'.format(1, sub_structures[0].name)])

            # calculate the interference function for N repetitions of
            # the substructure with the length z
            psi = self.get_interference_function(qz, z, sub_structures[1])
            # calculate the reflected field for N repetitions of
            # the substructure with the length z
            EpN = Ep * psi
            # remember the result
            A.append([EpN, '{:d}x {:s}'.format(sub_structures[1], sub_structures[0].name)])
            # add the reflected field of the current substructure
            # phase-correct to the already calculated substructures
            Ept = Ept+(EpN*np.exp(1j*qz*Z))
            # update the total length $Z$ of the already calculated
            # substructures
            Z = Z + z*sub_structures[1]

        # add static substrate to kinXRD
        if S.substrate != []:
            temp,  temp2 = self.homogeneous_reflected_field(S.substrate, energy, qz, theta)
            A.append([temp2, 'static substrate'])
            Ept = Ept+(temp*np.exp(1j*qz*Z))
        return Ept, A

    @u.wraps(None, (None, 'm**-1', 'm', None), strict=False)
    def get_interference_function(self, qz, z, N):
        r"""get_interference_function

        Calculates the interference function for :math:`N` repetitions of the
        structure with the length :math:`z`:

        .. math::

            \psi(q_z,z,N) & = \sum_{n=0}^{N-1} \exp(i q_z z n) \\
              & = \frac{1- \exp(i q_z z N)}{1- \exp(i q_z z)}

        Args:
            qz (ndarray[float, Quantity]): scattering vectors.
            z (float): thickness/length of the structure.
            N (int): repetitions of the structure.

        Returns:
            psi (ndarray[complex]): interference function.

        """
        psi = (1-np.exp(1j*qz*z*N)) / (1 - np.exp(1j*qz*z))
        return psi

    @u.wraps(None, (None, 'eV', 'm**-1', 'rad', None, None), strict=False)
    def get_Ep(self, energy, qz, theta, uc, strain):
        r"""get_Ep

        Calculates the reflected field :math:`E_p` for one unit cell
        with a given strain :math:`\epsilon`:

        .. math::

            E_p = \frac{i}{\varepsilon_0} \frac{e^2}{m_e c_0^2}
                  \frac{P S(E,q_z,\epsilon)}{A q_z}

        with :math:`e` as electron charge, :math:`m_e` as electron
        mass, :math:`c_0` as vacuum light velocity,
        :math:`\varepsilon_0` as vacuum permittivity,
        :math:`P` as polarization factor and :math:`S(E,q_z,\sigma)`
        as energy-, angle-, and strain-dependent unit cell structure
        factor.

        Args:
            energy (float, Quantity): photon energy.
            qz (ndarray[float, Quantity]): scattering vectors.
            theta (ndarray[float, Quantity]): scattering incidence angle.
            uc (UnitCell): unit cell object.
            strain (float, optional): strain of the unit cell 0 .. 1.
                Defaults to 0.

        Returns:
            Ep (ndarray[complex]): reflected field.

        """
        Ep = 1j/constants.epsilon_0*constants.elementary_charge**2 \
            / constants.electron_mass/constants.c**2 \
            * (self.get_polarization_factor(theta)
                * self.get_uc_structure_factor(energy, qz, uc, strain)
                / uc._area) / qz
        return Ep
