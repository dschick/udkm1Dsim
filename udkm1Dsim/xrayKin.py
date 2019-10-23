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

"""A :mod:`XrayKin` module """

__all__ = ["XrayKin"]

__docformat__ = "restructuredtext"

import numpy as np
from .xray import Xray
from .unitCell import UnitCell
from . import u
from time import time


class XrayKin(Xray):
    """XrayKin

    Kinetic Xray simulations

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Attributes:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results
        polarization (float): polarization state

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self):
        """String representation of this class"""
        class_str = 'Kinematical X-Ray Diffraction simulation properties:\n\n'
        class_str += super().__str__()
        return class_str

    @u.wraps(None, (None, 'eV', 'm**-1', None), strict=False)
    def get_uc_atomic_form_factors(self, energy, qz, uc):
        """ get_uc_atomic_form_factors
        Returns the energy- and angle-dependent atomic form factors
        .. math: `f(q_z, E)` of all atoms in the unit cell as a vector.
        """
        if (not np.isscalar(energy)) and (not isinstance(energy, object)):
            raise TypeError('Only scalars or pint quantities for the energy are allowd!')
        f = np.zeros([uc.num_atoms, len(qz)], dtype=complex)
        for i in range(uc.num_atoms):
            f[i, :] = uc.atoms[i][0].get_cm_atomic_form_factor(energy, qz)
        return f

    @u.wraps(None, (None, 'eV', 'm**-1', None, None), strict=False)
    def get_uc_structure_factor(self, energy, qz, uc, strain=0):
        """get_uc_structure_factor

        Returns the energy-, angle-, and strain-dependent structure
        factor .. math: `S(E,q_z,\epsilon)` of the unit cell

        .. math::

            S(E,q_z,\epsilon) = \sum_i^N f_i \, \exp(-i q_z z_i(\epsilon)

        """
        if (not np.isscalar(energy)) and (not isinstance(energy, object)):
            raise TypeError('Only scalars or pint quantities for the energy are allowd!')

        if np.isscalar(qz):
            qz = np.array([qz])

        S = np.sum(self.get_uc_atomic_form_factors(energy, qz, uc)
                   * np.exp(1j * uc._c_axis
                   * np.outer(uc.get_atom_positions(strain), qz)), 0)
        return S

    def homogeneous_reflectivity(self, strains=0):
        """homogeneous_reflectivity

        Returns the reflectivity .. math:`R = E_p^t\,(E_p^t)^*` of a
        homogeneous sample structure as well as the reflected field
        .. math:`E_p^N` of all substructures.

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
        """homogeneous_reflected_field

        Calculates the reflected field :math:`E_p^t` of the whole
        sample structure as well as for each sub structure
        (:math:`E_p^N`). The reflected wave field :math:`E_p` from a
        single layer of unit cells at the detector is calculated as
        follows:[Ref. 1]

        .. math::

            E_p = \\frac{i}{\\varepsilon_0}\\frac{e^2}{m_e c_0^2}
                  \\frac{P(\\vartheta)  S(E,q_z,\\epsilon)}{A q_z}

        For the case of :math:`N` similar planes of unit cells one can
        write:

        .. math::

            E_p^N = \sum_{n=0}^{N-1} E_p \exp(i q_z z n )

        where :math:`z` is the distance between the planes (c-axis).
        The above equation can be simplified to

        .. math::

            E_p^N = E_p \psi(q_z,z,N)

        introducing the interference function

        .. math::

            \psi(q_z,z,N) = \sum_{n=0}^{N-1} \exp(i q_z z n)
           = \\frac{1- \exp(i q_z  z  N)}{1- \exp(i q_z z)}

        The total reflected wave field of all :math:`i = 1\ldots M`
        homogeneous layers (:math:`E_p^t`) is the phase-correct
        summation of all individual :math:`E_p^{N,i}`:

        .. math::

            E_p^t = \sum_{i=1}^M E_p^{N,i} \exp(i q_z Z_i)

        where :math:`Z_i = \sum_{j=1}^{i-1} N_j z_j` is the distance
        of the i-th layer from the surface.

        """
        # if no strains are given we assume no strain (1)
        if np.isscalar(strains) and strains == 0:
            strains = np.zeros([self.S.get_number_of_sub_structures(), 1])

        K = len(qz)  # nb of qz
        Ept = np.zeros([1, K])  # total reflected field
        Z = 0  # total length of the substructure from the surface
        A = list([0, 2])  # cell matrix of reflected fields EpN of substructures
        strainCounter = 0  # the is the index of the strain vector if applied

        # traverse substructures
        for i, sub_structures in enumerate(S.sub_structures):
            if isinstance(sub_structures[0], UnitCell):
                # the substructure is an unit cell and we can calculate
                # Ep directly
                Ep = self.get_Ep(energy, qz, theta, sub_structures[0], strains[strainCounter])
                z = sub_structures[0]._c_axis
                strainCounter = strainCounter+1
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

            # calculate the interferece function for N repetitions of
            # the substructure with the length z
            psi = self.get_interference_function(qz, z, sub_structures[1])
            # calculate the reflected field for N repetitions of
            # the substructure with the length z
            EpN = Ep * psi
            # remember the result
            A.append([EpN, '{:d}x {:s}'.format(sub_structures[1], sub_structures[0].name)])
            # add the reflected field of the current substructre
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
        """get_interference_function

        Calculates the interferece function for :math:`N`
        repetitions of the structure with the length :math:`z`:

        .. math::

            \psi(q_z,z,N) = \sum_{n=0}^{N-1} \exp(i q_z z n)
             = \\frac{1- \exp(i q_z z N)}{1- \exp(i q_z z)}

        """
        psi = (1-np.exp(1j*qz*z*N)) / (1 - np.exp(1j*qz*z))
        return psi

    @u.wraps(None, (None, 'eV', 'm**-1', 'rad', None, None), strict=False)
    def get_Ep(self, energy, qz, theta, uc, strain):
        """get_Ep

        Calculates the reflected field :math:`E_p` for one unit cell
        with a given strain :math:`\epsilon`:

        .. math::

            E_p = \\frac{i}{\\varepsilon_0} \\frac{e^2}{m_e c_0^2}
                  \\frac{P S(E,q_z,\epsilon)}{A q_z}

        with :math:`e` as electron charge, :math:`m_e` as electron
        mass, :math:`c_0` as vacuum light velocity,
        :math:`\\varepsilon_0` as vacuum permittivity,
        :math:`P` as polarization factor and :math:`S(E,q_z,\sigma)`
        as energy-, angle-, and strain-dependent unit cell structure
        factor.

        """
        import scipy.constants as c
        Ep = 1j/c.epsilon_0*c.elementary_charge**2/c.electron_mass/c.c**2 \
            * (self.get_polarization_factor(theta)
                * self.get_uc_structure_factor(energy, qz, uc, strain)
                / uc._area) / qz
        return Ep
