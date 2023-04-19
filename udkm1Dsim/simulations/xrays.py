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

__all__ = ['Xray', 'XrayKin', 'XrayDyn', 'XrayDynMag']

__docformat__ = 'restructuredtext'

from .simulation import Simulation
from ..structures.layers import AmorphousLayer, UnitCell
from .. import u, Q_
from ..helpers import make_hash_md5, m_power_x, m_times_n, finderb
import numpy as np
import scipy.constants as constants
from time import time
from os import path
from tqdm.notebook import trange

r_0 = constants.physical_constants['classical electron radius'][0]


class Xray(Simulation):
    r"""Xray

    Base class for all X-ray scattering simulations.

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

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self._energy = np.array([])
        self._wl = np.array([])
        self._k = np.array([])
        self._theta = np.zeros([1, 1])
        self._qz = np.zeros([1, 1])

        self.polarizations = {0: 'unpolarized',
                              1: 'circ +',
                              2: 'circ -',
                              3: 'sigma',
                              4: 'pi'}

        self.pol_in_state = 3  # sigma
        self.pol_out_state = 0  # no-analyzer
        self.pol_in = None
        self.pol_out = None
        self.set_polarization(self.pol_in_state, self.pol_out_state)

    def __str__(self, output=[]):
        """String representation of this class"""
        output = [['energy', self.energy[0] if np.size(self.energy) == 1 else
                   '{:f} .. {:f}'.format(np.min(self.energy), np.max(self.energy))],
                  ['wavelength', self.wl[0] if np.size(self.wl) == 1 else
                   '{:f} .. {:f}'.format(np.min(self.wl), np.max(self.wl))],
                  ['wavenumber', self.k[0] if np.size(self.k) == 1 else
                   '{:f} .. {:f}'.format(np.min(self.k), np.max(self.k))],
                  ['theta', self.theta[0] if np.size(self.theta) == 1 else
                   '{:f} .. {:f}'.format(np.min(self.theta), np.max(self.theta))],
                  ['q_z', self.qz[0] if np.size(self.qz) == 1 else
                   '{:f} .. {:f}'.format(np.min(self.qz), np.max(self.qz))],
                  ['incoming polarization', self.polarizations[self.pol_in_state]],
                  ['analyzer polarization', self.polarizations[self.pol_out_state]],
                  ] + output
        return super().__str__(output)

    def set_incoming_polarization(self, pol_in_state):
        """set_incoming_polarization

        Must be overwritten by child classes.

        Args:
            pol_in_state (int): incoming polarization state id.

        """
        raise NotImplementedError

    def set_outgoing_polarization(self, pol_out_state):
        """set_outgoing_polarization

        Must be overwritten by child classes.

        Args:
            pol_out_state (int): outgoing polarization state id.

        """
        raise NotImplementedError

    def set_polarization(self, pol_in_state, pol_out_state):
        """set_polarization

        Sets the incoming and analyzer (outgoing) polarization.

        Args:
            pol_in_state (int): incoming polarization state id.
            pol_out_state (int): outgoing polarization state id.

        """
        self.set_incoming_polarization(pol_in_state)
        self.set_outgoing_polarization(pol_out_state)

    def get_hash(self, strain_vectors, **kwargs):
        """get_hash

        Calculates an unique hash given by the energy :math:`E`,
        :math:`q_z` range, polarization states and the ``strain_vectors`` as
        well as the sample structure hash for relevant x-ray parameters.
        Optionally, part of the strain_map is used.

        Args:
            strain_vectors (dict{ndarray[float]}): reduced strains per unique
                layer.
            **kwargs (ndarray[float]): spatio-temporal strain profile.

        Returns:
            hash (str): unique hash.

        """
        param = [self.pol_in_state, self.pol_out_state, self._qz, self._energy, strain_vectors]

        if 'strain_map' in kwargs:
            strain_map = kwargs.get('strain_map')
            if np.size(strain_map) > 1e6:
                strain_map = strain_map.flatten()[0:1000000]
            param.append(strain_map)

        return self.S.get_hash(types='xray') + '_' + make_hash_md5(param)

    def get_polarization_factor(self, theta):
        r"""get_polarization_factor

        Calculates the polarization factor :math:`P(\vartheta)` for a given
        incident angle :math:`\vartheta` for the case of `s`-polarization
        (pol = 0), or `p`-polarization (pol = 1), or unpolarized X-rays
        (pol = 0.5):

        .. math::

            P(\vartheta) = \sqrt{(1-\mbox{pol}) + \mbox{pol} \cdot \cos(2\vartheta)}

        Args:
            theta (ndarray[float]): incidence angle.

        Returns:
            P (ndarray[float]): polarization factor.

        """
        return np.sqrt((1-self.pol_in) + self.pol_in*np.cos(2*theta)**2)

    def update_experiment(self, caller):
        r"""update_experiment

        Recalculate energy, wavelength, and wavevector as well as theta
        and the scattering vector in case any of these has changed.

        .. math::

            \lambda & = \frac{hc}{E} \\
            E & = \frac{hc}{\lambda} \\
            k & = \frac{2\pi}{\lambda} \\
            \vartheta & = \arcsin{\frac{\lambda q_z}{4\pi}} \\
            q_z & = 2k \sin{\vartheta}

        Args:
            caller (str): name of calling method.

        """
        from scipy import constants
        if caller != 'energy':
            if caller == 'wl':  # calc energy from wavelength
                self._energy = Q_((constants.h*constants.c)/self._wl, 'J').to('eV').magnitude
            elif caller == 'k':  # calc energy von wavevector
                self._energy = \
                    Q_((constants.h*constants.c)/(2*np.pi/self._k), 'J').to('eV').magnitude
        if caller != 'wl':
            if caller == 'energy':  # calc wavelength from energy
                self._wl = (constants.h*constants.c)/self.energy.to('J').magnitude
            elif caller == 'k':  # calc wavelength from wavevector
                self._wl = 2*np.pi/self._k
        if caller != 'k':
            if caller == 'energy':  # calc wavevector from energy
                self._k = 2*np.pi/self._wl
            elif caller == 'wl':  # calc wavevector from wavelength
                self._k = 2*np.pi/self._wl

        if caller != 'theta':
            self._theta = np.arcsin(np.outer(self._wl, self._qz[0, :])/np.pi/4)
        if caller != 'qz':
            self._qz = np.outer(2*self._k, np.sin(self._theta[0, :]))

    @property
    def energy(self):
        return Q_(self._energy, u.eV)

    @energy.setter
    def energy(self, energy):
        self._energy = np.array(energy.to('eV').magnitude, ndmin=1)
        self.update_experiment('energy')

    @property
    def wl(self):
        return Q_(self._wl, u.m).to('nm')

    @wl.setter
    def wl(self, wl):
        self._wl = np.array(wl.to_base_units().magnitude, ndmin=1)
        self.update_experiment('wl')

    @property
    def k(self):
        return Q_(self._k, 1/u.m).to('1/nm')

    @k.setter
    def k(self, k):
        self._k = np.array(k.to_base_units().magnitude, ndmin=1)
        self.update_experiment('k')

    @property
    def theta(self):
        return Q_(self._theta, u.rad).to('deg')

    @theta.setter
    def theta(self, theta):
        self._theta = np.array(theta.to_base_units().magnitude, ndmin=1)
        if self._theta.ndim < 2:
            self._theta = np.tile(self._theta, (len(self._energy), 1))
        self.update_experiment('theta')

    @property
    def qz(self):
        return Q_(self._qz, 1/u.m).to('1/nm')

    @qz.setter
    def qz(self, qz):
        self._qz = np.array(qz.to_base_units().magnitude, ndmin=1)
        if self._qz.ndim < 2:
            self._qz = np.tile(self._qz, (len(self._energy), 1))
        self.update_experiment('qz')


class XrayKin(Xray):
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
        import scipy.constants as c
        Ep = 1j/c.epsilon_0*c.elementary_charge**2/c.electron_mass/c.c**2 \
            * (self.get_polarization_factor(theta)
                * self.get_uc_structure_factor(energy, qz, uc, strain)
                / uc._area) / qz
        return Ep


class XrayDyn(Xray):
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

    def homogeneous_reflectivity(self, *args):
        r"""homogeneous_reflectivity

        Calculates the reflectivity :math:`R` of the whole sample structure
        and the reflectivity-transmission matrices :math:`M_{RT}` for
        each substructure. The reflectivity of the :math:`2\times 2`
        matrices for each :math:`q_z` is calculates as follow:

        .. math:: R = \left|M_{RT}^t(0,1)/M_{RT}^t(1,1)\right|^2

        Args:
            *args (ndarray[float], optional): strains for each substructure.

        Returns:
            (tuple):
            - *R (ndarray[float])* - homogeneous reflectivity.
            - *A (ndarray[complex])* - reflectivity-transmission matrices of
              sub-structures.

        """
        # if no strains are given we assume no strain
        if len(args) == 0:
            strains = np.zeros([self.S.get_number_of_sub_structures(), 1])
        else:
            strains = args[0]
        t1 = time()
        self.disp_message('Calculating _homogenous_reflectivity_ ...')
        # get the reflectivity-transmission matrix of the structure
        RT, A = self.homogeneous_ref_trans_matrix(self.S, strains)
        # calculate the real reflectivity from the RT matrix
        R = self.calc_reflectivity_from_matrix(RT)
        self.disp_message('Elapsed time for _homogenous_reflectivity_: {:f} s'.format(time()-t1))
        return R, A

    def homogeneous_ref_trans_matrix(self, S, *args):
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
            *args (ndarray[float], optional): strains for each substructure.

        Returns:
            (tuple):
            - *RT (ndarray[complex])* - reflectivity-transmission matrix.
            - *A (ndarray[complex])* - reflectivity-transmission matrices of
              sub-structures.

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
        for sub_structure in S.sub_structures:
            if isinstance(sub_structure[0], UnitCell):
                # the sub_structure is an unitCell
                # calculate the ref-trans matrices for N unitCells
                temp = m_power_x(self.get_uc_ref_trans_matrix(
                        sub_structure[0], strains[strainCounter]),
                        sub_structure[1])
                strainCounter += 1
                # remember the result
                A.append([temp, '{:d}x {:s}'.format(sub_structure[1], sub_structure[0].name)])
            elif isinstance(sub_structure[0], AmorphousLayer):
                raise ValueError('The substructure cannot be an AmorphousLayer!')
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

    def inhomogeneous_reflectivity(self, strain_map, strain_vectors, **kwargs):
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
            strain_vectors (list[ndarray[float]]): reduced strains per unique
                layer.
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
                   + self.get_hash(strain_vectors, strain_map=strain_map) \
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

            self.disp_message('Elapsed time for _inhomogeneous_reflectivity_:'
                              ' {:f} s'.format(time()-t1))
            self.save(full_filename, {'R': R}, '_inhomogeneous_reflectivity_')
        return R

    def sequential_inhomogeneous_reflectivity(self, strain_map, strain_vectors, RTM):
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
        # structure for each time step of the strain map
        for i in iterator:
            R[i, :, :] = self.calc_inhomogeneous_reflectivity(strain_map[i, :],
                                                              strain_vectors,
                                                              RTM)
        return R

    def parallel_inhomogeneous_reflectivity(self, strain_map, strain_vectors,
                                            RTM, dask_client):
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
        uc_indices, _, _ = self.S.get_layer_vectors()
        # init unity matrix for matrix multiplication
        RTU = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (N, K, 1, 1))
        # make RTM available for all works
        remote_RTM = dask_client.scatter(RTM)
        remote_RTU = dask_client.scatter(RTU)
        remote_uc_indices = dask_client.scatter(uc_indices)
        remote_strain_vectors = dask_client.scatter(strain_vectors)

        # precalculate the substrate ref_trans_matrix if present
        if self.S.substrate != []:
            RTS, _ = self.homogeneous_ref_trans_matrix(self.S.substrate)
        else:
            RTS = RTU

        # create dask.delayed tasks for all delay steps
        for i in range(M):
            RT = delayed(XrayDyn.calc_inhomogeneous_ref_trans_matrix)(
                    remote_uc_indices,
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

    def distributed_inhomogeneous_reflectivity(self, strain_map, strain_vectors, RTM,
                                               job, num_worker):
        """distributed_inhomogeneous_reflectivity

        This is a stub. Not yet implemented in python.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            strain_vectors (list[ndarray[float]]): reduced strains per unique
                layer.
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.
            job (Dask.job): Dask job.
            num_workers (int): Dask number of workers.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        raise NotImplementedError

    def calc_inhomogeneous_reflectivity(self, strains, strain_vectors, RTM):
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
            strain_map (ndarray[float]): spatio-temporal strain profile.
            strain_vectors (list[ndarray[float]]): reduced strains per unique
                layer.
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        # initialize ref_trans_matrix
        N = np.shape(self._qz)[1]  # number of q_z
        M = np.shape(self._qz)[0]  # number of energies
        uc_indices, _, _ = self.S.get_layer_vectors()

        # initialize ref_trans_matrix
        RTU = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (M, N, 1, 1))

        RT = XrayDyn.calc_inhomogeneous_ref_trans_matrix(uc_indices,
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
    def calc_inhomogeneous_ref_trans_matrix(uc_indices, RT, strains,
                                            strain_vectors, RTM):
        r"""calc_inhomogeneous_ref_trans_matrix

        Sub-function of :meth:`calc_inhomogeneous_reflectivity` and for
        parallel computing (needs to be static) only for calculating the
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
            temp = RTM[int(uc_index)][strain_index]
            if temp is not []:
                RT = m_times_n(RT, temp)
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

    def get_uc_ref_trans_matrix(self, uc, *args):
        r"""get_uc_ref_trans_matrix

        Returns the reflection-transmission matrix of a unit cell:

        .. math:: M_{RT} = \prod_i H_i \  L_i

        where :math:`H_i` and :math:`L_i` are the atomic reflection-
        transmission matrix and the phase matrix for the atomic distances,
        respectively.

        Args:
            uc (UnitCell): unit cell object.
            args (float, optional): strain of unit cell.

        Returns:
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.

        """
        if len(args) == 0:
            strain = 0  # set the default strain to 0
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
            # The relative position is calculated by the function handle
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
        - :math:`M = 0.5(\mbox{dbf} \ q_z)^2)` where
          :math:`\mbox{dbf}^2 = \langle u^2\rangle` is the average
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
               * np.exp(-0.5*(deb_wal_fac*self._qz)**2))/(self._qz*area)
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
        - :math:`M = 0.5(\mbox{dbf} \ q_z)^2` where
          :math:`\mbox{dbf}^2 = \langle u^2\rangle` is the average
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
                   * np.exp(-0.5*(deb_wal_fac*self._qz)**2))/(self._qz*area)
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
        L = np.zeros([np.shape(self._qz)[0], np.shape(self._qz)[1], 2, 2], dtype=np.cfloat)
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


class XrayDynMag(Xray):
    r"""XrayDynMag

    Dynamical magnetic X-ray scattering simulations.

    Adapted from Elzo et.al. [10]_ and initially realized in `Project Dyna
    <http://dyna.neel.cnrs.fr>`_.

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

        .. [10] M. Elzo, E. Jal, O. Bunau, S. Grenier, Y. Joly, A. Y.
           Ramos, H. C. N. Tolentino, J. M. Tonnerre & N. Jaouen, *X-ray
           resonant magnetic reflectivity of stratified magnetic structures:
           Eigenwave formalism and application to a W/Fe/W trilayer*,
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

        Calculates an unique hash given by the energy :math:`E`, :math:`q_z`
        range, polarization states as well as the sample structure hash for
        relevant x-ray and magnetic parameters. Optionally, part of the
        ``strain_map`` and ``magnetization_map`` are used.

        Args:
            **kwargs (ndarray[float]): spatio-temporal strain and magnetization
                profile.

        Returns:
            hash (str): unique hash.

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

        Sets the incoming polarization factor for circular +, circular -, sigma,
        pi, and unpolarized polarization.

        Args:
            pol_in_state (int): incoming polarization state id.

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

        Sets the outgoing polarization factor for circular +, circular -, sigma,
        pi, and unpolarized polarization.

        Args:
            pol_out_state (int): outgoing polarization state id.

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
        r"""homogeneous_reflectivity

        Calculates the reflectivity :math:`R` of the whole sample structure
        allowing only for homogeneous strain and magnetization.

        The reflection-transmission matrices

        .. math:: RT = A_f^{-1} \prod_m \left( A_m P_m A_m^{-1} \right) A_0

        are calculated for every substructure :math:`m` before post-processing
        the incoming and analyzer polarizations and calculating the actual
        reflectivities as function of energy and :math:`q_z`.

        Args:
            args (ndarray[float], optional): strains and magnetization for each
                sub-structure.

        Returns:
            (tuple):
            - *R (ndarray[float])* - homogeneous reflectivity.
            - *R_phi (ndarray[float])* - homogeneous reflectivity for opposite
              magnetization.

        """
        t1 = time()
        self.disp_message('Calculating _homogeneous_reflectivity_ ...')
        # vacuum boundary
        A0, A0_phi, _, _, _, _, k_z_0 = self.get_atom_boundary_phase_matrix([], 0, 0)
        # calc the reflectivity-transmission matrix of the structure
        # and the inverse of the last boundary matrix
        RT, RT_phi, last_A, last_A_phi, last_A_inv, last_A_inv_phi, last_k_z = \
            self.calc_homogeneous_matrix(self.S, A0, A0_phi, k_z_0, *args)
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
        # calc the actual reflectivity and transmissivity from the matrix
        R, T = XrayDynMag.calc_reflectivity_transmissivity_from_matrix(
            RT, self.pol_in, self.pol_out)
        R_phi, T_phi = XrayDynMag.calc_reflectivity_transmissivity_from_matrix(
            RT_phi, self.pol_in, self.pol_out)
        self.disp_message('Elapsed time for _homogeneous_reflectivity_: {:f} s'.format(time()-t1))
        return R, R_phi, T, T_phi

    def calc_homogeneous_matrix(self, S, last_A, last_A_phi, last_k_z, *args):
        r"""calc_homogeneous_matrix

        Calculates the product of all reflection-transmission matrices of the
        sample structure

        .. math:: RT = \prod_m \left(P_m A_m^{-1} A_{m-1} \right)

        If the sub-structure :math:`m` consists of :math:`N` unit cells
        the matrix exponential rule is applied:

        .. math:: RT_m = \left( P_{UC} A_{UC}^{-1} A_{UC} \right)^N

        Roughness is also included by a gaussian width

        Args:
            S (Structure, UnitCell, AmorphousLayer): structure, sub-structure,
                unit cell or amorphous layer to calculate on.
            last_A (ndarray[complex]): last atom boundary matrix.
            last_A_phi (ndarray[complex]): last atom boundary matrix for opposite
                magnetization.
            last_k_z (ndarray[float]): last internal wave vector
            args (ndarray[float], optional): strains and magnetization for each
                sub-structure.

        Return:
            (tuple):
            - *RT (ndarray[complex])* - reflection-transmission matrix.
            - *RT_phi (ndarray[complex])* - reflection-transmission matrix for
              opposite magnetization.
            - *A (ndarray[complex])* - atom boundary matrix.
            - *A_phi (ndarray[complex])* - atom boundary matrix for opposite
              magnetization.
            - *A_inv (ndarray[complex])* - inverted atom boundary matrix.
            - *A_inv_phi (ndarray[complex])* - inverted atom boundary matrix for
              opposite magnetization.
            - *k_z (ndarray[float])* - internal wave vector.

        """
        # if no strains are given we assume no strain (1)
        if len(args) == 0:
            strains = np.zeros([S.get_number_of_sub_structures(), 1])
        else:
            strains = args[0]

        if len(args) < 2:
            # create non-working magnetizations
            magnetizations = np.zeros([S.get_number_of_sub_structures(), 1])
        else:
            magnetizations = args[1]

        layer_counter = 0
        # traverse substructures
        for i, sub_structure in enumerate(S.sub_structures):
            layer = sub_structure[0]
            repetitions = sub_structure[1]
            if isinstance(layer, UnitCell):
                # the sub_structure is an unitCell
                # calculate the ref-trans matrices for N unitCells
                RT_uc, RT_uc_phi, A, A_phi, A_inv, A_inv_phi, k_z = \
                    self.calc_uc_boundary_phase_matrix(
                        layer, last_A, last_A_phi, last_k_z, strains[layer_counter],
                        magnetizations[layer_counter])
                temp = RT_uc
                temp_phi = RT_uc_phi
                if repetitions > 1:
                    # use m_power_x for more than one repetition
                    temp2, temp2_phi, A, A_phi, A_inv, A_inv_phi, k_z = \
                        self.calc_uc_boundary_phase_matrix(
                            layer, A, A_phi, k_z, strains[layer_counter],
                            magnetizations[layer_counter])
                    temp2 = m_power_x(temp2, repetitions-1)
                    temp2_phi = m_power_x(temp2_phi, repetitions-1)
                    temp = m_times_n(temp2, temp)
                    temp_phi = m_times_n(temp2_phi, temp_phi)

                layer_counter += 1
            elif isinstance(layer, AmorphousLayer):
                # the sub_structure is an amorphous layer
                # calculate the ref-trans matrices for N layers

                A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = \
                    self.get_atom_boundary_phase_matrix(layer.atom,
                                                        layer._density*(
                                                            strains[layer_counter]+1),
                                                        layer._thickness*(
                                                            strains[layer_counter]+1),
                                                        False,
                                                        magnetizations[layer_counter])

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
                layer_counter += 1
            else:
                # its a structure
                # make a recursive call
                temp, temp_phi, A, A_phi, A_inv, A_inv_phi, k_z = self.calc_homogeneous_matrix(
                        layer, last_A, last_A_phi, last_k_z,
                        strains[layer_counter:(
                            layer_counter
                            + layer.get_number_of_sub_structures()
                            )],
                        magnetizations[layer_counter:(
                            layer_counter
                            + layer.get_number_of_sub_structures()
                            )])
                # calculate the ref-trans matrices for N sub structures
                if repetitions > 1:
                    # use m_power_x for more than one repetition
                    temp2, temp2_phi, A, A_phi, A_inv, A_inv_phi, k_z = \
                        self.calc_homogeneous_matrix(
                            layer, A, A_phi, k_z,
                            strains[layer_counter:(layer_counter
                                                   + layer.get_number_of_sub_structures())],
                            magnetizations[layer_counter:(layer_counter
                                                          + layer.get_number_of_sub_structures())])

                    temp = m_times_n(m_power_x(temp2, repetitions-1), temp)
                    temp_phi = m_times_n(m_power_x(temp2_phi, repetitions-1), temp_phi)

                layer_counter = layer_counter+layer.get_number_of_sub_structures()

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

    def inhomogeneous_reflectivity(self, strain_map=np.array([]),
                                   magnetization_map=np.array([]), **kwargs):
        """inhomogeneous_reflectivity

        Returns the reflectivity and transmissivity of an inhomogeneously
        strained and magnetized sample structure for a given _strain_map_
        and _magnetization_map_ in space and time for each unit cell or
        amorphous layer in the sample structure. If no reflectivity is
        saved in the cache it is caluclated. Providing the ``calc_type``
        for the calculation the corresponding sub-routines for the
        reflectivity computation are called:

        * ``parallel`` parallelization over the time steps utilizing
          `Dask <https://dask.org/>`_
        * ``distributed`` not implemented in Python, but should be possible
          with `Dask <https://dask.org/>`_ as well
        * ``sequential`` no parallelization at all

        Args:
            strain_map (ndarray[float], optional): spatio-temporal strain
                profile.
            magnetization_map (ndarray[float], optional): spatio-temporal
                magnetization profile.
            **kwargs:
                - *calc_type (str)* - type of calculation.
                - *dask_client (Dask.Client)* - Dask client.
                - *job (Dask.job)* - Dask job.
                - *num_workers (int)* - Dask number of workers.

        Returns:
            (tuple):
            - *R (ndarray[float])* - inhomogeneous reflectivity.
            - *R_phi (ndarray[float])* - inhomogeneous reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - inhomogeneous transmissivity.
            - *T_phi (ndarray[float])* - inhomogeneous transmissivity for opposite
              magnetization.

        """
        # create a hash of all simulation parameters
        filename = 'inhomogeneous_reflectivity_dynMag_' \
                   + self.get_hash(strain_map=strain_map, magnetization_map=magnetization_map) \
                   + '.npz'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            R = tmp['R']
            R_phi = tmp['R_phi']
            T = tmp['T']
            T_phi = tmp['T_phi']
            self.disp_message('_inhomogeneous_reflectivity_ loaded from file:\n\t' + filename)
        else:
            t1 = time()
            self.disp_message('Calculating _inhomogeneous_reflectivity_ ...')
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

            M = np.size(strain_map, 0)
            N = np.size(magnetization_map, 0)

            if (M == 0) and (N > 0):
                strain_map = np.zeros([np.size(magnetization_map, 0),
                                       np.size(magnetization_map, 1)])
            elif (M > 0) and (N == 0):
                magnetization_map = np.zeros_like(strain_map)
            elif (M == 0) and (N == 0):
                raise ValueError('At least a strain_map or magnetzation_map must be given!')
            else:
                if M != N:
                    raise ValueError('The strain_map and magnetzation_map must '
                                     'have the same number of delay steps!')

            # select the type of computation
            if calc_type == 'parallel':
                R, R_phi, T, T_phi = self.parallel_inhomogeneous_reflectivity(
                    strain_map, magnetization_map, dask_client)
            elif calc_type == 'distributed':
                R, R_phi, T, T_phi = self.distributed_inhomogeneous_reflectivity(
                    strain_map, magnetization_map, job, num_workers)
            else:  # sequential
                R, R_phi, T, T_phi = self.sequential_inhomogeneous_reflectivity(
                    strain_map, magnetization_map)

            self.disp_message('Elapsed time for _inhomogeneous_reflectivity_:'
                              ' {:f} s'.format(time()-t1))
            self.save(full_filename, {'R': R, 'R_phi': R_phi, 'T': T, 'T_phi': T_phi},
                      '_inhomogeneous_reflectivity_')
        return R, R_phi, T, T_phi

    def sequential_inhomogeneous_reflectivity(self, strain_map, magnetization_map):
        """sequential_inhomogeneous_reflectivity

        Returns the reflectivity and transmission of an inhomogeneously strained
        sample structure for a given ``strain_map`` and ``magnetization_map`` in
        space and time. The function calculates the results sequentially for every
        layer without parallelization.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            magnetization_map (ndarray[float]): spatio-temporal magnetization
                profile.

        Returns:
            (tuple):
            - *R (ndarray[float])* - inhomogeneous reflectivity.
            - *R_phi (ndarray[float])* - inhomogeneous reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - inhomogeneous transmission.
            - *T_phi (ndarray[float])* - inhomogeneous transmission for opposite
              magnetization.

        """
        # initialize
        M = np.size(strain_map, 0)  # delay steps
        R = np.zeros([M, np.size(self._qz, 0), np.size(self._qz, 1)])
        R_phi = np.zeros_like(R)
        T = np.zeros_like(R)
        T_phi = np.zeros_like(R)

        if self.progress_bar:
            iterator = trange(M, desc='Progress', leave=True)
        else:
            iterator = range(M)

        for i in iterator:
            # get the inhomogeneous reflectivity of the sample
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

            R[i, :, :], T[i, :, :] = XrayDynMag.calc_reflectivity_transmissivity_from_matrix(
                RT, self.pol_in, self.pol_out)
            R_phi[i, :, :], T_phi[i, :, :] = \
                XrayDynMag.calc_reflectivity_transmissivity_from_matrix(
                    RT_phi, self.pol_in, self.pol_out)

        return R, R_phi, T, T_phi

    def parallel_inhomogeneous_reflectivity(self, strain_map, magnetization_map, dask_client):
        """parallel_inhomogeneous_reflectivity

        Returns the reflectivity and transmission of an inhomogeneously strained
        sample structure for a given ``strain_map`` and ``magnetization_map`` in
        space and time. The function tries to parallelize the calculation over the
        time steps, since the results do not depend on each other.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            magnetization_map (ndarray[float]): spatio-temporal magnetization
                profile.
            dask_client (Dask.Client): Dask client.

        Returns:
            (tuple):
            - *R (ndarray[float])* - inhomogeneous reflectivity.
            - *R_phi (ndarray[float])* - inhomogeneous reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - inhomogeneous transmission.
            - *T_phi (ndarray[float])* - inhomogeneous transmission for opposite
              magnetization.

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
        T = np.zeros_like(R)
        T_phi = np.zeros_like(R)
        # vacuum boundary
        A0, A0_phi, _, _,  _, _, k_z_0 = self.get_atom_boundary_phase_matrix([], 0, 0)
        remote_A0 = dask_client.scatter(A0)
        remote_A0_phi = dask_client.scatter(A0_phi)
        remote_k_z_0 = dask_client.scatter(k_z_0)
        remote_pol_in = dask_client.scatter(self.pol_in)
        remote_pol_out = dask_client.scatter(self.pol_out)
        if self.S.substrate != []:
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
            if self.S.substrate != []:
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
            RTi = delayed(XrayDynMag.calc_reflectivity_transmissivity_from_matrix)(
                RT, remote_pol_in, remote_pol_out)
            RTi_phi = delayed(XrayDynMag.calc_reflectivity_transmissivity_from_matrix)(
                RT_phi, remote_pol_in, remote_pol_out)
            res.append(RTi[0])
            res.append(RTi_phi[0])
            res.append(RTi[1])
            res.append(RTi_phi[1])

        # compute results
        res = dask_client.compute(res, sync=True)

        # reorder results to reflectivity matrix
        for i in range(M):
            R[i, :, :] = res[4*i]
            R_phi[i, :, :] = res[4*i + 1]
            T[i, :, :] = res[4*i + 2]
            T_phi[i, :, :] = res[4*i + 3]

        return R, R_phi, T, T_phi

    def distributed_inhomogeneous_reflectivity(self, strain_map, magnetization_map,
                                               job, num_worker,):
        """distributed_inhomogeneous_reflectivity

        This is a stub. Not yet implemented in python.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            magnetization_map (ndarray[float]): spatio-temporal magnetization
                profile.
            job (Dask.job): Dask job.
            num_workers (int): Dask number of workers.

        Returns:
            (tuple):
            - *R (ndarray[float])* - inhomogeneous reflectivity.
            - *R_phi (ndarray[float])* - inhomogeneous reflectivity for opposite
              magnetization.

        """
        raise NotImplementedError

    def calc_inhomogeneous_matrix(self, last_A, last_A_phi, last_k_z, strains, magnetizations):
        r"""calc_inhomogeneous_matrix

        Calculates the product of all reflection-transmission matrices of the
        sample structure for every atomic layer.

        .. math:: RT = \prod_m \left( P_m A_m^{-1} A_{m-1}  \right)

        Args:
            last_A (ndarray[complex]): last atom boundary matrix.
            last_A_phi (ndarray[complex]): last atom boundary matrix for opposite
              magnetization.
            last_k_z (ndarray[float]): last internal wave vector
            strains (ndarray[float]): spatial strain profile for single time
                step.
            magnetizations (ndarray[float]): spatial magnetization profile for
                single time step.

        Returns:
            (tuple):
            - *RT (ndarray[complex])* - reflection-transmission matrix.
            - *RT_phi (ndarray[complex])* - reflection-transmission matrix for
              opposite magnetization.
            - *A (ndarray[complex])* - atom boundary matrix.
            - *A_phi (ndarray[complex])* - atom boundary matrix for opposite
              magnetization.
            - *A_inv (ndarray[complex])* - inverted atom boundary matrix.
            - *A_inv_phi (ndarray[complex])* - inverted atom boundary matrix for
              opposite magnetization.
            - *k_z (ndarray[float])* - internal wave vector.

        """
        L = self.S.get_number_of_layers()  # number of unit cells
        _, _, layer_handles = self.S.get_layer_vectors()
        # for inhomogeneous results we do not store results and force a re-calc
        force_recalc = True
        for i in range(L):
            layer = layer_handles[i]
            if isinstance(layer, UnitCell):
                RT_layer, RT_layer_phi, A, A_phi, A_inv, A_inv_phi, k_z = \
                    self.calc_uc_boundary_phase_matrix(
                        layer, last_A, last_A_phi, last_k_z, strains[i],
                        magnetizations[i], force_recalc)
            elif isinstance(layer, AmorphousLayer):
                A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = \
                    self.get_atom_boundary_phase_matrix(
                        layer.atom, layer._density/(strains[i]+1), layer._thickness*(strains[i]+1),
                        force_recalc, magnetizations[i])
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
                                      magnetization, force_recalc=False):
        r"""calc_uc_boundary_phase_matrix

        Calculates the product of all reflection-transmission matrices of
        a single unit cell for a given strain:

        .. math:: RT = \prod_m \left( P_m A_m^{-1} A_{m-1}\right)

        and returns also the last matrices :math:`A, A^{-1}, k_z`.

        Args:
            uc (UnitCell): unit cell
            last_A (ndarray[complex]): last atom boundary matrix.
            last_A_phi (ndarray[complex]): last atom boundary matrix for opposite
              magnetization.
            last_k_z (ndarray[float]): last internal wave vector
            strain (float): strain of unit cell for a single time
                step.
            magnetization (ndarray[float]): magnetization of unit cell for
                a single time step.
            force_recalc (boolean, optional): force recalculation of boundary
                phase matrix if True. Defaults to False.

        Returns:
            (tuple):
            - *RT (ndarray[complex])* - reflection-transmission matrix.
            - *RT_phi (ndarray[complex])* - reflection-transmission matrix for
              opposite magnetization.
            - *A (ndarray[complex])* - atom boundary matrix.
            - *A_phi (ndarray[complex])* - atom boundary matrix for opposite
              magnetization.
            - *A_inv (ndarray[complex])* - inverted atom boundary matrix.
            - *A_inv_phi (ndarray[complex])* - inverted atom boundary matrix for
              opposite magnetization.
            - *k_z (ndarray[float])* - internal wave vector.

        """
        K = uc.num_atoms  # number of atoms
        # force_recalc = True
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
                                                    force_recalc, magnetization)
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

    def get_atom_boundary_phase_matrix(self, atom, density, distance,
                                       force_recalc=False, *args):
        """get_atom_boundary_phase_matrix

        Returns the boundary and phase matrices of an atom from Elzo
        formalism [10]_. The results for a given atom, energy, :math:`q_z`,
        polarization, and magnetization are stored to RAM to avoid recalculation.

        Args:
            atom (Atom, AtomMixed): atom or mixed atom.
            density (float): density around the atom [kg/m³].
            distance (float): distance towards the next atomic [m].
            force_recalc (boolean, optional): force recalculation of boundary
                phase matrix if True. Defaults to False.
            args (ndarray[float]): magnetization vector.

        Returns:
            (tuple):
            - *A (ndarray[complex])* - atom boundary matrix.
            - *A_phi (ndarray[complex])* - atom boundary matrix for opposite
              magnetization.
            - *P (ndarray[complex])* - atom phase matrix.
            - *P_phi (ndarray[complex])* - atom phase matrix for opposite
              magnetization.
            - *A_inv (ndarray[complex])* - inverted atom boundary matrix.
            - *A_inv_phi (ndarray[complex])* - inverted atom boundary matrix for
              opposite magnetization.
            - *k_z (ndarray[float])* - internal wave vector.

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

        if force_recalc:
            # just calculate and and do not remember the results to save
            # computational time
            A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = \
                self.calc_atom_boundary_phase_matrix(atom, density, distance, *args)
        else:
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

        Calculates the boundary and phase matrices of an atom from Elzo
        formalism [10]_.

        Args:
            atom (Atom, AtomMixed): atom or mixed atom.
            density (float): density around the atom [kg/m³].
            distance (float): distance towards the next atomic [m].
            args (ndarray[float]): magnetization vector.

        Returns:
            (tuple):
            - *A (ndarray[complex])* - atom boundary matrix.
            - *A_phi (ndarray[complex])* - atom boundary matrix for opposite
              magnetization.
            - *P (ndarray[complex])* - atom phase matrix.
            - *P_phi (ndarray[complex])* - atom phase matrix for opposite
              magnetization.
            - *A_inv (ndarray[complex])* - inverted atom boundary matrix.
            - *A_inv_phi (ndarray[complex])* - inverted atom boundary matrix for
              opposite magnetization.
            - *k_z (ndarray[float])* - internal wave vector.

        """
        try:
            magnetization = args[0]
            mag_amplitude = magnetization[0]
            mag_phi = magnetization[1]
            mag_gamma = magnetization[2]
        except IndexError:
            # here we catch magnetizations with only one instead of three
            # elements
            try:
                mag_amplitude = atom.mag_amplitude
            except AttributeError:
                mag_amplitude = 0
            try:
                mag_phi = atom._mag_phi
            except AttributeError:
                mag_phi = 0
            try:
                mag_gamma = atom._mag_gamma
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
        A_phi[:, :, 3, 2] = alpha_z_left_up * n_left_up * A_phi[:, :, 0, 2]
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
    def calc_reflectivity_transmissivity_from_matrix(RT, pol_in, pol_out):
        """calc_reflectivity_transmissivity_from_matrix

        Calculates the actual reflectivity and transmissivity from the
        reflectivity-transmission matrix for a given incoming and analyzer
        polarization from Elzo formalism [10]_.

        Args:
            RT (ndarray[complex]): reflection-transmission matrix.
            pol_in (ndarray[complex]): incoming polarization factor.
            pol_out (ndarray[complex]): outgoing polarization factor.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity.
            - *T (ndarray[float])* - transmissivity.

        """

        Ref = np.tile(np.eye(2, 2, dtype=np.cfloat)[np.newaxis, np.newaxis, :, :],
                      (np.size(RT, 0), np.size(RT, 1), 1, 1))
        Trans = np.tile(np.eye(2, 2, dtype=np.cfloat)[np.newaxis, np.newaxis, :, :],
                        (np.size(RT, 0), np.size(RT, 1), 1, 1))

        d = np.divide(1, RT[:, :, 3, 3] * RT[:, :, 2, 2] - RT[:, :, 3, 2] * RT[:, :, 2, 3])
        Ref[:, :, 0, 0] = (-RT[:, :, 3, 3] * RT[:, :, 2, 0] + RT[:, :, 2, 3] * RT[:, :, 3, 0]) * d
        Ref[:, :, 0, 1] = (-RT[:, :, 3, 3] * RT[:, :, 2, 1] + RT[:, :, 2, 3] * RT[:, :, 3, 1]) * d
        Ref[:, :, 1, 0] = (RT[:, :, 3, 2] * RT[:, :, 2, 0] - RT[:, :, 2, 2] * RT[:, :, 3, 0]) * d
        Ref[:, :, 1, 1] = (RT[:, :, 3, 2] * RT[:, :, 2, 1] - RT[:, :, 2, 2] * RT[:, :, 3, 1]) * d

        Trans[:, :, 0, 0] = (RT[:, :, 0, 0] + RT[:, :, 0, 2] * Ref[:, :, 0, 0]
                             + RT[:, :, 0, 3] * Ref[:, :, 1, 0])
        Trans[:, :, 0, 1] = (RT[:, :, 0, 1] + RT[:, :, 0, 2] * Ref[:, :, 0, 1]
                             + RT[:, :, 0, 3] * Ref[:, :, 1, 1])
        Trans[:, :, 1, 0] = (RT[:, :, 1, 0] + RT[:, :, 1, 2] * Ref[:, :, 0, 0]
                             + RT[:, :, 1, 3] * Ref[:, :, 1, 0])
        Trans[:, :, 1, 1] = (RT[:, :, 1, 1] + RT[:, :, 1, 2] * Ref[:, :, 0, 1]
                             + RT[:, :, 1, 3] * Ref[:, :, 1, 1])

        Ref = np.matmul(np.matmul(np.array([[-1, 1], [-1j, -1j]]), Ref),
                        np.array([[-1, 1j], [1, 1j]])*0.5)
        Trans = np.matmul(np.matmul(np.array([[-1, 1], [-1j, -1j]]), Trans),
                          np.array([[-1, 1j], [1, 1j]])*0.5)

        if pol_out.size == 0:
            # no analyzer polarization
            R = np.real(np.matmul(np.square(np.absolute(np.matmul(Ref, pol_in))),
                        np.array([1, 1], dtype=np.cfloat)))
            T = np.real(np.matmul(np.square(np.absolute(np.matmul(Trans, pol_in))),
                        np.array([1, 1], dtype=np.cfloat)))
        else:
            R = np.real(np.square(np.absolute(np.matmul(np.matmul(Ref, pol_in), pol_out))))
            T = np.real(np.square(np.absolute(np.matmul(np.matmul(Trans, pol_in), pol_out))))

        return R, T

    @staticmethod
    def calc_kerr_effect_from_matrix(RT):
        """calc_kerr_effect_from_matrix

        Calculates the Kerr rotation and ellipticity for sigma and pi
        incident polarization from the reflectivity-transmission
        matrix independent of the given incoming and analyzer polarization
        from Elzo formalism [10]_.

        Args:
            RT (ndarray[complex]): reflection-transmission matrix.

        Returns:
            K (ndarray[float]): kerr.

        """

        raise NotImplementedError

    @staticmethod
    def calc_roughness_matrix(roughness, k_z, last_k_z):
        """calc_roughness_matrix

        Calculates the roughness matrix for an interface with a gaussian
        roughness for the Elzo formalism [10]_.

        Args:
            roughness (float): gaussian roughness of the interface [m].
            k_z (ndarray[float)]: internal wave vector.
            last_k_z (ndarray[float)]: last internal wave vector.

        Returns:
            W (ndarray[float]): roughness matrix.

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
