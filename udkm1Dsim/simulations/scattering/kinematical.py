#!/usr/bin/env python

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

"""Kinematical X-ray diffraction."""

__all__ = ["XrayKin"]

__docformat__ = "restructuredtext"

from time import time

import numpy as np
import pint
import scipy.constants as constants

from ...structures.layers import UnitCell, Vacuum
from ...structures.structure import Structure
from .scattering import Scattering

u = pint.get_application_registry()
Q_ = u.Quantity

r_0 = constants.physical_constants["classical electron radius"][0]


class XrayKin(Scattering):
    r"""XrayKin

    Kinetic X-ray scattering simulations following :cite:t:`warren1990`.

    The superstrate is neglected, and the substrate is only included if it
    is a :class:`.UnitCell`. Any other superstrate or substrate than
    :class:`.Vacuum` raises a warning, see
    :class:`~udkm1Dsim.simulations.scattering.elzo.Elzo` for homogeneous
    layers.

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

    _title = "Kinematical X-Ray Diffraction simulation"

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    @u.wraps(None, (None, "eV", "m**-1", None), strict=False)
    def get_uc_atomic_form_factors(self, energy, qz, uc):
        """get_uc_atomic_form_factors

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
            raise TypeError("Only scalars or Quantities are allowed for the energy!")
        f = np.zeros([uc.num_atoms, len(qz)], dtype=complex)
        for i in range(uc.num_atoms):
            f[i, :] = uc.atoms[i][0].get_cm_atomic_form_factor(energy, qz)
        return f

    @u.wraps(None, (None, "eV", "m**-1", None, None), strict=False)
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
            raise TypeError("Only scalars or Quantities for the energy are allowed!")

        if np.isscalar(qz):
            qz = np.array([qz])

        S = np.sum(
            self.get_uc_atomic_form_factors(energy, qz, uc)
            * np.exp(
                1j * uc.lattice.c_axis.magnitude * np.outer(uc.get_atom_positions(strain), qz)
            ),
            0,
        )
        return S

    def homogeneous_reflectivity(self, strains=0):
        r"""homogeneous_reflectivity

        Calculates the reflectivity :math:`R = E_p^t\,(E_p^t)^*` of a
        homogeneous sample structure as well as the reflected field
        :math:`E_p^N` of all substructures.

        Args:
            strains (ndarray[float], optional): strains of each sub-structure

        Returns:
            (tuple):
            - *R (ndarray[complex])* - homogeneous reflectivity.
            - *A (ndarray[complex])* - reflected fields of sub-structures.

        """
        self._warn_ignored_superstrate_and_substrate()
        if strains == 0:
            strains = np.zeros([self.S.get_number_of_sub_structures(), 1])

        t1 = time()
        self.disp_message("Calculating _homogenous_reflectivity_ ...")
        # get the reflected field of the structure for each energy
        R = np.zeros_like(self.experiment.qz.magnitude)
        for i, energy in enumerate(self.experiment.energy.magnitude):
            qz = self.experiment.qz.magnitude[i, :]
            theta = self.experiment.theta.magnitude[i, :]
            # a superstrate does not make too much sence here
            # calculate reflected field of the actual structure
            Ept, A = self.homogeneous_reflected_field(self.S, energy, qz, theta, strains)
            # add static substrate
            if isinstance(self.S.substrate[0], UnitCell):
                temp, temp2 = self.homogeneous_reflected_field(self.S.substrate, energy, qz, theta)
                A.append([temp2, "static substrate"])
                Ept = Ept + (temp * np.exp(1j * qz * self.S.thickness.to_base_units().magnitude))

            # calculate the real reflectivity from Ef
            R[i, :] = np.real(Ept * np.conj(Ept))
        self.disp_message(f"Elapsed time for _homogenous_reflectivity_: {time() - t1:f} s")
        return R, A

    @u.wraps((None, None), (None, None, "eV", "m**-1", "rad", None), strict=False)
    def homogeneous_reflected_field(self, S, energy, qz, theta, strains=0):
        r"""homogeneous_reflected_field

        Calculates the reflected field :math:`E_p^t` from a :class:`.UnitCell` or
        :class:Structure` as well as for each sub-structure (:math:`E_p^N`).
        The reflected wave field :math:`E_p` from a single :class:`.UnitCell` at
        the detector is calculated according to :cite:t:`warren1990`:

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
        if isinstance(S, Structure):
            sub_structures = S.sub_structures
        elif isinstance(S[0], (Vacuum, UnitCell)):
            sub_structures = [S]
        else:
            raise TypeError("XrayKin can only handle Layers of class UnitCell and Vacuum")

        # if no strains are given we assume no strain (1)
        if np.isscalar(strains) and strains == 0:
            try:
                strains = np.zeros([self.S.get_number_of_sub_structures(), 1])
            except AttributeError:
                # its a single Layer
                strains = np.array([0])

        N = len(qz)  # nb of qz
        Ept = np.zeros([1, N])  # total reflected field
        Z = 0  # total length of the substructure from the surface
        A = list([0, 2])  # cell matrix of reflected fields EpN of substructures
        strainCounter = 0  # the is the index of the strain vector if applied

        # traverse substructures
        for sub_structure in sub_structures:
            if isinstance(sub_structure[0], UnitCell):
                # the substructure is an unit cell and we can calculate
                # Ep directly
                Ep = self.get_Ep(energy, qz, theta, sub_structure[0], strains[strainCounter])
                z = sub_structure[0].lattice.c_axis.magnitude
                strainCounter = strainCounter + 1
            elif isinstance(sub_structure[0], Structure):
                # the substructure is a structure, so we do a recursive
                # call of this method
                d = sub_structure[0].get_number_of_sub_structures()
                Ep, temp = self.homogeneous_reflected_field(
                    sub_structure[0],
                    energy,
                    qz,
                    theta,
                    strains[strainCounter : (strainCounter + d)],
                )
                z = sub_structure[0].get_length().magnitude
                strainCounter = strainCounter + d
                A.append([temp, [sub_structure[0].name + " substructures"]])
                A.append([Ep, f"{1:d}x {sub_structure[0].name:s}"])
            else:
                raise TypeError("The substructure must be Vacuum, UnitCell or Structure type!")

            # calculate the interference function for N repetitions of
            # the substructure with the length z
            psi = self.get_interference_function(qz, z, sub_structure[1])
            # calculate the reflected field for N repetitions of
            # the substructure with the length z
            EpN = Ep * psi
            # remember the result
            A.append([EpN, f"{sub_structure[1]:d}x {sub_structure[0].name:s}"])
            # add the reflected field of the current substructure
            # phase-correct to the already calculated substructures
            Ept = Ept + (EpN * np.exp(1j * qz * Z))
            # update the total length $Z$ of the already calculated
            # substructures
            Z = Z + z * sub_structure[1]

        return Ept, A

    @u.wraps(None, (None, "m**-1", "m", None), strict=False)
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
        psi = (1 - np.exp(1j * qz * z * N)) / (1 - np.exp(1j * qz * z))
        return psi

    @u.wraps(None, (None, "eV", "m**-1", "rad", None, None), strict=False)
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

        Ep = (
            1j
            / c.epsilon_0
            * c.elementary_charge**2
            / c.electron_mass
            / c.c**2
            * (
                self.get_polarization_factor(theta)
                * self.get_uc_structure_factor(energy, qz, uc, strain)
                / uc.structural.area.magnitude
            )
            / qz
        )
        return Ep
