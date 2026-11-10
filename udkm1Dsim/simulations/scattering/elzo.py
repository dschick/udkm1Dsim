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

"""Dynamical magnetic scattering by the formalism of Elzo et al."""

__all__ = ["Elzo", "elzo_modes", "phase_matrix"]

__docformat__ = "restructuredtext"

from os import path
from time import time

import numpy as np
import pint
from tqdm.auto import trange

from ...helpers import m_power_x, m_times_n, make_hash_md5
from ...structures.layers import Layer
from ...structures.structure import Structure
from .multilayer import MultilayerScattering

u = pint.get_application_registry()
Q_ = u.Quantity


class Elzo(MultilayerScattering):
    r"""Elzo

    Dynamical magnetic scattering simulations by the formalism of
    :cite:t:`elzo2012` with the analytical eigenmodes of a dielectric tensor,
    which is isotropic except for its gyrotropic part to first order, and the
    assembly of the stack by transfer matrices.
    A :class:`.UnitCell` is treated as homogeneous layer with its dielectric
    tensor as all other layers, e.g. from the average of its atoms by
    :meth:`.Layer.set_dielectric_from_atoms`, so that it does not show Bragg
    peaks, see :class:`~udkm1Dsim.simulations.scattering.dynamical.XrayDyn`.

    .. deprecated:: 3.0.0
        The former name ``XrayDynMag`` was removed, use :class:`Elzo` instead.

    Adapted from :cite:t:`elzo2012` and initially realized in `Project Dyna
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
        last_layer_ref_trans_matrices (list): remember last result of
           layer ref_trans_matrices to speed up calculation.

    """

    _title = "Elzo multilayer scattering simulation"
    _cache_attributes = ("last_layer_ref_trans_matrices",)
    _assembly = "transfer"

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.last_layer_ref_trans_matrices = {
            "layer_ids": [],
            "hashes": [],
            "A": [],
            "A_phi": [],
            "P": [],
            "P_phi": [],
            "A_inv": [],
            "A_inv_phi": [],
            "k_z": [],
        }

    def calc_modes(self, eps):
        """calc_modes

        Calculates the analytical eigenmodes of a homogeneous layer with the
        dielectric tensor ``eps``, see :func:`elzo_modes`.

        Args:
            eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3).

        Returns:
            (tuple):
            - *A (ndarray[complex])* - boundary matrix of shape (M, N, 4, 4).
            - *A_phi (ndarray[complex])* - boundary matrix for opposite
              magnetization.
            - *q (ndarray[complex])* - normal wave-vector components of the
              modes of shape (M, N, 4).
            - *q_phi (ndarray[complex])* - normal wave-vector components for
              opposite magnetization.
            - *k_z (ndarray[complex])* - normal wave-vector component of the
              non-magnetic layer of shape (M, N).

        """
        return elzo_modes(eps, np.cos(self.experiment.theta.magnitude))

    def homogeneous_jones_matrices(self, *args, polar=False):
        """homogeneous_jones_matrices

        Calculates the Jones matrices of the reflection and transmission of
        the whole sample structure in the linear :math:`(\\sigma, \\pi)` basis
        allowing only for homogeneous strain and magnetization, see
        :meth:`homogeneous_reflectivity`.

        Args:
            args (ndarray[float], optional): strains and magnetization for each
                sub-structure. The magnetization is given as :math:`L \times 3`
                array in cartesian coordinates ``(x, y, z)`` by default, see
                :meth:`inhomogeneous_reflectivity`.
            polar (bool, optional): the magnetization is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad.
                Defaults to False.

        Returns:
            (tuple):
            - *R (ndarray[complex])* - Jones matrix of the reflection of shape
              (M, N, 2, 2).
            - *R_phi (ndarray[complex])* - Jones matrix of the reflection for
              opposite magnetization.
            - *T (ndarray[complex])* - Jones matrix of the transmission.
            - *T_phi (ndarray[complex])* - Jones matrix of the transmission for
              opposite magnetization.

        """
        if len(args) > 1:
            args = (args[0], self.parse_magnetization_map(args[1], polar), *args[2:])
        RT, RT_phi = self.calc_homogeneous_transfer_matrix(*args)
        eps = self._substrate_dielectric_tensor()
        R, T = self._jones_of_fields(RT, eps)
        R_phi, T_phi = self._jones_of_fields(RT_phi, eps)
        return R, R_phi, T, T_phi

    def _substrate_dielectric_tensor(self):
        """dielectric tensor of the last layer of the substrate"""
        self._check_superstrate_and_substrate()
        return self.get_layer_dielectric_tensor(self.S.substrate[0])

    def _jones_of_fields(self, RT, eps=None):
        """Jones matrices of the reflection and transmission of the electric
        fields from the transfer matrix, see :meth:`transmitted_field` and
        :meth:`incident_field`"""
        if eps is None:
            eps = self._substrate_dielectric_tensor()
        Ref, Trans = self.jones_from_transfer_matrix(RT)
        Trans = self.transmitted_field(Trans, eps)
        return self.incident_field(
            Ref, Trans, self.get_layer_dielectric_tensor(self.S.superstrate[0])
        )

    def calc_intensities_from_matrix(self, RT, eps=None):
        """calc_intensities_from_matrix

        Calculates the reflectivity and transmissivity from the transfer
        matrix for the incoming and analyzer polarization with the
        transmission of the electric field, see :meth:`transmitted_field` and
        :meth:`incident_field`.

        Args:
            RT (ndarray[complex]): transfer matrix.
            eps (ndarray[complex], optional): dielectric tensor of the last
                layer of the substrate. Defaults to the one of the substrate.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity.
            - *T (ndarray[float])* - transmissivity.

        """
        Ref, Trans = self._jones_of_fields(RT, eps)
        return self.calc_intensities(Ref, Trans, self.pol_in, self.pol_out)

    def homogeneous_reflectivity(self, *args, polar=False):
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
                sub-structure. The magnetization is given as :math:`L \times 3`
                array in cartesian coordinates ``(x, y, z)`` by default, see
                :meth:`inhomogeneous_reflectivity`.
            polar (bool, optional): the magnetization is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad.
                Defaults to False.

        Returns:
            (tuple):
            - *R (ndarray[float])* - homogeneous reflectivity.
            - *R_phi (ndarray[float])* - homogeneous reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - homogeneous transmissivity.
            - *T_phi (ndarray[float])* - homogeneous transmissivity for
              opposite magnetization.

        """
        t1 = time()
        self.disp_message("Calculating _homogeneous_reflectivity_ ...")
        if len(args) > 1:
            args = (args[0], self.parse_magnetization_map(args[1], polar), *args[2:])
        RT, RT_phi = self.calc_homogeneous_transfer_matrix(*args)

        # calc the actual reflectivity and transmissivity from the matrix
        eps = self._substrate_dielectric_tensor()
        R, T = self.calc_intensities_from_matrix(RT, eps)
        R_phi, T_phi = self.calc_intensities_from_matrix(RT_phi, eps)
        self.disp_message(f"Elapsed time for _homogeneous_reflectivity_: {time() - t1:f} s")
        return R, R_phi, T, T_phi

    def calc_homogeneous_transfer_matrix(self, *args):
        """calc_homogeneous_transfer_matrix

        Calculates the transfer matrix of the whole sample structure from the
        superstrate to the substrate allowing only for homogeneous strain and
        magnetization.

        Args:
            args (ndarray[float], optional): strains and magnetization for each
                sub-structure.

        Returns:
            (tuple):
            - *RT (ndarray[complex])* - transfer matrix.
            - *RT_phi (ndarray[complex])* - transfer matrix for opposite
              magnetization.

        """
        # add superstrate
        self._check_superstrate_and_substrate()
        A0, A0_phi, _, _, _, _, k_z_0 = self.get_layer_boundary_phase_matrix(self.S.superstrate[0])

        # calc the reflectivity-transmission matrix of the structure
        # and the inverse of the last boundary matrix
        RT, RT_phi, last_A, last_A_phi, last_A_inv, last_A_inv_phi, last_k_z = (
            self.calc_homogeneous_matrix(self.S, A0, A0_phi, k_z_0, *args)
        )

        # add substrate
        RT_sub, RT_sub_phi, last_A, last_A_phi, last_A_inv, last_A_inv_phi, _ = (
            self.calc_homogeneous_matrix(self.S.substrate, last_A, last_A_phi, last_k_z)
        )
        RT = m_times_n(RT_sub, RT)
        RT_phi = m_times_n(RT_sub_phi, RT_phi)

        # multiply the result of the structure with the boundary matrix
        # of the substrate
        RT = m_times_n(last_A_inv, m_times_n(last_A, RT))
        RT_phi = m_times_n(last_A_inv_phi, m_times_n(last_A_phi, RT_phi))
        return RT, RT_phi

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
            last_A (ndarray[complex]): last layer boundary matrix.
            last_A_phi (ndarray[complex]): last layer boundary matrix for opposite
                magnetization.
            last_k_z (ndarray[float]): last internal wave vector
            args (ndarray[float], optional): strains and magnetization for each
                sub-structure.

        Return:
            (tuple):
            - *RT (ndarray[complex])* - reflection-transmission matrix.
            - *RT_phi (ndarray[complex])* - reflection-transmission matrix for
              opposite magnetization.
            - *A (ndarray[complex])* - layer boundary matrix.
            - *A_phi (ndarray[complex])* - layer boundary matrix for opposite
              magnetization.
            - *A_inv (ndarray[complex])* - inverted layer boundary matrix.
            - *A_inv_phi (ndarray[complex])* - inverted layer boundary matrix for
              opposite magnetization.
            - *k_z (ndarray[float])* - internal wave vector.

        """
        if isinstance(S, Structure):
            sub_structures = S.sub_structures
            L = S.get_number_of_sub_structures()
        elif isinstance(S[0], Layer):
            sub_structures = [S]
            L = 1
        else:
            raise TypeError(
                "Elzo can only handle Layers, e.g. AmorphousLayer, UnitCell, and Vacuum"
            )

        # if no strains are given we assume no strain (1)
        if len(args) == 0:
            strains = np.zeros([L, 1])
        else:
            strains = args[0]

        if len(args) < 2:
            # create non-working magnetizations
            magnetizations = np.zeros([L, 1])
        else:
            magnetizations = args[1]

        layer_counter = 0
        # traverse substructures
        for i, sub_structure in enumerate(sub_structures):
            layer = sub_structure[0]
            repetitions = sub_structure[1]
            if isinstance(layer, Layer):
                # calculate the ref-trans matrices for N homogeneous layers
                A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = self.get_layer_boundary_phase_matrix(
                    layer, strains[layer_counter], magnetizations[layer_counter]
                )

                roughness = layer.structural.roughness.magnitude
                F = m_times_n(A_inv, last_A)
                F_phi = m_times_n(A_inv_phi, last_A_phi)
                if roughness > 0:
                    W = Elzo.calc_roughness_matrix(roughness, k_z, last_k_z)
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
                    temp = m_times_n(m_power_x(RT_amorph, repetitions - 1), temp)
                    temp_phi = m_times_n(m_power_x(RT_amorph_phi, repetitions - 1), temp_phi)
                layer_counter += 1
            else:
                # its a structure
                # make a recursive call
                temp, temp_phi, A, A_phi, A_inv, A_inv_phi, k_z = self.calc_homogeneous_matrix(
                    layer,
                    last_A,
                    last_A_phi,
                    last_k_z,
                    strains[
                        layer_counter : (layer_counter + layer.get_number_of_sub_structures())
                    ],
                    magnetizations[
                        layer_counter : (layer_counter + layer.get_number_of_sub_structures())
                    ],
                )
                # calculate the ref-trans matrices for N sub structures
                if repetitions > 1:
                    # use m_power_x for more than one repetition
                    temp2, temp2_phi, A, A_phi, A_inv, A_inv_phi, k_z = (
                        self.calc_homogeneous_matrix(
                            layer,
                            A,
                            A_phi,
                            k_z,
                            strains[
                                layer_counter : (
                                    layer_counter + layer.get_number_of_sub_structures()
                                )
                            ],
                            magnetizations[
                                layer_counter : (
                                    layer_counter + layer.get_number_of_sub_structures()
                                )
                            ],
                        )
                    )

                    temp = m_times_n(m_power_x(temp2, repetitions - 1), temp)
                    temp_phi = m_times_n(m_power_x(temp2_phi, repetitions - 1), temp_phi)

                layer_counter = layer_counter + layer.get_number_of_sub_structures()

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

    def inhomogeneous_reflectivity(
        self, strain_map=np.array([]), magnetization_map=np.array([]), polar=False, **kwargs
    ):
        r"""inhomogeneous_reflectivity

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
                magnetization profile :math:`M \times N \times 3` with
                :math:`M` delays and :math:`N` layers. The magnetization is
                given in cartesian coordinates ``(x, y, z)`` by default, as
                returned by
                :meth:`~udkm1Dsim.simulations.magnetization.Magnetization.get_magnetization_map`.
                Its amplitude scales the magnetic part of the dielectric tensor.
            polar (bool, optional): the ``magnetization_map`` is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad,
                as before version 2.6.1. Defaults to False.
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
        if not isinstance(magnetization_map, np.ndarray):
            raise TypeError("magnetization_map must be a numpy ndarray!")
        magnetization_map = self.parse_magnetization_map(magnetization_map, polar)
        # create a hash of all simulation parameters
        filename = (
            "inhomogeneous_reflectivity_dynMag_"
            + self.get_hash(strain_map=strain_map, magnetization_map=magnetization_map)
            + ".npz"
        )
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            R = tmp["R"]
            R_phi = tmp["R_phi"]
            T = tmp["T"]
            T_phi = tmp["T_phi"]
            self.disp_message("_inhomogeneous_reflectivity_ loaded from file:\n\t" + filename)
        else:
            t1 = time()
            self.disp_message("Calculating _inhomogeneous_reflectivity_ ...")
            # parse the input arguments
            if not isinstance(strain_map, np.ndarray):
                raise TypeError("strain_map must be a numpy ndarray!")

            dask_client = kwargs.get("dask_client", [])
            calc_type = kwargs.get("calc_type", "sequential")
            if calc_type not in ["parallel", "sequential", "distributed"]:
                raise TypeError(
                    "calc_type must be either _parallel_, _sequential_, or _distributed_!"
                )
            job = kwargs.get("job")
            num_workers = kwargs.get("num_workers", 1)

            M = np.size(strain_map, 0)
            N = np.size(magnetization_map, 0)

            if (M == 0) and (N > 0):
                strain_map = np.zeros(
                    [np.size(magnetization_map, 0), np.size(magnetization_map, 1)]
                )
            elif (M > 0) and (N == 0):
                magnetization_map = np.zeros_like(strain_map)
            elif (M == 0) and (N == 0):
                raise ValueError("At least a strain_map or magnetzation_map must be given!")
            else:
                if M != N:
                    raise ValueError(
                        "The strain_map and magnetzation_map must "
                        "have the same number of delay steps!"
                    )

            # select the type of computation
            if calc_type == "parallel":
                R, R_phi, T, T_phi = self.parallel_inhomogeneous_reflectivity(
                    strain_map, magnetization_map, dask_client
                )
            elif calc_type == "distributed":
                R, R_phi, T, T_phi = self.distributed_inhomogeneous_reflectivity(
                    strain_map, magnetization_map, job, num_workers
                )
            else:  # sequential
                R, R_phi, T, T_phi = self.sequential_inhomogeneous_reflectivity(
                    strain_map, magnetization_map
                )

            self.disp_message(f"Elapsed time for _inhomogeneous_reflectivity_: {time() - t1:f} s")
            self.save(
                full_filename,
                {"R": R, "R_phi": R_phi, "T": T, "T_phi": T_phi},
                "_inhomogeneous_reflectivity_",
            )
        return R, R_phi, T, T_phi

    def sequential_inhomogeneous_reflectivity(self, strain_map, magnetization_map):
        """sequential_inhomogeneous_reflectivity

        Returns the reflectivity and transmission of an inhomogeneously strained
        sample structure for a given ``strain_map`` and ``magnetization_map`` in
        space and time. The function calculates the results sequentially for every
        layer without parallelization.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            magnetization_map (ndarray[float]): cartesian spatio-temporal
                magnetization profile.

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
        num_pol = self.num_pol
        shape = (M,) + self.experiment.qz.magnitude.shape + ((num_pol,) if num_pol > 1 else ())
        R = np.zeros(shape)
        R_phi = np.zeros_like(R)
        T = np.zeros_like(R)
        T_phi = np.zeros_like(R)

        if self.progress_bar:
            iterator = trange(M, desc="Progress", leave=True)
        else:
            iterator = range(M)

        eps = self._substrate_dielectric_tensor()
        for i in iterator:
            # get the inhomogeneous reflectivity of the sample
            # structure for each time step of the strain map

            # add superstrate
            self._check_superstrate_and_substrate()
            A0, A0_phi, _, _, _, _, k_z_0 = self.get_layer_boundary_phase_matrix(
                self.S.superstrate[0]
            )

            RT, RT_phi, last_A, last_A_phi, last_A_inv, last_A_inv_phi, last_k_z = (
                self.calc_inhomogeneous_matrix(
                    A0, A0_phi, k_z_0, strain_map[i, :], magnetization_map[i, :]
                )
            )

            # add substrate
            RT_sub, RT_sub_phi, last_A, last_A_phi, last_A_inv, last_A_inv_phi, _ = (
                self.calc_homogeneous_matrix(self.S.substrate, last_A, last_A_phi, last_k_z)
            )
            RT = m_times_n(RT_sub, RT)
            RT_phi = m_times_n(RT_sub_phi, RT_phi)

            # multiply vacuum and last layer
            RT = m_times_n(last_A_inv, m_times_n(last_A, RT))
            RT_phi = m_times_n(last_A_inv_phi, m_times_n(last_A_phi, RT_phi))

            R[i, :, :], T[i, :, :] = self.calc_intensities_from_matrix(RT, eps)
            R_phi[i, :, :], T_phi[i, :, :] = self.calc_intensities_from_matrix(RT_phi, eps)

        return R, R_phi, T, T_phi

    def parallel_inhomogeneous_reflectivity(self, strain_map, magnetization_map, dask_client):
        """parallel_inhomogeneous_reflectivity

        Returns the reflectivity and transmission of an inhomogeneously strained
        sample structure for a given ``strain_map`` and ``magnetization_map`` in
        space and time. The function tries to parallelize the calculation over the
        time steps, since the results do not depend on each other.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            magnetization_map (ndarray[float]): cartesian spatio-temporal
                magnetization profile.
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
            raise ValueError("no dask client set")
        # initialize
        M = np.size(strain_map, 0)  # delay steps
        num_pol = self.num_pol
        shape = (M,) + self.experiment.qz.magnitude.shape + ((num_pol,) if num_pol > 1 else ())

        R = np.zeros(shape)
        R_phi = np.zeros_like(R)
        T = np.zeros_like(R)
        T_phi = np.zeros_like(R)

        # add superstrate
        self._check_superstrate_and_substrate()
        A0, A0_phi, _, _, _, _, k_z_0 = self.get_layer_boundary_phase_matrix(self.S.superstrate[0])

        eps = self._substrate_dielectric_tensor()
        boundary = dask_client.scatter((A0, A0_phi, k_z_0, eps), hash=False)
        results = self._compute_in_parallel(
            dask_client,
            Elzo._parallel_inhomogeneous_delay,
            [(boundary, strain_map[i, :], magnetization_map[i, :]) for i in range(M)],
        )
        for i, result in enumerate(results):
            R[i, ...], R_phi[i, ...], T[i, ...], T_phi[i, ...] = result

        return R, R_phi, T, T_phi

    @staticmethod
    def _parallel_inhomogeneous_delay(sim, boundary, strains, magnetizations):
        """Reflectivity and transmissivity of a single delay on a dask worker."""
        A0, A0_phi, k_z_0, eps = boundary
        RT, RT_phi, A, A_phi, A_inv, A_inv_phi, k_z = sim.calc_inhomogeneous_matrix(
            A0, A0_phi, k_z_0, strains, magnetizations
        )
        RT_sub, RT_sub_phi, A, A_phi, A_inv, A_inv_phi, _ = sim.calc_homogeneous_matrix(
            sim.S.substrate, A, A_phi, k_z
        )
        RT = m_times_n(A_inv, m_times_n(A, m_times_n(RT_sub, RT)))
        RT_phi = m_times_n(A_inv_phi, m_times_n(A_phi, m_times_n(RT_sub_phi, RT_phi)))
        R, T = sim.calc_intensities_from_matrix(RT, eps)
        R_phi, T_phi = sim.calc_intensities_from_matrix(RT_phi, eps)
        return R, R_phi, T, T_phi

    def distributed_inhomogeneous_reflectivity(
        self,
        strain_map,
        magnetization_map,
        job,
        num_worker,
    ):
        """distributed_inhomogeneous_reflectivity

        This is a stub. Not yet implemented in python.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            magnetization_map (ndarray[float]): cartesian spatio-temporal
                magnetization profile.
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
        sample structure for every layer.

        .. math:: RT = \prod_m \left( P_m A_m^{-1} A_{m-1}  \right)

        Args:
            last_A (ndarray[complex]): last layer boundary matrix.
            last_A_phi (ndarray[complex]): last layer boundary matrix for opposite
              magnetization.
            last_k_z (ndarray[float]): last internal wave vector
            strains (ndarray[float]): spatial strain profile for single time
                step.
            magnetizations (ndarray[float]): cartesian spatial magnetization
                profile for single time step.

        Returns:
            (tuple):
            - *RT (ndarray[complex])* - reflection-transmission matrix.
            - *RT_phi (ndarray[complex])* - reflection-transmission matrix for
              opposite magnetization.
            - *A (ndarray[complex])* - layer boundary matrix.
            - *A_phi (ndarray[complex])* - layer boundary matrix for opposite
              magnetization.
            - *A_inv (ndarray[complex])* - inverted layer boundary matrix.
            - *A_inv_phi (ndarray[complex])* - inverted layer boundary matrix for
              opposite magnetization.
            - *k_z (ndarray[float])* - internal wave vector.

        """
        L = self.S.get_number_of_layers()  # number of unit cells
        _, _, layer_handles = self.S.get_layer_vectors()
        # for inhomogeneous results we do not store results and force a re-calc
        force_recalc = True
        for i in range(L):
            layer = layer_handles[i]
            if isinstance(layer, Layer):
                A, A_phi, P, P_phi, A_inv, A_inv_phi, k_z = self.get_layer_boundary_phase_matrix(
                    layer, strains[i], magnetizations[i], force_recalc
                )
                roughness = layer.structural.roughness.magnitude
                F = m_times_n(A_inv, last_A)
                F_phi = m_times_n(A_inv_phi, last_A_phi)
                if roughness > 0:
                    W = Elzo.calc_roughness_matrix(roughness, k_z, last_k_z)
                    F = F * W
                    F_phi = F_phi * W
                RT_layer = m_times_n(P, F)
                RT_layer_phi = m_times_n(P_phi, F_phi)
            else:
                raise TypeError("All layers must be Layers!")
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

    def get_layer_boundary_phase_matrix(
        self, layer, strain=0.0, magnetization=None, force_recalc=False
    ):
        """get_layer_boundary_phase_matrix

        Returns the boundary and phase matrices of a homogeneous layer, e.g.
        an :class:`.AmorphousLayer`, from its dielectric tensor, see
        :meth:`.Layer.get_dielectric_tensor`. The results for a given layer,
        energy, :math:`q_z`, polarization, strain, and magnetization are
        stored to RAM to avoid recalculation.

        Args:
            layer (Layer): homogeneous layer with an isotropic dielectric tensor.
            strain (float, optional): strain of the layer. Defaults to 0.
            magnetization (ndarray[float], optional): cartesian
                magnetization ``(x, y, z)``. Defaults to the magnetization of
                the layer.
            force_recalc (boolean, optional): force recalculation of boundary
                phase matrix if True. Defaults to False.

        Returns:
            (tuple):
            - *A (ndarray[complex])* - boundary matrix.
            - *A_phi (ndarray[complex])* - boundary matrix for opposite
              magnetization.
            - *P (ndarray[complex])* - phase matrix.
            - *P_phi (ndarray[complex])* - phase matrix for opposite
              magnetization.
            - *A_inv (ndarray[complex])* - inverted boundary matrix.
            - *A_inv_phi (ndarray[complex])* - inverted boundary matrix for
              opposite magnetization.
            - *k_z (ndarray[float])* - internal wave vector.

        """
        dielectric = layer.optical.dielectric
        if dielectric.is_set and not dielectric.is_isotropic:
            raise ValueError(
                f"The formalism of Elzo requires an isotropic dielectric tensor, but the "
                f"layer '{layer.id}' is anisotropic!"
            )
        strain = float(np.asarray(strain).item())
        if magnetization is None or np.size(magnetization) < 3:
            magnetization = layer._magnetization_for_dielectric()
        magnetization = tuple(float(m) for m in magnetization)
        distance = layer.structural.thickness.magnitude * (strain + 1)
        if force_recalc:
            eps = layer.get_dielectric_tensor(
                self.experiment.energy.magnitude, strain, magnetization
            )
            return self.calc_boundary_phase_matrix(eps, distance)

        key = layer.id
        _hash = make_hash_md5(
            [
                self.experiment.energy.magnitude,
                self.experiment.qz.magnitude,
                self.pol_in,
                self.pol_out,
                repr(dielectric),
                layer.structural.density.magnitude,
                distance,
                strain,
                *magnetization,
            ]
        )
        cache = self.last_layer_ref_trans_matrices
        try:
            index = cache["layer_ids"].index(key)
        except ValueError:
            index = -1
        names = ["A", "A_phi", "P", "P_phi", "A_inv", "A_inv_phi", "k_z"]
        if index >= 0 and cache["hashes"][index] == _hash:
            return tuple(cache[name][index] for name in names)

        eps = layer.get_dielectric_tensor(self.experiment.energy.magnitude, strain, magnetization)
        result = self.calc_boundary_phase_matrix(eps, distance)
        if index >= 0:
            cache["hashes"][index] = _hash
            for name, value in zip(names, result):
                cache[name][index] = value
        else:
            cache["layer_ids"].append(key)
            cache["hashes"].append(_hash)
            for name, value in zip(names, result):
                cache[name].append(value)
        return result

    def calc_boundary_phase_matrix(self, eps, distance):
        """calc_boundary_phase_matrix

        Calculates the boundary and phase matrices of a homogeneous slab with
        the dielectric tensor ``eps`` from :cite:t:`elzo2012`, which is
        isotropic except for the gyrotropic, magnetic contributions, see
        :func:`elzo_modes`.

        Args:
            eps (ndarray[complex]): dielectric tensor per energy of shape
                (M, 3, 3).
            distance (float): thickness of the slab (m).

        Returns:
            (tuple):
            - *A (ndarray[complex])* - boundary matrix.
            - *A_phi (ndarray[complex])* - boundary matrix for opposite
              magnetization.
            - *P (ndarray[complex])* - phase matrix.
            - *P_phi (ndarray[complex])* - phase matrix for opposite
              magnetization.
            - *A_inv (ndarray[complex])* - inverted boundary matrix.
            - *A_inv_phi (ndarray[complex])* - inverted boundary matrix for
              opposite magnetization.
            - *k_z (ndarray[float])* - internal wave vector.

        """
        k = self.experiment.k.magnitude[:, np.newaxis]
        A, A_phi, q, q_phi, k_z = elzo_modes(eps, np.cos(self.experiment.theta.magnitude))
        P = phase_matrix(q, k * distance)
        P_phi = phase_matrix(q_phi, k * distance)
        return A, A_phi, P, P_phi, np.linalg.inv(A), np.linalg.inv(A_phi), k * k_z


def phase_matrix(q, phase):
    r"""phase_matrix

    Diagonal propagation matrix :math:`P = \mathrm{diag}(e^{i k d q_j})` of the
    four eigenmodes of a slab with the normal wave-vector components
    :math:`q_j = k_{z,j} / k` and the phase :math:`k d`.

    Args:
        q (ndarray[complex]): normal wave-vector components of shape
            (M, N, 4).
        phase (ndarray[float]): phase :math:`k d` broadcastable to (M, N).

    Returns:
        P (ndarray[complex]): propagation matrix of shape (M, N, 4, 4).

    """
    P = np.zeros(q.shape + (4,), dtype=np.complex128)
    for j in range(4):
        P[..., j, j] = np.exp(1j * phase * q[..., j])
    return P


def elzo_modes(eps, cos_theta):
    r"""elzo_modes

    Analytical eigenmodes of a dielectric tensor :math:`\varepsilon`, which is
    isotropic except for its gyrotropic part, from :cite:t:`elzo2012`. The four
    circularly polarized modes are ordered as right and left down, i.e. into
    the sample, and right and left up. The boundary matrix holds the in-plane
    electric and magnetic fields of the modes.

    The inverse permittivity is expanded to first order in the off-diagonal
    part :math:`\varepsilon_A` of :math:`\varepsilon = \varepsilon_{00} I +
    \varepsilon_A`, i.e. :math:`\varepsilon^{-1} = (I - \varepsilon_A /
    \varepsilon_{00}) / \varepsilon_{00}`. Eq. (20) of :cite:t:`elzo2012`
    omits the division by :math:`\varepsilon_{00}`, which is exact only for
    :math:`\varepsilon_{00} = 1`, as found and corrected in Dyna after version 6.13.1.

    Args:
        eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3) or
            (M, N, 3, 3).
        cos_theta (ndarray[float]): cosine of the grazing incidence angles
            :math:`\theta` of shape (M, N), i.e. the conserved in-plane
            wave-vector component :math:`k_y / k`.

    Returns:
        (tuple):
        - *A (ndarray[complex])* - boundary matrix of shape (M, N, 4, 4).
        - *A_phi (ndarray[complex])* - boundary matrix for opposite
          magnetization.
        - *q (ndarray[complex])* - normal wave-vector components
          :math:`k_{z,j} / k` of the modes of shape (M, N, 4).
        - *q_phi (ndarray[complex])* - normal wave-vector components for
          opposite magnetization.
        - *k_z (ndarray[complex])* - normal wave-vector component
          :math:`k_z / k` of the non-magnetic slab of shape (M, N).

    """
    M, N = np.shape(cos_theta)
    if eps.ndim == 3:
        # the dielectric tensor does not depend on q_z
        eps = np.broadcast_to(eps[:, np.newaxis, :, :], (M, N, 3, 3))
    e00 = eps[:, :, 0, 0]
    e01 = eps[:, :, 0, 1]
    e02 = eps[:, :, 0, 2]

    alpha_y = np.divide(cos_theta, np.sqrt(e00))
    alpha_z = np.sqrt(1 - alpha_y**2)
    k_z = np.sqrt(e00) * alpha_z

    n_rd = np.sqrt(e00 - 1j * e02 * alpha_y - 1j * e01 * alpha_z)
    n_ld = np.sqrt(e00 + 1j * e02 * alpha_y + 1j * e01 * alpha_z)
    n_ru = np.sqrt(e00 - 1j * e02 * alpha_y + 1j * e01 * alpha_z)
    n_lu = np.sqrt(e00 + 1j * e02 * alpha_y - 1j * e01 * alpha_z)

    ay_rd = cos_theta / n_rd
    az_rd = np.sqrt(1 - ay_rd**2)
    ay_ld = cos_theta / n_ld
    az_ld = np.sqrt(1 - ay_ld**2)
    ay_ru = cos_theta / n_ru
    az_ru = np.sqrt(1 - ay_ru**2)
    ay_lu = cos_theta / n_lu
    az_lu = np.sqrt(1 - ay_lu**2)

    # first-order inverse permittivity, see above
    e01 = e01 / e00
    e02 = e02 / e00
    e12 = eps[:, :, 1, 2] / e00

    A = np.zeros([M, N, 4, 4], dtype=np.complex128)
    A_phi = np.zeros_like(A)

    A[:, :, 0, 0] = -1 - 1j * e01 * az_rd - 1j * e02 * ay_rd
    A[:, :, 0, 1] = 1 - 1j * e01 * az_ld - 1j * e02 * ay_ld
    A[:, :, 0, 2] = -1 + 1j * e01 * az_ru - 1j * e02 * ay_ru
    A[:, :, 0, 3] = 1 + 1j * e01 * az_lu - 1j * e02 * ay_lu

    A[:, :, 1, 0] = 1j * az_rd - e01 - 1j * e12 * ay_rd
    A[:, :, 1, 1] = 1j * az_ld + e01 - 1j * e12 * ay_ld
    A[:, :, 1, 2] = -1j * az_ru - e01 - 1j * e12 * ay_ru
    A[:, :, 1, 3] = -1j * az_lu + e01 - 1j * e12 * ay_lu

    A[:, :, 2, 0] = -1j * n_rd * A[:, :, 0, 0]
    A[:, :, 2, 1] = 1j * n_ld * A[:, :, 0, 1]
    A[:, :, 2, 2] = -1j * n_ru * A[:, :, 0, 2]
    A[:, :, 2, 3] = 1j * n_lu * A[:, :, 0, 3]

    A[:, :, 3, 0] = -az_rd * n_rd * A[:, :, 0, 0]
    A[:, :, 3, 1] = -az_ld * n_ld * A[:, :, 0, 1]
    A[:, :, 3, 2] = az_ru * n_ru * A[:, :, 0, 2]
    A[:, :, 3, 3] = az_lu * n_lu * A[:, :, 0, 3]

    A_phi[:, :, 0, 0] = -1 + 1j * e01 * az_ld + 1j * e02 * ay_ld
    A_phi[:, :, 0, 1] = 1 + 1j * e01 * az_rd + 1j * e02 * ay_rd
    A_phi[:, :, 0, 2] = -1 - 1j * e01 * az_lu + 1j * e02 * ay_lu
    A_phi[:, :, 0, 3] = 1 - 1j * e01 * az_ru + 1j * e02 * ay_ru

    A_phi[:, :, 1, 0] = 1j * az_ld + e01 + 1j * e12 * ay_ld
    A_phi[:, :, 1, 1] = 1j * az_rd - e01 + 1j * e12 * ay_rd
    A_phi[:, :, 1, 2] = -1j * az_lu + e01 + 1j * e12 * ay_lu
    A_phi[:, :, 1, 3] = -1j * az_ru - e01 + 1j * e12 * ay_ru

    A_phi[:, :, 2, 0] = 1j * n_ld * A_phi[:, :, 0, 0]
    A_phi[:, :, 2, 1] = -1j * n_rd * A_phi[:, :, 0, 1]
    A_phi[:, :, 2, 2] = 1j * n_lu * A_phi[:, :, 0, 2]
    A_phi[:, :, 2, 3] = -1j * n_ru * A_phi[:, :, 0, 3]

    A_phi[:, :, 3, 0] = -az_ld * n_ld * A_phi[:, :, 0, 0]
    A_phi[:, :, 3, 1] = -az_rd * n_rd * A_phi[:, :, 0, 1]
    A_phi[:, :, 3, 2] = az_lu * n_lu * A_phi[:, :, 0, 2]
    A_phi[:, :, 3, 3] = az_ru * n_ru * A_phi[:, :, 0, 3]

    norm = (np.sqrt(2) * e00)[:, :, np.newaxis, np.newaxis]
    A /= norm
    A_phi /= norm

    q = np.stack((n_rd * az_rd, n_ld * az_ld, -n_ru * az_ru, -n_lu * az_lu), axis=-1)
    # the reversed magnetization swaps the right and left modes
    q_phi = q[..., [1, 0, 3, 2]]
    return A, A_phi, q, q_phi, k_z
