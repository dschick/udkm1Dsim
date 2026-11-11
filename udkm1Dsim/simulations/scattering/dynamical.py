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

"""Dynamical X-ray diffraction of unit cells."""

__all__ = ["XrayDyn"]

__docformat__ = "restructuredtext"

import warnings
from os import path
from time import time

import numpy as np
import pint
import scipy.constants as constants
from tqdm.auto import trange

from ...helpers import finderb, m_power_x, m_times_n, make_hash_md5
from ...structures.layers import UnitCell, Vacuum
from ...structures.structure import Structure
from .scattering import Scattering

u = pint.get_application_registry()
Q_ = u.Quantity

r_0 = constants.physical_constants["classical electron radius"][0]


class XrayDyn(Scattering):
    r"""XrayDyn

    Dynamical X-ray scattering simulations following :cite:t:`alsnielsen2011`.

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
        last_atom_ref_trans_matrices (list): remember last result of
           atom ref_trans_matrices to speed up calculation.

    """

    _title = "Dynamical X-Ray Diffraction simulation"
    _cache_attributes = ("last_atom_ref_trans_matrices",)

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.last_atom_ref_trans_matrices = {"atom_ids": [], "hashes": [], "H": []}

    def homogeneous_reflectivity(self, strains=[], temps=[]):
        r"""homogeneous_reflectivity

        Calculates the reflectivity :math:`R` of the whole sample structure
        and the reflectivity-transmission matrices :math:`M_{RT}` for
        each substructure. The reflectivity of the :math:`2\times 2`
        matrices for each :math:`q_z` is calculates as follow:

        .. math:: R = \left|M_{RT}^t(0,1)/M_{RT}^t(1,1)\right|^2

        Args:
            strains (ndarray[float], optional): strains of each sub-structure
            temps (ndarray[float], optional): temperatures of each sub-structure

        Returns:
            (tuple):
            - *R (ndarray[float])* - homogeneous reflectivity.
            - *A (ndarray[complex])* - reflectivity-transmission matrices of
              sub-structures.

        """
        self._check_experiment()
        self._warn_ignored_superstrate_and_substrate()
        t1 = time()
        self.disp_message("Calculating _homogenous_reflectivity_ ...")

        # a superstrate does not make too much sense here
        # get the reflectivity-transmission matrix of the actual structure
        RT, A = self.homogeneous_ref_trans_matrix(self.S, strains, temps)
        # add static substrate
        if isinstance(self.S.substrate[0], UnitCell):
            tmp, tmp2 = self.homogeneous_ref_trans_matrix(self.S.substrate)
            A.append([tmp2, "static substrate"])
            RT = m_times_n(RT, tmp)

        # calculate the real reflectivity from the RT matrix
        R = self.calc_reflectivity_from_matrix(RT)
        self.disp_message(f"Elapsed time for _homogenous_reflectivity_: {time() - t1:f} s")
        return R, A

    def homogeneous_ref_trans_matrix(self, S, strains=[], temps=[]):
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
            strains (ndarray[float], optional): strains of each sub-structure
            temps (ndarray[float], optional): temperatures of each sub-structure

        Returns:
            (tuple):
            - *RT (ndarray[complex])* - reflectivity-transmission matrix.
            - *A (ndarray[complex])* - reflectivity-transmission matrices of
              sub-structures.

        """
        if isinstance(S, Structure):
            sub_structures = S.sub_structures
            L = S.get_number_of_sub_structures()
        elif isinstance(S[0], (Vacuum, UnitCell)):
            sub_structures = [S]
            L = 1
        else:
            raise TypeError("XrayDyn can only handle Layers of class UnitCell and Vacuum")

        # if no strains are given we assume no strain (1)
        if len(strains) == 0:
            strains = np.zeros([L])
        else:
            strains = np.array(strains)
        if len(strains) != L:
            raise IndexError(f"Number of strains must match the number of substructures: {L:d}")

        if len(temps) == 0:
            temps = np.zeros([L, 1])
        else:
            temps = np.array(temps)
            if temps.ndim == 1:
                # add second dimension for temperature
                temps = temps[:, np.newaxis]
            if temps.shape[0] != L:
                raise IndexError(
                    "First dimension of temperatures must match the number of "
                    f"substructures {L:d}."
                )

            # check length (number of sub-systems) of Debye-Waller factor, which is not checked
            # in setter method
            numel_deb_wal_fac = self.S.get_numel_of_layer_property("deb_wal_fac")
            if temps.shape[1] != numel_deb_wal_fac:
                raise IndexError(
                    "Second dimension of temperatures must match the number of "
                    f"subsystems for the Debye-Waller factor: {numel_deb_wal_fac:d}"
                )
        # initialize
        RT = np.tile(
            np.eye(2, 2)[np.newaxis, np.newaxis, :, :],
            (
                np.size(self.experiment.qz.magnitude, 0),
                np.size(self.experiment.qz.magnitude, 1),
                1,
                1,
            ),
        )  # ref_trans_matrix
        A = []  # list of ref_trans_matrices of substructures
        counter = 0

        # traverse substructures
        for sub_structure in sub_structures:
            if isinstance(sub_structure[0], UnitCell):
                # the sub_structure is an unitCell
                # calculate the ref-trans matrices for N unitCells
                tmp = m_power_x(
                    self.get_uc_ref_trans_matrix(
                        sub_structure[0], strains[counter], temps[counter, :]
                    ),
                    sub_structure[1],
                )
                counter += 1
                # remember the result
                A.append([tmp, f"{sub_structure[1]:d}x {sub_structure[0].name:s}"])
            elif isinstance(sub_structure[0], Structure):
                # its a structure
                # make a recursive call
                idx = np.r_[counter : (counter + sub_structure[0].get_number_of_sub_structures())]
                tmp, tmp2 = self.homogeneous_ref_trans_matrix(
                    sub_structure[0], strains[idx], temps[idx, :]
                )
                A.append([tmp2, sub_structure[0].name + " substructures"])
                counter = counter + sub_structure[0].get_number_of_sub_structures()
                A.append([tmp, f"{sub_structure[1]:d}x {sub_structure[0].name:s}"])
                # calculate the ref-trans matrices for N sub structures
                tmp = m_power_x(tmp, sub_structure[1])
                A.append([tmp, f"{sub_structure[1]:d}x {sub_structure[0].name:s}"])
            else:
                raise TypeError("The substructure must be Vacuum, UnitCell or Structure type!")

            # multiply it to the output
            RT = m_times_n(RT, tmp)

        return RT, A

    def inhomogeneous_reflectivity(
        self, strain_map, strain_vectors=[], temp_map=np.array([]), **kwargs
    ):
        """inhomogeneous_reflectivity

        Returns the reflectivity of an inhomogeneously strained sample
        structure for a given ``strain_map`` in position and time, as well
        as for a given set of possible strains for each unit cell in the
        sample structure (``strain_vectors``).
        If no reflectivity is saved in the cache it is caluclated.
        Providing the ``calc_type`` for the calculation the corresponding
        sub-routines for the reflectivity computation are called:

        * ``parallel`` parallelization over the time steps utilizing
          `Dask <https://dask.org/>`_ on a local machine or a cluster,
          depending on the ``dask_client``
        * ``sequential`` no parallelization at all

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            strain_vectors (list[ndarray[float]], optional): reduced strains per unique
                layer.
            temp_map (ndarray[float], optional): spatio-temporal temperature profile.
            **kwargs:
                - *calc_type (str)* - type of calculation.
                - *dask_client (Dask.Client)* - Dask client.
                - *job (Dask.job)* - Dask job.
                - *num_workers (int)* - Dask number of workers.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        self._check_experiment()
        self._warn_ignored_superstrate_and_substrate()
        # create a hash of all simulation parameters
        filename = (
            "inhomogeneous_reflectivity_dyn_"
            + self.get_hash(strain_vectors, strain_map=strain_map, temp_map=temp_map)
            + ".npz"
        )
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            R = tmp["R"]
            self.disp_message("_inhomogeneous_reflectivity_ loaded from file:\n\t" + filename)
        else:
            t1 = time()
            self.disp_message("Calculating _inhomogeneousReflectivity_ ...")
            # parse the input arguments
            if not isinstance(strain_map, np.ndarray):
                raise TypeError("strain_map must be a numpy ndarray!")
            if not isinstance(strain_vectors, list):
                raise TypeError("strain_vectors must be a list!")
            if not isinstance(temp_map, np.ndarray):
                raise TypeError("temp_map must be a numpy ndarray!")

            (M, L) = strain_map.shape
            # check length (number of sub-systems) of Debye-Waller factor, which is not checked
            # in setter method
            numel_deb_wal_fac = self.S.get_numel_of_layer_property("deb_wal_fac")

            if len(temp_map) == 0:
                temp_map = np.zeros([M, L, numel_deb_wal_fac])
            else:
                try:
                    temp_map = np.reshape(temp_map, [M, L, numel_deb_wal_fac])
                except ValueError:
                    raise ValueError(
                        "Third dimension of temp_map must match the number of sub-"
                        f"systems for the Debye-Waller factor: {numel_deb_wal_fac:d}"
                    )

                if len(strain_vectors) > 0:
                    warnings.warn(
                        "strain_vectors and temp_map are not compatible "
                        "with each other! strain_vectors takes over."
                    )

            dask_client = kwargs.get("dask_client", [])
            calc_type = kwargs.get("calc_type", "sequential")
            if calc_type not in ["parallel", "sequential"]:
                raise TypeError("calc_type must be either _parallel_ or _sequential_!")

            # optinally calculate all ref-trans matrices for all unique unitCells
            # and for all possible strains in advance, if strain_vectors are given
            if len(strain_vectors) > 0:
                RTM = self.get_all_ref_trans_matrices(strain_vectors)
            else:
                RTM = []

            # select the type of computation
            if calc_type == "parallel":
                R = self.parallel_inhomogeneous_reflectivity(
                    strain_map, strain_vectors, RTM, temp_map, dask_client
                )
            else:  # sequential
                R = self.sequential_inhomogeneous_reflectivity(
                    strain_map, strain_vectors, RTM, temp_map
                )

            self.disp_message(f"Elapsed time for _inhomogeneous_reflectivity_: {time() - t1:f} s")
            self.save(full_filename, {"R": R}, "_inhomogeneous_reflectivity_")
        return R

    def sequential_inhomogeneous_reflectivity(self, strain_map, strain_vectors, RTM, temp_map):
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
            temp_map (ndarray[float], optional): spatio-temporal temperature profile.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        # initialize
        M = np.size(strain_map, 0)  # delay steps
        R = np.zeros(
            [M, np.size(self.experiment.qz.magnitude, 0), np.size(self.experiment.qz.magnitude, 1)]
        )
        if self.progress_bar:
            iterator = trange(M, desc="Progress", leave=True)
        else:
            iterator = range(M)
        # get the inhomogeneous reflectivity of the sample
        # structure for each time step of the strain_map and temp_map
        for i in iterator:
            R[i, :, :] = self.calc_inhomogeneous_reflectivity(
                strain_map[i, :],
                strain_vectors,
                RTM,
                temp_map[i, :, :],
            )
        return R

    def parallel_inhomogeneous_reflectivity(
        self, strain_map, strain_vectors, RTM, temp_map, dask_client
    ):
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
            temp_map (ndarray[float], optional): spatio-temporal temperature profile.
            dask_client (Dask.Client): Dask client.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        if not dask_client:
            raise ValueError("no dask client set")
        # initialize
        M = np.size(strain_map, 0)  # delay steps
        N = np.size(self.experiment.qz.magnitude, 0)  # energy steps
        K = np.size(self.experiment.qz.magnitude, 1)  # qz steps

        R = np.zeros([M, N, K])
        # init unity matrix for matrix multiplication
        RTU = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (N, K, 1, 1))

        # precalculate the substrate ref_trans_matrix if present
        if isinstance(self.S.substrate[0], UnitCell):
            RTS, _ = self.homogeneous_ref_trans_matrix(self.S.substrate)
        else:
            RTS = RTU

        if len(strain_vectors) > 0:
            uc_indices, _, _ = self.S.get_layer_vectors()
            # make RTM available for all works
            shared = dask_client.scatter((uc_indices, RTU, strain_vectors, RTM, RTS), hash=False)
            results = self._compute_in_parallel(
                dask_client,
                XrayDyn._parallel_lookup_delay,
                [(shared, strain_map[i, :]) for i in range(M)],
            )
        else:
            remote_RTS = dask_client.scatter(RTS, hash=False)
            results = self._compute_in_parallel(
                dask_client,
                XrayDyn._parallel_recalc_delay,
                [(remote_RTS, strain_map[i, :], temp_map[i, :, :]) for i in range(M)],
            )

        # reorder results to reflectivity matrix
        for i, result in enumerate(results):
            R[i, :, :] = result

        return R

    @staticmethod
    def _parallel_lookup_delay(sim, shared, strains):
        """Reflectivity of a single delay from the given matrices on a dask worker."""
        uc_indices, RTU, strain_vectors, RTM, RTS = shared
        RT = XrayDyn.lookup_inhomogeneous_ref_trans_matrix(
            uc_indices, RTU, strains, strain_vectors, RTM
        )
        return XrayDyn.calc_reflectivity_from_matrix(m_times_n(RT, RTS))

    @staticmethod
    def _parallel_recalc_delay(sim, RTS, strains, temps):
        """Reflectivity of a single delay recalculated on a dask worker."""
        RT = sim.calc_inhomogeneous_ref_trans_matrix(strains, temps)
        return XrayDyn.calc_reflectivity_from_matrix(m_times_n(RT, RTS))

    def calc_inhomogeneous_reflectivity(self, strains, strain_vectors, RTM, temps):
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
            strains (ndarray[float]): spatial strain profile for single time
                step.
            strain_vectors (list[ndarray[float]]): reduced strains per unique
                layer.
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.
            temps (ndarray[float]): spatial temperature profile for single time
                step.

        Returns:
            R (ndarray[float]): inhomogeneous reflectivity.

        """
        # initialize ref_trans_matrix
        N = np.shape(self.experiment.qz.magnitude)[1]  # number of q_z
        M = np.shape(self.experiment.qz.magnitude)[0]  # number of energies
        uc_indices, _, _ = self.S.get_layer_vectors()

        # initialize ref_trans_matrix
        RTU = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (M, N, 1, 1))

        if len(strain_vectors) > 0:
            RT = XrayDyn.lookup_inhomogeneous_ref_trans_matrix(
                uc_indices, RTU, strains, strain_vectors, RTM
            )
        else:
            RT = self.calc_inhomogeneous_ref_trans_matrix(strains, temps)

        # if a substrate is included add it at the end
        if isinstance(self.S.substrate[0], UnitCell):
            RTS, _ = self.homogeneous_ref_trans_matrix(self.S.substrate)
            RT = m_times_n(RT, RTS)
        # calculate reflectivity from ref-trans matrix
        R = self.calc_reflectivity_from_matrix(RT)
        return R

    def calc_inhomogeneous_ref_trans_matrix(self, strains, temps):
        r"""calc_inhomogeneous_ref_trans_matrix

        Sub-function of :meth:`calc_inhomogeneous_reflectivity` and for
        parallel computing (needs to be static) only for calculating the
        total reflection-transmission matrix :math:`M_{RT}^t`:

        .. math:: M_{RT}^t = \prod_{j=1}^M M_{RT,j}

        Args:
            strains (ndarray[float]): spatial strain profile for single time
                step.
            temps (ndarray[float]): spatial temperature profile for single time
                step.

        Returns:
            RT (ndarray[complex]): reflection-transmission matrix.

        """
        _, _, uc_handles = self.S.get_layer_vectors()

        N = np.shape(self.experiment.qz.magnitude)[1]  # number of q_z
        M = np.shape(self.experiment.qz.magnitude)[0]  # number of energies
        RT = np.tile(np.eye(2, 2)[np.newaxis, np.newaxis, :, :], (M, N, 1, 1))
        # traverse all unit cells in the sample structure
        for i, uc in enumerate(uc_handles):
            if not isinstance(uc, UnitCell):
                raise TypeError("All layers  must be of type UnitCell!")
            RT = m_times_n(RT, self.get_uc_ref_trans_matrix(uc, strains[i], temps[i, :]))

        return RT

    @staticmethod
    def lookup_inhomogeneous_ref_trans_matrix(uc_indices, RT, strains, strain_vectors, RTM):
        r"""lookup_inhomogeneous_ref_trans_matrix

        Sub-function of :meth:`calc_inhomogeneous_reflectivity` and for
        parallel computing (needs to be static) only for looking up the
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
            tmp = RTM[int(uc_index)][strain_index]
            if tmp.size:
                RT = m_times_n(RT, tmp)
            else:
                raise ValueError("RTM not found")

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
            strain_vectors = [np.array([1])] * self.S.get_number_of_unique_layers()
        else:
            strain_vectors = args[0]
        # create a hash of all simulation parameters
        filename = "all_ref_trans_matrices_dyn_" + self.get_hash(strain_vectors) + ".npz"
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            RTM = tmp["RTM"]
            self.disp_message("_all_ref_trans_matrices_dyn_ loaded from file:\n\t" + filename)
        else:
            # nothing found so calculate it and save it
            RTM = self.calc_all_ref_trans_matrices(strain_vectors)
            self.save(full_filename, {"RTM": RTM}, "_all_ref_trans_matrices_dyn_")
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
        self.disp_message("Calculate all _ref_trans_matrices_ ...")
        # initialize
        uc_ids, uc_handles = self.S.get_unique_layers()
        # if no strain_vectors are given we just do it for no strain (1)
        if len(args) == 0:
            strain_vectors = [np.array([1])] * len(uc_ids)
        else:
            strain_vectors = args[0]
        # check if there are strains for each unique unitCell
        if len(strain_vectors) is not len(uc_ids):
            raise TypeError(
                "The strain vector has not the same size as number of unique unit cells"
            )

        # initialize ref_trans_matrices
        RTM = []

        # traverse all unique unit_cells
        for i, uc in enumerate(uc_handles):
            # traverse all strains in the strain_vector for this unique
            # unit_cell
            if not isinstance(uc, UnitCell):
                raise TypeError("All layers  must be UnitCells!")
            temp = []
            for strain in strain_vectors[i]:
                temp.append(self.get_uc_ref_trans_matrix(uc, strain))
            RTM.append(temp)
        self.disp_message(f"Elapsed time for _ref_trans_matrices_: {time() - t1:f} s")
        return RTM

    def get_uc_ref_trans_matrix(self, uc, strain=0, temp=np.array([0])):
        r"""get_uc_ref_trans_matrix

        Returns the reflection-transmission matrix of a unit cell:

        .. math:: M_{RT} = \prod_i H_i \  L_i

        where :math:`H_i` and :math:`L_i` are the atomic reflection-
        transmission matrix and the phase matrix for the atomic distances,
        respectively.

        Args:
            uc (UnitCell): unit cell object.
            strain (float, optional): strain of unit cell.
            temp (ndarray[float], optional): temperature of unit cell.

        Returns:
            RTM (list[ndarray[complex]]): reflection-transmission matrices for
                all given strains per unique layer.

        """
        M = len(self.experiment.energy.magnitude)  # number of energies
        N = np.shape(self.experiment.qz.magnitude)[1]  # number of q_z
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
            if i == (K - 1):  # its the last atom
                rel_dist = (strain + 1) - uc.atoms[i][1](strain)
            else:
                rel_dist = uc.atoms[i + 1][1](strain) - uc.atoms[i][1](strain)

            # sum Debye-Waller factors for all sub-systems
            deb_wal_fac = np.sum(
                np.array([dbf(T) for dbf, T in zip(uc.thermal.deb_wal_fac.functional, temp)])
            )

            # get the reflection-transmission matrix and phase matrix
            # from all atoms in the unit cell and multiply them
            # together
            RTM = m_times_n(
                RTM,
                self.get_atom_ref_trans_matrix(
                    uc.atoms[i][0], uc.structural.area.magnitude, deb_wal_fac
                ),
            )
            RTM = m_times_n(
                RTM, self.get_atom_phase_matrix(rel_dist * uc.lattice.c_axis.magnitude)
            )
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
            area (float): area of the unit cell (m²)
            deb_wal_fac (float): Debye-Waller factor for unit cell

        Returns:
            H (ndarray[complex]): reflection-transmission matrix

        """
        # check for already calculated data
        _hash = make_hash_md5(
            [
                self.experiment.energy.magnitude,
                self.experiment.qz.magnitude,
                self.pol_in_state,
                self.pol_out_state,
                area,
                deb_wal_fac,
            ]
        )
        try:
            index = self.last_atom_ref_trans_matrices["atom_ids"].index(atom.id)
        except ValueError:
            index = -1

        if (index >= 0) and (_hash == self.last_atom_ref_trans_matrices["hashes"][index]):
            # These are the same X-ray parameters as last time so we
            # can use the same matrix again for this atom
            H = self.last_atom_ref_trans_matrices["H"][index]
        else:
            # These are new parameters so we have to calculate.
            # Get the reflection-transmission-factors
            rho = self.get_atom_reflection_factor(atom, area, deb_wal_fac)
            tau = self.get_atom_transmission_factor(atom, area)
            # calculate the reflection-transmission matrix
            H = np.zeros(
                [
                    np.shape(self.experiment.qz.magnitude)[0],
                    np.shape(self.experiment.qz.magnitude)[1],
                    2,
                    2,
                ],
                dtype=np.complex128,
            )
            H[:, :, 0, 0] = (1 / tau) * (tau**2 - rho**2)
            H[:, :, 0, 1] = (1 / tau) * (rho)
            H[:, :, 1, 0] = (1 / tau) * (-rho)
            H[:, :, 1, 1] = 1 / tau
            # remember this matrix for next use with the same
            # parameters for this atom
            if index >= 0:
                self.last_atom_ref_trans_matrices["atom_ids"][index] = atom.id
                self.last_atom_ref_trans_matrices["hashes"][index] = _hash
                self.last_atom_ref_trans_matrices["H"][index] = H
            else:
                self.last_atom_ref_trans_matrices["atom_ids"].append(atom.id)
                self.last_atom_ref_trans_matrices["hashes"].append(_hash)
                self.last_atom_ref_trans_matrices["H"].append(H)
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
        - :math:`M = 0.5 \mbox{dbf} q_z^2` where
          :math:`\mbox{dbf} = \langle u^2\rangle` is the average
          thermal vibration of the atoms - Debye-Waller factor

        Args:
            atom (Atom, AtomMixed): atom or mixed atom
            area (float): area of the unit cell (m²)
            deb_wal_fac (float): Debye-Waller factor for unit cell

        Returns:
            rho (complex): reflection factor

        """
        rho = (
            -4j
            * np.pi
            * r_0
            * atom.get_cm_atomic_form_factor(
                self.experiment.energy.magnitude, self.experiment.qz.magnitude
            )
            * self.get_polarization_factor(self.experiment.theta.magnitude)
            * np.exp(-0.5 * deb_wal_fac * self.experiment.qz.magnitude**2)
        ) / (self.experiment.qz.magnitude * area)
        return rho

    def get_atom_transmission_factor(self, atom, area):
        r"""get_atom_transmission_factor

        Calculates the transmission factor from dynamical x-ray theory:

        .. math:: \tau = 1 - \frac{i 4 \pi r_e f(E,0)}{q_z A}

        - :math:`r_e` is the electron radius
        - :math:`f(E,0)` is the energy dispersive atomic form factor
          (no angle correction)
        - :math:`A` is the area in :math:`x-y` plane on which the atom
          is placed

        The Debye-Waller factor does not apply to the forward scattering
        without momentum transfer, which gives the absorption and refraction.

        Args:
            atom (Atom, AtomMixed): atom or mixed atom
            area (float): area of the unit cell (m²)

        Returns:
            tau (complex): transmission factor

        """
        tau = 1 - (
            4j
            * np.pi
            * r_0
            * atom.get_cm_atomic_form_factor(
                self.experiment.energy.magnitude, np.zeros_like(self.experiment.qz.magnitude)
            )
        ) / (self.experiment.qz.magnitude * area)
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
        L = np.zeros(
            [
                np.shape(self.experiment.qz.magnitude)[0],
                np.shape(self.experiment.qz.magnitude)[1],
                2,
                2,
            ],
            dtype=np.complex128,
        )
        L[:, :, 0, 0] = np.exp(1j * phi)
        L[:, :, 1, 1] = np.exp(-1j * phi)
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
        phi = distance * self.experiment.qz.magnitude / 2
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
        return np.abs(M[:, :, 0, 1] / M[:, :, 1, 1]) ** 2
