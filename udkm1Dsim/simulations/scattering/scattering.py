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

"""Base class of all scattering simulations."""

__all__ = ["Scattering"]

__docformat__ = "restructuredtext"

import copy
import warnings

import numpy as np
import pint
from tqdm.auto import tqdm

from ...helpers import make_hash_md5
from ...structures.layers import Layer, UnitCell, Vacuum
from ..parameter_groups import ExperimentParameters
from ..simulation import Simulation

u = pint.get_application_registry()
Q_ = u.Quantity


class Scattering(Simulation):
    r"""Scattering

    Base class for all X-ray and light scattering simulations, which handles
    the photon energy, wavelength, wavevector, incidence angle, and
    scattering vector.

    .. deprecated:: 3.0.0
        The former name ``Xray`` was removed, use :class:`Scattering` instead.

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
        experiment (ExperimentParameters): photon energies and angles in SI
            units and eV, which also hold the depending quantities.
        polarizations (dict): polarization states and according names.
        pol_in_state (int): read-only incoming polarization state as defined
            in polarizations dict, see :meth:`set_incoming_polarization`.
        pol_out_state (int): read-only outgoing polarization state as defined
            in polarizations dict, see :meth:`set_outgoing_polarization`.
        pol_in (float): incoming polarization factor (can be a complex ndarray).
        pol_out (float): outgoing polarization factor (can be a complex ndarray).

    """

    _title = "Scattering simulation"
    # attributes with cached results, which are not sent to the dask workers
    _cache_attributes = ()
    polarizations = {
        0: "unpolarized",
        1: "circ +",
        2: "circ -",
        3: "sigma",
        4: "pi",
        5: "elliptical",
    }
    # the polarization states by their names, where "none" is the no analyzer
    _polarization_ids = {name: state for state, name in polarizations.items()} | {"none": 0}
    # states of the incoming and analyzer polarization of the formalism
    _pol_in_states = (0, 3, 4)
    _pol_out_states = (0,)

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.experiment = ExperimentParameters()

        self._pol_in_state = 3  # sigma
        self._pol_out_state = 0  # no analyzer
        self.pol_in = self.calc_incoming_polarization(self._pol_in_state)
        self.pol_out = self.calc_outgoing_polarization(self._pol_out_state)

    def _check_experiment(self):
        """Warn for scattering vectors beyond :math:`2k`, which have no
        incidence angle at the photon energy."""
        invalid = ~np.isfinite(self.experiment.theta.magnitude)
        if np.any(invalid):
            energy = self.experiment.energy.magnitude[np.any(invalid, axis=1)]
            warnings.warn(
                f"{np.sum(invalid)} scattering vectors qz exceed 2k at photon energies from "
                f"{np.min(energy):g} eV to {np.max(energy):g} eV and have no incidence angle!",
                stacklevel=3,
            )

    def _check_superstrate_and_substrate(self):
        """Raise an error if the superstrate or substrate of the structure is
        not a Layer with its number of repetitions, as set by
        :meth:`.Structure.add_superstrate` and :meth:`.Structure.add_substrate`."""
        for name in ("superstrate", "substrate"):
            value = getattr(self.S, name, None)
            if not (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and isinstance(value[0], Layer)
            ):
                raise ValueError(
                    f"The {name} of the structure '{self.S.name}' must be a Layer, "
                    f"e.g. Vacuum, set by Structure.add_{name}(), but is {value!r}!"
                )

    def _warn_ignored_superstrate_and_substrate(self):
        """The atomistic formalisms neglect the superstrate and include only a
        substrate of unit cells, which is warned for any other layer than
        Vacuum."""
        self._check_superstrate_and_substrate()
        self._check_experiment()
        superstrate, substrate = self.S.superstrate[0], self.S.substrate[0]
        if not isinstance(superstrate, Vacuum):
            warnings.warn(
                f"{type(self).__name__} neglects the superstrate '{superstrate.id}'!",
                stacklevel=3,
            )
        if not isinstance(substrate, (UnitCell, Vacuum)):
            warnings.warn(
                f"{type(self).__name__} only includes a substrate of UnitCells and "
                f"neglects the substrate '{substrate.id}'!",
                stacklevel=3,
            )

    def _worker_copy(self):
        """Copy of the simulation for the dask workers without cached results."""
        worker = copy.copy(self)
        for name in self._cache_attributes:
            setattr(worker, name, {key: [] for key in getattr(self, name)})
        return worker

    def _compute_in_parallel(self, dask_client, function, arguments):
        """Calculates ``function(simulation, *args)`` for all arguments on the
        dask workers. The function must be static, so that only a single copy
        of the simulation is sent to the workers instead of one per task.

        Args:
            dask_client (Dask.Client): Dask client.
            function (callable): static function.
            arguments (list[tuple]): arguments of all tasks.

        Returns:
            results (list): results of all tasks in the order of the arguments.

        """
        from dask.distributed import as_completed

        simulation = dask_client.scatter(self._worker_copy(), hash=False)
        futures = [
            dask_client.submit(function, simulation, *args, pure=False) for args in arguments
        ]
        if self.progress_bar:
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Progress", leave=True):
                pass
        return dask_client.gather(futures)

    @staticmethod
    def _format_range(values):
        """single value or range of a Quantity, which may not be set yet"""
        if np.size(values) == 0:
            return "not set"
        if np.size(values) == 1:
            return values[0]
        return f"{np.min(values):.4g~P} .. {np.max(values):.4g~P}"

    def _property_rows(self):
        """[parameter, value] rows of the simulation properties"""
        output = [
            ["energy", self._format_range(self.energy)],
            ["wavelength", self._format_range(self.wl)],
            ["wavenumber", self._format_range(self.k)],
            ["theta", self._format_range(self.theta)],
            ["q_z", self._format_range(self.qz)],
            ["incoming polarization", self.polarizations[self.pol_in_state]],
            ["analyzer polarization", self.polarizations[self.pol_out_state]],
        ]
        return super()._property_rows() + output

    @property
    def pol_in_state(self):
        """int: incoming polarization state, see :meth:`set_incoming_polarization`."""
        return self._pol_in_state

    @property
    def pol_out_state(self):
        """int: analyzer polarization state, see :meth:`set_outgoing_polarization`."""
        return self._pol_out_state

    def _parse_polarization_state(self, state, states, kind):
        """State id of a polarization given by its id or name, which must be
        one of the given states of the formalism."""
        state_id = self._polarization_ids.get(state, state) if isinstance(state, str) else state
        if state_id not in states:
            names = ", ".join(f"{i}: {self.polarizations[i]}" for i in states)
            raise ValueError(
                f"{type(self).__name__} does not support the {kind} polarization "
                f"{state!r}, but only {names}!"
            )
        return int(state_id)

    def _store_incoming_polarization(self, state, pol_in):
        self._pol_in_state, self.pol_in = state, pol_in
        self.disp_message(f"incoming polarizations set to: {self.polarizations[state]:s}")

    def _store_outgoing_polarization(self, state, pol_out):
        self._pol_out_state, self.pol_out = state, pol_out
        self.disp_message(f"analyzer polarizations set to: {self.polarizations[state]:s}")

    @staticmethod
    def calc_incoming_polarization(pol_in_state):
        """calc_incoming_polarization

        Calculates the incoming polarization factor :math:`\\mbox{pol}` of
        :meth:`get_polarization_factor` for sigma, pi, and unpolarized X-rays.

        Args:
            pol_in_state (int): incoming polarization state id.

        Returns:
            pol_in (float): incoming polarization factor.

        """
        return {0: 0.5, 3: 0, 4: 1}[pol_in_state]

    @staticmethod
    def calc_outgoing_polarization(pol_out_state):
        """calc_outgoing_polarization

        The formalisms without an analyzer have no outgoing polarization
        factor.

        Args:
            pol_out_state (int): outgoing polarization state id.

        Returns:
            pol_out (None): no outgoing polarization factor.

        """
        return None

    def set_incoming_polarization(self, pol_in_state):
        """set_incoming_polarization

        Sets the incoming polarization for a state supported by the
        formalism.

        Args:
            pol_in_state (int, str): incoming polarization state id or its
                name in ``polarizations``, e.g. ``3`` or ``"sigma"``.

        Raises:
            ValueError: the state is not supported by the formalism.

        """
        state = self._parse_polarization_state(pol_in_state, self._pol_in_states, "incoming")
        self._store_incoming_polarization(state, self.calc_incoming_polarization(state))

    def set_outgoing_polarization(self, pol_out_state):
        """set_outgoing_polarization

        Sets the analyzer polarization for a state supported by the
        formalism.

        Args:
            pol_out_state (int, str): outgoing polarization state id or its
                name in ``polarizations``, where ``"none"`` is no analyzer.

        Raises:
            ValueError: the state is not supported by the formalism.

        """
        state = self._parse_polarization_state(pol_out_state, self._pol_out_states, "analyzer")
        self._store_outgoing_polarization(state, self.calc_outgoing_polarization(state))

    def set_polarization(self, pol_in_state, pol_out_state):
        """set_polarization

        Sets the incoming and analyzer (outgoing) polarization.

        Args:
            pol_in_state (int, str): incoming polarization state id or name.
            pol_out_state (int, str): outgoing polarization state id or name.

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
            **kwargs (ndarray[float]): spatio-temporal strain/temperature profile.

        Returns:
            hash (str): unique hash.

        """
        param = [
            self.pol_in_state,
            self.pol_out_state,
            self.experiment.qz.magnitude,
            self.experiment.energy.magnitude,
            strain_vectors,
        ]

        if "strain_map" in kwargs:
            strain_map = kwargs.get("strain_map")
            if np.size(strain_map) > 1e6:
                strain_map = strain_map.flatten()[0:1000000]
            param.append(strain_map)

        if "temp_map" in kwargs:
            temp_map = kwargs.get("temp_map")
            if np.size(temp_map) > 1e6:
                temp_map = temp_map.flatten()[0:1000000]
            param.append(temp_map)

        return self.S.get_hash(types="xray") + "_" + make_hash_md5(param)

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
        return np.sqrt((1 - self.pol_in) + self.pol_in * np.cos(2 * theta) ** 2)

    @property
    def energy(self):
        return self.experiment.energy.quantity

    @energy.setter
    def energy(self, energy):
        self.experiment.energy = energy.to("eV")

    @property
    def wl(self):
        return self.experiment.wl.quantity.to("nm")

    @wl.setter
    def wl(self, wl):
        self.experiment.wl = wl.to("m")

    @property
    def k(self):
        return self.experiment.k.quantity.to("1/nm")

    @k.setter
    def k(self, k):
        self.experiment.k = k.to("1/m")

    @property
    def theta(self):
        return self.experiment.theta.quantity.to("deg")

    @theta.setter
    def theta(self, theta):
        self.experiment.theta = theta.to("rad")

    @property
    def qz(self):
        return self.experiment.qz.quantity.to("1/nm")

    @qz.setter
    def qz(self, qz):
        self.experiment.qz = qz.to("1/m")
