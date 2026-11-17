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

__all__ = ["ALIASES", "Scattering"]

__docformat__ = "restructuredtext"

import copy
import warnings

import numpy as np
import pint
from tqdm.auto import tqdm

from ...helpers import make_hash_md5
from ...structures.layers import Layer, UnitCell, Vacuum
from ..parameter_groups import ExperimentParameters, InstrumentParameters
from ..result import Result
from ..simulation import Simulation
from . import instrument as _instrument

u = pint.get_application_registry()
Q_ = u.Quantity


#: short aliases of the entries of the results of the scattering simulations
ALIASES = {
    "reflectivity": "R",
    "reflectivity_phi": "R_phi",
    "transmissivity": "T",
    "transmissivity_phi": "T_phi",
    "jones_reflection": "jones_R",
    "jones_reflection_phi": "jones_R_phi",
    "jones_transmission": "jones_T",
    "jones_transmission_phi": "jones_T_phi",
    "reflection_rotation": "rot_R",
    "reflection_ellipticity": "ell_R",
    "reflection_rotation_phi": "rot_R_phi",
    "reflection_ellipticity_phi": "ell_R_phi",
    "transmission_rotation": "rot_T",
    "transmission_ellipticity": "ell_T",
    "transmission_rotation_phi": "rot_T_phi",
    "transmission_ellipticity_phi": "ell_T_phi",
    "electric_field": "E",
    "electric_field_phi": "E_phi",
    "magnetic_field": "H",
    "magnetic_field_phi": "H_phi",
    "absorption": "dAdz",
    "absorption_phi": "dAdz_phi",
    "absorption_matrix": "K",
    "absorption_matrix_phi": "K_phi",
    "layer_absorption": "A_layers",
    "layer_absorption_phi": "A_layers_phi",
    "layer_absorption_matrix": "K_layers",
    "layer_absorption_matrix_phi": "K_layers_phi",
    "substrate_flux": "S_sub",
    "substrate_flux_phi": "S_sub_phi",
    "substrate_flux_matrix": "S_sub_matrix",
    "substrate_flux_matrix_phi": "S_sub_matrix_phi",
    "sub_structure_fields": "A",
    "sub_structure_matrices": "A",
}


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
        instrument (InstrumentParameters): angular and spectral resolution,
            source spectrum, footprint, offsets, intensity, and background,
            which are applied to the intensities of the results.
        polarizations (dict): polarization states and according names, where
            ``c+``, ``c-``, ``s``, and ``p`` can also be given for ``circ +``,
            ``circ -``, ``sigma``, and ``pi``.
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
    # and "c+", "c-", "s", and "p" are short names of circ +, circ -, sigma, and pi
    _polarization_ids = {name: state for state, name in polarizations.items()} | {
        "none": 0,
        "c+": 1,
        "c-": 2,
        "s": 3,
        "p": 4,
    }
    # states of the incoming and analyzer polarization of the formalism
    _pol_in_states = (0, 3, 4)
    _pol_out_states = (0,)

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.experiment = ExperimentParameters()
        self.instrument = InstrumentParameters()

        self._pol_in_state = 3  # sigma
        self._pol_out_state = 0  # no analyzer
        self.pol_in = self.calc_incoming_polarization(self._pol_in_state)
        self.pol_out = self.calc_outgoing_polarization(self._pol_out_state)

    def new_scattering_result(
        self, title, num_delays=None, source=None, inputs=None, input_maps=None
    ):
        """new_scattering_result

        New :class:`~udkm1Dsim.simulations.result.Result` with the
        coordinates ``energy``, ``qz``, and ``theta`` of the experiment, see
        :meth:`~udkm1Dsim.simulations.simulation.Simulation.new_result`. The
        ``qz`` and ``theta`` have the dimensions ``energy`` and ``angles``.
        The ``delays`` are taken from an upstream ``source`` result, if
        their number agrees.

        Args:
            title (str): title of the result.
            num_delays (int, optional): number of delays of an inhomogeneous
                result.
            source (Result, optional): upstream result.
            inputs (dict, optional): inputs of the method, see
                :meth:`~udkm1Dsim.simulations.simulation.Simulation.new_result`.
            input_maps (dict, optional): input maps of the method, see
                :meth:`~udkm1Dsim.simulations.simulation.Simulation.new_result`.

        Returns:
            result (Result): empty result.

        """
        result = self.new_result(title, inputs=inputs, input_maps=input_maps)
        if (
            isinstance(source, Result)
            and "delays" in source.coords
            and np.size(source["delays"]) == num_delays
        ):
            result.copy_coords(source, ["delays"])
        result.add_coord("energy", np.atleast_1d(self.energy.to("eV").magnitude), "eV")
        dims = ("energy", "angles")
        result.add_coord("qz", np.atleast_2d(self.qz.to("1/nm").magnitude), "1/nm", dims)
        result.add_coord("theta", np.atleast_2d(self.theta.to("deg").magnitude), "deg", dims)
        return result

    def _scattering_result(
        self,
        title,
        names,
        values,
        unit="",
        inhomogeneous=False,
        source=None,
        inputs=None,
        input_maps=None,
    ):
        """result of the ``values`` with the ``names`` and their
        :data:`ALIASES`, which have a first axis of the delays for
        ``inhomogeneous=True`` with the coordinates of an upstream ``source``
        result and the ``inputs`` and ``input_maps`` of the method, see
        :meth:`new_scattering_result`"""
        num_delays = np.size(values[0], 0) if inhomogeneous else None
        result = self.new_scattering_result(title, num_delays, source, inputs, input_maps)
        dims = ("delays", "energy", "angles") if inhomogeneous else ("energy", "angles")
        for name, value in zip(names, values):
            # e.g. the lists of the sub-structures have no dimensions
            value_dims = dims if isinstance(value, np.ndarray) else ()
            result.add(name, value, value_dims, unit, ALIASES.get(name, ()))
        return result

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
        of the simulation is sent to the workers instead of one per task. The
        array arguments are sent to the workers before the tasks as well.

        Args:
            dask_client (Dask.Client): Dask client.
            function (callable): static function.
            arguments (list[tuple]): arguments of all tasks.

        Returns:
            results (list): results of all tasks in the order of the arguments.

        """
        from dask.distributed import as_completed

        simulation = dask_client.scatter(self._worker_copy(), hash=False)
        # the arrays, e.g. blocks of the strain map, are sent to the workers
        # before the tasks, which keeps the task graph small
        arguments = [list(args) for args in arguments]
        positions = [
            (i, j)
            for i, args in enumerate(arguments)
            for j, arg in enumerate(args)
            if isinstance(arg, np.ndarray)
        ]
        if positions:
            scattered = dask_client.scatter([arguments[i][j] for i, j in positions], hash=False)
            for (i, j), future in zip(positions, scattered):
                arguments[i][j] = future
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
            ["instrument", "active" if self.instrument.is_active else "none"],
        ]
        return super()._property_rows() + output

    def _parameters(self):
        """dict of the parameters of the simulation, whose photon energies
        and angles are the coordinates of the results"""
        return {
            **super()._parameters(),
            "incoming_polarization": self.polarizations[self.pol_in_state],
            "analyzer_polarization": self.polarizations[self.pol_out_state],
            "pol_in": self.pol_in,
            "pol_out": self.pol_out,
            "instrument": self.instrument.to_dict(),
        }

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
        :meth:`calc_polarization_factor` for sigma, pi, and unpolarized X-rays.

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
        if self.instrument.is_active:
            # only an instrument changes the hash of former results
            param.append(("instrument", self.instrument._hash_items()))

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

    def calc_polarization_factor(self, theta):
        r"""calc_polarization_factor

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

    # ---- instrument function ----------------------------------------------------

    def apply_instrument(self, calc, kinds, finish=None, lead=0):
        r"""apply_instrument

        Applies the :attr:`instrument` to the intensities of
        ``calc()``, which is repeated for all nodes of the quadrature with
        the experiment set to the photon energies and angles of the nodes:

        1. The ``theta_offset`` and ``energy_offset`` are added.
        2. For each component of the ``spectrum`` and each energy node, the
           photon energies, or the angles for the ``'qz'`` energy mode, are
           set, and for each angular node the angles.
        3. The reflected intensities are multiplied by the ``footprint`` at
           the angles of the node and all are summed with the weights of the
           nodes and components.
        4. The ``'convolution'`` modes convolve the sums along the energies
           and the angles of the experiment.
        5. The sums are reduced by ``finish``, e.g. to the rotation of the
           polarization, and the intensities are scaled to
           :math:`I_0 I + I_\mathrm{bkg}`.

        Without an active instrument, the values of ``calc()`` are returned
        as they are, reduced by ``finish``.

        Args:
            calc (callable): returns a tuple of arrays of the experiment with
                the energies at the axis ``lead`` and the angles at the axis
                ``lead + 1``.
            kinds (tuple[str]): kind of each value, i.e. ``'reflected'`` and
                ``'transmitted'`` intensities, which are scaled by
                :math:`I_0` and the background, where only the reflected ones
                are multiplied by the footprint, as well as ``'reflected
                terms'`` and ``'transmitted terms'``, which are only averaged,
                e.g. for the polarization of the light.
            finish (callable, optional): reduces the averaged tuple of values.
            lead (int, optional): number of leading axes, e.g. 1 for the
                delays. Defaults to 0.

        Returns:
            values (tuple[ndarray[float]]): values with the instrument.

        """
        ins = self.instrument
        if not ins.is_active:
            values = calc()
            return finish(values) if finish is not None else values
        exp = self.experiment
        scan = exp._scan
        energy0 = np.array(exp.energy.magnitude, dtype=float)
        scan0 = np.array(exp.__dict__[scan].magnitude, dtype=float)
        theta0 = np.array(exp.theta.magnitude, dtype=float)
        M, N = theta0.shape
        energy = energy0 + ins.energy_offset.magnitude
        theta = theta0 + np.deg2rad(ins.theta_offset.magnitude)
        theta_width = np.deg2rad(self._instrument_width(ins.theta_width, (M, N), "theta_width"))
        energy_width = self._instrument_width(ins.energy_width, (M,), "energy_width")
        theta_nodes = (
            self._instrument_nodes("theta")
            if ins.theta_mode == "quadrature"
            else (np.zeros(1), np.ones(1))
        )
        energy_nodes = (
            self._instrument_nodes("energy")
            if ins.energy_mode != "convolution"
            else (np.zeros(1), np.ones(1))
        )
        if ins.energy_mode == "convolution" and ins.energy_resolution != "none":
            if M > 1 and not np.allclose(theta0, theta0[:1, :]):
                raise ValueError(
                    "The 'convolution' of the energy is done at fixed angles, which must be "
                    "the same for all photon energies, i.e. the angles must be set by theta, "
                    "not by qz, otherwise use the 'quadrature' energy mode!"
                )
        num_components = len(ins.components)
        num_energy, num_theta = np.size(energy_nodes[0]), np.size(theta_nodes[0])
        num_calculations = num_components * num_energy * num_theta
        if num_calculations > 1:
            self.disp_message(
                f"Instrument: {num_calculations} calculations ({num_components} spectral "
                f"component{'s' * (num_components > 1)} x {num_energy} energy "
                f"node{'s' * (num_energy > 1)} x {num_theta} angle node{'s' * (num_theta > 1)})"
            )
        total = None
        try:
            for component in ins.components:
                energy_c = component["factor"] * energy + component["offset"]
                width_c = component.get("width")
                width_c = (
                    component["factor"] * energy_width if width_c is None else np.full(M, width_c)
                )
                part = None
                for x_e, w_e in zip(*energy_nodes):
                    for x_t, w_t in zip(*theta_nodes):
                        th = theta + x_t * theta_width
                        if ins.energy_mode == "qz":
                            exp.energy = energy_c
                            scale = 1 + x_e * width_c / energy_c
                            th = np.arcsin(np.clip(np.sin(th) * scale[:, np.newaxis], -1, 1))
                        else:
                            exp.energy = energy_c + x_e * width_c
                        exp.theta = th
                        weights = w_e * w_t * np.ones((M, N))
                        foot = weights * _instrument.footprint(
                            ins.footprint,
                            th,
                            ins.beam_width.magnitude,
                            ins.sample_length.magnitude,
                        )
                        values = calc()
                        node = tuple(
                            self._expand(
                                foot if kind.startswith("reflected") else weights, v, lead
                            )
                            * v
                            for kind, v in zip(kinds, values)
                        )
                        part = node if part is None else tuple(p + n for p, n in zip(part, node))
                if ins.energy_mode == "convolution" and ins.energy_resolution != "none" and M > 1:
                    # the spread of the component maps to the energies of the experiment
                    part = tuple(
                        _instrument.convolve_on_grid(
                            p,
                            lead,
                            energy0,
                            width_c / component["factor"],
                            ins.energy_resolution,
                            ins.energy_range.magnitude,
                            ins.energy_eta.magnitude,
                        )
                        for p in part
                    )
                part = tuple(component["weight"] * p for p in part)
                total = part if total is None else tuple(t + p for t, p in zip(total, part))
        finally:
            exp.energy = energy0
            setattr(exp, scan, scan0)
        if ins.theta_mode == "convolution" and ins.theta_resolution != "none":
            convolved = [np.array(t, dtype=float) for t in total]
            for i in range(M):
                index = (slice(None),) * lead + (i,)
                for t, c in zip(total, convolved):
                    c[index] = _instrument.convolve_on_grid(
                        t[index],
                        lead,
                        theta0[i, :],
                        theta_width[i, :],
                        ins.theta_resolution,
                        ins.theta_range.magnitude,
                        ins.theta_eta.magnitude,
                    )
            total = tuple(convolved)
        if finish is not None:
            return finish(total)
        I0, background = ins.I0.magnitude, ins.background.magnitude
        return tuple(
            I0 * t + background if kind in ("reflected", "transmitted") else t
            for kind, t in zip(kinds, total)
        )

    def _apply_instrument_with_nominal(self, calc):
        """reflectivity of ``calc()`` with the :attr:`instrument` and the
        second value of ``calc()``, e.g. the fields of the sub-structures, at
        the photon energies and angles of the experiment, which is taken
        from a node without shift, if there is one, e.g. for a convolution,
        or calculated once more"""
        if not self.instrument.is_active:
            return calc()
        energy = np.array(self.experiment.energy.magnitude)
        theta = np.array(self.experiment.theta.magnitude)
        nominal = []

        def node():
            R, A = calc()
            if (
                not nominal
                and np.array_equal(self.experiment.energy.magnitude, energy)
                and np.array_equal(self.experiment.theta.magnitude, theta)
            ):
                nominal.append(A)
            return (R,)

        (R,) = self.apply_instrument(node, ("reflected",))
        return R, nominal[0] if nominal else calc()[1]

    def _instrument_nodes(self, axis):
        """nodes in units of the FWHM and weights of the ``axis`` of the
        instrument, i.e. ``'theta'`` or ``'energy'``"""
        ins = self.instrument
        return _instrument.profile_nodes(
            getattr(ins, f"{axis}_resolution"),
            getattr(ins, f"{axis}_nodes"),
            getattr(ins, f"{axis}_points").magnitude,
            getattr(ins, f"{axis}_range").magnitude,
            getattr(ins, f"{axis}_eta").magnitude,
        )

    @staticmethod
    def _instrument_width(parameter, shape, name):
        """width of the instrument broadcast to the ``shape`` of the
        experiment, where widths along the angles are given for all
        energies"""
        width = np.asarray(parameter.magnitude, dtype=float)
        try:
            return np.broadcast_to(width, shape).copy()
        except ValueError:
            raise ValueError(
                f"The {name} of the instrument of shape {width.shape} must be a scalar or "
                f"broadcast to the shape {shape} of the experiment!"
            ) from None

    @staticmethod
    def _expand(weights, value, lead):
        """weights of shape (M, N) with axes for the ``lead`` axes and the
        trailing axes of the ``value``, e.g. of several polarizations"""
        trail = np.ndim(value) - lead - 2
        return weights.reshape((1,) * lead + weights.shape + (1,) * trail)

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
