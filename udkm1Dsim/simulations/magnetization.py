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

__all__ = ["Magnetization", "LLG", "iLLG", "LLB"]

__docformat__ = "restructuredtext"

import warnings
from time import time

import numpy as np
import pint

from ..helpers import (
    convert_cartesian_to_polar,
    convert_polar_to_cartesian,
    finderb,
    make_hash_md5,
)
from ..structures.layers import UnitCell
from ..structures.parameters import VectorParameter
from .result import Result, find_source, get_entry
from .simulation import Simulation
from .solvers import magnetization_scipy
from .solvers.mean_field import TABLE, interpolate_tables_map, sublattice_signs

u = pint.get_application_registry()
Q_ = u.Quantity


class Magnetization(Simulation):
    """Magnetization

    Base class for all magnetization simulations. It prepares the inputs and
    the material parameters of the magnetic layers and caches the results.
    The magnetization maps are given in cartesian coordinates ``(x, y, z)``
    by default and in polar coordinates ``(amplitude, phi, gamma)`` for
    ``polar=True``, see :meth:`get_magnetization_map`. The physics - the right-hand sides
    and the time integration - lives in
    :mod:`~udkm1Dsim.simulations.solvers.magnetization_scipy` and the
    equilibrium quantities of the mean-field model in
    :mod:`~udkm1Dsim.simulations.solvers.mean_field`.

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
        ode_options (dict): options for scipy solve_ivp ode solver. The
            default ``method`` is ``LSODA``, which switches automatically
            between a stiff and a non-stiff integrator, since the strong
            exchange between sublattices, the inertia of the iLLG, and the
            longitudinal relaxation of the LLB at low temperatures make the
            equations stiff.
        temp_sub_system (int): index of the sub-system of the ``temp_map``
            whose temperature drives the magnetization dynamics.
        equilibrium_options (dict): ``tol``, the largest remaining torque (T),
            and ``max_time``, the largest time of the overdamped flow (s), of
            the relaxation into the equilibrium, see
            :meth:`calc_equilibrium_magnetization`.

    """

    _title = "Magnetization simulation"

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.ode_options = {
            "method": "LSODA",
            "first_step": None,
            "max_step": np.inf,
            "rtol": 1e-3,
            "atol": 1e-6,
        }
        self.temp_sub_system = 0
        self.equilibrium_options = {"tol": 1e-6, "max_time": 1e-6}

    def _property_rows(self):
        """extend the simulation properties by the temperature sub-system"""
        return super()._property_rows() + [["temperature sub-system", self.temp_sub_system]]

    def _hash_params(self):
        """simulation settings that enter the hash of the results"""
        return [type(self).__name__, self.temp_sub_system]

    def get_hash(self, **kwargs):
        """get_hash

        Calculates an unique hash given by the delays as well as the sample
        structure hash for relevant magnetic parameters, the type of the
        simulation and the temperature sub-system.
        Optionally, part of the ``strain_map`` and ``temp_map`` are used.

        Args:
            **kwargs (ndarray[float], optional): optional delays, strain and
                temperature profile as well as external magnetic field and
                initial magnetization.

        Returns:
            hash (str): unique hash.

        """
        param = self._hash_params()
        if kwargs.get("equilibrate"):
            param.append(self.equilibrium_options)

        for name in ("strain_map", "temp_map", "mag_amplitude_map"):
            if name in kwargs:
                value = kwargs.pop(name)
                if value is not None and np.size(value) > 1e6:
                    value = np.asarray(value).flatten()[0:1000000]
                param.append(value)

        for value in kwargs.values():
            param.append(value)

        return self.S.get_hash(types="magnetic") + "_" + make_hash_md5(param)

    @property
    def num_sublattices(self):
        """int: largest number of magnetic sublattices of all layers."""
        layers = self.S.unique_layers[1]
        return max((layer.magnetic.num_sublattices for layer in layers), default=1)

    def check_initial_magnetization(self, init_magnetization, distances=[]):
        r"""check_initial_magnetization

        Check if a given initial magnetization profile is valid and return it
        for all sublattices of all layers as polar array
        :math:`N \times P \times 3` with the components
        ``[amplitude, phi, gamma]`` in radians, where :math:`N` is the number
        of layers or the length of the specific spatial grid and :math:`P`
        the largest number of sublattices. Layers with fewer sublattices are
        padded with zeros.

        If no initial magnetization is given, the profile is determined from
        the magnetization of the layers. Otherwise it is given in any form of
        :meth:`parse_initial_magnetization`, i.e. in cartesian coordinates by
        default, and is either a single
        vector or an :math:`N \times 3` array with one vector per layer for
        structures with a single sublattice, and a :math:`P \times 3` array
        with one vector per sublattice for all layers or the full
        :math:`N \times P \times 3` array for several sublattices. Since the
        sublattices of a layer can point in arbitrary directions, e.g.
        antiparallel for a ferrimagnet, a vector per layer is not accepted for
        several sublattices. For as many layers as sublattices, an array
        :math:`P \times 3` is one vector per sublattice. In addition, a
        spatial grid can be provided.

        Args:
            init_magnetization (ndarray[float], dict, VectorParameter): initial
                global or local magnetization profile, see
                :meth:`parse_initial_magnetization`.
            distances (ndarray[float, Quantity], optional): spatial grid of the
                initial magnetization.

        Returns:
            init_mag (ndarray[float]): checked initial magnetization
            :math:`N \times P \times 3` on the according spatial grid.

        """
        layers = self.S.unique_layers
        for layer in layers[1]:
            if isinstance(layer, UnitCell):
                raise TypeError(
                    "UnitCells are not yet supported in Magnetization simulations! See issue #129."
                )

        try:
            distances = distances.to("m").magnitude
        except AttributeError:
            pass

        N = len(distances)
        if N == 0:
            # no spatial grid is provided
            N = self.S.num_layers
            [distances, _, _] = self.S._layer_distances()
        # use finderb search to find the corresponding indices between the
        # internal and external spatial grids
        [d_start, _, _] = self.S._layer_distances()
        idx = finderb(distances, d_start)
        handles = self.S.layer_vectors[2]
        sizes = np.array([handles[i].magnetic.num_sublattices for i in idx])
        P = int(np.max(sizes, initial=1))

        init_mag = self.parse_initial_magnetization(init_magnetization)
        if len(init_mag) == 0:
            self.disp_message(
                "No explicit initial magnetization given - use magnetization of layers instead."
            )
            result = np.zeros([N, P, 3])
            for k, i in enumerate(idx):
                for a, vector in enumerate(handles[i].magnetic.sublattice_magnetizations()):
                    result[k, a] = vector.polar_rad
            return result

        init_mag = np.array(init_mag, dtype=float)
        if P == 1 and init_mag.shape == (3,):
            # it is the same initial magnetization for all layers
            result = np.tile(init_mag, (N, 1, 1))
        elif P == 1 and init_mag.shape == (N, 3):
            result = init_mag[:, np.newaxis, :]
        elif P > 1 and init_mag.shape == (P, 3):
            # the same initial magnetization of the sublattices for all layers
            result = np.tile(init_mag, (N, 1, 1))
        elif init_mag.shape == (N, P, 3):
            result = init_mag.copy()
        elif P > 1:
            raise ValueError(
                f"The initial magnetization of the {P:d} sublattices must be given as "
                "array of Px3 for all layers or NxPx3, where N is the number of layers "
                "in the structure or the length of the spatial grid provided as distances "
                "vector!"
            )
        else:
            raise ValueError(
                "The initial magnetization array must have 3 or Nx3 elements, where N is "
                "the number of layers in the structure or the length of the spatial grid "
                "provided as distances vector!"
            )
        # sublattices beyond the number of a layer are padding
        result[np.arange(P)[np.newaxis, :] >= sizes[:, np.newaxis]] = 0.0
        return result

    def get_magnetization_map(self, delays, polar=False, **kwargs):
        r"""get_magnetization_map

        Returns an absolute ``magnetization_map`` for the sample structure with
        the dimension :math:`M \times N \times 3` with :math:`M` being the
        number of delays and :math:`N` the number of layers in the structure or
        the length of the given spatial grid. Each element of the map contains
        the three components of the magnetization as plain numpy array

        - in cartesian coordinates ``(x, y, z)`` by default, as the
          ``magnetization`` of the layers, or
        - in polar coordinates ``(amplitude, phi, gamma)`` for ``polar=True``
          with the polar angle ``phi`` from the :math:`z`-axis and the
          azimuthal angle ``gamma`` in the :math:`xy`-plane from the
          :math:`x`-axis in radians, see
          :func:`~udkm1Dsim.helpers.convert_cartesian_to_polar`.

        The default changed from polar to cartesian coordinates in version
        2.6.1. Both forms are calculated from the same cached results and are
        accepted by :class:`~udkm1Dsim.simulations.scattering.elzo.Elzo` with the
        same ``polar`` switch.

        For layers with several sublattices it is the net magnetization, i.e.
        the sum of the sublattice magnetizations weighted by their saturation
        magnetizations :math:`M_{s,a}` and normalized to their sum
        :math:`\sum_a M_{s,a}\mathbf{m}_a / \sum_a M_{s,a}`, see
        :meth:`get_sublattice_magnetization_map` for the sublattices.
        The ``magnetization_map`` can depend on the ``temp_map`` and
        ``strain_map`` that can be also calculated for the sample structure.
        More over an external magnetic field ``B_ext`` and initial magnetization
        profile ``init_magnetization`` can be provided.

        Args:
            delays (ndarray[Quantity]): delays range of simulation (s).
            polar (bool, optional): return the map in polar instead of
                cartesian coordinates. Defaults to False.
            **kwargs (ndarray[float], Result, optional): optional strain and
                temperature profile, also as results of
                :meth:`~udkm1Dsim.simulations.phonons.Phonon.get_strain_map`
                and :meth:`~udkm1Dsim.simulations.heat.Heat.get_temp_map`, as
                well external magnetic field in T also as quantity,
                either constant or one vector per delay, and initial
                magnetization.

        Returns:
            result (Result): with the spatio-temporal ``magnetization_map`` of
            the dimensions ``delays``, ``distances`` of the layer centers, and
            ``components``, and the metadata ``polar``, see
            :class:`~udkm1Dsim.simulations.result.Result`.

        """
        return self._select_coordinates(
            self._get_magnetization_maps(delays, **kwargs), "magnetization_map", polar
        )

    def get_sublattice_magnetization_map(self, delays, polar=False, **kwargs):
        r"""get_sublattice_magnetization_map

        Returns the ``sublattice_magnetization_map`` of all sublattices of the
        layers with the dimension :math:`M \times N \times P \times 3`, with
        :math:`P` being the largest number of sublattices, see
        :meth:`get_magnetization_map` for the arguments and the cartesian or
        polar coordinates. The sublattices of layers with fewer sublattices are
        zero. It is calculated and cached together with the net
        ``magnetization_map``.

        Args:
            delays (ndarray[Quantity]): delays range of simulation (s).
            polar (bool, optional): return the map in polar instead of
                cartesian coordinates. Defaults to False.
            **kwargs (ndarray[float], Result, optional): optional strain and
                temperature profile, also as results of
                :meth:`~udkm1Dsim.simulations.phonons.Phonon.get_strain_map`
                and :meth:`~udkm1Dsim.simulations.heat.Heat.get_temp_map`, as
                well external magnetic field in T also as quantity,
                either constant or one vector per delay, and initial
                magnetization.

        Returns:
            result (Result): with the spatio-temporal
            ``sublattice_magnetization_map`` of the dimensions ``delays``,
            ``distances`` of the layer centers, ``sublattices``, and
            ``components``, and the metadata ``polar``, see
            :class:`~udkm1Dsim.simulations.result.Result`.

        """
        return self._select_coordinates(
            self._get_magnetization_maps(delays, **kwargs), "sublattice_magnetization_map", polar
        )

    @staticmethod
    def _to_coordinates(magnetization_map, polar):
        """the cartesian magnetization map in polar coordinates for
        ``polar=True``"""
        return convert_cartesian_to_polar(magnetization_map) if polar else magnetization_map

    @staticmethod
    def _select_coordinates(maps, name, polar):
        """result with the cartesian magnetization map ``name`` of the
        ``maps``, which is converted to polar coordinates for ``polar=True``"""
        result = maps.select(name)
        if polar:
            result.add(name, Magnetization._to_coordinates(result[name], polar), result.dims(name))
        result.attrs["polar"] = polar
        return result

    def _map_result(self, title, shape, delays=None, source=None):
        """new result for maps of the ``shape`` (M, N, ...), see
        :meth:`~udkm1Dsim.simulations.simulation.Simulation.new_map_result`"""
        return self.new_map_result(
            title, delays, source=source, num_delays=shape[0], num_distances=shape[1]
        )

    @staticmethod
    def _parse_input_maps(kwargs):
        """the input maps of the ``kwargs`` given as arrays or as results of
        the upstream simulations and the first of these results"""
        names = {
            "temp_map": "temp_map",
            "strain_map": "strain_map",
            "mag_amplitude_map": "mean_field_magnetization_map",
        }
        source = find_source(*(kwargs.get(key) for key in names))
        for key, name in names.items():
            if key in kwargs:
                kwargs[key] = get_entry(kwargs[key], name)
        return kwargs, source

    def _get_magnetization_maps(self, delays, **kwargs):
        """the cartesian net and sublattice magnetization maps, which are
        calculated and cached together, see :meth:`get_magnetization_map`"""
        if "init_mag" in kwargs:
            raise TypeError(
                "init_mag is replaced by init_magnetization, which is given in cartesian "
                "coordinates (x, y, z), or as dict or VectorParameter in polar coordinates, "
                "see parse_initial_magnetization!"
            )
        kwargs, source = self._parse_input_maps(dict(kwargs))
        if kwargs.get("B_ext") is not None:
            kwargs["B_ext"] = self.parse_field(kwargs["B_ext"])
        # create a hash of all simulation parameters with the initial
        # magnetization in a unique form
        hash_kwargs = dict(kwargs)
        if "init_magnetization" in kwargs:
            hash_kwargs["init_magnetization"] = self.parse_initial_magnetization(
                kwargs["init_magnetization"]
            )

        def calc():
            # parse the input arguments
            if kwargs.get("strain_map") is not None:
                if not isinstance(kwargs["strain_map"], np.ndarray):
                    raise TypeError("strain_map must be a numpy ndarray!")
            for name in ("temp_map", "mag_amplitude_map"):
                if kwargs.get(name) is not None and not isinstance(kwargs[name], np.ndarray):
                    raise TypeError(f"{name} must be a numpy ndarray!")
            if "B_ext" in kwargs:
                if not isinstance(kwargs["B_ext"], np.ndarray):
                    raise TypeError("B_ext must be a numpy ndarray!")
                elif kwargs["B_ext"].shape not in [(3,), (np.size(delays), 3)]:
                    raise ValueError(
                        "B_ext must be a vector with 3 components (B_x, B_y, B_z) or an "
                        "array of Mx3 with M being the number of delays!"
                    )
            if len(hash_kwargs.get("init_magnetization", [])) > 0:
                # the shape is checked by check_initial_magnetization
                self.check_initial_magnetization(kwargs["init_magnetization"])

            maps = self.calc_magnetization_map(delays, **kwargs)
            if isinstance(maps, tuple):
                magnetization_map, sublattice_map = maps
            else:
                # simulations without sublattices
                magnetization_map = maps
                sublattice_map = maps[:, :, np.newaxis, :]

            result = self._map_result(
                "magnetization map", np.shape(magnetization_map), delays, source
            )
            dims = ("delays", "distances")
            result.add("magnetization_map", magnetization_map, dims + ("components",))
            result.add(
                "sublattice_magnetization_map",
                sublattice_map,
                dims + ("sublattices", "components"),
            )
            return result

        return self._cached_result(
            "magnetization_map", self.get_hash(delays=delays, **hash_kwargs), calc
        )

    def calc_magnetization_map(self, delays, **kwargs):
        """calc_magnetization_map

        Calculates an absolute ``magnetization_map`` in cartesian coordinates
        ``(x, y, z)`` - see :meth:`get_magnetization_map` for details.

        This method is just an interface and should be overwritten for the
        actual simulations. Since version 2.6.1 it must return cartesian
        maps, which can be converted from polar coordinates by
        :func:`~udkm1Dsim.helpers.convert_polar_to_cartesian`.

        Args:
            delays (ndarray[Quantity]): delays range of simulation (s).
            **kwargs (ndarray[float], optional): optional strain and
                temperature profile as well external magnetic field and initial
                magnetization.

        Returns:
            (tuple):
            - *magnetization_map (ndarray[float])* - spatio-temporal absolute
              cartesian net magnetization profile :math:`M \times N \times 3`.
            - *sublattice_magnetization_map (ndarray[float])* - spatio-temporal
              absolute cartesian magnetization profile of the sublattices
              :math:`M \times N \times P \times 3`.

            Simulations without sublattices may only return the net map.

        """
        raise NotImplementedError

    def get_mean_field_mag_map(self, temp_map):
        r"""get_mean_field_mag_map

        Returns the mean-field magnetization map see
        :meth:`calc_mean_field_mag_map` for details. The dimension of the map
        are :math:`M \times N` with :math:`M` being the number of delays and
        :math:`N` the number of layers in the structure. For several
        sublattices it is the amplitude of the net magnetization, see
        :meth:`get_sublattice_mean_field_mag_map` for the sublattices.

        Args:
            temp_map (ndarray[float], Result): spatio-temporal electron
                temperature map or result of
                :meth:`~udkm1Dsim.simulations.heat.Heat.get_temp_map`, whose
                ``temp_sub_system`` is used.

        Returns:
            result (Result): with the spatio-temporal
            ``mean_field_magnetization_map`` of the dimensions ``delays`` and
            ``distances``, whose coordinates are taken from a given result of
            the heat simulation, see
            :class:`~udkm1Dsim.simulations.result.Result`.

        """
        return self._get_mean_field_mag_maps(temp_map).select("mean_field_magnetization_map")

    def get_sublattice_mean_field_mag_map(self, temp_map):
        r"""get_sublattice_mean_field_mag_map

        Returns the mean-field magnetization map of all sublattices with the
        dimension :math:`M \times N \times P`, see
        :meth:`calc_mean_field_mag_map` for details. It is calculated and
        cached together with the net mean-field magnetization.

        Args:
            temp_map (ndarray[float], Result): spatio-temporal electron
                temperature map or result of
                :meth:`~udkm1Dsim.simulations.heat.Heat.get_temp_map`, whose
                ``temp_sub_system`` is used.

        Returns:
            result (Result): with the spatio-temporal
            ``sublattice_mean_field_magnetization_map`` of the dimensions
            ``delays``, ``distances``, and ``sublattices``, see
            :meth:`get_mean_field_mag_map`.

        """
        return self._get_mean_field_mag_maps(temp_map).select(
            "sublattice_mean_field_magnetization_map"
        )

    def _get_mean_field_mag_maps(self, temp_map):
        """the net and the sublattice mean-field magnetization maps, which are
        calculated and cached together, see :meth:`get_mean_field_mag_map`"""
        source = temp_map
        temp_map = get_entry(temp_map, "temp_map")
        if isinstance(source, Result) and np.ndim(temp_map) == 3:
            temp_map = temp_map[:, :, self.temp_sub_system]

        def calc():
            if not isinstance(temp_map, np.ndarray):
                raise TypeError("temp_map must be a numpy ndarray!")
            mf_mag_map, sublattice_map = self._calc_mean_field_mag_maps(temp_map)
            result = self._map_result(
                "mean-field magnetization map", np.shape(mf_mag_map), source=source
            )
            dims = ("delays", "distances")
            result.add("mean_field_magnetization_map", mf_mag_map, dims, aliases="mf_mag_map")
            result.add(
                "sublattice_mean_field_magnetization_map",
                sublattice_map,
                dims + ("sublattices",),
                aliases="sublattice_mf_mag_map",
            )
            return result

        return self._cached_result("mf_magnetization_map", self.get_hash(temp_map=temp_map), calc)

    def calc_mean_field_mag_map(self, temp_map):
        r"""calc_mean_field_mag_map

        Calculate the mean-field magnetization map :math:`m_\mathrm{eq}` of
        every layer, the non-trivial solution of the self-consistent equation

        .. math::

            m_\mathrm{eq} = B_S\left(\frac{J\, m_\mathrm{eq}}{k_\mathrm{B} T}\right)

        with the Brillouin function :math:`B_S` of the effective spin - the
        Langevin function for an infinite spin - or of the coupled equations
        of several sublattices, see
        :func:`~udkm1Dsim.simulations.solvers.mean_field.mean_field_magnetization_coupled`,
        which is tabulated on a temperature grid, see
        :func:`~udkm1Dsim.simulations.solvers.mean_field.equilibrium_tables`,
        and interpolated onto the ``temp_map``. The magnetization is zero for
        non-magnetic layers and at and above the Curie temperature. The net
        mean-field magnetization of several sublattices is the amplitude of
        their sum in the collinear ground state, weighted by their saturation
        magnetizations, see :meth:`get_magnetization_map`.

        Args:
            temp_map (ndarray[float]): spatio-temporal electron temperature map
                :math:`M \times N`.

        Returns:
            mf_mag_map (ndarray[float]): spatio-temporal net mean-field
            magnetization map :math:`M \times N`.

        """
        return self._calc_mean_field_mag_maps(temp_map)[0]

    def _calc_mean_field_mag_maps(self, temp_map):
        """the net mean-field magnetization map M x N and the one of the
        sublattices M x N x P, see :meth:`calc_mean_field_mag_map`"""
        temp_map = np.asarray(temp_map, dtype=float)
        L = self.S.num_layers
        layers = self.calc_magnetic_parameters(np.ones(L, dtype=bool))
        temp_grid, tables = magnetization_scipy.tabulate_equilibrium(
            layers, temp_max=float(np.max(temp_map))
        )
        sublattice_map = interpolate_tables_map(temp_grid, tables, temp_map)[..., TABLE["mag"]]
        # the signs of the sublattices in the collinear ground state
        mag = magnetization_scipy.stack_magnetic_parameters(layers)
        signs = np.zeros_like(mag.active)
        for k, layer in enumerate(layers):
            P = layer.num_sublattices
            signs[k, :P] = sublattice_signs(
                layer.sublattice_values("mf_exch_coupling"), layer.sublattice_values("eff_spin")
            )
        weights = self._net_weights(mag) * signs
        mf_mag_map = np.abs(np.einsum("mnp,np->mn", sublattice_map, weights))
        return mf_mag_map, sublattice_map

    @staticmethod
    def _net_weights(mag):
        """weights of the sublattices for the net magnetization of the layers,
        their saturation magnetizations normalized to their sum, or equal
        weights for layers without saturation magnetization"""
        weights = mag.mag_saturation * mag.active if hasattr(mag, "mag_saturation") else mag.active
        total = np.sum(weights, axis=1)
        weights = np.where((total > 0)[:, np.newaxis], weights, mag.active)
        total = np.sum(weights, axis=1)
        return weights / np.where(total > 0, total, 1.0)[:, np.newaxis]

    def calc_directional_exchange_stiffnesses(self):
        r"""calc_directional_exchange_stiffnesses

        Returns the directional exchange stiffnesses with size
        :math:`N \times 2`, with :math:`N` being the number of layers, between
        the :math:`(i-1)^\mathrm{th}` and :math:`i^\mathrm{th}` layers as well
        as between the :math:`i^\mathrm{th}` and :math:`(i+1)^\mathrm{th}`
        layers in the structure.
        In case two neighboring layers are identical the center entry
        of the 3-component `exchange_stiffness` is used. In case of an interface
        to the top :math:`(i-1)\rightarrow i` it will be the first entry and
        for an interface to the bottom :math:`i\rightarrow (i+1)` it will be
        the third entry.

        Returns:
            A (ndarray[float]): directional exchange stiffnesses.

        """
        exch_stiffnesses = self.S.get_layer_property_vector("magnetic.exch_stiffness.magnitude")

        indices, _, _ = self.S.layer_vectors

        A = np.zeros([len(indices), 2])
        interfaces = np.r_[1, np.diff(indices), 1]
        interfaces[interfaces != 0] = -1
        select = (interfaces + 1).astype(np.int16)

        A[:, 0] = exch_stiffnesses[np.arange(len(select[0:-1])), select[0:-1]]

        interfaces[interfaces != 0] = 1
        select = (interfaces + 1).astype(np.int16)
        A[:, 1] = exch_stiffnesses[np.arange(len(select[1:])), select[1:]]

        return A

    def calc_magnetic_parameters(self, is_magnetic):
        r"""calc_magnetic_parameters

        Returns the
        :class:`~udkm1Dsim.structures.parameter_groups.MagneticParameters` of
        the magnetic layers, as they are passed to the solver. Repeated
        layers share the same parameters.

        Args:
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.

        Returns:
            layers (list[MagneticParameters]): magnetic parameters of the
            magnetic layers.

        """
        handles = self.S.layer_vectors[2]
        return [handles[i].magnetic for i in np.flatnonzero(is_magnetic)]

    def calc_interlayer_exchange_stiffnesses(self, is_magnetic):
        r"""calc_interlayer_exchange_stiffnesses

        Returns the :meth:`calc_directional_exchange_stiffnesses` of the
        magnetic layers with size :math:`N \times 2` towards the upper and
        lower magnetic layer. The exchange coupling between two magnetic
        layers that are not direct neighbours in the structure, i.e.
        separated by a non-magnetic spacer, as well as at the top and bottom
        of the stack is set to zero.

        Args:
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.

        Returns:
            A (ndarray[float]): exchange stiffnesses of the magnetic layers.

        """
        idx = np.flatnonzero(is_magnetic)
        A = self.calc_directional_exchange_stiffnesses()[idx, :].copy()
        if len(idx) > 1:
            adjacent = np.diff(idx) == 1
            A[1:, 0] *= adjacent
            A[:-1, 1] *= adjacent
        if len(idx) > 0:
            A[0, 0] = 0
            A[-1, 1] = 0
        return A

    def check_temp_map(self, temp_map, M, L):
        r"""check_temp_map

        Checks the ``temp_map`` and returns the temperature of the
        ``temp_sub_system`` as :math:`M \times L` array.

        Args:
            temp_map (ndarray[float]): temperature map :math:`M \times L` or
                :math:`M \times L \times K`.
            M (int): number of delays.
            L (int): number of layers.

        Returns:
            temps (ndarray[float]): temperature map :math:`M \times L`.

        """
        return self.check_map("temp_map", temp_map, M, L, self.temp_sub_system)

    @staticmethod
    def check_map(name, values, M, L, sub_system=None):
        r"""check_map

        Checks a spatio-temporal map of the layers, e.g. the ``strain_map``,
        and returns it as :math:`M \times L` array. For a map
        :math:`M \times L \times K` with several sub-systems, the given
        ``sub_system`` is selected.

        Args:
            name (str): name of the map for the error messages.
            values (ndarray[float]): map :math:`M \times L` or
                :math:`M \times L \times K`.
            M (int): number of delays.
            L (int): number of layers.
            sub_system (int, optional): sub-system of a map with several
                sub-systems. Defaults to None.

        Returns:
            values (ndarray[float]): map :math:`M \times L`.

        """
        values = np.asarray(values, dtype=float)
        if values.ndim == 3 and sub_system is not None:
            if sub_system >= values.shape[2]:
                raise ValueError(f"Sub-system {sub_system:d} is not contained in {name}!")
            values = values[:, :, sub_system]
        if values.shape != (M, L):
            raise ValueError(
                f"{name} must have the dimension M x L"
                + (" (x K)" if sub_system is not None else "")
                + ", where M is the number of delays and L the number of layers!"
            )
        return values

    @staticmethod
    def parse_delays(delays):
        """parse_delays

        Returns the delays as plain array in seconds.

        Args:
            delays (ndarray[Quantity, float]): delays (s).

        Returns:
            delays (ndarray[float]): delays (s).

        """
        try:
            delays = delays.to("s").magnitude
        except AttributeError:
            pass
        return np.asarray(delays, dtype=float)

    @staticmethod
    def parse_field(B_ext):
        """parse_field

        Returns the external magnetic field as plain array in tesla, if it is
        given as quantity, otherwise unchanged.

        Args:
            B_ext (ndarray[Quantity, float]): external magnetic field (T).

        Returns:
            B_ext (ndarray[float]): external magnetic field (T).

        """
        if isinstance(B_ext, u.Quantity):
            return np.asarray(B_ext.to("T").magnitude, dtype=float)
        return B_ext

    @staticmethod
    def parse_initial_magnetization(init_magnetization):
        r"""parse_initial_magnetization

        Returns the initial magnetization as plain polar array
        ``[amplitude, phi, gamma]`` in radians, with the vectors in its last
        dimension. As for the ``magnetization`` of the layers, see
        :class:`~udkm1Dsim.structures.parameters.VectorParameter`, it can be
        given as

        - cartesian components ``(x, y, z)``, e.g. ``(1, 0, 0)`` or an array
          of them with the components in its last dimension,
        - dict of the polar components ``{"r": ..., "phi": ..., "gamma": ...}``
          with angles as quantities or bare numbers in degrees, where a
          negative ``r`` reverses the direction given by the angles,
        - dict of the cartesian components ``{"x": ..., "y": ..., "z": ...}``,
        - :class:`~udkm1Dsim.structures.parameters.VectorParameter` or
          (nested) lists of them, e.g. one per sublattice.

        - :class:`~udkm1Dsim.simulations.result.Result` of
          :meth:`get_equilibrium_magnetization` in cartesian or polar
          coordinates.

        The components of the dicts can be arrays of the same shape. A
        cartesian ``magnetization_map`` at a single delay, e.g. of
        :meth:`calc_equilibrium_magnetization`, can be used directly as array.
        An empty input gives an empty list, i.e. the magnetization of the
        layers is used.

        Args:
            init_magnetization (ndarray[float], dict, VectorParameter, Result):
                initial magnetization.

        Returns:
            init_mag (ndarray[float]): initial magnetization in polar
            coordinates in radians.

        """
        if init_magnetization is None:
            return []
        if isinstance(init_magnetization, Result):
            equilibrium = init_magnetization["equilibrium_magnetization"]
            if init_magnetization.attrs.get("polar", False):
                return np.array(equilibrium, dtype=float)
            return convert_cartesian_to_polar(np.asarray(equilibrium, dtype=float))
        if isinstance(init_magnetization, VectorParameter):
            return np.array(init_magnetization.polar_rad, dtype=float)
        if isinstance(init_magnetization, dict):
            keys = set(init_magnetization)
            if keys == {"r", "phi", "gamma"}:
                r = init_magnetization["r"]
                r = r.m_as("") if isinstance(r, u.Quantity) else r
                angles = [
                    value.m_as("rad") if isinstance(value, u.Quantity) else np.radians(value)
                    for value in (init_magnetization["phi"], init_magnetization["gamma"])
                ]
                return np.stack(np.broadcast_arrays(r, *angles), axis=-1).astype(float)
            if keys == {"x", "y", "z"}:
                components = [
                    value.m_as("") if isinstance(value, u.Quantity) else value
                    for value in (init_magnetization[k] for k in "xyz")
                ]
                cartesian = np.stack(np.broadcast_arrays(*components), axis=-1)
                return convert_cartesian_to_polar(cartesian.astype(float))
            raise ValueError(
                "A dict of the initial magnetization must hold either the polar components "
                "r, phi, and gamma or the cartesian components x, y, and z!"
            )

        def contains_vectors(value):
            return isinstance(value, VectorParameter) or (
                isinstance(value, (list, tuple)) and any(contains_vectors(v) for v in value)
            )

        def polar_of_vectors(value):
            if isinstance(value, VectorParameter):
                return value.polar_rad
            return [polar_of_vectors(v) for v in value]

        if isinstance(init_magnetization, (list, tuple)) and contains_vectors(init_magnetization):
            return np.array(polar_of_vectors(init_magnetization), dtype=float)
        if isinstance(init_magnetization, u.Quantity):
            init_magnetization = init_magnetization.m_as("")
        cartesian = np.asarray(init_magnetization, dtype=float)
        if cartesian.size == 0:
            return []
        if cartesian.shape[-1] != 3:
            raise ValueError(
                "The initial magnetization must be given with the 3 cartesian components "
                "(x, y, z) in its last dimension!"
            )
        return convert_cartesian_to_polar(cartesian)

    def get_equilibrium_magnetization(
        self, B_ext=np.array([0, 0, 0]), init_magnetization=None, polar=False, **profiles
    ):
        r"""get_equilibrium_magnetization

        Returns the magnetization of all sublattices of all layers in the
        equilibrium of the given external field and spatial profiles, see
        :meth:`calc_equilibrium_magnetization` for the arguments. The result
        can be used directly as ``init_magnetization`` of
        :meth:`get_magnetization_map`, also in polar coordinates.

        Args:
            B_ext (ndarray[Quantity, float], optional): external magnetic
                field (B_x, B_y, B_z) (T).
            init_magnetization (ndarray, dict, VectorParameter, Result, optional):
                initial magnetization, see :meth:`parse_initial_magnetization`.
            polar (bool, optional): return the magnetization in polar instead
                of cartesian coordinates. Defaults to False.
            **profiles (ndarray[float], optional): the spatial profiles of
                the model at a single time, see
                :meth:`calc_equilibrium_magnetization`.

        Returns:
            result (Result): with the ``equilibrium_magnetization`` of the
            dimensions ``distances`` of the layer centers, ``sublattices``,
            and ``components``, and the metadata ``polar``, see
            :class:`~udkm1Dsim.simulations.result.Result`.

        """
        equilibrium = self.calc_equilibrium_magnetization(
            B_ext, init_magnetization, polar, **profiles
        )
        result = self.new_map_result("equilibrium magnetization")
        result.add(
            "equilibrium_magnetization",
            equilibrium,
            ("distances", "sublattices", "components"),
        )
        result.attrs["polar"] = polar
        return result

    def calc_equilibrium_magnetization(
        self, B_ext=np.array([0, 0, 0]), init_magnetization=None, polar=False, **profiles
    ):
        r"""calc_equilibrium_magnetization

        Returns the magnetization of all sublattices of all layers in the
        equilibrium of the given external field and spatial profiles, e.g. of
        the temperature and strain before an excitation, as cartesian array
        :math:`L \times P \times 3` (or polar for ``polar=True``, as before
        version 2.6.1), see :meth:`get_magnetization_map`. The cartesian array
        can be used directly as ``init_magnetization`` of
        :meth:`get_magnetization_map`, see
        :meth:`parse_initial_magnetization`. Starting from the initial
        magnetization ``init_magnetization`` or the magnetization of the
        layers, the directions are
        relaxed by an overdamped flow without precession into the nearest
        equilibrium, see
        :func:`~udkm1Dsim.simulations.solvers.magnetization_scipy.relax_directions`
        and :attr:`equilibrium_options`. The amplitudes are those of the model
        at the given profiles, i.e. the mean-field magnetization for the
        :class:`LLB`, which neglects the tiny increase of the amplitude by
        the longitudinal components of the field. The initial directions
        select the equilibrium if there are several, e.g. the branch of a
        hysteresis, and an unstable
        equilibrium, e.g. a magnetization exactly perpendicular to an easy
        axis, is not left.

        Args:
            B_ext (ndarray[Quantity, float], optional): external magnetic
                field (B_x, B_y, B_z) (T).
            init_magnetization (ndarray, dict, VectorParameter, Result, optional):
                initial magnetization, see :meth:`parse_initial_magnetization`.
            polar (bool, optional): return the magnetization in polar instead
                of cartesian coordinates. Defaults to False.
            **profiles (ndarray[float], optional): the spatial profiles of
                the maps of :meth:`get_magnetization_map` of the model at a
                single time, e.g. ``temp_map`` of the layers :math:`L` or
                :math:`L \times K`, ``strain_map`` :math:`L`, or
                ``mag_amplitude_map`` :math:`L` or :math:`L \times P`.

        Returns:
            equilibrium_mag (ndarray[float]): magnetization of all
            sublattices in equilibrium :math:`L \times P \times 3`.

        """
        if "init_mag" in profiles:
            raise TypeError(
                "init_mag is replaced by init_magnetization, which is given in cartesian "
                "coordinates (x, y, z), or as dict or VectorParameter in polar coordinates, "
                "see parse_initial_magnetization!"
            )
        maps = {
            name: None if value is None else np.asarray(value, dtype=float)[np.newaxis, ...]
            for name, value in profiles.items()
        }
        _, sublattice_map = self.calc_magnetization_map(
            np.zeros(1),
            B_ext=np.asarray(self.parse_field(B_ext), dtype=float),
            init_magnetization=init_magnetization,
            equilibrate=True,
            **maps,
        )
        return self._to_coordinates(sublattice_map[0], polar)

    def equilibrate_directions(self, init_dir, amplitudes, inputs, is_magnetic):
        r"""equilibrate_directions

        Relaxes the directions of the magnetization of the magnetic layers into
        the equilibrium of the inputs at the first delay, see
        :meth:`calc_equilibrium_magnetization`. Warns if the remaining torque
        exceeds the ``tol`` of the :attr:`equilibrium_options` after their
        ``max_time``.

        Args:
            init_dir (ndarray[float]): initial unit vectors of the magnetic
                layers :math:`N \times P \times 3`.
            amplitudes (ndarray[float]): magnetization amplitudes
                :math:`N \times P`.
            inputs (Inputs): time-dependent inputs of the magnetic layers.
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.

        Returns:
            directions (ndarray[float]): unit vectors in equilibrium
            :math:`N \times P \times 3`.

        """
        B_ext, _, _, strains = magnetization_scipy.interpolate_inputs(inputs.delays[0], inputs)
        directions, torque = magnetization_scipy.relax_directions(
            init_dir,
            amplitudes,
            B_ext,
            strains,
            magnetization_scipy.stack_magnetic_parameters(
                self.calc_magnetic_parameters(is_magnetic)
            ),
            self.calc_interlayer_exchange_stiffnesses(is_magnetic),
            self.S.get_layer_property_vector("structural.thickness.magnitude")[is_magnetic],
            **self.equilibrium_options,
        )
        if torque > self.equilibrium_options["tol"]:
            warnings.warn(
                f"The magnetization did not reach its equilibrium - the largest remaining "
                f"torque is {torque:.3g} T. Consider a larger max_time or tol of the "
                "equilibrium_options."
            )
        return directions

    def solve(self, model, init_state, inputs, is_magnetic, **kwargs):
        r"""solve

        Solves the ``model`` for the magnetic layers, see
        :func:`~udkm1Dsim.simulations.solvers.magnetization_scipy.solve`.

        Args:
            model (str): one of ``llg``, ``illg`` or ``llb``.
            init_state (ndarray[float]): initial state of the solver.
            inputs (Inputs): time-dependent inputs of the magnetic layers.
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.
            **kwargs: ``ode_options`` replacing the ones of the simulation and
                ``exchange_relaxation`` of the LLB, see
                :func:`~udkm1Dsim.simulations.solvers.magnetization_scipy.solve`.

        Returns:
            mag_map (ndarray[float]): cartesian magnetization map of the
            magnetic layers :math:`M \times N \times P \times 3`.

        """
        return magnetization_scipy.solve(
            model,
            init_state,
            inputs,
            self.calc_magnetic_parameters(is_magnetic),
            self.calc_interlayer_exchange_stiffnesses(is_magnetic),
            self.S.get_layer_property_vector("structural.thickness.magnitude")[is_magnetic],
            kwargs.get("ode_options", self.ode_options),
            self.progress_bar,
            exchange_relaxation=kwargs.get("exchange_relaxation", False),
        )

    def embed_magnetization(self, mag_map, is_magnetic):
        r"""embed_magnetization

        Embeds the cartesian magnetization of the magnetic layers into the
        cartesian magnetization maps of all layers, which are zero for the
        non-magnetic layers. The net magnetization of a layer is the sum of its
        sublattice magnetizations weighted by their saturation magnetizations
        and normalized to their sum, see :meth:`get_magnetization_map`.

        Args:
            mag_map (ndarray[float]): cartesian magnetization map of the
                magnetic layers :math:`M \times N \times P \times 3`.
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.

        Returns:
            (tuple):
            - *magnetization_map (ndarray[float])* - cartesian net
              magnetization map :math:`M \times L \times 3`.
            - *sublattice_magnetization_map (ndarray[float])* - cartesian
              magnetization map of the sublattices
              :math:`M \times L \times P \times 3`.

        """
        M, N, P = mag_map.shape[0:3]
        P = max(P, self.num_sublattices)
        L = len(is_magnetic)
        sublattice_map = np.zeros([M, L, P, 3])
        sublattice_map[:, is_magnetic, : mag_map.shape[2], :] = mag_map
        magnetization_map = np.zeros([M, L, 3])
        if N > 0:
            mag = magnetization_scipy.stack_magnetic_parameters(
                self.calc_magnetic_parameters(is_magnetic)
            )
            weights = self._net_weights(mag)
            magnetization_map[:, is_magnetic, :] = np.einsum("mnpc,np->mnc", mag_map, weights)
        return magnetization_map, sublattice_map


class LLG(Magnetization):
    r"""LLG

    Landau-Lifshitz-Gilbert simulations with a prescribed magnetization amplitude.

    Ported from `friweber/udkm1Dsim_LLG-LLB
    <https://github.com/friweber/udkm1Dsim_LLG-LLB>`_ by
    `@friweber <https://github.com/friweber>`_.

    The LLG describes the precession and the transverse relaxation of the
    unit vector :math:`\mathbf{n}` of the magnetization of each layer

    .. math::

        \frac{d\mathbf{n}}{dt} = \frac{\gamma}{1+\alpha^2}\left(
            \mathbf{n} \times \mathbf{B}
            + \alpha\, \mathbf{n} \times (\mathbf{n} \times
            \mathbf{B})\right)

    with the Gilbert damping :math:`\alpha` (``gilbert_damping`` of the
    layers) and the gyromagnetic ratio :math:`\gamma` (``gyromagnetic_ratio``
    of the layers), which defaults to the one of a free electron
    :math:`\gamma_e = -1.761\times10^{11}\,\mathrm{rad\,s^{-1}\,T^{-1}}`,
    see
    :func:`~udkm1Dsim.simulations.solvers.magnetization_scipy.llg_rhs`. The
    amplitude :math:`m(t)` of the magnetization :math:`\mathbf{m} = m(t)\,
    \mathbf{n}`, i.e. the reduced magnetization :math:`m = |\mathbf{M}|/M_s`,
    which is 1 at saturation, is not a dynamic variable but an input, which is

    #. given directly as ``mag_amplitude_map``,
    #. taken from the sub-system ``mag_sub_system`` of the ``temp_map``,
       e.g. of a microscopic three-temperature model solved with the
       :class:`~.heat.Heat` class, where one sub-system holds the
       magnetization amplitude,
    #. derived from the temperature of the ``temp_sub_system`` of the
       ``temp_map`` by the mean-field model, see
       :meth:`~Magnetization.calc_mean_field_mag_map`, or
    #. kept at the initial amplitude of the layers.

    The amplitude must not be negative, a reversed direction is given by a
    negative amplitude of the initial magnetization only.

    The field :math:`\mathbf{B}` (T) is the sum of the external field
    ``B_ext``, the :func:`anisotropy field
    <udkm1Dsim.simulations.solvers.magnetization_scipy.anisotropy_field>`,
    the :func:`interlayer exchange field
    <udkm1Dsim.simulations.solvers.magnetization_scipy.interlayer_exchange_field>`
    the :func:`magnetoelastic field
    <udkm1Dsim.simulations.solvers.magnetization_scipy.magnetoelastic_field>`
    of the ``strain_map`` and, for layers with several sublattices, the
    :func:`mean-field exchange field of the other sublattices
    <udkm1Dsim.simulations.solvers.magnetization_scipy.sublattice_exchange_field>`,
    which couples the sublattices of ferri- and antiferromagnets. The
    temperature dependence of the anisotropy and magnetoelastic fields is
    scaled with the magnetization amplitude :math:`m(t)`. Magnetic layers are
    those with a saturation magnetization. Every sublattice :math:`a` of a
    layer, see :class:`~udkm1Dsim.structures.parameter_groups.MagneticParameters`,
    has its own unit vector :math:`\mathbf{n}_a`, amplitude :math:`m_a(t)` and
    parameters, see :meth:`~Magnetization.get_sublattice_magnetization_map`.

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
        ode_options (dict): options for scipy solve_ivp ode solver.
        temp_sub_system (int): index of the sub-system of the ``temp_map``
            whose temperature yields the mean-field magnetization amplitude.
        mag_sub_system (int): index of the sub-system of the ``temp_map``
            that holds the magnetization amplitude. Defaults to None, in
            which case the amplitude is derived from the temperature of the
            ``temp_sub_system`` by the mean-field model.

    """

    _title = "Landau-Lifshitz-Gilbert Magnetization Dynamics simulation"
    model = "llg"

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.mag_sub_system = None

    def _property_rows(self):
        """extend the simulation properties by the magnetization sub-system"""
        return super()._property_rows() + [["magnetization sub-system", self.mag_sub_system]]

    def _hash_params(self):
        return super()._hash_params() + [self.mag_sub_system]

    def calc_magnetization_map(
        self,
        delays,
        mag_amplitude_map=None,
        temp_map=None,
        B_ext=np.array([0, 0, 0]),
        init_magnetization=None,
        strain_map=None,
        equilibrate=False,
        **kwargs,
    ):
        r"""calc_magnetization_map

        Calculates the magnetization map using the Landau-Lifshitz-Gilbert
        equation for a given delay range and magnetization amplitude, see
        :class:`LLG`.

        Args:
            delays (ndarray[Quantity]): delays range of simulation (s).
            mag_amplitude_map (ndarray[float], optional): spatio-temporal
                amplitude of the magnetization :math:`M \times L`, i.e. the
                reduced magnetization :math:`m = |\mathbf{M}|/M_s`, which is 1
                at saturation and must not be negative. The LLG keeps it as
                prescribed and only evolves the direction.
            temp_map (ndarray[float], optional): spatio-temporal temperature
                map, used for the magnetization amplitude if no
                ``mag_amplitude_map`` is given, see :class:`LLG`.
            B_ext (ndarray[Quantity, float], optional): external magnetic
                field (B_x, B_y, B_z) (T), either constant or one vector per
                delay.
            init_magnetization (ndarray[float], dict, VectorParameter, optional):
                initial magnetization in cartesian coordinates by
                default, see :meth:`check_initial_magnetization`.
                Its amplitude is kept if neither ``mag_amplitude_map`` nor
                ``temp_map`` are given, otherwise only its direction is used.
            strain_map (ndarray[float], optional): spatio-temporal strain map
                :math:`M \times L` for the magnetoelastic field.
            equilibrate (boolean, optional): relax the directions of the
                initial magnetization into the equilibrium of the inputs at the
                first delay, see :meth:`calc_equilibrium_magnetization`.
                Defaults to False.
            **kwargs: model-specific options, see :class:`iLLG`.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
            cartesian magnetization profile.

        """
        t1 = time()
        delays = self.parse_delays(delays)
        M = len(delays)
        L = self.S.num_layers

        init_mag = self.check_initial_magnetization(init_magnetization)
        P = init_mag.shape[1]
        # the directions of the initial magnetization, a negative amplitude
        # reverses the direction
        init_dir = init_mag.copy()
        init_dir[..., 0] = np.where(init_mag[..., 0] < 0, -1.0, 1.0)
        init_dir = convert_polar_to_cartesian(init_dir)

        # the magnetization amplitude of all sublattices
        if mag_amplitude_map is not None:
            mag_amplitude_map = np.asarray(mag_amplitude_map, dtype=float)
            if mag_amplitude_map.ndim == 3:
                if mag_amplitude_map.shape != (M, L, P):
                    raise ValueError(
                        "mag_amplitude_map must have the dimension M x L or M x L x P, where "
                        "M is the number of delays, L the number of layers, and P the number "
                        "of sublattices!"
                    )
                amplitude_map = mag_amplitude_map
            else:
                amplitude_map = self.check_map("mag_amplitude_map", mag_amplitude_map, M, L)
        elif temp_map is not None:
            if self.mag_sub_system is not None:
                amplitude_map = self.check_map("temp_map", temp_map, M, L, self.mag_sub_system)
            else:
                amplitude_map = self.get_sublattice_mean_field_mag_map(
                    self.check_temp_map(temp_map, M, L)
                ).sublattice_mean_field_magnetization_map
        else:
            amplitude_map = np.tile(np.abs(init_mag[..., 0]), (M, 1, 1))
        if amplitude_map.ndim == 2:
            # the same amplitude for all sublattices
            amplitude_map = np.repeat(amplitude_map[:, :, np.newaxis], P, axis=2)
        if np.any(amplitude_map < 0):
            raise ValueError(
                "The magnetization amplitude must not be negative - a reversed direction "
                "is given by the direction of init_magnetization only!"
            )
        if strain_map is not None:
            strain_map = self.check_map("strain_map", strain_map, M, L)

        # magnetic layers carry a saturation magnetization
        handles = self.S.layer_vectors[2]
        is_magnetic = np.array(
            [np.any(np.asarray(h.magnetic.mag_saturation.magnitude) > 0) for h in handles],
            dtype=bool,
        )
        layers = self.calc_magnetic_parameters(is_magnetic)
        self.check_parameters(layers)
        P = magnetization_scipy.stack_magnetic_parameters(layers).active.shape[1]

        inputs = magnetization_scipy.prepare_inputs(
            delays,
            len(layers),
            B_ext,
            amplitude_map=amplitude_map[:, is_magnetic, :P],
            strain_map=None if strain_map is None else strain_map[:, is_magnetic],
        )
        init_dir = init_dir[is_magnetic, :P, :]
        if equilibrate:
            init_dir = self.equilibrate_directions(
                init_dir, inputs.amplitude_map[0], inputs, is_magnetic
            )
        init_state = self.calc_initial_state(init_dir, inputs, is_magnetic, **kwargs)
        mag_map = self.solve(self.model, init_state, inputs, is_magnetic)

        maps = self.embed_magnetization(mag_map, is_magnetic)
        self.disp_message(f"Elapsed time for _{type(self).__name__:s}_: {time() - t1:f} s")
        return maps

    def check_parameters(self, layers):
        """check_parameters

        Checks the parameters of the magnetic layers for the model.

        Args:
            layers (list[MagneticParameters]): magnetic parameters of the
                magnetic layers.

        """

    def calc_initial_state(self, init_dir, inputs, is_magnetic, **kwargs):
        r"""calc_initial_state

        Returns the initial state of the solver, which are the unit vectors of
        the magnetic layers for the LLG.

        Args:
            init_dir (ndarray[float]): initial unit vectors of the magnetic
                layers :math:`N \times P \times 3`.
            inputs (Inputs): time-dependent inputs of the magnetic layers.
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.
            **kwargs: model-specific options.

        Returns:
            init_state (ndarray[float]): initial state.

        """
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword argument(s) "
                + ", ".join(f"'{key}'" for key in kwargs)
                + "!"
            )
        return init_dir


class iLLG(LLG):
    r"""iLLG

    Inertial Landau-Lifshitz-Gilbert simulations with a prescribed
    magnetization amplitude.

    Ported from `friweber/udkm1Dsim_LLG-LLB
    <https://github.com/friweber/udkm1Dsim_LLG-LLB>`_ by
    `@friweber <https://github.com/friweber>`_.

    The inertial LLG of :cite:t:`ciornei2011` adds the inertia of the
    magnetization to the :class:`LLG`

    .. math::

        \frac{d\mathbf{n}}{dt} = \gamma\, \mathbf{n} \times
            \mathbf{B} + \alpha\, \mathbf{n} \times \left(
            \frac{d\mathbf{n}}{dt} + \tau \frac{d^2\mathbf{n}}{dt^2}\right)

    with the inertial relaxation time :math:`\tau` (``inertial_time`` of the
    layers), which gives rise to a nutation of the magnetization at the
    frequency :math:`1/(\alpha\tau)`. The second-order equation is split into
    two first-order equations for the unit vector and its velocity
    :math:`\mathbf{v} = d\mathbf{n}/dt`, see
    :func:`~udkm1Dsim.simulations.solvers.magnetization_scipy.illg_rhs`.
    All magnetic layers must have a non-zero Gilbert damping and inertial
    relaxation time. Apart from that everything is identical to the
    :class:`LLG`.

    The velocity relaxes on the time scale of the inertial time :math:`\tau`,
    which makes the equations stiff. The default ``method`` ``LSODA`` of the
    ``ode_options`` handles this, whereas the explicit ``RK45`` has to resolve
    the inertial time and is orders of magnitude slower.
    For a small damping, the nutation at the frequency :math:`1/(\alpha\tau)`,
    e.g. :math:`10^{17}\,\mathrm{s^{-1}}` for :math:`\alpha = 0.01` and
    :math:`\tau = 1\,\mathrm{fs}`, is only weakly damped. Tight tolerances
    ``rtol`` and ``atol`` of the ``ode_options`` require to resolve it, which
    makes the simulation very slow, e.g. seconds per picosecond for
    ``rtol=1e-8``, whereas the default tolerances are typically sufficient.

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
        ode_options (dict): options for scipy solve_ivp ode solver.
        temp_sub_system (int): index of the sub-system of the ``temp_map``
            whose temperature yields the mean-field magnetization amplitude.
        mag_sub_system (int): index of the sub-system of the ``temp_map``
            that holds the magnetization amplitude.

    """

    _title = "Inertial Landau-Lifshitz-Gilbert Magnetization Dynamics simulation"
    model = "illg"

    def calc_magnetization_map(
        self,
        delays,
        mag_amplitude_map=None,
        temp_map=None,
        B_ext=np.array([0, 0, 0]),
        init_magnetization=None,
        strain_map=None,
        equilibrate=False,
        init_velocity=None,
    ):
        r"""calc_magnetization_map

        Calculates the magnetization map using the inertial
        Landau-Lifshitz-Gilbert equation for a given delay range and
        magnetization amplitude, see :class:`iLLG` and :meth:`LLG.calc_magnetization_map`.

        Args:
            delays (ndarray[Quantity]): delays range of simulation (s).
            mag_amplitude_map (ndarray[float], optional): spatio-temporal
                amplitude of the magnetization :math:`M \times L`, i.e. the
                reduced magnetization :math:`m = |\mathbf{M}|/M_s`, which is 1
                at saturation and must not be negative. The LLG keeps it as
                prescribed and only evolves the direction.
            temp_map (ndarray[float], optional): spatio-temporal temperature
                map, used for the magnetization amplitude if no
                ``mag_amplitude_map`` is given, see :class:`LLG`.
            B_ext (ndarray[Quantity, float], optional): external magnetic
                field (B_x, B_y, B_z) (T), either constant or one vector per
                delay.
            init_magnetization (ndarray[float], dict, VectorParameter, optional):
                initial magnetization in cartesian coordinates by
                default, see :meth:`check_initial_magnetization`.
            strain_map (ndarray[float], optional): spatio-temporal strain map
                :math:`M \times L`.
            equilibrate (boolean, optional): relax the directions of the
                initial magnetization into the equilibrium of the inputs at the
                first delay, see :meth:`calc_equilibrium_magnetization`.
                Defaults to False.
            init_velocity (ndarray[float], optional): initial velocity
                :math:`d\mathbf{n}/dt` of the unit vectors (1/s), a single
                vector, one per layer :math:`L \times 3`, or one per
                sublattice :math:`L \times P \times 3`. Defaults to None,
                in which case the velocity of the LLG for the initial state is
                used, which avoids an artificial nutation at the start.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
            cartesian magnetization profile.

        """
        return super().calc_magnetization_map(
            delays,
            mag_amplitude_map=mag_amplitude_map,
            temp_map=temp_map,
            B_ext=B_ext,
            init_magnetization=init_magnetization,
            strain_map=strain_map,
            equilibrate=equilibrate,
            init_velocity=init_velocity,
        )

    def check_parameters(self, layers):
        """check_parameters

        The inertial LLG requires a non-zero Gilbert damping and inertial
        relaxation time for all magnetic layers.

        Args:
            layers (list[MagneticParameters]): magnetic parameters of the
                magnetic layers.

        """
        for layer in layers:
            if np.any(layer.sublattice_values("gilbert_damping") <= 0) or np.any(
                layer.sublattice_values("inertial_time") <= 0
            ):
                raise ValueError(
                    "The inertial LLG requires a gilbert_damping > 0 and an "
                    "inertial_time > 0 for all sublattices of the magnetic layers!"
                )

    def calc_initial_state(self, init_dir, inputs, is_magnetic, init_velocity=None, **kwargs):
        r"""calc_initial_state

        Returns the initial state of the solver, which are the unit vectors of
        the magnetic layers and their velocities for the inertial LLG.

        Args:
            init_dir (ndarray[float]): initial unit vectors of the magnetic
                layers :math:`N \times P \times 3`.
            inputs (Inputs): time-dependent inputs of the magnetic layers.
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.
            init_velocity (ndarray[float], optional): initial velocities
                (1/s), see :meth:`calc_magnetization_map`. Defaults to None.

        Returns:
            init_state (ndarray[float]): initial state :math:`2 \times N
            \times P \times 3`.

        """
        N, P = init_dir.shape[0:2]
        if init_velocity is None and N == 0:
            velocity = np.zeros_like(init_dir)
        elif init_velocity is None:
            # the velocity of the LLG for the initial state
            velocity = magnetization_scipy.llg_rhs(
                inputs.delays[0],
                init_dir.ravel(),
                inputs,
                magnetization_scipy.stack_magnetic_parameters(
                    self.calc_magnetic_parameters(is_magnetic)
                ),
                self.calc_interlayer_exchange_stiffnesses(is_magnetic),
                self.S.get_layer_property_vector("structural.thickness.magnitude")[is_magnetic],
            ).reshape(N, P, 3)
        else:
            init_velocity = np.asarray(init_velocity, dtype=float)
            if init_velocity.shape == (3,):
                velocity = np.tile(init_velocity, (N, P, 1))
            elif init_velocity.shape == (len(is_magnetic), 3):
                velocity = np.tile(init_velocity[is_magnetic, np.newaxis, :], (1, P, 1))
            elif init_velocity.shape[0:1] == (len(is_magnetic),) and init_velocity.shape[2:] == (
                3,
            ):
                velocity = np.zeros_like(init_dir)
                p = min(P, init_velocity.shape[1])
                velocity[:, :p] = init_velocity[is_magnetic, :p]
            else:
                raise ValueError(
                    "init_velocity must be a vector with 3 components or an array of "
                    "Lx3 or LxPx3 with L being the number of layers and P the number of "
                    "sublattices!"
                )
            # no velocity of the padded sublattices
            velocity = velocity * (np.linalg.norm(init_dir, axis=-1) > 0)[..., np.newaxis]
        return np.stack((init_dir, velocity))


class LLB(Magnetization):
    r"""LLB

    Mean-field Landau-Lifshitz-Bloch simulations.

    Please find a detailed review on the Landau-Lifshitz-Bloch equation by
    :cite:t:`atxitia2017` and its quantum version by :cite:t:`nieves2014`.

    In collaboration with Theodor Griepe
    (`@Nilodirf <https://github.com/Nilodirf>`_) from the group of
    Unai Atxitia at Instituto de Ciencia de Materiales de Madrid (ICMM-CSIC).

    The LLB for the magnetization :math:`\mathbf{m}` of each layer reads

    .. math::

        \frac{d\mathbf{m}}{dt}=\gamma \left(\mathbf{m} \times
          \mathbf{B} + \frac{\alpha_{\perp}}{m^2}\mathbf{m}
          \times (\mathbf{m} \times \mathbf{B}) -
          \frac{\alpha_{\parallel}}{m^2}(\mathbf{m} \cdot
          \mathbf{B}) \cdot \mathbf{m}\right)

    see :func:`~udkm1Dsim.simulations.solvers.magnetization_scipy.llb_rhs`.
    The three terms describe

    #. **precession** at Larmor frequency,
    #. **transversal damping** (conserving the macrospin length), and
    #. **longitudinal damping** (changing macrospin length due to incoherent
       atomistic spin excitations within the layer the macrospin is
       defined on).

    :math:`\alpha_{\parallel}` and :math:`\alpha_{\perp}` are the longitudinal
    and transverse :func:`damping parameters
    <udkm1Dsim.simulations.solvers.mean_field.damping_parameters>` and
    :math:`\gamma` is the gyromagnetic ratio (``gyromagnetic_ratio`` of the
    layers), which defaults to the one of a free electron
    :math:`\gamma_e = -1.761\times10^{11}\,\mathrm{rad\,s^{-1}\,T^{-1}}`.

    The field :math:`\mathbf{B}` (T) is the sum of all relevant magnetic
    interactions:

    * the external magnetic field ``B_ext``
    * the :func:`anisotropy field
      <udkm1Dsim.simulations.solvers.magnetization_scipy.anisotropy_field>`
    * the :func:`interlayer exchange field
      <udkm1Dsim.simulations.solvers.magnetization_scipy.interlayer_exchange_field>`
    * the :func:`magnetoelastic field
      <udkm1Dsim.simulations.solvers.magnetization_scipy.magnetoelastic_field>`
      of the ``strain_map``
    * the :func:`transverse exchange field of the other sublattices
      <udkm1Dsim.simulations.solvers.magnetization_scipy.transverse_sublattice_exchange_field>`
      of layers with several sublattices
    * the :func:`longitudinal field
      <udkm1Dsim.simulations.solvers.magnetization_scipy.longitudinal_field>`,
      which relaxes the amplitude of the magnetization towards its equilibrium
      with the :func:`longitudinal susceptibility
      <udkm1Dsim.simulations.solvers.mean_field.longitudinal_susceptibility>`
      and carries the longitudinal exchange between the sublattices

    The equilibrium magnetization, the damping parameters and the
    susceptibility of every layer depend on the temperature of the
    ``temp_sub_system`` only and are tabulated by the mean-field model, see
    :func:`~udkm1Dsim.simulations.solvers.mean_field.equilibrium_tables`.
    The effective spin ``eff_spin`` of the layers selects the model: finite
    spins give the quantum LLB with Brillouin functions, e.g. :math:`S = 1/2`
    for Ni, an infinite spin the classical LLB with the Langevin function.
    Layers with several sublattices, e.g. ferrimagnets, follow the LLB of
    :cite:t:`raposo2022` with the coupled mean-field equations of the
    sublattices, whose equilibrium magnetizations, damping parameters and
    susceptibilities are those of the collinear ground state.

    **Exchange relaxation and all-optical switching.** The relaxation terms
    above transfer angular momentum from every sublattice to the bath only
    and relax each sublattice towards the mean field of the current direction
    of the other sublattice. Hence, they can quench a sublattice of a
    ferrimagnet under a strong excitation but never reverse it. The reversal
    requires a transfer of angular momentum directly between the sublattices
    (:cite:t:`mentink2012`, :cite:t:`baryakhtar2013`), the
    :func:`exchange-relaxation torque
    <udkm1Dsim.simulations.solvers.magnetization_scipy.exchange_relaxation_torque>`
    of the extended LLB of :cite:t:`raposo2022`. It is added with
    ``exchange_relaxation=True`` for layers with two sublattices and the
    ``exch_relax_rate`` :math:`\lambda_\mathrm{ex}` of the layers, e.g. 0.013
    for GdFeCo, using the ``concentration`` of the sublattices. The torque
    diverges as :math:`1/m` for a quenched sublattice, which the adaptive
    integrators carry through - tighter tolerances of the ``ode_options``
    check the convergence of the switching.

    Layers with an anisotropy or a magnetoelastic coupling require an
    ``aniso_exponent`` of at least 2, since these fields scale as
    :math:`m_\mathrm{eq}^{\beta-2}` and would diverge at the Curie temperature
    otherwise.

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

    """

    _title = "Landau-Lifshitz-Bloch Magnetization Dynamics simulation"

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def calc_magnetization_map(
        self,
        delays,
        temp_map,
        B_ext=np.array([0, 0, 0]),
        init_magnetization=None,
        strain_map=None,
        equilibrate=False,
        exchange_relaxation=False,
    ):
        r"""calc_magnetization_map

        Calculates the magnetization map using the mean-field
        Landau-Lifshitz-Bloch equation (LLB) for a given delay range and
        according temperature map, see :class:`LLB`.

        Args:
            delays (ndarray[Quantity]): delays range of simulation (s).
            temp_map (ndarray[float]): spatio-temporal temperature map.
            B_ext (ndarray[Quantity, float], optional): external magnetic
                field (B_x, B_y, B_z) (T), either constant or one vector per
                delay.
            init_magnetization (ndarray[float], dict, VectorParameter, optional):
                initial magnetization in cartesian coordinates by
                default, see :meth:`check_initial_magnetization`. If none is
                given, the layers start in the equilibrium of their
                initial temperature along the magnetization of the layers.
            strain_map (ndarray[float], optional): spatio-temporal strain map
                :math:`M \times L` for the magnetoelastic field.
            equilibrate (boolean, optional): start in the equilibrium of the
                inputs at the first delay, i.e. with the mean-field
                magnetization of the initial temperature and the directions
                relaxed into the equilibrium, see
                :meth:`calc_equilibrium_magnetization`. Defaults to False.
            exchange_relaxation (boolean, optional): add the
                exchange-relaxation torque between two sublattices with the
                ``exch_relax_rate`` of the layers, see :class:`LLB`. Defaults
                to False.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
            cartesian magnetization profile.

        """
        t1 = time()
        delays = self.parse_delays(delays)
        M = len(delays)  # nb of delay steps
        L = self.S.num_layers

        temps = self.check_temp_map(temp_map, M, L)
        if strain_map is not None:
            strain_map = self.check_map("strain_map", strain_map, M, L)

        implicit_init_mag = len(self.parse_initial_magnetization(init_magnetization)) == 0
        init_mag = self.check_initial_magnetization(init_magnetization)

        # magnetic layers carry a Curie temperature
        is_magnetic = self.S.get_layer_property_vector("magnetic.curie_temp.magnitude") > 0
        layers = self.calc_magnetic_parameters(is_magnetic)
        N = len(layers)
        P = magnetization_scipy.stack_magnetic_parameters(layers).active.shape[1]
        if any(np.any(layer.sublattice_values("eff_spin") <= 0) for layer in layers):
            raise ValueError(
                "All sublattices of the layers with a Curie temperature must have an "
                "eff_spin > 0 for the LLB, which can also be infinite for the classical LLB!"
            )
        for layer in layers:
            anisotropic = np.any(layer.sublattice_values("anisotropy") != 0, axis=1) | (
                layer.sublattice_values("magnetoelastic_coupling") != 0
            )
            if np.any(anisotropic & (layer.sublattice_values("aniso_exponent") < 2)):
                raise ValueError(
                    "The anisotropy and magnetoelastic fields of the LLB scale as "
                    "m_eq**(aniso_exponent - 2) and diverge at the Curie temperature for "
                    "aniso_exponent < 2 - it is 3 for a uniaxial anisotropy following "
                    "the Callen-Callen law!"
                )

        if (implicit_init_mag or equilibrate) and N > 0:
            # start in the equilibrium of the initial temperature, tabulated
            # on the same grid as by the solver
            temp_grid, tables = magnetization_scipy.tabulate_equilibrium(
                layers, temp_max=np.max(temps[:, is_magnetic])
            )
            mean_mags = interpolate_tables_map(temp_grid, tables, temps[:1, is_magnetic])[
                0, ..., TABLE["mag"]
            ]
            if equilibrate:
                # the mean-field amplitude, keeping a reversed direction
                init_mag[is_magnetic, :P, 0] = np.where(
                    init_mag[is_magnetic, :P, 0] < 0, -mean_mags, mean_mags
                )
            else:
                init_mag[is_magnetic, :P, 0] *= mean_mags
        init_state = convert_polar_to_cartesian(init_mag)[is_magnetic, :P, :]

        inputs = magnetization_scipy.prepare_inputs(
            delays,
            N,
            B_ext,
            temp_map=temps[:, is_magnetic],
            strain_map=None if strain_map is None else strain_map[:, is_magnetic],
        )
        if equilibrate and N > 0:
            amplitudes = np.linalg.norm(init_state, axis=-1)
            directions = self.equilibrate_directions(init_state, amplitudes, inputs, is_magnetic)
            init_state = amplitudes[..., np.newaxis] * directions
        mag_map = self.solve(
            "llb", init_state, inputs, is_magnetic, exchange_relaxation=exchange_relaxation
        )
        self.check_demagnetization_collapse(mag_map, temps[:, is_magnetic], layers)

        # final magnetization maps are zero for all non-magnetic layers
        maps = self.embed_magnetization(mag_map, is_magnetic)
        self.disp_message(f"Elapsed time for _LLB_: {time() - t1:f} s")

        return maps

    def check_demagnetization_collapse(self, mag_map, temps, layers):
        r"""check_demagnetization_collapse

        Warns if a layer ends at :math:`|\mathbf{m}| < 10^{-6}` although it is
        below its Curie temperature at the end of the delay range. Without an
        external field :math:`\mathbf{m} = 0` is an exact fixed point of the
        LLB, which is never left again once reached in floating point, and a
        magnetization that was quenched to a tiny value regrows only
        exponentially slowly from it.

        Args:
            mag_map (ndarray[float]): cartesian magnetization map of the
                magnetic layers :math:`M \times N \times P \times 3`.
            temps (ndarray[float]): temperature map of the magnetic layers
                :math:`M \times N` (K).
            layers (list[MagneticParameters]): magnetic parameters of the
                magnetic layers.

        Returns:
            collapsed (ndarray[boolean]): mask :math:`N \times P` of the
            collapsed layers.

        """
        if len(mag_map) == 0 or mag_map.shape[1] == 0:
            return np.zeros(mag_map.shape[1:3], dtype=bool)
        mag = magnetization_scipy.stack_magnetic_parameters(layers)
        norms = np.sqrt(np.sum(mag_map[-1] ** 2, axis=2))
        cold = temps[-1] < mag.curie_temp[:, 0]
        collapsed = (norms < 1e-6) & (mag.active > 0) & cold[:, np.newaxis]
        if np.any(collapsed):
            warnings.warn(
                f"{int(np.count_nonzero(collapsed)):d} layer(s) ended at |m| < 1e-6 although "
                "the final temperature is below the Curie temperature. m = 0 is an exact "
                "fixed point of the LLB without an external field and a quenched "
                "magnetization regrows only exponentially slowly - consider a small "
                "external field or a longer delay range."
            )
        return collapsed

    @property
    def distances(self):
        return Q_(self._distances, u.meter).to("nm")

    @distances.setter
    def distances(self, distances):
        self._distances = distances.to_base_units().magnitude
