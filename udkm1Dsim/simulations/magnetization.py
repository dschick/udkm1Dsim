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
from os import path
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
from .simulation import Simulation
from .solvers import magnetization_scipy
from .solvers.mean_field import TABLE, interpolate_tables_map

u = pint.get_application_registry()
Q_ = u.Quantity


class Magnetization(Simulation):
    """Magnetization

    Base class for all magnetization simulations. It prepares the inputs and
    the material parameters of the magnetic layers, caches the results and
    converts them to polar coordinates. The physics - the right-hand sides
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
        ode_options (dict): options for scipy solve_ivp ode solver.
        temp_sub_system (int): index of the sub-system of the ``temp_map``
            whose temperature drives the magnetization dynamics.

    """

    _title = "Magnetization simulation"

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.ode_options = {
            "method": "RK45",
            "first_step": None,
            "max_step": np.inf,
            "rtol": 1e-3,
            "atol": 1e-6,
        }
        self.temp_sub_system = 0

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

        for name in ("strain_map", "temp_map", "mag_amplitude_map"):
            if name in kwargs:
                value = kwargs.pop(name)
                if value is not None and np.size(value) > 1e6:
                    value = np.asarray(value).flatten()[0:1000000]
                param.append(value)

        for value in kwargs.values():
            param.append(value)

        return self.S.get_hash(types="magnetic") + "_" + make_hash_md5(param)

    def check_initial_magnetization(self, init_mag, distances=[]):
        """check_initial_magnetization

        Check if a given initial magnetization profile is valid. The profile
        must be either a vector `[amplitude, phi, gamma]` describing a global
        initial magnetization or an array of shape `[Nx3]` with N being the
        number of layers or the length of the specific spatial grid. If no
        initial magnetization is given, the initial profile is determined from
        the magnetization of the layers on creation. In addition, a spatial
        grid can be provided.

        Args:
            init_mag (ndarray[float]): initial global or local magnetization
                profile.
            distances (ndarray[float, Quantity], optional): spatial grid of the
                initial magnetization.

        Returns:
            init_mag (ndarray[float]): checked initial magnetization as array on
                the according spatial grid.

        """
        layers = self.S.get_unique_layers()
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
            N = self.S.get_number_of_layers()
            [distances, _, _] = self.S.get_distances_of_layers(False)

        if len(init_mag) == 0:
            self.disp_message(
                "No explicit initial magnetization given - use magnetization of layers instead."
            )
            init_mag = np.zeros([N, 3])
            # use finderb search to find the corresponding indices between the
            # internal and external spatial grids
            [d_start, _, _] = self.S.get_distances_of_layers(False)
            idx = finderb(distances, d_start)

            init_mag = self.S.get_layer_property_vector("magnetic.magnetization.polar")[idx, :]
        else:
            if np.size(init_mag) == 3:
                # it is the same initial magnetization for all layers
                init_mag = np.tile(init_mag, (N, 1))
            elif np.shape(init_mag) != (N, 3):
                # init_temp is a vector but has not as many elements as layers
                raise ValueError(
                    "The initial magnetization array must have 3 or "
                    "Nx3 elements, where N is the number of layers "
                    "in the structure or the length of the spatial "
                    "grid provided as distances vector!"
                )

            # convert phi and gamma to rad and store only magnitudes
            try:
                init_mag[:, 1] = init_mag[:, 1].to("rad").magnitude
            except AttributeError:
                pass
            try:
                init_mag[:, 2] = init_mag[:, 2].to("rad").magnitude
            except AttributeError:
                pass
        return init_mag

    def get_magnetization_map(self, delays, **kwargs):
        r"""get_magnetization_map

        Returns an absolute ``magnetization_map`` for the sample structure with
        the dimension :math:`M \times N \times 3` with :math:`M` being the
        number of delays and :math:`N` the number of layers in the structure or
        the length of the given spatial grid. Each element of the map contains
        the three magnetization components ``[amplitude, phi, gamma]``.
        The angles ``phi`` and ``gamma`` must be returned in radians as pure
        numpy arrays.
        The ``magnetization_map`` can depend on the ``temp_map`` and
        ``strain_map`` that can be also calculated for the sample structure.
        More over an external magnetic field ``H_ext`` and initial magnetization
        profile ``init_mag`` can be provided.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            **kwargs (ndarray[float], optional): optional strain and
                temperature profile as well external magnetic field in [T],
                either constant or one vector per delay, and initial
                magnetization.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
               magnetization profile.

        """
        # create a hash of all simulation parameters
        filename = "magnetization_map_" + self.get_hash(delays=delays, **kwargs) + ".npz"
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            magnetization_map = tmp["magnetization_map"]
            self.disp_message("_magnetization_map_ loaded from file:\n\t" + filename)
        else:
            t1 = time()
            self.disp_message("Calculating _magnetization_map_ ...")
            # parse the input arguments

            if kwargs.get("strain_map") is not None:
                if not isinstance(kwargs["strain_map"], np.ndarray):
                    raise TypeError("strain_map must be a numpy ndarray!")
            for name in ("temp_map", "mag_amplitude_map"):
                if kwargs.get(name) is not None and not isinstance(kwargs[name], np.ndarray):
                    raise TypeError(f"{name} must be a numpy ndarray!")
            if "H_ext" in kwargs:
                if not isinstance(kwargs["H_ext"], np.ndarray):
                    raise TypeError("H_ext must be a numpy ndarray!")
                elif kwargs["H_ext"].shape not in [(3,), (np.size(delays), 3)]:
                    raise ValueError(
                        "H_ext must be a vector with 3 components (H_x, H_y, H_z) or an "
                        "array of Mx3 with M being the number of delays!"
                    )
            if "init_mag" in kwargs and len(kwargs["init_mag"]) > 0:
                if not isinstance(kwargs["init_mag"], np.ndarray):
                    raise TypeError(
                        "init_mag must be a numpy ndarray with all in radians without units!"
                    )
                elif kwargs["init_mag"].shape not in [(3,), (self.S.get_number_of_layers(), 3)]:
                    raise ValueError(
                        "init_mag must be a vector with 3 components or an array of Nx3 "
                        "with N being the number of layers."
                    )

            magnetization_map = self.calc_magnetization_map(delays, **kwargs)

            self.disp_message(f"Elapsed time for _magnetization_map_: {time() - t1:f} s")
            self.save(
                full_filename, {"magnetization_map": magnetization_map}, "_magnetization_map_"
            )
        return magnetization_map

    def calc_magnetization_map(self, delays, **kwargs):
        """calc_magnetization_map

        Calculates an absolute ``magnetization_map`` - see
        :meth:`get_magnetization_map` for details.

        This method is just an interface and should be overwritten for the
        actual simulations.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            **kwargs (ndarray[float], optional): optional strain and
                temperature profile as well external magnetic field and initial
                magnetization.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
                magnetization profile.

        """
        raise NotImplementedError

    def get_mean_field_mag_map(self, temp_map):
        r"""get_mean_field_mag_map

        Returns the mean-field magnetization map see
        :meth:`calc_mean_field_mag_map` for details. The dimension of the map
        are :math:`M \times N` with :math:`M` being the number of delays and
        :math:`N` the number of layers in the structure.

        Args:
            temp_map (ndarray[float]): spatio-temporal electron temperature map.

        Returns:
            mf_mag_map (ndarray[float]): spatio-temporal mean_field
            magnetization map.

        """
        # create a hash of all simulation parameters
        filename = "mf_magnetization_map_" + self.get_hash(temp_map=temp_map) + ".npz"
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            mf_mag_map = tmp["mf_mag_map"]
            self.disp_message("_mean_field_magnetization_map_ loaded from file:\n\t" + filename)
        else:
            t1 = time()
            self.disp_message("Calculating _mean_field_magnetization_map_ ...")
            # parse the input arguments
            if not isinstance(temp_map, np.ndarray):
                raise TypeError("temp_map must be a numpy ndarray!")

            mf_mag_map = self.calc_mean_field_mag_map(temp_map)

            self.disp_message(
                f"Elapsed time for _mean_field_magnetization_map_: {time() - t1:f} s"
            )
            self.save(full_filename, {"mf_mag_map": mf_mag_map}, "_mean_field_magnetization_map_")
        return mf_mag_map

    def calc_mean_field_mag_map(self, temp_map):
        r"""calc_mean_field_mag_map

        Calculate the mean-field magnetization map :math:`m_\mathrm{eq}` of
        every layer, the non-trivial solution of the self-consistent equation

        .. math::

            m_\mathrm{eq} = B_S\left(\frac{J\, m_\mathrm{eq}}{k_\mathrm{B} T}\right)

        with the Brillouin function :math:`B_S` of the effective spin - the
        Langevin function for an infinite spin - which is tabulated on a
        temperature grid, see
        :func:`~udkm1Dsim.simulations.solvers.mean_field.equilibrium_tables`,
        and interpolated onto the ``temp_map``. The magnetization is zero for
        non-magnetic layers and at and above the Curie temperature.

        Args:
            temp_map (ndarray[float]): spatio-temporal electron temperature map
                :math:`M \times N`.

        Returns:
            mf_mag_map (ndarray[float]): spatio-temporal mean_field
            magnetization map.

        """
        temp_map = np.asarray(temp_map, dtype=float)
        L = self.S.get_number_of_layers()
        layers = self.get_magnetic_parameters(np.ones(L, dtype=bool))
        temp_grid, tables = magnetization_scipy.tabulate_equilibrium(
            layers, temp_max=float(np.max(temp_map))
        )
        values = interpolate_tables_map(temp_grid, tables, temp_map)
        return values[..., 0, TABLE["mag"]]

    def get_directional_exchange_stiffnesses(self):
        r"""get_directional_exchange_stiffnesses

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

        indices, _, _ = self.S.get_layer_vectors()

        A = np.zeros([len(indices), 2])
        interfaces = np.r_[1, np.diff(indices), 1]
        interfaces[interfaces != 0] = -1
        select = (interfaces + 1).astype(np.int16)

        A[:, 0] = exch_stiffnesses[np.arange(len(select[0:-1])), select[0:-1]]

        interfaces[interfaces != 0] = 1
        select = (interfaces + 1).astype(np.int16)
        A[:, 1] = exch_stiffnesses[np.arange(len(select[1:])), select[1:]]

        return A

    def get_magnetic_parameters(self, is_magnetic):
        r"""get_magnetic_parameters

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
        handles = self.S.get_layer_vectors()[2]
        return [handles[i].magnetic for i in np.flatnonzero(is_magnetic)]

    def get_interlayer_exchange_stiffnesses(self, is_magnetic):
        r"""get_interlayer_exchange_stiffnesses

        Returns the :meth:`get_directional_exchange_stiffnesses` of the
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
        A = self.get_directional_exchange_stiffnesses()[idx, :].copy()
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
            delays (ndarray[Quantity, float]): delays [s].

        Returns:
            delays (ndarray[float]): delays [s].

        """
        try:
            delays = delays.to("s").magnitude
        except AttributeError:
            pass
        return np.asarray(delays, dtype=float)

    def solve(self, model, init_state, inputs, is_magnetic):
        r"""solve

        Solves the ``model`` for the magnetic layers, see
        :func:`~udkm1Dsim.simulations.solvers.magnetization_scipy.solve`.

        Args:
            model (str): one of ``llg``, ``illg`` or ``llb``.
            init_state (ndarray[float]): initial state of the solver.
            inputs (Inputs): time-dependent inputs of the magnetic layers.
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.

        Returns:
            mag_map (ndarray[float]): cartesian magnetization map of the
            magnetic layers :math:`M \times N \times P \times 3`.

        """
        return magnetization_scipy.solve(
            model,
            init_state,
            inputs,
            self.get_magnetic_parameters(is_magnetic),
            self.get_interlayer_exchange_stiffnesses(is_magnetic),
            self.S.get_layer_property_vector("structural.thickness.magnitude")[is_magnetic],
            self.ode_options,
            self.progress_bar,
        )

    @staticmethod
    def embed_magnetization(mag_map, is_magnetic):
        r"""embed_magnetization

        Embeds the cartesian magnetization of the magnetic layers into the
        polar magnetization map of all layers, which is zero for the
        non-magnetic layers.

        Args:
            mag_map (ndarray[float]): cartesian magnetization map of the
                magnetic layers :math:`M \times N \times P \times 3`.
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.

        Returns:
            magnetization_map (ndarray[float]): polar magnetization map
            :math:`M \times L \times 3`.

        """
        magnetization_map = np.zeros([len(mag_map), len(is_magnetic), 3])
        magnetization_map[:, is_magnetic, :] = mag_map[:, :, 0, :]
        return convert_cartesian_to_polar(magnetization_map)


class LLG(Magnetization):
    r"""LLG

    Landau-Lifshitz-Gilbert simulations with a prescribed magnetization amplitude.

    Ported from `friweber/udkm1Dsim_LLG-LLB
    <https://github.com/friweber/udkm1Dsim_LLG-LLB>`_ by
    `@friweber <https://github.com/friweber>`_.

    The LLG describes the precession and the transverse relaxation of the
    unit vector :math:`\mathbf{n}` of the magnetization of each layer

    .. math::

        \frac{d\mathbf{n}}{dt} = \frac{\gamma_e}{1+\alpha^2}\left(
            \mathbf{n} \times \mathbf{B}
            + \alpha\, \mathbf{n} \times (\mathbf{n} \times
            \mathbf{B})\right)

    with the Gilbert damping :math:`\alpha` (``gilbert_damping`` of the
    layers) and :math:`\gamma_e = -1.761\times10^{11}\,\mathrm{rad\,s^{-1}\,
    T^{-1}}` the gyromagnetic ratio of an electron, see
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

    The field :math:`\mathbf{B}` [T] is the sum of the external field
    ``H_ext``, the :func:`anisotropy field
    <udkm1Dsim.simulations.solvers.magnetization_scipy.anisotropy_field>`,
    the :func:`interlayer exchange field
    <udkm1Dsim.simulations.solvers.magnetization_scipy.interlayer_exchange_field>`
    and the :func:`magnetoelastic field
    <udkm1Dsim.simulations.solvers.magnetization_scipy.magnetoelastic_field>`
    of the ``strain_map``. The temperature dependence of the anisotropy and
    magnetoelastic fields is scaled with the magnetization amplitude :math:`m(t)`.
    Magnetic layers are those with a saturation magnetization.

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
        H_ext=np.array([0, 0, 0]),
        init_mag=[],
        strain_map=None,
        **kwargs,
    ):
        r"""calc_magnetization_map

        Calculates the magnetization map using the Landau-Lifshitz-Gilbert
        equation for a given delay range and magnetization amplitude, see
        :class:`LLG`.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            mag_amplitude_map (ndarray[float], optional): spatio-temporal
                amplitude of the magnetization :math:`M \times L`, i.e. the
                reduced magnetization :math:`m = |\mathbf{M}|/M_s`, which is 1
                at saturation and must not be negative. The LLG keeps it as
                prescribed and only evolves the direction.
            temp_map (ndarray[float], optional): spatio-temporal temperature
                map, used for the magnetization amplitude if no
                ``mag_amplitude_map`` is given, see :class:`LLG`.
            H_ext (ndarray[float], optional): external magnetic field
                (H_x, H_y, H_z) [T], either constant or one vector per delay.
            init_mag (ndarray[float], optional): initial magnetization in
                polar coordinates, see :meth:`check_initial_magnetization`.
                Its amplitude is kept if neither ``mag_amplitude_map`` nor
                ``temp_map`` are given, otherwise only its direction is used.
            strain_map (ndarray[float], optional): spatio-temporal strain map
                :math:`M \times L` for the magnetoelastic field.
            **kwargs: model-specific options, see :class:`iLLG`.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
            magnetization profile.

        """
        t1 = time()
        delays = self.parse_delays(delays)
        M = len(delays)
        L = self.S.get_number_of_layers()

        init_mag = np.array(self.check_initial_magnetization(init_mag), dtype=float)
        # the directions of the initial magnetization, a negative amplitude
        # reverses the direction
        init_dir = init_mag.copy()
        init_dir[:, 0] = np.where(init_mag[:, 0] < 0, -1.0, 1.0)
        init_dir = convert_polar_to_cartesian(init_dir)

        # the magnetization amplitude
        if mag_amplitude_map is not None:
            amplitude_map = self.check_map("mag_amplitude_map", mag_amplitude_map, M, L)
        elif temp_map is not None:
            if self.mag_sub_system is not None:
                amplitude_map = self.check_map("temp_map", temp_map, M, L, self.mag_sub_system)
            else:
                amplitude_map = self.get_mean_field_mag_map(self.check_temp_map(temp_map, M, L))
        else:
            amplitude_map = np.tile(np.abs(init_mag[:, 0]), (M, 1))
        if np.any(amplitude_map < 0):
            raise ValueError(
                "The magnetization amplitude must not be negative - a reversed direction "
                "is given by a negative amplitude of init_mag only!"
            )
        if strain_map is not None:
            strain_map = self.check_map("strain_map", strain_map, M, L)

        # magnetic layers carry a saturation magnetization
        is_magnetic = self.S.get_layer_property_vector("magnetic.mag_saturation.magnitude") > 0
        layers = self.get_magnetic_parameters(is_magnetic)
        self.check_parameters(layers)

        inputs = magnetization_scipy.prepare_inputs(
            delays,
            len(layers),
            H_ext,
            amplitude_map=amplitude_map[:, is_magnetic],
            strain_map=None if strain_map is None else strain_map[:, is_magnetic],
        )
        init_state = self.get_initial_state(
            init_dir[is_magnetic, np.newaxis, :], inputs, is_magnetic, **kwargs
        )
        mag_map = self.solve(self.model, init_state, inputs, is_magnetic)

        magnetization_map = self.embed_magnetization(mag_map, is_magnetic)
        self.disp_message(f"Elapsed time for _{type(self).__name__:s}_: {time() - t1:f} s")
        return magnetization_map

    def check_parameters(self, layers):
        """check_parameters

        Checks the parameters of the magnetic layers for the model.

        Args:
            layers (list[MagneticParameters]): magnetic parameters of the
                magnetic layers.

        """

    def get_initial_state(self, init_dir, inputs, is_magnetic, **kwargs):
        r"""get_initial_state

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

        \frac{d\mathbf{n}}{dt} = \gamma_e\, \mathbf{n} \times
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
    which makes the equations stiff. Hence, the default ``method`` of the
    ``ode_options`` is ``LSODA``, which switches automatically between a stiff
    and a non-stiff integrator. The explicit ``RK45`` has to resolve the
    inertial time and is orders of magnitude slower.

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

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.ode_options["method"] = "LSODA"

    def calc_magnetization_map(
        self,
        delays,
        mag_amplitude_map=None,
        temp_map=None,
        H_ext=np.array([0, 0, 0]),
        init_mag=[],
        strain_map=None,
        init_velocity=None,
    ):
        r"""calc_magnetization_map

        Calculates the magnetization map using the inertial
        Landau-Lifshitz-Gilbert equation for a given delay range and
        magnetization amplitude, see :class:`iLLG` and :meth:`LLG.calc_magnetization_map`.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            mag_amplitude_map (ndarray[float], optional): spatio-temporal
                amplitude of the magnetization :math:`M \times L`, i.e. the
                reduced magnetization :math:`m = |\mathbf{M}|/M_s`, which is 1
                at saturation and must not be negative. The LLG keeps it as
                prescribed and only evolves the direction.
            temp_map (ndarray[float], optional): spatio-temporal temperature
                map, used for the magnetization amplitude if no
                ``mag_amplitude_map`` is given, see :class:`LLG`.
            H_ext (ndarray[float], optional): external magnetic field
                (H_x, H_y, H_z) [T], either constant or one vector per delay.
            init_mag (ndarray[float], optional): initial magnetization in
                polar coordinates.
            strain_map (ndarray[float], optional): spatio-temporal strain map
                :math:`M \times L`.
            init_velocity (ndarray[float], optional): initial velocity
                :math:`d\mathbf{n}/dt` of the unit vectors [1/s], a single
                vector or one per layer :math:`L \times 3`. Defaults to None,
                in which case the velocity of the LLG for the initial state is
                used, which avoids an artificial nutation at the start.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
            magnetization profile.

        """
        return super().calc_magnetization_map(
            delays,
            mag_amplitude_map=mag_amplitude_map,
            temp_map=temp_map,
            H_ext=H_ext,
            init_mag=init_mag,
            strain_map=strain_map,
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
            if layer.gilbert_damping.magnitude <= 0 or layer.inertial_time.magnitude <= 0:
                raise ValueError(
                    "The inertial LLG requires a gilbert_damping > 0 and an "
                    "inertial_time > 0 for all magnetic layers!"
                )

    def get_initial_state(self, init_dir, inputs, is_magnetic, init_velocity=None, **kwargs):
        r"""get_initial_state

        Returns the initial state of the solver, which are the unit vectors of
        the magnetic layers and their velocities for the inertial LLG.

        Args:
            init_dir (ndarray[float]): initial unit vectors of the magnetic
                layers :math:`N \times P \times 3`.
            inputs (Inputs): time-dependent inputs of the magnetic layers.
            is_magnetic (ndarray[boolean]): mask of the magnetic layers.
            init_velocity (ndarray[float], optional): initial velocities
                [1/s], see :meth:`calc_magnetization_map`. Defaults to None.

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
                    self.get_magnetic_parameters(is_magnetic)
                ),
                self.get_interlayer_exchange_stiffnesses(is_magnetic),
                self.S.get_layer_property_vector("structural.thickness.magnitude")[is_magnetic],
            ).reshape(N, P, 3)
        else:
            init_velocity = np.asarray(init_velocity, dtype=float)
            if init_velocity.shape == (3,):
                velocity = np.tile(init_velocity, (N, P, 1))
            elif init_velocity.shape == (len(is_magnetic), 3):
                velocity = np.tile(init_velocity[is_magnetic, np.newaxis, :], (1, P, 1))
            else:
                raise ValueError(
                    "init_velocity must be a vector with 3 components or an array of "
                    "Lx3 with L being the number of layers!"
                )
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

        \frac{d\mathbf{m}}{dt}=\gamma_e \left(\mathbf{m} \times
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
    :math:`\gamma_e = -1.761\times10^{11}\,\mathrm{rad\,s^{-1}\,T^{-1}}` is
    the gyromagnetic ratio of an electron.

    The field :math:`\mathbf{B}` [T] is the sum of all relevant magnetic
    interactions:

    * the external magnetic field ``H_ext``
    * the :func:`anisotropy field
      <udkm1Dsim.simulations.solvers.magnetization_scipy.anisotropy_field>`
    * the :func:`interlayer exchange field
      <udkm1Dsim.simulations.solvers.magnetization_scipy.interlayer_exchange_field>`
    * the :func:`magnetoelastic field
      <udkm1Dsim.simulations.solvers.magnetization_scipy.magnetoelastic_field>`
      of the ``strain_map``
    * the :func:`longitudinal field
      <udkm1Dsim.simulations.solvers.magnetization_scipy.longitudinal_field>`,
      which relaxes the amplitude of the magnetization towards its equilibrium
      with the :func:`longitudinal susceptibility
      <udkm1Dsim.simulations.solvers.mean_field.longitudinal_susceptibility>`

    The equilibrium magnetization, the damping parameters and the
    susceptibility of every layer depend on the temperature of the
    ``temp_sub_system`` only and are tabulated by the mean-field model, see
    :func:`~udkm1Dsim.simulations.solvers.mean_field.equilibrium_tables`.
    The effective spin ``eff_spin`` of the layers selects the model: finite
    spins give the quantum LLB with Brillouin functions, e.g. :math:`S = 1/2`
    for Ni, an infinite spin the classical LLB with the Langevin function.
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
        self, delays, temp_map, H_ext=np.array([0, 0, 0]), init_mag=[], strain_map=None
    ):
        r"""calc_magnetization_map

        Calculates the magnetization map using the mean-field
        Landau-Lifshitz-Bloch equation (LLB) for a given delay range and
        according temperature map, see :class:`LLB`.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            temp_map (ndarray[float]): spatio-temporal temperature map.
            H_ext (ndarray[float], optional): external magnetic field
                (H_x, H_y, H_z) [T], either constant or one vector per delay.
            init_mag (ndarray[float], optional): initial magnetization in
                polar coordinates, see :meth:`check_initial_magnetization`. If
                none is given, the layers start in the equilibrium of their
                initial temperature along the magnetization of the layers.
            strain_map (ndarray[float], optional): spatio-temporal strain map
                :math:`M \times L` for the magnetoelastic field.

        Returns:
            magnetization_map (ndarray[float]): spatio-temporal absolute
            magnetization profile.

        """
        t1 = time()
        delays = self.parse_delays(delays)
        M = len(delays)  # nb of delay steps
        L = self.S.get_number_of_layers()

        temps = self.check_temp_map(temp_map, M, L)
        if strain_map is not None:
            strain_map = self.check_map("strain_map", strain_map, M, L)

        implicit_init_mag = len(init_mag) == 0
        init_mag = np.array(self.check_initial_magnetization(init_mag), dtype=float)

        # magnetic layers carry a Curie temperature
        is_magnetic = self.S.get_layer_property_vector("magnetic.curie_temp.magnitude") > 0
        layers = self.get_magnetic_parameters(is_magnetic)
        N = len(layers)
        if any(layer.eff_spin.magnitude <= 0 for layer in layers):
            raise ValueError(
                "All layers with a Curie temperature must have an eff_spin > 0 for the "
                "LLB, which can also be infinite for the classical LLB!"
            )
        for layer in layers:
            anisotropic = np.any(np.asarray(layer.anisotropy.magnitude) != 0) or np.any(
                np.asarray(layer.magnetoelastic_coupling.magnitude) != 0
            )
            if anisotropic and layer.aniso_exponent.magnitude < 2:
                raise ValueError(
                    "The anisotropy and magnetoelastic fields of the LLB scale as "
                    "m_eq**(aniso_exponent - 2) and diverge at the Curie temperature for "
                    "aniso_exponent < 2 - it is 3 for a uniaxial anisotropy following "
                    "the Callen-Callen law!"
                )

        if implicit_init_mag and N > 0:
            # start in the equilibrium of the initial temperature, tabulated
            # on the same grid as by the solver
            temp_grid, tables = magnetization_scipy.tabulate_equilibrium(
                layers, temp_max=np.max(temps[:, is_magnetic])
            )
            mean_mags = interpolate_tables_map(temp_grid, tables, temps[:1, is_magnetic])[
                0, :, 0, TABLE["mag"]
            ]
            init_mag[is_magnetic, 0] *= mean_mags
        init_state = convert_polar_to_cartesian(init_mag)[is_magnetic, np.newaxis, :]

        inputs = magnetization_scipy.prepare_inputs(
            delays,
            N,
            H_ext,
            temp_map=temps[:, is_magnetic],
            strain_map=None if strain_map is None else strain_map[:, is_magnetic],
        )
        mag_map = self.solve("llb", init_state, inputs, is_magnetic)
        self.check_demagnetization_collapse(mag_map, temps[:, is_magnetic], layers)

        # final magnetization map is zero for all non-magnetic layers
        magnetization_map = self.embed_magnetization(mag_map, is_magnetic)
        self.disp_message(f"Elapsed time for _LLB_: {time() - t1:f} s")

        return magnetization_map

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
                :math:`M \times N` [K].
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
