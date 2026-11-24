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

__all__ = ["Heat"]

__docformat__ = "restructuredtext"

import operator
import warnings
from contextlib import contextmanager
from copy import copy, deepcopy
from time import time

import numpy as np
import pint
import scipy.constants as constants
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

from ..helpers import finderb, make_hash_md5, to_angle
from .result import find_source, get_entry
from .scattering import instrument as _instrument
from .scattering.multilayer import MultilayerScattering
from .scattering.parratt import Parratt
from .simulation import Simulation
from .solvers import HAS_NUMBA

u = pint.get_application_registry()
Q_ = u.Quantity

# incoming polarizations of the excitation by their names, i.e. those of the
# multilayer formalisms without the elliptical polarization
_POLARIZATIONS = tuple(MultilayerScattering.polarizations[state] for state in range(5))
# excitation parameters, which are defined by a configured formalism instance as
# absorption
_INSTANCE_KEYS = ("wavelength", "theta", "azimuth", "polarization")
# spectral bandwidth and divergence of the pulses with the defaults of the
# sampling of their profiles by the nodes of the instrument function
_SPREADS = ("bandwidth", "divergence")
_SAMPLING_DEFAULTS = {
    "profile": "gauss",
    "nodes": "uniform",
    "points": 7,
    "range": 2.0,
    "eta": 0.5,
}
_SAMPLING_KEYS = tuple(f"{spread}_{key}" for spread in _SPREADS for key in _SAMPLING_DEFAULTS)
# h c (eV m) to convert between photon energies and wavelengths
_HC = constants.h * constants.c / constants.e
# excitation parameters, which can be given per pulse besides the fluence,
# delay_pump, and pulse_width, where the bandwidth_domain is the wavelength or
# the photon energy of the profile of the bandwidth by its unit
_PULSE_KEYS = (
    *_INSTANCE_KEYS,
    "absorption",
    "backside",
    "bandwidth",
    "bandwidth_domain",
    "divergence",
    *_SAMPLING_KEYS,
)


class Heat(Simulation):
    """Heat

    Heat simulations including laser excitation and N-temperature model.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        heat_diffusion (boolean): true when including heat diffusion in the
            calculations.
        intp_at_interface (int): number of additional spacial points at the
            interface of each layer.
        backend (str): pde solver backend - either default scipy or numba.

    Attributes:
        S (Structure): sample structure to calculate simulations on.
        force_recalc (boolean): force recalculation of results.
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        heat_diffusion (boolean): true when including heat diffusion in the
            calculations.
        intp_at_interface (int): number of additional spacial points at the
            interface of each layer.
        backend (str): pde solver backend - either default scipy or numba.
        excitation (dict{ndarray[float, Quantity]}): excitation parameters
            fluence, delay_pump, pulse_width, wavelength, theta, azimuth,
            polarization, absorption, backside, bandwidth, divergence, and
            the sampling of the bandwidth and divergence. The ``fluence``,
            ``delay_pump``, and ``pulse_width`` are given for each pulse,
            while all other parameters are either a single value for all
            pulses or a list of one value for each pulse, which are sorted by
            the ``delay_pump`` together with all other per-pulse parameters,
            so that the indices of the pulses, e.g. ``pulse_index`` of
            :meth:`get_absorption_profile`, refer to the sorted pulses. The
            ``polarization`` is ``sigma`` or ``s``,
            ``pi`` or ``p``, ``unpolarized``, ``circ +`` or ``c+``, or ``circ -`` or ``c-``, as for
            the multilayer formalisms, while an ``elliptical`` polarization
            requires a configured instance as ``absorption``. The ``absorption`` is
            ``lambert_beer`` or a multilayer formalism, by default
            :class:`.Parratt`, which can also be a configured instance
            created for the structure ``S``, which then defines the
            ``wavelength``, ``theta``, ``azimuth``, and ``polarization``. The
            ``azimuth`` of the plane of incidence relative to the sample
            applies to the multilayer formalisms only, see
            :attr:`.MultilayerScattering.azimuth`. The ``theta`` and
            ``azimuth`` are pint Quantities with an angular unit, e.g.
            ``45 * u.deg``, or lists of them, see
            :func:`~udkm1Dsim.helpers.to_angle`. The spectral ``bandwidth``
            and the ``divergence`` in the plane of incidence are the FWHM of
            the spectrum and of the angular spread of a pulse, by default 0
            for a monochromatic and collimated pulse, see
            :meth:`get_absorption_profile`. A ``bandwidth`` in a length unit,
            e.g. ``10 * u.nm``, is a profile in the wavelength, and in an
            energy unit, e.g. ``0.1 * u.eV``, a profile in the photon energy.
            The ``divergence`` is an angle Quantity. Their profiles are
            sampled by ``bandwidth_profile`` and ``divergence_profile``
            (``'gauss'`` by default, ``'box'``, ``'lorentz'``, or
            ``'pseudo_voigt'``), ``*_nodes`` (``'uniform'`` by default or
            ``'gaussian_quadrature'``), ``*_points`` (7), ``*_range`` (2,
            in units of the FWHM), and ``*_eta`` (0.5, the Lorentzian
            fraction of a pseudo-Voigt profile), see
            :func:`.instrument.profile_nodes`.
        boundary_conditions (dict{str, float, Quantity}): boundary conditions of
            the top and bottom boundary for the heat diffusion calculation.
            ``top_type`` or ``bottom_type`` must be one of ``boundary_types``.
            For the last two types the corresponding value, ``top_value`` and
            ``bottom_value`` have to be set as :math:`K \\times 1` array, where
            :math:`K` is the number of sub-systems.
        boundary_types (list[str]): description of boundary types.

            * isolator
            * temperature
            * flux

        distances (ndarray[float, Quantity]): spatial grid for heat diffusion (m)
        ode_options (dict): options for scipy solve_ivp ode solver

    """

    _title = "Heat simulation"

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.heat_diffusion = kwargs.get("heat_diffusion", False)
        self.intp_at_interface = kwargs.get("intp_at_interface", 11)
        self._excitation = {
            "fluence": [],
            "delay_pump": [0],
            "pulse_width": [0],
            "wavelength": 800e-9,
            "theta": np.pi / 2,
            "azimuth": 0.0,
            "polarization": "pi",
            "absorption": Parratt,
            "backside": False,
            "bandwidth": 0.0,
            "bandwidth_domain": "wavelength",
            "divergence": 0.0,
            **{
                f"{spread}_{key}": value
                for spread in _SPREADS
                for key, value in _SAMPLING_DEFAULTS.items()
            },
        }
        self._distances = np.array([])
        # absorption profiles of a run of calc_temp_map
        self._profile_cache = None
        self.backend = kwargs.get("backend", "scipy")
        self.boundary_types = ["isolator", "temperature", "flux"]
        self._boundary_conditions = {
            "top_type": 0,
            "top_value": np.array([]),
            "bottom_type": 0,
            "bottom_value": np.array([]),
        }
        self.ode_options = {
            "method": "Radau",
            "first_step": None,
            "max_step": np.inf,
            "rtol": 1e-3,
            "atol": 1e-6,
        }

    def _property_rows(self):
        """[parameter, value] rows of the simulation properties"""

        def pulses(value, unit):
            # a single pulse without the brackets of the array
            value = value.to(unit)
            value = value[0] if np.size(value.magnitude) == 1 else value
            return f"{value:.4g~P}"

        output = [
            ["excitation fluence", pulses(self.excitation["fluence"], "mJ/cm**2")],
            ["excitation delay", pulses(self.excitation["delay_pump"], "ps")],
            ["excitation pulse length", pulses(self.excitation["pulse_width"], "ps")],
            ["excitation wavelength", "{:.4g~P}".format(self.excitation["wavelength"].to("nm"))],
            ["excitation theta", "{:.4g~P}".format(self.excitation["theta"].to("deg"))],
            ["excitation azimuth", "{:.4g~P}".format(self.excitation["azimuth"].to("deg"))],
            ["excitation polarization", self.excitation["polarization"]],
            ["excitation absorption", self._absorption_name()],
            ["excitation backside", self.excitation["backside"]],
            ["excitation bandwidth", self._spread_text("bandwidth")],
            ["excitation divergence", self._spread_text("divergence")],
            ["heat diffusion", self.heat_diffusion],
            ["backend", self.backend],
            ["interpolate at interfaces", self.intp_at_interface],
            [
                "distances",
                "no distance mesh is set for heat diffusion calculations"
                if self.distances.size == 0
                else "a distance mesh is set for heat diffusion calculations.",
            ],
            ["top boundary type", self.boundary_conditions["top_type"]],
        ]

        if self._boundary_conditions["top_type"] == 1:
            output += [
                [
                    "top boundary temperature",
                    "{:.4g~P}".format(self.boundary_conditions["top_value"].to("K")),
                ]
            ]
        elif self._boundary_conditions["top_type"] == 2:
            output += [
                [
                    "top boundary flux",
                    "{:.4g~P}".format(self.boundary_conditions["top_value"].to("W/m**2")),
                ]
            ]

        output += [["bottom boundary type", self.boundary_conditions["bottom_type"]]]

        if self._boundary_conditions["bottom_type"] == 1:
            output += [
                [
                    "bottom boundary temperature",
                    "{:.4g~P}".format(self.boundary_conditions["bottom_value"].to("K")),
                ]
            ]
        elif self._boundary_conditions["bottom_type"] == 2:
            output += [
                [
                    "bottom boundary flux",
                    "{:.4g~P}".format(self.boundary_conditions["bottom_value"].to("W/m**2")),
                ]
            ]

        return super()._property_rows() + output

    def _parameters(self):
        """dict of the parameters of the simulation"""
        return {
            **super()._parameters(),
            "excitation": {**self.excitation, "absorption": self._absorption_name()},
            "heat_diffusion": self.heat_diffusion,
            "backend": self.backend,
            "intp_at_interface": self.intp_at_interface,
            "distances": self.distances,
            "boundary_conditions": self.boundary_conditions,
            "ode_options": dict(self.ode_options),
        }

    def get_hash(self, delays, init_temp, **kwargs):
        """get_hash

        Returns a unique hash given by the ``delays`` and ``init_temp``, the
        parameters of the simulation including the ``ode_options``, but not
        the ``backend``, which yields the same results, as well as the sample
        structure hash for relevant thermal parameters and the hash of the
        absorption formalism of the excitation. The ``bandwidth`` and
        ``divergence`` of the excitation and their sampling enter only, if
        they are not 0 for at least one pulse.

        Args:
            delays (ndarray[float]): delay grid for the simulation.
            init_temp (ndarray[float]): initial spatial temperature profile.
            **kwargs (float, optional): optional parameters.

        Returns:
            hash (str): unique hash.

        """
        excitation = {**self._excitation, "absorption": self._absorption_name()}
        if any(
            isinstance(self._pulse(index)["absorption"], MultilayerScattering)
            for index in range(self.num_pulses)
        ):
            # the current values of configured instances as absorption
            excitation.update({key: self._pulse_values(key) for key in _INSTANCE_KEYS})
        for spread in _SPREADS:
            if not self._spread_active(spread):
                # a monochromatic and collimated excitation keeps former hashes
                for key in [spread, *(f"{spread}_{key}" for key in _SAMPLING_DEFAULTS)]:
                    del excitation[key]
                if spread == "bandwidth":
                    del excitation["bandwidth_domain"]
        param = [
            delays,
            init_temp,
            self.heat_diffusion,
            self.intp_at_interface,
            excitation,
            self._boundary_conditions,
            self.distances,
            self.ode_options,
        ]
        for index in range(self.num_pulses):
            if not isinstance(self._pulse(index)["absorption"], str):
                # the formalism depends also on the roughness and magnetization
                param.append(self.create_multilayer_formalism(pulse_index=index).get_hash())

        for value in kwargs.values():
            param.append(value)

        return self.S.get_hash(types="heat") + "_" + make_hash_md5(param)

    def check_initial_temperature(self, init_temp, distances=[]):
        """check_initial_temperature

        Initial temperature for heat simulations can be either a scalar
        temperature which is assumed to be valid for all layers in the
        structure or a temperature profile is given with one temperature for
        each layer in the structure and for each subsystem.
        Alternatively a spatial grid can be provided.

        Args:
            init_temp (float, Quantity, ndarray[float, Quantity]): initial
                temperature scalar or array (K).
            distances (ndarray[float, Quantity], optional): spatial grid of the
                initial temperatrue.

        Returns:
            init_temp (ndarray[float]): checked initial temperature as array on
            the according spatial grid.

        """

        try:
            init_temp = init_temp.to("K").magnitude
        except AttributeError:
            pass

        N = len(distances)
        if N == 0:
            N = self.S.num_layers

        K = self.S.num_sub_systems
        # check size of initTemp
        if np.size(init_temp) == 1:
            # it is the same initial temperature for all layers
            init_temp = init_temp * np.ones([N, K])
        elif np.shape(init_temp) != (N, K):
            # init_temp is a vector but has not as many elements as layers
            raise ValueError(
                "The initial temperature vector must have 1 or "
                "NxK elements, where N is the number of layers "
                "in the structure and K the number of subsystems!"
            )

        return init_temp

    def check_excitation(self, delays):
        r"""check_excitation

        The optical excitation is a dictionary with fluence
        :math:`F` (J/m²), delays :math:`t` (s) of the pump events, and pulse
        width :math:`\tau` (s). :math:`N` is the number of pump events.

        Traverse excitation vector to update the ``delay_pump`` :math:`t_p`
        vector for finite pulse durations :math:`w(i)` as follows

        .. math::

            [t_p(i)-\mbox{window}\cdot w(i)]:[t_p(i)+\mbox{window}\cdot w(i)]:
            [w(i)/\mbox{intp}]

        and to combine excitations which have overlapping intervals.

        Moreover the incidence angle :math:`\vartheta` is taken into account for
        the user-defined incidence fluence in order to project the laser
        footprint onto the sample surface. For a ``divergence`` of the pulse,
        the central angle ``theta`` gives the footprint, as all rays hit the
        same spot on the sample.

        Args:
            delays (ndarray[Quantity]): delays range of simulation (s).

        Returns:
            (tuple):
            - *res (list[ndarray[float]])* - resulting list of excitations with
              interpolated delay density around excitations.
            - *fluence (ndarray[float])* - projected excitation fluences.
            - *delay_pump (ndarray[float])* - delays of the excitations.
            - *pulse_width (ndarray[float])* - pulse widths of the excitations.

        """
        delays = delays.to("s").magnitude
        fluence = self._excitation["fluence"]
        delay_pump = self._excitation["delay_pump"]
        pulse_width = self._excitation["pulse_width"]
        # the incidence angle of every pulse
        theta = np.array([self._pulse(i)["theta"] for i in range(len(fluence))])

        fluence = fluence * np.sin(theta)
        if np.size(np.unique(theta)) == 1:
            self.disp_message(
                f"Surface incidence fluence scaled by factor {np.sin(theta[0]):5.4f}"
                f" due to incidence angle theta={np.rad2deg(theta[0]):5.2f} deg"
            )
        elif len(theta) > 1:
            self.disp_message(
                "Surface incidence fluences scaled by factors "
                f"{np.array2string(np.sin(theta), precision=4)} due to incidence angles "
                f"theta={np.array2string(np.rad2deg(theta), precision=2)} deg"
            )

        # throw warnings if heat diffusion should be enabled
        if (self.S.num_sub_systems > 1) and not self.heat_diffusion:
            warnings.warn(
                "If you are introducing more than 1 subsystem you should enable heat diffusion!"
            )

        if np.sum(pulse_width) > 0 and not self.heat_diffusion:
            pulse_width = np.zeros_like(fluence)
            warnings.warn(
                "The effect of finite pulse duration of the excitation "
                "is only considered if heat diffusion is enabled! "
                "All pulse durations are set to 0!"
            )

        n_excitation = []  # the result of the traversed excitation is a cell vector
        window = 1.5  # window factor for finite pulse duration
        intp = 1000  # interpolation factor for finite pulse duration

        i = 0  # start counter
        while i < len(delay_pump):
            k = i
            temp = []
            # check all upcoming excitations if they overlap with the current
            while k < len(delay_pump):
                temp.append([delay_pump[k], pulse_width[k], fluence[k]])
                if (k + 1 < len(delay_pump)) and (
                    (delay_pump[k] + window * pulse_width[k])
                    >= (delay_pump[k + 1] - window * pulse_width[k + 1])
                ):
                    # there is an overlap in time so add the next
                    # excitation to the current element
                    k += 1
                    if pulse_width[k] == 0:
                        # an overlapping pulse cannot have a pulseWidth
                        # of 0! Throw an error!
                        raise ValueError("Overlapping pulse must have duration > 0!")
                else:
                    # no overlap, so go to the next iteration of the outer while loop
                    break

            # calculate the new time vector of the current excitation
            delta_delay = np.min(pulse_width[i : (k + 1)]) / intp
            if delta_delay == 0:
                # its pulse_width = 0 or no heat diffusion was enabled
                # so calculate just at a single delay step
                interval = np.array([delay_pump[i]])
            else:
                interval = np.r_[
                    (delay_pump[i] - window * pulse_width[i]) : (
                        delay_pump[k] + window * pulse_width[k]
                    ) : delta_delay
                ]
            # update the new excitation list
            n_excitation.append(
                [interval, [t[0] for t in temp], [t[1] for t in temp], [t[2] for t in temp]]
            )
            i = k + 1  # increase counter

        # traverse the n_excitation list and add additional time vectors between
        # the pump events for the later temperature calculation
        res = []  # initialize the result list

        # check for delay < delay_pump[0]
        if np.size(delays[delays < np.min(n_excitation[0][0])]) > 0:
            res.append([delays[delays < n_excitation[0][0][0]], [], [], []])
        else:
            warnings.warn("Please add more delay steps before the first excitation!")

        # traverse n_excitation
        for i, excitation in enumerate(n_excitation):
            res.append(excitation)
            # print(excitation[0][-1])
            # print(n_excitation[i+1][0][0])
            if i + 1 < len(n_excitation):
                # there is an upcoming pump event
                if (
                    np.size(
                        delays[
                            np.logical_and(
                                delays > excitation[0][-1], delays < n_excitation[i + 1][0][0]
                            )
                        ]
                    )
                    > 0
                ):
                    # there are times between the current and next excitation
                    temp = [
                        delays[
                            np.logical_and(
                                delays > excitation[0][-1], delays < n_excitation[i + 1][0][0]
                            )
                        ],
                        [],
                        [],
                        [],
                    ]
                    res.append(temp)
            else:  # this is the last pump event
                if np.size(delays[delays > excitation[0][-1]]) > 0:
                    # there are times after the current last excitation
                    temp = [delays[delays > excitation[0][-1]], [], [], []]
                    res.append(temp)
        return res, fluence, delay_pump, pulse_width

    # ---- absorption -------------------------------------------------------------

    def get_absorption_profile(self, distances=[], pulse_index=None):
        r"""get_absorption_profile

        Returns the differential absorption profile :math:`\mbox{d}A/\mbox{d}z`
        of every pulse of the excitation by Lambert-Beer's law or its
        multilayer formalism, depending on its ``absorption``. Pulses with the
        same parameters, besides their ``fluence``, ``delay_pump``, and
        ``pulse_width``, are calculated only once.

        A pulse with a spectral ``bandwidth`` or a ``divergence`` has the
        weighted mean profile of its wavelengths :math:`\lambda_e` and
        angles :math:`\vartheta_t` of :meth:`get_pulse_nodes`

        .. math::

            \left\langle\frac{\mbox{d}A}{\mbox{d}z}\right\rangle(z) =
            \sum_e \sum_t w_e w_t \frac{\mbox{d}A}{\mbox{d}z}
            (\lambda_e, \vartheta_t, z),

        which is exact for the deposited energy of a linear medium, as the
        spectral components of a pulse add incoherently in its time
        integral, independent of their spectral phase. All rays of a pulse
        hit the same spot on the sample, so that the incident fluence is
        projected by :math:`\sin\vartheta` of the central angle only, see
        :meth:`check_excitation`. The divergence acts in the plane of
        incidence only, and Lambert-Beer's law does not depend on the angle
        at all. The nodes must resolve the spectral and angular features of
        the absorption, e.g. interference fringes of thin films, which is
        checked best by increasing the ``*_points`` of the nodes.

        Args:
            distances (ndarray[float], optional): spatial grid for calculation.
            pulse_index (int, optional): index of the pulse, sorted by
                ``delay_pump``. Defaults to all pulses.

        Returns:
            dAdz (ndarray[float]): differential absorption profile (1/m) of
            shape (N,) for a single pulse or (P, N) for P pulses.

        """
        return self._absorption_profiles(distances, pulse_index, "auto")

    def get_pulse_nodes(self, pulse_index=0):
        r"""get_pulse_nodes

        Wavelengths :math:`\lambda_e`, angles of incidence
        :math:`\vartheta_t`, and normalized weights :math:`w_e w_t` of the
        spectral ``bandwidth`` and the ``divergence`` of a pulse of the
        excitation, see :meth:`get_absorption_profile`, by the nodes of
        :func:`.instrument.profile_nodes` around its ``wavelength`` and
        ``theta``. A ``bandwidth`` in a length unit shifts the wavelength
        :math:`\lambda_e = \lambda + x_e \Delta\lambda`, and in an energy
        unit the photon energy :math:`E_e = E + x_e \Delta E`, for the nodes
        :math:`x_e` in units of the FWHM. A monochromatic and collimated
        pulse has a single node.

        Args:
            pulse_index (int, optional): index of the pulse, sorted by
                ``delay_pump``. Defaults to the first pulse.

        Returns:
            (tuple):
            - *wavelength (Quantity)* - wavelengths (nm) of shape (E,).
            - *theta (Quantity)* - angles of incidence (deg) of shape (T,).
            - *weights (ndarray[float])* - normalized weights of shape
              (E, T).

        """
        wavelength, w_e, theta, w_t = self._pulse_nodes(self._check_pulse_index(pulse_index))
        return (
            Q_(wavelength, u.m).to("nm"),
            Q_(theta, u.rad).to("deg"),
            np.outer(w_e, w_t),
        )

    def get_absorption_from_lambert_beer(self, distances=[], pulse_index=None):
        r"""get_absorption_from_lambert_beer

        The transmission is given by:

        .. math:: \tau = \frac{I}{I_0} =  \exp(-z/ \zeta)

        and the absorption by:

        .. math:: A = 1 - \tau =  1 - \exp(-z/\zeta)

        The absorption profile can be derived from the spatial derivative:

        .. math::

            \frac{\mbox{d}A(z)}{\mbox{d}z} = \frac{1}{\zeta}
            \exp(-z/\zeta)

        Args:
            distances (ndarray[float], optional): spatial grid for calculation.
            pulse_index (int, optional): index of the pulse, sorted by
                ``delay_pump``, for its ``wavelength`` and ``backside``.
                Defaults to all pulses.

        Returns:
            dAdz (ndarray[float]): differential absorption profile (1/m) of
            shape (N,) for a single pulse or (P, N) for P pulses calculated
            by Lambert-Beers law.

        """
        return self._absorption_profiles(distances, pulse_index, "lambert_beer")

    def get_absorption_from_multilayer(self, distances=[], pulse_index=None):
        """get_absorption_from_multilayer

        Differential absorption profile :math:`\\mbox{d}A/\\mbox{d}z` from
        the decrease of the Poynting flux of the multilayer formalism of the
        pulses of the excitation, see
        :meth:`~.MultilayerScattering.calc_homogeneous_absorption`, which
        requires a multilayer formalism as ``absorption`` of the pulses.

        Args:
            distances (ndarray[float], optional): spatial grid for calculation.
            pulse_index (int, optional): index of the pulse, sorted by
                ``delay_pump``. Defaults to all pulses.

        Returns:
            dAdz (ndarray[float]): differential absorption profile (1/m) of
            shape (N,) for a single pulse or (P, N) for P pulses.

        """
        return self._absorption_profiles(distances, pulse_index, "multilayer")

    def create_multilayer_formalism(self, cls=None, pulse_index=0, nodes=False):
        """create_multilayer_formalism

        Multilayer formalism for the ``wavelength``, ``theta``, ``azimuth``,
        ``polarization``, and ``backside`` of a pulse of the excitation, e.g.
        to calculate its reflectivity or the fields inside the structure. A
        configured instance as ``absorption`` of the pulse is returned as it
        is, or for a backside excitation as a copy with all its settings for
        the reversed structure.

        Args:
            cls (type[MultilayerScattering], optional): formalism, e.g.
                :class:`.Parratt`. Defaults to the ``absorption`` of the
                pulse.
            pulse_index (int, optional): index of the pulse, sorted by
                ``delay_pump``. Defaults to the first pulse.
            nodes (bool, optional): set the formalism to the wavelengths and
                angles of the ``bandwidth`` and ``divergence`` of the pulse,
                see :meth:`get_pulse_nodes`, where a configured instance is
                copied without its :attr:`~.Scattering.instrument`. Defaults
                to False for the ``wavelength`` and ``theta`` of the pulse.

        Returns:
            formalism (MultilayerScattering): formalism for the pulse.

        """
        index = self._check_pulse_index(pulse_index)
        pulse = self._pulse(index)
        backside = pulse["backside"]
        if cls is None:
            cls = pulse["absorption"]
        if isinstance(cls, MultilayerScattering):
            if cls.S is not self.S:
                raise ValueError(
                    "The absorption formalism of the excitation must be created for the "
                    "structure of the heat simulation!"
                )
            if not (backside or nodes):
                return cls
            # the formalisms read their structure only during the calculation,
            # so that a copy keeps all settings for the reversed structure
            formalism = copy(cls)
            if backside:
                formalism.S = self.S.reverse()
            if nodes:
                # the nodes must not change the experiment of the instance,
                # while its instrument describes a probe and not the pump
                formalism.experiment = deepcopy(cls.experiment)
                formalism._instrument = None
                wavelength, _, theta, _ = self._pulse_nodes(index)
                formalism.wl = Q_(wavelength, u.m)
                formalism.theta = Q_(theta, u.rad)
            return formalism
        if not (isinstance(cls, type) and issubclass(cls, MultilayerScattering)):
            raise TypeError(
                "The absorption formalism must be a multilayer formalism, e.g. Parratt!"
            )
        formalism = cls(
            self.S.reverse() if backside else self.S,
            True,
            save_data=False,
            disp_messages=False,
            progress_bar=False,
        )
        if nodes:
            wavelength, _, theta, _ = self._pulse_nodes(index)
        else:
            wavelength, theta = np.r_[pulse["wavelength"]], np.r_[pulse["theta"]]
        formalism.wl = Q_(wavelength, u.m)
        formalism.theta = Q_(theta, u.rad)
        formalism.azimuth = Q_(pulse["azimuth"], u.rad)
        polarization = pulse["polarization"]
        if polarization not in _POLARIZATIONS:
            raise ValueError(
                f"The polarization {polarization!r} of a former configured instance must be "
                f"replaced by one of {', '.join(_POLARIZATIONS)}!"
            )
        formalism.set_incoming_polarization(polarization)
        return formalism

    def _absorption_profiles(self, distances, pulse_index, kind):
        """absorption profiles of a single or all pulses by Lambert-Beer's law,
        their multilayer formalism, or ``auto`` by their ``absorption``, which
        are calculated once for identical pulses and on the same grid during
        a run of :meth:`calc_temp_map`, while each pulse is reported once"""
        if pulse_index is None:
            indices = range(self.num_pulses)
        else:
            indices = [self._check_pulse_index(pulse_index)]
        cache = self._profile_cache
        if cache is None:
            cache = {"profiles": {}, "reported": set()}
        if len(distances) == 0:
            # the layer centers, as for an explicit grid of the same depths
            distances = self.S._layer_distances()[2]
        grid = np.asarray(distances, dtype=float).tobytes()
        rows = []
        for index in indices:
            pulse_key = self._pulse_key(index)
            lambert_beer = kind == "lambert_beer" or (
                kind == "auto" and isinstance(self._pulse(index)["absorption"], str)
            )
            key = (lambert_beer, pulse_key, grid)
            if key not in cache["profiles"]:
                if lambert_beer:
                    profile, report = self._lambert_beer_absorption(distances, index)
                else:
                    profile, report = self._multilayer_absorption(distances, index)
                cache["profiles"][key] = profile
                if (lambert_beer, pulse_key) not in cache["reported"]:
                    cache["reported"].add((lambert_beer, pulse_key))
                    self.disp_message(self._absorption_message(pulse_key, report))
            rows.append(cache["profiles"][key])
        rows = np.asarray(rows)
        return rows[0] if len(rows) == 1 else rows

    @contextmanager
    def _profile_run(self):
        """context of a run, in which every absorption profile is calculated
        and reported only once"""
        outer = self._profile_cache is not None
        if not outer:
            self._profile_cache = {"profiles": {}, "reported": set()}
        try:
            yield
        finally:
            if not outer:
                self._profile_cache = None

    def _absorption_message(self, pulse_key, report):
        """message of the absorption of all pulses with the same parameters"""
        indices = [i for i in range(self.num_pulses) if self._pulse_key(i) == pulse_key]
        pulse = self._pulse(indices[0])
        delays = np.atleast_1d(self._excitation["delay_pump"])
        delays = [f"{delays[i] * 1e12:.4g}" for i in indices if i < len(delays)]
        label = "pulse" if len(indices) == 1 else "pulses"
        label = f"{label} {', '.join(map(str, indices))}"
        if delays:
            label += f" at {', '.join(delays)} ps"
        side = "backside" if pulse["backside"] else "frontside"
        wavelength = f"{pulse['wavelength'] * 1e9:.4g} nm" + self._spread_note(pulse, "bandwidth")
        # percentages without the sign of rounding residues around 0, i.e. -0.0 %
        percent = {
            key: round(value * 100, 1) + 0.0 for key, value in report.items() if key != "formalism"
        }
        if "reflection" not in report:
            return (
                f"Absorption of {label} by Lambert-Beer's law for {wavelength}, {side}: "
                f"absorption in the sample of {percent['absorption']:.1f} %."
            )
        return (
            f"Absorption of {label} by {report['formalism']} for {wavelength}, "
            f"theta = {np.degrees(pulse['theta']):.4g} deg"
            f"{self._spread_note(pulse, 'divergence')}, "
            f"azimuth = {np.degrees(pulse['azimuth']):.4g} deg, "
            f"{pulse['polarization']} polarization, {side}: "
            f"reflection of {percent['reflection']:.1f} %, "
            f"absorption in the sample of {percent['absorption']:.1f} %, and "
            f"transmission into the substrate of {percent['substrate']:.1f} %."
        )

    def _lambert_beer_absorption(self, distances, index):
        """absorption profile of a pulse by Lambert-Beer's law and the
        absorbed fraction of the structure"""
        pulse = self._pulse(index)
        backside = pulse["backside"]
        if backside:
            structure = self.S.reverse()
        else:
            structure = self.S

        if len(distances) == 0:
            # if no distances are set, calculate the extinction on
            # the middle of each unit cell
            d_start, _, distances = structure._layer_distances()
        else:
            d_start, d_end, _ = structure._layer_distances()
            if backside:
                # np.clip() prevents negative distances after flipping
                # due to floating point rounding errors
                distances = np.clip(np.flip(d_end[-1] - distances), 0, d_end[-1])

        interfaces = structure._interface_distances()
        # the wavelengths of the bandwidth, while the angle does not matter
        wavelengths, weights, _, _ = self._pulse_nodes(index)

        N = len(distances)
        dAdz = np.zeros(N)  # initialize relative absorbed energies
        I0 = np.ones(len(wavelengths))  # initial intensity of every wavelength
        k = 0  # counter for first layer
        for i in range(len(interfaces) - 1):
            # find the first layer and get properties
            layer_index = finderb(interfaces[i], d_start)
            layer = structure.get_layer_handle(layer_index[0])

            # get all distances in the current layer we have to
            # calculate the absorption profile for
            if i >= len(interfaces) - 2:  # last layer
                z = distances[
                    np.logical_and(distances >= interfaces[i], distances <= interfaces[i + 1])
                ]
            else:
                z = distances[
                    np.logical_and(distances >= interfaces[i], distances < interfaces[i + 1])
                ]
            m = len(z)
            for e, wavelength in enumerate(wavelengths):
                opt_pen_depth = layer.get_opt_pen_depth(wavelength * u.m).to("m").magnitude
                if not np.isinf(opt_pen_depth):
                    # the layer is absorbing
                    dAdz[k : k + m] += (
                        weights[e]
                        * I0[e]
                        / opt_pen_depth
                        * np.exp(-(z - interfaces[i]) / opt_pen_depth)
                    )
                    # calculate the remaining intensity for the next layer
                    I0[e] = I0[e] * np.exp(-(interfaces[i + 1] - interfaces[i]) / opt_pen_depth)
            k = k + m  # set the counter

        if backside:
            # for backside excitation the result must be reversed
            dAdz = np.flip(dAdz)

        return dAdz, {"absorption": float(np.sum(weights * (1 - I0)))}

    def _multilayer_absorption(self, distances, index):
        """absorption profile of a pulse by its multilayer formalism and its
        reflection, absorption in the structure, and flux into the substrate"""
        pulse = self._pulse(index)
        if isinstance(pulse["absorption"], str):
            raise ValueError(
                f"The pulse {index} is absorbed by Lambert-Beer's law and not by a "
                "multilayer formalism!"
            )
        backside = pulse["backside"]
        formalism = self.create_multilayer_formalism(pulse_index=index, nodes=True)
        weights = self.get_pulse_nodes(index)[2]
        d_start, d_end, d_mid = formalism.S._layer_distances()
        if len(distances) == 0:
            distances = d_mid
        else:
            distances = np.asarray(distances, dtype=float)
            if backside:
                # np.clip() prevents negative distances after flipping
                # due to floating point rounding errors
                distances = np.clip(np.flip(d_end[-1] - distances), 0, d_end[-1])
        # the bottom of the structure belongs to its last layer and not to
        # the substrate, as for all other interfaces of the formalism
        z = np.minimum(distances, d_end[-1] * (1 - 128 * np.finfo(float).eps))
        A, _, _, _, A_layers, _, _, _, S_sub, *_ = formalism.calc_homogeneous_absorption(z)
        if np.ndim(A) != 3:
            raise ValueError(
                "The absorption formalism of the excitation must have a single incoming "
                "polarization!"
            )
        # the mean over the nodes of the bandwidth and divergence
        R = formalism.calc_homogeneous_reflectivity_transmissivity()[0]
        report = {
            "formalism": type(formalism).__name__,
            "reflection": float(np.sum(weights * R)),
            "absorption": float(np.sum(weights * np.sum(A_layers, axis=-1))),
            "substrate": float(np.sum(weights * S_sub)),
        }
        dAdz = A[0, 0] if weights.size == 1 else np.einsum("et,etn->n", weights, A)
        if backside:
            # for backside excitation the results must be reversed
            dAdz = np.flip(dAdz)
        return dAdz, report

    # ---- temperature ------------------------------------------------------------

    def get_temperature_after_delta_excitation(
        self, fluence, init_temp, distances=[], pulse_index=0
    ):
        """get_temperature_after_delta_excitation

        Returns the final temperature and temperature change for each layer of
        the sample structure or the given ``distances`` after an optical
        excitation, see :meth:`calc_temperature_after_delta_excitation`.

        Args:
            fluence (float, Quantity): fluence on the surface (J/m²), i.e.
                already projected by :math:`\\sin\\vartheta` of the incidence
                angle, unlike the ``fluence`` of the ``excitation``, see
                :meth:`check_excitation`.
            init_temp (float, Quantity, ndarray[float, Quantity]): initial
                temperature scalar or array (K).
            distances (ndarray[float], optional): spatial grid for calculation.
            pulse_index (int, optional): index of the pulse of the excitation,
                sorted by ``delay_pump``, for its absorption profile.
                Defaults to the first pulse.

        Returns:
            result (Result): with the final temperature ``final_temp`` and
            the temperature change ``delta_temp`` (K) of the dimensions
            ``distances`` (m) of the layer centers or of the given grid and
            ``subsystems``, see :class:`~udkm1Dsim.simulations.result.Result`.

        """
        final_temp, delta_temp = self.calc_temperature_after_delta_excitation(
            fluence, init_temp, distances, pulse_index
        )
        fluence = fluence if isinstance(fluence, Q_) else Q_(fluence, "J/m**2")
        result = self.new_map_result(
            "temperature after delta excitation",
            distances=distances if len(distances) > 0 else None,
            inputs={"fluence": fluence.to("mJ/cm**2"), "pulse_index": pulse_index},
        )
        result.add("final_temp", final_temp, ("distances", "subsystems"), "K")
        result.add("delta_temp", delta_temp, ("distances", "subsystems"), "K")
        return result

    def calc_temperature_after_delta_excitation(
        self, fluence, init_temp, distances=[], pulse_index=0
    ):
        r"""calc_temperature_after_delta_excitation

        Calculate the final temperature and temperature change for each layer
        of the sample structure after an optical excitation with a fluence
        :math:`F` (J/m^2) and an initial temperature :math:`T_1` (K):

        .. math:: \Delta E = \int_{T_1}^{T_2} m \, c(T)\, \mbox{d}T

        where :math:`\Delta E` is the absorbed energy in each layer and
        :math:`c(T)` is the temperature-dependent heat capacity (J/kg K) and
        :math:`m` is the mass (kg).

        The absorbed energy per layer can be linearized from the absorption
        profile :math:`\mbox{d} \alpha / \mbox{d}z` as

        .. math:: \Delta E = \frac{\mbox{d} \alpha}{\mbox{d}z} E_0 \Delta z

        where :math:`E_0` is the initial energy impinging on the first layer
        given by the fluence :math:`F = E / A` on the surface area :math:`A`.
        :math:`\Delta z` is equal to the thickness of each layer.

        Finally, one has to minimize the following modulus to obtain the
        final temperature :math:`T_2` of each layer:

        .. math::

             \left| \int_{T_1}^{T_2} m c(T)\, \mbox{d}T - \frac{\mbox{d}\alpha}
             {\mbox{d}z} E_0 \Delta z \right| \stackrel{!}{=} 0

        Args:
            fluence (float, Quantity): fluence on the surface (J/m²), i.e.
                already projected by :math:`\sin\vartheta` of the incidence
                angle, unlike the ``fluence`` of the ``excitation``, see
                :meth:`check_excitation`.
            init_temp (float, Quantity, ndarray[float, Quantity]): initial
                temperature scalar or array (K).
            distances (ndarray[float], optional): spatial grid for calculation.
            pulse_index (int, optional): index of the pulse of the excitation,
                sorted by ``delay_pump``, for its absorption profile.
                Defaults to the first pulse.

        Returns:
            (tuple):
            - *final_temp (ndarray[float])* - final temperature after delta
              excitation.
            - *delta_T (ndarray[float])* - temperature change.

        """
        # initialize
        t1 = time()
        if len(distances) == 0:
            # if no distances are set, calculate the extinction on
            # the middle of each unit cell
            d_start, _, distances = self.S._layer_distances()
        else:
            d_start, _, _ = self.S._layer_distances()

        # absorption profile from Lambert-Beer's law or multilayer absorption
        dAdz = self.get_absorption_profile(distances=distances, pulse_index=pulse_index)

        # normalize fluence to a scalar (single delta excitation)
        try:
            fluence = fluence.to("J/m**2").magnitude
        except AttributeError:
            # fluence might be a list/array; squeeze to scalar if possible
            fluence = np.asarray(fluence, dtype=float).squeeze()
        if np.ndim(fluence) != 0:
            raise ValueError(
                f"Delta excitation expects a single fluence value; got shape {np.shape(fluence)}"
            )

        int_heat_capacities = self.S.get_layer_property_vector("thermal.heat_capacity.integral")
        thicknesses = self.S.get_layer_property_vector("structural.thickness.magnitude")
        masses = self.S.get_layer_property_vector("structural.mass_unit_area.magnitude")
        # masses are normalized to 1Ang^2
        E0 = np.array(fluence) * (1 * u.angstrom**2).to_base_units().magnitude

        # check the initial temperature
        init_temp = self.check_initial_temperature(init_temp, distances)
        final_temp = init_temp.copy()
        # traverse layers
        for i, dist in enumerate(distances):
            idx = finderb(dist, d_start)[0]
            if dAdz[i] > 0:
                # if there is absorption in the current layer
                del_E = dAdz[i] * E0 * thicknesses[idx]

                def fun(final_temp):
                    return (
                        masses[idx]
                        * (
                            int_heat_capacities[idx][0](final_temp)
                            - int_heat_capacities[idx][0](init_temp[i, 0])
                        )
                        - del_E
                    )

                final_temp[i, 0] = brentq(fun, init_temp[i, 0], 1e5)
        delta_T = final_temp - init_temp  # this is the temperature change
        self.disp_message(
            f"Elapsed time for _temperature_after_delta_excitation_: {time() - t1:f} s"
        )
        return final_temp, delta_T

    def get_temp_map(self, delays, init_temp):
        """get_temp_map

        Returns a temperature profile for the sample structure after optical
        excitation. The result can be saved using an unique hash of the sample
        and the simulation parameters in order to reuse it.

        Args:
            delays (ndarray[Quantity]): delays range of simulation (s).
            init_temp (float, Quantity, ndarray[float, Quantity]): initial
                temperature scalar or array (K).

        Returns:
            result (Result): with the spatio-temporal ``temp_map`` and its
            change ``delta_temp_map`` (K) of the dimensions ``delays``,
            ``distances`` of the layer centers, and ``subsystems``, the
            checked initial temperature ``init_temp``, and the absorption
            profile ``absorption`` (1/m) of every pulse of the dimensions
            ``pulses``, sorted by ``delay_pump``, and ``distances``, see
            :meth:`get_absorption_profile` and
            :class:`~udkm1Dsim.simulations.result.Result`.

        """
        init_temp = self.check_initial_temperature(init_temp)  # check the initial temperature

        def calc():
            with self._profile_run():
                return calc_result()

        def calc_result():
            temp_map, delta_temp_map, _ = self.calc_temp_map(delays, init_temp)
            result = self.new_map_result("temperature map", delays)
            # the singleton dimensions are squeezed out of the maps
            sizes = (np.size(delays), self.S.num_layers, np.shape(init_temp)[-1])
            dims = [
                dim for dim, size in zip(("delays", "distances", "subsystems"), sizes) if size > 1
            ]
            result.add("temp_map", temp_map, dims, "K")
            result.add("delta_temp_map", delta_temp_map, dims, "K")
            result.add("init_temp", init_temp, ("distances", "subsystems"), "K")
            if len(self._excitation["fluence"]) > 0:
                # the absorption profiles at the layer centers, which are
                # squeezed for a single pulse
                absorption = self.get_absorption_profile()
                dims = ("pulses", "distances") if np.ndim(absorption) == 2 else ("distances",)
                result.add("absorption", absorption, dims, "1/m")
            return result

        return self._cached_result("temp_map", self.get_hash(delays, init_temp), calc, False)

    def calc_temp_map(self, delays, input_init_temp):
        """calc_temp_map

        Calculates the temperature profile for the sample structure after optical
        excitation. Heat diffusion can be included if ``heat_diffusion = true``.

        Args:
            delays (ndarray[Quantity]): delays range of simulation (s).
            input_init_temp (float, Quantity, ndarray[float, Quantity]): initial
                temperature scalar or array (K).

        Returns:
            (tuple):
            - *temp_map (ndarray[float])* - spatio-temporal temperature map.
            - *delta_temp_map (ndarray[float])* - spatio-temporal temperature
              change map.
            - *checked_excitations (list[ndarray[float]])* - resulting list of
              checked excitations.

        """
        with self._profile_run():
            return self._calc_temp_map(delays, input_init_temp)

    def _calc_temp_map(self, delays, input_init_temp):
        """:meth:`calc_temp_map` in the context of a run, in which every
        absorption profile is calculated and reported only once"""
        t1 = time()
        # all thermal parameters of the heat simulation must have the same number of subsystems
        self.S.check_num_sub_systems("heat_capacity", "therm_cond", "sub_system_coupling")
        # initialize
        L = self.S.num_layers
        K = self.S.num_sub_systems
        F = len(self._excitation["fluence"])  # total number of excitations

        # there is an initial time step for the init_temp - we will remove it later on
        # check the initial temperature
        input_init_temp = self.check_initial_temperature(input_init_temp)
        checked_excitation, _, _, _ = self.check_excitation(delays)  # check excitation
        _, _, d_mid = self.S._layer_distances()

        if self.heat_diffusion:
            # in case heat diffusion is enabled all calculation are first done
            # on an interpolated grid at the interfaces (or defined by the user)
            if len(self.distances) == 0:
                # no user-defined distaces are given, so calculate heat diffusion
                # by layer and also interpolate at the interfaces
                distances, _ = self.S.interp_distance_at_interfaces(self.intp_at_interface, False)
            else:
                # a user-defined distances vector is given, so determine the
                # indices for final assignment per layer
                distances = self._distances
        else:
            distances = d_mid

        N = len(distances)
        if np.any(self._excitation["fluence"]):
            # report the absorption of all pulses at the beginning
            self.get_absorption_profile(distances)
        temp_map = np.zeros([1, N, K])
        # interpolate the initial temperature onto distance grid
        init_temp = np.zeros([N, K])
        for iii in range(K):
            init_temp[:, iii] = np.interp(distances, d_mid, input_init_temp[:, iii])
        temp_map[0, :, :] = init_temp  # this is initial temperature before the simulation starts

        excitation_delays = np.array([])
        temp = []
        # traverse excitations
        for i, excitation in enumerate(checked_excitation):
            # reset initial temperature for delta excitation with heat diffusion enabled
            special_init_temp = []
            # extract excitation parameters for the current iteration
            sub_delays = excitation[0]
            excitation_delays = np.concatenate((excitation_delays, sub_delays))
            delay_pump = excitation[1]
            pulse_width = excitation[2]
            fluence = excitation[3]
            # determine if a temperature gradient is present and if
            # heat diffusion is required
            temp_gradient = np.sum(np.sum(np.abs(np.diff(temp_map[-1, :, :], 1, 0))))
            if (
                self.heat_diffusion
                and (len(sub_delays) > 2)
                and (
                    (np.sum(fluence) == 0 and temp_gradient != 0)
                    or (np.sum(fluence) > 0 and np.sum(pulse_width) > 0)
                    or (
                        self._boundary_conditions["top_type"]
                        + self._boundary_conditions["bottom_type"]
                    )
                    > 0
                )
            ):
                # heat diffusion enabled and more than 2 time steps AND
                # either no excitation with temperature gradient or excitation with finite pulse
                # duration
                if np.sum(fluence) == 0:
                    self.disp_message("Calculating _heat_diffusion_ without excitation...")
                else:
                    # the pulses by their indices, as for pulse_index
                    pulses = self._pulse_indices(delay_pump)
                    pulses = (
                        f"pulse {pulses[0]}"
                        if len(pulses) == 1
                        else f"pulses {pulses[0]}-{pulses[-1]}"
                    )
                    self.disp_message(f"Calculating _heat_diffusion_ for {pulses} of {F} ...")

                start = 0
                stop = 0
                if i > 0:
                    # check if there was a interval before and add
                    # last time of this interval to the current
                    sub_delays = np.r_[checked_excitation[i - 1][0][-1], sub_delays]
                    start = 1
                if (
                    i < len(checked_excitation) - 1
                    and np.sum(checked_excitation[i + 1][3]) > 0
                    and np.sum(checked_excitation[i + 1][2]) == 0
                ):
                    # there is a next interval of delta excitation so
                    # we add this time at the end of the current
                    # interval
                    sub_delays = np.r_[sub_delays, checked_excitation[i + 1][0][0]]
                    stop = 1

                # calc heat diffusion
                temp = self.calc_heat_diffusion(
                    init_temp, distances, sub_delays, delay_pump, pulse_width, fluence
                )

                if stop == 1:
                    # there is an upcoming delta excitation so we have
                    # to set the initial temperature for this next
                    # interval separately
                    special_init_temp = temp[-1, :, :]
                    temp = temp[start:-1, :, :]
                else:
                    temp = temp[start:, :, :].copy()
                # cut the before added time steps
            elif np.sum(fluence) > 0 and (
                not self.heat_diffusion or (self.heat_diffusion and np.sum(pulse_width) == 0)
            ):
                # excitation with no heat diffusion -> only delta excitation
                # possible in this case OR excitation with heat diffusion and
                # pulse_width equal to 0
                temp, _ = self.calc_temperature_after_delta_excitation(
                    fluence, init_temp, distances, self._pulse_indices(delay_pump)[0]
                )
                temp = np.reshape(temp, [1, np.size(temp, 0), np.size(temp, 1)])
            else:
                # no excitation and no heat diffusion or not enough time
                # step to calculate heat diffusion, so just repeat the
                # initial temperature + every unhandled case
                temp = np.tile(np.reshape(init_temp, [1, N, K]), [len(sub_delays), 1, 1])

            # concat results
            temp_map = np.append(temp_map, temp, axis=0)
            # set the initial temperature for the next iteration
            if len(special_init_temp) > 0:
                init_temp = special_init_temp
            else:
                init_temp = temp_map[-1, :, :].copy()

        if not np.array_equal(excitation_delays, delays.to("s").magnitude) or self.heat_diffusion:
            # if the time grid for the calculation is not the same as
            # the grid to return the results on. Then extrapolate the
            # results on the original delay array but keep the first
            # element in time for the deltaTempMap calculation.
            temp = temp_map.copy()
            temp_map = np.zeros([len(delays) + 1, L, K])
            for iii in range(K):
                init = np.interp(d_mid, distances, temp[0, :, iii]).reshape([1, L])
                f = RectBivariateSpline(
                    distances, excitation_delays, temp[1:, :, iii].T, kx=1, ky=1
                )
                temp_map[:, :, iii] = np.append(init, f(d_mid, delays.to("s").magnitude).T, axis=0)

        # calculate the difference temperature map
        delta_temp_map = np.diff(temp_map, axis=0)
        # delete the initial temperature that was added at the beginning
        temp_map = temp_map[1:, :, :]
        self.disp_message(f"Elapsed time for _temp_map_: {time() - t1:f} s")
        return np.squeeze(temp_map), np.squeeze(delta_temp_map), checked_excitation

    def calc_heat_diffusion(self, init_temp, distances, delays, delay_pump, pulse_width, fluence):
        r"""calc_heat_diffusion

        Calculates a temperature profile including heat diffusion for a given
        delay array and initial temperature profile. Here we have to solve the
        1D heat diffusion equation for :math:`N` subsystems:

        .. math::

            c_1(T_1) \rho \frac{\partial T_1}{\partial t} & = & \frac{\partial}{\partial z}
              \left( \kappa_1(T_1) \frac{\partial T_1}{\partial z} \right)
              + G_1(T_1,...,T_N) + S(z,t) \\
            & \vdots &  \\
            c_N(T_N) \rho \frac{\partial T_N}{\partial t} & = & \frac{\partial}{\partial z}
              \left( \kappa_N(T_N) \frac{\partial T_N}{\partial z} \right) + G_N(T_1,...,T_N)

        where :math:`T_i` is the temperature (K), :math:`z` the distance (m),
        :math:`t` the delay (s), :math:`c_i(T)` the temperature dependent heat
        capacity (J/kg K), :math:`\rho` the density (kg/m³) and :math:`\kappa_i(T)`
        is the temperature-dependent thermal conductivity (W/m K) and
        :math:`S(z,t)` is a source term (W/m³).
        The energy flow between the subsystems is given by the ``sub_system_coupling``
        parameter :math:`G_i(T_1,...,T_N)` of the individual layers.
        The index :math:`i` refers to the :math:`i`-th subsystem.

        Args:
            init_temp (float, Quantity, ndarray[float, Quantity]): initial
                temperature scalar or array (K).
            distances (ndarray[float]): spatial grid for calculation.
            delays (ndarray[Quantity]): delays range of simulation (s).
            delay_pump (ndarray[float]): delays of the excitations, which
                select the pulses of the ``excitation`` for their absorption
                profiles.
            pulse_width (ndarray[float]): pulse widths of the excitations.
            fluence (ndarray[float]): excitation fluences.

        Returns:
            temp_map (ndarray[float]): spatio-temporal temperature map.

        """
        t1 = time()
        init_temp = self.check_initial_temperature(init_temp, distances).copy()
        # the nodes of a temperature boundary are at the boundary value
        bc = self._boundary_conditions
        if bc["top_type"] == 1:
            init_temp[0, :] = bc["top_value"]
        if bc["bottom_type"] == 1:
            init_temp[-1, :] = bc["bottom_value"]
        d_start, _, _ = self.S._layer_distances()

        if np.any(fluence):
            # one absorption profile per pulse
            dAdz = np.array(
                [
                    self.get_absorption_profile(distances=distances, pulse_index=index)
                    for index in self._pulse_indices(delay_pump)
                ]
            )
        else:
            dAdz = np.zeros((len(fluence), len(distances)))

        # indices of the unique layers in respect to the spatial grid, so that
        # the solver can group the grid points of identical layers
        _, unique_layers = self.S.unique_layers
        unique_indices = np.asarray(self.S.layer_vectors[0], dtype=int)
        indices = unique_indices[finderb(distances, d_start)]
        # properties of the unique layers
        therm_conds = [layer.thermal.therm_cond.functional for layer in unique_layers]
        therm_cond_takes_vector = [
            layer.thermal.therm_cond.takes_temperature_vector for layer in unique_layers
        ]
        heat_capacities = [layer.thermal.heat_capacity.functional for layer in unique_layers]
        sub_system_couplings = [
            layer.thermal.sub_system_coupling.functional for layer in unique_layers
        ]
        densities = np.array([layer.structural.density.magnitude for layer in unique_layers])

        if self.backend == "numba":
            from .solvers.heat_numba import solve
        else:
            from .solvers.heat_scipy import solve

        temp_map = solve(
            init_temp,
            distances,
            delays,
            delay_pump,
            pulse_width,
            fluence,
            therm_conds,
            therm_cond_takes_vector,
            heat_capacities,
            sub_system_couplings,
            densities,
            indices,
            dAdz,
            self._boundary_conditions,
            self.ode_options,
            self.progress_bar,
        )

        if np.any(fluence):
            self.disp_message(
                f"Elapsed time for _heat_diffusion_ with {len(fluence):d} "
                f"excitation(s): {time() - t1:f} s"
            )
        else:
            self.disp_message(f"Elapsed time for _heat_diffusion_: {time() - t1:f} s")

        return temp_map

    # ---- energy -----------------------------------------------------------------

    def get_energy_map(self, temp_map, init_temp=None):
        """get_energy_map

        Returns the energy map for a given temperature map and initial
        temperature, see :meth:`calc_energy_map`.

        Args:
            temp_map (ndarray[float], Result): spatio-temporal temperature map
                or result of :meth:`get_temp_map`.
            init_temp (float, Quantity, ndarray[float, Quantity], optional):
                initial temperature scalar or array (K), which is taken from a
                given result of :meth:`get_temp_map` by default.

        Returns:
            result (Result): with the spatio-temporal ``energy_map`` (J) of
            the dimensions ``delays``, ``distances``, and ``subsystems``, see
            :class:`~udkm1Dsim.simulations.result.Result`.

        """
        source = find_source(temp_map)
        if init_temp is None:
            if source is None:
                raise ValueError("init_temp must be given for a temp_map array!")
            init_temp = source["init_temp"]
        temp_map = get_entry(temp_map, "temp_map")
        energy_map = self.calc_energy_map(temp_map, init_temp)
        init_temp = init_temp if isinstance(init_temp, Q_) else Q_(init_temp, "K")
        result = self.new_map_result(
            "energy map",
            source=source,
            inputs={"init_temp": init_temp},
            input_maps={"temp_map": temp_map},
        )
        result.add("energy_map", energy_map, ("delays", "distances", "subsystems"), "J")
        return result

    def get_energy_flux_map(self, temp_map, delta_temp_map=None, delays=None):
        """get_energy_flux_map

        Returns the energy flux maps for a given temperature map, see
        :meth:`calc_energy_flux_map`.

        Args:
            temp_map (ndarray[float], Result): spatio-temporal temperature map
                or result of :meth:`get_temp_map`.
            delta_temp_map (ndarray[float], optional): spatio-temporal
                differential temperature map, which is taken from a given
                result by default.
            delays (ndarray[Quantity], optional): delays range of simulation
                (s), which are taken from a given result by default.

        Returns:
            result (Result): with the spatio-temporal total energy flux
            ``total_energy_flux_map`` into the sub-systems, the one due to
            their coupling ``coupling_energy_flux_map``, and the one due to
            diffusion ``diffusion_energy_flux_map`` (W) of the dimensions
            ``delays``, ``distances``, and ``subsystems``, see
            :class:`~udkm1Dsim.simulations.result.Result`.

        """
        source = find_source(temp_map)
        if source is not None:
            delta_temp_map = source["delta_temp_map"] if delta_temp_map is None else delta_temp_map
            delays = source.quantity("delays") if delays is None else delays
        if delta_temp_map is None or delays is None:
            raise ValueError("delta_temp_map and delays must be given for a temp_map array!")
        temp_map = get_entry(temp_map, "temp_map")
        flux_map = self.calc_energy_flux_map(temp_map, delta_temp_map, delays)
        result = self.new_map_result(
            "energy flux map",
            delays,
            source=source,
            input_maps={"temp_map": temp_map, "delta_temp_map": delta_temp_map},
        )
        dims = ("delays", "distances", "subsystems")
        for i, name in enumerate(("total", "coupling", "diffusion")):
            result.add(f"{name}_energy_flux_map", flux_map[..., i], dims, "W")
        return result

    def calc_energy_map(self, temp_map, init_temp):
        r"""calc_energy_map

        Calculates a energy profile for a given temperature map and inital temperature.

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature map.
            init_temp (float, Quantity, ndarray[float, Quantity]): initial
                temperature scalar or array (K).

        Returns:
            energy_map (ndarray[float]): spatio-temporal energy map.

        """
        t1 = time()

        temp_map = np.atleast_3d(temp_map)
        (M, N, K) = temp_map.shape
        init_temp = self.check_initial_temperature(init_temp)

        energy_map = np.zeros_like(temp_map)

        int_heat_capacities = self.S.get_layer_property_vector("thermal.heat_capacity.integral")
        masses = self.S.get_layer_property_vector("structural.mass.magnitude")

        for k in range(K):
            for i in range(M):
                for j in range(N):
                    energy_map[i, j, k] = masses[j] * (
                        int_heat_capacities[j][k](temp_map[i, j, k])
                        - int_heat_capacities[j][k](init_temp[j, k])
                    )

        self.disp_message(f"Elapsed time for _energy_map_: {time() - t1:f} s")

        return energy_map

    def calc_energy_flux_map(self, temp_map, delta_temp_map, delays):
        r"""calc_energy_flux_map

        Calculates a energy flux profile for a given temperature map and inital
        temperature. For the flux calculation also the time-intervalls are required
        given by the according delays of the heat diffusion simulations.

        The fourth dimension of the ``energy_flux_map`` contains

        index 0 - total energy flux into the respective subsystems
        index 1 - energy flux due to ``sub_system_coupling_str``
        index 2 - energy flux due to diffusion into each subsystems

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature map.
            delta_temp_map (ndarray[float]): spatio-temporal differential
              temperature map.
            delays (ndarray[Quantity]): delays range of simulation (s).

        Returns:
            energy_flux_map (ndarray[float]): spatio-temporal energy flux map.

        """
        t1 = time()

        try:
            delays = delays.to("s").magnitude
        except AttributeError:
            pass

        delay_intervals = np.diff(delays, prepend=2 * delays[0] - delays[1])

        temp_map = np.atleast_3d(temp_map)
        delta_temp_map = np.atleast_3d(delta_temp_map)
        (M, N, K) = temp_map.shape

        energy_flux_map = np.zeros((M, N, K, 3))

        heat_capacities = self.S.get_layer_property_vector("thermal.heat_capacity.functional")
        sub_system_couplings = self.S.get_layer_property_vector(
            "thermal.sub_system_coupling.functional"
        )
        masses = self.S.get_layer_property_vector("structural.mass.magnitude")
        volumes = self.S.get_layer_property_vector("structural.volume.magnitude")

        for k in range(K):
            for i in range(M):
                for j in range(N):
                    energy_flux_map[i, j, k, 0] = (
                        masses[j]
                        * heat_capacities[j][k](temp_map[i, j, k])
                        * delta_temp_map[i, j, k]
                        / delay_intervals[i]
                    )
                    if K > 1:
                        energy_flux_map[i, j, k, 1] = volumes[j] * sub_system_couplings[j][k](
                            temp_map[i, j, :]
                        )
                        energy_flux_map[i, j, k, 2] = (
                            energy_flux_map[i, j, k, 0] - energy_flux_map[i, j, k, 1]
                        )

        self.disp_message(f"Elapsed time for _energy_flux_map_: {time() - t1:f} s")

        return energy_flux_map

    # ---- properties -------------------------------------------------------------

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if backend not in ["scipy", "numba"]:
            raise ValueError("Backend must be either 'scipy' or 'numba'!")

        if backend == "numba" and not HAS_NUMBA:
            warnings.warn("Backend 'numba' is not installed. Fallback to default 'scipy'.")
            backend = "scipy"

        self._backend = backend

    @property
    def excitation(self):
        # Convert to from default SI units to real quantities, where the
        # per-pulse parameters are given once, if they are the same for all
        # pulses
        values = {key: self._pulse_values(key) for key in _PULSE_KEYS}
        excitation = {
            "fluence": Q_(self._excitation["fluence"], u.J / u.m**2).to("mJ/cm**2"),
            "delay_pump": Q_(self._excitation["delay_pump"], u.s).to("ps"),
            "pulse_width": Q_(self._excitation["pulse_width"], u.s).to("ps"),
            "wavelength": Q_(values["wavelength"], u.m).to("nm"),
            "theta": Q_(values["theta"], u.rad).to("deg"),
            "azimuth": Q_(values["azimuth"], u.rad).to("deg"),
            "polarization": values["polarization"],
            "absorption": values["absorption"],
            "backside": values["backside"],
            "bandwidth": self._bandwidth_quantity(),
            "divergence": Q_(values["divergence"], u.rad).to("deg"),
            **{key: values[key] for key in _SAMPLING_KEYS},
        }

        return excitation

    def _bandwidth_quantity(self):
        """``bandwidth`` of all pulses in nm or eV by its domain, as a list
        for different domains of the pulses"""
        widths = [
            Q_(pulse["bandwidth"], u.m).to("nm")
            if pulse["bandwidth_domain"] == "wavelength"
            else Q_(pulse["bandwidth"], u.eV)
            for pulse in (self._pulse(i) for i in range(self.num_pulses))
        ]
        if all(width == widths[0] for width in widths):
            return widths[0]
        if all(width.units == widths[0].units for width in widths):
            return Q_(np.array([width.magnitude for width in widths]), widths[0].units)
        return widths

    @excitation.setter
    def excitation(self, excitation):
        # the excitation is kept as it was for an invalid excitation
        previous = {
            key: value.copy() if isinstance(value, (np.ndarray, list)) else value
            for key, value in self._excitation.items()
        }
        try:
            self._set_excitation(excitation)
        except Exception:
            self._excitation = previous
            raise

    @property
    def num_pulses(self):
        """int: number of pulses of the excitation, at least 1."""
        return max(len(self._excitation["fluence"]), 1)

    @property
    def boundary_conditions(self):
        # Converted from default SI units to real quantities.
        boundary_conditions = {
            "top_type": self.boundary_types[self._boundary_conditions["top_type"]],
        }

        if self._boundary_conditions["top_type"] == 1:
            boundary_conditions["top_value"] = Q_(self._boundary_conditions["top_value"], "K")
        elif self._boundary_conditions["top_type"] == 2:
            boundary_conditions["top_value"] = Q_(self._boundary_conditions["top_value"], "W/m**2")

        boundary_conditions["bottom_type"] = self.boundary_types[
            self._boundary_conditions["bottom_type"]
        ]

        if self._boundary_conditions["bottom_type"] == 1:
            boundary_conditions["bottom_value"] = Q_(
                self._boundary_conditions["bottom_value"], "K"
            )
        elif self._boundary_conditions["bottom_type"] == 2:
            boundary_conditions["bottom_value"] = Q_(
                self._boundary_conditions["bottom_value"], "W/m**2"
            )
        return boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        if isinstance(boundary_conditions, dict):
            if "top_type" in boundary_conditions:
                try:
                    btype = self.boundary_types.index(boundary_conditions["top_type"])
                except ValueError:
                    raise ValueError(
                        "boundary_type must be either _isolator_, _temperature_ or _flux_!"
                    )

                self._boundary_conditions["top_type"] = btype
            if "bottom_type" in boundary_conditions:
                try:
                    btype = self.boundary_types.index(boundary_conditions["bottom_type"])
                except ValueError:
                    raise ValueError(
                        "boundary_type must be either _isolator_, _temperature_ or _flux_!"
                    )

                self._boundary_conditions["bottom_type"] = btype
            if "top_value" in boundary_conditions:
                if self._boundary_conditions["top_type"] == 1:
                    self._boundary_conditions["top_value"] = np.array(
                        boundary_conditions["top_value"].to("K").magnitude
                    )
                elif self._boundary_conditions["top_type"] == 2:
                    self._boundary_conditions["top_value"] = np.array(
                        boundary_conditions["top_value"].to("W/m**2").magnitude
                    )
            if "bottom_value" in boundary_conditions:
                if self._boundary_conditions["bottom_type"] == 1:
                    self._boundary_conditions["bottom_value"] = np.array(
                        boundary_conditions["bottom_value"].to("K").magnitude
                    )
                elif self._boundary_conditions["bottom_type"] == 2:
                    self._boundary_conditions["bottom_value"] = np.array(
                        boundary_conditions["bottom_value"].to("W/m**2").magnitude
                    )
        else:
            raise ValueError("_boundary_conditions_ must be a dict!")

        K = self.S.num_sub_systems
        if (self._boundary_conditions["top_type"] > 0) and (
            np.size(self._boundary_conditions["top_value"]) != K
        ):
            raise ValueError(
                "Non-isolating top boundary conditions must have the "
                "same dimensionality as the number of sub-systems K!"
            )
        if (self._boundary_conditions["bottom_type"] > 0) and (
            np.size(self._boundary_conditions["bottom_value"]) != K
        ):
            raise ValueError(
                "Non-isolating bottom boundary conditions must have the "
                "same dimensionality as the number of sub-systems K!"
            )

    @property
    def distances(self):
        return Q_(self._distances, u.meter).to("nm")

    @distances.setter
    def distances(self, distances):
        self._distances = distances.to_base_units().magnitude

    # ---- excitation -------------------------------------------------------------

    def _set_excitation(self, excitation):
        """sets and checks the excitation, see :attr:`excitation`"""
        # check the size of excitation, if we have a multipulse excitation
        if isinstance(excitation, Q_):
            # just a fluence is given
            self._excitation["fluence"] = np.atleast_1d(
                np.asarray(excitation.to("J/m**2").magnitude, dtype=float)
            )
        elif isinstance(excitation, dict):
            if "multilayer_absorption" in excitation:
                raise TypeError(
                    "The excitation key multilayer_absorption was replaced by absorption, "
                    "which is lambert_beer or a multilayer formalism, e.g. Parratt by default!"
                )
            # a scalar is a single pulse
            for key, unit in [("fluence", "J/m**2"), ("delay_pump", "s"), ("pulse_width", "s")]:
                if key in excitation:
                    self._excitation[key] = np.atleast_1d(
                        np.asarray(excitation[key].to(unit).magnitude, dtype=float)
                    )
            # the per-pulse parameters are scalars for all pulses or given per pulse
            absorption = self._excitation["absorption"]
            if "absorption" in excitation:
                absorption = self._check_per_pulse(
                    excitation["absorption"], self._check_absorption
                )
            given = [key for key in _INSTANCE_KEYS if key in excitation]
            if isinstance(absorption, MultilayerScattering) and given:
                raise ValueError(
                    f"The {', '.join(given)} of the excitation must not be given for a "
                    "configured instance as absorption, which defines them!"
                )
            for key in ("wavelength", "theta", "azimuth"):
                if key in excitation:
                    value = (
                        np.asarray(excitation[key].to("m").magnitude, dtype=float)
                        if key == "wavelength"
                        else to_angle(excitation[key], name=f"excitation {key}")
                    )
                    self._excitation[key] = float(value) if value.ndim == 0 else value.ravel()
            if "polarization" in excitation:
                self._excitation["polarization"] = self._check_per_pulse(
                    excitation["polarization"], self._check_polarization
                )
            if "backside" in excitation:
                backside = excitation["backside"]
                self._excitation["backside"] = (
                    np.asarray(backside, dtype=bool)
                    if self._is_per_pulse(backside)
                    else bool(backside)
                )
            if "bandwidth" in excitation:
                self._set_bandwidth(excitation["bandwidth"])
            if "divergence" in excitation:
                divergence = excitation["divergence"]
                divergence = (
                    np.zeros(())
                    if divergence is None
                    else to_angle(divergence, name="excitation divergence")
                )
                if np.any(divergence < 0):
                    raise ValueError("The divergence of the excitation must not be negative!")
                self._excitation["divergence"] = (
                    float(divergence) if divergence.ndim == 0 else divergence.ravel()
                )
            for key in _SAMPLING_KEYS:
                if key in excitation:
                    # checked by profile_nodes() for every pulse below
                    self._excitation[key] = self._check_per_pulse(excitation[key], lambda v: v)
            self._excitation["absorption"] = absorption
            self._set_instance_parameters(given)
        else:
            raise ValueError("_excitation_ must be either a float/int or dict!")

        if not (
            len(self._excitation["fluence"])
            == len(self._excitation["delay_pump"])
            == len(self._excitation["pulse_width"])
        ):
            raise ValueError(
                "Elements of excitation dict, fluence, delay_pump, "
                "and pulse_width must have "
                "the same number of elements!"
            )
        num_pulses = len(self._excitation["delay_pump"])
        for key in _PULSE_KEYS:
            value = self._excitation[key]
            if self._is_per_pulse(value) and len(value) != num_pulses:
                raise ValueError(
                    f"The {key} of the excitation must be a single value for all pulses or "
                    f"one value for each of the {num_pulses} pulse(s)!"
                )

        # check the elements of the delay_pump vector
        if len(self._excitation["delay_pump"]) != len(np.unique(self._excitation["delay_pump"])):
            raise ValueError("The excitations have to be unique in delays!")
        # sort the pulses and their per-pulse parameters by their delays
        order = np.argsort(self._excitation["delay_pump"])
        for key in ("fluence", "delay_pump", "pulse_width"):
            self._excitation[key] = np.asarray(self._excitation[key], dtype=float)[order]
        for key in _PULSE_KEYS:
            value = self._excitation[key]
            if isinstance(value, np.ndarray):
                self._excitation[key] = value[order]
            elif self._is_per_pulse(value):
                self._excitation[key] = [value[i] for i in order]
        # the sampling and nodes of the bandwidth and divergence are valid for
        # every pulse, while too few nodes of an active spread are warned for
        for index in range(self.num_pulses):
            pulse = self._pulse(index)
            for spread in _SPREADS:
                self._spread_nodes(index, spread)
                if pulse[spread] > 0:
                    _instrument.check_nodes(
                        spread,
                        f"{spread} of the pulse {index}",
                        pulse[f"{spread}_profile"],
                        pulse[f"{spread}_nodes"],
                        pulse[f"{spread}_points"],
                        pulse[f"{spread}_range"],
                        stacklevel=3,
                    )
                    if pulse[f"{spread}_nodes"] == "uniform":
                        _instrument.check_range(
                            spread,
                            f"{spread} of the pulse {index}",
                            pulse[f"{spread}_profile"],
                            pulse[f"{spread}_range"],
                            pulse[f"{spread}_eta"],
                            stacklevel=3,
                        )
            self._pulse_nodes(index)

    def _set_bandwidth(self, bandwidth):
        """sets the ``bandwidth`` of the excitation in m or eV and its
        ``bandwidth_domain`` by its unit for all or each pulse"""

        def check(value):
            if value is None or (not isinstance(value, Q_) and value == 0):
                return 0.0, "wavelength"
            if not isinstance(value, Q_):
                raise TypeError(
                    "The bandwidth of the excitation must be a Quantity in a length or energy "
                    f"unit, e.g. 10 * u.nm or 0.1 * u.eV, not {value!r}!"
                )
            if value.check("[length]"):
                width, domain = float(value.to("m").magnitude), "wavelength"
            elif value.check("[energy]"):
                width, domain = float(value.to("eV").magnitude), "energy"
            else:
                raise ValueError(
                    "The bandwidth of the excitation must be given in a length or energy unit, "
                    f"not in {value.units}!"
                )
            if not (np.isfinite(width) and width >= 0):
                raise ValueError(
                    f"The bandwidth of the excitation must not be negative, not {value}!"
                )
            return width, domain

        if isinstance(bandwidth, Q_) and np.ndim(bandwidth.magnitude) > 0:
            bandwidth = list(bandwidth.flatten())
        if self._is_per_pulse(bandwidth):
            widths, domains = zip(*(check(value) for value in bandwidth))
            self._excitation["bandwidth"] = np.asarray(widths, dtype=float)
            self._excitation["bandwidth_domain"] = list(domains)
        else:
            self._excitation["bandwidth"], self._excitation["bandwidth_domain"] = check(bandwidth)

    def _spread_active(self, spread):
        """the ``bandwidth`` or ``divergence`` is not 0 for at least one pulse"""
        return bool(np.any(np.asarray(self._excitation[spread], dtype=float) > 0))

    def _spread_nodes(self, index, spread):
        """nodes in units of the FWHM and weights of the ``bandwidth`` or
        ``divergence`` of a pulse by :func:`.instrument.profile_nodes`, which
        also checks their sampling"""
        pulse = self._pulse(index)
        try:
            return _instrument.profile_nodes(
                *(pulse[f"{spread}_{key}"] for key in _SAMPLING_DEFAULTS)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid {spread}_profile, {spread}_nodes, {spread}_points, {spread}_range, "
                f"or {spread}_eta of the pulse {index}: {error}"
            ) from None

    def _pulse_nodes(self, index):
        """wavelengths (m) and angles (rad) of a pulse with the weights of the
        nodes of its ``bandwidth`` and ``divergence``"""
        pulse = self._pulse(index)

        def nodes(spread):
            if pulse[spread] == 0:
                return np.zeros(1), np.ones(1)
            return self._spread_nodes(index, spread)

        x, w_e = nodes("bandwidth")
        if pulse["bandwidth_domain"] == "energy":
            wavelength = _HC / (_HC / pulse["wavelength"] + x * pulse["bandwidth"])
        else:
            wavelength = pulse["wavelength"] + x * pulse["bandwidth"]
        if not np.all(np.isfinite(wavelength) & (wavelength > 0)):
            raise ValueError(
                f"The bandwidth of the pulse {index} yields photon energies or wavelengths of "
                "the nodes below 0, which needs a smaller bandwidth or bandwidth_range!"
            )
        x, w_t = nodes("divergence")
        theta = pulse["theta"] + x * pulse["divergence"]
        if not np.all((theta > 0) & (theta < np.pi)):
            raise ValueError(
                f"The divergence of the pulse {index} yields angles of incidence of the nodes "
                "outside of (0, 180) deg, which needs a smaller divergence or "
                "divergence_range!"
            )
        return wavelength, w_e, theta, w_t

    def _spread_note(self, pulse, spread):
        """note on the FWHM and the nodes of the bandwidth or divergence of a
        pulse for the messages"""
        width = pulse[spread]
        if width == 0:
            return ""
        if spread == "divergence":
            width = f"{np.degrees(width):.4g} deg"
        elif pulse["bandwidth_domain"] == "wavelength":
            width = f"{width * 1e9:.4g} nm"
        else:
            width = f"{width:.4g} eV"
        return f" (FWHM {width}, {self._sampling_text(pulse, spread)})"

    @staticmethod
    def _sampling_text(pulse, spread):
        """profile and nodes of the bandwidth or divergence of a pulse"""
        nodes = pulse[f"{spread}_nodes"].replace("_", " ")
        return f"{pulse[f'{spread}_profile']}, {pulse[f'{spread}_points']} {nodes} nodes"

    def _spread_text(self, spread):
        """``bandwidth`` or ``divergence`` with its sampling, if it is the
        same for all pulses with this spread, for the properties of the
        simulation"""
        value = self.excitation[spread]
        if isinstance(value, list):
            text = ", ".join(f"{item:.4g~P}" for item in value)
        else:
            text = f"{value:.4g~P}"
        pulses = [self._pulse(i) for i in range(self.num_pulses)]
        samplings = {self._sampling_text(pulse, spread) for pulse in pulses if pulse[spread] > 0}
        if len(samplings) == 1:
            text += f" ({samplings.pop()})"
        return text

    def _check_per_pulse(self, value, check):
        """a single checked value for all pulses or a list of checked values
        for each pulse"""
        if self._is_per_pulse(value):
            return [check(item) for item in value]
        return check(value)

    @staticmethod
    def _check_polarization(polarization):
        """name of the polarization of the multilayer formalisms, e.g. pi for p"""
        state = MultilayerScattering._polarization_ids.get(polarization)
        if polarization == "none" or state not in range(5):
            raise ValueError(
                f"The polarization of the excitation must be one of "
                f"{', '.join(_POLARIZATIONS)}, or c+, c-, s, and p for short, "
                f"but not {polarization!r}!"
            )
        return MultilayerScattering.polarizations[state]

    def _check_absorption(self, absorption):
        """checked ``absorption`` of a pulse"""
        if isinstance(absorption, str):
            if absorption != "lambert_beer":
                raise ValueError(
                    f"The absorption of the excitation must be lambert_beer or a multilayer "
                    f"formalism, but not {absorption!r}!"
                )
        elif isinstance(absorption, MultilayerScattering):
            if absorption.S is not self.S:
                raise ValueError(
                    "The absorption formalism of the excitation must be created for the "
                    "structure of the heat simulation!"
                )
            if (
                np.size(absorption.energy.magnitude) != 1
                or np.size(absorption.theta.magnitude) != 1
                or np.atleast_2d(absorption.pol_in).shape[0] != 1
            ):
                raise ValueError(
                    "The absorption formalism of the excitation must have a single energy, "
                    "angle, and incoming polarization!"
                )
        elif not (isinstance(absorption, type) and issubclass(absorption, MultilayerScattering)):
            raise TypeError(
                "The absorption of the excitation must be lambert_beer or a multilayer "
                "formalism, e.g. Parratt!"
            )
        return absorption

    @staticmethod
    def _instance_parameters(formalism):
        """``wavelength``, ``theta``, ``azimuth``, and ``polarization`` of a
        configured formalism, which must have a single energy, angle, and
        incoming polarization, also after changes of the formalism"""
        if (
            np.size(formalism.energy.magnitude) != 1
            or np.size(formalism.theta.magnitude) != 1
            or np.atleast_2d(formalism.pol_in).shape[0] != 1
        ):
            raise ValueError(
                "The absorption formalism of the excitation must have a single energy, "
                "angle, and incoming polarization!"
            )
        return {
            "wavelength": float(formalism.wl.to("m").magnitude.flat[0]),
            "theta": float(formalism.theta.to("rad").magnitude.flat[0]),
            "azimuth": float(formalism.azimuth.to("rad").magnitude),
            "polarization": formalism.polarizations[formalism.pol_in_state],
        }

    def _set_instance_parameters(self, given):
        """sets the ``wavelength``, ``theta``, ``azimuth``, and
        ``polarization`` of the pulses with a configured formalism as
        ``absorption`` by it, where the ``given`` per-pulse values of these
        pulses must agree with the formalism"""
        absorption = self._excitation["absorption"]
        if isinstance(absorption, MultilayerScattering):
            self._excitation.update(self._instance_parameters(absorption))
            return
        if not self._is_per_pulse(absorption):
            return
        for index, formalism in enumerate(absorption):
            if not isinstance(formalism, MultilayerScattering):
                continue
            for key, value in self._instance_parameters(formalism).items():
                stored = self._excitation[key]
                if self._is_per_pulse(stored):
                    if key in given and not (
                        stored[index] == value
                        if key == "polarization"
                        else np.isclose(stored[index], value)
                    ):
                        raise ValueError(
                            f"The {key} of the pulse {index} is defined by its configured "
                            "instance as absorption and must not be given differently!"
                        )
                else:
                    # one value per pulse
                    stored = [stored] * len(absorption)
                    stored = stored if key == "polarization" else np.asarray(stored, dtype=float)
                stored[index] = value
                self._excitation[key] = stored

    @staticmethod
    def _is_per_pulse(value):
        """the value is given per pulse"""
        return isinstance(value, (list, tuple)) or (
            isinstance(value, np.ndarray) and value.ndim > 0
        )

    def _check_pulse_index(self, pulse_index):
        """the checked non-negative index of a pulse"""
        index = operator.index(pulse_index)
        if not -self.num_pulses <= index < self.num_pulses:
            raise IndexError(
                f"The pulse index {pulse_index} is out of range for {self.num_pulses} pulse(s)!"
            )
        return index % self.num_pulses

    def _pulse_indices(self, delay_pump):
        """indices of the pulses of the excitation by their delays"""
        delays = np.asarray(self._excitation["delay_pump"], dtype=float)
        indices = []
        for delay in np.atleast_1d(delay_pump):
            match = np.flatnonzero(np.isclose(delays, delay, rtol=1e-12, atol=0))
            if match.size == 0:
                raise ValueError(f"There is no pulse of the excitation at the delay {delay} s!")
            indices.append(int(match[0]))
        return indices

    def _pulse(self, index):
        """the per-pulse excitation parameters of a pulse, where a configured
        instance as ``absorption`` gives its current ``wavelength``,
        ``theta``, ``azimuth``, and ``polarization``, also after changes of
        the instance"""
        pulse = {}
        for key in _PULSE_KEYS:
            value = self._excitation[key]
            pulse[key] = value[index] if self._is_per_pulse(value) else value
        pulse["backside"] = bool(pulse["backside"])
        if isinstance(pulse["absorption"], MultilayerScattering):
            pulse.update(self._instance_parameters(pulse["absorption"]))
        return pulse

    def _pulse_key(self, index):
        """hashable key of the parameters of a pulse, which define its
        absorption profile"""
        pulse = self._pulse(index)
        absorption = pulse["absorption"]
        if isinstance(absorption, MultilayerScattering):
            # a configured instance by its identity
            absorption = id(absorption)
        pulse["absorption"] = absorption
        for spread in _SPREADS:
            if pulse[spread] == 0:
                # the sampling of a monochromatic or collimated pulse does not
                # change its absorption
                for key in _SAMPLING_DEFAULTS:
                    pulse[f"{spread}_{key}"] = None
                if spread == "bandwidth":
                    pulse["bandwidth_domain"] = None
        return tuple(pulse[key] for key in _PULSE_KEYS)

    def _pulse_values(self, key):
        """a per-pulse parameter of all pulses, which is a single value, if
        it is the same for all pulses"""
        values = [self._pulse(i)[key] for i in range(self.num_pulses)]
        if all(value is values[0] or value == values[0] for value in values):
            return values[0]
        if key in ("wavelength", "theta", "azimuth", "bandwidth", "divergence"):
            return np.asarray(values, dtype=float)
        return values

    def _absorption_name(self):
        """name of the absorption of the excitation, i.e. lambert_beer or the
        name of the multilayer formalism, or a list of them for each pulse"""

        def name(absorption):
            if isinstance(absorption, str):
                return absorption
            if isinstance(absorption, MultilayerScattering):
                return type(absorption).__name__
            return absorption.__name__

        names = [name(self._pulse(i)["absorption"]) for i in range(self.num_pulses)]
        return names[0] if len(set(names)) == 1 else names
