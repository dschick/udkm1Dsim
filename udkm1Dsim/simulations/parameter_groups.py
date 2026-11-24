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

"""Parameter groups of the simulations."""

__all__ = ["ExperimentParameters", "InstrumentParameters"]

__docformat__ = "restructuredtext"

import warnings

import numpy as np
import pint
import scipy.constants as constants
from tabulate import tabulate

from ..helpers import to_angle
from ..structures.parameter_groups import ParameterGroup
from ..structures.parameters import Parameter
from .scattering import instrument as _instrument

u = pint.get_application_registry()


class ExperimentParameters(ParameterGroup):
    r"""Photon energies and angles of a scattering experiment.

    The photon energy :math:`E` is the independent spectral parameter, from
    which the wavelength and the wavenumber follow, and setting one of them
    sets the photon energy. Either the scattering vector :math:`q_z` or the
    incidence angle :math:`\vartheta` is independent, namely the one which
    was set last, and is kept for changes of the photon energy, while the
    other one follows from

    .. math::

        \lambda & = \frac{hc}{E} \\
        k & = \frac{2\pi}{\lambda} \\
        \vartheta & = \arcsin{\frac{\lambda q_z}{4\pi}} \\
        q_z & = 2k \sin{\vartheta}

    The incidence angle and the scattering vector have the shape (M, N) for
    M photon energies, where one-dimensional inputs are used for all of them.

    Attributes:
        energy (ndarray[float]): photon energies :math:`E` (eV).
        wl (ndarray[float]): wavelengths :math:`\lambda` (m).
        k (ndarray[float]): wavenumbers :math:`k` (1/m).
        theta (ndarray[float]): incidence angles :math:`\vartheta` (rad).
        qz (ndarray[float]): scattering vectors :math:`q_z` (1/m).

    """

    _derived = ("wl", "k", "theta")

    def __init__(self):
        self._scan = "qz"
        self.energy = Parameter("eV", np.array([]), symbol="E")
        self.wl = Parameter("m", np.array([]), symbol=r"\lambda")
        self.k = Parameter("1/m", np.array([]), symbol="k")
        self.theta = Parameter("rad", np.zeros([1, 1]), symbol=r"\theta")
        self.qz = Parameter("1/m", np.zeros([1, 1]), symbol="q_z")

        self._connect_parameters()

    def _array(self, name, value):
        """Value of a parameter as array of at least one dimension in the
        unit of the parameter."""
        if isinstance(value, Parameter):
            value = value.quantity
        if isinstance(value, u.Quantity):
            value = value.to(self.__dict__[name].unit).magnitude
        return np.array(value, dtype=float, ndmin=1)

    def __setattr__(self, name, value):
        """The wavelength and the wavenumber set the photon energy, and the
        incidence angle and the scattering vector become the independent one
        of both on assignment."""
        if name not in self.__dict__ or not isinstance(self.__dict__[name], Parameter):
            super().__setattr__(name, value)
        elif name == "energy":
            super().__setattr__(name, self._array(name, value))
        elif name == "wl":
            self.energy = self._hc / self._array(name, value)
        elif name == "k":
            self.energy = self._hc * self._array(name, value) / (2 * np.pi)
        elif name in ("theta", "qz"):
            value = self._array(name, value)
            if value.ndim < 2:
                if np.size(self.energy.magnitude) == 0:
                    raise IndexError("Set energy, wl, or k first!")
                value = np.tile(value, (np.size(self.energy.magnitude), 1))
            self._switch_scan(name)
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    # Planck constant times the speed of light (eV m)
    _hc = constants.h * constants.c / constants.e

    def _switch_scan(self, name):
        """Make the given one of the incidence angle and the scattering
        vector the independent one."""
        other = "theta" if name == "qz" else "qz"
        self._scan = name
        self._derived = ("wl", "k", other)
        self.__dict__[other]._caller = None
        self.__dict__[name]._caller = self

    def _update_group(self):
        M = np.size(self.energy.magnitude)
        if M == 0:
            return
        self.wl.magnitude = self._hc / self.energy.magnitude
        self.k.magnitude = 2 * np.pi / self.wl.magnitude
        scan = self.__dict__[self._scan]
        if scan.magnitude.shape[0] != M:
            scan.magnitude = np.tile(scan.magnitude[0, :], (M, 1))
        if self._scan == "qz":
            # a scattering vector beyond 2k has no incidence angle, which is
            # warned for by the simulations only on a calculation, as it may
            # be set right after the photon energy
            with np.errstate(invalid="ignore"):
                self.theta.magnitude = np.arcsin(
                    self.wl.magnitude[:, np.newaxis] * self.qz.magnitude / (4 * np.pi)
                )
        else:
            self.qz.magnitude = 2 * self.k.magnitude[:, np.newaxis] * np.sin(self.theta.magnitude)


class InstrumentParameters(ParameterGroup):
    r"""Instrument function of a scattering experiment.

    The intensities of the scattering simulations are averaged over the
    angular and spectral distribution of the instrument, multiplied by the
    footprint of the beam on the sample, and scaled by an intensity and a
    background, see :meth:`.Scattering.apply_instrument`. All widths
    are full widths at half maximum (FWHM), also of the Gaussian beam, and
    the full width of a box. The instrument is applied to the reflectivity
    and transmissivity, as well as to the rotation and ellipticity of the
    multilayer formalisms, but not to the Jones matrices, the fields, and
    the absorption.

    The angular, spectral, and temporal resolution are given by a profile, i.e.
    ``'none'``, ``'box'``, ``'gauss'``, ``'lorentz'``, or ``'pseudo_voigt'``,
    the latter of the mixing :math:`\eta` of a Lorentzian and a Gaussian
    profile of the same FWHM. A resolution needs both, its profile, e.g.
    ``theta_resolution``, and its FWHM, e.g. ``theta_width``: a width with
    the profile ``'none'`` is not applied, and a profile with a width of 0
    has no effect, which are both warned for. Their ``mode`` is either

    * ``'quadrature'``: the integral over the profile is approximated by a
      weighted sum over ``points`` nodes, i.e. the simulation is calculated
      once for every node with the whole grid of the experiment shifted by
      the node, so that every point gets its nodes at the correct distance
      and any grid works,
    * ``'convolution'``: the integral is evaluated with the neighboring
      points of the grid as nodes, i.e. the result of a single calculation
      is convolved on the grid of the experiment, which must be dense,
      e.g. for sharp Bragg peaks, or
    * ``'qz'``: only for the energy, the spectral spread at a fixed angle is
      approximated by the spread of the angle
      :math:`\sin\vartheta' = \sin\vartheta\,(1 + \Delta E / E)` of
      the same scattering vector :math:`q_z` at the photon energy,
      which is valid away from resonances only.

    The ``nodes`` of the quadrature are either ``'uniform'`` with the
    weights of the profile on ``points`` equidistant nodes within
    :math:`\pm` ``range`` times the width, which is robust for narrow
    features, or ``'gaussian_quadrature'``, i.e. Gauss-Hermite nodes of a Gaussian
    and Gauss-Legendre nodes of a box profile, which are more efficient for
    smooth curves. The widths are scalars or arrays along the energies of
    the experiment for the energy and along its angles or of its shape
    (M, N) for the angle, e.g. for a curvature of the sample, whose
    footprint spreads the angle, or a spectral bandwidth of constant
    :math:`\Delta E / E`.

    The ``spectrum`` of the source is a list of components, e.g. the Cu
    :math:`K_{\alpha 1}` and :math:`K_{\alpha 2}` lines or higher
    harmonics of a monochromator, given as dicts of the photon energy
    :math:`E_m = f\,E + \Delta E` with the ``factor`` :math:`f`
    (default 1) and the ``offset`` :math:`\Delta E` (default 0 eV) relative
    to the photon energy :math:`E` of the experiment, as well as the
    ``weight`` (default 1) and the optional ``width``, which defaults to
    :math:`f` times the ``energy_width``, as for a constant
    :math:`\Delta E / E` of a monochromator. The weights are normalized.

    The temporal resolution, e.g. of the pump-probe cross-correlation, is
    given by the ``delay_resolution`` profile of the ``delay_width``, which
    is a scalar or an array along the delays. It applies only to the
    inhomogeneous simulations, whose intensities are convolved along their
    delays, see :func:`~udkm1Dsim.simulations.scattering.instrument.convolve_delays`.
    These need the ``delays`` then, which are given or are taken from the
    upstream result of the maps.

    The ``footprint`` of a ``'box'`` or ``'gauss'`` beam of the
    ``beam_width`` on a sample of the ``sample_length`` scales the
    reflected intensities, while the ``theta_offset`` and ``energy_offset``
    are added to the angles and photon energies of the experiment, and the
    intensities are scaled to :math:`I_0 I + I_\mathrm{bkg}`.

    The nodes of the quadrature and the formulas of the instrument function
    are described in the module
    :mod:`~udkm1Dsim.simulations.scattering.instrument`.

    The angles ``theta_width`` and ``theta_offset`` are pint Quantities
    with an angular unit, e.g. ``0.05 * u.deg``, or lists of them, see
    :func:`~udkm1Dsim.helpers.to_angle`, while bare numbers of the energies
    are eV, of the widths of the delays ps, and of the lengths mm.

    Attributes:
        theta_resolution (str): profile of the angular resolution, i.e.
            ``'none'`` (default) for no angular resolution, ``'box'``,
            ``'gauss'``, ``'lorentz'``, or ``'pseudo_voigt'``.
        theta_width (Quantity): FWHM of the angular resolution (deg), a
            scalar or an array along the angles or of the shape (M, N) of the
            experiment.
        theta_eta (float): Lorentzian fraction of a pseudo-Voigt profile.
        theta_mode (str): ``'quadrature'`` or ``'convolution'``.
        theta_nodes (str): ``'uniform'`` or ``'gaussian_quadrature'`` nodes.
        theta_points (int): number of nodes.
        theta_range (float): truncation of the profile in units of its
            FWHM, i.e. the extent :math:`\pm` range of the uniform nodes and
            of the convolution, not used by ``'gaussian_quadrature'`` nodes and box
            profiles.
        energy_resolution (str): profile of the spectral resolution, i.e.
            ``'none'`` (default) for no spectral resolution, ``'box'``,
            ``'gauss'``, ``'lorentz'``, or ``'pseudo_voigt'``.
        energy_width (ndarray[float]): FWHM of the spectral resolution (eV),
            a scalar or an array along the photon energies.
        energy_eta (float): Lorentzian fraction of a pseudo-Voigt profile.
        energy_mode (str): ``'quadrature'``, ``'convolution'``, or ``'qz'``.
        energy_nodes (str): ``'uniform'`` or ``'gaussian_quadrature'`` nodes.
        energy_points (int): number of nodes.
        energy_range (float): truncation of the profile in units of its
            FWHM, i.e. the extent :math:`\pm` range of the uniform nodes and
            of the convolution, not used by ``'gaussian_quadrature'`` nodes and box
            profiles.
        spectrum (list[dict], None): components of the source spectrum.
        delay_resolution (str): profile of the temporal resolution, i.e.
            ``'none'`` (default) for no temporal resolution, ``'box'``,
            ``'gauss'``, ``'lorentz'``, or ``'pseudo_voigt'``.
        delay_width (ndarray[float]): FWHM of the temporal resolution (ps).
        delay_eta (float): Lorentzian fraction of a pseudo-Voigt profile.
        delay_range (float): truncation of the profile in units of its
            FWHM for the convolution, not used by box profiles.
        footprint (str): ``'none'``, ``'box'``, or ``'gauss'`` beam.
        beam_width (float): FWHM or full width of the beam (mm).
        sample_length (float): length of the sample along the beam (mm).
        theta_offset (Quantity): offset of the angles (deg).
        energy_offset (float): offset of the photon energies (eV).
        I0 (float): scale of the intensities :math:`I_0`.
        background (float): background of the intensities
            :math:`I_\mathrm{bkg}`.

    """

    profiles = ("none", *_instrument.PROFILES)
    footprints = ("none", "box", "gauss")
    _options = {
        "theta_resolution": profiles,
        "theta_mode": ("quadrature", "convolution"),
        "theta_nodes": _instrument.NODES,
        "energy_resolution": profiles,
        "energy_mode": ("quadrature", "convolution", "qz"),
        "energy_nodes": _instrument.NODES,
        "delay_resolution": profiles,
        "footprint": footprints,
    }
    _spectrum_keys = ("offset", "factor", "weight", "width")
    # angles, which must be given as Quantities, see to_angle
    _angles = ("theta_width", "theta_offset")
    # settings of the default instrument, see is_active
    _default_hash_items = None

    def __init__(self):
        self.theta_resolution = "none"
        self.theta_width = Parameter("deg", 0.0, symbol=r"\Delta\theta")
        self.theta_eta = Parameter("", 0.5, symbol=r"\eta_\theta")
        self.theta_mode = "quadrature"
        self.theta_nodes = "uniform"
        self.theta_points = Parameter("", 7, symbol=r"n_\theta")
        self.theta_range = Parameter("", 2.0, symbol=r"r_\theta")
        self.energy_resolution = "none"
        self.energy_width = Parameter("eV", 0.0, symbol=r"\Delta E")
        self.energy_eta = Parameter("", 0.5, symbol=r"\eta_E")
        self.energy_mode = "quadrature"
        self.energy_nodes = "uniform"
        self.energy_points = Parameter("", 7, symbol="n_E")
        self.energy_range = Parameter("", 2.0, symbol="r_E")
        self.spectrum = None
        self.delay_resolution = "none"
        self.delay_width = Parameter("ps", 0.0, symbol=r"\Delta t")
        self.delay_eta = Parameter("", 0.5, symbol=r"\eta_t")
        self.delay_range = Parameter("", 2.0, symbol="r_t")
        self.footprint = "none"
        self.beam_width = Parameter("mm", 0.0, symbol="w")
        self.sample_length = Parameter("mm", 0.0, symbol="L")
        self.theta_offset = Parameter("deg", 0.0, symbol=r"\theta_0")
        self.energy_offset = Parameter("eV", 0.0, symbol="E_0")
        self.I0 = Parameter("", 1.0, symbol="I_0")
        self.background = Parameter("", 0.0, symbol=r"I_\mathrm{bkg}")

        self._connect_parameters()

    def __setattr__(self, name, value):
        """The options and angles are checked and the spectrum is parsed on
        assignment."""
        if name in self._angles:
            if isinstance(value, Parameter):
                to_angle(value.quantity, name=name)
            else:
                value = to_angle(value, "deg", name=name)
                value = float(value) if value.ndim == 0 else value
        elif name in self._options:
            if value not in self._options[name]:
                raise ValueError(
                    f"The {name} must be one of {', '.join(map(repr, self._options[name]))}, "
                    f"but is {value!r}!"
                )
        elif name == "spectrum" and value is not None:
            value = self._parse_spectrum(value)
        super().__setattr__(name, value)

    @classmethod
    def _parse_spectrum(cls, spectrum):
        """components of the spectrum as dicts of floats in eV with the
        defaults"""
        if isinstance(spectrum, dict):
            spectrum = [spectrum]
        if len(spectrum) == 0:
            raise ValueError("The spectrum must have at least one component!")
        components = []
        for component in spectrum:
            if not isinstance(component, dict):
                raise TypeError(
                    f"The components of the spectrum must be dicts, not {component!r}!"
                )
            unknown = set(component) - set(cls._spectrum_keys)
            if unknown:
                raise ValueError(
                    f"Unknown keys {sorted(unknown)} of a component of the spectrum, which has "
                    f"the keys {', '.join(cls._spectrum_keys)}!"
                )
            parsed = {
                "offset": cls._to_float(component.get("offset", 0.0), "eV"),
                "factor": cls._to_float(component.get("factor", 1.0), ""),
                "weight": cls._to_float(component.get("weight", 1.0), ""),
            }
            if component.get("width") is not None:
                parsed["width"] = cls._to_float(component["width"], "eV")
            if parsed["factor"] <= 0 or parsed["weight"] < 0 or parsed.get("width", 0) < 0:
                raise ValueError(
                    "The factor of a component of the spectrum must be positive and its "
                    f"weight and width must not be negative, but it is {component!r}!"
                )
            components.append(parsed)
        if sum(c["weight"] for c in components) <= 0:
            raise ValueError("The weights of the spectrum must not all be zero!")
        return components

    @staticmethod
    def _to_float(value, unit):
        """float of a value, which can be a Quantity, in the unit"""
        if isinstance(value, u.Quantity):
            return float(value.to(unit).magnitude) if unit else float(value.to("").magnitude)
        return float(value)

    @property
    def components(self):
        """list[dict]: components of the spectrum with normalized weights,
        which is the photon energy of the experiment for no spectrum."""
        spectrum = self.spectrum or [{"offset": 0.0, "factor": 1.0, "weight": 1.0}]
        total = sum(c["weight"] for c in spectrum)
        return [dict(c, weight=c["weight"] / total) for c in spectrum if c["weight"] > 0]

    def _check_resolutions(self):
        """Warn for an angular, spectral, or temporal resolution, which is set
        only partially, i.e. a width without a profile, which is not applied,
        or a profile without a width, which repeats the same calculation for
        all its nodes."""
        widths = {
            "theta": np.any(self.theta_width.magnitude > 0),
            "energy": np.any(self.energy_width.magnitude > 0)
            or any(c.get("width", 0) > 0 for c in self.spectrum or []),
            "delay": np.any(self.delay_width.magnitude > 0),
        }
        for axis, width in widths.items():
            resolution = getattr(self, f"{axis}_resolution")
            if width and resolution == "none":
                warnings.warn(
                    f"The {axis}_width is set, but the {axis}_resolution is 'none', so that "
                    f"no {axis} resolution is applied!",
                    stacklevel=4,
                )
            elif not width and resolution != "none":
                warnings.warn(
                    f"The {axis}_resolution is {resolution!r}, but the {axis}_width is 0, so "
                    f"that the {axis} resolution has no effect!",
                    stacklevel=4,
                )
            elif width:
                if axis != "delay":
                    self._check_nodes(axis)
                if axis == "delay" or getattr(self, f"{axis}_nodes") == "uniform":
                    # the uniform nodes and the convolution are truncated
                    _instrument.check_range(
                        axis,
                        f"{axis} resolution",
                        resolution,
                        getattr(self, f"{axis}_range").magnitude,
                        getattr(self, f"{axis}_eta").magnitude,
                        stacklevel=4,
                    )

    def _check_nodes(self, axis):
        """Warn for too few nodes of the quadrature of the ``axis``, see
        :func:`.instrument.check_nodes`."""
        if getattr(self, f"{axis}_mode") == "convolution":
            return
        _instrument.check_nodes(
            axis,
            f"{axis} resolution",
            getattr(self, f"{axis}_resolution"),
            getattr(self, f"{axis}_nodes"),
            getattr(self, f"{axis}_points").magnitude,
            getattr(self, f"{axis}_range").magnitude,
            stacklevel=5,
        )

    @property
    def is_delay_active(self):
        """bool: the temporal resolution convolves the inhomogeneous
        results along their delays."""
        return self.delay_resolution != "none"

    @property
    def is_active(self):
        """bool: the instrument changes the results of the simulations,
        otherwise they are calculated as without an instrument."""
        if InstrumentParameters._default_hash_items is None:
            # the defaults are compared for every calculation, so they are
            # created only once
            InstrumentParameters._default_hash_items = InstrumentParameters()._hash_items()
        return self._hash_items() != InstrumentParameters._default_hash_items

    def to_dict(self):
        """to_dict

        All settings of the instrument, e.g. for the metadata of the results.

        Returns:
            settings (dict): settings with Quantities for values with units.

        """
        settings = {}
        for name, value in vars(self).items():
            if name.startswith("_"):
                continue
            if isinstance(value, Parameter):
                settings[name] = value.quantity if str(value.unit) else value.magnitude
            elif value is not None:
                settings[name] = value
        return settings

    def _hash_items(self):
        """settings of the instrument as plain values for the hash of the
        results"""
        settings = {}
        for name, value in vars(self).items():
            if name.startswith("_"):
                continue
            if isinstance(value, Parameter):
                # the unit of a parameter is fixed, as values are converted to it
                value = np.asarray(value.magnitude, dtype=float).tolist()
            settings[name] = value
        return settings

    def _option_rows(self):
        rows = [[name, getattr(self, name)] for name in self._options]
        rows.append(["spectrum", self._format_spectrum()])
        return rows

    def _format_spectrum(self):
        """components of the spectrum as text, e.g. ``E - 20 eV (1)``"""
        if self.spectrum is None:
            return "none"
        texts = []
        for c in self.spectrum:
            text = "E" if c["factor"] == 1 else f"{c['factor']:g} E"
            if c["offset"] != 0:
                text += f" {'+' if c['offset'] > 0 else '-'} {abs(c['offset']):g} eV"
            text += f" ({c['weight']:g}"
            if "width" in c:
                text += f", {c['width']:g} eV"
            texts.append(text + ")")
        return ", ".join(texts)

    def __repr__(self):
        options = tabulate(
            self._option_rows(), headers=["Option", "Value"], tablefmt="double_grid"
        )
        return super().__repr__() + "\n" + options

    def _repr_html_(self):
        options = tabulate(self._option_rows(), headers=["Option", "Value"], tablefmt="html")
        return super()._repr_html_() + options
