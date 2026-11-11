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

__all__ = ["ExperimentParameters"]

__docformat__ = "restructuredtext"

import numpy as np
import pint
import scipy.constants as constants

from ..structures.parameter_groups import ParameterGroup
from ..structures.parameters import Parameter

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
