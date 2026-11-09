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

"""X-ray and light scattering simulations.

The formalisms are named by their method:

- :class:`XrayKin` and :class:`XrayDyn` - kinematical and dynamical X-ray
  diffraction of the atoms in unit cells,
- :class:`Parratt` - scalar recursion of isotropic layers for :math:`s` and
  :math:`p` polarization,
- :class:`Elzo` - analytical magnetic eigenmodes with transfer matrices,
- :class:`Stepanov` - the same eigenmodes with the stable scattering-matrix
  recursion,
- :class:`BerremanElzo` and :class:`BerremanStepanov` - exact eigenmodes of an
  arbitrary dielectric tensor with transfer matrices or the scattering-matrix
  recursion.

All formalisms except :class:`XrayKin` and :class:`XrayDyn` take the dielectric
tensor of the layers, see :class:`.Dielectric`.

The photon energies and angles of a simulation are held by its
:class:`~udkm1Dsim.simulations.parameter_groups.ExperimentParameters` in
``sim.experiment``, with ``sim.energy``, ``sim.wl``, ``sim.k``, ``sim.theta``,
and ``sim.qz`` as shortcuts. The polarizations are set by
:meth:`~Scattering.set_polarization` with their ids or names in
``polarizations``, e.g. ``sim.set_polarization("sigma", "none")``, and states
which are not supported by a formalism raise a ``ValueError``.

The code ported from `Dyna <https://gitlab.com/dynadevgroup/Dyna>`_ by
Stéphane Grenier is included under the MIT license with the author's written
permission.

"""

from .berreman import BerremanElzo, BerremanStepanov
from .dynamical import XrayDyn
from .elzo import Elzo
from .kinematical import XrayKin
from .multilayer import MultilayerScattering
from .parratt import Parratt
from .scattering import Scattering
from .stepanov import Stepanov

__all__ = [
    "Scattering",
    "XrayKin",
    "XrayDyn",
    "MultilayerScattering",
    "Parratt",
    "Elzo",
    "Stepanov",
    "BerremanElzo",
    "BerremanStepanov",
]
