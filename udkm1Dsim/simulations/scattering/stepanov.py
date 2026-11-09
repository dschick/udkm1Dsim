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

"""Multilayer formalism with the analytical eigenmodes of
:cite:t:`elzo2012` and the numerically stable scattering-matrix recursion of
:cite:t:`stepanov2000`, as in
`Dyna <https://gitlab.com/dynadevgroup/Dyna>`_."""

__all__ = ["Stepanov"]

__docformat__ = "restructuredtext"

import numpy as np

from .elzo import elzo_modes
from .multilayer import MultilayerScattering


class Stepanov(MultilayerScattering):
    r"""Stepanov

    Multilayer scattering simulations with the analytical eigenmodes of
    :cite:t:`elzo2012`, see
    :func:`~udkm1Dsim.simulations.scattering.elzo.elzo_modes`, which are
    assembled by the scattering-matrix recursion of :cite:t:`stepanov2000`
    instead of transfer matrices. Hence, it gives the same results as
    :class:`~udkm1Dsim.simulations.scattering.elzo.Elzo` for homogeneous
    layers, but stays numerically stable for thick, strongly absorbing, or
    highly repeated stacks, e.g. for visible light. Ported from
    `Dyna <https://gitlab.com/dynadevgroup/Dyna>`_.

    The dielectric tensor of every layer must be isotropic except for its
    gyrotropic part, and a :class:`.UnitCell` is treated as homogeneous layer
    with its dielectric tensor.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.

    """

    _title = "Stepanov multilayer scattering simulation"
    _assembly = "scattering"

    def calc_modes(self, eps):
        """calc_modes

        Calculates the analytical eigenmodes of a homogeneous layer with the
        dielectric tensor ``eps``, see
        :func:`~udkm1Dsim.simulations.scattering.elzo.elzo_modes`.

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

    def get_layer_dielectric_tensor(self, layer, strain=0.0, magnetization=None):
        """get_layer_dielectric_tensor

        Dielectric tensor of a layer at the photon energies of the simulation,
        which must be isotropic except for its gyrotropic part, see
        :meth:`.Layer.get_dielectric_tensor`.

        Args:
            layer (Layer): homogeneous layer.
            strain (float, optional): strain of the layer. Defaults to 0.
            magnetization (ndarray[float], optional): cartesian
                magnetization ``(x, y, z)``. Defaults to the magnetization of
                the layer.

        Returns:
            eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3).

        """
        dielectric = layer.optical.dielectric
        if dielectric.is_set and not dielectric.is_isotropic:
            raise ValueError(
                f"The formalism of {type(self).__name__} requires an isotropic dielectric tensor, "
                f"but the layer '{layer.id}' is anisotropic! Use BerremanStepanov instead."
            )
        return layer.get_dielectric_tensor(self.experiment.energy.magnitude, strain, magnetization)
