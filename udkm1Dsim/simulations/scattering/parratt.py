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

"""Scalar multilayer formalism of :cite:t:`parratt1954` for isotropic layers,
as in `Dyna <https://gitlab.com/dynadevgroup/Dyna>`_."""

__all__ = ["Parratt"]

__docformat__ = "restructuredtext"

import numpy as np

from .berreman import _forward_sqrt
from .multilayer import MultilayerScattering


class Parratt(MultilayerScattering):
    r"""Parratt

    Multilayer scattering simulations of isotropic layers by the recursion of
    :cite:t:`parratt1954` for the uncoupled :math:`\sigma` (:math:`s`) and
    :math:`\pi` (:math:`p`) polarizations from the X-ray to the optical range.
    It is the fastest formalism, but neglects the magnetization and requires
    isotropic dielectric tensors of all layers. The interface roughness
    :math:`\sigma_j` is included by the factor of :cite:t:`nevot1980`

    .. math:: r_j \rightarrow r_j \exp(-2 k_{z,j-1} k_{z,j} \sigma_j^2)

    with the normal wave-vector components
    :math:`k_{z,j} = k \sqrt{\varepsilon_j - \cos^2\theta}` above and below the
    interface, which differs slightly from the roughness matrices of
    :class:`~udkm1Dsim.simulations.scattering.elzo.Elzo`.

    The reflection coefficients of the interfaces are

    .. math::

        r_{s,j} = \frac{k_{z,j-1} - k_{z,j}}{k_{z,j-1} + k_{z,j}}, \quad
        r_{p,j} = \frac{\varepsilon_j k_{z,j-1} - \varepsilon_{j-1} k_{z,j}}
        {\varepsilon_j k_{z,j-1} + \varepsilon_{j-1} k_{z,j}}

    and the transmission is given for the amplitude of the electric field at
    the bottom of the last layer of the substrate, as for the other
    multilayer formalisms. Ported from
    `Dyna <https://gitlab.com/dynadevgroup/Dyna>`_.

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

    _title = "Parratt multilayer scattering simulation"

    def get_layer_dielectric_tensor(self, layer, strain=0.0, magnetization=None):
        """get_layer_dielectric_tensor

        Non-magnetic dielectric tensor of a layer at the photon energies of
        the simulation, which must be isotropic, see
        :meth:`.Layer.get_dielectric_tensor`. The magnetization is neglected.

        Args:
            layer (Layer): homogeneous layer.
            strain (float, optional): strain of the layer. Defaults to 0.
            magnetization (ndarray[float], optional): neglected
                magnetization.

        Returns:
            eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3).

        """
        dielectric = layer.optical.dielectric
        if dielectric.is_set and not dielectric.is_isotropic:
            raise ValueError(
                f"The formalism of {type(self).__name__} requires an isotropic dielectric tensor, "
                f"but the layer '{layer.id}' is anisotropic! Use BerremanStepanov instead."
            )
        return layer.get_dielectric_tensor(
            self.experiment.energy.magnitude, strain, (0.0, 0.0, 0.0)
        )

    def _layer_constants(self, layer, strain, magnetization, cache):
        """dielectric function, normal wave-vector component, thickness, and
        roughness of a layer, where an array of strains of shape (D,), e.g.
        for D delays, gives constants of shape (D, M, N)"""
        if np.ndim(strain) > 0 and np.size(strain) > 1:
            strain = np.asarray(strain, dtype=float)
            eps = self.get_layer_dielectric_tensor(layer, strain)[..., 0, 0][..., np.newaxis]
            thickness = layer.structural.thickness.magnitude * (1 + strain)
            thickness = thickness[:, np.newaxis, np.newaxis]
        else:
            strain, _ = self._parse_layer_inputs(layer, strain, magnetization)
            key = (id(layer), strain)
            if cache is not None and key in cache:
                return cache[key]
            eps = self.get_layer_dielectric_tensor(layer, strain)[:, 0, 0][:, np.newaxis]
            thickness = layer.structural.thickness.magnitude * (1 + strain)
        k_z = self.experiment.k.magnitude[:, np.newaxis] * _forward_sqrt(
            eps - np.cos(self.experiment.theta.magnitude) ** 2
        )
        result = (eps, k_z, thickness, layer.structural.roughness.magnitude)
        if cache is not None and np.ndim(thickness) == 0:
            cache[key] = result
        return result

    def calc_jones_matrices(self, layers, cache=None):
        """calc_jones_matrices

        Calculates the Jones matrices of the reflection and transmission of a
        stack of layers in the linear :math:`(\\sigma, \\pi)` basis, which are
        diagonal and independent of the magnetization.

        Args:
            layers (list[tuple]): layer, strain, and magnetization from the
                superstrate to the substrate, see :meth:`get_homogeneous_layers`.
            cache (dict, optional): cache of the constants of identical
                layers. Defaults to None.

        Returns:
            (tuple):
            - *R (ndarray[complex])* - Jones matrix of the reflection of shape
              (M, N, 2, 2).
            - *R_phi (ndarray[complex])* - the same Jones matrix for opposite
              magnetization.
            - *T (ndarray[complex])* - Jones matrix of the transmission.
            - *T_phi (ndarray[complex])* - the same Jones matrix for opposite
              magnetization.

        """
        # leading dimension of the delays for arrays of strains
        shape = self.experiment.theta.magnitude.shape
        for _, strain, _ in layers:
            if np.size(strain) > 1:
                shape = np.shape(strain) + shape
                break
        X_s = np.zeros(shape, dtype=np.complex128)
        X_p = np.zeros_like(X_s)
        t_s = np.ones_like(X_s)
        t_p = np.ones_like(X_s)
        # recursion from the substrate to the superstrate, where X is the
        # ratio of the up and down amplitudes at the top of the layer below;
        # the constants are calculated on the fly to keep only two layers
        # in memory for arrays of strains
        below = self._layer_constants(*layers[-1], cache)
        for j in range(len(layers) - 1, 0, -1):
            above = self._layer_constants(*layers[j - 1], cache)
            eps_a, k_a, d_a, _ = above
            eps_b, k_b, d_b, sigma = below
            r_s = (k_a - k_b) / (k_a + k_b)
            r_p = (eps_b * k_a - eps_a * k_b) / (eps_b * k_a + eps_a * k_b)
            tau_s = 1 + r_s
            # transmission of the electric field from the magnetic field
            tau_p = (1 + r_p) * np.sqrt(eps_a) / np.sqrt(eps_b)
            if sigma > 0:
                r_s = r_s * np.exp(-2 * k_a * k_b * sigma**2)
                r_p = r_p * np.exp(-2 * k_a * k_b * sigma**2)
                damping = np.exp((k_b - k_a) ** 2 * sigma**2 / 2)
                tau_s = tau_s * damping
                tau_p = tau_p * damping
            phase = np.exp(1j * k_b * d_b)
            denominator_s = 1 + r_s * X_s
            denominator_p = 1 + r_p * X_p
            t_s = t_s * tau_s / denominator_s * phase
            t_p = t_p * tau_p / denominator_p * phase
            X_s = (r_s + X_s) / denominator_s
            X_p = (r_p + X_p) / denominator_p
            if j > 1:
                # propagate to the top of the layer above
                phase_a = np.exp(2j * k_a * d_a)
                X_s = X_s * phase_a
                X_p = X_p * phase_a
            below = above

        R = np.zeros(shape + (2, 2), dtype=np.complex128)
        T = np.zeros_like(R)
        R[..., 0, 0], R[..., 1, 1] = X_s, X_p
        T[..., 0, 0], T[..., 1, 1] = t_s, t_p
        return R, R, T, T

    def calc_inhomogeneous_reflectivities(self, strain_map, magnetization_map):
        """calc_inhomogeneous_reflectivities

        Calculates the reflectivity and transmissivity of the sample structure
        for a block of delays of the ``strain_map`` at once, where the
        recursion runs over the layers for all delays, photon energies, and
        angles in parallel. The magnetization is neglected.

        Args:
            strain_map (ndarray[float]): strain of every layer for each delay.
            magnetization_map (ndarray[float]): neglected magnetization of
                every layer for each delay.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity for each delay.
            - *R_phi (ndarray[float])* - the same reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - transmissivity.
            - *T_phi (ndarray[float])* - the same transmissivity for opposite
              magnetization.

        """
        strain_map = np.asarray(strain_map, dtype=float)
        D = strain_map.shape[0]
        if D == 1:
            return super().calc_inhomogeneous_reflectivities(strain_map, magnetization_map)
        _, _, handles = self.S.get_layer_vectors()
        superstrate, substrate = self._superstrate_and_substrate()
        layers = (
            [superstrate]
            + [(handle, strain_map[:, i], None) for i, handle in enumerate(handles)]
            + substrate
        )
        Ref, _, Trans, _ = self.calc_jones_matrices(layers, {})
        # the intensities are calculated for all delays and photon energies
        shape = Ref.shape
        Ref = Ref.reshape((-1,) + shape[2:])
        Trans = Trans.reshape((-1,) + shape[2:])
        R, T = self.calc_intensities(Ref, Trans, self.pol_in, self.pol_out)
        R = R.reshape(shape[:3] + R.shape[2:])
        T = T.reshape(shape[:3] + T.shape[2:])
        return R, R, T, T

    def _delay_block_size(self):
        """number of delays, which are calculated at once, for about 2^20
        elements per array"""
        return max(1, 2**20 // max(1, np.size(self.experiment.theta.magnitude)))
