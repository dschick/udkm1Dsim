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

"""Common base of the multilayer formalisms, which calculate the reflectivity
and transmissivity of a stack of homogeneous layers from their dielectric
tensors, see :class:`.Dielectric`.

The magnetization and the dielectric tensor are given in the frame of the
sample of :cite:t:`elzo2012`: the surface is the :math:`xy`-plane, the surface
normal :math:`z` points out of the sample towards the superstrate, and
:math:`yz` is the plane of incidence, so that the :math:`\\sigma` polarization
is along :math:`x`. The grazing incidence angle :math:`\\theta` is measured
from the surface.

Every layer has four eigenmodes, two propagating down, i.e. into the sample,
and two up. Their boundary matrix :math:`A` holds the in-plane electric and
magnetic fields of the modes, and their normal wave-vector components
:math:`q_j = k_{z,j} / k` define the propagation through the layer. The stack
is assembled either by transfer matrices

.. math:: M = \\prod_j P_j A_j^{-1} A_{j-1}

or by the numerically stable scattering-matrix recursion of
:cite:t:`stepanov2000`, which avoids the exponentially growing terms of the
upward modes in thick and absorbing stacks.

"""

__all__ = ["MultilayerScattering", "circular_to_linear", "compose_scattering_matrices"]

__docformat__ = "restructuredtext"

from os import path
from time import time

import numpy as np
import pint
from tqdm.auto import tqdm

from ...helpers import convert_polar_to_cartesian, make_hash_md5
from ...structures.layers import Layer
from ...structures.structure import Structure
from .scattering import Scattering

u = pint.get_application_registry()
Q_ = u.Quantity

# change of the basis of the circular modes to the linear sigma/pi basis
_CIRCULAR_TO_LINEAR = np.array([[-1, 1], [-1j, -1j]], dtype=np.complex128)
_LINEAR_TO_CIRCULAR = np.array([[-1, 1j], [1, 1j]], dtype=np.complex128) * 0.5
# incoming polarization vectors in the (sigma, pi) basis
_POLARIZATION_VECTORS = {
    1: np.array([-np.sqrt(0.5), -1j * np.sqrt(0.5)], dtype=np.complex128),
    2: np.array([np.sqrt(0.5), -1j * np.sqrt(0.5)], dtype=np.complex128),
    3: np.array([1, 0], dtype=np.complex128),
    4: np.array([0, 1], dtype=np.complex128),
}


def circular_to_linear(matrix):
    """circular_to_linear

    Converts a Jones matrix of shape (..., 2, 2) from the basis of the right
    and left circular modes to the linear :math:`(\\sigma, \\pi)` basis.

    Args:
        matrix (ndarray[complex]): Jones matrix in the circular basis.

    Returns:
        matrix (ndarray[complex]): Jones matrix in the linear basis.

    """
    return np.matmul(np.matmul(_CIRCULAR_TO_LINEAR, matrix), _LINEAR_TO_CIRCULAR)


class MultilayerScattering(Scattering):
    r"""MultilayerScattering

    Base class of the multilayer formalisms, which take the dielectric tensor
    of every layer, see :meth:`.Layer.get_dielectric_tensor`, and the
    incoming and analyzer polarization. The child classes define the
    eigenmodes of the layers by :meth:`calc_modes` and select the assembly of
    the stack by transfer matrices or by the scattering-matrix recursion.

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
        theta (ndarray[float]): grazing incidence angles :math:`\theta` of
            scattering light
        qz (ndarray[float]): scattering vector :math:`q_z` of scattering light
        polarizations (dict): polarization states and according names.
        pol_in_state (int): incoming polarization state as defined in
            polarizations dict.
        pol_out_state (int): outgoing polarization state as defined in
            polarizations dict.
        pol_in (float): incoming polarization factor (can be a complex ndarray).
        pol_out (float): outgoing polarization factor (can be a complex ndarray).

    """

    _title = "Multilayer scattering simulation"
    _pol_in_states = (0, 1, 2, 3, 4, 5)
    _pol_out_states = (0, 1, 2, 3, 4, 5)
    # assembly of the stack, either "transfer" or "scattering"
    _assembly = "scattering"

    def get_hash(self, **kwargs):
        """get_hash

        Calculates an unique hash given by the energy :math:`E`, :math:`q_z`
        range, polarization states as well as the sample structure hash for
        relevant x-ray and magnetic parameters. Optionally, part of the
        ``strain_map`` and ``magnetization_map`` are used.

        Args:
            **kwargs (ndarray[float]): spatio-temporal strain and magnetization
                profile.

        Returns:
            hash (str): unique hash.

        """
        param = [
            self.pol_in_state,
            self.pol_out_state,
            self.pol_in,
            self.pol_out,
            self.experiment.qz.magnitude,
            self.experiment.energy.magnitude,
        ]

        if "strain_map" in kwargs:
            strain_map = kwargs.get("strain_map")
            if np.size(strain_map) > 1e6:
                strain_map = strain_map.flatten()[0:1000000]
            param.append(strain_map)
        if "magnetization_map" in kwargs:
            magnetization_map = kwargs.get("magnetization_map")
            if np.size(magnetization_map) > 1e6:
                magnetization_map = magnetization_map.flatten()[0:1000000]
            # cartesian magnetization maps since version 2.6.1
            param.append("cartesian")
            param.append(magnetization_map)

        return self.S.get_hash(types=["xray", "magnetic", "optical"]) + "_" + make_hash_md5(param)

    def set_polarization(
        self, pol_in_state, pol_out_state, polarization_in=None, polarization_out=None
    ):
        r"""set_polarization

        Sets the incoming and analyzer (outgoing) polarization at the same time.
        This is required to change the number of multiple elliptical
        polarizations of both, the incoming and outgoing polarization, as the
        individual setters check consistency against the currently set
        counterpart.

        Args:
            pol_in_state (int, str): incoming polarization state id or name.
            pol_out_state (int, str): outgoing polarization state id or name.
            polarization_in (list[tuple[alpha (Quantity), ellipticity (float)]],
                optional): elliptical incoming polarization, see
                :meth:`set_incoming_polarization`.
            polarization_out (list[tuple[alpha (Quantity), ellipticity (float)]],
                optional): elliptical outgoing polarization, see
                :meth:`set_outgoing_polarization`.

        """
        pol_in_state = self._parse_polarization_state(
            pol_in_state, self._pol_in_states, "incoming"
        )
        pol_out_state = self._parse_polarization_state(
            pol_out_state, self._pol_out_states, "analyzer"
        )
        pol_in = self.calc_incoming_polarization(pol_in_state, polarization_in)
        pol_out = self.calc_outgoing_polarization(pol_out_state, polarization_out)
        self.match_polarizations(pol_in, pol_out)

        self._store_incoming_polarization(pol_in_state, pol_in)
        self._store_outgoing_polarization(pol_out_state, pol_out)

    def set_incoming_polarization(self, pol_in_state, polarization=None):
        r"""set_incoming_polarization

        Sets the incoming polarization factor for circular +, circular -, sigma,
        pi, unpolarized, and elliptical polarization.
        In the case of elliptical polarization a single or list of tuple of the
        azimuth angle :math:`\alpha` and the ellipticity :math:`e` of the
        polarization can be input.
        :math:`0° \leq \alpha \leq +180°`
        :math:`-1 \leq e \leq +1`

        The number of multiple incoming polarizations must be either 1 or equal
        to the number of currently set outgoing polarizations. Use
        :meth:`set_polarization` to change both at the same time.

        Args:
            pol_in_state (int, str): incoming polarization state id or name.
            polarization (list[tuple[alpha (Quantity), ellipticity (float)]]):
                azimuth angle alpha of polarization (0 -> s; 90 -> p) (deg)
                ellipticity (-+1 -> circular left/right; 0 -> linear)

        """
        state = self._parse_polarization_state(pol_in_state, self._pol_in_states, "incoming")
        pol_in = self.calc_incoming_polarization(state, polarization)
        self._check_polarizations(pol_in, self.pol_out)
        self._store_incoming_polarization(state, pol_in)

    def set_outgoing_polarization(self, pol_out_state, polarization=None):
        r"""set_outgoing_polarization

        Sets the outgoing polarization factor for circular +, circular -, sigma,
        pi, unpolarized, and elliptical polarization.
        In the case of elliptical polarization a single or list of tuple of the
        azimuth angle :math:`\alpha` and the ellipticity :math:`e` of the
        polarization can be input.
        :math:`0° \leq \alpha \leq +180°`
        :math:`-1 \leq e \leq +1`

        The number of multiple outgoing polarizations must be either 1 or equal
        to the number of currently set incoming polarizations. Use
        :meth:`set_polarization` to change both at the same time.

        Args:
            pol_out_state (int, str): outgoing polarization state id or name.
            polarization (list[tuple[alpha (Quantity), ellipticity (float)]]):
                azimuth angle alpha of polarization (0 -> s; 90 -> p) (deg)
                ellipticity (-+1 -> circular left/right; 0 -> linear)

        """
        state = self._parse_polarization_state(pol_out_state, self._pol_out_states, "analyzer")
        pol_out = self.calc_outgoing_polarization(state, polarization)
        self._check_polarizations(self.pol_in, pol_out)
        self._store_outgoing_polarization(state, pol_out)

    @staticmethod
    def calc_incoming_polarization(pol_in_state, polarization=None):
        """calc_incoming_polarization

        Calculates the incoming polarization factor for a given state.

        Args:
            pol_in_state (int): incoming polarization state id.
            polarization (list[tuple[alpha (Quantity), ellipticity (float)]]):
                elliptical polarization, see :meth:`set_incoming_polarization`.

        Returns:
            pol_in (ndarray[complex]): incoming polarization factor.

        """
        if pol_in_state == 0:  # unpolarized
            return np.array([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.complex128)
        if pol_in_state == 5:  # elliptical
            return MultilayerScattering.calc_elliptical_polarization(polarization)
        return _POLARIZATION_VECTORS[pol_in_state].copy()

    @staticmethod
    def calc_outgoing_polarization(pol_out_state, polarization=None):
        """calc_outgoing_polarization

        Calculates the outgoing polarization factor for a given state.

        Args:
            pol_out_state (int): outgoing polarization state id.
            polarization (list[tuple[alpha (Quantity), ellipticity (float)]]):
                elliptical polarization, see :meth:`set_outgoing_polarization`.

        Returns:
            pol_out (ndarray[complex]): outgoing polarization factor.

        """
        if pol_out_state == 0:  # no analyzer
            return np.array([], dtype=np.complex128)
        if pol_out_state == 5:  # elliptical
            return MultilayerScattering.calc_elliptical_polarization(polarization)
        return _POLARIZATION_VECTORS[pol_out_state].conj()

    @staticmethod
    def calc_elliptical_polarization(polarization):
        r"""calc_elliptical_polarization

        Calculates the polarization factors for a single or list of tuple of
        the azimuth angle :math:`\alpha` and the ellipticity :math:`e`.

        Args:
            polarization (list[tuple[alpha (Quantity), ellipticity (float)]]):
                azimuth angle alpha of polarization (0 -> s; 90 -> p) (deg)
                ellipticity (-+1 -> circular left/right; 0 -> linear)

        Returns:
            pol (ndarray[complex]): polarization factors of shape (N, 2).

        """
        if polarization is None:
            raise ValueError(
                "For elliptical polarization a single or list tuple of azimuth "
                "angle alpha and ellipticity e must be provided."
            )
        if type(polarization) is tuple:
            polarization = [polarization]

        N = len(polarization)
        pol = np.zeros((N, 2), dtype=np.complex128)
        for i, (alpha, ellipticity) in enumerate(polarization):
            try:
                alpha = alpha.to("rad").magnitude
            except AttributeError:
                pass

            if ellipticity > 1 or ellipticity < -1:
                raise ValueError("ellipticity must be -1 <= e <= +1")
            else:
                epsilon = np.arctan(ellipticity)

            pol[i, :] = np.array(
                [
                    np.cos(alpha) * np.cos(epsilon) - 1j * np.sin(alpha) * np.sin(epsilon),
                    np.sin(alpha) * np.cos(epsilon) + 1j * np.cos(alpha) * np.sin(epsilon),
                ],
                dtype=np.complex128,
            )
        return pol

    @staticmethod
    def match_polarizations(pol_in, pol_out):
        """match_polarizations

        Checks that the number of incoming and outgoing polarizations is either
        equal or one of them is 1, and tiles the single one to the length of
        the other.

        Args:
            pol_in (ndarray[complex]): incoming polarization factor.
            pol_out (ndarray[complex]): outgoing polarization factor.

        Returns:
            (tuple):
            - *pol_in (ndarray[complex])* - incoming polarization factors of
              shape (num_pol, 2).
            - *pol_out (ndarray[complex])* - outgoing polarization factors of
              shape (num_pol, 2) or (num_pol, 0) for no analyzer.
            - *num_pol (int)* - number of polarizations.

        """
        # add second dimension to polarization vectors for iteration
        pol_in = np.atleast_2d(pol_in)
        pol_out = np.atleast_2d(pol_out)

        num_pol_in = pol_in.shape[0]
        num_pol_out = pol_out.shape[0]

        if (num_pol_in > 1) and (num_pol_out > 1) and (num_pol_in != num_pol_out):
            raise ValueError(
                f"the number of multiple incoming (#{num_pol_in:d}) and outgoing "
                f"(#{num_pol_out:d}) elliptical polarizations must be the same."
            )
        num_pol = max(num_pol_in, num_pol_out)
        if num_pol_in < num_pol:
            pol_in = np.tile(pol_in, (num_pol, 1))
        if num_pol_out < num_pol:
            pol_out = np.tile(pol_out, (num_pol, 1))
        return pol_in, pol_out, num_pol

    def _check_polarizations(self, pol_in, pol_out):
        """_check_polarizations

        Checks the consistency of incoming and outgoing polarizations and adds
        a hint to :meth:`set_polarization` to the error message.

        Args:
            pol_in (ndarray[complex]): incoming polarization factor.
            pol_out (ndarray[complex]): outgoing polarization factor.

        """
        try:
            self.match_polarizations(pol_in, pol_out)
        except ValueError as e:
            raise ValueError(
                f"{e} Use set_polarization() to change the incoming and "
                "outgoing polarizations at the same time."
            ) from None

    @property
    def num_pol(self):
        """int: number of incoming/outgoing polarization pairs."""
        return self.match_polarizations(self.pol_in, self.pol_out)[2]

    # ---- Jones matrices and intensities ------------------------------------

    @staticmethod
    def jones_from_transfer_matrix(RT):
        """jones_from_transfer_matrix

        Calculates the Jones matrices of the reflection and transmission in
        the linear :math:`(\\sigma, \\pi)` basis from the transfer matrix
        :math:`M` of the stack from :cite:t:`elzo2012`, which fulfills
        :math:`(t, 0) = M (1, r)` for the down and up modes.

        Args:
            RT (ndarray[complex]): transfer matrix of shape (M, N, 4, 4).

        Returns:
            (tuple):
            - *Ref (ndarray[complex])* - Jones matrix of the reflection of
              shape (M, N, 2, 2).
            - *Trans (ndarray[complex])* - Jones matrix of the transmission.

        """
        Ref = np.zeros(RT.shape[:-2] + (2, 2), dtype=np.complex128)
        Trans = np.zeros_like(Ref)

        d = np.divide(1, RT[..., 3, 3] * RT[..., 2, 2] - RT[..., 3, 2] * RT[..., 2, 3])
        Ref[..., 0, 0] = (-RT[..., 3, 3] * RT[..., 2, 0] + RT[..., 2, 3] * RT[..., 3, 0]) * d
        Ref[..., 0, 1] = (-RT[..., 3, 3] * RT[..., 2, 1] + RT[..., 2, 3] * RT[..., 3, 1]) * d
        Ref[..., 1, 0] = (RT[..., 3, 2] * RT[..., 2, 0] - RT[..., 2, 2] * RT[..., 3, 0]) * d
        Ref[..., 1, 1] = (RT[..., 3, 2] * RT[..., 2, 1] - RT[..., 2, 2] * RT[..., 3, 1]) * d

        Trans[..., 0, 0] = (
            RT[..., 0, 0] + RT[..., 0, 2] * Ref[..., 0, 0] + RT[..., 0, 3] * Ref[..., 1, 0]
        )
        Trans[..., 0, 1] = (
            RT[..., 0, 1] + RT[..., 0, 2] * Ref[..., 0, 1] + RT[..., 0, 3] * Ref[..., 1, 1]
        )
        Trans[..., 1, 0] = (
            RT[..., 1, 0] + RT[..., 1, 2] * Ref[..., 0, 0] + RT[..., 1, 3] * Ref[..., 1, 0]
        )
        Trans[..., 1, 1] = (
            RT[..., 1, 1] + RT[..., 1, 2] * Ref[..., 0, 1] + RT[..., 1, 3] * Ref[..., 1, 1]
        )
        return circular_to_linear(Ref), circular_to_linear(Trans)

    @staticmethod
    def calc_intensities(Ref, Trans, pol_in, pol_out):
        """calc_intensities

        Calculates the reflectivity and transmissivity from the Jones matrices
        in the linear :math:`(\\sigma, \\pi)` basis for a given incoming and
        analyzer polarization.

        Args:
            Ref (ndarray[complex]): Jones matrix of the reflection of shape
                (M, N, 2, 2).
            Trans (ndarray[complex]): Jones matrix of the transmission.
            pol_in (ndarray[complex]): incoming polarization factor.
            pol_out (ndarray[complex]): outgoing polarization factor.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity.
            - *T (ndarray[float])* - transmissivity.

        """
        # enable multiple polarizations
        pol_in, pol_out, num_pol = MultilayerScattering.match_polarizations(pol_in, pol_out)

        R = np.empty((Ref.shape[0], Ref.shape[1], num_pol))
        T = np.empty((Ref.shape[0], Ref.shape[1], num_pol))

        for i in range(num_pol):
            if pol_out.size == 0:
                # no analyzer polarization
                R[:, :, i] = np.real(
                    np.matmul(
                        np.square(np.absolute(np.matmul(Ref, pol_in[i, :]))),
                        np.array([1, 1], dtype=np.complex128),
                    )
                )
                T[:, :, i] = np.real(
                    np.matmul(
                        np.square(np.absolute(np.matmul(Trans, pol_in[i, :]))),
                        np.array([1, 1], dtype=np.complex128),
                    )
                )
            else:
                R[:, :, i] = np.real(
                    np.square(np.absolute(np.matmul(np.matmul(Ref, pol_in[i, :]), pol_out[i, :])))
                )
                T[:, :, i] = np.real(
                    np.square(
                        np.absolute(np.matmul(np.matmul(Trans, pol_in[i, :]), pol_out[i, :]))
                    )
                )

        if num_pol == 1:
            return np.squeeze(R, axis=2), np.squeeze(T, axis=2)
        else:
            return R, T

    @staticmethod
    def calc_reflectivity_transmissivity_from_matrix(RT, pol_in, pol_out):
        """calc_reflectivity_transmissivity_from_matrix

        Calculates the actual reflectivity and transmissivity from the
        reflectivity-transmission matrix for a given incoming and analyzer
        polarization from :cite:t:`elzo2012`, see
        :meth:`jones_from_transfer_matrix` and :meth:`calc_intensities`.

        Args:
            RT (ndarray[complex]): reflection-transmission matrix.
            pol_in (ndarray[complex]): incoming polarization factor.
            pol_out (ndarray[complex]): outgoing polarization factor.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity.
            - *T (ndarray[float])* - transmissivity.

        """
        Ref, Trans = MultilayerScattering.jones_from_transfer_matrix(RT)
        return MultilayerScattering.calc_intensities(Ref, Trans, pol_in, pol_out)

    @staticmethod
    def calc_kerr_angles(jones, pi_incident=False):
        r"""calc_kerr_angles

        Calculates the rotation :math:`\theta_K` and ellipticity
        :math:`\epsilon_K` of the polarization ellipse of the reflected or
        transmitted light, i.e. the Kerr or Faraday effect, for an incoming
        :math:`\sigma` or :math:`\pi` polarization from the ratio
        :math:`\chi = r_{\pi\sigma} / r_{\sigma\sigma}` or
        :math:`\chi = r_{\sigma\pi} / r_{\pi\pi}` of the Jones matrix:

        .. math::

            \tan(2 \theta_K) = \frac{2 \Re(\chi)}{1 - |\chi|^2}, \quad
            \sin(2 \epsilon_K) = \frac{2 \Im(\chi)}{1 + |\chi|^2}

        which reduce to :math:`\theta_K + i \epsilon_K \approx \chi` for small
        angles. In the Fresnel convention of the :math:`\pi` polarization,
        i.e. :math:`r_{\pi\pi} = -r_{\sigma\sigma}` at normal incidence, the
        polar Kerr rotation for :math:`\sigma` and :math:`\pi` incidence have
        opposite signs at normal incidence.

        Args:
            jones (ndarray[complex]): Jones matrix of shape (..., 2, 2) in the
                linear :math:`(\sigma, \pi)` basis, see
                :meth:`homogeneous_jones_matrices`.
            pi_incident (bool, optional): incoming :math:`\pi` instead of
                :math:`\sigma` polarization. Defaults to False.

        Returns:
            (tuple):
            - *rotation (Quantity)* - Kerr or Faraday rotation (deg).
            - *ellipticity (Quantity)* - Kerr or Faraday ellipticity (deg).

        """
        if pi_incident:
            ratio = jones[..., 0, 1] / jones[..., 1, 1]
        else:
            ratio = jones[..., 1, 0] / jones[..., 0, 0]
        abs2 = np.abs(ratio) ** 2
        rotation = 0.5 * np.arctan2(2 * np.real(ratio), 1 - abs2)
        ellipticity = 0.5 * np.arcsin(np.clip(2 * np.imag(ratio) / (1 + abs2), -1, 1))
        return Q_(rotation, u.rad).to("deg"), Q_(ellipticity, u.rad).to("deg")

    @staticmethod
    def calc_kerr_effect_from_matrix(RT):
        """calc_kerr_effect_from_matrix

        Calculates the Kerr rotation and ellipticity for :math:`\\sigma` and
        :math:`\\pi` incident polarization from the transfer matrix
        independent of the given incoming and analyzer polarization, see
        :meth:`calc_kerr_angles`.

        Args:
            RT (ndarray[complex]): reflection-transmission matrix.

        Returns:
            (tuple):
            - *kerr_sigma (tuple[Quantity])* - rotation and ellipticity for
              :math:`\\sigma` incident polarization.
            - *kerr_pi (tuple[Quantity])* - rotation and ellipticity for
              :math:`\\pi` incident polarization.

        """
        Ref, _ = MultilayerScattering.jones_from_transfer_matrix(RT)
        return (
            MultilayerScattering.calc_kerr_angles(Ref),
            MultilayerScattering.calc_kerr_angles(Ref, pi_incident=True),
        )

    @staticmethod
    def calc_roughness_matrix(roughness, k_z, last_k_z):
        """calc_roughness_matrix

        Calculates the roughness matrix for an interface with a gaussian
        roughness for the :cite:t:`elzo2012`.

        Args:
            roughness (float): gaussian roughness of the interface (m).
            k_z (ndarray[float)]: internal wave vector.
            last_k_z (ndarray[float)]: last internal wave vector.

        Returns:
            W (ndarray[float]): roughness matrix.

        """
        W = np.zeros([k_z.shape[0], k_z.shape[1], 4, 4], dtype=np.complex128)
        rugosp = np.exp(-((k_z + last_k_z) ** 2) * roughness**2 / 2)
        rugosn = np.exp(-((-k_z + last_k_z) ** 2) * roughness**2 / 2)
        W[:, :, 0, 0] = rugosn
        W[:, :, 0, 1] = rugosn
        W[:, :, 0, 2] = rugosp
        W[:, :, 0, 3] = rugosp
        W[:, :, 1, 0] = rugosn
        W[:, :, 1, 1] = rugosn
        W[:, :, 1, 2] = rugosp
        W[:, :, 1, 3] = rugosp
        W[:, :, 2, 0] = rugosp
        W[:, :, 2, 1] = rugosp
        W[:, :, 2, 2] = rugosn
        W[:, :, 2, 3] = rugosn
        W[:, :, 3, 0] = rugosp
        W[:, :, 3, 1] = rugosp
        W[:, :, 3, 2] = rugosn
        W[:, :, 3, 3] = rugosn

        return W

    # ---- eigenmodes and assembly of the stack ------------------------------

    def calc_modes(self, eps):
        """calc_modes

        Calculates the eigenmodes of a homogeneous layer with the dielectric
        tensor ``eps`` for all photon energies and incidence angles. Must be
        overwritten by the child classes.

        Args:
            eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3).

        Returns:
            (tuple):
            - *A (ndarray[complex])* - boundary matrix of shape (M, N, 4, 4).
            - *A_phi (ndarray[complex])* - boundary matrix for opposite
              magnetization.
            - *q (ndarray[complex])* - normal wave-vector components
              :math:`k_{z,j} / k` of the down and up modes of shape (M, N, 4).
            - *q_phi (ndarray[complex])* - normal wave-vector components for
              opposite magnetization.
            - *k_z (ndarray[complex])* - mean normal wave-vector component
              :math:`k_z / k` of shape (M, N) for the roughness.

        """
        raise NotImplementedError

    def get_layer_dielectric_tensor(self, layer, strain=0.0, magnetization=None):
        """get_layer_dielectric_tensor

        Dielectric tensor of a layer at the photon energies of the simulation,
        see :meth:`.Layer.get_dielectric_tensor`.

        Args:
            layer (Layer): homogeneous layer.
            strain (float, optional): strain of the layer. Defaults to 0.
            magnetization (ndarray[float], optional): cartesian
                magnetization ``(x, y, z)``. Defaults to the magnetization of
                the layer.

        Returns:
            eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3).

        """
        return layer.get_dielectric_tensor(self.experiment.energy.magnitude, strain, magnetization)

    @staticmethod
    def _parse_layer_inputs(layer, strain, magnetization):
        """strain as float and magnetization as tuple, where magnetizations
        with fewer than three elements are replaced by the one of the layer"""
        strain = float(np.ravel(strain)[0]) if np.size(strain) > 0 else 0.0
        if magnetization is None or np.size(magnetization) < 3:
            magnetization = layer._magnetization_for_dielectric()
        return strain, tuple(float(m) for m in np.ravel(magnetization)[:3])

    def _layer_modes(self, layer, strain, magnetization, cache=None):
        """eigenmodes, thickness, and roughness of a layer, which are cached
        for identical layers, strains, and magnetizations"""
        if not isinstance(layer, Layer):
            raise TypeError(f"{type(self).__name__} can only handle Layers and their sub-classes!")
        strain, magnetization = self._parse_layer_inputs(layer, strain, magnetization)
        key = (id(layer), strain, magnetization)
        if cache is not None and key in cache:
            return cache[key]
        eps = self.get_layer_dielectric_tensor(layer, strain, magnetization)
        modes = self.calc_modes(eps)
        result = modes + (
            layer.structural.thickness.magnitude * (1 + strain),
            layer.structural.roughness.magnitude,
            eps,
        )
        if cache is not None:
            cache[key] = result
        return result

    def _expand_sub_structures(self, S, strains, magnetizations):
        """layers of all sub-structures with their strain and magnetization
        from the values per sub-structure, see :meth:`homogeneous_reflectivity`"""
        entries = []
        counter = 0
        for sub_structure, repetitions in S.sub_structures:
            if isinstance(sub_structure, Layer):
                entry = (sub_structure, strains[counter], magnetizations[counter])
                entries += [entry] * repetitions
                counter += 1
            else:
                num = sub_structure.get_number_of_sub_structures()
                entries += (
                    self._expand_sub_structures(
                        sub_structure,
                        strains[counter : counter + num],
                        magnetizations[counter : counter + num],
                    )
                    * repetitions
                )
                counter += num
        return entries

    def _superstrate_and_substrate(self):
        """the superstrate and the layers of the substrate"""
        superstrate = self.S.superstrate[0]
        if not isinstance(superstrate, Layer):
            raise TypeError(
                f"In {type(self).__name__} the superstrate must be a Layer, "
                "e.g. Vacuum or AmorphousLayer, and should be present!"
            )
        substrate, N = self.S.substrate
        if not isinstance(substrate, Layer):
            raise ValueError("There should be a substrate present.")
        return (superstrate, 0.0, None), [(substrate, 0.0, None)] * N

    def get_homogeneous_layers(self, *args):
        """get_homogeneous_layers

        List of all layers of the stack from the superstrate to the substrate
        with their strain and magnetization for homogeneous strains and
        magnetizations of the sub-structures.

        Args:
            args (ndarray[float], optional): strains and magnetization for each
                sub-structure.

        Returns:
            layers (list[tuple]): layer, strain, and magnetization.

        """
        if not isinstance(self.S, Structure):
            raise TypeError("The sample must be a Structure!")
        L = self.S.get_number_of_sub_structures()
        strains = args[0] if len(args) > 0 else np.zeros([L, 1])
        magnetizations = args[1] if len(args) > 1 else np.zeros([L, 1])
        superstrate, substrate = self._superstrate_and_substrate()
        return (
            [superstrate]
            + self._expand_sub_structures(self.S, strains, magnetizations)
            + substrate
        )

    def get_inhomogeneous_layers(self, strains, magnetizations):
        """get_inhomogeneous_layers

        List of all layers of the stack from the superstrate to the substrate
        with their strain and magnetization for a single delay of the
        ``strain_map`` and ``magnetization_map``.

        Args:
            strains (ndarray[float]): strain of every layer.
            magnetizations (ndarray[float]): magnetization of every layer.

        Returns:
            layers (list[tuple]): layer, strain, and magnetization.

        """
        _, _, handles = self.S.get_layer_vectors()
        superstrate, substrate = self._superstrate_and_substrate()
        return (
            [superstrate]
            + [(handles[i], strains[i], magnetizations[i]) for i in range(len(handles))]
            + substrate
        )

    def calc_jones_matrices(self, layers, cache=None):
        """calc_jones_matrices

        Calculates the Jones matrices of the reflection and transmission of a
        stack of layers in the linear :math:`(\\sigma, \\pi)` basis for the
        given and the opposite magnetization.

        Args:
            layers (list[tuple]): layer, strain, and magnetization from the
                superstrate to the substrate, see :meth:`get_homogeneous_layers`.
            cache (dict, optional): cache of the eigenmodes of identical
                layers. Defaults to None.

        Returns:
            (tuple):
            - *R (ndarray[complex])* - Jones matrix of the reflection of shape
              (M, N, 2, 2).
            - *R_phi (ndarray[complex])* - Jones matrix of the reflection for
              opposite magnetization.
            - *T (ndarray[complex])* - Jones matrix of the transmission.
            - *T_phi (ndarray[complex])* - Jones matrix of the transmission for
              opposite magnetization.

        """
        k = self.experiment.k.magnitude[:, np.newaxis]
        self._check_superstrate(layers[0])
        top = self._layer_modes(*layers[0], cache=cache)
        if self._assembly == "transfer":
            assemble = self._transfer_step
        else:
            assemble = self._scattering_step
        stack = stack_phi = None
        last, last_phi = (top[0], top[4]), (top[1], top[4])
        for entry in layers[1:]:
            A, A_phi, q, q_phi, k_z, distance, roughness, eps = self._layer_modes(
                *entry, cache=cache
            )
            W = None
            if roughness > 0:
                W = self.calc_roughness_matrix(roughness, k * k_z, k * last[1])
            stack = assemble(stack, A, q, last[0], W, k * distance)
            stack_phi = assemble(stack_phi, A_phi, q_phi, last_phi[0], W, k * distance)
            last, last_phi = (A, k_z), (A_phi, k_z)

        if self._assembly == "transfer":
            R, T = self.jones_from_transfer_matrix(stack)
            R_phi, T_phi = self.jones_from_transfer_matrix(stack_phi)
        else:
            R, R_phi = circular_to_linear(stack[2]), circular_to_linear(stack_phi[2])
            T, T_phi = circular_to_linear(stack[0]), circular_to_linear(stack_phi[0])
        T, T_phi = self.transmitted_field(T, eps), self.transmitted_field(T_phi, eps)
        R, T = self.incident_field(R, T, top[7])
        R_phi, T_phi = self.incident_field(R_phi, T_phi, top[7])
        return R, R_phi, T, T_phi

    @staticmethod
    def incident_field(R, T, eps):
        r"""incident_field

        Converts the Jones matrices of the reflection and transmission from the
        amplitudes of the eigenmodes of the superstrate to the amplitudes of
        its electric field. The transmission of the eigenmodes of
        :func:`~udkm1Dsim.simulations.scattering.elzo.elzo_modes` is
        :math:`\\varepsilon_{00}` times smaller, where :math:`\\varepsilon` is the
        dielectric tensor of the superstrate, while their reflection is
        given for the electric field. Must be overwritten for other
        eigenmodes.

        Args:
            R (ndarray[complex]): Jones matrix of the reflection of the modes
                of shape (M, N, 2, 2).
            T (ndarray[complex]): Jones matrix of the transmission.
            eps (ndarray[complex]): dielectric tensor of the superstrate of
                shape (M, 3, 3).

        Returns:
            (tuple):
            - *R (ndarray[complex])* - Jones matrix of the reflected electric
              field.
            - *T (ndarray[complex])* - Jones matrix of the transmission for
              the incident electric field.

        """
        return R, T * eps[:, 0, 0][:, np.newaxis, np.newaxis, np.newaxis]

    @staticmethod
    def transmitted_field(T, eps):
        r"""transmitted_field

        Converts the Jones matrix of the transmission from the amplitudes of
        the eigenmodes of :func:`~udkm1Dsim.simulations.scattering.elzo.elzo_modes`
        to the amplitudes of the transmitted electric field, which are
        :math:`\\varepsilon_{00}` times smaller, where :math:`\\varepsilon` is the
        dielectric tensor of the last layer of the substrate. Must be
        overwritten for other eigenmodes.

        Args:
            T (ndarray[complex]): Jones matrix of the transmission of the modes
                of shape (M, N, 2, 2).
            eps (ndarray[complex]): dielectric tensor of the last layer of
                shape (M, 3, 3).

        Returns:
            T (ndarray[complex]): Jones matrix of the transmitted electric
            field.

        """
        return T / eps[:, 0, 0][:, np.newaxis, np.newaxis, np.newaxis]

    def _check_superstrate(self, entry):
        """check of the superstrate, which must be isotropic to define the
        polarizations, by the child classes"""

    @staticmethod
    def _transfer_step(stack, A, q, last_A, W, phase):
        """transfer matrix of the stack including the next layer"""
        F = np.linalg.solve(A, last_A)
        if W is not None:
            F = F * W
        M = np.exp(1j * phase[..., np.newaxis] * q)[..., np.newaxis] * F
        return M if stack is None else np.matmul(M, stack)

    @staticmethod
    def _scattering_step(stack, A, q, last_A, W, phase):
        """scattering matrices of the stack including the next layer by the
        recursion of :cite:t:`stepanov2000`"""
        X = np.linalg.solve(A, last_A)
        if W is not None:
            X = X * W
        Xtt, Xtr = X[..., :2, :2], X[..., :2, 2:]
        Xrt, Xrr = X[..., 2:, :2], X[..., 2:, 2:]
        # the down modes decay and the inverse propagation of the up modes as
        # well, so that no exponentially growing terms appear
        down = np.exp(1j * phase[..., np.newaxis] * q[..., :2])
        up_inv = np.exp(-1j * phase[..., np.newaxis] * q[..., 2:])
        inv_Xrr = np.linalg.inv(Xrr)
        Mtt = down[..., :, np.newaxis] * (Xtt - Xtr @ inv_Xrr @ Xrt)
        Mtr = down[..., :, np.newaxis] * (Xtr @ inv_Xrr) * up_inv[..., np.newaxis, :]
        Mrt = -inv_Xrr @ Xrt
        Mrr = inv_Xrr * up_inv[..., np.newaxis, :]
        if stack is None:
            return Mtt, Mtr, Mrt, Mrr
        return compose_scattering_matrices(stack, (Mtt, Mtr, Mrt, Mrr))

    # ---- simulations -------------------------------------------------------

    def homogeneous_jones_matrices(self, *args, polar=False):
        """homogeneous_jones_matrices

        Calculates the Jones matrices of the reflection and transmission of
        the whole sample structure in the linear :math:`(\\sigma, \\pi)` basis
        allowing only for homogeneous strain and magnetization, e.g. for the
        Kerr effect, see :meth:`calc_kerr_angles`.

        Args:
            args (ndarray[float], optional): strains and magnetization for each
                sub-structure. The magnetization is given as :math:`L \times 3`
                array in cartesian coordinates ``(x, y, z)`` by default, see
                :meth:`inhomogeneous_reflectivity`.
            polar (bool, optional): the magnetization is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad.
                Defaults to False.

        Returns:
            (tuple):
            - *R (ndarray[complex])* - Jones matrix of the reflection of shape
              (M, N, 2, 2).
            - *R_phi (ndarray[complex])* - Jones matrix of the reflection for
              opposite magnetization.
            - *T (ndarray[complex])* - Jones matrix of the transmission.
            - *T_phi (ndarray[complex])* - Jones matrix of the transmission for
              opposite magnetization.

        """
        if len(args) > 1:
            args = (args[0], self.parse_magnetization_map(args[1], polar), *args[2:])
        return self.calc_jones_matrices(self.get_homogeneous_layers(*args), cache={})

    def homogeneous_reflectivity(self, *args, polar=False):
        r"""homogeneous_reflectivity

        Calculates the reflectivity :math:`R` and transmissivity :math:`T` of
        the whole sample structure allowing only for homogeneous strain and
        magnetization for the incoming and analyzer polarization, see
        :meth:`homogeneous_jones_matrices`.

        Args:
            args (ndarray[float], optional): strains and magnetization for each
                sub-structure. The magnetization is given as :math:`L \times 3`
                array in cartesian coordinates ``(x, y, z)`` by default, see
                :meth:`inhomogeneous_reflectivity`.
            polar (bool, optional): the magnetization is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad.
                Defaults to False.

        Returns:
            (tuple):
            - *R (ndarray[float])* - homogeneous reflectivity.
            - *R_phi (ndarray[float])* - homogeneous reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - homogeneous transmissivity.
            - *T_phi (ndarray[float])* - homogeneous transmissivity for
              opposite magnetization.

        """
        t1 = time()
        self.disp_message("Calculating _homogeneous_reflectivity_ ...")
        Ref, Ref_phi, Trans, Trans_phi = self.homogeneous_jones_matrices(*args, polar=polar)
        R, T = self.calc_intensities(Ref, Trans, self.pol_in, self.pol_out)
        R_phi, T_phi = self.calc_intensities(Ref_phi, Trans_phi, self.pol_in, self.pol_out)
        self.disp_message(f"Elapsed time for _homogeneous_reflectivity_: {time() - t1:f} s")
        return R, R_phi, T, T_phi

    def calc_inhomogeneous_reflectivity(self, strains, magnetizations):
        """calc_inhomogeneous_reflectivity

        Calculates the reflectivity and transmissivity of the sample structure
        for a single delay of the ``strain_map`` and ``magnetization_map``.

        Args:
            strains (ndarray[float]): strain of every layer.
            magnetizations (ndarray[float]): magnetization of every layer.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity.
            - *R_phi (ndarray[float])* - reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - transmissivity.
            - *T_phi (ndarray[float])* - transmissivity for opposite
              magnetization.

        """
        Ref, Ref_phi, Trans, Trans_phi = self.calc_jones_matrices(
            self.get_inhomogeneous_layers(strains, magnetizations)
        )
        R, T = self.calc_intensities(Ref, Trans, self.pol_in, self.pol_out)
        R_phi, T_phi = self.calc_intensities(Ref_phi, Trans_phi, self.pol_in, self.pol_out)
        return R, R_phi, T, T_phi

    def calc_inhomogeneous_reflectivities(self, strain_map, magnetization_map):
        """calc_inhomogeneous_reflectivities

        Calculates the reflectivity and transmissivity of the sample structure
        for a block of delays of the ``strain_map`` and ``magnetization_map``,
        see :meth:`calc_inhomogeneous_reflectivity` for every delay.
        Formalisms, which calculate all delays at once, overwrite this method
        and :meth:`_delay_block_size`.

        Args:
            strain_map (ndarray[float]): strain of every layer for each delay.
            magnetization_map (ndarray[float]): magnetization of every layer
                for each delay.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity for each delay.
            - *R_phi (ndarray[float])* - reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - transmissivity.
            - *T_phi (ndarray[float])* - transmissivity for opposite
              magnetization.

        """
        results = [
            self.calc_inhomogeneous_reflectivity(strains, magnetizations)
            for strains, magnetizations in zip(strain_map, magnetization_map)
        ]
        return tuple(np.stack(result) for result in zip(*results))

    def _delay_block_size(self):
        """number of delays, which are calculated at once"""
        return 1

    @staticmethod
    def parse_magnetization_map(magnetization_map, polar=False):
        r"""parse_magnetization_map

        Returns the magnetization as plain cartesian array, which is used
        internally. The ``magnetization_map`` is given in cartesian
        coordinates ``(x, y, z)`` by default or in polar coordinates
        ``(amplitude, phi, gamma)`` with the angles in rad for ``polar=True``
        in its last dimension, see :meth:`inhomogeneous_reflectivity`. Maps
        with only a single component in the last dimension or empty maps are
        returned unchanged, since the magnetization of the layers is used for
        them.

        Args:
            magnetization_map (ndarray[float]): magnetization map.
            polar (bool, optional): the magnetization map is given in polar
                coordinates. Defaults to False.

        Returns:
            magnetization_map (ndarray[float]): cartesian magnetization map.

        """
        if not polar:
            return magnetization_map
        magnetization_map = np.asarray(magnetization_map, dtype=float)
        if magnetization_map.ndim == 0 or magnetization_map.shape[-1] != 3:
            return magnetization_map
        return convert_polar_to_cartesian(magnetization_map)

    def inhomogeneous_reflectivity(
        self, strain_map=np.array([]), magnetization_map=np.array([]), polar=False, **kwargs
    ):
        r"""inhomogeneous_reflectivity

        Returns the reflectivity and transmissivity of an inhomogeneously
        strained and magnetized sample structure for a given ``strain_map``
        and ``magnetization_map`` in space and time for each layer of the
        sample structure. If no reflectivity is saved in the cache it is
        calculated. Providing the ``calc_type`` for the calculation the
        corresponding sub-routines for the reflectivity computation are
        called:

        * ``parallel`` parallelization over the time steps utilizing
          `Dask <https://dask.org/>`_
        * ``sequential`` no parallelization at all

        Args:
            strain_map (ndarray[float], optional): spatio-temporal strain
                profile.
            magnetization_map (ndarray[float], optional): spatio-temporal
                magnetization profile :math:`M \times N \times 3` with
                :math:`M` delays and :math:`N` layers. The magnetization is
                given in cartesian coordinates ``(x, y, z)`` by default, as
                returned by
                :meth:`~udkm1Dsim.simulations.magnetization.Magnetization.get_magnetization_map`.
                Its amplitude scales the magnetic part of the dielectric tensor.
            polar (bool, optional): the ``magnetization_map`` is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad,
                as before version 2.6.1. Defaults to False.
            **kwargs:
                - *calc_type (str)* - type of calculation.
                - *dask_client (Dask.Client)* - Dask client.

        Returns:
            (tuple):
            - *R (ndarray[float])* - inhomogeneous reflectivity.
            - *R_phi (ndarray[float])* - inhomogeneous reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - inhomogeneous transmissivity.
            - *T_phi (ndarray[float])* - inhomogeneous transmissivity for opposite
              magnetization.

        """
        if not isinstance(magnetization_map, np.ndarray):
            raise TypeError("magnetization_map must be a numpy ndarray!")
        magnetization_map = self.parse_magnetization_map(magnetization_map, polar)
        # create a hash of all simulation parameters
        filename = (
            f"inhomogeneous_reflectivity_{type(self).__name__}_"
            + self.get_hash(strain_map=strain_map, magnetization_map=magnetization_map)
            + ".npz"
        )
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        # check if we find some corresponding data in the cache dir
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            self.disp_message("_inhomogeneous_reflectivity_ loaded from file:\n\t" + filename)
            return tmp["R"], tmp["R_phi"], tmp["T"], tmp["T_phi"]

        t1 = time()
        self.disp_message("Calculating _inhomogeneous_reflectivity_ ...")
        if not isinstance(strain_map, np.ndarray):
            raise TypeError("strain_map must be a numpy ndarray!")
        calc_type = kwargs.get("calc_type", "sequential")
        if calc_type not in ["parallel", "sequential"]:
            raise TypeError("calc_type must be either _parallel_ or _sequential_!")

        M = np.size(strain_map, 0)
        N = np.size(magnetization_map, 0)
        if (M == 0) and (N > 0):
            strain_map = np.zeros([np.size(magnetization_map, 0), np.size(magnetization_map, 1)])
        elif (M > 0) and (N == 0):
            magnetization_map = np.zeros_like(strain_map)
        elif (M == 0) and (N == 0):
            raise ValueError("At least a strain_map or magnetzation_map must be given!")
        elif M != N:
            raise ValueError(
                "The strain_map and magnetzation_map must have the same number of delay steps!"
            )

        M = np.size(strain_map, 0)
        num_pol = self.num_pol
        shape = (M,) + self.experiment.qz.magnitude.shape + ((num_pol,) if num_pol > 1 else ())
        R, R_phi, T, T_phi = (np.zeros(shape) for _ in range(4))

        # blocks of delays, which are calculated at once
        size = self._delay_block_size()
        blocks = [slice(i, min(i + size, M)) for i in range(0, M, size)]
        if calc_type == "parallel":
            dask_client = kwargs.get("dask_client")
            if not dask_client:
                raise ValueError("no dask client set")
            from dask import delayed  # to allow parallel computation

            tasks = [
                delayed(self.calc_inhomogeneous_reflectivities)(
                    strain_map[block, :], magnetization_map[block, :]
                )
                for block in blocks
            ]
            results = dask_client.compute(tasks, sync=True)
        else:
            iterator = (
                tqdm(blocks, desc="Progress", leave=True) if self.progress_bar else iter(blocks)
            )
            results = (
                self.calc_inhomogeneous_reflectivities(
                    strain_map[block, :], magnetization_map[block, :]
                )
                for block in iterator
            )
        for block, result in zip(blocks, results):
            R[block], R_phi[block], T[block], T_phi[block] = result

        self.disp_message(f"Elapsed time for _inhomogeneous_reflectivity_: {time() - t1:f} s")
        self.save(
            full_filename,
            {"R": R, "R_phi": R_phi, "T": T, "T_phi": T_phi},
            "_inhomogeneous_reflectivity_",
        )
        return R, R_phi, T, T_phi


def compose_scattering_matrices(stack, following):
    """compose_scattering_matrices

    Composes the scattering matrices :math:`(W_{tt}, W_{tr}, W_{rt}, W_{rr})`
    of a stack with the ones of the following part of the stack by Eqs.
    (63)-(67) of :cite:t:`stepanov2000`, where :math:`W_{rt}` is the
    reflection and :math:`W_{tt}` the transmission in the basis of the modes
    of the superstrate and the last layer.

    Args:
        stack (tuple[ndarray[complex]]): scattering matrices of the stack of
            shape (..., 2, 2).
        following (tuple[ndarray[complex]]): scattering matrices of the
            following part of the stack.

    Returns:
        stack (tuple[ndarray[complex]]): scattering matrices of the combined
        stack.

    """
    W_tt, W_tr, W_rt, W_rr = stack
    Mtt, Mtr, Mrt, Mrr = following
    identity = np.eye(2, dtype=np.complex128)
    A = Mtt @ np.linalg.inv(identity - W_tr @ Mrt)
    B = W_rr @ np.linalg.inv(identity - Mrt @ W_tr)
    return A @ W_tt, Mtr + A @ W_tr @ Mrr, W_rt + B @ Mrt @ W_tt, B @ Mrr
