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
from the surface in the superstrate, so that the in-plane wave vector
:math:`k_y = n k \\cos\\theta` with the real part of the refractive index
:math:`n` of the superstrate is conserved in the stack, which also allows for
the coupling by a prism.

Every layer has four eigenmodes, two propagating down, i.e. into the sample,
and two up. Their boundary matrix :math:`A` holds the in-plane electric and
magnetic fields of the modes, and their normal wave-vector components
:math:`q_j = k_{z,j} / k` define the propagation through the layer. The stack
is assembled either by transfer matrices

.. math:: M = \\prod_j P_j A_j^{-1} A_{j-1}

or by the numerically stable scattering-matrix recursion of
:cite:t:`stepanov2000`, which avoids the exponentially growing terms of the
upward modes in thick and absorbing stacks.

The fields inside the stack and the absorption from their Poynting vector
follow from :meth:`MultilayerScattering.homogeneous_fields` and
:meth:`MultilayerScattering.homogeneous_absorption`.

"""

__all__ = ["MultilayerScattering", "circular_to_linear", "compose_scattering_stacks"]

__docformat__ = "restructuredtext"


import numpy as np
import pint
from tqdm.auto import tqdm

from ...helpers import convert_polar_to_cartesian, make_hash_md5
from ...structures.layers import Layer, Vacuum
from ...structures.structure import Structure
from ..result import Result, find_source, get_entry
from .scattering import ALIASES, Scattering

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


class _Period:
    """Period of a stack, which is assembled once and then repeated by
    integer powers for its number of repetitions, see
    :meth:`MultilayerScattering.calc_jones_matrices`.

    The homogeneous simulations keep the tree of the sub-structures and their
    repetitions, e.g. 100 periods of a superlattice, as list of layers and
    periods, see :meth:`MultilayerScattering._homogeneous_layer_tree`, so
    that every period is assembled only once and then raised to the power of
    its repetitions with :math:`\\mathcal{O}(\\log N)` matrix products. This
    tree is not available from the :class:`.Structure`, which only provides
    flattened vectors of all layers, e.g. by
    :attr:`.Structure.layer_vectors`, as used by the inhomogeneous
    simulations with a strain or magnetization per layer.
    """

    __slots__ = ("entries", "repetitions")

    def __init__(self, entries, repetitions):
        self.entries = entries
        self.repetitions = repetitions


def _repeated(entries, repetitions):
    """entries of a part of the stack, which is repeated, as :class:`_Period`
    or as plain entries for a single repetition"""
    if repetitions == 1:
        return list(entries)
    return [_Period(list(entries), repetitions)]


def _flatten_layers(entries):
    """all layers of a stack tree with :class:`_Period` entries as plain list
    in the order from the top to the bottom, e.g. for :class:`.Parratt`"""
    layers = []
    for entry in entries:
        if isinstance(entry, _Period):
            layers += _flatten_layers(entry.entries) * entry.repetitions
        else:
            layers.append(entry)
    return layers


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


#: names of the Jones matrices of a :class:`~udkm1Dsim.simulations.result.Result`
JONES_NAMES = (
    "jones_reflection",
    "jones_reflection_phi",
    "jones_transmission",
    "jones_transmission_phi",
)
#: names of the reflectivities and transmissivities of a result
REFLECTIVITY_NAMES = ("reflectivity", "reflectivity_phi", "transmissivity", "transmissivity_phi")
#: names of the fields of a result
FIELD_NAMES = ("electric_field", "electric_field_phi", "magnetic_field", "magnetic_field_phi")
#: names of the absorptions of a result
ABSORPTION_NAMES = (
    "absorption",
    "absorption_phi",
    "absorption_matrix",
    "absorption_matrix_phi",
    "layer_absorption",
    "layer_absorption_phi",
    "layer_absorption_matrix",
    "layer_absorption_matrix_phi",
    "substrate_flux",
    "substrate_flux_phi",
    "substrate_flux_matrix",
    "substrate_flux_matrix_phi",
)
#: units of the fields and absorptions of a result
FIELD_UNITS = {
    "absorption": "1/m",
    "absorption_phi": "1/m",
    "absorption_matrix": "1/m",
    "absorption_matrix_phi": "1/m",
}
#: names of the rotations and ellipticities of a result
ROTATION_NAMES = (
    "reflection_rotation",
    "reflection_ellipticity",
    "reflection_rotation_phi",
    "reflection_ellipticity_phi",
    "transmission_rotation",
    "transmission_ellipticity",
    "transmission_rotation_phi",
    "transmission_ellipticity_phi",
)


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
    # signs of the in-plane fields (E_x, E_y, H_x, H_y) of the modes in the
    # sample frame of Elzo et al., see calc_stack_fields()
    _field_frame = np.ones(4)
    # the fields depend on the magnetization
    _magnetic_fields = True

    def get_hash(self, **kwargs):
        """get_hash

        Calculates an unique hash given by the energy :math:`E`, :math:`q_z`
        range, polarization states as well as the sample structure hash for
        relevant x-ray and magnetic parameters. Optionally, part of the
        ``strain_map``, ``magnetization_map``, and ``temp_map`` are used.

        Args:
            **kwargs (ndarray[float]): spatio-temporal strain, magnetization,
                and temperature profile.

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
        if np.size(kwargs.get("temp_map", [])) > 0:
            # only a given temp_map changes the hash of former results
            temp_map = kwargs.get("temp_map")
            if np.size(temp_map) > 1e6:
                temp_map = temp_map.flatten()[0:1000000]
            param.append(temp_map)

        return self.S.get_hash(types=["xray", "magnetic", "optical"]) + "_" + make_hash_md5(param)

    # ---- polarizations ----------------------------------------------------------

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

        Calculates the incoming polarization factor, i.e. the Jones vector in
        the linear :math:`(\\sigma, \\pi)` basis, for a given state.
        Unpolarized light has no Jones vector, so that it is marked by the zero
        vector, for which :meth:`calc_reflectivity_transmissivity` averages
        the intensities for the incoming :math:`\\sigma` and :math:`\\pi`
        polarization.

        Args:
            pol_in_state (int): incoming polarization state id.
            polarization (list[tuple[alpha (Quantity), ellipticity (float)]]):
                elliptical polarization, see :meth:`set_incoming_polarization`.

        Returns:
            pol_in (ndarray[complex]): incoming polarization factor.

        """
        if pol_in_state == 0:  # unpolarized, see calc_reflectivity_transmissivity
            return np.zeros(2, dtype=np.complex128)
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

    def _pi_incident(self):
        """True for an incoming pi and False for a sigma polarization, which
        define the rotation and ellipticity."""
        if self.pol_in_state not in (3, 4):
            raise ValueError(
                "The rotation and ellipticity are defined for an incoming sigma or pi "
                f"polarization, but it is {self.polarizations[self.pol_in_state]}!"
            )
        return self.pol_in_state == 4

    @property
    def num_pol(self):
        """int: number of incoming/outgoing polarization pairs."""
        return self.match_polarizations(self.pol_in, self.pol_out)[2]

    # ---- Jones matrices and intensities -----------------------------------------

    def calc_reflectivity_transmissivity(self, Ref, Trans, pol_in=None, pol_out=None):
        """calc_reflectivity_transmissivity

        Calculates the reflectivity and transmissivity from the Jones matrices
        in the linear :math:`(\\sigma, \\pi)` basis for the incoming and
        analyzer polarization, e.g. from :meth:`homogeneous_jones_matrices`
        or :meth:`inhomogeneous_jones_matrices`.

        Unpolarized incoming light is an incoherent mixture of two orthogonal
        polarizations, so that its intensities are the average of the ones
        for the incoming :math:`\\sigma` and :math:`\\pi` polarization, also
        for any analyzer, e.g.
        :math:`R = (|r_{\\sigma\\sigma}|^2 + |r_{\\sigma\\pi}|^2) / 2` for the
        :math:`\\sigma` analyzer. It has no Jones vector and is marked by the
        zero vector of :meth:`calc_incoming_polarization`.

        Args:
            Ref (ndarray[complex]): Jones matrix of the reflection of shape
                (..., 2, 2).
            Trans (ndarray[complex]): Jones matrix of the transmission.
            pol_in (ndarray[complex], optional): incoming polarization factor.
                Defaults to the one of the simulation.
            pol_out (ndarray[complex], optional): outgoing polarization
                factor. Defaults to the one of the simulation.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity of shape (...) or
              (..., P) for P polarizations.
            - *T (ndarray[float])* - transmissivity.

        """
        pol_in = self.pol_in if pol_in is None else pol_in
        pol_out = self.pol_out if pol_out is None else pol_out
        # enable multiple polarizations
        pol_in, pol_out, num_pol = self.match_polarizations(pol_in, pol_out)
        R = np.empty(Ref.shape[:-2] + (num_pol,))
        T = np.empty(Trans.shape[:-2] + (num_pol,))
        for i in range(num_pol):
            if np.any(pol_in[i, :]):
                incoming = [(1.0, pol_in[i, :])]
            else:
                # unpolarized: incoherent average of sigma and pi
                incoming = [(0.5, _POLARIZATION_VECTORS[3]), (0.5, _POLARIZATION_VECTORS[4])]
            R[..., i] = 0
            T[..., i] = 0
            for weight, vector in incoming:
                E_R = np.matmul(Ref, vector)
                E_T = np.matmul(Trans, vector)
                if pol_out.size == 0:
                    # no analyzer polarization
                    R[..., i] += weight * np.sum(np.abs(E_R) ** 2, axis=-1)
                    T[..., i] += weight * np.sum(np.abs(E_T) ** 2, axis=-1)
                else:
                    R[..., i] += weight * np.abs(np.matmul(E_R, pol_out[i, :])) ** 2
                    T[..., i] += weight * np.abs(np.matmul(E_T, pol_out[i, :])) ** 2
        if num_pol == 1:
            return R[..., 0], T[..., 0]
        return R, T

    @staticmethod
    def calc_rotation_ellipticity(jones, pi_incident=False):
        r"""calc_rotation_ellipticity

        Calculates the rotation :math:`\theta` and ellipticity
        :math:`\epsilon` of the polarization ellipse of the reflected or
        transmitted light for an incoming :math:`\sigma` or :math:`\pi`
        polarization, e.g. the Kerr or Faraday effect or the change by a
        birefringent layer, from the ratio
        :math:`\chi = r_{\pi\sigma} / r_{\sigma\sigma}` or
        :math:`\chi = r_{\sigma\pi} / r_{\pi\pi}` of the Jones matrix:

        .. math::

            \tan(2 \theta) = \frac{2 \Re(\chi)}{1 - |\chi|^2}, \quad
            \sin(2 \epsilon) = \frac{2 \Im(\chi)}{1 + |\chi|^2}

        which reduce to :math:`\theta + i \epsilon \approx \chi` for small
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
            - *rotation (Quantity)* - rotation of the polarization (deg).
            - *ellipticity (Quantity)* - ellipticity of the polarization
              (deg).

        """
        if pi_incident:
            ratio = jones[..., 0, 1] / jones[..., 1, 1]
        else:
            ratio = jones[..., 1, 0] / jones[..., 0, 0]
        abs2 = np.abs(ratio) ** 2
        rotation = 0.5 * np.arctan2(2 * np.real(ratio), 1 - abs2)
        ellipticity = 0.5 * np.arcsin(np.clip(2 * np.imag(ratio) / (1 + abs2), -1, 1))
        return Q_(rotation, u.rad).to("deg"), Q_(ellipticity, u.rad).to("deg")

    # ---- eigenmodes and assembly of the stack -----------------------------------

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

    def calc_layer_dielectric_tensor(self, layer, strain=0.0, magnetization=None):
        """calc_layer_dielectric_tensor

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

    def calc_homogeneous_layers(self, strains=np.array([]), magnetizations=np.array([])):
        """calc_homogeneous_layers

        List of all layers of the stack from the superstrate to the substrate
        with their strain and magnetization for homogeneous strains and
        magnetizations of the sub-structures.

        Args:
            strains (ndarray[float], optional): strain of each sub-structure.
            magnetizations (ndarray[float], optional): cartesian magnetization
                of each sub-structure.

        Returns:
            layers (list[tuple]): layer, strain, and magnetization.

        """
        return _flatten_layers(self._homogeneous_layer_tree(strains, magnetizations))

    def calc_inhomogeneous_layers(self, strains, magnetizations):
        """calc_inhomogeneous_layers

        List of all layers of the stack from the superstrate to the substrate
        with their strain and magnetization for a single delay of the
        ``strain_map`` and ``magnetization_map``.

        Args:
            strains (ndarray[float]): strain of every layer.
            magnetizations (ndarray[float]): magnetization of every layer.

        Returns:
            layers (list[tuple]): layer, strain, and magnetization.

        """
        _, _, handles = self.S.layer_vectors
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
                superstrate to the substrate, see :meth:`calc_homogeneous_layers`,
                where repeated parts of the stack are assembled only once and
                raised to the power of their repetitions by the transfer
                matrices or the scattering matrices.
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
        # state of the assembly from the superstrate to the substrate for the
        # given and the opposite magnetization:
        # - stack: transfer or scattering matrices of the layers assembled so far
        # - last: boundary matrix A and normal wave vector k_z of the last layer
        # - eps: dielectric tensor of the last layer for the transmitted field
        state = {
            "stack": [None, None],
            "last": [(top[0], top[4]), (top[1], top[4])],
            "eps": top[7],
        }
        self._assemble(layers[1:], state, cache, k)
        stack, stack_phi = state["stack"]
        eps = state["eps"]

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

        The transmission of all multilayer formalisms is the electric field
        at the bottom of the last layer of the substrate per incident
        electric field, i.e. after the propagation through the thickness of
        the substrate layers, below which the substrate is semi-infinite.
        Hence, a substrate layer of zero thickness gives the transmitted
        field at the top of the substrate, as many other codes do.

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

    def _in_plane_wavevector(self):
        """In-plane component :math:`k_y / k_0 = n \\cos\\vartheta` of the wave
        vector, which is conserved in the stack, for the grazing angles
        :math:`\\vartheta` in the superstrate with the real part of its
        refractive index :math:`n`, e.g. for the coupling by a prism, where
        it exceeds 1."""
        cos_theta = np.cos(self.experiment.theta.magnitude)
        superstrate = self.S.superstrate[0] if isinstance(self.S, Structure) else None
        if superstrate is None or isinstance(superstrate, Vacuum):
            return cos_theta
        n = superstrate.get_ref_index(self.experiment.energy.magnitude)
        return np.real(np.atleast_1d(n))[:, np.newaxis] * cos_theta

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
        eps = self.calc_layer_dielectric_tensor(layer, strain, magnetization)
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
        from the values per sub-structure, see :meth:`homogeneous_reflectivity_transmissivity`"""
        entries = []
        counter = 0
        for sub_structure, repetitions in S.sub_structures:
            if isinstance(sub_structure, Layer):
                entry = (sub_structure, strains[counter], magnetizations[counter])
                entries += _repeated([entry], repetitions)
                counter += 1
            else:
                num = sub_structure.num_sub_structures
                entries += _repeated(
                    self._expand_sub_structures(
                        sub_structure,
                        strains[counter : counter + num],
                        magnetizations[counter : counter + num],
                    ),
                    repetitions,
                )
                counter += num
        return entries

    def _superstrate_and_substrate(self):
        """the superstrate and the layers of the substrate"""
        self._check_superstrate_and_substrate()
        superstrate = self.S.superstrate[0]
        substrate, N = self.S.substrate
        return (superstrate, 0.0, None), _repeated([(substrate, 0.0, None)], N)

    def _homogeneous_layer_tree(self, strains=np.array([]), magnetizations=np.array([])):
        """_homogeneous_layer_tree

        List of all layers of the stack from the superstrate to the substrate
        with their strain and magnetization for homogeneous strains and
        magnetizations of the sub-structures.

        Args:
            strains (ndarray[float], optional): strain of each sub-structure.
            magnetizations (ndarray[float], optional): cartesian magnetization
                of each sub-structure.

        Returns:
            layers (list[tuple, _Period]): layer, strain, and
            magnetization, where repeated parts are kept as repetitions.

        """
        if not isinstance(self.S, Structure):
            raise TypeError("The sample must be a Structure!")
        L = self.S.num_sub_structures
        if np.size(strains) == 0:
            strains = np.zeros([L, 1])
        if np.size(magnetizations) == 0:
            magnetizations = np.zeros([L, 1])
        superstrate, substrate = self._superstrate_and_substrate()
        return (
            [superstrate]
            + self._expand_sub_structures(self.S, strains, magnetizations)
            + substrate
        )

    def _check_superstrate(self, entry):
        """check of the superstrate, which must be isotropic to define the
        polarizations, by the child classes"""

    def _assemble(self, entries, state, cache, k):
        """Adds the layers and repeated parts of the stack to the transfer or
        scattering matrices in ``state`` for both magnetizations."""
        step = self._transfer_step if self._assembly == "transfer" else self._scattering_step
        for entry in entries:
            if isinstance(entry, _Period):
                self._assemble_period(entry, state, cache, k)
                continue
            A, A_phi, q, q_phi, k_z, distance, roughness, eps = self._layer_modes(
                *entry, cache=cache
            )
            W = None
            if roughness > 0:
                W = self.calc_roughness_matrix(roughness, k * k_z, k * state["last"][0][1])
            for i, (A_i, q_i) in enumerate([(A, q), (A_phi, q_phi)]):
                state["stack"][i] = step(
                    state["stack"][i], A_i, q_i, state["last"][i][0], W, k * distance
                )
                state["last"][i] = (A_i, k_z)
            state["eps"] = eps

    def _assemble_period(self, period, state, cache, k):
        """Adds a period of the stack, where the first repetition starts at
        the layer above and all others at the end of the period."""
        N = period.repetitions
        if N < 1 or not period.entries:
            return
        first = {"stack": [None, None], "last": list(state["last"]), "eps": state["eps"]}
        self._assemble(period.entries, first, cache, k)
        blocks = first["stack"]
        end = first
        if N > 1:
            end = {"stack": [None, None], "last": list(first["last"]), "eps": first["eps"]}
            self._assemble(period.entries, end, cache, k)
            blocks = [
                self._compose_stacks(block, self._repeat_stack(inner, N - 1))
                for block, inner in zip(first["stack"], end["stack"])
            ]
        state["stack"] = [
            block if stack is None else self._compose_stacks(stack, block)
            for stack, block in zip(state["stack"], blocks)
        ]
        state["last"], state["eps"] = end["last"], end["eps"]

    def _compose_stacks(self, stack, following):
        """transfer or scattering matrices of a stack and the following part"""
        if self._assembly == "transfer":
            return np.matmul(following, stack)
        return compose_scattering_stacks(stack, following)

    def _repeat_stack(self, stack, n):
        """transfer or scattering matrices of a period repeated ``n`` times"""
        if self._assembly == "transfer":
            return _repeat_transfer_matrix(stack, n)
        return _repeat_scattering_stack(stack, n)

    @staticmethod
    def _transfer_step(stack, A, q, last_A, W, phase):
        """transfer matrix of the stack including the next layer, i.e. its
        projection A_j^-1 A_j-1 at the interface and propagation P_j"""
        interface = np.linalg.solve(A, last_A)
        if W is not None:
            interface = interface * W
        M = np.exp(1j * phase[..., np.newaxis] * q)[..., np.newaxis] * interface
        return M if stack is None else np.matmul(M, stack)

    @staticmethod
    def _scattering_step(stack, A, q, last_A, W, phase):
        """scattering matrices of the stack including the next layer by the
        recursion of :cite:t:`stepanov2000`"""
        interface = np.linalg.solve(A, last_A)
        if W is not None:
            interface = interface * W
        Xtt, Xtr = interface[..., :2, :2], interface[..., :2, 2:]
        Xrt, Xrr = interface[..., 2:, :2], interface[..., 2:, 2:]
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
        return compose_scattering_stacks(stack, (Mtt, Mtr, Mrt, Mrr))

    # ---- homogeneous simulations ------------------------------------------------

    def homogeneous_jones_matrices(
        self,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
    ):
        """homogeneous_jones_matrices

        Calculates the Jones matrices of the reflection and transmission of
        the whole sample structure in the linear :math:`(\\sigma, \\pi)` basis
        allowing only for homogeneous strain and magnetization, from which
        :meth:`calc_reflectivity_transmissivity` and
        :meth:`calc_rotation_ellipticity` follow.

        Args:
            strains (ndarray[float], optional): strain of each sub-structure.
                Defaults to no strain.
            magnetizations (ndarray[float], optional): magnetization of each
                sub-structure as :math:`L \times 3` array in cartesian
                coordinates ``(x, y, z)`` by default, see
                :meth:`inhomogeneous_jones_matrices`. Defaults to no
                magnetization.
            temps (ndarray[float], optional): temperature of each sub-structure,
                also for :math:`K` sub-systems, e.g. electrons and phonons. The
                temperature dependence of the dielectric tensor is planned for
                a future version, so that given ``temps`` raise a
                ``NotImplementedError``.
            polar (bool, optional): the magnetization is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad.
                Defaults to False.

        Returns:
            result (Result): with the Jones matrices of the reflection
            ``jones_reflection`` and the transmission ``jones_transmission``,
            see :meth:`transmitted_field`, also for opposite magnetization
            ``jones_reflection_phi`` and ``jones_transmission_phi``, of the
            dimensions ``energy`` and ``angles`` and shape (M, N, 2, 2), see
            :meth:`~udkm1Dsim.simulations.scattering.scattering.Scattering.new_scattering_result`.

        """
        results = self._timed(
            "homogeneous_jones_matrices",
            self.calc_homogeneous_jones_matrices,
            strains,
            magnetizations,
            temps,
            polar=polar,
        )
        return self._scattering_result(
            "homogeneous Jones matrices",
            JONES_NAMES,
            results,
            inputs={
                "strains": strains,
                "magnetizations": magnetizations,
                "temps": temps,
                "polar": polar,
            },
        )

    def calc_homogeneous_jones_matrices(
        self,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
    ):
        """calc_homogeneous_jones_matrices

        Plain arrays of :meth:`homogeneous_jones_matrices` without the
        metadata of a :class:`~udkm1Dsim.simulations.result.Result`, e.g. for
        fits with many calls.

        Args:
            strains (ndarray[float], optional): strain of each sub-structure.
            magnetizations (ndarray[float], optional): magnetization of each
                sub-structure.
            temps (ndarray[float], optional): temperature of each
                sub-structure, which raises a ``NotImplementedError`` yet.
            polar (bool, optional): the magnetization is given in polar
                coordinates. Defaults to False.

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
        self._check_experiment()
        self._reject_temperatures(temps, "temps")
        if np.size(magnetizations) > 0:
            magnetizations = self.parse_magnetization_map(magnetizations, polar)
        return self.calc_jones_matrices(
            self._homogeneous_layer_tree(strains, magnetizations), cache={}
        )

    def homogeneous_reflectivity_transmissivity(
        self,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
    ):
        r"""homogeneous_reflectivity_transmissivity

        Calculates the reflectivity :math:`R` and transmissivity :math:`T` of
        the whole sample structure allowing only for homogeneous strain and
        magnetization for the incoming and analyzer polarization, see
        :meth:`homogeneous_jones_matrices` and
        :meth:`calc_reflectivity_transmissivity`.

        Args:
            strains (ndarray[float], optional): strain of each sub-structure.
                Defaults to no strain.
            magnetizations (ndarray[float], optional): magnetization of each
                sub-structure as :math:`L \times 3` array in cartesian
                coordinates ``(x, y, z)`` by default, see
                :meth:`inhomogeneous_jones_matrices`. Defaults to no
                magnetization.
            temps (ndarray[float], optional): temperature of each sub-structure,
                also for :math:`K` sub-systems, e.g. electrons and phonons. The
                temperature dependence of the dielectric tensor is planned for
                a future version, so that given ``temps`` raise a
                ``NotImplementedError``.
            polar (bool, optional): the magnetization is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad.
                Defaults to False.

        Returns:
            result (Result): with the ``reflectivity`` and ``transmissivity``,
            also for opposite magnetization ``reflectivity_phi`` and
            ``transmissivity_phi``, of the dimensions ``energy`` and ``angles``, see
            :meth:`~udkm1Dsim.simulations.scattering.scattering.Scattering.new_scattering_result`.

        """
        results = self._timed(
            "homogeneous_reflectivity_transmissivity",
            self.calc_homogeneous_reflectivity_transmissivity,
            strains,
            magnetizations,
            temps,
            polar=polar,
        )
        return self._scattering_result(
            "homogeneous reflectivity and transmissivity",
            REFLECTIVITY_NAMES,
            results,
            inputs={
                "strains": strains,
                "magnetizations": magnetizations,
                "temps": temps,
                "polar": polar,
            },
        )

    def calc_homogeneous_reflectivity_transmissivity(
        self,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
    ):
        """calc_homogeneous_reflectivity_transmissivity

        Plain arrays of :meth:`homogeneous_reflectivity_transmissivity`
        without the metadata of a
        :class:`~udkm1Dsim.simulations.result.Result`, e.g. for fits with many
        calls, see :meth:`calc_homogeneous_jones_matrices` for the arguments.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity.
            - *R_phi (ndarray[float])* - reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - transmissivity.
            - *T_phi (ndarray[float])* - transmissivity for opposite
              magnetization.

        """
        jones = self.calc_homogeneous_jones_matrices(strains, magnetizations, temps, polar=polar)
        return self._reduce_reflectivity_transmissivity(self, jones)

    def homogeneous_rotation_ellipticity(
        self,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
    ):
        r"""homogeneous_rotation_ellipticity

        Calculates the rotation and ellipticity of the polarization of the
        reflected and transmitted light, e.g. the Kerr and Faraday effect, of
        the whole sample structure allowing only for homogeneous strain and
        magnetization for the incoming :math:`\sigma` or :math:`\pi`
        polarization, see :meth:`homogeneous_jones_matrices` and
        :meth:`calc_rotation_ellipticity`.

        Args:
            strains (ndarray[float], optional): strain of each sub-structure.
                Defaults to no strain.
            magnetizations (ndarray[float], optional): magnetization of each
                sub-structure as :math:`L \times 3` array in cartesian
                coordinates ``(x, y, z)`` by default, see
                :meth:`inhomogeneous_jones_matrices`. Defaults to no
                magnetization.
            temps (ndarray[float], optional): temperature of each sub-structure,
                also for :math:`K` sub-systems, e.g. electrons and phonons. The
                temperature dependence of the dielectric tensor is planned for
                a future version, so that given ``temps`` raise a
                ``NotImplementedError``.
            polar (bool, optional): the magnetization is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad.
                Defaults to False.

        Returns:
            result (Result): with the ``reflection_rotation`` and
            ``reflection_ellipticity``, e.g. the Kerr effect, and the
            ``transmission_rotation`` and ``transmission_ellipticity``, e.g.
            the Faraday effect, also for opposite magnetization with the
            suffix ``_phi``, in (deg) of the dimensions ``energy`` and ``angles``, see
            :meth:`~udkm1Dsim.simulations.scattering.scattering.Scattering.new_scattering_result`.

        """
        results = self._timed(
            "homogeneous_rotation_ellipticity",
            self.calc_homogeneous_rotation_ellipticity,
            strains,
            magnetizations,
            temps,
            polar=polar,
        )
        return self._scattering_result(
            "homogeneous rotation and ellipticity",
            ROTATION_NAMES,
            results,
            "deg",
            inputs={
                "strains": strains,
                "magnetizations": magnetizations,
                "temps": temps,
                "polar": polar,
            },
        )

    def calc_homogeneous_rotation_ellipticity(
        self,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
    ):
        """calc_homogeneous_rotation_ellipticity

        Plain arrays of :meth:`homogeneous_rotation_ellipticity` without the
        metadata of a :class:`~udkm1Dsim.simulations.result.Result`, e.g. for
        fits with many calls, see :meth:`calc_homogeneous_jones_matrices` for
        the arguments.

        Returns:
            (tuple): rotation and ellipticity (deg) of the reflection, of the
            reflection for opposite magnetization, of the transmission, and of
            the transmission for opposite magnetization.

        """
        self._pi_incident()
        jones = self.calc_homogeneous_jones_matrices(strains, magnetizations, temps, polar=polar)
        return self._reduce_rotation_ellipticity(self, jones)

    # ---- fields and absorption --------------------------------------------------

    def calc_absorption(self, K, pol_in=None):
        r"""calc_absorption

        Absorption :math:`p^\dagger K p` for the Jones vector :math:`p` of the
        incoming polarization from the hermitian absorption matrix :math:`K`
        in the linear :math:`(\sigma, \pi)` basis, e.g. of
        :meth:`homogeneous_absorption`, and :math:`(K_{\sigma\sigma} +
        K_{\pi\pi}) / 2` for unpolarized light.

        For partially polarized light with the coherency matrix :math:`J`
        :cite:p:`born1999`, the absorption is :math:`\mathrm{tr}(K J)`. Hence,
        the four real parameters of :math:`K` are the absorption of
        unpolarized light :math:`\mathrm{tr}(K) / 2`, the linear dichroism
        :math:`K_{\sigma\sigma} - K_{\pi\pi}` and :math:`2\,\mathrm{Re}\,
        K_{\sigma\pi}` along :math:`\pm 45°`, and the circular dichroism
        :math:`2\,\mathrm{Im}\, K_{\sigma\pi}`.

        Args:
            K (ndarray[complex]): absorption matrices of shape (..., 2, 2).
            pol_in (ndarray[complex], optional): incoming polarization factor.
                Defaults to the one of the simulation.

        Returns:
            A (ndarray[float]): absorption of shape (...) or (..., P) for P
            polarizations.

        """
        pol_in = np.atleast_2d(self.pol_in if pol_in is None else pol_in)
        num_pol = pol_in.shape[0]
        A = np.empty(K.shape[:-2] + (num_pol,))
        for i in range(num_pol):
            vector = pol_in[i, :]
            if np.any(vector):
                A[..., i] = np.real(np.einsum("...ij,i,j->...", K, vector.conj(), vector))
            else:
                # unpolarized: incoherent average of sigma and pi
                A[..., i] = np.real(K[..., 0, 0] + K[..., 1, 1]) / 2
        if num_pol == 1:
            return A[..., 0]
        return A

    def homogeneous_fields(
        self,
        distances=None,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
        dichroic=False,
    ):
        r"""homogeneous_fields

        Electric field :math:`E` and magnetic field :math:`Z_0 H` inside the
        sample structure with homogeneous strain and magnetization for an
        incident :math:`\sigma` or :math:`\pi` polarization of unit amplitude,
        see :meth:`calc_homogeneous_fields`.

        Args:
            distances (ndarray[float, Quantity], optional): depths below the
                surface (m), negative in the superstrate. Defaults to the
                centers of the layers.
            strains (ndarray[float], optional): strain of each sub-structure.
            magnetizations (ndarray[float], optional): magnetization of each
                sub-structure, see :meth:`homogeneous_jones_matrices`.
            temps (ndarray[float], optional): temperature of each
                sub-structure, which raises a ``NotImplementedError`` yet.
            polar (bool, optional): the magnetization is given in polar
                coordinates. Defaults to False.
            dichroic (bool, optional): also for the opposite magnetization,
                which needs a second calculation for magnetic layers.
                Defaults to False.

        Returns:
            result (Result): ``electric_field`` and ``magnetic_field`` of
            shape (M, N, Z, 3, 2) for the components :math:`(x, y, z)` in the
            frame of the sample and the incident :math:`\sigma` and
            :math:`\pi` polarization, and with the suffix ``_phi`` for the
            opposite magnetization.

        """
        distances, results = self._timed(
            "homogeneous_fields",
            self._calc_homogeneous_fields,
            distances,
            strains,
            magnetizations,
            temps,
            polar=polar,
            dichroic=dichroic,
            absorption=False,
        )
        return self._field_result(
            "homogeneous fields",
            FIELD_NAMES,
            results,
            distances,
            {"strains": strains, "magnetizations": magnetizations, "temps": temps, "polar": polar},
        )

    def calc_homogeneous_fields(
        self,
        distances=None,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
        dichroic=False,
    ):
        r"""calc_homogeneous_fields

        Plain arrays of :meth:`homogeneous_fields`.

        The fields are the superposition of the eigenmodes of each layer.
        Their amplitudes follow from the reflection matrices of the stack
        below each layer from the substrate upwards and the transmission from
        the superstrate downwards, as in the scattering matrices of
        :cite:t:`ko1988,stepanov2000`, so that no exponentially growing terms
        appear for any formalism.

        Returns:
            (tuple):
            - *E (ndarray[complex])* - electric field of shape (M, N, Z, 3, 2).
            - *E_phi (ndarray[complex])* - for opposite magnetization or None.
            - *H (ndarray[complex])* - magnetic field :math:`Z_0 H`.
            - *H_phi (ndarray[complex])* - for opposite magnetization or None.

        """
        return self._calc_homogeneous_fields(
            distances,
            strains,
            magnetizations,
            temps,
            polar=polar,
            dichroic=dichroic,
            absorption=False,
        )[1]

    def homogeneous_absorption(
        self,
        distances=None,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
        dichroic=False,
        fields=False,
    ):
        r"""homogeneous_absorption

        Absorption profile :math:`\mathrm{d}A/\mathrm{d}z`, absorption of
        each layer, and flux into the substrate of the sample structure with
        homogeneous strain and magnetization for the incoming polarization,
        see :meth:`calc_homogeneous_absorption`.

        Args:
            distances (ndarray[float, Quantity], optional): depths below the
                surface (m). Defaults to the centers of the layers.
            strains (ndarray[float], optional): strain of each sub-structure.
            magnetizations (ndarray[float], optional): magnetization of each
                sub-structure, see :meth:`homogeneous_jones_matrices`.
            temps (ndarray[float], optional): temperature of each
                sub-structure, which raises a ``NotImplementedError`` yet.
            polar (bool, optional): the magnetization is given in polar
                coordinates. Defaults to False.
            dichroic (bool, optional): also for the opposite magnetization,
                which needs a second calculation for magnetic layers.
                Defaults to False.
            fields (bool, optional): also return the fields of
                :meth:`homogeneous_fields` at the ``distances``. Defaults to
                False.

        Returns:
            result (Result): ``absorption`` (1/m) of shape (M, N, Z) and its
            ``absorption_matrix`` of shape (M, N, Z, 2, 2), the absorbed
            fraction ``layer_absorption`` of each layer of shape (M, N, L)
            and its ``layer_absorption_matrix``, the ``substrate_flux`` of
            shape (M, N), i.e. the absorption of a semi-infinite absorbing
            substrate, and its ``substrate_flux_matrix``, and with the suffix
            ``_phi`` for the opposite magnetization.

        """
        distances, results = self._timed(
            "homogeneous_absorption",
            self._calc_homogeneous_fields,
            distances,
            strains,
            magnetizations,
            temps,
            polar=polar,
            dichroic=dichroic,
            absorption=True,
            fields=fields,
        )
        names = ABSORPTION_NAMES + (FIELD_NAMES if fields else ())
        return self._field_result(
            "homogeneous absorption",
            names,
            results,
            distances,
            {"strains": strains, "magnetizations": magnetizations, "temps": temps, "polar": polar},
        )

    def calc_homogeneous_absorption(
        self,
        distances=None,
        strains=np.array([]),
        magnetizations=np.array([]),
        temps=np.array([]),
        *,
        polar=False,
        dichroic=False,
        fields=False,
    ):
        r"""calc_homogeneous_absorption

        Plain arrays of :meth:`homogeneous_absorption`.

        The absorption is the decrease of the normal component of the
        Poynting vector :math:`S_z = \mathrm{Re}(E_y H_x^* - E_x H_y^*) / 2`
        of the fields of :meth:`calc_homogeneous_fields` into the sample,
        relative to the incident flux, see :cite:t:`passler2020` and
        :cite:t:`byrnes2016`. It is consistent with the reflectivity, except
        for rough interfaces, which do not conserve the flux. As :math:`S_z`
        is quadratic in the fields, the absorption for any incoming
        polarization follows from the hermitian absorption matrix :math:`K`,
        see :meth:`calc_absorption`. The absorption of each layer is the
        difference of :math:`S_z` at its top and bottom, and the flux into
        the substrate is :math:`S_z` at its top, which is absorbed for a
        semi-infinite absorbing substrate and transmitted otherwise.

        Returns:
            (tuple):
            - *A (ndarray[float])* - absorption profile (1/m) of shape
              (M, N, Z) or (M, N, Z, P) for P polarizations.
            - *A_phi (ndarray[float])* - for opposite magnetization or None.
            - *K (ndarray[complex])* - absorption matrix (1/m) of shape
              (M, N, Z, 2, 2).
            - *K_phi (ndarray[complex])* - for opposite magnetization or None.
            - *A_layers (ndarray[float])* - absorption of each layer of shape
              (M, N, L) or (M, N, L, P).
            - *A_layers_phi (ndarray[float])* - for opposite magnetization or
              None.
            - *K_layers (ndarray[complex])* - absorption matrix of each layer
              of shape (M, N, L, 2, 2).
            - *K_layers_phi (ndarray[complex])* - for opposite magnetization
              or None.
            - *S_sub (ndarray[float])* - flux into the substrate of shape
              (M, N) or (M, N, P).
            - *S_sub_phi (ndarray[float])* - for opposite magnetization or
              None.
            - *S_sub_matrix (ndarray[complex])* - flux matrix into the
              substrate of shape (M, N, 2, 2).
            - *S_sub_matrix_phi (ndarray[complex])* - for opposite
              magnetization or None.
            - *E, E_phi, H, H_phi (ndarray[complex])* - the fields of
              :meth:`calc_homogeneous_fields` for ``fields=True`` only.

        """
        return self._calc_homogeneous_fields(
            distances,
            strains,
            magnetizations,
            temps,
            polar=polar,
            dichroic=dichroic,
            absorption=True,
            fields=fields,
        )[1]

    def _calc_homogeneous_fields(
        self,
        distances,
        strains,
        magnetizations,
        temps,
        *,
        polar,
        dichroic,
        absorption,
        fields=False,
    ):
        """distances and the fields or, for ``absorption=True``, the absorptions
        and, for ``fields=True``, also the fields"""
        self._check_experiment()
        self._reject_temperatures(temps, "temps")
        if np.size(magnetizations) > 0:
            magnetizations = self.parse_magnetization_map(magnetizations, polar)
        layers = _flatten_layers(self._homogeneous_layer_tree(strains, magnetizations))
        tops, bottoms = self._layer_bounds(layers)
        if distances is None:
            distances = (tops + bottoms) / 2
        elif isinstance(distances, Q_):
            distances = distances.to("m").magnitude
        distances = np.atleast_1d(np.asarray(distances, dtype=float))
        if not absorption:
            stack = self._stack_fields(layers, distances, {}, dichroic)
            return distances, (stack["E"], stack.get("E_phi"), stack["H"], stack.get("H_phi"))
        # the flux at the top and bottom of each layer within the layer and at
        # the top of the substrate within the substrate
        Z, L = len(distances), len(tops)
        centers = (tops + bottoms) / 2
        substrate = bottoms[-1:] if L > 0 else np.zeros(1)
        stack = self._stack_fields(
            layers,
            np.concatenate([distances, tops, bottoms, substrate]),
            {},
            dichroic,
            z_slab=np.concatenate([distances, centers, centers, substrate]),
        )
        results = {}
        for suffix in ("", "_phi"):
            if "K" + suffix not in stack:
                results[suffix] = [None] * 6
                continue
            K, S = stack["K" + suffix][:, :, :Z], stack["S" + suffix]
            K_layers = S[:, :, Z : Z + L] - S[:, :, Z + L : Z + 2 * L]
            S_sub = S[:, :, -1]
            results[suffix] = [
                self.calc_absorption(K),
                K,
                self.calc_absorption(K_layers),
                K_layers,
                self.calc_absorption(S_sub),
                S_sub,
            ]
        # order of ABSORPTION_NAMES: each value followed by its opposite magnetization
        values = [value for pair in zip(results[""], results["_phi"]) for value in pair]
        if fields:
            values += [stack["E"][:, :, :Z], None, stack["H"][:, :, :Z], None]
            if "E_phi" in stack:
                values[-3], values[-1] = stack["E_phi"][:, :, :Z], stack["H_phi"][:, :, :Z]
        return distances, tuple(values)

    def _field_result(self, title, names, values, distances, inputs=None):
        """result of the fields or absorptions with the dimensions
        ``energy``, ``angles``, and ``distances`` or ``layers`` and the
        ``inputs`` of the method"""
        result = self.new_scattering_result(title, inputs=inputs)
        result.add_coord("distances", distances, "m")
        for name, value in zip(names, values):
            if value is None:
                # no opposite magnetization for dichroic=False
                continue
            if name.startswith("layer"):
                dims = ("energy", "angles", "layers")
            elif name.startswith("substrate"):
                dims = ("energy", "angles")
            else:
                dims = ("energy", "angles", "distances")
            result.add(name, value, dims, FIELD_UNITS.get(name, ""), ALIASES.get(name, ()))
        return result

    def _layer_bounds(self, layers):
        """tops and bottoms of the layers of the structure without the
        superstrate and the substrate (m) for the strains of the ``layers``"""
        num_substrate = len(_flatten_layers(self._superstrate_and_substrate()[1]))
        thickness = [
            layer.structural.thickness.magnitude
            * (1 + self._parse_layer_inputs(layer, strain, magnetization)[0])
            for layer, strain, magnetization in layers[1 : len(layers) - num_substrate]
        ]
        bottoms = np.cumsum(np.asarray(thickness, dtype=float))
        return bottoms - thickness, bottoms

    def calc_stack_fields(self, layers, distances, cache=None, dichroic=False):
        r"""calc_stack_fields

        Fields, Poynting flux, and absorption matrices of a stack of layers at
        the given depths for incident :math:`\sigma` and :math:`\pi`
        polarization of unit amplitude, see :meth:`calc_homogeneous_fields`.

        Args:
            layers (list[tuple]): layer, strain, and magnetization from the
                superstrate to the substrate, see :meth:`calc_homogeneous_layers`.
            distances (ndarray[float]): depths below the surface (m).
            cache (dict, optional): cache of the eigenmodes. Defaults to None.
            dichroic (bool, optional): also for the opposite magnetization.
                Defaults to False.

        Returns:
            fields (dict): ``E`` and ``H`` of shape (M, N, Z, 3, 2), the
            absorption matrices ``K`` (1/m), and the flux matrices ``S``
            relative to the incident flux of shape (M, N, Z, 2, 2), with the
            suffix ``_phi`` for the opposite magnetization.

        """
        return self._stack_fields(layers, distances, cache, dichroic)

    def _stack_fields(self, layers, distances, cache=None, dichroic=False, z_slab=None):
        """:meth:`calc_stack_fields`, where the slab of each depth is the one
        at ``z_slab``"""
        layers = _flatten_layers(layers)
        self._check_superstrate(layers[0])
        k = self.experiment.k.magnitude[:, np.newaxis]
        b = np.broadcast_to(self._in_plane_wavevector(), self.experiment.theta.magnitude.shape)
        z = np.atleast_1d(np.asarray(distances, dtype=float))
        magnetizations = [self._parse_layer_inputs(*entry)[1] for entry in layers]
        stacks = {"": layers}
        magnetic = self._magnetic_fields and any(np.any(m) for m in magnetizations)
        if dichroic and magnetic:
            # the opposite magnetization from the modes of the reversed
            # magnetizations, as the modes of calc_modes() for the opposite
            # magnetization are not given in the sample frame for all formalisms
            stacks["_phi"] = [
                (layer, strain, tuple(-m for m in magnetization))
                for (layer, strain, _), magnetization in zip(layers, magnetizations)
            ]
        result = {}
        for suffix, stack in stacks.items():
            slabs, interfaces = self._field_slabs(stack, cache)
            result.update(
                {
                    name + suffix: value
                    for name, value in self._slab_stack_fields(
                        slabs, interfaces, z, k, b, z_slab
                    ).items()
                }
            )
        if dichroic and not magnetic:
            for name in ("E", "H", "K", "S"):
                result[name + "_phi"] = result[name]
        return result

    def _slab_stack_fields(self, slabs, interfaces, z, k, b, z_slab=None):
        """fields, flux matrices, and absorption matrices at the depths ``z``
        of the slabs ``(A, q, thickness, eps)`` and their interface
        matrices, see :meth:`calc_stack_fields`, where the slab of each depth
        is the one at ``z_slab``, e.g. to evaluate the bottom of a layer in
        its own slab"""
        A = [slab[0] for slab in slabs]
        q = [slab[1] for slab in slabs]
        thickness = np.array([slab[2] for slab in slabs])
        tops = np.zeros(len(slabs))
        tops[1:] = np.concatenate([[0.0], np.cumsum(thickness[1:-1])])
        # depths at an interface belong to the slab below, also for the
        # rounding errors of the sums of the thicknesses
        z_slab = z if z_slab is None else z_slab
        z_tol = z_slab + 64 * np.finfo(float).eps * max(np.max(tops), np.max(np.abs(z_slab)))
        index = np.where(z_tol < 0, 0, np.searchsorted(tops[1:], z_tol, side="right"))
        frame = self._field_frame[:, np.newaxis]

        incident = self._incident_amplitudes(A[0], slabs[0][3], b)
        d_top, u_bot = _mode_amplitudes(interfaces, q, thickness, k, incident)
        psi = np.zeros(b.shape + (len(z), 4, 2), dtype=np.complex128)
        dpsi = np.zeros_like(psi)
        E = np.zeros(b.shape + (len(z), 3, 2), dtype=np.complex128)
        H = np.zeros_like(E)
        for j in np.unique(index):
            mask = index == j
            # no up modes in the semi-infinite substrate
            u_j = None if j == len(slabs) - 1 else u_bot[j]
            psi_j, dpsi_j = _slab_fields(
                A[j], q[j], k, z[mask] - tops[j], thickness[j], d_top[j], u_j
            )
            psi[:, :, mask], dpsi[:, :, mask] = psi_j * frame, dpsi_j * frame
            E[:, :, mask], H[:, :, mask] = _full_fields(psi_j * frame, b, slabs[j][3])
        # normalization to an incident electric field of unit amplitude and to
        # the incident flux
        psi_in = _slab_fields(A[0], q[0], k, np.zeros(1), 0.0, incident, None)[0] * frame
        E_in = _full_fields(psi_in, b, slabs[0][3])[0][:, :, 0]
        norm = np.sqrt(np.sum(np.abs(E_in) ** 2, axis=-2))
        psi_in = psi_in[:, :, 0] / norm[:, :, np.newaxis, :]
        flux_in = np.real(np.trace(_flux_matrix(psi_in, psi_in), axis1=-2, axis2=-1)) / 2
        flux_in = flux_in[:, :, np.newaxis, np.newaxis, np.newaxis]
        norm = norm[:, :, np.newaxis, np.newaxis, :]
        psi, dpsi = psi / norm, dpsi / norm
        return {
            "E": E / norm,
            "H": H / norm,
            "S": _flux_matrix(psi, psi) / flux_in,
            "K": -(_flux_matrix(dpsi, psi) + _flux_matrix(psi, dpsi)) / flux_in,
        }

    def _field_slabs(self, layers, cache=None):
        """eigenmodes ``(A, q, thickness, eps)`` of the homogeneous slabs of a
        flat list of layers from the superstrate to the substrate, where
        consecutive identical layers are combined, and the interface matrices
        :math:`X` from the amplitudes at the bottom of a slab to the ones at
        the top of the next slab including the roughness, see
        :meth:`_assemble`. The superstrate has no thickness."""
        k = self.experiment.k.magnitude[:, np.newaxis]
        slabs, interfaces, keys = [], [], []
        for i, entry in enumerate(layers):
            layer, strain, magnetization = entry
            key = (id(layer), *self._parse_layer_inputs(layer, strain, magnetization))
            A, _, q, _, k_z, thickness, roughness, eps = self._layer_modes(*entry, cache=cache)
            if i > 1 and key == keys[-1]:
                # identical layers have no interface and no roughness
                slabs[-1][2] += thickness
                continue
            if i > 0:
                X = np.linalg.solve(A, slabs[-1][0])
                if roughness > 0:
                    X = X * self.calc_roughness_matrix(roughness, k * k_z, k * slabs[-1][4])
                interfaces.append(X)
            slabs.append([A, q, 0.0 if i == 0 else thickness, eps, k_z])
            keys.append(key)
        return slabs, interfaces

    def _incident_amplitudes(self, A, eps, b):
        """amplitudes of the down modes of the superstrate for the incident
        sigma and pi polarization of shape (M, N, 2, 2), which follow from
        the conversion of the Jones matrices by :meth:`incident_field`"""
        identity = np.broadcast_to(np.eye(2, dtype=np.complex128), b.shape + (2, 2))
        incident = self.incident_field(identity, identity, eps)[1]
        return np.matmul(_LINEAR_TO_CIRCULAR, incident)

    # ---- inhomogeneous simulations ----------------------------------------------

    def inhomogeneous_jones_matrices(
        self,
        strain_map=np.array([]),
        magnetization_map=np.array([]),
        temp_map=np.array([]),
        *,
        polar=False,
        **kwargs,
    ):
        r"""inhomogeneous_jones_matrices

        Calculates the Jones matrices of the reflection and transmission of an
        inhomogeneously strained and magnetized sample structure for a given
        ``strain_map`` and ``magnetization_map`` in space and time for each
        layer of the sample structure, from which
        :meth:`calc_reflectivity_transmissivity` and
        :meth:`calc_rotation_ellipticity` follow. They are not saved to the
        cache, as they need much more memory than these.

        Args:
            strain_map (ndarray[float], Result, optional): spatio-temporal
                strain profile or result of
                :meth:`~udkm1Dsim.simulations.phonons.Phonon.get_strain_map`.
            magnetization_map (ndarray[float], Result, optional): spatio-temporal
                magnetization profile :math:`M \times N \times 3` with
                :math:`M` delays and :math:`N` layers. The magnetization is
                given in cartesian coordinates ``(x, y, z)`` by default, as
                returned by
                :meth:`~udkm1Dsim.simulations.magnetization.Magnetization.get_magnetization_map`.
                Its amplitude scales the magnetic part of the dielectric tensor.
                For its result, the metadata ``polar`` overrides ``polar``.
            temp_map (ndarray[float], Result, optional): spatio-temporal temperature
                profile :math:`M \times N` or :math:`M \times N \times K` of
                :meth:`~udkm1Dsim.simulations.heat.Heat.get_temp_map` with
                :math:`K` sub-systems, e.g. electrons and phonons. The
                temperature dependence of the dielectric tensor is planned for
                a future version, so that a given ``temp_map`` raises a
                ``NotImplementedError``.
            polar (bool, optional): the ``magnetization_map`` is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad,
                as before version 2.6.1. Defaults to False.
            **kwargs:
                - *calc_type (str)* - ``sequential`` (default) or ``parallel``
                  over blocks of delays utilizing `Dask <https://dask.org/>`_
                  on a local machine or a cluster, depending on the
                  ``dask_client``.
                - *dask_client (Dask.Client)* - Dask client.

        Returns:
            result (Result): with the Jones matrices ``jones_reflection``,
            ``jones_reflection_phi``, ``jones_transmission``, and
            ``jones_transmission_phi`` of the dimensions ``delays``,
            ``energy``, and ``angles`` and shape (D, M, N, 2, 2) for D
            delays, see :meth:`homogeneous_jones_matrices`.

        """
        strain_map, magnetization_map, temp_map, polar, source = self.parse_result_maps(
            strain_map, magnetization_map, temp_map, polar
        )
        input_maps = {
            "strain_map": strain_map,
            "magnetization_map": magnetization_map,
            "temp_map": temp_map,
        }
        strain_map, magnetization_map = self._parse_maps(
            strain_map, magnetization_map, temp_map, polar
        )
        results = self._timed(
            "inhomogeneous_jones_matrices",
            self._calc_inhomogeneous,
            strain_map,
            magnetization_map,
            None,
            **kwargs,
        )
        return self._scattering_result(
            "inhomogeneous Jones matrices",
            JONES_NAMES,
            results,
            inhomogeneous=True,
            source=source,
            inputs={"polar": polar},
            input_maps=input_maps,
        )

    def calc_inhomogeneous_jones_matrices(self, strain_map, magnetization_map, temp_map=None):
        """calc_inhomogeneous_jones_matrices

        Calculates the Jones matrices of the sample structure for a block of
        delays of the ``strain_map`` and ``magnetization_map``, see
        :meth:`calc_jones_matrices` for every delay. Formalisms, which
        calculate all delays at once, overwrite this method and
        :meth:`_delay_block_size`.

        Args:
            strain_map (ndarray[float]): strain of every layer for each delay.
            magnetization_map (ndarray[float]): magnetization of every layer
                for each delay.
            temp_map (ndarray[float], optional): temperature of every layer
                for each delay, which is reserved for the planned temperature
                dependence of the dielectric tensor. Defaults to None.

        Returns:
            (tuple):
            - *R (ndarray[complex])* - Jones matrix of the reflection of shape
              (D, M, N, 2, 2) for D delays.
            - *R_phi (ndarray[complex])* - the same Jones matrix for opposite
              magnetization.
            - *T (ndarray[complex])* - Jones matrix of the transmission.
            - *T_phi (ndarray[complex])* - the same Jones matrix for opposite
              magnetization.

        """
        results = [
            self.calc_jones_matrices(self.calc_inhomogeneous_layers(strains, magnetizations))
            for strains, magnetizations in zip(strain_map, magnetization_map)
        ]
        return tuple(np.stack(result) for result in zip(*results))

    def inhomogeneous_reflectivity_transmissivity(
        self,
        strain_map=np.array([]),
        magnetization_map=np.array([]),
        temp_map=np.array([]),
        *,
        polar=False,
        **kwargs,
    ):
        r"""inhomogeneous_reflectivity_transmissivity

        Returns the reflectivity and transmissivity of an inhomogeneously
        strained and magnetized sample structure for a given ``strain_map``
        and ``magnetization_map`` in space and time for each layer of the
        sample structure and the incoming and analyzer polarization. If no
        result is saved in the cache it is calculated, see
        :meth:`inhomogeneous_jones_matrices`.

        Args:
            strain_map (ndarray[float], Result, optional): spatio-temporal
                strain profile or result of
                :meth:`~udkm1Dsim.simulations.phonons.Phonon.get_strain_map`.
            magnetization_map (ndarray[float], Result, optional): spatio-temporal
                magnetization profile :math:`M \times N \times 3` with
                :math:`M` delays and :math:`N` layers. The magnetization is
                given in cartesian coordinates ``(x, y, z)`` by default, as
                returned by
                :meth:`~udkm1Dsim.simulations.magnetization.Magnetization.get_magnetization_map`.
                Its amplitude scales the magnetic part of the dielectric tensor.
                For its result, the metadata ``polar`` overrides ``polar``.
            temp_map (ndarray[float], Result, optional): spatio-temporal temperature
                profile :math:`M \times N` or :math:`M \times N \times K` of
                :meth:`~udkm1Dsim.simulations.heat.Heat.get_temp_map` with
                :math:`K` sub-systems, e.g. electrons and phonons. The
                temperature dependence of the dielectric tensor is planned for
                a future version, so that a given ``temp_map`` raises a
                ``NotImplementedError``.
            polar (bool, optional): the ``magnetization_map`` is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad,
                as before version 2.6.1. Defaults to False.
            **kwargs:
                - *calc_type (str)* - ``sequential`` (default) or ``parallel``
                  over blocks of delays utilizing `Dask <https://dask.org/>`_
                  on a local machine or a cluster, depending on the
                  ``dask_client``.
                - *dask_client (Dask.Client)* - Dask client.

        Returns:
            result (Result): with the ``reflectivity`` and ``transmissivity``,
            also for opposite magnetization ``reflectivity_phi`` and
            ``transmissivity_phi``, of the dimensions ``delays``, ``energy``, and ``angles``,
            whose ``delays`` are taken from a given upstream result, see
            :meth:`~udkm1Dsim.simulations.scattering.scattering.Scattering.new_scattering_result`.

        """
        return self._cached_inhomogeneous(
            "reflectivity_transmissivity",
            REFLECTIVITY_NAMES,
            "",
            MultilayerScattering._reduce_reflectivity_transmissivity,
            strain_map,
            magnetization_map,
            temp_map,
            polar,
            kwargs,
        )

    def calc_inhomogeneous_reflectivity_transmissivity(
        self,
        strain_map=np.array([]),
        magnetization_map=np.array([]),
        temp_map=np.array([]),
        *,
        polar=False,
        **kwargs,
    ):
        """calc_inhomogeneous_reflectivity_transmissivity

        Plain arrays of :meth:`inhomogeneous_reflectivity_transmissivity`,
        which are always calculated without the cache and the metadata of a
        :class:`~udkm1Dsim.simulations.result.Result`, e.g. for fits with many
        calls, see :meth:`inhomogeneous_reflectivity_transmissivity` for the
        arguments.

        Returns:
            (tuple):
            - *R (ndarray[float])* - reflectivity.
            - *R_phi (ndarray[float])* - reflectivity for opposite
              magnetization.
            - *T (ndarray[float])* - transmissivity.
            - *T_phi (ndarray[float])* - transmissivity for opposite
              magnetization.

        """
        strain_map, magnetization_map = self._parse_result_maps_to_arrays(
            strain_map, magnetization_map, temp_map, polar
        )
        return self._calc_inhomogeneous(
            strain_map,
            magnetization_map,
            MultilayerScattering._reduce_reflectivity_transmissivity,
            **kwargs,
        )

    def inhomogeneous_rotation_ellipticity(
        self,
        strain_map=np.array([]),
        magnetization_map=np.array([]),
        temp_map=np.array([]),
        *,
        polar=False,
        **kwargs,
    ):
        r"""inhomogeneous_rotation_ellipticity

        Returns the rotation and ellipticity of the polarization of the
        reflected and transmitted light, e.g. the time-resolved Kerr and
        Faraday effect, of an inhomogeneously strained and magnetized sample
        structure for a given ``strain_map`` and ``magnetization_map`` for the
        incoming :math:`\sigma` or :math:`\pi` polarization. If no result is
        saved in the cache it is calculated, see
        :meth:`inhomogeneous_jones_matrices`.

        Args:
            strain_map (ndarray[float], Result, optional): spatio-temporal
                strain profile or result of
                :meth:`~udkm1Dsim.simulations.phonons.Phonon.get_strain_map`.
            magnetization_map (ndarray[float], Result, optional): spatio-temporal
                magnetization profile :math:`M \times N \times 3` with
                :math:`M` delays and :math:`N` layers. The magnetization is
                given in cartesian coordinates ``(x, y, z)`` by default, as
                returned by
                :meth:`~udkm1Dsim.simulations.magnetization.Magnetization.get_magnetization_map`.
                Its amplitude scales the magnetic part of the dielectric tensor.
                For its result, the metadata ``polar`` overrides ``polar``.
            temp_map (ndarray[float], Result, optional): spatio-temporal temperature
                profile :math:`M \times N` or :math:`M \times N \times K` of
                :meth:`~udkm1Dsim.simulations.heat.Heat.get_temp_map` with
                :math:`K` sub-systems, e.g. electrons and phonons. The
                temperature dependence of the dielectric tensor is planned for
                a future version, so that a given ``temp_map`` raises a
                ``NotImplementedError``.
            polar (bool, optional): the ``magnetization_map`` is given in polar
                coordinates ``(amplitude, phi, gamma)`` with the angles in rad,
                as before version 2.6.1. Defaults to False.
            **kwargs:
                - *calc_type (str)* - ``sequential`` (default) or ``parallel``
                  over blocks of delays utilizing `Dask <https://dask.org/>`_
                  on a local machine or a cluster, depending on the
                  ``dask_client``.
                - *dask_client (Dask.Client)* - Dask client.

        Returns:
            result (Result): with the ``reflection_rotation`` and
            ``reflection_ellipticity``, e.g. the Kerr effect, and the
            ``transmission_rotation`` and ``transmission_ellipticity``, e.g.
            the Faraday effect, also for opposite magnetization with the
            suffix ``_phi``, in (deg) of the dimensions ``delays``, ``energy``, and ``angles``,
            whose ``delays`` are taken from a given upstream result, see
            :meth:`~udkm1Dsim.simulations.scattering.scattering.Scattering.new_scattering_result`.

        """
        self._pi_incident()
        return self._cached_inhomogeneous(
            "rotation_ellipticity",
            ROTATION_NAMES,
            "deg",
            MultilayerScattering._reduce_rotation_ellipticity,
            strain_map,
            magnetization_map,
            temp_map,
            polar,
            kwargs,
        )

    def calc_inhomogeneous_rotation_ellipticity(
        self,
        strain_map=np.array([]),
        magnetization_map=np.array([]),
        temp_map=np.array([]),
        *,
        polar=False,
        **kwargs,
    ):
        """calc_inhomogeneous_rotation_ellipticity

        Plain arrays of :meth:`inhomogeneous_rotation_ellipticity`, which are
        always calculated without the cache and the metadata of a
        :class:`~udkm1Dsim.simulations.result.Result`, e.g. for fits with many
        calls, see :meth:`inhomogeneous_rotation_ellipticity` for the
        arguments.

        Returns:
            (tuple): rotation and ellipticity (deg) of the reflection, of the
            reflection for opposite magnetization, of the transmission, and of
            the transmission for opposite magnetization.

        """
        self._pi_incident()
        strain_map, magnetization_map = self._parse_result_maps_to_arrays(
            strain_map, magnetization_map, temp_map, polar
        )
        return self._calc_inhomogeneous(
            strain_map,
            magnetization_map,
            MultilayerScattering._reduce_rotation_ellipticity,
            **kwargs,
        )

    # ---- input maps and results -------------------------------------------------

    @staticmethod
    def parse_magnetization_map(magnetization_map, polar=False):
        r"""parse_magnetization_map

        Returns the magnetization as plain cartesian array, which is used
        internally. The ``magnetization_map`` is given in cartesian
        coordinates ``(x, y, z)`` by default or in polar coordinates
        ``(amplitude, phi, gamma)`` with the angles in rad for ``polar=True``
        in its last dimension, see :meth:`inhomogeneous_jones_matrices`. Maps
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

    def parse_result_maps(self, strain_map, magnetization_map, temp_map, polar=False):
        """parse_result_maps

        The input maps given as arrays or as the
        :class:`~udkm1Dsim.simulations.result.Result` of the upstream
        simulations, e.g. of
        :meth:`~udkm1Dsim.simulations.magnetization.Magnetization.get_magnetization_map`,
        whose metadata ``polar`` overrides the given ``polar``.

        Args:
            strain_map (ndarray[float], Result): spatio-temporal strain
                profile.
            magnetization_map (ndarray[float], Result): spatio-temporal
                magnetization profile.
            temp_map (ndarray[float], Result): spatio-temporal temperature
                profile.
            polar (bool, optional): the ``magnetization_map`` array is given
                in polar coordinates. Defaults to False.

        Returns:
            (tuple):
            - *strain_map (ndarray[float])* - strain map.
            - *magnetization_map (ndarray[float])* - magnetization map.
            - *temp_map (ndarray[float])* - temperature map.
            - *polar (bool)* - the magnetization map is in polar coordinates.
            - *source (Result, None)* - first given result for its coordinates.

        """
        source = find_source(strain_map, magnetization_map, temp_map)
        if isinstance(magnetization_map, Result):
            polar = bool(magnetization_map.attrs.get("polar", polar))
        return (
            get_entry(strain_map, "strain_map"),
            get_entry(magnetization_map, "magnetization_map"),
            get_entry(temp_map, "temp_map"),
            polar,
            source,
        )

    def _parse_result_maps_to_arrays(self, strain_map, magnetization_map, temp_map, polar):
        """strain and cartesian magnetization map of arrays or results"""
        strain_map, magnetization_map, temp_map, polar, _ = self.parse_result_maps(
            strain_map, magnetization_map, temp_map, polar
        )
        return self._parse_maps(strain_map, magnetization_map, temp_map, polar)

    def _parse_maps(self, strain_map, magnetization_map, temp_map, polar):
        """strain and cartesian magnetization map with the same delays"""
        self._check_experiment()
        if not isinstance(strain_map, np.ndarray):
            raise TypeError("strain_map must be a numpy ndarray!")
        if not isinstance(magnetization_map, np.ndarray):
            raise TypeError("magnetization_map must be a numpy ndarray!")
        if not isinstance(temp_map, np.ndarray):
            raise TypeError("temp_map must be a numpy ndarray!")
        self._reject_temperatures(temp_map, "temp_map")
        magnetization_map = self.parse_magnetization_map(magnetization_map, polar)
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
        return strain_map, magnetization_map

    @staticmethod
    def _reject_temperatures(temps, name):
        """The temperature dependence of the dielectric tensor is planned, so
        that only empty temperatures are accepted yet."""
        if np.size(temps) > 0:
            raise NotImplementedError(
                "The temperature dependence of the dielectric tensor is planned for a future "
                f"version, so that the {name} of the multilayer formalisms must be empty!"
            )

    # ---- delay blocks, cache, and parallel calculation --------------------------

    def _delay_block_size(self):
        """number of delays, which are calculated at once"""
        return 1

    def _delay_blocks(self, M, dask_client=None):
        """slices of the blocks of delays of the size of
        :meth:`_delay_block_size`, which are split into at least one block
        per worker thread of the ``dask_client`` for the parallel
        calculation"""
        size = self._delay_block_size()
        if dask_client is not None:
            workers = max(1, sum(dask_client.nthreads().values()))
            size = max(1, min(size, -(-M // workers)))
        return [slice(i, min(i + size, M)) for i in range(0, M, size)]

    def _cached_inhomogeneous(
        self, name, names, unit, reduce, strain_map, magnetization_map, temp_map, polar, kwargs
    ):
        """Result of :meth:`_calc_inhomogeneous` with the ``names`` and
        ``unit``, which is saved to and loaded from the cache by its
        ``name``."""
        strain_map, magnetization_map, temp_map, polar, source = self.parse_result_maps(
            strain_map, magnetization_map, temp_map, polar
        )
        input_maps = {
            "strain_map": strain_map,
            "magnetization_map": magnetization_map,
            "temp_map": temp_map,
        }
        strain_map, magnetization_map = self._parse_maps(
            strain_map, magnetization_map, temp_map, polar
        )

        def calc():
            results = self._calc_inhomogeneous(strain_map, magnetization_map, reduce, **kwargs)
            title = f"inhomogeneous {name.replace('_', ' and ')}"
            return self._scattering_result(
                title,
                names,
                results,
                unit,
                True,
                source,
                inputs={"polar": polar},
                input_maps=input_maps,
            )

        return self._cached_result(
            f"inhomogeneous_{name}_{type(self).__name__}",
            self.get_hash(
                strain_map=strain_map, magnetization_map=magnetization_map, temp_map=temp_map
            ),
            calc,
        )

    def _calc_inhomogeneous(self, strain_map, magnetization_map, reduce, **kwargs):
        """Results of :meth:`calc_inhomogeneous_jones_matrices` for all blocks
        of delays, which are reduced by the static function ``reduce`` on the
        dask workers, e.g. to the reflectivity, or are the Jones matrices for
        ``reduce=None``, see :meth:`_delay_blocks`."""
        calc_type = kwargs.get("calc_type", "sequential")
        if calc_type not in ["parallel", "sequential"]:
            raise TypeError("calc_type must be either _parallel_ or _sequential_!")
        dask_client = kwargs.get("dask_client")
        if calc_type == "parallel" and not dask_client:
            raise ValueError("no dask client set")
        blocks = self._delay_blocks(
            np.size(strain_map, 0), dask_client if calc_type == "parallel" else None
        )
        if calc_type == "parallel":
            results = self._compute_in_parallel(
                dask_client,
                MultilayerScattering._parallel_inhomogeneous_block,
                [(reduce, strain_map[block, :], magnetization_map[block, :]) for block in blocks],
            )
        else:
            iterator = (
                tqdm(blocks, desc="Progress", leave=True) if self.progress_bar else iter(blocks)
            )
            results = [
                MultilayerScattering._parallel_inhomogeneous_block(
                    self, reduce, strain_map[block, :], magnetization_map[block, :]
                )
                for block in iterator
            ]
        return tuple(np.concatenate(parts) for parts in zip(*results))

    @staticmethod
    def _parallel_inhomogeneous_block(sim, reduce, strain_map, magnetization_map):
        """Results of a block of delays on a dask worker."""
        jones = sim.calc_inhomogeneous_jones_matrices(strain_map, magnetization_map)
        return jones if reduce is None else reduce(sim, jones)

    @staticmethod
    def _reduce_reflectivity_transmissivity(sim, jones):
        """reflectivity and transmissivity of a block of Jones matrices"""
        Ref, Ref_phi, Trans, Trans_phi = jones
        R, T = sim.calc_reflectivity_transmissivity(Ref, Trans)
        R_phi, T_phi = sim.calc_reflectivity_transmissivity(Ref_phi, Trans_phi)
        return R, R_phi, T, T_phi

    @staticmethod
    def _reduce_rotation_ellipticity(sim, jones):
        """rotation and ellipticity (deg) of a block of Jones matrices"""
        pi_incident = sim._pi_incident()
        result = []
        for j in jones:
            rotation, ellipticity = MultilayerScattering.calc_rotation_ellipticity(j, pi_incident)
            result += [rotation.to("deg").magnitude, ellipticity.to("deg").magnitude]
        return tuple(result)


def compose_scattering_stacks(stack, following):
    r"""compose_scattering_stacks

    Composes the scattering matrices
    :math:`S^{(1)} = (S^{(1)}_{tt}, S^{(1)}_{tr}, S^{(1)}_{rt}, S^{(1)}_{rr})`
    of the upper part of a stack with the ones :math:`S^{(2)}` of its lower
    part by Eqs. (63)-(67) of :cite:t:`stepanov2000`

    .. math::

        S_{tt} = X S^{(1)}_{tt}, \quad
        S_{tr} = S^{(2)}_{tr} + X S^{(1)}_{tr} S^{(2)}_{rr}, \quad
        S_{rt} = S^{(1)}_{rt} + Y S^{(2)}_{rt} S^{(1)}_{tt}, \quad
        S_{rr} = Y S^{(2)}_{rr}

    with :math:`X = S^{(2)}_{tt} (1 - S^{(1)}_{tr} S^{(2)}_{rt})^{-1}` and
    :math:`Y = S^{(1)}_{rr} (1 - S^{(2)}_{rt} S^{(1)}_{tr})^{-1}`, where
    :math:`t` and :math:`r` label the down- and up-going modes, so that
    :math:`S_{rt}` is the reflection and :math:`S_{tt}` the transmission in
    the basis of the modes of the superstrate and the last layer.

    Args:
        stack (tuple[ndarray[complex]]): scattering matrices :math:`S^{(1)}`
            of the upper part of shape (..., 2, 2).
        following (tuple[ndarray[complex]]): scattering matrices
            :math:`S^{(2)}` of the lower part.

    Returns:
        stack (tuple[ndarray[complex]]): scattering matrices :math:`S` of the
        combined stack.

    """
    S1_tt, S1_tr, S1_rt, S1_rr = stack
    S2_tt, S2_tr, S2_rt, S2_rr = following
    identity = np.eye(2, dtype=np.complex128)
    X = S2_tt @ np.linalg.inv(identity - S1_tr @ S2_rt)
    Y = S1_rr @ np.linalg.inv(identity - S2_rt @ S1_tr)
    return X @ S1_tt, S2_tr + X @ S1_tr @ S2_rr, S1_rt + Y @ S2_rt @ S1_tt, Y @ S2_rr


def _repeat_transfer_matrix(matrix, n):
    """Repeat a batched transfer matrix ``n`` times by integer powers."""
    return np.linalg.matrix_power(matrix, n)


def _scattering_stack_power(stack, n):
    """Raise a scattering stack to a positive integer power by binary
    powers, see :func:`compose_scattering_stacks`."""
    result = None
    while n > 0:
        if n & 1:
            result = stack if result is None else compose_scattering_stacks(result, stack)
        n >>= 1
        if n:
            stack = compose_scattering_stacks(stack, stack)
    return result


def _repeat_scattering_stack(stack, n):
    """Repeat a scattering stack ``n`` times by binary powers."""
    return _scattering_stack_power(stack, n)


# weights of the in-plane fields (E_x, E_y, Z_0 H_x, Z_0 H_y) for the normal
# component S_z = Re(E_y (Z_0 H_x)^* - E_x (Z_0 H_y)^*) / 2 of the Poynting
# vector into the sample, i.e. along -z of the sample frame, as bilinear form
_FLUX_WEIGHTS = (
    np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=np.complex128) / 4
)


def _mode_amplitudes(interfaces, q, thickness, k, incident):
    r"""Amplitudes of the down modes at the top and of the up modes at the
    bottom of all slabs for the given amplitudes of the incident down modes
    at the bottom of the superstrate.

    The reflection matrices :math:`\Gamma_j` of the stack below each slab,
    i.e. :math:`u_j = \Gamma_j d_j` at the bottom of the slab, follow from
    the substrate, where no up modes are incident, to the superstrate by

    .. math::

        \Gamma_{j-1} = (\Gamma'_j X_{tr} - X_{rr})^{-1}
        (X_{rt} - \Gamma'_j X_{tt}), \quad
        \Gamma'_j = P_u^{-1} \Gamma_j P_d

    with the blocks of the interface matrix :math:`X` from the amplitudes at
    the bottom of slab :math:`j-1` to the ones at the top of slab :math:`j`
    and the decaying propagations :math:`P_d` of the down and
    :math:`P_u^{-1}` of the up modes through slab :math:`j`. The down
    amplitudes then follow from the superstrate by
    :math:`d_j = (X_{tt} + X_{tr} \Gamma_{j-1}) P_d d_{j-1}`. All terms are
    bounded, also for thick and absorbing slabs.

    Args:
        interfaces (list[ndarray[complex]]): interface matrices of shape
            (M, N, 4, 4) above the slabs 1 to J-1.
        q (list[ndarray[complex]]): normal wave-vector components of the
            modes of the J slabs of shape (M, N, 4).
        thickness (ndarray[float]): thickness of the slabs (m).
        k (ndarray[float]): wavenumbers of shape (M, 1).
        incident (ndarray[complex]): amplitudes of the incident down modes of
            shape (M, N, 2, P).

    Returns:
        (tuple):
        - *d_top (list[ndarray[complex]])* - amplitudes of the down modes at
          the top of the slabs of shape (M, N, 2, P).
        - *u_bot (list[ndarray[complex]])* - amplitudes of the up modes at the
          bottom of the slabs.

    """
    J = len(q)
    k = k[..., np.newaxis]
    down = [np.exp(1j * k * q[j][..., :2] * thickness[j]) for j in range(J)]
    up_inv = [np.exp(-1j * k * q[j][..., 2:] * thickness[j]) for j in range(J)]
    gamma = [None] * J
    # no up modes are incident from the semi-infinite substrate
    gamma[-1] = np.zeros(q[-1].shape[:-1] + (2, 2), dtype=np.complex128)
    for j in range(J - 1, 0, -1):
        gamma_top = up_inv[j][..., :, np.newaxis] * gamma[j] * down[j][..., np.newaxis, :]
        X = interfaces[j - 1]
        Xtt, Xtr, Xrt, Xrr = X[..., :2, :2], X[..., :2, 2:], X[..., 2:, :2], X[..., 2:, 2:]
        gamma[j - 1] = np.linalg.solve(gamma_top @ Xtr - Xrr, Xrt - gamma_top @ Xtt)
    d_top = [incident]
    u_bot = [gamma[0] @ incident]
    d_bot = incident
    for j in range(1, J):
        X = interfaces[j - 1]
        d_top.append((X[..., :2, :2] + X[..., :2, 2:] @ gamma[j - 1]) @ d_bot)
        d_bot = down[j][..., :, np.newaxis] * d_top[j]
        u_bot.append(gamma[j] @ d_bot)
    return d_top, u_bot


def _slab_fields(A, q, k, zeta, thickness, d_top, u_bot):
    """in-plane fields of the modes of a slab and their derivatives along the
    depth of shape (M, N, Z, 4, P) at the depths ``zeta`` below its top for
    the amplitudes of its down modes at the top and of its up modes at the
    bottom, which are neglected for ``u_bot=None``, e.g. in the
    semi-infinite substrate"""
    k = k[..., np.newaxis, np.newaxis]
    zeta = zeta[:, np.newaxis]
    q = q[:, :, np.newaxis, :]
    down = np.exp(1j * k * q[..., :2] * zeta)[..., np.newaxis] * d_top[:, :, np.newaxis]
    if u_bot is None:
        up = np.zeros_like(down)
    else:
        up = np.exp(1j * k * q[..., 2:] * (zeta - thickness))[..., np.newaxis]
        up = up * u_bot[:, :, np.newaxis]
    amplitudes = np.concatenate([down, up], axis=-2)
    derivatives = 1j * k[..., np.newaxis] * q[..., np.newaxis] * amplitudes
    A = A[:, :, np.newaxis]
    return A @ amplitudes, A @ derivatives


def _full_fields(psi, b, eps):
    r"""electric field :math:`E` and magnetic field :math:`Z_0 H` of shape
    (M, N, Z, 3, P) in the sample frame from the in-plane fields ``psi`` of
    shape (M, N, Z, 4, P), the in-plane wave vector :math:`b = k_y / k` of
    shape (M, N), and the dielectric tensor of shape (M, 3, 3) by
    :math:`\varepsilon_{zx} E_x + \varepsilon_{zy} E_y + \varepsilon_{zz} E_z
    = b Z_0 H_x` and :math:`Z_0 H_z = -b E_x`"""
    b = b[:, :, np.newaxis, np.newaxis]
    eps = eps[:, np.newaxis, np.newaxis]
    E_x, E_y, H_x, H_y = psi[..., 0, :], psi[..., 1, :], psi[..., 2, :], psi[..., 3, :]
    E_z = (b * H_x - eps[..., 2, 0, np.newaxis] * E_x - eps[..., 2, 1, np.newaxis] * E_y) / eps[
        ..., 2, 2, np.newaxis
    ]
    E = np.stack([E_x, E_y, E_z], axis=-2)
    H = np.stack([H_x, H_y, -b * E_x], axis=-2)
    return E, H


def _flux_matrix(psi_1, psi_2):
    """hermitian bilinear form of the normal component of the Poynting
    vector into the sample of shape (..., P, P) of the in-plane fields of
    shape (..., 4, P), see :data:`_FLUX_WEIGHTS`"""
    return np.einsum("...ia,ij,...jb->...ab", psi_1.conj(), _FLUX_WEIGHTS, psi_2)
