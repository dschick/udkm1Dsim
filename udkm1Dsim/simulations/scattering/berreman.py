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

"""Exact eigenmodes of an arbitrary dielectric tensor following
:cite:t:`berreman1972`, which are combined with the transfer matrices of
:cite:t:`elzo2012` in :class:`BerremanElzo` and with the scattering-matrix
recursion of :cite:t:`stepanov2000` in
:class:`BerremanStepanov`, as in
`Dyna <https://gitlab.com/dynadevgroup/Dyna>`_."""

__all__ = ["BerremanElzo", "BerremanStepanov", "berreman_matrix", "berreman_modes"]

__docformat__ = "restructuredtext"

import numpy as np

from .multilayer import _CIRCULAR_TO_LINEAR, MultilayerScattering

# tolerance of the classification of the modes in units of the machine epsilon
_TOLERANCE = 256 * np.finfo(float).eps
# mirror M eps M with M = diag(1, 1, -1) of a tensor at the surface
_MIRROR = np.array([[1, 1, -1], [1, 1, -1], [-1, -1, 1]])


def berreman_matrix(eps, b):
    r"""berreman_matrix

    Berreman matrix :math:`\Delta` of a dielectric tensor :math:`\varepsilon`
    for the conserved in-plane wave-vector component :math:`b = k_y / k`,
    whose eigenvalues are the normal wave-vector components
    :math:`q = k_z / k` of the four eigenmodes with the in-plane fields
    :math:`\Psi = (E_x, E_y, H_x, H_y)`, i.e. :math:`\Delta \Psi = q \Psi`,
    see :cite:t:`berreman1972`.

    Args:
        eps (ndarray[complex]): dielectric tensor of shape (..., 3, 3).
        b (ndarray[float]): in-plane wave-vector component of shape (...).

    Returns:
        delta (ndarray[complex]): Berreman matrix of shape (..., 4, 4).

    """
    exx, exy, exz = eps[..., 0, 0], eps[..., 0, 1], eps[..., 0, 2]
    eyx, eyy, eyz = eps[..., 1, 0], eps[..., 1, 1], eps[..., 1, 2]
    ezx, ezy, ezz = eps[..., 2, 0], eps[..., 2, 1], eps[..., 2, 2]
    scale = np.max(np.abs(eps), axis=(-2, -1))
    if np.any(np.abs(ezz) <= _TOLERANCE * np.maximum(1, scale)):
        raise np.linalg.LinAlgError("The Berreman matrix requires a non-zero eps_zz!")
    b = np.asarray(b, dtype=np.complex128)
    delta = np.zeros(eps.shape[:-2] + (4, 4), dtype=np.complex128)
    delta[..., 0, 3] = 1.0
    delta[..., 1, 0] = -b * ezx / ezz
    delta[..., 1, 1] = -b * ezy / ezz
    delta[..., 1, 2] = b**2 / ezz - 1.0
    delta[..., 2, 0] = -eyx + eyz * ezx / ezz
    delta[..., 2, 1] = -eyy + eyz * ezy / ezz
    delta[..., 2, 2] = -b * eyz / ezz
    delta[..., 3, 0] = exx - b**2 - exz * ezx / ezz
    delta[..., 3, 1] = exy - exz * ezy / ezz
    delta[..., 3, 2] = b * exz / ezz
    return delta


def _forward_sqrt(value):
    """square root with a positive real part, or a positive imaginary part
    for a vanishing real part, i.e. of a mode propagating or decaying into
    the sample"""
    # the principal root has a non-negative real part, so that only roots
    # with a negative imaginary part and a vanishing real part are flipped
    root = np.sqrt(np.asarray(value, dtype=np.complex128))
    negative = root.imag < 0
    if not np.any(negative):
        return root
    tol = _TOLERANCE * np.maximum(1, np.abs(root))
    flip = negative & (np.abs(root.real) <= tol)
    return np.where(flip, -root, root)


def _is_isotropic(eps):
    """mask of isotropic dielectric tensors of shape (..., 3, 3)"""
    diagonal = np.trace(eps, axis1=-2, axis2=-1) / 3
    deviation = np.max(
        np.abs(eps - diagonal[..., np.newaxis, np.newaxis] * np.eye(3)), axis=(-2, -1)
    )
    scale = np.maximum(1, np.max(np.abs(eps), axis=(-2, -1)))
    return deviation <= _TOLERANCE * scale


def _isotropic_modes(eps, b):
    """circularly polarized modes of isotropic tensors in the basis of
    :func:`~udkm1Dsim.simulations.scattering.elzo.elzo_modes`"""
    q = _forward_sqrt(eps - b**2)
    one, zero = np.ones_like(q), np.zeros_like(q)
    linear = np.stack(
        [
            np.stack([one, zero, one, zero], axis=-1),
            np.stack([zero, -q, zero, q], axis=-1),
            np.stack([zero, eps, zero, eps], axis=-1),
            np.stack([q, zero, -q, zero], axis=-1),
        ],
        axis=-2,
    )
    basis = np.zeros((4, 4), dtype=np.complex128)
    basis[:2, :2] = _CIRCULAR_TO_LINEAR
    basis[2:, 2:] = _CIRCULAR_TO_LINEAR
    return linear @ basis, np.stack([q, q, -q, -q], axis=-1)


def _ordered_modes(values, vectors):
    """eigenvalues and eigenvectors of shape (K, 4) and (K, 4, 4) with the two
    down modes, i.e. with a positive energy flux along z, before the two up
    modes and normalized eigenvectors"""
    K = values.shape[0]
    # normal component of the Poynting vector of the in-plane fields
    flux = np.real(
        vectors[:, 0, :] * np.conj(vectors[:, 3, :]) - vectors[:, 1, :] * np.conj(vectors[:, 2, :])
    )
    flux_scale = np.maximum(1, np.max(np.abs(flux), axis=1, keepdims=True))
    value_scale = np.maximum(1, np.max(np.abs(values), axis=1, keepdims=True))
    metric = np.where(
        np.abs(flux) > _TOLERANCE * flux_scale,
        flux / flux_scale,
        np.where(
            np.abs(values.real) > _TOLERANCE * value_scale,
            values.real / value_scale,
            values.imag / value_scale,
        ),
    )
    ranked = np.argsort(metric, axis=1, kind="stable")
    rows = np.arange(K)[:, np.newaxis]
    down = ranked[:, 2:].copy()
    up = ranked[:, :2].copy()
    # down modes in ascending and up modes in descending order of their values
    v_down = values[rows, down]
    swap = (v_down[:, 0].real > v_down[:, 1].real) | (
        (v_down[:, 0].real == v_down[:, 1].real) & (v_down[:, 0].imag > v_down[:, 1].imag)
    )
    down[swap] = down[swap][:, ::-1]
    v_up = values[rows, up]
    swap = (v_up[:, 0].real < v_up[:, 1].real) | (
        (v_up[:, 0].real == v_up[:, 1].real) & (v_up[:, 0].imag < v_up[:, 1].imag)
    )
    up[swap] = up[swap][:, ::-1]
    order = np.concatenate([down, up], axis=1)
    values = values[rows, order]
    vectors = np.take_along_axis(vectors, order[:, np.newaxis, :], axis=2)
    # normalize to the largest component with a real and positive phase
    pivot = np.argmax(np.abs(vectors), axis=1)[:, np.newaxis, :]
    reference = np.take_along_axis(vectors, pivot, axis=1)
    if np.any(np.abs(reference) == 0):
        raise np.linalg.LinAlgError("A Berreman eigenvector has a zero norm!")
    vectors = vectors / np.abs(reference) * np.exp(-1j * np.angle(reference))
    return values, vectors


def berreman_modes(eps, cos_theta):
    r"""berreman_modes

    Exact eigenmodes of a dielectric tensor :math:`\varepsilon` from the
    eigenvalue problem of the Berreman matrix, see :func:`berreman_matrix`.
    The two down modes, i.e. with a positive energy flux into the sample,
    precede the two up modes. Isotropic tensors take the analytical
    circularly polarized modes of
    :func:`~udkm1Dsim.simulations.scattering.elzo.elzo_modes` to avoid the
    numerical ambiguity of their degenerate polarizations.

    Args:
        eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3) or
            (M, N, 3, 3).
        cos_theta (ndarray[float]): cosine of the grazing incidence angles
            :math:`\theta` of shape (M, N), i.e. the conserved in-plane
            wave-vector component :math:`b = k_y / k`.

    Returns:
        (tuple):
        - *A (ndarray[complex])* - boundary matrix of the in-plane fields of
          shape (M, N, 4, 4).
        - *q (ndarray[complex])* - normal wave-vector components
          :math:`k_{z,j} / k` of the modes of shape (M, N, 4).
        - *k_z (ndarray[complex])* - mean normal wave-vector component
          :math:`k_z / k` of the down modes of shape (M, N).

    """
    M, N = np.shape(cos_theta)
    eps = np.asarray(eps, dtype=np.complex128)
    if eps.ndim == 3:
        eps = np.broadcast_to(eps[:, np.newaxis, :, :], (M, N, 3, 3))
    b = np.asarray(cos_theta, dtype=float)
    A = np.empty((M, N, 4, 4), dtype=np.complex128)
    q = np.empty((M, N, 4), dtype=np.complex128)

    isotropic = _is_isotropic(eps)
    if np.any(isotropic):
        A[isotropic], q[isotropic] = _isotropic_modes(eps[isotropic][..., 0, 0], b[isotropic])
    general = ~isotropic
    if np.any(general):
        values, vectors = np.linalg.eig(berreman_matrix(eps[general], b[general]))
        q[general], A[general] = _ordered_modes(values, vectors)
    return A, q, np.mean(q[..., :2], axis=-1)


class _BerremanModes:
    """eigenmodes of the Berreman formalisms"""

    def calc_modes(self, eps):
        """calc_modes

        Calculates the exact eigenmodes of a homogeneous layer with the
        dielectric tensor ``eps`` for the given and the opposite
        magnetization, i.e. the transposed tensor, see :func:`berreman_modes`.
        The normal axis of the Berreman matrix points into the sample,
        opposite to the one of the sample frame of :cite:t:`elzo2012`, in which
        the magnetization and the dielectric tensor are given. Hence, the
        tensor is mirrored at the surface by :math:`M \\varepsilon M` with
        :math:`M = \\mathrm{diag}(1, 1, -1)`, which maps the magnetization
        :math:`(m_x, m_y, m_z)` to :math:`(-m_x, -m_y, m_z)`.

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
            - *k_z (ndarray[complex])* - mean normal wave-vector component
              of the down modes of shape (M, N).

        """
        cos_theta = self._in_plane_wavevector()
        # the normal axis of the Berreman matrix points into the sample,
        # opposite to the one of the sample frame of Elzo et al., so that the
        # tensor is mirrored at the surface
        eps = eps * _MIRROR
        A, q, k_z = berreman_modes(eps, cos_theta)
        # the opposite magnetization transposes the gyrotropic part
        eps_phi = np.swapaxes(eps, -1, -2)
        if np.array_equal(eps_phi, eps):
            return A, A, q, q, k_z
        A_phi, q_phi, k_z_phi = berreman_modes(eps_phi, cos_theta)
        # the mean is invariant under the reversal of the magnetization
        k_z = (k_z + k_z_phi) / 2
        return A, A_phi, q, q_phi, k_z

    @staticmethod
    def transmitted_field(T, eps):
        r"""transmitted_field

        Converts the Jones matrix of the transmission from the amplitudes of
        the eigenmodes of :func:`berreman_modes` to the amplitudes of the
        transmitted electric field for an isotropic last layer of the
        substrate, whose :math:`\pi` modes are normalized to the electric field
        divided by its refractive index :math:`n`. The transmission into an
        anisotropic last layer is given in the basis of its eigenmodes.

        Args:
            T (ndarray[complex]): Jones matrix of the transmission of the modes
                of shape (M, N, 2, 2).
            eps (ndarray[complex]): dielectric tensor of the last layer of
                shape (M, 3, 3).

        Returns:
            T (ndarray[complex]): Jones matrix of the transmitted electric
            field.

        """
        if not np.all(_is_isotropic(eps)):
            return T
        T = T.copy()
        T[..., 1, :] *= np.sqrt(eps[:, 0, 0])[:, np.newaxis, np.newaxis]
        return T

    @staticmethod
    def incident_field(R, T, eps):
        r"""incident_field

        Converts the Jones matrices of the reflection and transmission from the
        amplitudes of the eigenmodes of :func:`berreman_modes` in the isotropic
        superstrate to the amplitudes of its electric field. Its :math:`\pi`
        modes are normalized to the electric field divided by its refractive
        index :math:`n`, so that the reflected :math:`\pi` row of :math:`R`
        is multiplied and the incident :math:`\pi` columns of :math:`R` and
        :math:`T` are divided by :math:`n`.

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
        n = np.sqrt(eps[:, 0, 0])[:, np.newaxis, np.newaxis]
        R, T = R.copy(), T.copy()
        R[..., 1, :] *= n
        R[..., :, 1] /= n
        T[..., :, 1] /= n
        return R, T

    def _check_superstrate(self, entry):
        """the polarizations are defined in an isotropic superstrate"""
        layer, strain, magnetization = entry
        strain, magnetization = self._parse_layer_inputs(layer, strain, magnetization)
        if not np.all(
            _is_isotropic(self.calc_layer_dielectric_tensor(layer, strain, magnetization))
        ):
            raise ValueError(
                f"The superstrate '{layer.id}' of {type(self).__name__} must be isotropic!"
            )


class BerremanElzo(_BerremanModes, MultilayerScattering):
    r"""BerremanElzo

    Multilayer scattering simulations with the exact eigenmodes of an
    arbitrary dielectric tensor of every layer following
    :cite:t:`berreman1972`, e.g. of anisotropic, birefringent, or strongly
    magnetic layers, which are assembled by the transfer matrices of
    :cite:t:`elzo2012`. Ported from `Dyna <https://gitlab.com/dynadevgroup/Dyna>`_.

    The superstrate must be isotropic to define the :math:`\sigma` and
    :math:`\pi` polarizations. For an anisotropic substrate, the
    transmission is given in the basis of its eigenmodes. Use
    :class:`BerremanStepanov` for
    thick or strongly absorbing stacks.

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

    _title = "Berreman-Elzo multilayer scattering simulation"
    _assembly = "transfer"


class BerremanStepanov(_BerremanModes, MultilayerScattering):
    r"""BerremanStepanov

    Multilayer scattering simulations with the exact eigenmodes of an
    arbitrary dielectric tensor of every layer following
    :cite:t:`berreman1972`, e.g. of anisotropic, birefringent, or strongly
    magnetic layers, which are assembled by the scattering-matrix recursion
    of :cite:t:`stepanov2000`. It is the most general and stable, but also the
    slowest formalism and serves as reference for the faster ones. Ported from
    `Dyna <https://gitlab.com/dynadevgroup/Dyna>`_.

    The superstrate must be isotropic to define the :math:`\sigma` and
    :math:`\pi` polarizations. For an anisotropic substrate, the
    transmission is given in the basis of its eigenmodes.

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

    _title = "Berreman-Stepanov multilayer scattering simulation"
    _assembly = "scattering"
