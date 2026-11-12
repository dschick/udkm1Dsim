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

"""Dynamical magnetic scattering by the formalism of Elzo et al."""

__all__ = ["Elzo", "elzo_modes"]

__docformat__ = "restructuredtext"


import numpy as np
import pint

from .multilayer import MultilayerScattering

u = pint.get_application_registry()
Q_ = u.Quantity


class _ElzoModes:
    """Analytical eigenmodes of :cite:t:`elzo2012` for the formalisms of
    :class:`Elzo` and
    :class:`~udkm1Dsim.simulations.scattering.stepanov.Stepanov`."""

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
        return elzo_modes(eps, self._in_plane_wavevector())

    def calc_layer_dielectric_tensor(self, layer, strain=0.0, magnetization=None):
        """calc_layer_dielectric_tensor

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
                f"but the layer '{layer.id}' is anisotropic! Use BerremanElzo or "
                "BerremanStepanov instead."
            )
        return layer.get_dielectric_tensor(self.experiment.energy.magnitude, strain, magnetization)


class Elzo(_ElzoModes, MultilayerScattering):
    r"""Elzo

    Dynamical magnetic scattering simulations by the formalism of
    :cite:t:`elzo2012` with the analytical eigenmodes of a dielectric tensor,
    which is isotropic except for its gyrotropic part to first order, and the
    assembly of the stack by transfer matrices.
    A :class:`.UnitCell` is treated as homogeneous layer with its dielectric
    tensor as all other layers, e.g. from the average of its atoms by
    :meth:`.Layer.set_dielectric_from_atoms`, so that it does not show Bragg
    peaks, see :class:`~udkm1Dsim.simulations.scattering.dynamical.XrayDyn`.

    .. deprecated:: 3.0.0
        The former name ``XrayDynMag`` was removed, use :class:`Elzo` instead.

    Adapted from :cite:t:`elzo2012` and initially realized in `Project Dyna
    <http://dyna.neel.cnrs.fr>`_.

    Original copyright notice:

    *Copyright Institut Neel, CNRS, Grenoble, France*

    **Project Collaborators:**

    - Stéphane Grenier, stephane.grenier@neel.cnrs.fr
    - Marta Elzo (PhD, 2009-2012)
    - Nicolas Jaouen Sextants beamline, Synchrotron Soleil,
      nicolas.jaouen@synchrotron-soleil.fr
    - Emmanuelle Jal (PhD, 2010-2013) now at `LCPMR CNRS, Paris
      <https://lcpmr.cnrs.fr/content/emmanuelle-jal>`_
    - Jean-Marc Tonnerre, jean-marc.tonnerre@neel.cnrs.fr
    - Ingrid Hallsteinsen - Padraic Shaffer’s group - Berkeley Nat. Lab.

    **Questions to:**

    - Stéphane Grenier, stephane.grenier@neel.cnrs.fr

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
        theta (ndarray[float]): incidence angles :math:`\theta` of scattering
            light
        qz (ndarray[float]): scattering vector :math:`q_z` of scattering light
        polarizations (dict): polarization states and according names.
        pol_in_state (int): incoming polarization state as defined in
            polarizations dict.
        pol_out_state (int): outgoing polarization state as defined in
            polarizations dict.
        pol_in (float): incoming polarization factor (can be a complex ndarray).
        pol_out (float): outgoing polarization factor (can be a complex ndarray).

    """

    _title = "Elzo multilayer scattering simulation"
    _assembly = "transfer"


def elzo_modes(eps, cos_theta):
    r"""elzo_modes

    Analytical eigenmodes of a dielectric tensor :math:`\varepsilon`, which is
    isotropic except for its gyrotropic part, from :cite:t:`elzo2012`. The four
    circularly polarized modes are ordered as right and left down, i.e. into
    the sample, and right and left up. The boundary matrix holds the in-plane
    electric and magnetic fields of the modes.

    The inverse permittivity is expanded to first order in the off-diagonal
    part :math:`\varepsilon_A` of :math:`\varepsilon = \varepsilon_{00} I +
    \varepsilon_A`, i.e. :math:`\varepsilon^{-1} = (I - \varepsilon_A /
    \varepsilon_{00}) / \varepsilon_{00}`. Eq. (20) of :cite:t:`elzo2012`
    omits the division by :math:`\varepsilon_{00}`, which is exact only for
    :math:`\varepsilon_{00} = 1`, as found and corrected in Dyna after version 6.13.1.

    Args:
        eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3) or
            (M, N, 3, 3).
        cos_theta (ndarray[float]): cosine of the grazing incidence angles
            :math:`\theta` of shape (M, N), i.e. the conserved in-plane
            wave-vector component :math:`k_y / k`.

    Returns:
        (tuple):
        - *A (ndarray[complex])* - boundary matrix of shape (M, N, 4, 4).
        - *A_phi (ndarray[complex])* - boundary matrix for opposite
          magnetization.
        - *q (ndarray[complex])* - normal wave-vector components
          :math:`k_{z,j} / k` of the modes of shape (M, N, 4).
        - *q_phi (ndarray[complex])* - normal wave-vector components for
          opposite magnetization.
        - *k_z (ndarray[complex])* - normal wave-vector component
          :math:`k_z / k` of the non-magnetic slab of shape (M, N).

    """
    M, N = np.shape(cos_theta)
    if eps.ndim == 3:
        # the dielectric tensor does not depend on q_z
        eps = np.broadcast_to(eps[:, np.newaxis, :, :], (M, N, 3, 3))
    e00 = eps[:, :, 0, 0]
    e01 = eps[:, :, 0, 1]
    e02 = eps[:, :, 0, 2]

    alpha_y = np.divide(cos_theta, np.sqrt(e00))
    alpha_z = np.sqrt(1 - alpha_y**2)
    k_z = np.sqrt(e00) * alpha_z

    n_rd = np.sqrt(e00 - 1j * e02 * alpha_y - 1j * e01 * alpha_z)
    n_ld = np.sqrt(e00 + 1j * e02 * alpha_y + 1j * e01 * alpha_z)
    n_ru = np.sqrt(e00 - 1j * e02 * alpha_y + 1j * e01 * alpha_z)
    n_lu = np.sqrt(e00 + 1j * e02 * alpha_y - 1j * e01 * alpha_z)

    ay_rd = cos_theta / n_rd
    az_rd = np.sqrt(1 - ay_rd**2)
    ay_ld = cos_theta / n_ld
    az_ld = np.sqrt(1 - ay_ld**2)
    ay_ru = cos_theta / n_ru
    az_ru = np.sqrt(1 - ay_ru**2)
    ay_lu = cos_theta / n_lu
    az_lu = np.sqrt(1 - ay_lu**2)

    # first-order inverse permittivity, see above
    e01 = e01 / e00
    e02 = e02 / e00
    e12 = eps[:, :, 1, 2] / e00

    A = np.zeros([M, N, 4, 4], dtype=np.complex128)
    A_phi = np.zeros_like(A)

    A[:, :, 0, 0] = -1 - 1j * e01 * az_rd - 1j * e02 * ay_rd
    A[:, :, 0, 1] = 1 - 1j * e01 * az_ld - 1j * e02 * ay_ld
    A[:, :, 0, 2] = -1 + 1j * e01 * az_ru - 1j * e02 * ay_ru
    A[:, :, 0, 3] = 1 + 1j * e01 * az_lu - 1j * e02 * ay_lu

    A[:, :, 1, 0] = 1j * az_rd - e01 - 1j * e12 * ay_rd
    A[:, :, 1, 1] = 1j * az_ld + e01 - 1j * e12 * ay_ld
    A[:, :, 1, 2] = -1j * az_ru - e01 - 1j * e12 * ay_ru
    A[:, :, 1, 3] = -1j * az_lu + e01 - 1j * e12 * ay_lu

    A[:, :, 2, 0] = -1j * n_rd * A[:, :, 0, 0]
    A[:, :, 2, 1] = 1j * n_ld * A[:, :, 0, 1]
    A[:, :, 2, 2] = -1j * n_ru * A[:, :, 0, 2]
    A[:, :, 2, 3] = 1j * n_lu * A[:, :, 0, 3]

    A[:, :, 3, 0] = -az_rd * n_rd * A[:, :, 0, 0]
    A[:, :, 3, 1] = -az_ld * n_ld * A[:, :, 0, 1]
    A[:, :, 3, 2] = az_ru * n_ru * A[:, :, 0, 2]
    A[:, :, 3, 3] = az_lu * n_lu * A[:, :, 0, 3]

    A_phi[:, :, 0, 0] = -1 + 1j * e01 * az_ld + 1j * e02 * ay_ld
    A_phi[:, :, 0, 1] = 1 + 1j * e01 * az_rd + 1j * e02 * ay_rd
    A_phi[:, :, 0, 2] = -1 - 1j * e01 * az_lu + 1j * e02 * ay_lu
    A_phi[:, :, 0, 3] = 1 - 1j * e01 * az_ru + 1j * e02 * ay_ru

    A_phi[:, :, 1, 0] = 1j * az_ld + e01 + 1j * e12 * ay_ld
    A_phi[:, :, 1, 1] = 1j * az_rd - e01 + 1j * e12 * ay_rd
    A_phi[:, :, 1, 2] = -1j * az_lu + e01 + 1j * e12 * ay_lu
    A_phi[:, :, 1, 3] = -1j * az_ru - e01 + 1j * e12 * ay_ru

    A_phi[:, :, 2, 0] = 1j * n_ld * A_phi[:, :, 0, 0]
    A_phi[:, :, 2, 1] = -1j * n_rd * A_phi[:, :, 0, 1]
    A_phi[:, :, 2, 2] = 1j * n_lu * A_phi[:, :, 0, 2]
    A_phi[:, :, 2, 3] = -1j * n_ru * A_phi[:, :, 0, 3]

    A_phi[:, :, 3, 0] = -az_ld * n_ld * A_phi[:, :, 0, 0]
    A_phi[:, :, 3, 1] = -az_rd * n_rd * A_phi[:, :, 0, 1]
    A_phi[:, :, 3, 2] = az_lu * n_lu * A_phi[:, :, 0, 2]
    A_phi[:, :, 3, 3] = az_ru * n_ru * A_phi[:, :, 0, 3]

    norm = (np.sqrt(2) * e00)[:, :, np.newaxis, np.newaxis]
    A /= norm
    A_phi /= norm

    q = np.stack((n_rd * az_rd, n_ld * az_ld, -n_ru * az_ru, -n_lu * az_lu), axis=-1)
    # the reversed magnetization swaps the right and left modes
    q_phi = q[..., [1, 0, 3, 2]]
    return A, A_phi, q, q_phi, k_z
