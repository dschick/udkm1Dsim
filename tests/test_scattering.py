#!/usr/bin/env python

import copy
import warnings

import numpy as np
import pytest
from pint.testing import assert_allclose as assert_quantity_allclose
from scipy.integrate import simpson

import udkm1Dsim as ud
from tests.conftest import N_SUB, interface, magnetic_film, optical_stack, substrate_layer
from udkm1Dsim import (
    Atom,
    BerremanElzo,
    BerremanStepanov,
    Dielectric,
    Dispersion,
    Elzo,
    Layer,
    Parratt,
    Stepanov,
    Structure,
    Vacuum,
    u,
)
from udkm1Dsim.helpers import m_power_x
from udkm1Dsim.simulations.result import Result, array_hash
from udkm1Dsim.simulations.scattering.berreman import berreman_modes
from udkm1Dsim.simulations.scattering.elzo import elzo_modes
from udkm1Dsim.simulations.scattering.multilayer import (
    JONES_NAMES,
    REFLECTIVITY_NAMES,
    ROTATION_NAMES,
    MultilayerScattering,
    _flatten_layers,
    _layer_runs,
    _Period,
)

FORMALISMS = [Parratt, Elzo, Stepanov, BerremanElzo, BerremanStepanov]
MAGNETIC = [Elzo, Stepanov, BerremanElzo, BerremanStepanov]


def simulation(cls, S, energy, theta):
    sim = cls(S, True, disp_messages=False, progress_bar=False, save_data=False)
    sim.energy = energy
    sim.theta = theta
    return sim


def relative(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# ============================================================================
# package
# ============================================================================


def test_former_names():
    from udkm1Dsim.simulations import xrays

    for name in ["XrayDynMag", "Xray"]:
        assert not hasattr(ud, name)
        assert not hasattr(xrays, name)
    assert xrays.XrayKin is ud.XrayKin
    for cls in FORMALISMS:
        assert issubclass(cls, MultilayerScattering)


def test_titles():
    S = interface(substrate_layer())
    for cls, title in [
        (Parratt, "Parratt"),
        (Elzo, "Elzo"),
        (Stepanov, "Stepanov"),
        (BerremanElzo, "Berreman-Elzo"),
        (BerremanStepanov, "Berreman-Stepanov"),
    ]:
        sim = simulation(cls, S, 1.55 * u.eV, np.r_[10] * u.deg)
        assert repr(sim).startswith(f"{title} multilayer scattering simulation properties:")
        assert str(sim) == repr(sim)
        assert sim._repr_html_().startswith(f"<h2>{title} multilayer scattering simulation</h2>")


# ============================================================================
# single interface
# ============================================================================


@pytest.mark.parametrize("cls", FORMALISMS)
def test_fresnel_equations(cls):
    """reflection and transmission of the electric field at an interface"""
    theta = np.r_[10, 30, 60, 85]
    sim = simulation(cls, interface(substrate_layer()), 1.55 * u.eV, theta * u.deg)
    res = sim.homogeneous_jones_matrices()
    R, R_phi, T, T_phi = (
        res.jones_reflection,
        res.jones_reflection_phi,
        res.jones_transmission,
        res.jones_transmission_phi,
    )
    eps = N_SUB**2
    b = np.cos(np.radians(theta))
    kz0, kz1 = np.sin(np.radians(theta)), np.sqrt(eps - b**2)
    r_s = (kz0 - kz1) / (kz0 + kz1)
    r_p = (eps * kz0 - kz1) / (eps * kz0 + kz1)
    np.testing.assert_allclose(R[0, :, 0, 0], r_s, atol=1e-12)
    np.testing.assert_allclose(R[0, :, 1, 1], r_p, atol=1e-12)
    np.testing.assert_allclose(T[0, :, 0, 0], 1 + r_s, atol=1e-12)
    np.testing.assert_allclose(T[0, :, 1, 1], (1 + r_p) / N_SUB, atol=1e-12)
    np.testing.assert_allclose(R[..., 0, 1], 0, atol=1e-14)
    np.testing.assert_allclose(R_phi, R, atol=1e-14)
    np.testing.assert_allclose(T_phi, T, atol=1e-14)


@pytest.mark.parametrize("cls", FORMALISMS)
def test_fresnel_equations_dielectric_superstrate(cls):
    """reflection and transmission of the electric field at an interface of a
    dielectric superstrate and the substrate for the grazing angles in the
    superstrate, also for in-plane wave vectors n cos(theta) > 1 beyond the
    vacuum, as for the coupling by a prism"""
    n_sup = 1.3
    superstrate = substrate_layer(Dispersion.from_ref_index(n_sup))
    S = Structure("interface")
    S.add_superstrate(superstrate)
    S.add_sub_structure(substrate_layer(Dispersion.from_ref_index(n_sup)), 1)
    S.add_substrate(substrate_layer())
    theta = np.r_[10, 30, 60, 85]
    sim = simulation(cls, S, 1.55 * u.eV, theta * u.deg)
    res = sim.homogeneous_jones_matrices()
    R, T = res.jones_reflection, res.jones_transmission
    eps0, eps1 = n_sup**2, N_SUB**2
    b = n_sup * np.cos(np.radians(theta))
    assert np.any(b > 1)
    kz0, kz1 = np.sqrt(eps0 - b**2), np.sqrt(eps1 - b**2)
    r_s = (kz0 - kz1) / (kz0 + kz1)
    r_p = (eps1 * kz0 - eps0 * kz1) / (eps1 * kz0 + eps0 * kz1)
    np.testing.assert_allclose(R[0, :, 0, 0], r_s, atol=1e-12)
    np.testing.assert_allclose(R[0, :, 1, 1], r_p, atol=1e-12)
    np.testing.assert_allclose(T[0, :, 0, 0], 1 + r_s, atol=1e-12)
    np.testing.assert_allclose(T[0, :, 1, 1], (1 + r_p) * n_sup / N_SUB, atol=1e-12)


@pytest.mark.parametrize("cls", MAGNETIC)
def test_polar_kerr_effect_dielectric_superstrate(cls):
    """polar Kerr effect of a semi-infinite magnet at normal incidence from a
    dielectric superstrate by its circularly polarized modes"""
    n_sup, eps_xx, eps_xy = 1.76, -8 + 27j, -0.6 + 0.2j
    magnet = substrate_layer(Dispersion.from_array(None, eps_xx))
    magnet.optical.dielectric = Dielectric(
        eps0=Dispersion.from_array(None, eps_xx),
        eps_mag=Dispersion.from_array(None, 1j * eps_xy),
    )
    magnet.magnetization = (0, 0, 1)
    S = Structure("interface")
    S.add_superstrate(substrate_layer(Dispersion.from_ref_index(n_sup)))
    S.add_sub_structure(substrate_layer(Dispersion.from_ref_index(n_sup)), 1)
    S.add_substrate(magnet)
    sim = simulation(cls, S, 1.96 * u.eV, np.r_[90] * u.deg)
    R = sim.homogeneous_jones_matrices().jones_reflection[0, 0]
    r = [(n_sup - n) / (n_sup + n) for n in np.sqrt(eps_xx + np.r_[1, -1] * 1j * eps_xy)]
    chi = abs((r[0] - r[1]) / (r[0] + r[1]))
    assert abs(R[1, 0] / R[0, 0]) == pytest.approx(chi, rel=1e-10)
    assert abs(R[0, 1] / R[1, 1]) == pytest.approx(chi, rel=1e-10)


def test_parratt_sublattices():
    """Parratt neglects the magnetization also of a layer with several
    sublattices, as do the magnetic formalisms without gyrotropic part, while
    they require the coupling of the gyrotropic part to the sublattices"""
    layer = ud.Layer(
        "ferri",
        "ferrimagnet",
        structural=ud.StructuralParameters(thickness=10 * u.nm, density=8000 * u.kg / u.m**3),
        optical=ud.OpticalParameters(dielectric=Dispersion.from_ref_index(3.0 + 3.5j)),
        magnetic=ud.MagneticParameters(magnetization=[(0, 0, 1), (0, 0, -1)]),
    )
    S = Structure("ferrimagnet")
    S.add_sub_structure(layer, 2)
    S.add_substrate(Vacuum())
    energy, theta = np.r_[1.55] * u.eV, np.r_[45] * u.deg
    sim = simulation(Parratt, S, energy, theta)
    absorption = sim.homogeneous_absorption()
    R = sim.homogeneous_reflectivity_transmissivity().R
    np.testing.assert_allclose(R + absorption.A_layers.sum(axis=-1) + absorption.S_sub, 1)
    elzo = simulation(Elzo, S, energy, theta)
    np.testing.assert_allclose(elzo.homogeneous_reflectivity_transmissivity().R, R)
    layer.optical.dielectric.eps_mag = Dispersion.from_array(None, 0.1, kind="eps")
    with pytest.raises(ValueError, match="Layer 'ferri'.*gyrotropy"):
        elzo.homogeneous_absorption()


def test_parratt_roughness():
    """roughness of the interface by the factor of Névot and Croce"""
    theta = np.r_[10, 30, 60]
    smooth = simulation(Parratt, interface(substrate_layer()), 1.55 * u.eV, theta * u.deg)
    sub = substrate_layer()
    sub.structural.roughness = 20 * u.nm
    rough = simulation(Parratt, interface(sub), 1.55 * u.eV, theta * u.deg)
    k = rough.experiment.k.magnitude[0]
    kz0 = k * np.sin(np.radians(theta))
    kz1 = k * np.sqrt(N_SUB**2 - np.cos(np.radians(theta)) ** 2)
    np.testing.assert_allclose(
        rough.homogeneous_jones_matrices().jones_reflection[0, :, 0, 0],
        smooth.homogeneous_jones_matrices().jones_reflection[0, :, 0, 0]
        * np.exp(-2 * kz0 * kz1 * (20e-9) ** 2),
    )


def test_rough_stacks_of_formalisms_agree():
    """Parratt has the transmission coefficients of the roughness matrices
    of the other formalisms, so that all agree for rough isotropic stacks"""
    S = optical_stack()
    for layer in S.unique_layers[1]:
        layer.structural.roughness = 5 * u.nm
    stack = Structure("periodic")
    stack.add_sub_structure(S, 3)
    sub = substrate_layer()
    sub.structural.roughness = 3 * u.nm
    stack.add_substrate(sub)
    results = []
    for cls in FORMALISMS:
        sim = simulation(cls, stack, np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg)
        jones = sim.homogeneous_jones_matrices()
        absorption = sim.homogeneous_absorption().absorption_matrix
        results.append((jones.jones_reflection, jones.jones_transmission, absorption))
    for other in results[1:]:
        for value, reference in zip(other, results[0]):
            assert relative(value, reference) < 1e-10


# ============================================================================
# comparison of the formalisms
# ============================================================================


@pytest.mark.parametrize("cls", [Parratt, Elzo, Stepanov, BerremanElzo])
@pytest.mark.parametrize("vacuum_substrate", [True, False])
def test_non_magnetic_stacks_equal_berreman_stepanov(cls, vacuum_substrate):
    """all formalisms give the same Jones matrices as the exact modes and
    stable assembly of BerremanStepanov for isotropic and non-magnetic layers
    in the optical range"""
    substrate = None if vacuum_substrate else substrate_layer()
    S = optical_stack(substrate)
    energy, theta = np.r_[1.55, 3.1] * u.eV, np.r_[5:85:10] * u.deg
    reference = simulation(BerremanStepanov, S, energy, theta).homogeneous_jones_matrices()
    result = simulation(cls, S, energy, theta).homogeneous_jones_matrices()
    for a, b in zip(result.values(), reference.values()):
        assert relative(a, b) < 1e-12


def test_parratt_x_rays_equal_berreman_stepanov(magnetic_multilayer):
    """Parratt equals BerremanStepanov for the non-magnetic X-ray multilayer
    for all polarizations"""
    energy, theta = np.r_[700, 707] * u.eV, np.r_[0.5:30:2] * u.deg
    parratt = simulation(Parratt, magnetic_multilayer, energy, theta)
    reference = simulation(BerremanStepanov, magnetic_multilayer, energy, theta)
    L = magnetic_multilayer.num_sub_structures
    no_magnetization = np.zeros([L, 3])
    for pol_in, pol_out in [(0, 0), (1, 0), (3, 0), (4, 0), (1, 2), (3, 4)]:
        parratt.set_polarization(pol_in, pol_out)
        reference.set_polarization(pol_in, pol_out)
        for a, b in zip(
            parratt.homogeneous_reflectivity_transmissivity(
                np.zeros(L), no_magnetization
            ).values(),
            reference.homogeneous_reflectivity_transmissivity(
                np.zeros(L), no_magnetization
            ).values(),
        ):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize(
    "transfer, scattering", [(Elzo, Stepanov), (BerremanElzo, BerremanStepanov)]
)
def test_transfer_equals_scattering_matrices(transfer, scattering, magnetic_multilayer):
    """the transfer and scattering matrices assemble the same eigenmodes to
    the same Jones matrices for a stable stack"""
    energy, theta = np.r_[700, 707, 710] * u.eV, np.r_[0.5:40:1.5] * u.deg
    reference = simulation(
        scattering, magnetic_multilayer, energy, theta
    ).homogeneous_jones_matrices()
    result = simulation(transfer, magnetic_multilayer, energy, theta).homogeneous_jones_matrices()
    for a, b in zip(result.values(), reference.values()):
        assert relative(a, b) < 1e-12


@pytest.mark.parametrize("cls", [Elzo, Stepanov])
def test_first_order_x_rays_close_to_berreman_stepanov(cls, magnetic_multilayer):
    """the first-order modes of Elzo and Stepanov agree with the exact modes
    of BerremanStepanov to second order in the gyrotropic part for all
    polarizations"""
    energy, theta = np.r_[700, 707, 710] * u.eV, np.r_[0.5:40:1.5] * u.deg
    first_order = simulation(cls, magnetic_multilayer, energy, theta)
    exact = simulation(BerremanStepanov, magnetic_multilayer, energy, theta)
    for a, b in zip(
        first_order.homogeneous_jones_matrices().values(),
        exact.homogeneous_jones_matrices().values(),
    ):
        assert relative(a, b) < 1e-2
    for pol_in, pol_out in [(1, 0), (2, 0), (3, 0), (4, 0), (3, 4)]:
        first_order.set_polarization(pol_in, pol_out)
        exact.set_polarization(pol_in, pol_out)
        res = first_order.homogeneous_reflectivity_transmissivity()
        R, R_phi = res.reflectivity, res.reflectivity_phi
        res = exact.homogeneous_reflectivity_transmissivity()
        R_b, R_b_phi = res.reflectivity, res.reflectivity_phi
        np.testing.assert_allclose(R_b, R, rtol=3e-2)
        # the magnetic asymmetry up to second order at the Fe L3 resonance
        np.testing.assert_allclose(
            (R_b - R_b_phi) / (R_b + R_b_phi), (R - R_phi) / (R + R_phi), atol=5e-3
        )


@pytest.mark.parametrize("cls", [Elzo, Stepanov])
@pytest.mark.parametrize(
    "magnetization", [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0.3, -0.5, 0.8)], ids=str
)
def test_magneto_optics_close_to_berreman_stepanov(cls, magnetization):
    """the first-order modes of Elzo and Stepanov with the division of the
    gyrotropic part by eps_00 agree with the exact modes of BerremanStepanov
    for visible light"""
    S = magnetic_film(magnetization)
    energy, theta = np.r_[1.55] * u.eV, np.r_[10:85:15] * u.deg
    elzo = simulation(cls, S, energy, theta).homogeneous_jones_matrices()
    exact = simulation(BerremanStepanov, S, energy, theta).homogeneous_jones_matrices()
    for a, b in zip(elzo.values(), exact.values()):
        assert relative(a, b) < 1e-3
    # the polarization conversion by the magnetization, which vanishes for
    # the transverse magnetization along x
    conversion = np.max(np.abs(exact.jones_reflection[..., 1, 0]))
    assert (conversion > 1e-4) == (magnetization != (1, 0, 0))
    np.testing.assert_allclose(
        elzo.jones_reflection[..., 1, 0],
        exact.jones_reflection[..., 1, 0],
        atol=max(1e-2 * conversion, 1e-14),
    )


def test_kerr_angles():
    """the Kerr angles reverse with the magnetization and equal the ratio of
    the Jones matrix elements for small angles"""
    S = magnetic_film((0, 0, 1))
    sim = simulation(BerremanStepanov, S, np.r_[1.55] * u.eV, np.r_[30, 60] * u.deg)
    res = sim.homogeneous_jones_matrices()
    R, R_phi = res.jones_reflection, res.jones_reflection_phi
    rotation, ellipticity = sim.calc_rotation_ellipticity(R)
    rotation_phi, ellipticity_phi = sim.calc_rotation_ellipticity(R_phi)
    np.testing.assert_allclose(rotation_phi.m, -rotation.m, rtol=1e-8)
    np.testing.assert_allclose(ellipticity_phi.m, -ellipticity.m, rtol=1e-8)
    ratio = R[..., 1, 0] / R[..., 0, 0]
    # up to second order in the ratio
    np.testing.assert_allclose(rotation.to("rad").m, ratio.real, rtol=1e-3)
    np.testing.assert_allclose(ellipticity.to("rad").m, ratio.imag, rtol=1e-3)
    assert rotation.units == u.deg
    rotation_p, _ = sim.calc_rotation_ellipticity(R, pi_incident=True)
    assert np.all(np.abs(rotation_p.m) > 0)
    # the polar Kerr rotation of sigma and pi incidence have opposite signs at
    # normal incidence in the Fresnel convention
    sim.theta = np.r_[90] * u.deg
    R = sim.homogeneous_jones_matrices().jones_reflection
    rotation_s, ellipticity_s = sim.calc_rotation_ellipticity(R)
    rotation_p, ellipticity_p = sim.calc_rotation_ellipticity(R, pi_incident=True)
    assert np.all(np.abs(rotation_s.m) > 1e-3)
    np.testing.assert_allclose(rotation_p.m, -rotation_s.m, rtol=1e-8)
    np.testing.assert_allclose(ellipticity_p.m, -ellipticity_s.m, rtol=1e-8)


@pytest.mark.parametrize("cls", MAGNETIC)
def test_rotation_ellipticity_from_one_calculation(cls, magnetic_multilayer):
    """the homogeneous and inhomogeneous rotation and ellipticity are
    calculated from a single calculation of the Jones matrices, i.e. they
    equal calc_rotation_ellipticity() of the Jones matrices of the reflection
    and transmission for both magnetizations and for the incident sigma and pi
    polarization, and a homogeneous map gives the homogeneous result for all
    delays"""
    sim = simulation(cls, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[5, 10] * u.deg)
    L = magnetic_multilayer.num_layers
    res = sim.homogeneous_jones_matrices()
    R, R_phi, T, T_phi = (
        res.jones_reflection,
        res.jones_reflection_phi,
        res.jones_transmission,
        res.jones_transmission_phi,
    )
    suffixes = [
        ("reflection", ""),
        ("reflection", "_phi"),
        ("transmission", ""),
        ("transmission", "_phi"),
    ]
    for pol, pi_incident in [("sigma", False), ("pi", True)]:
        sim.set_polarization(pol, "none")
        angles = sim.homogeneous_rotation_ellipticity()
        for (kind, phi), jones in zip(suffixes, (R, R_phi, T, T_phi)):
            rotation = angles[f"{kind}_rotation{phi}"]
            ellipticity = angles[f"{kind}_ellipticity{phi}"]
            reference = sim.calc_rotation_ellipticity(jones, pi_incident)
            np.testing.assert_allclose(rotation, reference[0].to("deg").m)
            np.testing.assert_allclose(ellipticity, reference[1].to("deg").m)
        assert angles.unit("reflection_rotation") == "deg"
        # all delays of a homogeneous map equal the homogeneous result
        maps = (np.zeros((2, L)), np.zeros((2, L, 1)))
        delays = sim.inhomogeneous_rotation_ellipticity(*maps)
        for name, reference in angles.items():
            np.testing.assert_allclose(delays[name], np.stack([reference] * 2), atol=1e-12)
    jones = sim.inhomogeneous_jones_matrices(*maps)
    for a, b in zip(jones.values(), (R, R_phi, T, T_phi)):
        np.testing.assert_allclose(a, np.stack([b] * 2), rtol=1e-10, atol=1e-14)
    sim.set_polarization("unpolarized", "none")
    with pytest.raises(ValueError, match="incoming sigma or pi polarization"):
        sim.homogeneous_rotation_ellipticity()


# ============================================================================
# anisotropic layers
# ============================================================================


@pytest.mark.parametrize("cls", [BerremanElzo, BerremanStepanov])
@pytest.mark.parametrize(
    "euler_angles, axis",
    [((0, 0, 0), 2), ((90, 0, 0), 1), ((90, 90, 0), 0)],
    ids=["axis z", "axis y", "axis x"],
)
def test_uniaxial_crystal(cls, euler_angles, axis):
    """reflection at a uniaxial crystal with the optic axis along x, y, or z
    from the Fresnel equations of a diagonal dielectric tensor"""
    n_o, n_e = 1.66 + 0.01j, 1.49 + 0.02j
    sub = substrate_layer(Dispersion.from_ref_index([n_o, n_e]), euler_angles * u.deg)
    eps = np.full(3, n_o**2)
    eps[axis] = n_e**2
    theta = np.r_[5:85:10]
    R = (
        simulation(cls, interface(sub), 1.55 * u.eV, theta * u.deg)
        .homogeneous_jones_matrices()
        .jones_reflection
    )
    b = np.cos(np.radians(theta))
    kz0 = np.sin(np.radians(theta))
    kz_s = np.sqrt(eps[0] - b**2)
    kz_p = np.sqrt(eps[1]) * np.sqrt(1 - b**2 / eps[2])
    np.testing.assert_allclose(R[0, :, 0, 0], (kz0 - kz_s) / (kz0 + kz_s), atol=1e-12)
    np.testing.assert_allclose(
        R[0, :, 1, 1], (eps[1] * kz0 - kz_p) / (eps[1] * kz0 + kz_p), atol=1e-12
    )
    np.testing.assert_allclose(R[0, :, 0, 1], 0, atol=1e-14)


def test_tilted_optic_axis():
    """a tilted optic axis out of the plane of incidence converts the
    polarization, while both Berreman formalisms agree"""
    sub = substrate_layer(Dispersion.from_ref_index([1.66, 1.49]), (45, 30, 0) * u.deg)
    S = interface(sub)
    S.add_sub_structure(substrate_layer(Dispersion.from_ref_index(1.3)), 1)
    energy, theta = np.r_[1.55, 2.5] * u.eV, np.r_[10:80:10] * u.deg
    elzo = simulation(BerremanElzo, S, energy, theta).homogeneous_jones_matrices()
    stepanov = simulation(BerremanStepanov, S, energy, theta).homogeneous_jones_matrices()
    for a, b in zip(elzo.values(), stepanov.values()):
        assert relative(a, b) < 1e-10
    assert np.max(np.abs(stepanov.jones_reflection[..., 1, 0])) > 1e-3


@pytest.mark.parametrize("cls", [Parratt, Elzo, Stepanov])
def test_isotropic_formalisms_reject_anisotropic_layers(cls):
    S = interface(substrate_layer(Dispersion.from_ref_index([1.66, 1.49])))
    with pytest.raises(ValueError, match="isotropic"):
        simulation(
            cls, S, 1.55 * u.eV, np.r_[10] * u.deg
        ).homogeneous_reflectivity_transmissivity()


def test_berreman_requires_isotropic_superstrate():
    S = interface(substrate_layer())
    S.add_superstrate(substrate_layer(Dispersion.from_ref_index([1.66, 1.49])))
    with pytest.raises(ValueError, match="superstrate 'sub' .* must be isotropic"):
        simulation(
            BerremanStepanov, S, 1.55 * u.eV, np.r_[10] * u.deg
        ).homogeneous_reflectivity_transmissivity()


@pytest.mark.parametrize("cls", FORMALISMS)
def test_formalisms_require_dielectric(cls):
    sub = Layer("sub", "substrate", thickness=1 * u.nm, density=1000 * u.kg / u.m**3)
    with pytest.raises(ValueError, match="Layer 'sub'.*not set"):
        simulation(
            cls, interface(sub), 1.55 * u.eV, np.r_[10] * u.deg
        ).homogeneous_reflectivity_transmissivity()


# ============================================================================
# eigenmodes
# ============================================================================


def test_berreman_modes_of_isotropic_and_magnetic_layers():
    """the analytical isotropic modes are eigenvectors of the Berreman matrix
    and the exact modes of a weakly gyrotropic layer approach the ones of Elzo"""
    from udkm1Dsim.simulations.scattering.berreman import berreman_matrix

    cos_theta = np.cos(np.radians([[10.0, 40.0]]))
    eps = np.eye(3)[np.newaxis] * (2.1 + 0.4j)
    A, q, k_z = berreman_modes(eps, cos_theta)
    delta = berreman_matrix(np.broadcast_to(eps[:, np.newaxis], (1, 2, 3, 3)), cos_theta)
    np.testing.assert_allclose(delta @ A, A * q[..., np.newaxis, :], atol=1e-12)
    np.testing.assert_allclose(k_z, np.sqrt(2.1 + 0.4j - cos_theta**2))
    # a gyrotropic tensor along z has the same normal wave vectors as Elzo
    gyro = eps.copy()
    gyro[:, 0, 1], gyro[:, 1, 0] = -1e-4j, 1e-4j
    _, q_b, _ = berreman_modes(gyro * np.array([[1, 1, -1], [1, 1, -1], [-1, -1, 1]]), cos_theta)
    _, _, q_e, _, _ = elzo_modes(gyro, cos_theta)
    np.testing.assert_allclose(np.sort_complex(q_b[0, 0]), np.sort_complex(q_e[0, 0]), atol=1e-8)


def test_berreman_down_modes_carry_flux_into_the_sample():
    cos_theta = np.cos(np.radians([[20.0, 50.0]]))
    eps = np.diag([2.2 + 0.1j, 2.7 + 0.2j, 2.4 + 0.05j])[np.newaxis]
    eps[:, 0, 2] = eps[:, 2, 0] = 0.2
    A, q, _ = berreman_modes(eps, cos_theta)
    flux = np.real(A[..., 0, :] * np.conj(A[..., 3, :]) - A[..., 1, :] * np.conj(A[..., 2, :]))
    assert np.all(flux[..., :2] > 0)
    assert np.all(flux[..., 2:] < 0)
    assert np.all(q[..., :2].imag >= 0)


# ============================================================================
# stability
# ============================================================================


def test_stepanov_is_stable_for_thick_absorbing_stacks():
    """the scattering-matrix recursion stays exact for thick metal layers,
    for which the transfer matrices of Elzo lose all precision"""
    # 27 µm of the metal in total
    S = optical_stack(substrate_layer(), thickness=3000)
    energy, theta = np.r_[1.55] * u.eV, np.r_[10:80:10] * u.deg
    parratt = simulation(Parratt, S, energy, theta).homogeneous_jones_matrices()
    for cls in (Stepanov, BerremanStepanov):
        result = simulation(cls, S, energy, theta).homogeneous_jones_matrices()
        np.testing.assert_allclose(result.jones_reflection, parratt.jones_reflection, atol=1e-12)
        np.testing.assert_allclose(
            result.jones_transmission, parratt.jones_transmission, atol=1e-14
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elzo = simulation(Elzo, S, energy, theta).homogeneous_jones_matrices()
    # the exponentially growing terms of the up modes overflow
    assert not np.all(np.isfinite(elzo.jones_reflection))


# ============================================================================
# inhomogeneous reflectivity
# ============================================================================


@pytest.mark.parametrize("cls", FORMALISMS)
def test_inhomogeneous_equals_homogeneous(cls, magnetic_multilayer, tmp_path):
    energy, theta = np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg
    sim = simulation(cls, magnetic_multilayer, energy, theta)
    sim.set_polarization(1, 0)
    sim.cache_dir = tmp_path
    res = sim.homogeneous_reflectivity_transmissivity()
    R, R_phi, T, T_phi = (
        res.reflectivity,
        res.reflectivity_phi,
        res.transmissivity,
        res.transmissivity_phi,
    )
    L = magnetic_multilayer.num_layers
    magnetization_map = np.zeros((2, L, 3))
    magnetization_map[..., 0] = 1
    magnetization_map[..., 1] = np.pi / 2
    magnetization_map[..., 2] = np.pi / 6
    result = sim.inhomogeneous_reflectivity_transmissivity(
        strain_map=np.zeros((2, L)), magnetization_map=magnetization_map, polar=True
    )
    for a, b in zip(result.values(), (R, R_phi, T, T_phi)):
        np.testing.assert_allclose(a[1], b, rtol=1e-10)
    assert result.attrs["inputs"] == {"polar": True}
    assert result.attrs["input_hashes"] == {
        "strain_map": array_hash(np.zeros((2, L))),
        "magnetization_map": array_hash(magnetization_map),
    }
    assert res.attrs["inputs"] == {"polar": False}
    # cached results
    sim.save_data = True
    sim.force_recalc = False
    sim.inhomogeneous_reflectivity_transmissivity(strain_map=np.zeros((2, L)))
    cached = sim.inhomogeneous_reflectivity_transmissivity(strain_map=np.zeros((2, L)))
    assert any(
        path.name.startswith(f"inhomogeneous_reflectivity_transmissivity_{cls.__name__}_")
        for path in tmp_path.iterdir()
    )
    np.testing.assert_allclose(
        cached.reflectivity[0],
        sim.homogeneous_reflectivity_transmissivity().reflectivity,
        rtol=1e-10,
    )


@pytest.mark.parametrize("cls", MAGNETIC)
def test_inhomogeneous_merges_identical_layers(cls, magnetic_multilayer, monkeypatch):
    """consecutive identical layers of an inhomogeneous stack are raised to
    the power of their repetitions and their eigenmodes are calculated once,
    which gives the Jones matrices of the stack of all single layers, also
    for a single layer of a run with a different strain or magnetization"""
    sim = simulation(cls, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg)
    L = magnetic_multilayer.num_layers
    strain_map = np.zeros((3, L))
    strain_map[1, 1] = 0.01
    magnetization_map = np.zeros((3, L, 3))
    magnetization_map[..., 1] = 1
    magnetization_map[2, 5, 1] = 0.5
    modes = []
    calc_modes = sim.calc_modes
    monkeypatch.setattr(sim, "calc_modes", lambda eps: modes.append(eps) or calc_modes(eps))
    jones = sim.calc_inhomogeneous_jones_matrices(strain_map, magnetization_map)
    # superstrate, fe, ox, and the substrate of ox with the magnetization of
    # its layer for each delay plus the two single different layers
    assert len(modes) == 3 * 4 + 2
    for i, (strains, magnetizations) in enumerate(zip(strain_map, magnetization_map)):
        single = sim.calc_jones_matrices(sim.calc_inhomogeneous_layers(strains, magnetizations))
        for a, b in zip(jones, single):
            np.testing.assert_allclose(a[i], b, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("cls", MAGNETIC)
def test_inhomogeneous_keeps_identical_periods(cls, periodic_multilayer):
    """the repetitions of a nested structure with identical strains and
    magnetizations of all their layers are kept as period, as in the
    homogeneous simulations, and otherwise expanded to their layers, which
    gives the Jones matrices of the stack of all single layers"""
    sim = simulation(cls, periodic_multilayer, np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg)
    L = periodic_multilayer.num_layers
    strain_map = np.full((2, L), 0.01)
    strain_map[1, 7] = 0.02  # the oxide of the 4th period
    magnetization_map = np.zeros((2, L, 3))
    magnetization_map[..., 2] = 1
    periodic, single = (
        sim._inhomogeneous_layer_tree(strains, magnetizations)
        for strains, magnetizations in zip(strain_map, magnetization_map)
    )
    assert isinstance(periodic[1], _Period) and periodic[1].repetitions == 5
    assert len(periodic) == 3 and len(periodic[1].entries) == 2
    assert not any(isinstance(entry, _Period) for entry in single[1:-1]) and len(single) == 12
    jones = sim.calc_inhomogeneous_jones_matrices(strain_map, magnetization_map)
    for i, (strains, magnetizations) in enumerate(zip(strain_map, magnetization_map)):
        flat = sim.calc_jones_matrices(sim.calc_inhomogeneous_layers(strains, magnetizations))
        for a, b in zip(jones, flat):
            np.testing.assert_allclose(a[i], b, rtol=1e-10, atol=1e-14)
    # the homogeneous result for the same strain and magnetization
    homogeneous = sim.calc_homogeneous_jones_matrices(
        np.full(2, 0.01), np.tile([0.0, 0.0, 1.0], (2, 1))
    )
    for a, b in zip(jones, homogeneous):
        np.testing.assert_allclose(a[0], b, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("cls", FORMALISMS)
def test_calc_methods_equal_results(cls, magnetic_multilayer):
    """the calc_* methods return the plain arrays of the results of the
    homogeneous and inhomogeneous methods"""
    sim = simulation(cls, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg)
    L = magnetic_multilayer.num_layers
    strain_map = np.linspace(0, 0.01, 3)[:, np.newaxis] * np.ones((1, L))
    magnetization_map = np.zeros((3, L, 3))
    magnetization_map[..., 1] = 0.5
    pairs = [
        (sim.homogeneous_jones_matrices, sim.calc_homogeneous_jones_matrices, JONES_NAMES, ()),
        (
            sim.homogeneous_reflectivity_transmissivity,
            sim.calc_homogeneous_reflectivity_transmissivity,
            REFLECTIVITY_NAMES,
            (),
        ),
        (
            sim.homogeneous_rotation_ellipticity,
            sim.calc_homogeneous_rotation_ellipticity,
            ROTATION_NAMES,
            (),
        ),
        (
            sim.inhomogeneous_reflectivity_transmissivity,
            sim.calc_inhomogeneous_reflectivity_transmissivity,
            REFLECTIVITY_NAMES,
            (strain_map, magnetization_map),
        ),
        (
            sim.inhomogeneous_rotation_ellipticity,
            sim.calc_inhomogeneous_rotation_ellipticity,
            ROTATION_NAMES,
            (strain_map, magnetization_map),
        ),
    ]
    for get, calc, names, maps in pairs:
        result = get(*maps)
        arrays = calc(*maps)
        assert isinstance(arrays, tuple) and len(arrays) == len(names)
        for name, value in zip(names, arrays):
            np.testing.assert_array_equal(result[name], value)


def test_inhomogeneous_results_of_upstream_simulations(magnetic_multilayer):
    """results of the upstream simulations are accepted as maps, with the
    polar coordinates of the magnetization and the delays passed on"""
    sim = simulation(Elzo, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg)
    L = magnetic_multilayer.num_layers
    delays = np.r_[0, 1] * u.ps
    polar = np.zeros((2, L, 3))
    polar[..., 0] = 1
    polar[..., 1] = np.pi / 2
    strain = ud.Result("strain map")
    strain.add_coord("delays", delays, "s")
    strain.add("strain_map", np.full((2, L), 0.01), ("delays", "distances"))
    magnetization = ud.Result("magnetization map", {"polar": True})
    magnetization.add("magnetization_map", polar, ("delays", "distances", "components"))
    result = sim.inhomogeneous_reflectivity_transmissivity(strain, magnetization)
    reference = sim.inhomogeneous_reflectivity_transmissivity(strain.strain_map, polar, polar=True)
    for name, value in reference.items():
        np.testing.assert_array_equal(result[name], value)
    np.testing.assert_allclose(result.delays, delays.to("s").magnitude)
    assert "delays" not in reference
    assert result.dims("reflectivity") == ("delays", "energy", "angles")
    assert result.qz.shape == result.reflectivity.shape[1:]
    np.testing.assert_allclose(result.energy, [700, 707])
    assert result.unit("theta") == "deg"


def test_results_of_polarization_lists(magnetic_multilayer):
    """a list of elliptical polarizations adds a last axis to the homogeneous
    and inhomogeneous results, which keep their first dimensions"""
    sim = simulation(Elzo, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg)
    sim.set_polarization(5, 0, polarization_in=[(alpha * u.deg, 0.2) for alpha in (0, 45, 90)])
    hom = sim.homogeneous_reflectivity_transmissivity()
    assert hom.reflectivity.shape == (2, 5, 3)
    assert hom.dims("reflectivity") == ("energy", "angles")
    L = magnetic_multilayer.num_layers
    inhom = sim.inhomogeneous_reflectivity_transmissivity(np.zeros((4, L)))
    assert inhom.reflectivity.shape == (4, 2, 5, 3)
    assert inhom.dims("reflectivity") == ("delays", "energy", "angles")


@pytest.mark.parametrize("cls", [Elzo, Stepanov, BerremanElzo])
def test_inhomogeneous_close_to_berreman_stepanov(cls, magnetic_multilayer):
    """strained and magnetized layers of the inhomogeneous reflectivity agree
    with BerremanStepanov, exactly for the same modes of BerremanElzo and to
    second order in the gyrotropic part for Elzo and Stepanov"""
    energy, theta = np.r_[707] * u.eV, np.r_[1:20:4] * u.deg
    L = magnetic_multilayer.num_layers
    rng = np.random.default_rng(0)
    strain_map = rng.uniform(-0.01, 0.01, (3, L))
    magnetization_map = np.stack(
        [
            rng.uniform(0, 1, (3, L)),
            rng.uniform(0, np.pi, (3, L)),
            rng.uniform(0, 2 * np.pi, (3, L)),
        ],
        axis=-1,
    )
    exact = simulation(BerremanStepanov, magnetic_multilayer, energy, theta)
    result = simulation(cls, magnetic_multilayer, energy, theta)
    for sim in (exact, result):
        sim.set_polarization(1, 0)
    reference = exact.inhomogeneous_reflectivity_transmissivity(
        strain_map, magnetization_map, polar=True
    )
    rtol = 1e-10 if cls is BerremanElzo else 3e-2
    for a, b in zip(
        result.inhomogeneous_reflectivity_transmissivity(
            strain_map, magnetization_map, polar=True
        ).values(),
        reference.values(),
    ):
        np.testing.assert_allclose(a, b, rtol=rtol)


@pytest.mark.parametrize("cls", FORMALISMS)
def test_strain_stretches_the_layers(cls):
    """a strain eps without change of the optical constants (strain model
    none) equals the unstrained stack with thicknesses d (1 + eps), also for
    a contraction"""
    S = optical_stack(substrate_layer())
    for layer in S.unique_layers[1]:
        layer.optical.dielectric.strain_model = "none"
    energy, theta = np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 70] * u.deg
    sim = simulation(cls, S, energy, theta)
    strains = np.r_[0.05, -0.05]
    strain_map = strains[:, np.newaxis] * np.ones([2, S.num_layers])
    R = sim.calc_inhomogeneous_reflectivity_transmissivity(strain_map)[0]
    for i, eps in enumerate(strains):
        stretched = simulation(cls, optical_stack(substrate_layer(), 1 + eps), energy, theta)
        for layer in stretched.S.unique_layers[1]:
            layer.optical.dielectric.strain_model = "none"
        reference = stretched.calc_homogeneous_reflectivity_transmissivity()[0]
        np.testing.assert_allclose(R[i], reference, rtol=1e-10)
        # the strain changes the reflectivity
        assert relative(R[i], sim.calc_homogeneous_reflectivity_transmissivity()[0]) > 1e-3


@pytest.mark.parametrize("pol_in", [1, 3])
def test_parratt_inhomogeneous_for_all_delays_at_once(pol_in):
    """the recursion of Parratt over the layers for all delays at once equals
    the one for every delay and Stepanov for strained, rough layers"""
    S = optical_stack(substrate_layer(Dispersion.from_ref_index(1.5 + 0.01j)))
    S.get_layer_handle(0).roughness = 0.5 * u.nm
    energy, theta = np.r_[1.55, 2.5] * u.eV, np.r_[10:80:20] * u.deg
    L = S.num_layers
    strain_map = np.random.default_rng(1).uniform(-0.02, 0.02, (5, L))
    parratt = simulation(Parratt, S, energy, theta)
    stepanov = simulation(Stepanov, S, energy, theta)
    for sim in (parratt, stepanov):
        sim.set_polarization(pol_in, 0)
    assert parratt._delay_block_size() >= 5
    result = parratt.inhomogeneous_reflectivity_transmissivity(strain_map)
    single = MultilayerScattering._reduce_reflectivity_transmissivity(
        parratt,
        MultilayerScattering.calc_inhomogeneous_jones_matrices(
            parratt, strain_map, np.zeros((5, L, 3))
        ),
    )
    S.get_layer_handle(0).roughness = 0 * u.nm
    reference = stepanov.inhomogeneous_reflectivity_transmissivity(strain_map)
    smooth = parratt.inhomogeneous_reflectivity_transmissivity(strain_map)
    for a, b, c, d in zip(result.values(), single, smooth.values(), reference.values()):
        assert a.shape == d.shape
        np.testing.assert_allclose(a, b, rtol=1e-12)
        np.testing.assert_allclose(c, d, rtol=1e-10)


def test_parratt_inhomogeneous_merges_identical_layers(magnetic_multilayer, monkeypatch):
    """consecutive identical layers with the same strains for all delays of a
    block are handled at once by the recursion of Parratt for all delays,
    which gives the Jones matrices of every single delay, also for a layer of
    a run with a different strain at a single delay"""
    sim = simulation(Parratt, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg)
    L = magnetic_multilayer.num_layers
    strain_map = np.full((3, L), 0.01)
    strain_map[1, 1] = 0.02  # the 2nd of the 3 fe layers at the 2nd delay
    constants = []
    layer_constants = sim._layer_constants
    monkeypatch.setattr(
        sim, "_layer_constants", lambda *args: constants.append(args[0]) or layer_constants(*args)
    )
    jones = sim.calc_inhomogeneous_jones_matrices(strain_map, np.zeros((3, L, 3)))
    # superstrate, 3 single fe, one run of 2 ox and of 2 fe, and the substrate
    assert len(constants) == 1 + 3 + 1 + 1 + 1
    for i, strains in enumerate(strain_map):
        single = sim.calc_jones_matrices(sim.calc_inhomogeneous_layers(strains, np.zeros((L, 3))))
        for a, b in zip(jones, single):
            np.testing.assert_allclose(a[i], b, rtol=1e-12, atol=1e-15)


def test_inhomogeneous_input_validation(magnetic_multilayer):
    sim = simulation(Stepanov, magnetic_multilayer, np.r_[707] * u.eV, np.r_[5] * u.deg)
    L = magnetic_multilayer.num_layers
    with pytest.raises(ValueError, match="At least"):
        sim.inhomogeneous_reflectivity_transmissivity()
    with pytest.raises(ValueError, match="same number of delay steps"):
        sim.inhomogeneous_reflectivity_transmissivity(np.zeros((2, L)), np.zeros((3, L, 3)))
    with pytest.raises(TypeError, match="numpy ndarray"):
        sim.inhomogeneous_reflectivity_transmissivity(list(np.zeros((2, L))))
    with pytest.raises(TypeError, match="calc_type"):
        sim.inhomogeneous_reflectivity_transmissivity(np.zeros((2, L)), calc_type="distributed")
    with pytest.raises(ValueError, match="dask client"):
        sim.inhomogeneous_reflectivity_transmissivity(np.zeros((2, L)), calc_type="parallel")
    # polar is a keyword only, so that it cannot be confused with a map
    with pytest.raises(TypeError):
        sim.inhomogeneous_reflectivity_transmissivity(
            np.zeros((2, L)), np.zeros((2, L, 3)), [], True
        )


def test_temperatures_are_not_implemented(magnetic_multilayer):
    """the temperature dependence of the dielectric tensor is planned, while
    an empty temp_map or temps keep the results and their cache"""
    sim = simulation(Stepanov, magnetic_multilayer, np.r_[707] * u.eV, np.r_[5] * u.deg)
    L = magnetic_multilayer.num_layers
    strain_map = np.zeros((2, L))
    for method in (
        sim.inhomogeneous_jones_matrices,
        sim.inhomogeneous_reflectivity_transmissivity,
        sim.inhomogeneous_rotation_ellipticity,
    ):
        with pytest.raises(NotImplementedError, match="temperature dependence"):
            method(strain_map, temp_map=np.full((2, L), 300.0))
    with pytest.raises(TypeError, match="temp_map"):
        sim.inhomogeneous_reflectivity_transmissivity(strain_map, temp_map=[300.0])
    N = magnetic_multilayer.num_sub_structures
    for method in (
        sim.homogeneous_jones_matrices,
        sim.homogeneous_reflectivity_transmissivity,
        sim.homogeneous_rotation_ellipticity,
    ):
        with pytest.raises(NotImplementedError, match="temps"):
            method(temps=np.full(N, 300.0))
    for a, b in zip(
        sim.homogeneous_jones_matrices(np.zeros(N), temps=[]).values(),
        sim.homogeneous_jones_matrices().values(),
    ):
        np.testing.assert_array_equal(a, b)
    reference = sim.inhomogeneous_reflectivity_transmissivity(strain_map)
    result = sim.inhomogeneous_reflectivity_transmissivity(strain_map, temp_map=np.array([]))
    for a, b in zip(result.values(), reference.values()):
        np.testing.assert_array_equal(a, b)
    assert sim.get_hash(strain_map=strain_map, temp_map=np.array([])) == sim.get_hash(
        strain_map=strain_map
    )
    assert sim.get_hash(strain_map=strain_map, temp_map=np.ones((2, L))) != sim.get_hash(
        strain_map=strain_map
    )


def test_homogeneous_layers_of_nested_structures():
    a = substrate_layer()
    b = substrate_layer(Dispersion.from_ref_index(1.3))
    inner = Structure("inner")
    inner.add_sub_structure(a, 2)
    inner.add_sub_structure(b, 1)
    S = Structure("outer")
    S.add_sub_structure(b, 1)
    S.add_sub_structure(inner, 3)
    sim = simulation(Parratt, S, 1.55 * u.eV, np.r_[10] * u.deg)
    layers = sim.calc_homogeneous_layers([0.1, 0.2, 0.3], np.zeros([3, 1]))
    # superstrate, 1 + 3 * 3 layers, and the substrate
    assert len(layers) == 12
    assert [strain for _, strain, _ in layers[1:5]] == [0.1, 0.2, 0.2, 0.3]
    assert layers[-1][0] is S.substrate[0]


@pytest.mark.parametrize("last", ["energy", "wl", "k", "theta", "qz"])
def test_experiment_angles_and_scattering_vector_are_consistent(last):
    """qz = 2 k sin(theta) for several energies and any order of the setters"""
    sim = ud.XrayKin(ud.Structure("empty"), True)
    sim.energy = np.array([500.0, 1000.0]) * u.eV
    sim.qz = np.array([1.0, 2.0, 3.0]) / u.nm
    new = {
        "energy": np.array([600.0, 1000.0, 1500.0]) * u.eV,
        "wl": np.array([2.0, 1.2]) * u.nm,
        "k": np.array([3.0, 5.0]) / u.nm,
        "theta": np.array([10.0, 20.0, 30.0]) * u.deg,
        "qz": np.array([0.5, 1.5]) / u.nm,
    }
    setattr(sim, last, new[last])
    np.testing.assert_allclose(
        sim.experiment.qz.magnitude,
        2 * sim.experiment.k.magnitude[:, np.newaxis] * np.sin(sim.experiment.theta.magnitude),
    )
    assert (
        sim.experiment.qz.magnitude.shape
        == sim.experiment.theta.magnitude.shape
        == (np.size(sim.experiment.energy.magnitude), sim.experiment.qz.magnitude.shape[1])
    )
    if last in ["energy", "wl", "k"]:
        np.testing.assert_allclose(sim.qz.to("1/nm").magnitude[-1], [1.0, 2.0, 3.0])


def test_experiment_keeps_the_angle_set_last():
    sim = ud.XrayKin(ud.Structure("empty"), True)
    sim.energy = np.array([500.0, 1000.0]) * u.eV
    sim.theta = np.array([10.0, 20.0]) * u.deg
    sim.energy = np.array([700.0, 1200.0]) * u.eV
    np.testing.assert_allclose(sim.theta.to("deg").magnitude, [[10.0, 20.0]] * 2)
    assert "theta" not in sim.experiment._derived and "qz" in sim.experiment._derived
    with pytest.raises(IndexError, match="Set energy, wl, or k first"):
        ud.XrayKin(ud.Structure("empty"), True).qz = np.array([1.0]) / u.nm
    # bare numbers and dimensionless quantities are ambiguous angles
    for theta in (np.array([10.0, 20.0]), np.array([0.1, 0.2]) * u.dimensionless):
        with pytest.raises(TypeError, match="incidence angle theta must be given"):
            sim.theta = theta
    np.testing.assert_allclose(sim.theta.to("deg").magnitude, [[10.0, 20.0]] * 2)


def test_display_before_the_experiment_is_set():
    sim = ud.XrayKin(ud.Structure("empty"), True)
    assert "not set" in sim._repr_html_()
    assert "not set" in str(sim)
    sim.energy = np.array([500.0, 1000.0]) * u.eV
    assert "500 eV .. 1000 eV" in sim._repr_html_()


def test_polarization_without_message_on_init(capsys):
    sim = ud.XrayKin(ud.Structure("empty"), True, disp_messages=True)
    assert capsys.readouterr().out == ""
    assert (sim.pol_in_state, sim.pol_out_state) == (3, 0)
    sim.set_incoming_polarization("pi")
    assert "incoming polarizations set to: pi" in capsys.readouterr().out


@pytest.mark.parametrize("cls", MAGNETIC)
def test_multilayer_polarization_names_and_invalid_states(cls, magnetic_multilayer):
    energy, theta = np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg
    sim = simulation(cls, magnetic_multilayer, energy, theta)
    names = {0: "unpolarized", 1: "circ +", 2: "circ -", 3: "sigma", 4: "pi"}
    for state, name in names.items():
        sim.set_incoming_polarization(state)
        by_id = sim.pol_in
        sim.set_incoming_polarization(name)
        assert sim.pol_in_state == state
        np.testing.assert_array_equal(sim.pol_in, by_id)
    # short names of circ +, circ -, sigma, and pi
    for alias, state in [("c+", 1), ("c-", 2), ("s", 3), ("p", 4)]:
        sim.set_polarization(alias, alias)
        assert (sim.pol_in_state, sim.pol_out_state) == (state, state)
    sim.set_polarization("circ +", "circ +")
    np.testing.assert_allclose(sim.pol_out, np.conj(sim.pol_in))
    sim.set_polarization("sigma", "none")
    assert sim.pol_out.size == 0
    with pytest.raises(ValueError, match="does not support the incoming polarization 7"):
        sim.set_polarization(7, 0)
    assert sim.pol_in_state == 3


@pytest.fixture
def elzo(magnetic_multilayer):
    return simulation(Elzo, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[1:10:2] * u.deg)


def test_elliptical_polarization_requires_valid_parameters(elzo):
    for setter in (elzo.set_incoming_polarization, elzo.set_outgoing_polarization):
        with pytest.raises(ValueError):
            setter(5)
        setter(5, (0 * u.deg, 0))
        with pytest.raises(ValueError):
            setter(5, (0 * u.deg, 2))
        # bare numbers and dimensionless quantities are ambiguous angles
        for alpha in (0, np.pi / 2, 0.5 * u.dimensionless):
            with pytest.raises(TypeError, match="angular unit"):
                setter(5, (alpha, 0))


def test_polarization_consistency(elzo):
    pols = [(alpha * u.deg, 0) for alpha in (0, 1, 2)]
    elzo.set_polarization(5, 5, pols, pols)
    assert elzo.num_pol == 3
    # changing only one side to a different number of polarizations must fail
    # and keep the previous state
    with pytest.raises(ValueError, match="set_polarization"):
        elzo.set_outgoing_polarization(5, pols[:2])
    assert elzo.pol_out.shape == (3, 2)
    with pytest.raises(ValueError, match="set_polarization"):
        elzo.set_incoming_polarization(5, pols[:2])
    assert elzo.pol_in.shape == (3, 2)
    # a single polarization on one side is always allowed
    elzo.set_outgoing_polarization(3)
    assert elzo.num_pol == 3
    elzo.set_outgoing_polarization(0)
    assert elzo.num_pol == 3
    # changing both at the same time
    elzo.set_polarization(5, 5, pols[:2], pols[:2])
    assert elzo.num_pol == 2
    with pytest.raises(ValueError):
        elzo.set_polarization(5, 5, pols[:2], pols)
    assert elzo.num_pol == 2
    elzo.set_polarization(3, 0)
    assert elzo.num_pol == 1


@pytest.mark.parametrize(
    "polarization, state",
    [((0 * u.deg, 1), 1), ((0 * u.deg, -1), 2), ((0 * u.deg, 0), 3), ((90 * u.deg, 0), 4)],
)
def test_elliptical_analyzer_transmits_its_own_polarization(polarization, state):
    """an elliptical analyzer transmits the light of the same elliptical or
    of the equivalent circular or linear polarization and blocks the
    orthogonal one, i.e. the opposite helicity"""
    calc_in = MultilayerScattering.calc_incoming_polarization
    calc_out = MultilayerScattering.calc_outgoing_polarization
    analyzer = calc_out(5, polarization)[0]
    alpha, ellipticity = polarization
    orthogonal = (alpha + 90 * u.deg, -ellipticity)
    assert abs(calc_in(5, polarization)[0] @ analyzer) ** 2 == pytest.approx(1)
    assert abs(calc_in(state) @ analyzer) ** 2 == pytest.approx(1)
    assert abs(calc_in(5, orthogonal)[0] @ analyzer) ** 2 == pytest.approx(0, abs=1e-15)
    assert abs(calc_in(state) @ calc_out(state)) ** 2 == pytest.approx(1)


def test_get_hash_of_elliptical_polarizations(elzo):
    elzo.set_polarization(5, 0, (0 * u.deg, 0))
    hash_1 = elzo.get_hash()
    elzo.set_polarization(5, 0, (45 * u.deg, 0))
    hash_2 = elzo.get_hash()
    elzo.set_polarization(5, 0, [(0 * u.deg, 0), (45 * u.deg, 0)])
    hash_3 = elzo.get_hash()
    assert len({hash_1, hash_2, hash_3}) == 3


def test_inhomogeneous_for_several_polarizations(elzo, magnetic_multilayer):
    L = magnetic_multilayer.num_layers
    strain_map = np.full((2, L), 0.01)
    magnetization_map = np.zeros((2, L, 3))
    elzo.set_incoming_polarization(5, [(0 * u.deg, 0), (45 * u.deg, 0), (90 * u.deg, 0)])
    elzo.set_outgoing_polarization(5, [(0 * u.deg, 0)])
    assert elzo.num_pol == 3
    res = elzo.inhomogeneous_reflectivity_transmissivity(strain_map, magnetization_map)
    R = res.reflectivity
    assert R.shape == (2,) + elzo.experiment.qz.magnitude.shape + (3,)
    # single polarization results equal the respective multi-polarization slices
    elzo.set_incoming_polarization(5, (45 * u.deg, 0))
    res = elzo.inhomogeneous_reflectivity_transmissivity(strain_map, magnetization_map)
    R_single = res.reflectivity
    assert R_single.ndim == 3
    np.testing.assert_allclose(R_single, R[..., 1])


def test_polar_and_cartesian_magnetization_maps(elzo, magnetic_multilayer):
    """the magnetization_map is cartesian by default and polar for
    polar=True, which give the same results"""
    elzo.set_polarization("circ +", "none")
    L = magnetic_multilayer.num_layers
    strain_map = np.full((2, L), 0.01)
    polar = np.zeros([2, L, 3])
    polar[..., 0] = np.linspace(0.2, 1, L)
    polar[..., 1] = np.pi / 3
    polar[..., 2] = np.pi / 4
    cartesian = ud.helpers.convert_polar_to_cartesian(polar)
    res = elzo.inhomogeneous_reflectivity_transmissivity(strain_map, cartesian)
    R, R_phi = res.reflectivity, res.reflectivity_phi
    res = elzo.inhomogeneous_reflectivity_transmissivity(strain_map, polar, polar=True)
    R_polar, R_phi_polar = res.reflectivity, res.reflectivity_phi
    np.testing.assert_allclose(R_polar, R)
    np.testing.assert_allclose(R_phi_polar, R_phi)
    # the magnetization matters for circular polarization
    res = elzo.inhomogeneous_reflectivity_transmissivity(strain_map, np.zeros([2, L, 3]))
    R_zero = res.reflectivity
    assert not np.allclose(R, R_zero)
    # the homogeneous reflectivity of the sub-structures
    N = magnetic_multilayer.num_sub_structures
    res = elzo.homogeneous_reflectivity_transmissivity(np.zeros(N), cartesian[0, :N])
    R_hom = res.reflectivity
    res = elzo.homogeneous_reflectivity_transmissivity(np.zeros(N), polar[0, :N], polar=True)
    R_hom_polar = res.reflectivity
    np.testing.assert_allclose(R_hom_polar, R_hom)


@pytest.mark.parametrize("cls", MAGNETIC)
def test_unpolarized_light_is_an_incoherent_average(cls, magnetic_multilayer):
    """unpolarized light is the incoherent average of sigma and pi, or of the
    circular polarizations, for any analyzer, also for magnetic layers with
    polarization conversion, and without analyzer and a transverse
    magnetization the average of a circular polarization for both directions
    of the magnetization"""
    sim = simulation(cls, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[5:40:5] * u.deg)
    res = sim.homogeneous_jones_matrices()
    R, T = res.jones_reflection, res.jones_transmission
    assert np.max(np.abs(R[..., 1, 0])) > 1e-2 * np.max(np.abs(R[..., 0, 0]))
    np.testing.assert_array_equal(sim.calc_incoming_polarization(0), [0, 0])

    def intensity(J, pol_in, pol_out):
        return sim.calc_reflectivity_transmissivity(
            J, T, sim.calc_incoming_polarization(pol_in), sim.calc_outgoing_polarization(pol_out)
        )[0]

    for pol_out in [0, 1, 3, 4]:
        unpolarized = intensity(R, 0, pol_out)
        linear = (intensity(R, 3, pol_out) + intensity(R, 4, pol_out)) / 2
        circular = (intensity(R, 1, pol_out) + intensity(R, 2, pol_out)) / 2
        np.testing.assert_allclose(unpolarized, linear, rtol=1e-12)
        np.testing.assert_allclose(unpolarized, circular, rtol=1e-12)
    # without a transverse magnetization along x, reversing the magnetization
    # switches the helicity, while a transverse one changes the reflectivity of
    # pi itself, which differs then for the unpolarized light of both directions
    N = magnetic_multilayer.num_sub_structures
    for magnetization, equal in [((0, 0.6, 0.8), True), ((1, 0, 0), False)]:
        res = sim.homogeneous_jones_matrices(np.zeros(N), np.tile(magnetization, (N, 1)))
        R_m, R_m_phi = res.jones_reflection, res.jones_reflection_phi
        average = (intensity(R_m, 1, 0) + intensity(R_m_phi, 1, 0)) / 2
        assert np.allclose(intensity(R_m, 0, 0), average, rtol=1e-10) == equal
    # the same for the high-level methods of the simulation
    sim.set_polarization("unpolarized", "sigma")
    reference = (intensity(R, 3, 3) + intensity(R, 4, 3)) / 2
    np.testing.assert_allclose(
        sim.homogeneous_reflectivity_transmissivity().reflectivity, reference
    )


def test_elzo_unit_cell_equals_layer_with_its_dielectric_tensor():
    """a UnitCell is a homogeneous layer with its dielectric tensor"""
    n = 1 - 1e-3 + 1e-3j
    dielectric = ud.Dispersion.from_ref_index(n)
    Si = Atom("Si")
    uc = ud.UnitCell(
        "uc", "uc", c_axis=5 * u.angstrom, a_axis=5 * u.angstrom, b_axis=5 * u.angstrom
    )
    uc.add_atom(Si, 0.0)
    uc.optical.dielectric = ud.Dielectric(eps0=dielectric, strain_model="none")
    layer = ud.Layer(
        "layer",
        "layer",
        structural=ud.StructuralParameters(thickness=5 * u.angstrom, density=2000 * u.kg / u.m**3),
        optical=ud.OpticalParameters(
            dielectric=ud.Dielectric(eps0=dielectric, strain_model="none")
        ),
    )
    energy, theta = np.r_[700.0] * u.eV, np.r_[5:45:10] * u.deg
    results = []
    for film in [uc, layer]:
        S = ud.Structure("film")
        S.add_sub_structure(film, 20)
        S.add_substrate(Vacuum())
        results.append(simulation(Elzo, S, energy, theta).homogeneous_jones_matrices())
    for a, b in zip(*(result.values() for result in results)):
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("cls", [Elzo, Parratt, Stepanov, BerremanElzo, BerremanStepanov])
def test_inhomogeneous_parallel_equals_sequential(cls, magnetic_multilayer, tmp_path):
    distributed = pytest.importorskip("distributed")
    energy, theta = np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg
    sim = simulation(cls, magnetic_multilayer, energy, theta)
    sim.cache_dir = tmp_path
    L = magnetic_multilayer.num_layers
    rng = np.random.default_rng(0)
    strain_map = rng.uniform(-0.01, 0.01, (3, L))
    magnetization_map = rng.uniform(-0.5, 0.5, (3, L, 3))
    reference = sim.inhomogeneous_reflectivity_transmissivity(strain_map, magnetization_map)
    with distributed.Client(processes=False, dashboard_address=":0") as client:
        result = sim.inhomogeneous_reflectivity_transmissivity(
            strain_map, magnetization_map, calc_type="parallel", dask_client=client
        )
    for a, b in zip(result.values(), reference.values()):
        np.testing.assert_allclose(a, b, rtol=1e-12)


def test_parratt_parallel_blocks_per_worker(magnetic_multilayer):
    """the delays of Parratt are split into at least one block per worker
    thread for the parallel calculation instead of its large sequential
    blocks, while the formalisms with a single delay per block keep it"""
    distributed = pytest.importorskip("distributed")
    energy, theta = np.r_[707] * u.eV, np.r_[1:20:4] * u.deg
    parratt = simulation(Parratt, magnetic_multilayer, energy, theta)
    elzo = simulation(Elzo, magnetic_multilayer, energy, theta)
    assert parratt._delay_block_size() >= 7
    assert parratt._delay_blocks(7) == [slice(0, 7)]
    with distributed.Client(
        processes=False, n_workers=1, threads_per_worker=3, dashboard_address=":0"
    ) as client:
        assert parratt._delay_blocks(7, client) == [slice(0, 3), slice(3, 6), slice(6, 7)]
        assert parratt._delay_blocks(2, client) == [slice(0, 1), slice(1, 2)]
        assert elzo._delay_blocks(3, client) == [slice(0, 1), slice(1, 2), slice(2, 3)]


@pytest.mark.parametrize("calc_type", ["sequential", "parallel"])
def test_xray_dyn_lookup_equals_recalculation(structure_crystalline, calc_type):
    """the matrices looked up from the strain_vectors equal the ones
    recalculated for the strains of each unit cell, sequentially and in
    parallel"""
    xray_dyn = ud.XrayDyn(
        structure_crystalline, True, save_data=False, disp_messages=False, progress_bar=False
    )
    xray_dyn.energy = 8000 * u.eV
    xray_dyn.qz = np.r_[1:5:100j] / u.angstrom
    strains = np.r_[-0.01, 0, 0.005, 0.01]
    rng = np.random.default_rng(0)
    strain_map = rng.choice(strains, (3, structure_crystalline.num_layers))
    strain_vectors = [strains] * structure_crystalline.num_unique_layers
    reference = xray_dyn.calc_inhomogeneous_reflectivity(strain_map)
    if calc_type == "parallel":
        distributed = pytest.importorskip("distributed")
        with distributed.Client(processes=False, dashboard_address=":0") as client:
            kwargs = {"calc_type": "parallel", "dask_client": client}
            lookup = xray_dyn.calc_inhomogeneous_reflectivity(strain_map, strain_vectors, **kwargs)
            recalc = xray_dyn.calc_inhomogeneous_reflectivity(strain_map, **kwargs)
        np.testing.assert_allclose(recalc, reference, rtol=1e-12)
    else:
        lookup = xray_dyn.calc_inhomogeneous_reflectivity(strain_map, strain_vectors)
    np.testing.assert_allclose(lookup, reference, rtol=1e-12)
    # the delays differ by their strains
    assert relative(reference[0], reference[1]) > 1e-3


def test_xray_dyn_temperature_dependent_debye_waller_factor(unit_cell_iron):
    """the Debye-Waller factor at the temperature of the temps or temp_map
    equals a constant one of the same value"""
    reflectivities = []
    sims = []
    for deb_wal_fac in ("3e-23*T", 0.9 * u.angstrom**2):
        unit_cell = copy.deepcopy(unit_cell_iron)  # do not alter the module-scoped fixture
        unit_cell.deb_wal_fac = deb_wal_fac
        S = ud.Structure("film")
        S.add_sub_structure(unit_cell, 20)
        xray_dyn = ud.XrayDyn(S, True, save_data=False, disp_messages=False, progress_bar=False)
        xray_dyn.energy = 8047.8 * u.eV
        xray_dyn.qz = np.r_[1:6:200j] / u.angstrom
        temps = np.full(1, 300.0)
        sims.append(xray_dyn)
        reflectivities.append(
            [
                xray_dyn.calc_homogeneous_reflectivity(temps=temps)[0],
                xray_dyn.calc_inhomogeneous_reflectivity(
                    np.zeros([1, 20]), temp_map=np.full([1, 20], 300.0)
                )[0],
            ]
        )
    np.testing.assert_allclose(reflectivities[0][0], reflectivities[1][0], rtol=1e-12)
    np.testing.assert_allclose(reflectivities[0][1], reflectivities[1][1], rtol=1e-12)
    np.testing.assert_allclose(reflectivities[0][0], reflectivities[0][1], rtol=1e-12)
    # at 0 K, the temperature-dependent factor vanishes and the (004) peak
    # at 5 1/angstrom is higher
    cold = sims[0].calc_homogeneous_reflectivity(temps=np.zeros(1))[0]
    assert np.max(cold[0, -50:]) > 1.1 * np.max(reflectivities[0][0][0, -50:])


def test_worker_copy_has_no_cached_matrices(unit_cell_iron):
    S = ud.Structure("crystal")
    S.add_sub_structure(unit_cell_iron, 5)
    sim = ud.XrayDyn(S, True, disp_messages=False, progress_bar=False, save_data=False)
    sim.energy = np.r_[8000.0] * u.eV
    sim.qz = np.r_[1.0, 2.0] / u.nm
    sim.homogeneous_reflectivity()
    assert sim.last_atom_ref_trans_matrices["H"]
    worker = sim._worker_copy()
    assert worker.last_atom_ref_trans_matrices["H"] == []
    assert worker.S is sim.S
    assert sim.last_atom_ref_trans_matrices["H"]


def test_cached_atom_matrices_distinguish_atoms_of_the_same_id(unit_cell_iron):
    """atoms of the same id, but e.g. of different ionicity, do not share
    their cached reflection-transmission matrices"""
    ion_cell = copy.deepcopy(unit_cell_iron)
    ion_cell.atoms[0][0].ionicity = 3

    def reflectivity(sim, S):
        sim.S = S
        return sim.homogeneous_reflectivity().R

    neutral, ion = ud.Structure("neutral"), ud.Structure("ion")
    neutral.add_sub_structure(unit_cell_iron, 10)
    ion.add_sub_structure(ion_cell, 10)
    sim = ud.XrayDyn(neutral, True, disp_messages=False, progress_bar=False, save_data=False)
    sim.energy = np.r_[8000.0] * u.eV
    sim.qz = np.r_[1.0, 2.0, 3.0, 4.0] / u.nm
    R_neutral = reflectivity(sim, neutral)
    R_ion = reflectivity(sim, ion)  # reuses the simulation and its cache
    fresh = ud.XrayDyn(ion, True, disp_messages=False, progress_bar=False, save_data=False)
    fresh.energy, fresh.qz = sim.energy, sim.qz
    np.testing.assert_allclose(R_ion, reflectivity(fresh, ion), rtol=1e-12)
    assert not np.allclose(R_ion, R_neutral, rtol=1e-3)


@pytest.mark.parametrize("cls", [ud.XrayKin, ud.XrayDyn, Elzo, Parratt, BerremanStepanov])
@pytest.mark.parametrize("name", ["superstrate", "substrate"])
def test_missing_superstrate_or_substrate_raises(cls, name):
    S = ud.Structure("empty")
    setattr(S, name, [])
    sim = cls(S, True, disp_messages=False, progress_bar=False, save_data=False)
    sim.energy = np.r_[700.0] * u.eV
    sim.qz = np.r_[1.0, 2.0] / u.nm
    calc = getattr(sim, "homogeneous_reflectivity_transmissivity", None)
    with pytest.raises(ValueError, match=f"The {name} of the structure 'empty' must be a Layer"):
        (calc or sim.homogeneous_reflectivity)()


@pytest.mark.parametrize("cls", [ud.XrayKin, ud.XrayDyn])
def test_atomistic_formalisms_warn_for_neglected_layers(cls, amorphous_layer_iron):
    S = ud.Structure("empty")
    sim = cls(S, True, disp_messages=False, progress_bar=False, save_data=False)
    sim.energy = np.r_[8000.0] * u.eV
    sim.qz = np.r_[1.0, 2.0] / u.nm
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim.homogeneous_reflectivity()  # Vacuum is no warning
    S.add_substrate(amorphous_layer_iron)
    S.add_superstrate(amorphous_layer_iron)
    with pytest.warns(UserWarning, match="neglects the superstrate"):
        with pytest.warns(UserWarning, match="neglects the substrate"):
            sim.homogeneous_reflectivity()


def test_unreachable_scattering_vector_warns_only_on_calculation():
    sim = ud.XrayKin(ud.Structure("empty"), True, disp_messages=False)
    sim.energy = np.r_[8000.0] * u.eV
    sim.qz = np.r_[1.0, 5.0] / u.angstrom
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim.energy = np.r_[3000.0] * u.eV  # 5 1/Angstrom > 2k, but qz is set again
        sim.qz = np.r_[1.0] / u.angstrom
        sim.homogeneous_reflectivity()
    sim.qz = np.r_[1.0, 5.0] / u.angstrom
    with pytest.warns(UserWarning, match="1 scattering vectors qz exceed 2k"):
        sim.homogeneous_reflectivity()


# ============================================================================
# atomistic formalisms
# ============================================================================


@pytest.fixture
def xray_kin(structure_crystalline, tmp_path):
    """a new simulation per test, as many tests set their own experiment"""
    return ud.XrayKin(
        structure_crystalline,
        force_recalc=True,
        cache_dir=tmp_path,
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture
def xray_dyn(structure_crystalline, tmp_path):
    """a new simulation per test, as many tests set their own experiment"""
    return ud.XrayDyn(
        structure_crystalline,
        force_recalc=True,
        cache_dir=tmp_path,
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def atomistic_maps(structure_amorph):
    """homogeneous strain and temperature maps for 11 delays"""
    distances, _, _ = structure_amorph.layer_distances
    return 0.01 * np.ones([11, len(distances)]), 50 * np.ones([11, len(distances)])


@pytest.mark.parametrize("fixture", ["xray_kin", "xray_dyn"])
def test_atomistic_without_experiment_raises(fixture, request):
    sim = request.getfixturevalue(fixture)
    with pytest.raises(ValueError, match="energy and the angles or qz"):
        sim.homogeneous_reflectivity()


def test_xray_kin_update_experiment(xray_kin):
    # energy, wl, or k must be set first
    with pytest.raises(IndexError):
        xray_kin.theta = 45 * u.deg
    xray_kin.energy = 1000 * u.eV
    xray_kin.theta = 45 * u.deg
    assert_quantity_allclose(xray_kin.qz, np.array([[7.1669]]) / u.nm, rtol=1e-2)
    xray_kin.qz = 1 / u.nm
    assert_quantity_allclose(xray_kin.theta, np.array([[5.66]]) * u.deg, rtol=1e-2)
    assert_quantity_allclose(xray_kin.wl, np.array([1.24]) * u.nm, rtol=1e-2)
    assert_quantity_allclose(xray_kin.k, np.array([5.07]) / u.nm, rtol=1e-2)
    xray_kin.k = 1 / u.nm
    xray_kin.wl = 1 * u.nm


def test_xray_kin_repr(xray_kin):
    assert repr(xray_kin).startswith("Kinematical X-Ray Diffraction simulation properties:")
    assert str(xray_kin) == repr(xray_kin)
    assert xray_kin._repr_html_().startswith("<h2>Kinematical X-Ray Diffraction simulation</h2>")


@pytest.mark.parametrize("fixture", ["xray_kin", "xray_dyn"])
def test_atomistic_set_polarization(fixture, request):
    """the atomistic formalisms support only the linear and unpolarized
    states"""
    sim = request.getfixturevalue(fixture)
    for pol_in in [0, 3, 4, "unpolarized", "sigma", "pi"]:
        sim.set_polarization(pol_in, 0)
    assert sim.pol_in_state == 4
    assert sim.pol_in == 1
    sim.set_polarization("sigma", "none")
    assert sim.pol_in_state == 3
    assert sim.pol_out_state == 0
    for pol_in in [1, 2, 5, 7, "circ +", "linear"]:
        with pytest.raises(ValueError, match="does not support the incoming polarization"):
            sim.set_incoming_polarization(pol_in)
    with pytest.raises(ValueError, match="does not support the analyzer polarization"):
        sim.set_outgoing_polarization(3)
    with pytest.raises(AttributeError):
        sim.pol_in_state = 4


def test_xray_kin_calc_polarization_factor(xray_kin):
    """P = 1 for sigma, |cos 2 theta| for pi, and the root of the mean
    intensity factor (1 + cos^2 2 theta)/2 for unpolarized X-rays"""
    theta = np.linspace(0, np.pi / 2, 7)
    expected = {
        "sigma": np.ones_like(theta),
        "pi": np.abs(np.cos(2 * theta)),
        "unpolarized": np.sqrt((1 + np.cos(2 * theta) ** 2) / 2),
    }
    for polarization, P in expected.items():
        xray_kin.set_polarization(polarization, "none")
        np.testing.assert_allclose(
            xray_kin.calc_polarization_factor(theta), P, rtol=1e-12, atol=1e-15
        )


@pytest.mark.parametrize("fixture", ["xray_kin", "xray_dyn"])
def test_atomistic_get_hash(fixture, request, atomistic_maps):
    """the hash follows the values of the strain and temperature maps and of
    the strain vectors, also for large maps"""
    sim = request.getfixturevalue(fixture)
    sim.energy = 8000 * u.eV
    sim.qz = np.r_[1:5:10j] / u.angstrom
    strain_map, temp_map = atomistic_maps
    hash = sim.get_hash({}, strain_map=strain_map, temp_map=temp_map)
    assert hash == sim.get_hash({}, strain_map=strain_map.copy(), temp_map=temp_map.copy())
    assert hash != sim.get_hash({}, strain_map=2 * strain_map, temp_map=temp_map)
    assert hash != sim.get_hash({}, strain_map=strain_map, temp_map=temp_map + 1)
    assert hash != sim.get_hash([np.r_[0:1]], strain_map=strain_map, temp_map=temp_map)
    # maps of more than 1e6 elements which differ only in their last element
    large = np.zeros([1001, 1000])
    changed = large.copy()
    changed[-1, -1] = 1e-3
    assert sim.get_hash({}, strain_map=large) != sim.get_hash({}, strain_map=changed)


def test_xray_kin_calc_uc_structure_factor(xray_kin, unit_cell_iron):
    # for q_z -> 0, the structure factor is the sum of the atomic form factors
    assert_quantity_allclose(
        xray_kin.calc_uc_structure_factor(1000, 0, unit_cell_iron, strain=0.01),
        sum(atom.get_atomic_form_factor(1000) for atom, _, _ in unit_cell_iron.atoms),
        rtol=1e-9,
    )


def test_xray_kin_homogeneous_reflectivity(xray_kin):
    """the kinematical reflectivity is linear in P^2, so that the
    unpolarized one is exactly the mean of the sigma and pi ones"""
    xray_kin.energy = 8000 * u.eV
    xray_kin.qz = np.r_[1:5:200j] / u.angstrom
    R = {}
    for polarization in ("sigma", "pi", "unpolarized"):
        xray_kin.set_polarization(polarization, "none")
        result = xray_kin.homogeneous_reflectivity()
        assert result.reflectivity.shape == (1, 200)
        R[polarization] = result.reflectivity
    assert np.all(R["pi"] < R["sigma"])
    np.testing.assert_allclose(R["unpolarized"], (R["sigma"] + R["pi"]) / 2, rtol=1e-12)


@pytest.mark.parametrize("sim_class", [ud.XrayKin, ud.XrayDyn])
def test_xray_homogeneous_reflectivity_of_nested_structures(
    sim_class, unit_cell_iron, unit_cell_oxygen, tmp_path
):
    """a nested sub-structure reflects as its layers in a flat structure"""
    inner = ud.Structure("double layer")
    inner.add_sub_structure(unit_cell_iron, 3)
    inner.add_sub_structure(unit_cell_oxygen, 2)
    nested = ud.Structure("nested")
    nested.add_sub_structure(unit_cell_oxygen, 4)
    nested.add_sub_structure(inner, 5)
    flat = ud.Structure("flat")
    flat.add_sub_structure(unit_cell_oxygen, 4)
    for _ in range(5):
        flat.add_sub_structure(unit_cell_iron, 3)
        flat.add_sub_structure(unit_cell_oxygen, 2)
    R = []
    for S in [nested, flat]:
        sim = sim_class(S, True, cache_dir=tmp_path, disp_messages=False)
        sim.energy = 8000 * u.eV
        sim.qz = np.r_[1:5:50j] / u.angstrom
        R.append(sim.calc_homogeneous_reflectivity()[0])
    assert nested.num_sub_structures == 3
    np.testing.assert_allclose(R[0], R[1], rtol=1e-6)


def test_xray_dyn_matrices_of_nested_structures(unit_cell_iron, unit_cell_oxygen):
    """the sub-structure matrices keep the one of a single period of a nested
    structure besides its N-th power"""
    inner = ud.Structure("double layer")
    inner.add_sub_structure(unit_cell_iron, 3)
    inner.add_sub_structure(unit_cell_oxygen, 2)
    S = ud.Structure("nested")
    S.add_sub_structure(inner, 5)
    xray_dyn = ud.XrayDyn(S, True, save_data=False, disp_messages=False, progress_bar=False)
    xray_dyn.energy = 8000 * u.eV
    xray_dyn.qz = np.r_[1:5:10j] / u.angstrom
    _, A = xray_dyn.calc_homogeneous_reflectivity()
    matrices = {label: matrix for matrix, label in A[-2:]}
    period, total = matrices["1x double layer"], matrices["5x double layer"]
    np.testing.assert_allclose(total, m_power_x(period, 5), rtol=1e-10)
    assert not np.allclose(total, period)


def test_xray_dyn_all_ref_trans_matrices_without_strains(xray_dyn):
    xray_dyn.energy = 8000 * u.eV
    xray_dyn.qz = np.r_[1:5:10j] / u.angstrom
    RTM = xray_dyn._get_all_ref_trans_matrices()
    assert len(RTM) == xray_dyn.S.num_unique_layers


def test_xray_dyn_repr(xray_dyn):
    xray_dyn.energy = 8000 * u.eV
    xray_dyn.theta = np.r_[1:10] * u.deg
    assert repr(xray_dyn).startswith("Dynamical X-Ray Diffraction simulation properties:")
    assert str(xray_dyn) == repr(xray_dyn)
    assert xray_dyn._repr_html_().startswith("<h2>Dynamical X-Ray Diffraction simulation</h2>")


def test_xray_dyn_homogeneous_reflectivity(xray_dyn):
    xray_dyn.energy = 8000 * u.eV
    xray_dyn.qz = np.r_[1:5:100j] / u.angstrom
    R0 = xray_dyn.homogeneous_reflectivity().reflectivity
    # zero strains and no temperatures are the defaults
    np.testing.assert_array_equal(
        xray_dyn.homogeneous_reflectivity(strains=np.zeros([2])).reflectivity, R0
    )
    xray_dyn.homogeneous_reflectivity(strains=np.zeros([2]))
    xray_dyn.homogeneous_reflectivity(temps=300 * np.ones([2]))
    result = xray_dyn.homogeneous_reflectivity(strains=np.zeros([2]), temps=300 * np.ones([2]))
    np.testing.assert_array_equal(result.attrs["inputs"]["strains"], np.zeros([2]))
    np.testing.assert_array_equal(result.attrs["inputs"]["temps"], 300 * np.ones([2]))
    assert "inputs" not in xray_dyn.homogeneous_reflectivity().attrs
    with pytest.raises(IndexError):
        xray_dyn.homogeneous_reflectivity(strains=np.zeros([3]))
    with pytest.raises(IndexError):
        xray_dyn.homogeneous_reflectivity(temps=300 * np.ones([3]))
    with pytest.raises(IndexError):
        xray_dyn.homogeneous_reflectivity(temps=300 * np.ones([2, 2]))


def test_xray_dyn_inhomogeneous_reflectivity(xray_dyn, atomistic_maps):
    strain_map, temp_map = atomistic_maps
    xray_dyn.energy = 8000 * u.eV
    xray_dyn.theta = np.r_[1:10] * u.deg
    xray_dyn.inhomogeneous_reflectivity(strain_map)
    xray_dyn.inhomogeneous_reflectivity(strain_map, temp_map=temp_map)
    strain_vectors = [np.r_[0:1], np.r_[0:1]]
    with pytest.warns(UserWarning):
        xray_dyn.inhomogeneous_reflectivity(
            strain_map, strain_vectors=strain_vectors, temp_map=temp_map
        )
    result = xray_dyn.inhomogeneous_reflectivity(strain_map, strain_vectors=strain_vectors)
    xray_dyn.force_recalc = False
    cached = xray_dyn.inhomogeneous_reflectivity(strain_map, strain_vectors=strain_vectors)
    xray_dyn.force_recalc = True
    for res in (result, cached):
        assert res.attrs["input_hashes"] == {"strain_map": array_hash(strain_map)}
        for a, b in zip(res.attrs["inputs"]["strain_vectors"], strain_vectors):
            np.testing.assert_array_equal(a, b)
    assert cached.attrs["parameters"]["incoming_polarization"] == "sigma"
    with pytest.raises(ValueError):
        xray_dyn.inhomogeneous_reflectivity(
            strain_map, strain_vectors=strain_vectors, calc_type="parallel"
        )
    with pytest.raises(TypeError, match="calc_type"):
        xray_dyn.inhomogeneous_reflectivity(
            strain_map, strain_vectors=strain_vectors, calc_type="distributed"
        )
    with pytest.raises(TypeError):
        xray_dyn.inhomogeneous_reflectivity([])
    with pytest.raises(TypeError):
        xray_dyn.inhomogeneous_reflectivity(strain_map, temp_map=[])
    with pytest.raises(TypeError):
        xray_dyn.inhomogeneous_reflectivity(strain_map, strain_vectors=np.array([]))
    with pytest.raises(ValueError):
        xray_dyn.inhomogeneous_reflectivity(strain_map, temp_map=np.ones([11, 20, 2]))


@pytest.mark.parametrize("polarization", ["sigma", "pi", "unpolarized"])
def test_xray_kin_equals_xray_dyn_for_thin_films(unit_cell_iron, polarization):
    """the kinematical theory is the limit of the dynamical one for thin
    films without Debye-Waller factor, which XrayKin neglects, while their
    difference grows with the thickness by the multiple scattering"""
    unit_cell = copy.deepcopy(unit_cell_iron)  # do not alter the module-scoped fixture
    unit_cell.deb_wal_fac = 0 * u.m**2
    differences = []
    for num_cells in (5, 20, 200):
        S = ud.Structure("film")
        S.add_sub_structure(unit_cell, num_cells)
        R = []
        for cls in (ud.XrayKin, ud.XrayDyn):
            sim = cls(S, True, save_data=False, disp_messages=False, progress_bar=False)
            sim.energy = 8047.8 * u.eV
            sim.set_polarization(polarization, "none")
            sim.qz = np.linspace(1.0, 1.5, 1001) / u.angstrom  # around the (001) peak
            R.append(sim.calc_homogeneous_reflectivity()[0][0])
        kin, dyn = R
        differences.append(abs(np.sum(kin) / np.sum(dyn) - 1))
        if num_cells == 5:
            assert np.argmax(kin) == np.argmax(dyn)
            assert np.max(kin) == pytest.approx(np.max(dyn), rel=1e-3)
    assert differences[0] < 1e-3
    assert differences[0] < differences[1] < differences[2]


@pytest.mark.parametrize("cls", [ud.XrayKin, ud.XrayDyn])
def test_strain_shifts_the_bragg_peak(unit_cell_iron, cls):
    """a homogeneous strain eps stretches the lattice and shifts the Bragg
    peak of a film from q0 to q0/(1 + eps), also for the static substrate
    below a strained film"""
    S = ud.Structure("film")
    S.add_sub_structure(unit_cell_iron, 20)
    sim = cls(S, True, save_data=False, disp_messages=False, progress_bar=False)
    sim.energy = 8047.8 * u.eV
    sim.qz = np.linspace(1.1, 1.4, 6001) / u.angstrom  # around the (001) peak
    qz = sim.qz[0].to("1/nm").magnitude
    peaks = {}
    for eps in (0, 0.02):
        R, _ = sim.calc_homogeneous_reflectivity(strains=np.full(1, eps))
        peaks[eps] = qz[np.argmax(R[0])]
    assert peaks[0.02] == pytest.approx(peaks[0] / 1.02, abs=2 * (qz[1] - qz[0]))


def test_xray_kin_strain_moves_the_static_substrate(unit_cell_iron):
    """the phase of the static substrate follows the strained thickness of
    the film above, so a strain of the film changes the interference with
    the substrate"""
    S = ud.Structure("film")
    S.add_sub_structure(unit_cell_iron, 20)
    S.add_substrate(unit_cell_iron, 20)
    xray_kin = ud.XrayKin(S, True, save_data=False, disp_messages=False, progress_bar=False)
    xray_kin.energy = 8047.8 * u.eV
    xray_kin.qz = np.linspace(1.1, 1.4, 201) / u.angstrom
    strains = np.full(1, 0.02)
    R, _ = xray_kin.calc_homogeneous_reflectivity(strains=strains)
    energy = xray_kin.energy[0]
    qz = xray_kin.qz[0]
    theta = xray_kin.theta[0]
    film, _ = xray_kin.homogeneous_reflected_field(S, energy, qz, theta, strains)
    substrate, _ = xray_kin.homogeneous_reflected_field(S.substrate, energy, qz, theta)
    # the film of 20 cells is 20 c (1 + eps) thick
    thickness = 20 * unit_cell_iron.lattice.c_axis.quantity.to("m").magnitude * 1.02
    phase = np.exp(1j * qz.to("1/m").magnitude * thickness)
    np.testing.assert_allclose(R[0], np.abs(film + substrate * phase)[0] ** 2, rtol=1e-10)


def test_xray_calc_methods_equal_results(xray_kin, xray_dyn, atomistic_maps):
    """the calc_* methods of XrayKin and XrayDyn return the plain arrays of
    the results of the homogeneous and inhomogeneous methods"""
    strain_map, temp_map = atomistic_maps
    for sim in (xray_kin, xray_dyn):
        sim.energy = 8000 * u.eV
        sim.theta = np.r_[1:10] * u.deg
        result = sim.homogeneous_reflectivity()
        R, A = sim.calc_homogeneous_reflectivity()
        np.testing.assert_array_equal(result.reflectivity, R)
        assert len(A) == len(
            result.get("sub_structure_fields", result.get("sub_structure_matrices"))
        )
    R, _ = xray_dyn.calc_homogeneous_reflectivity(strains=np.full(2, 0.01))
    np.testing.assert_array_equal(
        xray_dyn.homogeneous_reflectivity(strains=np.full(2, 0.01)).reflectivity, R
    )
    for maps in ((strain_map,), (strain_map, [], temp_map)):
        R = xray_dyn.calc_inhomogeneous_reflectivity(*maps)
        assert isinstance(R, np.ndarray)
        np.testing.assert_array_equal(xray_dyn.inhomogeneous_reflectivity(*maps).reflectivity, R)


def test_aliases_of_scattering_results(xray_kin, magnetic_multilayer):
    """the entries of the scattering results have the short aliases of ALIASES"""
    from udkm1Dsim.simulations.scattering.scattering import ALIASES

    sim = simulation(Elzo, magnetic_multilayer, np.r_[707] * u.eV, np.r_[5, 10] * u.deg)
    xray_kin.energy = 8000 * u.eV
    xray_kin.qz = np.r_[1:5:10j] / u.angstrom
    for result in (
        sim.homogeneous_jones_matrices(),
        sim.homogeneous_reflectivity_transmissivity(),
        sim.homogeneous_rotation_ellipticity(),
        xray_kin.homogeneous_reflectivity(),
    ):
        for name in result.keys():
            assert result.aliases_of(name) == (ALIASES[name],)
            assert result[ALIASES[name]] is result[name]
    assert np.all(sim.homogeneous_reflectivity_transmissivity().R_phi >= 0)


# ============================================================================
# fields and absorption
# ============================================================================

POLARIZATIONS = {
    "sigma": np.array([1, 0]),
    "pi": np.array([0, 1]),
    "circ +": MultilayerScattering.calc_incoming_polarization(1),
    "circ -": MultilayerScattering.calc_incoming_polarization(2),
}


def flux(matrix, pol):
    """normal component of the Poynting vector for the polarization"""
    return np.real(np.einsum("...ij,i,j->...", matrix, pol.conj(), pol))


def reflectivity(sim, pol):
    """reflectivity without analyzer from the Jones matrices"""
    R, _, T, _ = sim.calc_homogeneous_jones_matrices()
    return sim.calc_reflectivity_transmissivity(R, T, pol_in=pol, pol_out=np.array([]))[0]


def check_energy_conservation(sim, points=2001):
    """the flux in the superstrate is 1 - R and the absorption integrated
    over each layer plus the flux through the bottom equals the flux below
    the surface"""
    interfaces = sim.S.interface_distances.to("m").magnitude
    # the depths of each layer without its bottom interface, which belongs to
    # the layer below
    grids = [
        top + (bottom - top) * np.linspace(0, 1 - 1e-10, points)
        for top, bottom in zip(interfaces[:-1], interfaces[1:])
    ]
    z = np.concatenate([[-1e-12], *grids, [interfaces[-1]]])
    fields = sim.calc_stack_fields(sim.calc_homogeneous_layers(), z, cache={}, dichroic=True)
    for suffix in ("", "_phi"):
        S, K = fields["S" + suffix], fields["K" + suffix]
        np.testing.assert_allclose(
            K, np.conj(np.swapaxes(K, -1, -2)), atol=1e-12 * np.abs(K).max()
        )
        for pol in POLARIZATIONS.values():
            if suffix == "":
                np.testing.assert_allclose(flux(S[:, :, 0], pol), 1 - reflectivity(sim, pol))
            dAdz = sim.calc_absorption(K, pol)
            absorbed = sum(
                simpson(dAdz[:, :, 1 + i * points : 1 + (i + 1) * points], x=grid, axis=-1)
                for i, grid in enumerate(grids)
            )
            np.testing.assert_allclose(
                absorbed + flux(S[:, :, -1], pol), flux(S[:, :, 1], pol), rtol=1e-6
            )


# the transmission of the transfer matrices of Elzo and BerremanElzo loses its
# precision for the thick stack, but not their fields and absorption
@pytest.mark.filterwarnings("ignore:The transfer matrix")
@pytest.mark.parametrize("cls", FORMALISMS)
@pytest.mark.parametrize("thickness", [1, 100], ids=["thin", "thick"])
def test_absorption_conserves_energy(cls, thickness):
    """energy conservation of the absorption of an optical stack, also for
    micrometer thick absorbing layers, where the transfer matrices of the up
    modes grow exponentially"""
    S = optical_stack(substrate_layer(), thickness)
    sim = simulation(cls, S, np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg)
    check_energy_conservation(sim)


def test_magnetized_atoms_of_an_unmagnetized_layer_are_warned(unit_cell_iron, recwarn):
    """the gyrotropic part of the dielectric tensor of a unit cell follows the
    magnetization of the layer, not of its magnetized atoms"""
    unit_cell = copy.deepcopy(unit_cell_iron)
    unit_cell.magnetization = (0, 0, 0)
    S = ud.Structure("crystal")
    S.add_sub_structure(unit_cell, 5)
    sim = simulation(Elzo, S, np.r_[708] * u.eV, np.r_[10] * u.deg)
    with pytest.warns(UserWarning, match="atoms of the layer 'unit_cell_Fe' are magnetized"):
        sim.homogeneous_reflectivity_transmissivity()
    unit_cell.magnetization = (0, 0, 1)
    sim.homogeneous_reflectivity_transmissivity()
    assert not [w for w in recwarn if "are magnetized" in str(w.message)]


@pytest.mark.parametrize(
    "transfer, scattering", [(Elzo, Stepanov), (BerremanElzo, BerremanStepanov)]
)
def test_growing_transfer_matrix_is_warned(transfer, scattering, recwarn):
    """the transmission of the transfer matrices of thick absorbing layers
    loses its precision, while the one of the scattering matrices and the
    reflection keep it"""
    S = optical_stack(substrate_layer(), 100)
    energy, theta = np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg
    with pytest.warns(UserWarning, match=f"Use {scattering.__name__} with scattering"):
        R, _, T, _ = simulation(
            transfer, S, energy, theta
        ).calc_homogeneous_reflectivity_transmissivity()
    R_ref, _, T_ref, _ = simulation(
        scattering, S, energy, theta
    ).calc_homogeneous_reflectivity_transmissivity()
    np.testing.assert_allclose(R, R_ref, atol=1e-12)
    assert np.max(np.abs(T - T_ref)) > 1
    S = optical_stack(substrate_layer(), 1)
    simulation(transfer, S, energy, theta).calc_homogeneous_reflectivity_transmissivity()
    assert not [w for w in recwarn if "transfer matrix" in str(w.message)]


@pytest.mark.parametrize("cls", MAGNETIC)
@pytest.mark.parametrize(
    "magnetization", [(0, 0, 1), (0, 1, 0), (1, 0, 0)], ids=["polar", "longitudinal", "transverse"]
)
def test_magnetic_absorption(cls, magnetization):
    """energy conservation and circular dichroism of a magnetic film, whose
    absorption of circularly polarized light reverses with the
    magnetization for the polar and longitudinal geometry"""
    S = magnetic_film(magnetization)
    sim = simulation(cls, S, np.r_[1.55] * u.eV, np.r_[30, 60, 90] * u.deg)
    check_energy_conservation(sim)
    K, K_phi = sim.calc_homogeneous_absorption(np.linspace(0, 10e-9, 101), dichroic=True)[2:4]
    plus = sim.calc_absorption(K, POLARIZATIONS["circ +"])
    minus = sim.calc_absorption(K, POLARIZATIONS["circ -"])
    if magnetization != (1, 0, 0):
        assert np.max(np.abs(plus - minus)) > 1e-3 * np.max(plus)
        np.testing.assert_allclose(
            sim.calc_absorption(K_phi, POLARIZATIONS["circ -"]),
            plus,
            rtol=1e-10,
            atol=1e-12 * np.max(plus),
        )


@pytest.mark.parametrize("cls", MAGNETIC)
def test_dichroic_absorption(cls):
    """the opposite magnetization is calculated only for dichroic=True and
    equals the absorption of the reversed magnetization"""
    z = np.linspace(0, 10e-9, 11)
    sim = simulation(cls, magnetic_film((0, 0.6, 0.8)), np.r_[1.55] * u.eV, np.r_[30, 60] * u.deg)
    res = sim.homogeneous_absorption(z)
    assert "absorption_phi" not in res
    res = sim.homogeneous_absorption(z, dichroic=True)
    reversed_sim = simulation(
        cls, magnetic_film((0, -0.6, -0.8)), np.r_[1.55] * u.eV, np.r_[30, 60] * u.deg
    )
    np.testing.assert_allclose(res.K_phi, reversed_sim.homogeneous_absorption(z).K, rtol=1e-12)
    assert relative(res.K_phi, res.K) > 1e-4


@pytest.mark.parametrize("magnetization", [(0, 0, 1), (0, 1, 0)], ids=["polar", "longitudinal"])
def test_magnetic_absorption_of_elzo_and_berreman(magnetization):
    """the first-order modes of Elzo agree with the exact modes of Berreman for
    a weak gyrotropy"""
    S = magnetic_film(magnetization)
    K = [
        simulation(
            cls, S, np.r_[1.55] * u.eV, np.r_[30, 60] * u.deg
        ).calc_homogeneous_absorption()[2]
        for cls in (Elzo, BerremanStepanov)
    ]
    assert relative(K[0], K[1]) < 1e-3


@pytest.mark.parametrize("cls", [BerremanElzo, BerremanStepanov])
def test_anisotropic_absorption(cls):
    """energy conservation for a tilted uniaxial absorbing layer, which mixes
    the sigma and pi polarizations"""
    layer = substrate_layer(
        Dispersion.from_ref_index([1.66 + 0.05j, 1.49 + 0.1j]), (30, 20, 0) * u.deg
    )
    layer.structural.thickness = 200 * u.nm
    S = Structure("birefringent")
    S.add_sub_structure(layer, 1)
    S.add_substrate(substrate_layer(Dispersion.from_ref_index(1.5)))
    sim = simulation(cls, S, np.r_[1.55] * u.eV, np.r_[20, 50, 90] * u.deg)
    check_energy_conservation(sim)
    K = sim.calc_homogeneous_absorption(np.linspace(0, 200e-9, 201))[2]
    assert np.max(np.abs(K[..., 0, 1])) > 1e-3 * np.max(np.abs(K))


def test_absorption_of_formalisms_agree():
    """all formalisms give the same absorption of an isotropic stack"""
    S = optical_stack(substrate_layer())
    K = [
        simulation(cls, S, np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg)
        .homogeneous_absorption()
        .absorption_matrix
        for cls in FORMALISMS
    ]
    for other in K[1:]:
        assert relative(other, K[0]) < 1e-12


@pytest.mark.parametrize("cls", FORMALISMS)
def test_absorption_of_semi_infinite_absorber(cls):
    """the absorption of a semi-infinite absorber decays exponentially with the
    transmitted fraction 1 - R"""
    theta = np.r_[15, 45, 90]
    sim = simulation(cls, interface(substrate_layer()), np.r_[1.55] * u.eV, theta * u.deg)
    z = np.linspace(0, 1e-6, 11)
    k = sim.experiment.k.magnitude[0]
    alpha = 2 * k * np.imag(np.sqrt(N_SUB**2 - np.cos(np.radians(theta)) ** 2))
    for pol in POLARIZATIONS.values():
        dAdz = sim.calc_absorption(sim.calc_homogeneous_absorption(z)[2], pol)[0]
        expected = ((1 - reflectivity(sim, pol)[0]) * alpha)[:, np.newaxis] * np.exp(
            -alpha[:, np.newaxis] * z
        )
        np.testing.assert_allclose(dAdz, expected, rtol=1e-10)


@pytest.mark.parametrize("cls", FORMALISMS)
def test_fields_of_vacuum_and_interface(cls):
    """an incident field of unit amplitude in vacuum and the continuity of the
    in-plane fields and the normal displacement at an interface"""
    theta = np.r_[20, 45, 90]
    S = Structure("vacuum")
    S.add_sub_structure(Vacuum(thickness=10 * u.nm), 1)
    S.add_substrate(Vacuum())
    sim = simulation(cls, S, np.r_[1.55] * u.eV, theta * u.deg)
    res = sim.homogeneous_fields(np.r_[-5, 0, 5, 20] * u.nm)
    np.testing.assert_allclose(np.linalg.norm(res.E, axis=-2), 1, rtol=1e-12)
    np.testing.assert_allclose(sim.homogeneous_absorption().absorption, 0, atol=1e-6)

    sim = simulation(cls, interface(substrate_layer()), np.r_[1.55] * u.eV, theta * u.deg)
    E, _, H, _ = sim.calc_homogeneous_fields(np.r_[-1e-15, 1e-15])
    np.testing.assert_allclose(E[:, :, 0, :2], E[:, :, 1, :2], atol=1e-12)
    np.testing.assert_allclose(H[:, :, 0], H[:, :, 1], atol=1e-12)
    np.testing.assert_allclose(E[:, :, 0, 2], N_SUB**2 * E[:, :, 1, 2], atol=1e-12)


def test_fields_of_formalisms_agree():
    """all formalisms give the same electric and magnetic fields of an
    isotropic stack in the sample frame"""
    S = optical_stack(substrate_layer())
    z = np.linspace(-5e-9, 20e-9, 51)
    fields = [
        simulation(cls, S, np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg).homogeneous_fields(
            z
        )
        for cls in FORMALISMS
    ]
    for other in fields[1:]:
        assert relative(other.E, fields[0].E) < 1e-12
        assert relative(other.H, fields[0].H) < 1e-12


@pytest.mark.parametrize("cls", FORMALISMS)
def test_incident_polarizations(cls):
    """the incident sigma polarization is along x and the pi polarization is
    k x sigma for the wave vector k into the sample"""
    theta = np.radians(np.r_[20, 45, 90])
    S = Structure("vacuum")
    S.add_sub_structure(Vacuum(thickness=10 * u.nm), 1)
    S.add_substrate(Vacuum())
    sim = simulation(cls, S, np.r_[1.55] * u.eV, np.degrees(theta) * u.deg)
    E = sim.calc_homogeneous_fields(np.r_[0])[0][0, :, 0]
    k = np.stack([np.zeros_like(theta), np.cos(theta), -np.sin(theta)], axis=-1)
    np.testing.assert_allclose(E[..., 0], np.r_[1, 0, 0] * np.ones((3, 1)), atol=1e-12)
    np.testing.assert_allclose(E[..., 1], np.cross(k, E[..., 0]), atol=1e-12)


@pytest.mark.parametrize("cls", FORMALISMS)
def test_transmitted_field(cls):
    """the field in the substrate below a periodic stack with rough
    interfaces equals the Jones matrix of the transmission"""
    S = optical_stack()
    for layer in S.unique_layers[1]:
        layer.structural.roughness = 0.5 * u.nm
    stack = Structure("periodic")
    stack.add_sub_structure(S, 5)
    stack.add_substrate(substrate_layer())
    sim = simulation(cls, stack, np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg)
    T = sim.calc_homogeneous_jones_matrices()[2]
    bottom = stack.interface_distances[-1].to("m").magnitude
    E = sim.calc_homogeneous_fields(np.r_[bottom])[0][:, :, 0]
    # pi polarization k x sigma of the transmitted wave
    q = np.sqrt(N_SUB**2 - np.cos(sim.theta.to("rad").magnitude) ** 2) / N_SUB
    np.testing.assert_allclose(E[..., 0, 0], T[..., 0, 0], rtol=1e-10)
    np.testing.assert_allclose(E[..., 1, 1], -q * T[..., 1, 1], rtol=1e-10)


@pytest.mark.parametrize("cls", FORMALISMS)
@pytest.mark.parametrize("roughness", [0, 0.5])
def test_layer_absorption(cls, roughness):
    """the absorption of each layer equals the integrated absorption profile
    and, for smooth interfaces, the layers and the substrate absorb 1 - R"""
    S = optical_stack(substrate_layer())
    for layer in S.unique_layers[1]:
        layer.structural.roughness = roughness * u.nm
    sim = simulation(cls, S, np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg)
    sim.set_polarization("circ +", "none")
    res = sim.homogeneous_absorption()
    assert res.A_layers.shape == (2, 3, S.num_layers)
    assert res.K_layers.shape == (2, 3, S.num_layers, 2, 2)
    assert res.dims("layer_absorption") == ("energy", "angles", "layers")
    np.testing.assert_allclose(res.A_layers, sim.calc_absorption(res.K_layers))
    tops, bottoms = (d.to("m").magnitude for d in S.layer_distances[:2])
    points = 2001
    for j, (top, bottom) in enumerate(zip(tops, bottoms)):
        z = top + (bottom - top) * np.linspace(0, 1 - 1e-10, points)
        absorbed = simpson(sim.calc_homogeneous_absorption(z)[0], x=z, axis=-1)
        np.testing.assert_allclose(res.A_layers[..., j], absorbed, rtol=1e-6, atol=1e-12)
    S_sub = sim.calc_stack_fields(sim.calc_homogeneous_layers(), np.r_[bottoms[-1]])["S"]
    np.testing.assert_allclose(res.S_sub, flux(S_sub[:, :, 0], sim.pol_in), rtol=1e-12)
    np.testing.assert_allclose(res.S_sub, sim.calc_absorption(res.S_sub_matrix))
    assert res.dims("substrate_flux") == ("energy", "angles")
    if roughness == 0:
        R = reflectivity(sim, sim.pol_in)
        np.testing.assert_allclose(res.A_layers.sum(axis=-1) + res.S_sub, 1 - R, rtol=1e-12)


def birefringent_layer():
    """vacuum around a tilted uniaxial absorbing layer"""
    layer = substrate_layer(
        Dispersion.from_ref_index([1.66 + 0.05j, 1.49 + 0.1j]), (30, 20, 0) * u.deg
    )
    layer.structural.thickness = 200 * u.nm
    S = Structure("birefringent")
    S.add_sub_structure(layer, 1)
    S.add_substrate(Vacuum())
    return S


@pytest.mark.parametrize(
    "sample, classes",
    [
        (optical_stack, FORMALISMS),
        (lambda: magnetic_film((0, 0.6, 0.8)), MAGNETIC),
        (birefringent_layer, [BerremanElzo, BerremanStepanov]),
    ],
    ids=["isotropic", "magnetic", "anisotropic"],
)
def test_reflectivity_transmissivity_absorption(sample, classes):
    """the reflectivity, the transmissivity, and the absorption of the layers
    add up to 1 between vacuum, and the flux into the vacuum substrate is the
    transmissivity, for all incoming polarizations"""
    S = sample()
    for cls in classes:
        sim = simulation(cls, S, np.r_[1.2, 1.55, 3.1] * u.eV, np.r_[5, 20, 45, 70, 90] * u.deg)
        for pol in ["sigma", "pi", "circ +", "circ -", "unpolarized"]:
            sim.set_polarization(pol, "none")
            res = sim.homogeneous_reflectivity_transmissivity()
            absorption = sim.homogeneous_absorption()
            np.testing.assert_allclose(
                res.R + res.T + absorption.A_layers.sum(axis=-1), 1, atol=1e-12
            )
            np.testing.assert_allclose(absorption.S_sub, res.T, atol=1e-12)


@pytest.mark.parametrize("cls", FORMALISMS)
def test_repeated_layers(cls):
    """repetitions of identical layers, e.g. a substrate of many unit cells,
    are handled at once and give the same Jones matrices and fields as the
    single layers"""
    S = optical_stack()
    substrate = substrate_layer()
    substrate.structural.thickness = 2 * u.nm
    S.add_substrate(substrate, 1000)
    sim = simulation(cls, S, np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg)
    tree = sim._homogeneous_layer_tree()
    # the superstrate, the layers of the structure, and the substrate
    assert len(_layer_runs(tree)) == 5
    # the same layers as distinct entries, which are not combined
    layers = [(*entry,) for entry in _flatten_layers(tree)]
    assert len(_layer_runs(layers)) == len(layers) == 1007
    for repeated, single in zip(sim.calc_jones_matrices(tree), sim.calc_jones_matrices(layers)):
        assert relative(repeated, single) < 1e-12
    z = np.linspace(-5e-9, 30e-9, 36)
    repeated = sim.calc_stack_fields(tree, z)
    single = sim.calc_stack_fields(layers, z)
    for name in ("E", "H", "K", "S"):
        assert relative(repeated[name], single[name]) < 1e-12


# ============================================================================
# azimuth
# ============================================================================


def biaxial_layer(euler_angles):
    """a tilted biaxial absorbing layer on glass"""
    eps0 = Dispersion.from_ref_index([1.66 + 0.05j, 1.49 + 0.1j, 1.7 + 0.02j])
    layer = substrate_layer(eps0, euler_angles)
    layer.structural.thickness = 200 * u.nm
    S = Structure("biaxial")
    S.add_sub_structure(layer, 1)
    S.add_substrate(substrate_layer(Dispersion.from_ref_index(1.5)))
    return S


def beam_frame(vector, azimuth):
    """components of a vector of the sample in the frame of the beam"""
    c, s = np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))
    return (c * vector[0] + s * vector[1], -s * vector[0] + c * vector[1], vector[2])


@pytest.mark.parametrize("cls", [BerremanElzo, BerremanStepanov])
def test_azimuth_of_anisotropic_layer(cls):
    """the azimuth of the plane of incidence equals the opposite in-plane
    rotation of the crystal by its Euler angle phi"""
    energy, theta = np.r_[1.2, 1.55] * u.eV, np.r_[20, 45, 70] * u.deg
    sim = simulation(cls, biaxial_layer((20, 10, 15) * u.deg), energy, theta)
    sim.azimuth = 35 * u.deg
    with pytest.raises(TypeError, match="azimuth must be given"):
        sim.azimuth = 35
    rotated = simulation(cls, biaxial_layer((20, 10 - 35, 15) * u.deg), energy, theta)
    np.testing.assert_allclose(
        sim.calc_homogeneous_jones_matrices()[0],
        rotated.calc_homogeneous_jones_matrices()[0],
        atol=1e-12,
    )
    z = np.linspace(0, 200e-9, 11)
    K = sim.calc_homogeneous_absorption(z)[2]
    assert relative(K, rotated.calc_homogeneous_absorption(z)[2]) < 1e-12


@pytest.mark.parametrize("cls", MAGNETIC)
def test_azimuth_of_magnetic_layer(cls):
    """the azimuth equals the magnetization in the frame of the beam, also for
    a magnetization map, while the fields stay in the frame of the sample"""
    energy, theta = np.r_[1.55] * u.eV, np.r_[30, 60] * u.deg
    m = np.array([0.3, 0.5, 0.81]) / np.linalg.norm([0.3, 0.5, 0.81])
    sim = simulation(cls, magnetic_film(tuple(m)), energy, theta)
    sim.azimuth = 35 * u.deg
    with pytest.raises(TypeError, match="azimuth must be given"):
        sim.azimuth = 35
    assert sim.azimuth == 35 * u.deg
    rotated = simulation(cls, magnetic_film(beam_frame(m, 35)), energy, theta)
    np.testing.assert_allclose(
        sim.calc_homogeneous_jones_matrices()[0],
        rotated.calc_homogeneous_jones_matrices()[0],
        atol=1e-12,
    )
    z = np.linspace(-2e-9, 12e-9, 15)
    E, _, H, _ = sim.calc_homogeneous_fields(z)
    E_rotated, _, H_rotated, _ = rotated.calc_homogeneous_fields(z)
    np.testing.assert_allclose(sim.calc_fields_in_beam_frame(E), E_rotated, atol=1e-12)
    np.testing.assert_allclose(sim.calc_fields_in_beam_frame(H), H_rotated, atol=1e-12)
    # the fields are rotated back into the frame of the sample
    assert relative(E, E_rotated) > 0.1
    # the magnetization map in the frame of the sample
    strain_map = np.zeros((2, 1))
    magnetization_map = np.tile(m, (2, 1, 1))
    rotated_map = np.tile(beam_frame(m, 35), (2, 1, 1))
    np.testing.assert_allclose(
        sim.calc_inhomogeneous_jones_matrices(strain_map, magnetization_map)[0],
        rotated.calc_inhomogeneous_jones_matrices(strain_map, rotated_map)[0],
        atol=1e-12,
    )


def test_azimuth_parameters():
    """the azimuth does not change isotropic layers, is part of the hash
    only if it is set, and is a single angle"""
    S = optical_stack(substrate_layer())
    sim = simulation(Parratt, S, np.r_[1.55] * u.eV, np.r_[20, 45] * u.deg)
    jones = sim.calc_homogeneous_jones_matrices()[0]
    hash = sim.get_hash()
    assert sim.azimuth == 0 * u.deg
    sim.azimuth = np.pi / 4 * u.rad
    assert sim.azimuth.to("deg").magnitude == pytest.approx(45)
    np.testing.assert_allclose(sim.calc_homogeneous_jones_matrices()[0], jones, atol=1e-15)
    assert sim.get_hash() != hash
    assert sim._parameters()["azimuth"] == sim.azimuth
    assert "azimuth" in repr(sim)
    sim.azimuth = 0 * u.deg
    assert sim.get_hash() == hash
    with pytest.raises(ValueError, match="single"):
        sim.azimuth = [0, 10] * u.deg


@pytest.mark.parametrize("cls", FORMALISMS)
def test_absorption_with_roughness(cls):
    """the reflected flux in the superstrate agrees with the reflectivity also
    for rough interfaces and repeated sub-structures"""
    S = optical_stack()
    for layer in S.unique_layers[1]:
        layer.structural.roughness = 0.5 * u.nm
    stack = Structure("periodic")
    stack.add_sub_structure(S, 20)
    stack.add_substrate(substrate_layer())
    sim = simulation(cls, stack, np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg)
    S = sim.calc_stack_fields(sim.calc_homogeneous_layers(), np.r_[-1e-12])["S"]
    for pol in POLARIZATIONS.values():
        np.testing.assert_allclose(flux(S[:, :, 0], pol), 1 - reflectivity(sim, pol), rtol=1e-10)


def test_homogeneous_absorption_result():
    """result of the absorption and the fields with their coordinates,
    aliases, units, and polarizations"""
    S = optical_stack(substrate_layer())
    sim = simulation(Elzo, S, np.r_[1.55, 3.1] * u.eV, np.r_[20, 45, 90] * u.deg)
    res = sim.homogeneous_absorption()
    assert res.dims("absorption") == ("energy", "angles", "distances")
    assert res.unit("absorption") == "1/m"
    centers = S.layer_distances[2].to("m").magnitude
    np.testing.assert_allclose(res["distances"], centers)
    assert res.absorption.shape == (2, 3, S.num_layers)
    assert res.absorption_matrix.shape == (2, 3, S.num_layers, 2, 2)
    np.testing.assert_array_equal(res.dAdz, sim.calc_homogeneous_absorption()[0])
    np.testing.assert_allclose(res.dAdz, np.real(res.K[..., 0, 0]))
    # the opposite magnetization only on request
    assert "absorption_phi" not in res and "absorption_matrix_phi" not in res
    assert sim.calc_homogeneous_absorption()[1] is None
    assert sim.calc_homogeneous_fields()[3] is None
    res_phi = sim.homogeneous_absorption(dichroic=True)
    np.testing.assert_allclose(res_phi.dAdz_phi, res.dAdz)
    np.testing.assert_allclose(res_phi.K_phi, res.K)

    sim.set_polarization("unpolarized", "none")
    np.testing.assert_allclose(
        sim.homogeneous_absorption().dAdz, np.real(res.K[..., 0, 0] + res.K[..., 1, 1]) / 2
    )
    pols = [(0 * u.deg, 0), (90 * u.deg, 0), (45 * u.deg, 1)]
    sim.set_polarization("elliptical", "none", pols)
    assert sim.homogeneous_absorption().dAdz.shape == (2, 3, S.num_layers, 3)

    res = sim.homogeneous_fields(np.r_[0, 1, 2] * u.nm)
    assert res.E.shape == res.H.shape == (2, 3, 3, 3, 2)
    # the fields together with the absorption
    res_both = sim.homogeneous_absorption(np.r_[0, 1, 2] * u.nm, fields=True)
    np.testing.assert_allclose(res_both.E, res.E)
    np.testing.assert_allclose(res_both.H, res.H)
    assert "electric_field_phi" not in res_both
    assert len(sim.calc_homogeneous_absorption()) == 12
    assert len(sim.calc_homogeneous_absorption(fields=True)) == 16
    np.testing.assert_allclose(res["distances"], [0, 1e-9, 2e-9])
    np.testing.assert_array_equal(res.E, sim.calc_homogeneous_fields(np.r_[0, 1e-9, 2e-9])[0])

    strained = sim.homogeneous_absorption(strains=np.full((S.num_sub_structures, 1), 0.01))
    np.testing.assert_allclose(strained["distances"], 1.01 * centers)
    with pytest.raises(NotImplementedError):
        sim.homogeneous_absorption(temps=np.full(S.num_sub_structures, 300))


def test_aliases_of_field_results(magnetic_multilayer):
    """the field and absorption results have the short aliases of ALIASES"""
    from udkm1Dsim.simulations.scattering.scattering import ALIASES

    sim = simulation(Elzo, magnetic_multilayer, np.r_[707] * u.eV, np.r_[5, 10] * u.deg)
    for result in (
        sim.homogeneous_fields(dichroic=True),
        sim.homogeneous_absorption(dichroic=True, fields=True),
    ):
        for name in result.keys():
            assert result.aliases_of(name) == (ALIASES[name],)
            assert result[ALIASES[name]] is result[name]


def ferrimagnetic_stack(gyrotropy):
    """a ferrimagnetic layer with a gyrotropic dispersion per sublattice on a
    magnetic layer with a single sublattice"""
    ferri = Layer(
        "ferri",
        "ferrimagnet",
        thickness=5 * u.nm,
        density=8000 * u.kg / u.m**3,
        eff_spin=[0.5, 0.5],
        mf_exch_coupling=np.array([[2e-21, -1e-21], [-1e-21, 2e-21]]) * u.J,
        mag_saturation=[3e5, 1e5] * u.A / u.m,
        magnetization=[(0, 1, 0), (0, -1, 0)],
        dielectric=Dielectric(
            eps0=Dispersion.from_ref_index(0.99 + 0.01j),
            eps_mag=[
                Dispersion.from_array(None, 2e-3 + 1e-3j, kind="eps"),
                Dispersion.from_array(None, -1e-3 + 3e-3j, kind="eps"),
            ],
            gyrotropy=gyrotropy,
        ),
    )
    fe = Layer(
        "fe",
        "fe",
        thickness=3 * u.nm,
        density=7874 * u.kg / u.m**3,
        magnetization=(0, 1, 0),
        dielectric=Dielectric(
            eps0=Dispersion.from_ref_index(0.98 + 0.02j),
            eps_mag=Dispersion.from_array(None, 4e-3j, kind="eps"),
        ),
    )
    S = Structure("ferrimagnetic stack")
    S.add_sub_structure(ferri, 2)
    S.add_sub_structure(fe, 2)
    S.add_substrate(Vacuum())
    return S


@pytest.mark.parametrize("cls", [Elzo, BerremanStepanov])
def test_sublattice_magnetization_maps(cls):
    """the gyrotropic part of every sublattice couples to its magnetization,
    which equals the net one for parallel sublattices"""
    energy, theta = np.r_[700] * u.eV, np.r_[5, 20] * u.deg
    # parallel sublattices, padded to two sublattices for the single one of fe
    sublattice_map = np.zeros([2, 4, 2, 3])
    sublattice_map[0, :, 0, 1] = 1
    sublattice_map[0, :2, 1, 1] = 1
    sublattice_map[1] = -sublattice_map[0]
    net_map = sublattice_map[:, :, 0, :]

    sublattices = simulation(cls, ferrimagnetic_stack("sublattices"), energy, theta)
    net = simulation(cls, ferrimagnetic_stack("net"), energy, theta)
    R_sub = sublattices.calc_inhomogeneous_reflectivity_transmissivity(
        magnetization_map=sublattice_map
    )[0]
    R_net = net.calc_inhomogeneous_reflectivity_transmissivity(magnetization_map=net_map)[0]
    np.testing.assert_allclose(R_sub, R_net, rtol=1e-12)
    # also as result of Magnetization.get_sublattice_magnetization_map()
    result = Result()
    result.add(
        "sublattice_magnetization_map",
        sublattice_map,
        ("delays", "distances", "sublattices", "components"),
    )
    R_result = sublattices.calc_inhomogeneous_reflectivity_transmissivity(
        magnetization_map=result
    )[0]
    np.testing.assert_allclose(R_result, R_sub)
    # the antiparallel sublattices of the layer differ for both models
    R_static = sublattices.homogeneous_reflectivity_transmissivity().R
    R_static_net = net.homogeneous_reflectivity_transmissivity().R
    assert relative(R_static, R_static_net) > 1e-6
    # the net map cannot give the magnetizations of the sublattices
    with pytest.raises(ValueError, match="Layer 'ferri'.*requires the magnetizations of all"):
        sublattices.calc_inhomogeneous_reflectivity_transmissivity(magnetization_map=net_map)
