#!/usr/bin/env python

import warnings

import numpy as np
import pytest

import udkm1Dsim as ud
from udkm1Dsim import (
    AmorphousLayer,
    Atom,
    BerremanElzo,
    BerremanStepanov,
    Dielectric,
    Dispersion,
    Elzo,
    Parratt,
    Stepanov,
    Structure,
    Vacuum,
    u,
)
from udkm1Dsim.simulations.scattering.berreman import berreman_modes
from udkm1Dsim.simulations.scattering.elzo import elzo_modes
from udkm1Dsim.simulations.scattering.multilayer import MultilayerScattering

FORMALISMS = [Parratt, Elzo, Stepanov, BerremanElzo, BerremanStepanov]
MAGNETIC = [Elzo, Stepanov, BerremanElzo, BerremanStepanov]
N_SUB = 1.5 + 0.1j


def simulation(cls, S, energy, theta):
    sim = cls(S, True, disp_messages=False, progress_bar=False, save_data=False)
    sim.energy = energy
    sim.theta = theta
    return sim


def relative(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def substrate_layer(eps0=None, euler_angles=(0, 0, 0)):
    sub = AmorphousLayer("sub", "substrate", thickness=0 * u.nm, density=1000 * u.kg / u.m**3)
    eps0 = Dispersion.from_ref_index(N_SUB) if eps0 is None else eps0
    sub.optical.dielectric = Dielectric(eps0=eps0, euler_angles=euler_angles)
    return sub


def interface(substrate):
    """a single interface of vacuum and the substrate"""
    S = Structure("interface")
    S.add_sub_structure(Vacuum(thickness=0 * u.nm), 1)
    S.add_substrate(substrate)
    return S


@pytest.fixture(scope="module")
def magnetic_multilayer():
    """Fe/O multilayer at the Fe L3 edge with an in-plane magnetization"""
    Fe = Atom("Fe", magnetization={"r": 1, "phi": 90 * u.deg, "gamma": 30 * u.deg})
    fe = AmorphousLayer("fe", "fe", thickness=2 * u.nm, density=7874 * u.kg / u.m**3, atom=Fe)
    fe.set_dielectric_from_atoms()
    ox = AmorphousLayer(
        "ox", "ox", thickness=1.3 * u.nm, density=3000 * u.kg / u.m**3, atom=Atom("O")
    )
    ox.set_dielectric_from_atoms()
    S = Structure("magnetic multilayer")
    S.add_sub_structure(fe, 3)
    S.add_sub_structure(ox, 2)
    S.add_sub_structure(fe, 2)
    S.add_substrate(ox)
    return S


def optical_stack(substrate=None, thickness=1):
    """non-magnetic optical stack of a metal and a dielectric"""
    a = AmorphousLayer("a", "metal", thickness=3 * thickness * u.nm, density=7874 * u.kg / u.m**3)
    a.optical.dielectric = Dispersion.from_ref_index(2.9 + 3.4j)
    b = AmorphousLayer("b", "oxide", thickness=2 * thickness * u.nm, density=2500 * u.kg / u.m**3)
    b.optical.dielectric = Dispersion.from_ref_index(1.45)
    S = Structure("optical stack")
    S.add_sub_structure(a, 2)
    S.add_sub_structure(b, 3)
    S.add_sub_structure(a, 1)
    S.add_substrate(Vacuum() if substrate is None else substrate)
    return S


def magnetic_film(magnetization):
    """Fe film with a gyrotropic dielectric at 800 nm"""
    fe = AmorphousLayer("fe", "fe", thickness=10 * u.nm, density=7874 * u.kg / u.m**3)
    fe.optical.dielectric = Dielectric(
        eps0=Dispersion.from_ref_index(2.9 + 3.4j),
        eps_mag=Dispersion.from_array([0, 10], [0.6 + 0.2j] * 2, kind="eps"),
    )
    fe.magnetization = magnetization
    S = Structure("film")
    S.add_sub_structure(fe, 1)
    S.add_substrate(Vacuum())
    return S


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
    R, R_phi, T, T_phi = sim.homogeneous_jones_matrices()
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
    R, _, T, _ = sim.homogeneous_jones_matrices()
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
    R = sim.homogeneous_jones_matrices()[0][0, 0]
    r = [(n_sup - n) / (n_sup + n) for n in np.sqrt(eps_xx + np.r_[1, -1] * 1j * eps_xy)]
    chi = abs((r[0] - r[1]) / (r[0] + r[1]))
    assert abs(R[1, 0] / R[0, 0]) == pytest.approx(chi, rel=1e-10)
    assert abs(R[0, 1] / R[1, 1]) == pytest.approx(chi, rel=1e-10)


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
        rough.homogeneous_jones_matrices()[0][0, :, 0, 0],
        smooth.homogeneous_jones_matrices()[0][0, :, 0, 0] * np.exp(-2 * kz0 * kz1 * (20e-9) ** 2),
    )


# ============================================================================
# comparison of the formalisms
# ============================================================================


@pytest.mark.parametrize("cls", [Parratt, Stepanov, BerremanElzo, BerremanStepanov])
@pytest.mark.parametrize("vacuum_substrate", [True, False])
def test_non_magnetic_stacks_equal_elzo(cls, vacuum_substrate):
    """all formalisms give the same Jones matrices for isotropic and
    non-magnetic layers in the optical range"""
    substrate = None if vacuum_substrate else substrate_layer()
    S = optical_stack(substrate)
    energy, theta = np.r_[1.55, 3.1] * u.eV, np.r_[5:85:10] * u.deg
    reference = simulation(Elzo, S, energy, theta).homogeneous_jones_matrices()
    result = simulation(cls, S, energy, theta).homogeneous_jones_matrices()
    for a, b in zip(result, reference):
        assert relative(a, b) < 1e-12


def test_parratt_x_rays_equal_elzo(magnetic_multilayer):
    """Parratt equals Elzo for the non-magnetic X-ray multilayer for all
    polarizations"""
    energy, theta = np.r_[700, 707] * u.eV, np.r_[0.5:30:2] * u.deg
    parratt = simulation(Parratt, magnetic_multilayer, energy, theta)
    elzo = simulation(Elzo, magnetic_multilayer, energy, theta)
    L = magnetic_multilayer.get_number_of_sub_structures()
    no_magnetization = np.zeros([L, 3])
    for pol_in, pol_out in [(0, 0), (1, 0), (3, 0), (4, 0), (1, 2), (3, 4)]:
        parratt.set_polarization(pol_in, pol_out)
        elzo.set_polarization(pol_in, pol_out)
        for a, b in zip(
            parratt.homogeneous_reflectivity_transmissivity(np.zeros(L), no_magnetization),
            elzo.homogeneous_reflectivity_transmissivity(np.zeros(L), no_magnetization),
        ):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)


def test_stepanov_equals_elzo(magnetic_multilayer):
    """Stepanov assembles the same eigenmodes as Elzo"""
    energy, theta = np.r_[700, 707, 710] * u.eV, np.r_[0.5:40:1.5] * u.deg
    reference = simulation(Elzo, magnetic_multilayer, energy, theta).homogeneous_jones_matrices()
    result = simulation(Stepanov, magnetic_multilayer, energy, theta).homogeneous_jones_matrices()
    for a, b in zip(result, reference):
        assert relative(a, b) < 1e-12


@pytest.mark.parametrize("cls", [BerremanElzo, BerremanStepanov])
def test_berreman_x_rays_close_to_elzo(cls, magnetic_multilayer):
    """the exact modes agree with the first-order modes of Elzo to second
    order in the gyrotropic part for all polarizations"""
    energy, theta = np.r_[700, 707, 710] * u.eV, np.r_[0.5:40:1.5] * u.deg
    elzo = simulation(Elzo, magnetic_multilayer, energy, theta)
    berreman = simulation(cls, magnetic_multilayer, energy, theta)
    for a, b in zip(berreman.homogeneous_jones_matrices(), elzo.homogeneous_jones_matrices()):
        assert relative(a, b) < 1e-2
    for pol_in, pol_out in [(1, 0), (2, 0), (3, 0), (4, 0), (3, 4)]:
        elzo.set_polarization(pol_in, pol_out)
        berreman.set_polarization(pol_in, pol_out)
        R, R_phi, _, _ = elzo.homogeneous_reflectivity_transmissivity()
        R_b, R_b_phi, _, _ = berreman.homogeneous_reflectivity_transmissivity()
        np.testing.assert_allclose(R_b, R, rtol=3e-2)
        # the magnetic asymmetry up to second order at the Fe L3 resonance
        np.testing.assert_allclose(
            (R_b - R_b_phi) / (R_b + R_b_phi), (R - R_phi) / (R + R_phi), atol=5e-3
        )


@pytest.mark.parametrize(
    "magnetization", [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0.3, -0.5, 0.8)], ids=str
)
def test_magneto_optics_elzo_and_berreman(magnetization):
    """the first-order modes of Elzo with the division of the gyrotropic part
    by eps_00 agree with the exact Berreman modes for visible light"""
    S = magnetic_film(magnetization)
    energy, theta = np.r_[1.55] * u.eV, np.r_[10:85:15] * u.deg
    elzo = simulation(Elzo, S, energy, theta).homogeneous_jones_matrices()
    exact = simulation(BerremanStepanov, S, energy, theta).homogeneous_jones_matrices()
    for a, b in zip(elzo, exact):
        assert relative(a, b) < 1e-3
    # the polarization conversion by the magnetization, which vanishes for
    # the transverse magnetization along x
    conversion = np.max(np.abs(exact[0][..., 1, 0]))
    assert (conversion > 1e-4) == (magnetization != (1, 0, 0))
    np.testing.assert_allclose(
        elzo[0][..., 1, 0], exact[0][..., 1, 0], atol=max(1e-2 * conversion, 1e-14)
    )


def test_kerr_angles():
    """the Kerr angles reverse with the magnetization and equal the ratio of
    the Jones matrix elements for small angles"""
    S = magnetic_film((0, 0, 1))
    sim = simulation(BerremanStepanov, S, np.r_[1.55] * u.eV, np.r_[30, 60] * u.deg)
    R, R_phi, _, _ = sim.homogeneous_jones_matrices()
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


@pytest.mark.parametrize("cls", MAGNETIC)
def test_rotation_ellipticity_from_one_calculation(cls, magnetic_multilayer):
    sim = simulation(cls, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[5, 10] * u.deg)
    L = magnetic_multilayer.get_number_of_layers()
    R, R_phi, T, T_phi = sim.homogeneous_jones_matrices()
    for pol, pi_incident in [("sigma", False), ("pi", True)]:
        sim.set_polarization(pol, "none")
        angles = sim.homogeneous_rotation_ellipticity()
        for (rotation, ellipticity), jones in zip(angles, (R, R_phi, T, T_phi)):
            reference = sim.calc_rotation_ellipticity(jones, pi_incident)
            np.testing.assert_allclose(rotation.m, reference[0].m)
            np.testing.assert_allclose(ellipticity.m, reference[1].m)
        # all delays of a homogeneous map equal the homogeneous result
        maps = (np.zeros((2, L)), np.zeros((2, L, 1)))
        delays = sim.inhomogeneous_rotation_ellipticity(*maps)
        for (rotation, ellipticity), (reference, _) in zip(delays, angles):
            np.testing.assert_allclose(rotation.m, np.stack([reference.m] * 2), atol=1e-12)
    jones = sim.inhomogeneous_jones_matrices(*maps)
    for a, b in zip(jones, (R, R_phi, T, T_phi)):
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
    sub = substrate_layer(Dispersion.from_ref_index([n_o, n_e]), euler_angles)
    eps = np.full(3, n_o**2)
    eps[axis] = n_e**2
    theta = np.r_[5:85:10]
    R = simulation(cls, interface(sub), 1.55 * u.eV, theta * u.deg).homogeneous_jones_matrices()[0]
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
    sub = substrate_layer(Dispersion.from_ref_index([1.66, 1.49]), (45, 30, 0))
    S = interface(sub)
    S.add_sub_structure(substrate_layer(Dispersion.from_ref_index(1.3)), 1)
    energy, theta = np.r_[1.55, 2.5] * u.eV, np.r_[10:80:10] * u.deg
    elzo = simulation(BerremanElzo, S, energy, theta).homogeneous_jones_matrices()
    stepanov = simulation(BerremanStepanov, S, energy, theta).homogeneous_jones_matrices()
    for a, b in zip(elzo, stepanov):
        assert relative(a, b) < 1e-10
    assert np.max(np.abs(stepanov[0][..., 1, 0])) > 1e-3


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
    sub = AmorphousLayer("sub", "substrate", thickness=1 * u.nm, density=1000 * u.kg / u.m**3)
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
        np.testing.assert_allclose(result[0], parratt[0], atol=1e-12)
        np.testing.assert_allclose(result[2], parratt[2], atol=1e-14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elzo = simulation(Elzo, S, energy, theta).homogeneous_jones_matrices()
    # the exponentially growing terms of the up modes overflow
    assert not np.all(np.isfinite(elzo[0]))


# ============================================================================
# inhomogeneous reflectivity
# ============================================================================


@pytest.mark.parametrize("cls", FORMALISMS)
def test_inhomogeneous_equals_homogeneous(cls, magnetic_multilayer, tmp_path):
    energy, theta = np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg
    sim = simulation(cls, magnetic_multilayer, energy, theta)
    sim.set_polarization(1, 0)
    sim.cache_dir = tmp_path
    R, R_phi, T, T_phi = sim.homogeneous_reflectivity_transmissivity()
    L = magnetic_multilayer.get_number_of_layers()
    magnetization_map = np.zeros((2, L, 3))
    magnetization_map[..., 0] = 1
    magnetization_map[..., 1] = np.pi / 2
    magnetization_map[..., 2] = np.pi / 6
    result = sim.inhomogeneous_reflectivity_transmissivity(
        strain_map=np.zeros((2, L)), magnetization_map=magnetization_map, polar=True
    )
    for a, b in zip(result, (R, R_phi, T, T_phi)):
        np.testing.assert_allclose(a[1], b, rtol=1e-10)
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
        cached[0][0], sim.homogeneous_reflectivity_transmissivity()[0], rtol=1e-10
    )


@pytest.mark.parametrize("cls", [Stepanov, BerremanStepanov])
def test_inhomogeneous_equals_elzo(cls, magnetic_multilayer):
    """strained and magnetized layers of the inhomogeneous reflectivity"""
    energy, theta = np.r_[707] * u.eV, np.r_[1:20:4] * u.deg
    L = magnetic_multilayer.get_number_of_layers()
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
    elzo = simulation(Elzo, magnetic_multilayer, energy, theta)
    result = simulation(cls, magnetic_multilayer, energy, theta)
    for sim in (elzo, result):
        sim.set_polarization(1, 0)
    reference = elzo.inhomogeneous_reflectivity_transmissivity(
        strain_map, magnetization_map, polar=True
    )
    rtol = 1e-10 if cls is Stepanov else 3e-2
    for a, b in zip(
        result.inhomogeneous_reflectivity_transmissivity(
            strain_map, magnetization_map, polar=True
        ),
        reference,
    ):
        np.testing.assert_allclose(a, b, rtol=rtol)


@pytest.mark.parametrize("pol_in", [1, 3])
def test_parratt_inhomogeneous_for_all_delays_at_once(pol_in):
    """the recursion of Parratt over the layers for all delays at once equals
    the one for every delay and Stepanov for strained, rough layers"""
    S = optical_stack(substrate_layer(Dispersion.from_ref_index(1.5 + 0.01j)))
    S.get_layer_handle(0).roughness = 0.5 * u.nm
    energy, theta = np.r_[1.55, 2.5] * u.eV, np.r_[10:80:20] * u.deg
    L = S.get_number_of_layers()
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
    for a, b, c, d in zip(result, single, smooth, reference):
        assert a.shape == d.shape
        np.testing.assert_allclose(a, b, rtol=1e-12)
        np.testing.assert_allclose(c, d, rtol=1e-10)


def test_inhomogeneous_input_validation(magnetic_multilayer):
    sim = simulation(Stepanov, magnetic_multilayer, np.r_[707] * u.eV, np.r_[5] * u.deg)
    L = magnetic_multilayer.get_number_of_layers()
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
    L = magnetic_multilayer.get_number_of_layers()
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
    N = magnetic_multilayer.get_number_of_sub_structures()
    for method in (
        sim.homogeneous_jones_matrices,
        sim.homogeneous_reflectivity_transmissivity,
        sim.homogeneous_rotation_ellipticity,
    ):
        with pytest.raises(NotImplementedError, match="temps"):
            method(temps=np.full(N, 300.0))
    for a, b in zip(
        sim.homogeneous_jones_matrices(np.zeros(N), temps=[]), sim.homogeneous_jones_matrices()
    ):
        np.testing.assert_array_equal(a, b)
    reference = sim.inhomogeneous_reflectivity_transmissivity(strain_map)
    result = sim.inhomogeneous_reflectivity_transmissivity(strain_map, temp_map=np.array([]))
    for a, b in zip(result, reference):
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
    layers = sim.get_homogeneous_layers([0.1, 0.2, 0.3], np.zeros([3, 1]))
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
        setter(5, (0, 0))
        with pytest.raises(ValueError):
            setter(5, (0, 2))


def test_polarization_consistency(elzo):
    elzo.set_polarization(5, 5, [(0, 0), (1, 0), (2, 0)], [(0, 0), (1, 0), (2, 0)])
    assert elzo.num_pol == 3
    # changing only one side to a different number of polarizations must fail
    # and keep the previous state
    with pytest.raises(ValueError, match="set_polarization"):
        elzo.set_outgoing_polarization(5, [(0, 0), (1, 0)])
    assert elzo.pol_out.shape == (3, 2)
    with pytest.raises(ValueError, match="set_polarization"):
        elzo.set_incoming_polarization(5, [(0, 0), (1, 0)])
    assert elzo.pol_in.shape == (3, 2)
    # a single polarization on one side is always allowed
    elzo.set_outgoing_polarization(3)
    assert elzo.num_pol == 3
    elzo.set_outgoing_polarization(0)
    assert elzo.num_pol == 3
    # changing both at the same time
    elzo.set_polarization(5, 5, [(0, 0), (1, 0)], [(0, 0), (1, 0)])
    assert elzo.num_pol == 2
    with pytest.raises(ValueError):
        elzo.set_polarization(5, 5, [(0, 0), (1, 0)], [(0, 0), (1, 0), (2, 0)])
    assert elzo.num_pol == 2
    elzo.set_polarization(3, 0)
    assert elzo.num_pol == 1


def test_get_hash_of_elliptical_polarizations(elzo):
    elzo.set_polarization(5, 0, (0 * u.deg, 0))
    hash_1 = elzo.get_hash()
    elzo.set_polarization(5, 0, (45 * u.deg, 0))
    hash_2 = elzo.get_hash()
    elzo.set_polarization(5, 0, [(0 * u.deg, 0), (45 * u.deg, 0)])
    hash_3 = elzo.get_hash()
    assert len({hash_1, hash_2, hash_3}) == 3


def test_inhomogeneous_for_several_polarizations(elzo, magnetic_multilayer):
    L = magnetic_multilayer.get_number_of_layers()
    strain_map = np.full((2, L), 0.01)
    magnetization_map = np.zeros((2, L, 3))
    elzo.set_incoming_polarization(5, [(0 * u.deg, 0), (45 * u.deg, 0), (90 * u.deg, 0)])
    elzo.set_outgoing_polarization(5, [(0, 0)])
    assert elzo.num_pol == 3
    R, _, _, _ = elzo.inhomogeneous_reflectivity_transmissivity(strain_map, magnetization_map)
    assert R.shape == (2,) + elzo.experiment.qz.magnitude.shape + (3,)
    # single polarization results equal the respective multi-polarization slices
    elzo.set_incoming_polarization(5, (45 * u.deg, 0))
    R_single, _, _, _ = elzo.inhomogeneous_reflectivity_transmissivity(
        strain_map, magnetization_map
    )
    assert R_single.ndim == 3
    np.testing.assert_allclose(R_single, R[..., 1])


def test_polar_and_cartesian_magnetization_maps(elzo, magnetic_multilayer):
    """the magnetization_map is cartesian by default and polar for
    polar=True, which give the same results"""
    elzo.set_polarization("circ +", "none")
    L = magnetic_multilayer.get_number_of_layers()
    strain_map = np.full((2, L), 0.01)
    polar = np.zeros([2, L, 3])
    polar[..., 0] = np.linspace(0.2, 1, L)
    polar[..., 1] = np.pi / 3
    polar[..., 2] = np.pi / 4
    cartesian = ud.helpers.convert_polar_to_cartesian(polar)
    R, R_phi, _, _ = elzo.inhomogeneous_reflectivity_transmissivity(strain_map, cartesian)
    R_polar, R_phi_polar, _, _ = elzo.inhomogeneous_reflectivity_transmissivity(
        strain_map, polar, polar=True
    )
    np.testing.assert_allclose(R_polar, R)
    np.testing.assert_allclose(R_phi_polar, R_phi)
    # the magnetization matters for circular polarization
    R_zero, _, _, _ = elzo.inhomogeneous_reflectivity_transmissivity(
        strain_map, np.zeros([2, L, 3])
    )
    assert not np.allclose(R, R_zero)
    # the homogeneous reflectivity of the sub-structures
    N = magnetic_multilayer.get_number_of_sub_structures()
    R_hom, _, _, _ = elzo.homogeneous_reflectivity_transmissivity(np.zeros(N), cartesian[0, :N])
    R_hom_polar, _, _, _ = elzo.homogeneous_reflectivity_transmissivity(
        np.zeros(N), polar[0, :N], polar=True
    )
    np.testing.assert_allclose(R_hom_polar, R_hom)


@pytest.mark.parametrize("cls", MAGNETIC)
def test_unpolarized_light_is_an_incoherent_average(cls, magnetic_multilayer):
    """unpolarized light is the incoherent average of sigma and pi, or of the
    circular polarizations, for any analyzer, also for magnetic layers with
    polarization conversion, and without analyzer and a transverse
    magnetization the average of a circular polarization for both directions
    of the magnetization"""
    sim = simulation(cls, magnetic_multilayer, np.r_[700, 707] * u.eV, np.r_[5:40:5] * u.deg)
    R, _, T, _ = sim.homogeneous_jones_matrices()
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
    N = magnetic_multilayer.get_number_of_sub_structures()
    for magnetization, equal in [((0, 0.6, 0.8), True), ((1, 0, 0), False)]:
        R_m, R_m_phi, _, _ = sim.homogeneous_jones_matrices(
            np.zeros(N), np.tile(magnetization, (N, 1))
        )
        average = (intensity(R_m, 1, 0) + intensity(R_m_phi, 1, 0)) / 2
        assert np.allclose(intensity(R_m, 0, 0), average, rtol=1e-10) == equal
    # the same for the high-level methods of the simulation
    sim.set_polarization("unpolarized", "sigma")
    reference = (intensity(R, 3, 3) + intensity(R, 4, 3)) / 2
    np.testing.assert_allclose(sim.homogeneous_reflectivity_transmissivity()[0], reference)


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
    for a, b in zip(*results):
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("cls", [Elzo, Parratt, Stepanov, BerremanElzo, BerremanStepanov])
def test_inhomogeneous_parallel_equals_sequential(cls, magnetic_multilayer, tmp_path):
    distributed = pytest.importorskip("distributed")
    energy, theta = np.r_[700, 707] * u.eV, np.r_[1:20:4] * u.deg
    sim = simulation(cls, magnetic_multilayer, energy, theta)
    sim.cache_dir = tmp_path
    L = magnetic_multilayer.get_number_of_layers()
    rng = np.random.default_rng(0)
    strain_map = rng.uniform(-0.01, 0.01, (3, L))
    magnetization_map = rng.uniform(-0.5, 0.5, (3, L, 3))
    reference = sim.inhomogeneous_reflectivity_transmissivity(strain_map, magnetization_map)
    with distributed.Client(processes=False, dashboard_address=":0") as client:
        result = sim.inhomogeneous_reflectivity_transmissivity(
            strain_map, magnetization_map, calc_type="parallel", dask_client=client
        )
    for a, b in zip(result, reference):
        np.testing.assert_allclose(a, b, rtol=1e-12)


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
