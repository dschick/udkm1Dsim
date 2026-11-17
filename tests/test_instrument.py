#!/usr/bin/env python

import numpy as np
import pytest
from scipy.special import erf

import udkm1Dsim as ud
from udkm1Dsim import AmorphousLayer, Atom, InstrumentParameters, Parratt, Structure, u
from udkm1Dsim.simulations.result import Result
from udkm1Dsim.simulations.scattering import instrument as ins

PROFILES = ["box", "gauss", "lorentz", "pseudo_voigt"]
# energies of the Cu K-alpha1 and K-alpha2 lines (eV)
K_ALPHA_1, K_ALPHA_2 = 8047.8, 8027.8


def simulation(cls, S, energy, theta, **kwargs):
    sim = cls(S, True, disp_messages=False, progress_bar=False, save_data=False, **kwargs)
    sim.energy = energy
    sim.theta = theta
    return sim


@pytest.fixture(scope="module")
def multilayer():
    """Fe/O multilayer on a substrate of O for X-rays"""
    fe = AmorphousLayer(
        "fe", "fe", thickness=2 * u.nm, density=7874 * u.kg / u.m**3, atom=Atom("Fe")
    )
    fe.set_dielectric_from_atoms()
    fe.roughness = 0.3 * u.nm
    ox = AmorphousLayer(
        "ox", "ox", thickness=1.3 * u.nm, density=3000 * u.kg / u.m**3, atom=Atom("O")
    )
    ox.set_dielectric_from_atoms()
    S = Structure("multilayer")
    S.add_sub_structure(fe, 1)
    S.add_sub_structure(ox, 1)
    ML = Structure("periods")
    ML.add_sub_structure(S, 5)
    ML.add_substrate(ox)
    return ML


@pytest.fixture
def parratt(multilayer):
    return simulation(Parratt, multilayer, [8000] * u.eV, np.linspace(0.2, 4, 101) * u.deg)


# ============================================================================
# profiles, nodes, convolution, and footprint
# ============================================================================


@pytest.mark.parametrize("shape", PROFILES)
def test_profiles_have_unit_fwhm_and_area(shape):
    x = np.linspace(-200, 200, 2000001)
    p = ins.profile(shape, x, eta=0.3)
    np.testing.assert_allclose(np.trapezoid(p, x), 1, atol=2e-3)
    if shape != "box":
        np.testing.assert_allclose(ins.profile(shape, 0.5, 0.3), ins.profile(shape, 0, 0.3) / 2)


def test_gauss_nodes_have_the_moments_of_the_profile():
    sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
    x, w = ins.profile_nodes("gauss", "gauss", 5)
    np.testing.assert_allclose(np.sum(w), 1)
    np.testing.assert_allclose(np.sum(w * x**2), sigma**2)
    np.testing.assert_allclose(np.sum(w * x**4), 3 * sigma**4)
    x, w = ins.profile_nodes("box", "gauss", 4)
    np.testing.assert_allclose(np.sum(w * x**2), 1 / 12)
    x, w = ins.profile_nodes("box", "uniform", 4)
    np.testing.assert_allclose(x, [-0.375, -0.125, 0.125, 0.375])
    np.testing.assert_allclose(w, 0.25)


@pytest.mark.parametrize("shape", PROFILES)
def test_uniform_nodes(shape):
    x, w = ins.profile_nodes(shape, "uniform", 9, 3, 0.3)
    np.testing.assert_allclose(np.sum(w), 1)
    np.testing.assert_allclose(w, w[::-1])
    assert ins.profile_nodes("none")[0].tolist() == [0]
    assert ins.profile_nodes(shape, points=1)[0].tolist() == [0]


@pytest.mark.parametrize("shape", ["lorentz", "pseudo_voigt"])
def test_gauss_nodes_only_for_gauss_and_box(shape):
    with pytest.raises(ValueError, match="uniform"):
        ins.profile_nodes(shape, "gauss")


def test_convolution_matrix():
    x = np.sort(np.r_[np.linspace(0, 10, 201), 3.333])
    K = ins.convolution_matrix(x, 0.5, "gauss", 3)
    np.testing.assert_allclose(K.sum(axis=1), 1)
    # a smooth function is kept apart from the curvature and the ends
    y = x**2
    inner = (x > 2) & (x < 8)
    sigma = 0.5 / (2 * np.sqrt(2 * np.log(2)))
    np.testing.assert_allclose((K @ y)[inner], y[inner] + sigma**2, rtol=1e-3)
    # widths below the spacing keep the values
    np.testing.assert_allclose(ins.convolution_matrix(x, 1e-6, "lorentz") @ y, y)
    # the convolution along an axis
    values = np.stack([y, 2 * y])
    np.testing.assert_allclose(ins.convolve(values, 1, K), np.stack([K @ y, 2 * (K @ y)]))


@pytest.mark.parametrize("shape", PROFILES)
def test_fast_convolution_equals_the_matrix(shape):
    """the fast convolution of a constant width on an equidistant and a
    non-equidistant grid, e.g. the angles of a qz-scan"""
    for x in [np.linspace(0, 10, 2001), np.arcsin(np.linspace(0.1, 0.3, 2001))]:
        y = np.exp(-((x - x.mean()) ** 2) / (0.1 * np.ptp(x)) ** 2) + 0.1 * np.sin(
            20 * x / np.ptp(x)
        )
        values = np.stack([y, 2 * y])
        width = 0.01 * np.ptp(x)
        fast = ins.convolve_on_grid(values, 1, x, width, shape, 2, 0.3)
        exact = ins.convolve(values, 1, ins.convolution_matrix(x, width, shape, 2, 0.3))
        np.testing.assert_allclose(fast, exact, atol=2e-3 * np.max(np.abs(values)))
    # a varying width and widths below the spacing
    x = np.linspace(0, 1, 101)
    width = np.linspace(0.01, 0.1, 101)
    np.testing.assert_allclose(
        ins.convolve_on_grid(x**2, 0, x, width, "gauss"),
        ins.convolution_matrix(x, width, "gauss") @ x**2,
    )
    np.testing.assert_allclose(ins.convolve_on_grid(x**2, 0, x, 1e-4, "gauss"), x**2)


def test_footprint_equals_genx():
    """the footprint of GenX, whose Gaussian beam width is the sigma"""
    theta = np.radians(np.linspace(-0.1, 3, 50))
    beam, length = 0.1, 10
    sigma = beam / (2 * np.sqrt(2 * np.log(2)))
    genx_square = np.minimum(length / beam * np.sin(theta), 1)
    genx_gauss = erf(length / 2 * np.sin(theta) / (np.sqrt(2) * sigma))
    positive = theta > 0
    np.testing.assert_allclose(
        ins.footprint("box", theta, beam, length)[positive], genx_square[positive]
    )
    np.testing.assert_allclose(
        ins.footprint("gauss", theta, beam, length)[positive], genx_gauss[positive]
    )
    assert np.all(ins.footprint("box", theta, beam, length)[~positive] == 0)
    np.testing.assert_allclose(ins.footprint("none", theta, beam, length), 1)
    with pytest.raises(ValueError):
        ins.footprint("box", theta, 0, length)


# ============================================================================
# instrument parameters
# ============================================================================


def test_instrument_parameters():
    instrument = InstrumentParameters()
    assert not instrument.is_active
    for name in ("theta_resolution", "energy_mode", "footprint", "theta_nodes"):
        with pytest.raises(ValueError, match=name):
            setattr(instrument, name, "unknown")
    instrument.theta_width = [0.1, 0.2]
    assert instrument.is_active
    instrument.theta_width = 0.01 * u.rad
    assert instrument.theta_width.magnitude == pytest.approx(np.degrees(0.01))
    instrument = InstrumentParameters()
    instrument.spectrum = [{"weight": 2}, {"offset": -20 * u.eV, "weight": 1, "width": 3}]
    assert instrument.is_active
    components = instrument.components
    assert [c["weight"] for c in components] == pytest.approx([2 / 3, 1 / 3])
    assert components[1] == pytest.approx(
        {"offset": -20, "factor": 1, "weight": 1 / 3, "width": 3}
    )
    with pytest.raises(ValueError, match="Unknown keys"):
        instrument.spectrum = [{"energy": 1}]
    with pytest.raises(ValueError, match="factor"):
        instrument.spectrum = [{"factor": 0}]
    assert "spectrum" in repr(instrument)
    assert "theta_mode" in instrument._repr_html_()


def test_inactive_instrument_keeps_the_hash(parratt):
    hash = parratt.get_hash()
    parratt.instrument.I0 = 1
    assert parratt.get_hash() == hash
    parratt.instrument.background = 1e-6
    assert parratt.get_hash() != hash


def test_metadata_of_the_instrument(parratt, tmp_path):
    parratt.instrument.theta_resolution = "gauss"
    parratt.instrument.theta_width = 0.02
    parratt.instrument.spectrum = [{"weight": 2}, {"offset": -20 * u.eV, "weight": 1}]
    result = parratt.homogeneous_reflectivity_transmissivity()
    instrument = result.attrs["parameters"]["instrument"]
    assert instrument["theta_resolution"] == "gauss"
    assert instrument["theta_width"].to("deg").magnitude == pytest.approx(0.02)
    result.save(tmp_path / "result.h5")
    loaded = Result.load(tmp_path / "result.h5")
    assert loaded.attrs["parameters"]["instrument"]["theta_resolution"] == "gauss"
    np.testing.assert_allclose(loaded.reflectivity, result.reflectivity)


# ============================================================================
# angular resolution, offsets, and footprint
# ============================================================================


@pytest.mark.parametrize(
    "shape,nodes",
    [
        ("gauss", "uniform"),
        ("gauss", "gauss"),
        ("box", "gauss"),
        ("lorentz", "uniform"),
        ("pseudo_voigt", "uniform"),
    ],
)
def test_theta_quadrature_is_the_weighted_average(parratt, shape, nodes):
    theta = parratt.theta.to("deg").magnitude
    R0 = parratt.calc_homogeneous_reflectivity_transmissivity()[0]
    parratt.instrument.theta_resolution = shape
    parratt.instrument.theta_nodes = nodes
    parratt.instrument.theta_width = 0.05
    parratt.instrument.theta_points = 5
    R = parratt.calc_homogeneous_reflectivity_transmissivity()[0]
    # the experiment is restored
    np.testing.assert_allclose(parratt.theta.to("deg").magnitude, theta)
    x, w = ins.profile_nodes(shape, nodes, 5, 2, 0.5)
    expected = 0
    for x_i, w_i in zip(x, w):
        parratt.instrument.theta_offset = x_i * 0.05
        parratt.instrument.theta_resolution = "none"
        expected = expected + w_i * parratt.calc_homogeneous_reflectivity_transmissivity()[0]
        parratt.instrument.theta_resolution = shape
    np.testing.assert_allclose(R, expected, rtol=1e-12)
    assert not np.allclose(R, R0, rtol=1e-3)


def test_theta_offset_footprint_I0_and_background(parratt):
    R0, _, T0, _ = parratt.calc_homogeneous_reflectivity_transmissivity()
    parratt.instrument.footprint = "box"
    parratt.instrument.beam_width = 0.1 * u.mm
    parratt.instrument.sample_length = 2 * u.mm
    parratt.instrument.I0 = 3
    parratt.instrument.background = 1e-5
    R, R_phi, T, T_phi = parratt.calc_homogeneous_reflectivity_transmissivity()
    foot = ins.footprint("box", parratt.theta.to("rad").magnitude, 0.1, 2)
    np.testing.assert_allclose(R, 3 * foot * R0 + 1e-5)
    np.testing.assert_allclose(R_phi, R)
    # the footprint is not applied to the transmission
    np.testing.assert_allclose(T, 3 * T0 + 1e-5)
    # an offset of the angles
    parratt.instrument.footprint = "none"
    parratt.instrument.I0 = 1
    parratt.instrument.background = 0
    parratt.instrument.theta_offset = 0.1
    R = parratt.calc_homogeneous_reflectivity_transmissivity()[0]
    parratt.instrument.theta_offset = 0
    parratt.theta = parratt.theta + 0.1 * u.deg
    np.testing.assert_allclose(R, parratt.calc_homogeneous_reflectivity_transmissivity()[0])


def test_theta_convolution_close_to_quadrature(multilayer):
    sim = simulation(Parratt, multilayer, [8000] * u.eV, np.linspace(0.2, 4, 3801) * u.deg)
    sim.instrument.theta_resolution = "gauss"
    sim.instrument.theta_width = 0.05
    sim.instrument.theta_range = 3
    sim.instrument.theta_mode = "convolution"
    R_conv = sim.calc_homogeneous_reflectivity_transmissivity()[0]
    sim.instrument.theta_mode = "quadrature"
    sim.instrument.theta_points = 61
    R_quad = sim.calc_homogeneous_reflectivity_transmissivity()[0]
    inner = (sim.theta.magnitude > 0.5) & (sim.theta.magnitude < 3.7)
    np.testing.assert_allclose(R_conv[inner], R_quad[inner], rtol=2e-3)


def test_theta_width_along_the_angles(parratt):
    width = np.linspace(0.01, 0.1, 101)
    parratt.instrument.theta_resolution = "gauss"
    parratt.instrument.theta_width = width
    R = parratt.calc_homogeneous_reflectivity_transmissivity()[0]
    parratt.instrument.theta_width = width[:50]
    with pytest.raises(ValueError, match="theta_width"):
        parratt.calc_homogeneous_reflectivity_transmissivity()
    assert R.shape == (1, 101)


# ============================================================================
# spectral resolution and spectrum
# ============================================================================


@pytest.fixture
def xray_dyn(structure_crystalline):
    return simulation(
        ud.XrayDyn, structure_crystalline, [K_ALPHA_1] * u.eV, np.linspace(30, 34, 201) * u.deg
    )


def test_spectrum_of_k_alpha_lines(xray_dyn):
    """the K-alpha lines are reflected at the same angles"""
    theta = xray_dyn.theta
    R1 = xray_dyn.calc_homogeneous_reflectivity()[0]
    xray_dyn.energy = [K_ALPHA_2] * u.eV
    xray_dyn.theta = theta
    R2 = xray_dyn.calc_homogeneous_reflectivity()[0]
    xray_dyn.energy = [K_ALPHA_1] * u.eV
    xray_dyn.theta = theta
    xray_dyn.instrument.spectrum = [
        {"weight": 2},
        {"offset": (K_ALPHA_2 - K_ALPHA_1) * u.eV, "weight": 1},
    ]
    R, A = xray_dyn.calc_homogeneous_reflectivity()
    np.testing.assert_allclose(R, (2 * R1 + R2) / 3, rtol=1e-12)
    # the matrices of the sub-structures are the ones without the instrument
    assert len(A) > 0
    # the high-level method and the cache
    result = xray_dyn.homogeneous_reflectivity()
    np.testing.assert_allclose(result.reflectivity, R)


def test_harmonic_of_the_spectrum(xray_dyn):
    xray_dyn.energy = [3000] * u.eV
    xray_dyn.theta = np.linspace(20, 40, 51) * u.deg
    R1 = xray_dyn.calc_homogeneous_reflectivity()[0]
    xray_dyn.energy = [9000] * u.eV
    xray_dyn.theta = np.linspace(20, 40, 51) * u.deg
    R3 = xray_dyn.calc_homogeneous_reflectivity()[0]
    xray_dyn.energy = [3000] * u.eV
    xray_dyn.theta = np.linspace(20, 40, 51) * u.deg
    xray_dyn.instrument.spectrum = [{"weight": 0.9}, {"factor": 3, "weight": 0.1}]
    R = xray_dyn.calc_homogeneous_reflectivity()[0]
    np.testing.assert_allclose(R, 0.9 * R1 + 0.1 * R3, rtol=1e-12)


def test_energy_qz_mode_close_to_quadrature(xray_dyn):
    """away from resonances, the spectral spread is a spread of the angle"""
    xray_dyn.instrument.energy_resolution = "gauss"
    xray_dyn.instrument.energy_width = 5
    xray_dyn.instrument.energy_points = 9
    R_quad = xray_dyn.calc_homogeneous_reflectivity()[0]
    xray_dyn.instrument.energy_mode = "qz"
    R_qz = xray_dyn.calc_homogeneous_reflectivity()[0]
    assert np.max(np.abs(R_qz - R_quad)) < 1e-2 * np.max(R_quad)


def test_energy_convolution_close_to_quadrature(multilayer):
    """an energy scan at a fixed angle"""
    sim = simulation(Parratt, multilayer, np.linspace(690, 740, 1001) * u.eV, [10] * u.deg)
    sim.instrument.energy_resolution = "gauss"
    sim.instrument.energy_width = 0.5
    sim.instrument.energy_range = 3
    sim.instrument.energy_mode = "convolution"
    R_conv = sim.calc_homogeneous_reflectivity_transmissivity()[0]
    sim.instrument.energy_mode = "quadrature"
    sim.instrument.energy_points = 61
    R_quad = sim.calc_homogeneous_reflectivity_transmissivity()[0]
    inner = (sim.energy.magnitude > 695) & (sim.energy.magnitude < 735)
    np.testing.assert_allclose(R_conv[inner], R_quad[inner], rtol=5e-3)
    # the angles of a convolution must be the same for all energies
    sim.qz = [0.5] * u.nm**-1
    sim.instrument.energy_mode = "convolution"
    with pytest.raises(ValueError, match="fixed angles"):
        sim.calc_homogeneous_reflectivity_transmissivity()


# ============================================================================
# rotation and ellipticity and inhomogeneous results
# ============================================================================


@pytest.fixture
def magnetic_elzo():
    Fe = ud.Atom("Fe", magnetization={"r": 1, "phi": 90 * u.deg, "gamma": 30 * u.deg})
    fe = AmorphousLayer("fe", "fe", thickness=5 * u.nm, density=7874 * u.kg / u.m**3, atom=Fe)
    fe.set_dielectric_from_atoms()
    S = Structure("magnetic film")
    S.add_sub_structure(fe, 2)
    return simulation(ud.Elzo, S, [708] * u.eV, np.linspace(5, 30, 26) * u.deg)


def test_rotation_ellipticity_of_a_single_node(magnetic_elzo):
    """the averaged polarization of a single node is the one without the
    instrument"""
    expected = magnetic_elzo.calc_homogeneous_rotation_ellipticity()
    magnetic_elzo.instrument.I0 = 2
    for value, reference in zip(magnetic_elzo.calc_homogeneous_rotation_ellipticity(), expected):
        np.testing.assert_allclose(
            value,
            reference.magnitude if hasattr(reference, "magnitude") else reference,
            atol=1e-10,
        )
    magnetic_elzo.instrument.energy_resolution = "gauss"
    magnetic_elzo.instrument.energy_width = 1
    rotation = magnetic_elzo.calc_homogeneous_rotation_ellipticity()[0]
    assert rotation.shape == (1, 26)
    assert not np.allclose(rotation, expected[0])


def test_inhomogeneous_equals_homogeneous_with_instrument(magnetic_elzo, tmp_path):
    magnetic_elzo.instrument.theta_resolution = "gauss"
    magnetic_elzo.instrument.theta_width = 1
    magnetic_elzo.instrument.energy_resolution = "lorentz"
    magnetic_elzo.instrument.energy_width = 0.5
    magnetic_elzo.instrument.energy_points = 5
    magnetic_elzo.instrument.background = 1e-6
    magnetization = magnetic_elzo.S.get_layer_property_vector("magnetization").magnitude
    mag_map = np.repeat(magnetization[np.newaxis], 3, axis=0)
    homogeneous = magnetic_elzo.calc_homogeneous_reflectivity_transmissivity(
        magnetizations=magnetization
    )
    inhomogeneous = magnetic_elzo.calc_inhomogeneous_reflectivity_transmissivity(
        magnetization_map=mag_map
    )
    for h, i in zip(homogeneous, inhomogeneous):
        np.testing.assert_allclose(i, np.repeat(h[np.newaxis], 3, axis=0), rtol=1e-10)
    rot_h = magnetic_elzo.calc_homogeneous_rotation_ellipticity(magnetizations=magnetization)
    rot_i = magnetic_elzo.calc_inhomogeneous_rotation_ellipticity(magnetization_map=mag_map)
    for h, i in zip(rot_h, rot_i):
        np.testing.assert_allclose(i, np.repeat(np.asarray(h)[np.newaxis], 3, axis=0), atol=1e-10)
    # the cached result
    magnetic_elzo.save_data = True
    magnetic_elzo.cache_dir = str(tmp_path)
    magnetic_elzo.force_recalc = False
    result = magnetic_elzo.inhomogeneous_reflectivity_transmissivity(magnetization_map=mag_map)
    np.testing.assert_allclose(result.reflectivity, inhomogeneous[0])
    loaded = magnetic_elzo.inhomogeneous_reflectivity_transmissivity(magnetization_map=mag_map)
    np.testing.assert_allclose(loaded.reflectivity, inhomogeneous[0])
    magnetic_elzo.instrument.background = 0
    other = magnetic_elzo.inhomogeneous_reflectivity_transmissivity(magnetization_map=mag_map)
    assert other.attrs["hash"] != result.attrs["hash"]


def test_xray_dyn_inhomogeneous_with_instrument(xray_dyn, structure_crystalline):
    xray_dyn.theta = np.linspace(31, 33, 41) * u.deg
    xray_dyn.instrument.theta_resolution = "gauss"
    xray_dyn.instrument.theta_width = 0.05
    xray_dyn.instrument.spectrum = [{"weight": 2}, {"offset": -20 * u.eV, "weight": 1}]
    R = xray_dyn.calc_homogeneous_reflectivity()[0]
    strain_map = np.zeros([2, structure_crystalline.num_layers])
    R_inhom = xray_dyn.calc_inhomogeneous_reflectivity(strain_map)
    np.testing.assert_allclose(R_inhom, np.repeat(R[np.newaxis], 2, axis=0), rtol=1e-10)


def test_xray_kin_with_instrument(structure_crystalline):
    sim = simulation(
        ud.XrayKin, structure_crystalline, [K_ALPHA_1] * u.eV, np.linspace(30, 34, 101) * u.deg
    )
    R0 = sim.calc_homogeneous_reflectivity()[0]
    sim.instrument.I0 = 2
    R, A = sim.calc_homogeneous_reflectivity()
    np.testing.assert_allclose(R, 2 * R0)
    assert len(A) > 0
