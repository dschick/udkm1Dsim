#!/usr/bin/env python

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import erf

import udkm1Dsim as ud
from tests.conftest import fe_layer
from udkm1Dsim import InstrumentParameters, Parratt, Structure, u
from udkm1Dsim.simulations.result import Result
from udkm1Dsim.simulations.scattering import instrument as ins
from udkm1Dsim.structures.parameters import Parameter

PROFILES = ["box", "gauss", "lorentz", "pseudo_voigt"]
# energies of the Cu K-alpha1 and K-alpha2 lines (eV)
K_ALPHA_1, K_ALPHA_2 = 8047.8, 8027.8


def simulation(cls, S, energy, theta, **kwargs):
    sim = cls(S, True, disp_messages=False, progress_bar=False, save_data=False, **kwargs)
    sim.energy = energy
    sim.theta = theta
    return sim


@pytest.fixture
def parratt(periodic_multilayer):
    return simulation(
        Parratt, periodic_multilayer, [8000] * u.eV, np.linspace(0.2, 4, 101) * u.deg
    )


# ============================================================================
# profiles, nodes, convolution, and footprint
# ============================================================================


@pytest.mark.parametrize("shape", PROFILES)
def test_profiles_have_unit_fwhm_and_area(shape):
    x = np.linspace(-200, 200, 2000001)
    p = ins.profile(shape, x, eta=0.3)
    np.testing.assert_allclose(trapezoid(p, x), 1, atol=2e-3)
    if shape != "box":
        np.testing.assert_allclose(ins.profile(shape, 0.5, 0.3), ins.profile(shape, 0, 0.3) / 2)


def test_gauss_nodes_have_the_moments_of_the_profile():
    sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
    x, w = ins.profile_nodes("gauss", "gaussian_quadrature", 5)
    np.testing.assert_allclose(np.sum(w), 1)
    np.testing.assert_allclose(np.sum(w * x**2), sigma**2)
    np.testing.assert_allclose(np.sum(w * x**4), 3 * sigma**4)
    x, w = ins.profile_nodes("box", "gaussian_quadrature", 4)
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
        ins.profile_nodes(shape, "gaussian_quadrature")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"shape": "voigt"}, "profile must be one of"),
        ({"nodes": "gauss"}, "nodes must be"),
        ({"points": 0}, "positive integer"),
        ({"points": 2.5}, "positive integer"),
        ({"points": True}, "positive integer"),
        ({"range": 0}, "range of the nodes"),
        ({"eta": 1.5}, r"\[0, 1\]"),
    ],
)
def test_profile_nodes_are_checked(kwargs, match):
    """the sampling is checked for all profiles, e.g. also for a box"""
    kwargs = {"shape": "box", **kwargs}
    with pytest.raises(ValueError, match=match):
        ins.profile_nodes(**kwargs)
    assert len(ins.profile_nodes("gauss", points=7.0)[0]) == 7


def test_check_nodes(recwarn):
    with pytest.warns(UserWarning, match="x_points is 1, .* the x spread has no effect"):
        ins.check_nodes("x", "x spread", "gauss", "uniform", 1, 2)
    with pytest.warns(UserWarning, match="Use at least 7 x_points"):
        ins.check_nodes("x", "x spread", "gauss", "uniform", 3, 2)
    recwarn.clear()
    # enough, Gaussian quadrature, or box nodes, and no profile
    ins.check_nodes("x", "x spread", "gauss", "uniform", 7, 2)
    ins.check_nodes("x", "x spread", "gauss", "gaussian_quadrature", 3, 2)
    ins.check_nodes("x", "x spread", "box", "uniform", 2, 2)
    ins.check_nodes("x", "x spread", "none", "uniform", 1, 2)
    assert len(recwarn) == 0


@pytest.mark.parametrize("shape", ["box", "gauss", "lorentz", "pseudo_voigt"])
@pytest.mark.parametrize("range_", [0.5, 2, 7])
def test_profile_area_is_the_integral_within_the_range(shape, range_):
    x = np.linspace(-range_, range_, 200001)
    area = trapezoid(ins.profile(shape, x, 0.3), x)
    # the trapezoidal rule is less accurate at the jumps of the box
    assert ins.profile_area(shape, range_, 0.3) == pytest.approx(
        area, abs=1e-4 if shape == "box" else 1e-6
    )


def test_check_range(recwarn):
    """the wings of a Lorentzian are cut by the default range of 2 FWHM"""
    with pytest.warns(UserWarning, match="Only 84.4% of the lorentz.*at least 6.4 FWHM"):
        ins.check_range("x", "x spread", "lorentz", 2)
    with pytest.warns(UserWarning, match="Only 92.2% of the pseudo_voigt.*at least 3.2 FWHM"):
        ins.check_range("x", "x spread", "pseudo_voigt", 2, 0.5)
    recwarn.clear()
    ins.check_range("x", "x spread", "lorentz", 6.4)
    ins.check_range("x", "x spread", "pseudo_voigt", 2, 0.02)
    ins.check_range("x", "x spread", "gauss", 1)
    assert len(recwarn) == 0


def test_truncated_lorentzian_resolution_is_warned(parratt, recwarn):
    parratt.instrument.theta_resolution = "lorentz"
    parratt.instrument.theta_width = 0.02 * u.deg  # all nodes above the surface
    with pytest.warns(UserWarning, match="lorentz profile of the theta resolution"):
        parratt.homogeneous_reflectivity_transmissivity()
    recwarn.clear()
    parratt.instrument.theta_range = 6.4
    parratt.instrument.theta_points = 21
    parratt.homogeneous_reflectivity_transmissivity()
    assert not [str(w.message) for w in recwarn if "resolution" in str(w.message)]


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
    instrument.theta_width = [0.1, 0.2] * u.deg
    assert instrument.is_active
    instrument.theta_width = 0.01 * u.rad
    assert instrument.theta_width.magnitude == pytest.approx(np.degrees(0.01))
    # bare numbers and dimensionless values are ambiguous angles
    for name in ("theta_width", "theta_offset"):
        for value in (0.1, [0.1, 0.2], 0.1 * u.dimensionless, Parameter("", 0.1)):
            with pytest.raises(TypeError, match=f"{name} must be given"):
                setattr(instrument, name, value)
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


@pytest.mark.parametrize(
    "settings, match",
    [
        ({"theta_width": 0.1 * u.deg}, "theta_width is set, but the theta_resolution is 'none'"),
        ({"theta_resolution": "gauss"}, "theta_resolution is 'gauss', but the theta_width is 0"),
        ({"energy_width": 1}, "energy_width is set, but the energy_resolution is 'none'"),
        ({"energy_resolution": "box"}, "energy_resolution is 'box', but the energy_width is 0"),
        ({"delay_width": 0.5}, "delay_width is set, but the delay_resolution is 'none'"),
        ({"delay_resolution": "gauss"}, "delay_resolution is 'gauss', but the delay_width is 0"),
    ],
)
def test_partial_resolution_is_warned(parratt, settings, match):
    for name, value in settings.items():
        setattr(parratt.instrument, name, value)
    with pytest.warns(UserWarning, match=match):
        parratt.homogeneous_reflectivity_transmissivity()


@pytest.mark.parametrize(
    "settings, match",
    [
        ({"theta_points": 1}, "single node of the quadrature"),
        ({"theta_points": 3}, "3 uniform nodes .* 2 FWHM apart"),
        pytest.param(
            {"theta_points": 9, "theta_range": 4},
            "Use at least 13 theta_points",
            # the wide range reaches the surface at the smallest angle
            marks=pytest.mark.filterwarnings("ignore:Some nodes of the theta resolution"),
        ),
        (
            {"theta_mode": "convolution", "theta_width": 1e-3 * u.deg},
            "larger than the theta_width",
        ),
        ({"energy_mode": "convolution"}, "a single photon energy"),
    ],
)
def test_resolution_without_effect_is_warned(parratt, settings, match):
    parratt.instrument.theta_resolution = "gauss"
    parratt.instrument.theta_width = 0.05 * u.deg
    if "energy_mode" in settings:
        parratt.instrument.theta_resolution = "none"
        parratt.instrument.theta_width = 0 * u.deg
        parratt.instrument.energy_resolution = "gauss"
        parratt.instrument.energy_width = 10
    for name, value in settings.items():
        setattr(parratt.instrument, name, value)
    with pytest.warns(UserWarning, match=match):
        parratt.homogeneous_reflectivity_transmissivity()


@pytest.mark.parametrize(
    "shape, nodes, points",
    [("gauss", "uniform", 7), ("gauss", "gaussian_quadrature", 3), ("box", "uniform", 2)],
)
def test_enough_nodes_are_not_warned(parratt, recwarn, shape, nodes, points):
    parratt.instrument.theta_resolution = shape
    parratt.instrument.theta_nodes = nodes
    parratt.instrument.theta_points = points
    parratt.instrument.theta_width = 0.05 * u.deg
    parratt.homogeneous_reflectivity_transmissivity()
    assert not [w for w in recwarn if "resolution" in str(w.message)]


def test_is_resolved():
    x = np.linspace(0, 1, 11)
    assert ins.is_resolved(x, 0.1, "gauss")
    assert not ins.is_resolved(x, 0.01, "gauss")
    # a box of the width of the spacing reaches half the spacing only
    assert not ins.is_resolved(x, 0.1, "box")
    assert ins.is_resolved(x, 0.2, "box")
    assert not ins.is_resolved(x[:1], 1, "gauss")
    assert not ins.is_resolved(x, 0.1, "none")
    # a single point with a wide profile is enough
    assert ins.is_resolved(x, np.where(x == 0.5, 0.1, 0.01), "gauss")


def test_width_of_the_spectrum_completes_the_resolution(parratt, recwarn):
    parratt.instrument.energy_resolution = "gauss"
    parratt.instrument.spectrum = [{"weight": 1, "width": 2}]
    parratt.homogeneous_reflectivity_transmissivity()
    assert not [w for w in recwarn if "resolution" in str(w.message)]


def test_metadata_of_the_instrument(parratt, tmp_path):
    parratt.instrument.theta_resolution = "gauss"
    parratt.instrument.theta_width = 0.02 * u.deg
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
        ("gauss", "gaussian_quadrature"),
        ("box", "gaussian_quadrature"),
        ("lorentz", "uniform"),
        ("pseudo_voigt", "uniform"),
    ],
)
@pytest.mark.filterwarnings("ignore:Only")  # the truncated Lorentzian wings
def test_theta_quadrature_is_the_weighted_average(parratt, shape, nodes):
    theta = parratt.theta.to("deg").magnitude
    R0 = parratt.calc_homogeneous_reflectivity_transmissivity()[0]
    parratt.instrument.theta_resolution = shape
    parratt.instrument.theta_nodes = nodes
    parratt.instrument.theta_width = 0.05 * u.deg
    parratt.instrument.theta_points = 7
    R = parratt.calc_homogeneous_reflectivity_transmissivity()[0]
    # the experiment is restored
    np.testing.assert_allclose(parratt.theta.to("deg").magnitude, theta)
    x, w = ins.profile_nodes(shape, nodes, 7, 2, 0.5)
    expected = 0
    for x_i, w_i in zip(x, w):
        parratt.instrument.theta_offset = x_i * 0.05 * u.deg
        parratt.instrument.theta_resolution = "none"
        parratt.instrument.theta_width = 0 * u.deg
        expected = expected + w_i * parratt.calc_homogeneous_reflectivity_transmissivity()[0]
        parratt.instrument.theta_resolution = shape
        parratt.instrument.theta_width = 0.05 * u.deg
    np.testing.assert_allclose(R, expected, rtol=1e-12)
    assert not np.allclose(R, R0, rtol=1e-3)


def test_theta_nodes_below_the_surface_do_not_contribute(xray_dyn):
    """the nodes of a divergent beam at angles <= 0 are rays, which do not
    hit the sample from above, instead of NaN, e.g. by q_z = 0"""
    xray_dyn.theta = np.r_[0.05, 0.1, 0.2, 0.3] * u.deg
    xray_dyn.instrument.theta_resolution = "gauss"
    xray_dyn.instrument.theta_width = 0.1 * u.deg
    with pytest.warns(UserWarning, match="below the surface"):
        R = xray_dyn.calc_homogeneous_reflectivity()[0]
    assert np.all(np.isfinite(R))
    x, w = ins.profile_nodes("gauss", "uniform", 7, 2, 0.5)
    expected = 0
    for x_i, w_i in zip(x, w):
        theta = np.r_[0.05, 0.1, 0.2, 0.3] + x_i * 0.1
        R_i = np.zeros(4)
        above = theta > 0
        if np.any(above):
            xray_dyn.instrument.theta_resolution = "none"
            xray_dyn.instrument.theta_width = 0 * u.deg
            xray_dyn.theta = theta[above] * u.deg
            R_i[above] = xray_dyn.calc_homogeneous_reflectivity()[0][0]
        expected = expected + w_i * R_i
    np.testing.assert_allclose(R[0], expected, rtol=1e-12)


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
    parratt.instrument.theta_offset = 0.1 * u.deg
    R = parratt.calc_homogeneous_reflectivity_transmissivity()[0]
    parratt.instrument.theta_offset = 0 * u.deg
    parratt.theta = parratt.theta + 0.1 * u.deg
    np.testing.assert_allclose(R, parratt.calc_homogeneous_reflectivity_transmissivity()[0])


def test_energy_offset(periodic_multilayer):
    """an offset of the photon energy equals the shifted energy at the same
    incidence angles"""
    theta = np.linspace(0.2, 4, 101) * u.deg
    sim = simulation(Parratt, periodic_multilayer, [700, 707] * u.eV, theta)
    sim.instrument.energy_offset = 2 * u.eV
    R = sim.calc_homogeneous_reflectivity_transmissivity()[0]
    shifted = simulation(Parratt, periodic_multilayer, [702, 709] * u.eV, theta)
    np.testing.assert_allclose(R, shifted.calc_homogeneous_reflectivity_transmissivity()[0])
    sim.instrument.energy_offset = 0 * u.eV
    assert not np.allclose(R, sim.calc_homogeneous_reflectivity_transmissivity()[0], rtol=1e-3)


def test_theta_convolution_close_to_quadrature(periodic_multilayer):
    sim = simulation(
        Parratt, periodic_multilayer, [8000] * u.eV, np.linspace(0.2, 4, 3801) * u.deg
    )
    sim.instrument.theta_resolution = "gauss"
    sim.instrument.theta_width = 0.05 * u.deg
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
    parratt.instrument.theta_width = width * u.deg
    R = parratt.calc_homogeneous_reflectivity_transmissivity()[0]
    assert R.shape == (1, 101)
    # each angle has the resolution of its own width
    for i in (10, 50, 90):
        parratt.instrument.theta_width = width[i] * u.deg
        np.testing.assert_allclose(
            R[:, i], parratt.calc_homogeneous_reflectivity_transmissivity()[0][:, i], rtol=1e-12
        )
    parratt.instrument.theta_width = width[:50] * u.deg
    with pytest.raises(ValueError, match="theta_width"):
        parratt.calc_homogeneous_reflectivity_transmissivity()


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
    # the footprint is taken at the angle of incidence, not at the angle of
    # the same qz
    xray_dyn.instrument.footprint = "box"
    xray_dyn.instrument.beam_width = 0.5 * u.mm
    xray_dyn.instrument.sample_length = 0.6 * u.mm
    foot = ins.footprint("box", xray_dyn.theta.to("rad").magnitude, 0.5, 0.6)
    np.testing.assert_allclose(xray_dyn.calc_homogeneous_reflectivity()[0], foot * R_qz)


def test_energy_convolution_close_to_quadrature(periodic_multilayer):
    """an energy scan at a fixed angle"""
    sim = simulation(
        Parratt, periodic_multilayer, np.linspace(690, 740, 1001) * u.eV, [10] * u.deg
    )
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
    fe = fe_layer(5 * u.nm, magnetization={"r": 1, "phi": 90 * u.deg, "gamma": 30 * u.deg})
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


@pytest.mark.filterwarnings("ignore:Only")  # the truncated Lorentzian wings
def test_inhomogeneous_equals_homogeneous_with_instrument(magnetic_elzo, tmp_path):
    magnetic_elzo.instrument.theta_resolution = "gauss"
    magnetic_elzo.instrument.theta_width = 1 * u.deg
    magnetic_elzo.instrument.energy_resolution = "lorentz"
    magnetic_elzo.instrument.energy_width = 0.5
    magnetic_elzo.instrument.energy_points = 7
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
    xray_dyn.instrument.theta_width = 0.05 * u.deg
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


# ============================================================================
# temporal resolution
# ============================================================================


def test_convolve_delays_holds_the_ends():
    """a step at zero delay convolved by a Gaussian is an error function on
    a non-equidistant grid, which is held at its ends"""
    sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
    delays = np.r_[np.linspace(-3, -0.5, 6), np.linspace(-0.45, 2, 50), np.linspace(2.5, 20, 8)]
    step = (delays >= 0).astype(float)
    values = np.stack([step, 2 * step + 1], axis=1)
    result = ins.convolve_delays(values, 0, delays, 1, "gauss", 4)
    expected = 0.5 * (1 + erf(delays / (np.sqrt(2) * sigma)))
    np.testing.assert_allclose(result[:, 0], expected, atol=2e-2)
    np.testing.assert_allclose(result[:, 1], 2 * result[:, 0] + 1)
    # the ends are held instead of normalized, e.g. constant values
    np.testing.assert_allclose(ins.convolve_delays(np.ones(10), 0, np.arange(10), 3, "lorentz"), 1)
    np.testing.assert_allclose(result[[0, -1], 0], [0, 1], atol=1e-10)
    # unsorted delays
    order = np.random.default_rng(1).permutation(delays.size)
    unsorted = ins.convolve_delays(values[order], 0, delays[order], 1, "gauss", 4)
    np.testing.assert_allclose(unsorted, result[order])
    with pytest.raises(ValueError, match="distinct"):
        ins.convolve_delays(np.ones(3), 0, [0, 1, 1], 1, "gauss")


@pytest.mark.parametrize("shape", PROFILES)
def test_convolve_delays_equals_the_convolution_inside(shape):
    """away from the ends, the temporal convolution is the one on the grid"""
    delays = np.linspace(-5, 5, 201)
    values = np.exp(-(delays**2)) * np.cos(3 * delays)
    result = ins.convolve_delays(values, 0, delays, 0.8, shape, 3, 0.3)
    expected = ins.convolve_on_grid(values, 0, delays, 0.8, shape, 3, 0.3)
    inside = np.abs(delays) < 2
    np.testing.assert_allclose(result[inside], expected[inside], atol=1e-12)


@pytest.fixture
def demagnetization(magnetic_elzo):
    """map of a demagnetization of 50 % at zero delay and its delays (ps)"""
    delays = np.r_[-2:-0.5:0.5, -0.5:1:0.05, 1:6:1]
    magnetization = magnetic_elzo.S.get_layer_property_vector("magnetization").magnitude
    amplitude = 1 - 0.5 * (delays >= 0)
    return delays, amplitude[:, np.newaxis, np.newaxis] * magnetization[np.newaxis]


def test_temporal_resolution_of_the_intensities_and_polarization(magnetic_elzo, demagnetization):
    delays, mag_map = demagnetization
    sim = magnetic_elzo
    R = sim.calc_inhomogeneous_reflectivity_transmissivity(magnetization_map=mag_map)
    rotation = sim.calc_inhomogeneous_rotation_ellipticity(magnetization_map=mag_map)
    terms = sim._calc_inhomogeneous(
        np.zeros(mag_map.shape[:2]),
        mag_map,
        type(sim)._reduce_polarization_terms,
    )
    homogeneous = sim.calc_homogeneous_reflectivity_transmissivity()
    sim.instrument.delay_resolution = "gauss"
    sim.instrument.delay_width = 0.3 * u.ps
    assert sim.instrument.is_active and sim.instrument.is_delay_active
    # the homogeneous results have no delays
    for h, reference in zip(sim.calc_homogeneous_reflectivity_transmissivity(), homogeneous):
        np.testing.assert_allclose(h, reference)
    with pytest.raises(ValueError, match="needs the delays"):
        sim.calc_inhomogeneous_reflectivity_transmissivity(magnetization_map=mag_map)
    convolved = sim.calc_inhomogeneous_reflectivity_transmissivity(
        magnetization_map=mag_map, delays=delays * u.ps
    )
    for c, r in zip(convolved, R):
        expected = ins.convolve_delays(r, 0, delays * 1e-12, 0.3e-12, "gauss")
        np.testing.assert_allclose(c, expected, rtol=1e-10)
        np.testing.assert_allclose(c[[0, -1]], r[[0, -1]], rtol=1e-10)
        assert not np.allclose(c, r)
    # the rotation follows from the convolved polarization terms
    rotation_convolved = sim.calc_inhomogeneous_rotation_ellipticity(
        magnetization_map=mag_map, delays=delays * u.ps
    )
    terms = tuple(ins.convolve_delays(t, 0, delays * 1e-12, 0.3e-12, "gauss") for t in terms)
    expected = type(sim)._finish_rotation_ellipticity(terms)
    for c, e, r in zip(rotation_convolved, expected, rotation):
        np.testing.assert_allclose(c, e, atol=1e-10)
        np.testing.assert_allclose(c[[0, -1]], r[[0, -1]], atol=1e-10)


def test_delays_of_the_inhomogeneous_results(magnetic_elzo, demagnetization, tmp_path):
    delays, mag_map = demagnetization
    sim = magnetic_elzo
    # the given delays label the result
    result = sim.inhomogeneous_reflectivity_transmissivity(
        magnetization_map=mag_map, delays=delays * u.ps
    )
    np.testing.assert_allclose(result.quantity("delays", "ps").magnitude, delays)
    jones = sim.inhomogeneous_jones_matrices(magnetization_map=mag_map, delays=delays * 1e-12)
    np.testing.assert_allclose(jones.delays, delays * 1e-12)
    assert "delays" not in sim.inhomogeneous_reflectivity_transmissivity(magnetization_map=mag_map)
    with pytest.raises(ValueError, match="1D array"):
        sim.calc_inhomogeneous_reflectivity_transmissivity(
            magnetization_map=mag_map, delays=delays[1:] * u.ps
        )
    # the delays of an upstream result
    upstream = Result("magnetization map")
    upstream.add_coord("delays", delays * u.ps, "ps")
    upstream.add("magnetization_map", mag_map, ("delays", "distances", "components"))
    sim.instrument.delay_resolution = "box"
    sim.instrument.delay_width = 0.5
    from_result = sim.calc_inhomogeneous_reflectivity_transmissivity(magnetization_map=upstream)
    given = sim.calc_inhomogeneous_reflectivity_transmissivity(
        magnetization_map=mag_map, delays=delays * u.ps
    )
    for r, g in zip(from_result, given):
        np.testing.assert_allclose(r, g)
    np.testing.assert_allclose(
        sim.calc_inhomogeneous_reflectivity_transmissivity(
            magnetization_map=upstream, delays=delays * u.ps
        )[0],
        given[0],
    )
    with pytest.raises(ValueError, match="differ"):
        sim.calc_inhomogeneous_reflectivity_transmissivity(
            magnetization_map=upstream, delays=2 * delays * u.ps
        )
    # the delays enter the hash with a temporal resolution
    sim.save_data = True
    sim.cache_dir = str(tmp_path)
    sim.force_recalc = False
    first = sim.inhomogeneous_reflectivity_transmissivity(magnetization_map=upstream)
    other = sim.inhomogeneous_reflectivity_transmissivity(
        magnetization_map=mag_map, delays=(delays + 0.1) * u.ps
    )
    assert other.attrs["hash"] != first.attrs["hash"]
    assert first.attrs["parameters"]["instrument"]["delay_width"] == 0.5 * u.ps


def test_xray_dyn_temporal_resolution(xray_dyn, structure_crystalline):
    xray_dyn.theta = np.linspace(31, 33, 41) * u.deg
    L = structure_crystalline.num_layers
    delays = np.linspace(-1, 4, 11) * u.ps
    strain_map = np.zeros([11, L])
    strain_map[delays.magnitude >= 0, :] = 1e-3
    R = xray_dyn.calc_inhomogeneous_reflectivity(strain_map)
    xray_dyn.instrument.delay_resolution = "gauss"
    xray_dyn.instrument.delay_width = 1
    with pytest.raises(ValueError, match="needs the delays"):
        xray_dyn.calc_inhomogeneous_reflectivity(strain_map)
    result = xray_dyn.inhomogeneous_reflectivity(strain_map, delays=delays)
    expected = ins.convolve_delays(R, 0, delays.to("s").magnitude, 1e-12, "gauss")
    np.testing.assert_allclose(result.reflectivity, expected, rtol=1e-10)
    np.testing.assert_allclose(result.quantity("delays", "ps").magnitude, delays.magnitude)
