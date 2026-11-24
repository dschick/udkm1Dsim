#!/usr/bin/env python

import numpy as np
import pytest
from pint.testing import assert_allclose

from udkm1Dsim import (
    AmorphousLayer,
    BerremanElzo,
    BerremanStepanov,
    Dielectric,
    Dispersion,
    Elzo,
    Heat,
    Layer,
    Parratt,
    Stepanov,
    Structure,
    Vacuum,
    u,
)
from udkm1Dsim.simulations.solvers import HAS_NUMBA

# fixtures


@pytest.fixture(scope="module")
def heat(structure, tmp_path_factory):
    return Heat(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


# tests


def test_heat_repr(heat):
    assert repr(heat).startswith("Heat simulation properties:")
    assert str(heat) == repr(heat)
    assert heat._repr_html_().startswith("<h2>Heat simulation</h2>")


def test_get_hash(heat):
    heat.get_hash(np.r_[0:10:1], 300)


def test_check_initial_temperature(heat):
    assert np.allclose(heat.check_initial_temperature(300), 300)
    with pytest.raises(ValueError):
        heat.check_initial_temperature(np.r_[300, 300])


def test_check_excitation(heat):
    heat.heat_diffusion = False
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": Parratt,
        "backside": False,
    }
    res, fluence, delay_pump, pulse_width = heat.check_excitation(np.r_[-10:10:1] * u.ps)
    assert np.allclose(fluence, [70.71067812])
    assert np.allclose(delay_pump, [0.0])
    assert np.allclose(pulse_width, [0.0])

    heat.heat_diffusion = True
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [1] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "s",
        "theta": 45 * u.deg,
        "absorption": Parratt,
        "backside": False,
    }
    res, fluence, delay_pump, pulse_width = heat.check_excitation(np.r_[-10:10:1] * u.ps)
    assert np.allclose(fluence, [70.71067812])
    assert np.allclose(delay_pump, [0.0])
    assert np.allclose(pulse_width, [1.0e-12])

    # the angles are quantities, also per pulse, but no bare numbers
    for key in ("theta", "azimuth"):
        with pytest.raises(TypeError, match=f"excitation {key} must be given"):
            heat.excitation = {key: 0.5}
    np.testing.assert_allclose(heat.excitation["theta"].to("deg").magnitude, 45)

    with pytest.raises(ValueError, match="polarization"):
        heat.excitation = {
            "fluence": [10] * u.mJ / u.cm**2,
            "delay_pump": [0] * u.ps,
            "pulse_width": [1] * u.ps,
            "wavelength": 800 * u.nm,
            "polarization": "elliptical",
            "theta": 45 * u.deg,
            "absorption": Parratt,
            "backside": False,
        }


def test_excitation_pulses(tmp_path):
    """scalars are a single pulse, and the pulses are sorted by their delays
    together with their fluences and pulse widths"""
    heat = optical_heat(optical_stack(), tmp_path)
    heat.excitation = {
        "fluence": 5 * u.mJ / u.cm**2,
        "delay_pump": 1 * u.ps,
        "pulse_width": 0 * u.ps,
    }
    assert_allclose(heat.excitation["fluence"], [5] * u.mJ / u.cm**2)
    assert_allclose(heat.excitation["delay_pump"], [1] * u.ps)
    heat.excitation = 3 * u.mJ / u.cm**2
    assert_allclose(heat.excitation["fluence"], [3] * u.mJ / u.cm**2)
    heat.excitation = {
        "fluence": [1, 2, 3] * u.mJ / u.cm**2,
        "delay_pump": [20, 0, 10] * u.ps,
        "pulse_width": [0.3, 0.1, 0.2] * u.ps,
    }
    assert_allclose(heat.excitation["fluence"], [2, 3, 1] * u.mJ / u.cm**2)
    assert_allclose(heat.excitation["delay_pump"], [0, 10, 20] * u.ps)
    assert_allclose(heat.excitation["pulse_width"], [0.1, 0.2, 0.3] * u.ps)


def test_get_absorption_profile(heat):
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": "lambert_beer",
        "backside": False,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": "lambert_beer",
        "backside": True,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": Parratt,
        "backside": False,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": Parratt,
        "backside": True,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "s",
        "theta": 45 * u.deg,
        "absorption": Parratt,
        "backside": False,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": "lambert_beer",
        "backside": False,
    }
    heat.get_absorption_profile(distances=np.r_[0:10e-9:1e-10])


def layer(id, n, thickness=1 * u.nm):
    return Layer(
        id, id, thickness=thickness, dielectric=Dispersion.from_ref_index(n), therm_cond=1
    )


def optical_stack(substrate_index=3.7 + 0.01j):
    """stack of an absorber and a dielectric on an absorbing substrate"""
    absorber = layer("absorber", 3.5 + 4.0j)
    S = Structure("stack")
    S.add_sub_structure(absorber, 20)
    S.add_sub_structure(layer("dielectric", 1.5 + 0.01j), 50)
    S.add_sub_structure(absorber, 10)
    S.add_substrate(layer("substrate", substrate_index, 0 * u.nm))
    return S


def optical_heat(S, tmp_path, **excitation):
    heat = Heat(S, True, cache_dir=tmp_path, save_data=False, disp_messages=False)
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "wavelength": 800 * u.nm,
        "theta": 60 * u.deg,
        "polarization": "p",
        **excitation,
    }
    return heat


@pytest.mark.parametrize("polarization", ["s", "p"])
@pytest.mark.parametrize("theta", [90, 60, 30])
@pytest.mark.parametrize("backside", [False, True])
def test_absorption_conserves_energy(tmp_path, polarization, theta, backside):
    """the absorbed fraction from the absorption profile equals 1 - R - T,
    also for the reversed structure of a backside excitation with total
    reflection at the vacuum, for which the incident medium is transparent
    to separate the incident and reflected flux"""
    S = optical_stack(3.7 if backside else 3.7 + 0.01j)
    heat = optical_heat(
        S, tmp_path, polarization=polarization, theta=theta * u.deg, backside=backside
    )
    # midpoint rule, as the absorption jumps at the interfaces
    thickness = S.interface_distances[-1].to("m").magnitude
    distances = (np.arange(20000) + 0.5) * thickness / 20000
    dAdz = heat.get_absorption_profile(distances)
    formalism = heat.create_multilayer_formalism()
    R = formalism.calc_homogeneous_reflectivity_transmissivity()[0]
    T = formalism.calc_homogeneous_absorption()[8]
    absorbed = np.sum(dAdz) * thickness / 20000
    assert np.isclose(absorbed, 1 - R.item() - T.item(), rtol=1e-6)


def test_absorption_at_interfaces(tmp_path):
    """the depths at the interfaces belong to the layer below and the bottom
    of the structure to its last layer, as for the grid of the heat
    diffusion"""
    S = optical_stack()
    heat = optical_heat(S, tmp_path)
    interfaces = S.interface_distances.to("m").magnitude
    dAdz = heat.get_absorption_profile(interfaces)
    formalism = heat.create_multilayer_formalism()
    inside = interfaces + np.r_[1, 1, 1, -1] * 1e-15
    np.testing.assert_allclose(dAdz, formalism.calc_homogeneous_absorption(inside)[0][0, 0], 1e-6)
    assert np.all(dAdz[[0, 2, 3]] > 10 * dAdz[1])


@pytest.mark.parametrize("cls", [Elzo, Stepanov, BerremanElzo, BerremanStepanov])
def test_absorption_formalisms_agree(tmp_path, cls):
    """all formalisms give the absorption of Parratt for isotropic layers"""
    heat = optical_heat(optical_stack(), tmp_path, polarization="s")
    dAdz = heat.get_absorption_profile()
    heat.excitation = {"absorption": cls}
    np.testing.assert_allclose(heat.get_absorption_profile(), dAdz, rtol=1e-8)


def test_absorption_polarizations(tmp_path):
    """the absorption of unpolarized light is the mean of s and p, while
    circular light is absorbed differently by a magnetic film"""
    heat = optical_heat(optical_stack(), tmp_path)
    profiles = {}
    for polarization in ["s", "p", "unpolarized", "sigma", "pi"]:
        heat.excitation = {"polarization": polarization}
        profiles[polarization] = heat.get_absorption_profile()
    np.testing.assert_allclose(
        profiles["unpolarized"], (profiles["s"] + profiles["p"]) / 2, rtol=1e-12
    )
    np.testing.assert_array_equal(profiles["sigma"], profiles["s"])
    np.testing.assert_array_equal(profiles["pi"], profiles["p"])
    assert heat.excitation["polarization"] == "pi"

    fe = AmorphousLayer("fe", "fe", thickness=1 * u.nm, density=7874 * u.kg / u.m**3, therm_cond=1)
    fe.optical.dielectric = Dielectric(
        eps0=Dispersion.from_ref_index(2.9 + 3.4j),
        eps_mag=Dispersion.from_array([0, 10], [0.6 + 0.2j] * 2, kind="eps"),
    )
    fe.magnetization = {"r": 1, "phi": 0 * u.deg, "gamma": 0 * u.deg}
    S = Structure("film")
    S.add_sub_structure(fe, 10)
    S.add_substrate(Vacuum())
    heat = optical_heat(S, tmp_path, absorption=Elzo)
    circular = []
    for polarization in ["circ +", "circ -"]:
        heat.excitation = {"polarization": polarization}
        circular.append(heat.get_absorption_profile())
    assert np.max(np.abs(circular[0] - circular[1])) > 1e-3 * np.max(circular[0])


def test_absorption_instance_changed_after_the_excitation(tmp_path):
    """changes of a configured instance after setting the excitation change
    its absorption profile, the footprint of the fluence, the metadata, and
    the hash"""
    S = optical_stack()
    heat = optical_heat(S, tmp_path)
    formalism = heat.create_multilayer_formalism(Parratt)
    formalism.theta = [60] * u.deg
    heat.excitation = {"absorption": formalism}
    delays = np.r_[-1:5:0.1] * u.ps
    old_hash = heat.get_hash(delays, 300)
    formalism.theta = [30] * u.deg
    formalism.wl = [400] * u.nm
    assert heat.get_hash(delays, 300) != old_hash
    assert_allclose(heat.excitation["theta"], 30 * u.deg)
    assert_allclose(heat.excitation["wavelength"], 400 * u.nm)
    heat.excitation = {"absorption": Parratt, "theta": 30 * u.deg, "wavelength": 400 * u.nm}
    reference = heat.get_absorption_profile()
    heat.excitation = {"absorption": formalism}
    formalism.theta = [60] * u.deg
    formalism.theta = [30] * u.deg  # changed after setting the excitation
    np.testing.assert_allclose(heat.get_absorption_profile(), reference, rtol=1e-12)
    _, fluence, _, _ = heat.check_excitation(delays)
    np.testing.assert_allclose(fluence, 100 * np.sin(np.deg2rad(30)))
    formalism.theta = [30, 40] * u.deg
    with pytest.raises(ValueError, match="single"):
        heat.get_absorption_profile()


def test_absorption_instance(tmp_path):
    """a configured formalism defines the wavelength, angle, and polarization
    of the excitation"""
    S = optical_stack()
    heat = optical_heat(S, tmp_path)
    formalism = heat.create_multilayer_formalism(Elzo)
    formalism.wl = [400] * u.nm
    formalism.theta = [30] * u.deg
    formalism.set_incoming_polarization("sigma")
    formalism.azimuth = 20 * u.deg
    heat.excitation = {"absorption": formalism}
    excitation = heat.excitation
    assert excitation["absorption"] is formalism
    assert_allclose(excitation["wavelength"], 400 * u.nm)
    assert_allclose(excitation["theta"], 30 * u.deg)
    assert excitation["polarization"] == "sigma"
    assert_allclose(excitation["azimuth"], 20 * u.deg)
    assert heat.create_multilayer_formalism() is formalism
    dAdz = heat.get_absorption_profile()
    heat.excitation = {"absorption": Parratt}
    np.testing.assert_allclose(heat.get_absorption_profile(), dAdz, rtol=1e-8)
    assert heat._parameters()["excitation"]["absorption"] == "Parratt"

    heat.excitation = {"absorption": formalism}
    assert heat._parameters()["excitation"]["absorption"] == "Elzo"
    for key, value in [
        ("wavelength", 800 * u.nm),
        ("theta", 45 * u.deg),
        ("azimuth", 10 * u.deg),
        ("polarization", "p"),
    ]:
        with pytest.raises(ValueError, match=key):
            heat.excitation = {key: value}
        with pytest.raises(ValueError, match=key):
            heat.excitation = {"absorption": formalism, key: value}
    # a backside excitation by a copy of the formalism for the reversed structure
    heat.excitation = {"backside": True}
    dAdz = heat.get_absorption_profile()
    assert formalism.S is S
    heat.excitation = {"absorption": Elzo}
    np.testing.assert_allclose(heat.get_absorption_profile(), dAdz, rtol=1e-12)
    heat.excitation = {"absorption": formalism, "backside": False}
    with pytest.raises(ValueError, match="structure"):
        heat.excitation = {"absorption": Elzo(optical_stack(), True)}
    formalism.theta = [30, 40] * u.deg
    with pytest.raises(ValueError, match="single"):
        heat.excitation = {"absorption": formalism}


def test_absorption_azimuth(tmp_path):
    """the azimuth of the excitation is the one of the formalism, which
    changes the circular dichroism of an in-plane magnetization"""
    fe = AmorphousLayer("fe", "fe", thickness=1 * u.nm, density=7874 * u.kg / u.m**3, therm_cond=1)
    fe.optical.dielectric = Dielectric(
        eps0=Dispersion.from_ref_index(2.9 + 3.4j),
        eps_mag=Dispersion.from_array([0, 10], [0.6 + 0.2j] * 2, kind="eps"),
    )
    # longitudinal magnetization along the in-plane wave vector for azimuth 0
    fe.magnetization = (0, 1, 0)
    S = Structure("film")
    S.add_sub_structure(fe, 10)
    S.add_substrate(Vacuum())
    heat = optical_heat(S, tmp_path, absorption=Elzo)
    dichroism = {}
    for azimuth in [0, 90]:
        heat.excitation = {"azimuth": azimuth * u.deg}
        assert heat.create_multilayer_formalism().azimuth == azimuth * u.deg
        profiles = []
        for polarization in ["c+", "c-"]:
            heat.excitation = {"polarization": polarization}
            profiles.append(heat.get_absorption_profile())
        dichroism[azimuth] = np.max(np.abs(profiles[0] - profiles[1]))
    # the transverse geometry for an azimuth of 90 deg has no circular dichroism
    assert dichroism[90] < 1e-6 * dichroism[0]
    assert "excitation azimuth" in repr(heat)
    hash = heat.get_hash(np.r_[0:10:1], 300)
    heat.excitation = {"azimuth": 0 * u.deg}
    assert heat.get_hash(np.r_[0:10:1], 300) != hash
    with pytest.raises(ValueError, match="single"):
        heat.excitation = {"azimuth": [0, 90] * u.deg}


def test_absorption_errors(tmp_path):
    heat = optical_heat(optical_stack(), tmp_path)
    with pytest.raises(TypeError, match="absorption"):
        heat.excitation = {"multilayer_absorption": True}
    with pytest.raises(ValueError, match="lambert_beer"):
        heat.excitation = {"absorption": "Parratt"}
    with pytest.raises(TypeError, match="multilayer formalism"):
        heat.excitation = {"absorption": Heat}
    with pytest.raises(TypeError, match="multilayer formalism"):
        heat.create_multilayer_formalism("lambert_beer")


def test_absorption_hash(tmp_path):
    """the roughness changes the hash of a heat simulation with a multilayer
    formalism, but not with Lambert-Beer's law"""
    absorber = layer("absorber", 3.5 + 4.0j)
    S = Structure("stack")
    S.add_sub_structure(absorber, 20)
    S.add_substrate(layer("substrate", 1.5, 0 * u.nm))
    heat = optical_heat(S, tmp_path)
    hashes = {}
    for absorption in [Parratt, "lambert_beer"]:
        heat.excitation = {"absorption": absorption}
        hash = heat.get_hash(np.r_[0:10:1], 300)
        absorber.roughness = 0.5 * u.nm
        hashes[absorption] = hash != heat.get_hash(np.r_[0:10:1], 300)
        absorber.roughness = 0 * u.nm
    assert hashes == {Parratt: True, "lambert_beer": False}


def test_absorption_monochromatic(tmp_path):
    """a bandwidth and divergence of 0 or a single node is the monochromatic
    and collimated excitation with its former hash, independent of the
    sampling"""
    heat = optical_heat(optical_stack(), tmp_path)
    dAdz = heat.get_absorption_profile()
    hash = heat.get_hash(np.r_[0:10:1], 300)
    excitation = heat.excitation
    assert_allclose(excitation["bandwidth"], 0 * u.nm)
    assert_allclose(excitation["divergence"], 0 * u.deg)
    assert excitation["bandwidth_profile"] == "gauss"
    assert excitation["divergence_points"] == 7
    for spreads in [
        {"bandwidth": None, "divergence": None},
        {"bandwidth": 0, "divergence": 0 * u.deg, "bandwidth_points": 3},
        {"bandwidth": 0 * u.eV, "divergence_profile": "box", "divergence_nodes": "uniform"},
    ]:
        heat.excitation = spreads
        np.testing.assert_array_equal(heat.get_absorption_profile(), dAdz)
        assert heat.get_hash(np.r_[0:10:1], 300) == hash
    heat.excitation = {"bandwidth": 20 * u.nm, "divergence": 5 * u.deg, "bandwidth_points": 7}
    assert heat.get_hash(np.r_[0:10:1], 300) != hash
    # a single node has no effect, which is warned for
    with pytest.warns(UserWarning) as record:
        heat.excitation = {"bandwidth_points": 1, "divergence_points": 1}
    assert [str(warning.message).split(",")[0] for warning in record] == [
        "The bandwidth_points is 1",
        "The divergence_points is 1",
    ]
    np.testing.assert_allclose(heat.get_absorption_profile(), dAdz, rtol=1e-12)
    wavelength, theta, weights = heat.get_pulse_nodes()
    assert_allclose(wavelength, [800] * u.nm)
    assert_allclose(theta, [60] * u.deg)
    np.testing.assert_array_equal(weights, [[1]])


@pytest.mark.parametrize("absorption", [Parratt, "lambert_beer"])
def test_absorption_bandwidth(tmp_path, absorption):
    """the absorption of a pulse with a bandwidth and divergence is the
    weighted mean of the monochromatic and collimated absorption of its
    nodes"""
    S = optical_stack()
    heat = optical_heat(
        S, tmp_path, absorption=absorption, bandwidth=30 * u.nm, divergence=4 * u.deg
    )
    wavelength, theta, weights = heat.get_pulse_nodes()
    assert weights.shape == (7, 7)
    assert np.isclose(np.sum(weights), 1)
    assert_allclose(wavelength, np.linspace(740, 860, 7) * u.nm)
    assert_allclose(theta, np.linspace(52, 68, 7) * u.deg)
    expected = 0
    for e, wl in enumerate(wavelength):
        for t, angle in enumerate(theta):
            single = optical_heat(S, tmp_path, absorption=absorption, wavelength=wl, theta=angle)
            expected = expected + weights[e, t] * single.get_absorption_profile()
    dAdz = heat.get_absorption_profile()
    np.testing.assert_allclose(dAdz, expected, rtol=1e-10)
    monochromatic = optical_heat(S, tmp_path, absorption=absorption).get_absorption_profile()
    assert np.max(np.abs(dAdz - monochromatic)) > 1e-4 * np.max(monochromatic)
    # the photon energies of a bandwidth in an energy unit
    heat.excitation = {
        "bandwidth": 0.1 * u.eV,
        "bandwidth_nodes": "gaussian_quadrature",
        "bandwidth_points": 5,
    }
    wavelength = heat.get_pulse_nodes()[0]
    energy = (u.planck_constant * u.speed_of_light / wavelength).to("eV")
    center = (u.planck_constant * u.speed_of_light / (800 * u.nm)).to("eV")
    x = np.polynomial.hermite_e.hermegauss(5)[0] / (2 * np.sqrt(2 * np.log(2)))
    assert_allclose(energy, center + x * 0.1 * u.eV, rtol=1e-12)


@pytest.mark.parametrize("backside", [False, True])
def test_absorption_bandwidth_conserves_energy(tmp_path, backside):
    """the mean absorption of the nodes equals the mean of 1 - R - T"""
    S = optical_stack(3.7 if backside else 3.7 + 0.01j)
    heat = optical_heat(
        S,
        tmp_path,
        bandwidth=50 * u.nm,
        divergence=10 * u.deg,
        divergence_profile="box",
        backside=backside,
    )
    thickness = S.interface_distances[-1].to("m").magnitude
    distances = (np.arange(20000) + 0.5) * thickness / 20000
    absorbed = np.sum(heat.get_absorption_profile(distances)) * thickness / 20000
    formalism = heat.create_multilayer_formalism(nodes=True)
    assert formalism.theta.shape == (7, 7)
    R = formalism.calc_homogeneous_reflectivity_transmissivity()[0]
    T = formalism.calc_homogeneous_absorption()[8]
    weights = heat.get_pulse_nodes()[2]
    assert np.isclose(absorbed, np.sum(weights * (1 - R - T)), rtol=1e-6)


def test_absorption_bandwidth_instance(tmp_path):
    """a configured instance is copied to the nodes without its instrument,
    while the instance itself is kept"""
    S = optical_stack()
    heat = optical_heat(S, tmp_path, absorption=Elzo, bandwidth=20 * u.nm, divergence=3 * u.deg)
    dAdz = heat.get_absorption_profile()
    formalism = heat.create_multilayer_formalism()
    formalism.instrument.theta_resolution = "gauss"
    formalism.instrument.theta_width = 1 * u.deg
    heat.excitation = {"absorption": formalism}
    for backside in [False, True]:
        heat.excitation = {"backside": backside}
        nodes = heat.create_multilayer_formalism(nodes=True)
        assert nodes is not formalism
        assert nodes._instrument is None
        assert nodes.theta.shape == (7, 7)
        assert formalism.theta.shape == (1, 1)
        assert formalism.instrument.is_active
        if not backside:
            np.testing.assert_allclose(heat.get_absorption_profile(), dAdz, rtol=1e-12)
    assert_allclose(formalism.wl, [800] * u.nm)


def test_absorption_bandwidth_per_pulse(tmp_path, monkeypatch, capsys):
    """the bandwidth and divergence are given per pulse and sorted with the
    pulses, where pulses with the same spreads are calculated once"""
    heat = optical_heat(optical_stack(), tmp_path)
    heat.disp_messages = True
    heat.excitation = two_pulses(
        wavelength=800 * u.nm,
        theta=45 * u.deg,
        backside=False,
        polarization="pi",
        bandwidth=[0.05 * u.eV, 20 * u.nm],
        divergence=[0, 2] * u.deg,
        bandwidth_points=[7, 9],
    )
    excitation = heat.excitation
    # sorted by the delays of the pulses
    assert [f"{width:.4g~P}" for width in excitation["bandwidth"]] == ["20 nm", "0.05 eV"]
    assert_allclose(excitation["divergence"], [2, 0] * u.deg)
    assert excitation["bandwidth_points"] == [9, 7]
    assert [heat.get_pulse_nodes(i)[2].shape for i in range(2)] == [(9, 7), (7, 1)]
    capsys.readouterr()
    heat.get_absorption_profile()
    out = capsys.readouterr().out
    assert "800 nm (FWHM 20 nm, gauss, 9 uniform nodes), theta = 45 deg (FWHM 2 deg" in out
    assert "800 nm (FWHM 0.05 eV, gauss, 7 uniform nodes), theta = 45 deg, " in out
    assert "excitation bandwidth" in repr(heat)
    # the same spreads of both pulses
    calls = []
    calculate = heat._multilayer_absorption
    monkeypatch.setattr(
        heat, "_multilayer_absorption", lambda *args: calls.append(args[1]) or calculate(*args)
    )
    heat.excitation = {"bandwidth": 20 * u.nm, "divergence": 2 * u.deg, "bandwidth_points": 9}
    assert "excitation bandwidth  20 nm (gauss, 9 uniform nodes)" in repr(heat)
    profiles = heat.get_absorption_profile()
    assert calls == [0]
    np.testing.assert_array_equal(profiles[0], profiles[1])


def test_absorption_bandwidth_metadata(heat, tmp_path):
    """the bandwidth and divergence of the pulses are saved in the metadata"""
    heat.excitation = {
        "fluence": [5, 5] * u.mJ / u.cm**2,
        "delay_pump": [0, 3] * u.ps,
        "pulse_width": [0, 0] * u.ps,
        "absorption": "lambert_beer",
        "bandwidth": [10 * u.nm, 0.1 * u.eV],
        "divergence": 2 * u.deg,
    }
    res = heat.get_temp_map(np.r_[-1:5:0.5] * u.ps, 300 * u.K)
    res.save(tmp_path / "temp_map.h5")
    excitation = type(res).load(tmp_path / "temp_map.h5").attrs["parameters"]["excitation"]
    assert "bandwidth" in excitation
    assert "divergence_profile" in excitation
    heat.excitation = {"bandwidth": 0, "divergence": None}


def test_absorption_bandwidth_errors(tmp_path):
    heat = optical_heat(optical_stack(), tmp_path)
    for spreads, error, match in [
        ({"bandwidth": 10}, TypeError, "Quantity"),
        ({"bandwidth": 10 * u.ps}, ValueError, "length or energy"),
        ({"bandwidth": -10 * u.nm}, ValueError, "negative"),
        ({"divergence": 2}, TypeError, "divergence"),
        ({"divergence": -2 * u.deg}, ValueError, "negative"),
        ({"bandwidth_profile": "voigt"}, ValueError, "bandwidth_profile"),
        ({"divergence_nodes": "gauss"}, ValueError, "divergence_nodes"),
        ({"bandwidth_points": 0}, ValueError, "bandwidth_points"),
        ({"bandwidth_points": 2.5}, ValueError, "bandwidth_points"),
        ({"divergence_range": 0}, ValueError, "divergence_range"),
        ({"bandwidth_eta": 2}, ValueError, "bandwidth_eta"),
        ({"bandwidth": 500 * u.nm}, ValueError, "below 0"),
        ({"bandwidth": 2 * u.eV}, ValueError, "below 0"),
        ({"divergence": 40 * u.deg}, ValueError, r"\(0, 180\) deg"),
        (
            {
                "bandwidth": 10 * u.nm,
                "bandwidth_profile": "lorentz",
                "bandwidth_nodes": "gaussian_quadrature",
            },
            ValueError,
            "Gaussian quadrature",
        ),
        ({"bandwidth": [10, 20] * u.nm}, ValueError, "each of the 1 pulse"),
    ]:
        with pytest.raises(error, match=match):
            heat.excitation = spreads
        # the excitation is kept for an invalid excitation
        assert_allclose(heat.excitation["bandwidth"], 0 * u.nm)
        assert heat.excitation["bandwidth_profile"] == "gauss"
        assert heat.excitation["bandwidth_nodes"] == "uniform"
    # too coarse uniform nodes of the profile, as for the instrument function
    with pytest.warns(UserWarning, match="3 uniform nodes of the divergence of the pulse 0"):
        heat.excitation = {"divergence": 2 * u.deg, "divergence_points": 3}


def test_get_temperature_after_delta_excitation(heat):
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": Parratt,
        "backside": False,
    }
    result = heat.get_temperature_after_delta_excitation(10, 300)
    assert result.attrs["inputs"]["fluence"] == 1 * u.mJ / u.cm**2
    final_temp, delta_temp = heat.calc_temperature_after_delta_excitation(10, 300)
    np.testing.assert_array_equal(result.final_temp, final_temp)
    # the temperature change is the difference to the initial temperature
    assert np.max(delta_temp) > 0
    np.testing.assert_allclose(delta_temp, final_temp - 300)
    assert result.dims("delta_temp") == ("distances", "subsystems")
    distances = np.r_[0:10e-9:1e-10]
    result = heat.get_temperature_after_delta_excitation(10, 300, distances=distances)
    np.testing.assert_array_equal(result.distances, distances)


def test_get_temp_map(heat):
    delays = np.r_[-1:10:0.01] * u.ps
    init_temp = 300 * u.K
    heat.heat_diffusion = False
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": Parratt,
        "backside": False,
    }
    result = heat.get_temp_map(delays, init_temp)
    parameters = result.attrs["parameters"]
    assert parameters["excitation"]["fluence"].units == u.mJ / u.cm**2
    np.testing.assert_allclose(parameters["excitation"]["fluence"].magnitude, [10])
    assert parameters["excitation"]["polarization"] == "pi"
    assert parameters["boundary_conditions"]["bottom_type"] == "isolator"
    assert parameters["ode_options"] == heat.ode_options
    assert "inputs" not in result.attrs and "input_hashes" not in result.attrs

    # load data from cache
    heat.force_recalc = False
    cached = heat.get_temp_map(delays, init_temp)
    assert cached.attrs["parameters"]["excitation"] == parameters["excitation"]
    heat.force_recalc = True
    # the ode_options change the hash
    hash = heat.get_hash(delays, init_temp)
    heat.ode_options["rtol"] = 1e-4
    assert heat.get_hash(delays, init_temp) != hash
    heat.ode_options["rtol"] = 1e-3

    # the energy maps hold the hash of their input map
    energy = heat.get_energy_map(result)
    assert energy.attrs["input_hashes"] == {"temp_map": result.entry_hash("temp_map")}
    assert energy.attrs["inputs"]["init_temp"].units == u.K


def test_get_diffusion_temp_map(heat):
    """the heat diffuses from the excited surface into the depth, while the
    isolating boundaries conserve the absorbed energy"""
    delays = np.r_[-1:10:0.01] * u.ps
    init_temp = 300 * u.K
    heat.heat_diffusion = True
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": Parratt,
        "backside": False,
    }
    heat.boundary_conditions = {"top_type": "isolator", "bottom_type": "isolator"}
    res = heat.get_temp_map(delays, init_temp)
    temp_map = res.temp_map
    after = delays.to("ps").magnitude >= 0
    # energy per area of the layers with the constant heat capacity of the fixtures
    thicknesses = heat.S.get_layer_property_vector("structural.thickness.magnitude")
    densities = heat.S.get_layer_property_vector("structural.density.magnitude")
    energy = (temp_map[after] - 300) @ (500 * densities * thicknesses)
    assert energy[0] > 0
    # up to the tolerance of the solver and the sampling of the layers
    np.testing.assert_allclose(energy, energy[0], rtol=1e-2)
    # the excited surface cools, while the deeper layers warm up
    excited = temp_map[after][0]
    assert temp_map[-1, 0] < excited[0]
    assert np.any(temp_map[-1] > excited + 1e-3)


def two_pulses(**excitation):
    """two pulses of different wavelength, angle, polarization, and side"""
    return {
        "fluence": [3, 5] * u.mJ / u.cm**2,
        "delay_pump": [10, 0] * u.ps,
        "pulse_width": [0, 0] * u.ps,
        "wavelength": [400, 800] * u.nm,
        "theta": [70, 45] * u.deg,
        "polarization": ["s", "pi"],
        "absorption": Parratt,
        "backside": [True, False],
        **excitation,
    }


def test_per_pulse_excitation(tmp_path):
    """the per-pulse parameters are sorted by the delays together with the
    pulses and are given once for identical pulses"""
    heat = optical_heat(optical_stack(), tmp_path)
    heat.excitation = two_pulses()
    excitation = heat.excitation
    assert heat.num_pulses == 2
    assert_allclose(excitation["delay_pump"], [0, 10] * u.ps)
    assert_allclose(excitation["fluence"], [5, 3] * u.mJ / u.cm**2)
    assert_allclose(excitation["wavelength"], [800, 400] * u.nm)
    assert_allclose(excitation["theta"], [45, 70] * u.deg)
    assert excitation["polarization"] == ["pi", "sigma"]
    assert excitation["backside"] == [False, True]
    assert excitation["absorption"] is Parratt
    assert_allclose(excitation["azimuth"], 0 * u.deg)
    # the footprint of each pulse
    _, fluence, _, _ = heat.check_excitation(np.r_[-1:20:1] * u.ps)
    np.testing.assert_allclose(fluence, [50 * np.sin(np.pi / 4), 30 * np.sin(np.radians(70))])
    assert "['pi', 'sigma']" in repr(heat)
    # per-pulse values of the wrong length
    with pytest.raises(ValueError, match="each of the 2 pulse"):
        heat.excitation = {"wavelength": [400, 800, 600] * u.nm}
    # mixed absorption of a configured instance and Lambert-Beer
    # created for the frontside pulse, i.e. the structure of the simulation
    formalism = heat.create_multilayer_formalism(Elzo, pulse_index=0)
    heat.excitation = {"absorption": ["lambert_beer", formalism]}
    assert heat._absorption_name() == ["lambert_beer", "Elzo"]
    assert heat.create_multilayer_formalism(pulse_index=1) is not formalism  # backside copy
    heat.excitation = {"wavelength": 600 * u.nm}
    # the instance keeps the wavelength of its pulse
    assert_allclose(heat.excitation["wavelength"], [600, 800] * u.nm)
    with pytest.raises(ValueError, match="pulse 1"):
        heat.excitation = {"wavelength": [600, 500] * u.nm}

    # a list of angle quantities is a per-pulse angle as well
    heat.excitation = two_pulses(theta=[70 * u.deg, 0.25 * u.turn])
    assert_allclose(heat.excitation["theta"], [90, 70] * u.deg)


def test_per_pulse_absorption_profiles(tmp_path, monkeypatch):
    """every pulse has its own absorption profile, which is calculated once
    for identical pulses"""
    S = optical_stack()
    heat = optical_heat(S, tmp_path)
    heat.excitation = two_pulses()
    profiles = heat.get_absorption_profile()
    assert profiles.shape == (2, S.num_layers)
    for index in range(2):
        np.testing.assert_array_equal(
            heat.get_absorption_profile(pulse_index=index), profiles[index]
        )
    # the profiles of the single pulses
    for index, (wavelength, theta, polarization, backside) in enumerate(
        [(800, 45, "pi", False), (400, 70, "sigma", True)]
    ):
        single = optical_heat(
            S,
            tmp_path,
            wavelength=wavelength * u.nm,
            theta=theta * u.deg,
            polarization=polarization,
            backside=backside,
        )
        np.testing.assert_allclose(single.get_absorption_profile(), profiles[index], rtol=1e-12)
    np.testing.assert_array_equal(heat.get_absorption_profile(pulse_index=-1), profiles[1])
    with pytest.raises(IndexError):
        heat.get_absorption_profile(pulse_index=2)
    # identical pulses share their calculation
    calls = []
    calculate = heat._multilayer_absorption
    monkeypatch.setattr(
        heat, "_multilayer_absorption", lambda *args: calls.append(args[1]) or calculate(*args)
    )
    heat.excitation = two_pulses(wavelength=800 * u.nm, theta=45 * u.deg, backside=False)
    heat.excitation = {"polarization": "pi"}
    profiles = heat.get_absorption_profile()
    assert calls == [0]
    np.testing.assert_array_equal(profiles[0], profiles[1])
    # Lambert-Beer for all pulses, but the multilayer formalism only for its pulses
    heat.excitation = {"absorption": ["lambert_beer", Parratt]}
    assert heat.get_absorption_from_lambert_beer().shape == (2, S.num_layers)
    heat.get_absorption_from_multilayer(pulse_index=1)
    with pytest.raises(ValueError, match="Lambert-Beer"):
        heat.get_absorption_from_multilayer()


def test_per_pulse_temp_map(heat, tmp_path, monkeypatch, capsys):
    """the absorbed energy of pulses with different wavelengths, angles, and
    polarizations, which is conserved by isolating boundaries, and the
    absorption profiles in the result, which are calculated once per grid and
    reported once per pulse"""
    delays = np.r_[-1:10:0.01] * u.ps
    heat.heat_diffusion = True
    heat.boundary_conditions = {"top_type": "isolator", "bottom_type": "isolator"}
    heat.excitation = {
        "fluence": [5, 5] * u.mJ / u.cm**2,
        "delay_pump": [0, 3] * u.ps,
        "pulse_width": [0.1, 0.1] * u.ps,
        "wavelength": [800, 400] * u.nm,
        "theta": [45, 70] * u.deg,
        "polarization": ["pi", "sigma"],
        "absorption": Parratt,
        "backside": False,
    }
    calls = []
    calculate = heat._multilayer_absorption
    monkeypatch.setattr(
        heat, "_multilayer_absorption", lambda *args: calls.append(args[1]) or calculate(*args)
    )
    capsys.readouterr()
    res = heat.get_temp_map(delays, 300 * u.K)
    # on the grid of the heat diffusion and at the layer centers of the result
    assert sorted(calls) == [0, 0, 1, 1]
    reports = [line for line in capsys.readouterr().out.splitlines() if "Absorption of" in line]
    assert len(reports) == 2
    assert "pulse 1 at 3 ps by Parratt for 400 nm, theta = 70 deg" in reports[1]
    assert "reflection of" in reports[1]
    monkeypatch.undo()
    assert res.dims("absorption") == ("pulses", "distances")
    np.testing.assert_allclose(res.absorption, heat.get_absorption_profile())
    assert np.max(np.abs(res.absorption[0] - res.absorption[1])) > 0.1 * np.max(res.absorption)
    thicknesses = heat.S.get_layer_property_vector("structural.thickness.magnitude")
    densities = heat.S.get_layer_property_vector("structural.density.magnitude")
    energy = (res.temp_map - 300) @ (500 * densities * thicknesses)
    # absorbed energy per area of each pulse on the surface
    surface = np.r_[50 * np.sin(np.pi / 4), 50 * np.sin(np.radians(70))]
    absorbed = surface * (res.absorption @ thicknesses)
    t = delays.to("ps").magnitude
    np.testing.assert_allclose(energy[(t > 1) & (t < 2)], absorbed[0], rtol=2e-2)
    np.testing.assert_allclose(energy[t > 4], absorbed.sum(), rtol=2e-2)
    # the per-pulse parameters are saved in the metadata
    res.save(tmp_path / "temp_map.h5")
    loaded = type(res).load(tmp_path / "temp_map.h5")
    assert list(loaded.attrs["parameters"]["excitation"]["polarization"]) == ["pi", "sigma"]
    heat.excitation = {"wavelength": [800, 600] * u.nm}
    assert heat.get_hash(delays, 300) != res.attrs["hash"]


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")
def test_get_diffusion_temp_map_numba(heat):
    """the numba backend for the heat diffusion is not implemented, yet"""
    heat.heat_diffusion = True
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0.1] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "absorption": Parratt,
        "backside": False,
    }
    heat.backend = "numba"
    try:
        # calc_temp_map, since the cached results do not depend on the backend
        with pytest.raises(NotImplementedError):
            heat.calc_temp_map(np.r_[-1:10:0.01] * u.ps, 300 * u.K)
    finally:
        heat.backend = "scipy"


def test_calc_energy_map(heat):
    dists, _, _ = heat.S.layer_distances
    temp_map = np.zeros((10, len(dists)))
    heat.calc_energy_map(temp_map, 0)


def test_calc_energy_flux_map(heat):
    delays = np.r_[-1:10:0.01] * u.ps
    dists, _, _ = heat.S.layer_distances

    temp_map = np.zeros((len(delays), len(dists)))
    delta_map = temp_map
    heat.calc_energy_flux_map(temp_map, delta_map, delays)


def test_get_energy_maps_of_temp_map(heat):
    """the energy and energy flux maps of the result of get_temp_map equal the
    ones of its arrays"""
    delays = np.r_[-1:5:0.1] * u.ps
    heat.heat_diffusion = False
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
    }
    temp = heat.get_temp_map(delays, 300 * u.K)
    np.testing.assert_array_equal(temp.init_temp, heat.check_initial_temperature(300))
    energy = heat.get_energy_map(temp)
    reference = heat.calc_energy_map(temp.temp_map, 300 * u.K)
    np.testing.assert_array_equal(energy.energy_map, reference)
    assert np.max(energy.energy_map) > 0
    np.testing.assert_array_equal(energy.delays, temp.delays)
    flux = heat.get_energy_flux_map(temp)
    reference = heat.calc_energy_flux_map(temp.temp_map, temp.delta_temp_map, delays)
    for i, name in enumerate(("total", "coupling", "diffusion")):
        np.testing.assert_array_equal(flux[f"{name}_energy_flux_map"], reference[..., i])
    assert flux.unit("total_energy_flux_map") == "W"
    with pytest.raises(ValueError, match="init_temp must be given"):
        heat.get_energy_map(temp.temp_map)
    with pytest.raises(ValueError, match="delays must be given"):
        heat.get_energy_flux_map(temp.temp_map)


def test_boundary_conditions(heat):
    heat.boundary_conditions = {"top_type": "isolator", "bottom_type": "isolator"}
    assert heat.boundary_conditions == {"top_type": "isolator", "bottom_type": "isolator"}

    heat.boundary_conditions = {
        "top_type": "temperature",
        "top_value": 300 * u.K,
        "bottom_type": "temperature",
        "bottom_value": 300 * u.K,
    }
    assert heat.boundary_conditions == {
        "top_type": "temperature",
        "top_value": 300 * u.K,
        "bottom_type": "temperature",
        "bottom_value": 300 * u.K,
    }

    heat.boundary_conditions = {
        "top_type": "flux",
        "top_value": 1 * u.W / u.m**2,
        "bottom_type": "flux",
        "bottom_value": 1 * u.W / u.m**2,
    }
    assert heat.boundary_conditions == {
        "top_type": "flux",
        "top_value": 1 * u.W / u.m**2,
        "bottom_type": "flux",
        "bottom_value": 1 * u.W / u.m**2,
    }

    with pytest.raises(ValueError):
        heat.boundary_conditions = {"top_type": "test"}
    with pytest.raises(ValueError):
        heat.boundary_conditions = {"bottom_type": "test"}
    with pytest.raises(ValueError):
        heat.boundary_conditions = {"top_value": [1, 2] * u.W / u.m**2}
    with pytest.raises(ValueError):
        heat.boundary_conditions = {"bottom_value": [1, 2] * u.W / u.m**2}
    with pytest.raises(ValueError):
        heat.boundary_conditions = "test"


def test_distances(heat):
    heat.distances = np.r_[0:10:0.1] * u.nm
    assert_allclose(heat.distances, np.r_[0:10:0.1] * u.nm)
    assert_allclose(heat.distances.to("m"), np.r_[0:10e-9:1e-10] * u.m)
    heat.distances = np.array([]) * u.nm


def test_backend(heat):
    assert heat.backend == "scipy"

    with pytest.raises(ValueError):
        heat.backend = "matlab"

    heat.backend = "numba"
    if HAS_NUMBA:
        assert heat.backend == "numba"
    else:
        # without numba being installed the backend falls back to scipy
        assert heat.backend == "scipy"

    heat.backend = "scipy"
    assert heat.backend == "scipy"


def test_get_hash_is_backend_independent(heat):
    """both backends must yield the same results, hence cached results must
    not be invalidated by a change of the backend"""
    hash_scipy = heat.get_hash(np.r_[0:10:1], 300)
    heat.backend = "numba"
    assert heat.get_hash(np.r_[0:10:1], 300) == hash_scipy
    heat.backend = "scipy"


def test_heat_inconsistent_num_sub_systems_raises(tmp_path):
    layer = Layer("a", "a", thickness=1 * u.nm, heat_capacity=[10, 20], therm_cond=1)
    S = Structure("s")
    S.add_sub_structure(layer, 10)
    heat = Heat(S, True, cache_dir=tmp_path, save_data=False, disp_messages=False)
    with pytest.raises(ValueError, match="'therm_cond' has 1"):
        heat.get_temp_map(np.r_[0:1:0.1] * u.ps, 300 * u.K)


@pytest.mark.parametrize("hot", [0, -1], ids=["surface", "bottom"])
def test_heat_diffusion_conserves_energy_at_isolating_boundaries(tmp_path, hot):
    """the boundary nodes have half cells, so that a hot boundary node
    equilibrates to the mean temperature of the trapezoidal rule"""
    layer = Layer(
        "a",
        "a",
        thickness=1 * u.nm,
        density=1000 * u.kg / u.m**3,
        heat_capacity=500 * u.J / u.kg / u.K,
        therm_cond=10 * u.W / u.m / u.K,
    )
    S = Structure("s")
    S.add_sub_structure(layer, 10)
    heat = Heat(S, True, cache_dir=tmp_path, save_data=False, disp_messages=False)
    heat.boundary_conditions = {"top_type": "isolator", "bottom_type": "isolator"}
    distances = np.linspace(0, 10e-9, 11)
    init_temp = np.full((11, 1), 300.0)
    init_temp[hot] = 400
    delays = np.r_[0, 1e-9]
    temp_map = heat.calc_heat_diffusion(
        init_temp, distances, delays, np.r_[0.0], np.r_[0.0], np.r_[0.0]
    )
    np.testing.assert_allclose(temp_map[-1], 305.0, rtol=1e-6)


def test_temperature_boundary_holds_its_node(tmp_path):
    layer = Layer(
        "a",
        "a",
        thickness=1 * u.nm,
        density=1000 * u.kg / u.m**3,
        heat_capacity=500 * u.J / u.kg / u.K,
        therm_cond=10 * u.W / u.m / u.K,
    )
    S = Structure("s")
    S.add_sub_structure(layer, 10)
    heat = Heat(S, True, cache_dir=tmp_path, save_data=False, disp_messages=False)
    heat.boundary_conditions = {
        "top_type": "temperature",
        "top_value": 350 * u.K,
        "bottom_type": "isolator",
    }
    distances = np.linspace(0, 10e-9, 11)
    delays = np.r_[0, 1e-12, 1e-9]
    temp_map = heat.calc_heat_diffusion(
        np.full((11, 1), 300.0), distances, delays, np.r_[0.0], np.r_[0.0], np.r_[0.0]
    )
    np.testing.assert_allclose(temp_map[:, 0], 350)
    np.testing.assert_allclose(temp_map[-1], 350, rtol=1e-6)
