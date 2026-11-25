#!/usr/bin/env python

import numpy as np
import pytest

from udkm1Dsim import BerremanElzo, BerremanStepanov, Elzo, Parratt, Stepanov, XrayDyn, XrayKin, u

MULTILAYER = [Parratt, Elzo, Stepanov, BerremanElzo, BerremanStepanov]

# fixtures


@pytest.fixture(scope="module")
def xray_kin(structure_crystalline, tmp_path_factory):
    xray_kin = XrayKin(
        structure_crystalline,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )
    xray_kin.energy = np.r_[5000, 8000] * u.eV
    xray_kin.qz = np.r_[1:5:0.001] / u.nm
    return xray_kin


@pytest.fixture
def xray_dyn(structure_crystalline, tmp_path):
    """a new simulation per benchmark, as XrayDyn keeps the matrices of its
    atoms of the last calculation"""
    xray_dyn = XrayDyn(
        structure_crystalline,
        force_recalc=True,
        cache_dir=tmp_path,
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )
    xray_dyn.energy = np.r_[5000, 8000] * u.eV
    xray_dyn.qz = np.r_[1:5:0.004] / u.nm
    return xray_dyn


@pytest.fixture(scope="module", params=MULTILAYER, ids=lambda cls: cls.__name__)
def multilayer(request, structure, tmp_path_factory):
    return multilayer_simulation(request.param, structure, tmp_path_factory)


def multilayer_simulation(cls, S, tmp_path_factory):
    multilayer = cls(
        S,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )
    multilayer.energy = np.r_[700, 800] * u.eV
    # above the critical angle, where the transfer matrices of Elzo stay precise
    multilayer.qz = np.r_[0.2:5:0.05] / u.nm
    multilayer.set_polarization(3, 0)
    return multilayer


@pytest.fixture(scope="module")
def berreman_stepanov(structure, tmp_path_factory):
    """the exact formalism for the benchmarks of the polarization analysis"""
    return multilayer_simulation(BerremanStepanov, structure, tmp_path_factory)


@pytest.fixture(scope="module")
def strain_map_crystalline(structure_crystalline):
    L = structure_crystalline.num_layers
    M = 10  # delay steps
    rng = np.random.default_rng(0)
    return 0.01 * rng.random((M, L))


@pytest.fixture(scope="module")
def temp_map_crystalline(strain_map_crystalline):
    return 300 + 1e4 * strain_map_crystalline


@pytest.fixture(scope="module")
def strain_map_mixed(structure):
    """a single delay, as the formalisms besides Parratt calculate every
    layer of every delay, which is beyond the benchmarks of 1 s for more
    delays of the large structure"""
    L = structure.num_layers
    M = 1  # delay steps
    return 0.01 * np.ones((M, L))


@pytest.fixture(scope="module")
def strain_map_delays(structure):
    L = structure.num_layers
    M = 10  # delay steps
    rng = np.random.default_rng(0)
    return 0.01 * rng.random((M, L))


@pytest.fixture(scope="module")
def magnetization_map(structure):
    L = structure.num_layers
    M = 1  # delay steps
    magnetization_map = np.zeros((M, L, 3))
    magnetization_map[:, :, 2] = 0.5  # cartesian m_z
    return magnetization_map


@pytest.fixture(scope="module")
def polarizations():
    # 18 elliptical polarizations
    return [(alpha * u.deg, 0.2) for alpha in range(0, 180, 10)]


@pytest.fixture
def parratt_instrument(periodic_multilayer):
    """X-ray reflectivity of a multilayer at 1001 angles, as for a fit"""
    parratt = Parratt(
        periodic_multilayer, True, save_data=False, disp_messages=False, progress_bar=False
    )
    parratt.energy = np.r_[8047.8] * u.eV
    parratt.theta = np.linspace(0.2, 4, 1001) * u.deg
    return parratt


# benchmarks


# XrayKin


@pytest.mark.benchmark
def test_xray_kin_homogeneous_reflectivity(xray_kin):
    xray_kin.calc_homogeneous_reflectivity()


# XrayDyn


@pytest.mark.benchmark
def test_xray_dyn_homogeneous_reflectivity(xray_dyn):
    xray_dyn.calc_homogeneous_reflectivity()


@pytest.mark.benchmark
def test_xray_dyn_inhomogeneous_reflectivity(xray_dyn, strain_map_crystalline):
    xray_dyn.calc_inhomogeneous_reflectivity(strain_map_crystalline)


@pytest.mark.benchmark
def test_xray_dyn_inhomogeneous_reflectivity_strain_vectors(xray_dyn, strain_map_crystalline):
    strain_vectors = [np.linspace(0, 0.01, 20)] * xray_dyn.S.num_unique_layers
    xray_dyn.calc_inhomogeneous_reflectivity(strain_map_crystalline, strain_vectors=strain_vectors)


@pytest.mark.benchmark
def test_xray_dyn_inhomogeneous_reflectivity_temp_map(
    xray_dyn, strain_map_crystalline, temp_map_crystalline
):
    """the Debye-Waller factor at the temperatures of all unit cells"""
    xray_dyn.calc_inhomogeneous_reflectivity(strain_map_crystalline, temp_map=temp_map_crystalline)


# multilayer formalisms


@pytest.mark.benchmark
def test_multilayer_homogeneous_reflectivity(multilayer):
    multilayer.calc_homogeneous_reflectivity_transmissivity()


@pytest.mark.benchmark
def test_multilayer_inhomogeneous_reflectivity(multilayer, strain_map_mixed, magnetization_map):
    multilayer.calc_inhomogeneous_reflectivity_transmissivity(
        strain_map=strain_map_mixed, magnetization_map=magnetization_map
    )


def test_parratt_inhomogeneous_reflectivity_delays(
    benchmark, structure, strain_map_delays, tmp_path_factory
):
    """the recursion of Parratt over the layers for a block of delays at once"""
    parratt = multilayer_simulation(Parratt, structure, tmp_path_factory)
    benchmark(parratt.calc_inhomogeneous_reflectivity_transmissivity, strain_map=strain_map_delays)


@pytest.mark.parametrize("cls", [Elzo, BerremanStepanov], ids=lambda cls: cls.__name__)
def test_multilayer_homogeneous_absorption(benchmark, cls, structure, tmp_path_factory):
    """the absorption profile of the fields of a first-order and an exact
    formalism"""
    sim = multilayer_simulation(cls, structure, tmp_path_factory)
    benchmark(sim.calc_homogeneous_absorption)


def test_multilayer_homogeneous_reflectivity_elliptical(
    benchmark, berreman_stepanov, polarizations
):
    berreman_stepanov.set_polarization(5, 0, polarization_in=polarizations)
    try:
        benchmark(berreman_stepanov.calc_homogeneous_reflectivity_transmissivity)
    finally:
        berreman_stepanov.set_polarization(3, 0)


@pytest.mark.benchmark
def test_multilayer_homogeneous_rotation_ellipticity(berreman_stepanov):
    berreman_stepanov.calc_homogeneous_rotation_ellipticity()


# instrument function


def test_instrument_theta_quadrature(benchmark, parratt_instrument):
    parratt_instrument.instrument.theta_resolution = "gauss"
    parratt_instrument.instrument.theta_width = 0.01 * u.deg
    parratt_instrument.instrument.footprint = "gauss"
    parratt_instrument.instrument.beam_width = 0.1 * u.mm
    parratt_instrument.instrument.sample_length = 10 * u.mm
    benchmark(parratt_instrument.calc_homogeneous_reflectivity_transmissivity)


def test_instrument_theta_convolution(benchmark, parratt_instrument):
    parratt_instrument.instrument.theta_resolution = "gauss"
    parratt_instrument.instrument.theta_width = 0.01 * u.deg
    parratt_instrument.instrument.theta_mode = "convolution"
    benchmark(parratt_instrument.calc_homogeneous_reflectivity_transmissivity)


def test_instrument_energy_quadrature(benchmark, parratt_instrument):
    parratt_instrument.instrument.energy_resolution = "gauss"
    parratt_instrument.instrument.energy_width = 2
    parratt_instrument.instrument.energy_points = 9
    benchmark(parratt_instrument.calc_homogeneous_reflectivity_transmissivity)


def test_instrument_delay_convolution(benchmark, parratt_instrument):
    """the temporal resolution of 50 delays of a strain wave"""
    parratt_instrument.instrument.delay_resolution = "gauss"
    parratt_instrument.instrument.delay_width = 0.3 * u.ps
    delays = np.linspace(-1, 4, 50)
    L = parratt_instrument.S.num_layers
    strain_map = 0.01 * np.exp(-(((np.r_[0:L] - 2 * delays[:, np.newaxis]) / 2) ** 2))
    strain_map[delays < 0] = 0
    benchmark(
        parratt_instrument.calc_inhomogeneous_reflectivity_transmissivity,
        strain_map=strain_map,
        delays=delays * u.ps,
    )
