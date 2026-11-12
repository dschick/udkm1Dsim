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


@pytest.fixture(scope="module")
def xray_dyn(structure_crystalline, tmp_path_factory):
    xray_dyn = XrayDyn(
        structure_crystalline,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )
    xray_dyn.energy = np.r_[5000, 8000] * u.eV
    xray_dyn.qz = np.r_[1:5:0.004] / u.nm
    return xray_dyn


@pytest.fixture(scope="module", params=MULTILAYER, ids=lambda cls: cls.__name__)
def multilayer(request, structure, tmp_path_factory):
    multilayer = request.param(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )
    multilayer.energy = np.r_[700, 800] * u.eV
    multilayer.qz = np.r_[0.01:5:0.05] / u.nm
    return multilayer


@pytest.fixture(scope="module")
def strain_map_crystalline(structure_crystalline):
    L = structure_crystalline.get_number_of_layers()
    M = 10  # delay steps
    rng = np.random.default_rng(0)
    return 0.01 * rng.random((M, L))


@pytest.fixture(scope="module")
def strain_map_mixed(structure):
    L = structure.get_number_of_layers()
    M = 1  # delay steps
    return 0.01 * np.ones((M, L))


@pytest.fixture(scope="module")
def strain_map_delays(structure):
    L = structure.get_number_of_layers()
    M = 10  # delay steps
    rng = np.random.default_rng(0)
    return 0.01 * rng.random((M, L))


@pytest.fixture(scope="module")
def magnetization_map(structure):
    L = structure.get_number_of_layers()
    M = 1  # delay steps
    magnetization_map = np.zeros((M, L, 3))
    magnetization_map[:, :, 2] = 0.5  # cartesian m_z
    return magnetization_map


@pytest.fixture(scope="module")
def polarizations():
    # 18 elliptical polarizations
    return [(alpha * u.deg, 0.2) for alpha in range(0, 180, 10)]


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
    strain_vectors = [np.linspace(0, 0.01, 20)] * xray_dyn.S.get_number_of_unique_layers()
    xray_dyn.calc_inhomogeneous_reflectivity(strain_map_crystalline, strain_vectors=strain_vectors)


# multilayer formalisms


@pytest.mark.benchmark
def test_multilayer_homogeneous_reflectivity(multilayer):
    multilayer.set_polarization(3, 0)
    multilayer.calc_homogeneous_reflectivity_transmissivity()


@pytest.mark.benchmark
def test_multilayer_inhomogeneous_reflectivity(multilayer, strain_map_mixed, magnetization_map):
    multilayer.set_polarization(3, 0)
    multilayer.calc_inhomogeneous_reflectivity_transmissivity(
        strain_map=strain_map_mixed, magnetization_map=magnetization_map
    )


@pytest.mark.benchmark
def test_multilayer_homogeneous_reflectivity_elliptical(multilayer, polarizations):
    multilayer.set_polarization(5, 0, polarization_in=polarizations)
    multilayer.calc_homogeneous_reflectivity_transmissivity()


@pytest.mark.benchmark
def test_multilayer_homogeneous_rotation_ellipticity(multilayer):
    multilayer.set_polarization(3, 0)
    multilayer.calc_homogeneous_rotation_ellipticity()


def test_parratt_inhomogeneous_reflectivity_delays(
    benchmark, structure, strain_map_delays, tmp_path_factory
):
    """the recursion of Parratt over the layers for a block of delays at once"""
    parratt = Parratt(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )
    parratt.energy = np.r_[700, 800] * u.eV
    parratt.qz = np.r_[0.01:5:0.05] / u.nm
    parratt.set_polarization(3, 0)
    benchmark(parratt.calc_inhomogeneous_reflectivity_transmissivity, strain_map=strain_map_delays)
