#!/usr/bin/env python

import numpy as np
import pytest

from udkm1Dsim import XrayDyn, XrayDynMag, XrayKin, u

# fixtures


@pytest.fixture(scope='module')
def xray_kin(structure_crystalline, tmp_path_factory):
    xray_kin = XrayKin(structure_crystalline, force_recalc=True,
                       cache_dir=tmp_path_factory.mktemp('cache'),
                       save_data=False, disp_messages=False, progress_bar=False,
                       )
    xray_kin.energy = np.r_[5000, 8000]*u.eV
    xray_kin.qz = np.r_[1:5:0.001]/u.nm
    return xray_kin


@pytest.fixture(scope='module')
def xray_dyn(structure_crystalline, tmp_path_factory):
    xray_dyn = XrayDyn(structure_crystalline, force_recalc=True,
                       cache_dir=tmp_path_factory.mktemp('cache'),
                       save_data=False, disp_messages=False, progress_bar=False,
                       )
    xray_dyn.energy = np.r_[5000, 8000]*u.eV
    xray_dyn.qz = np.r_[1:5:0.004]/u.nm
    return xray_dyn


@pytest.fixture(scope='module')
def xray_dyn_mag(structure, tmp_path_factory):
    xray_dyn_mag = XrayDynMag(structure, force_recalc=True,
                              cache_dir=tmp_path_factory.mktemp('cache'),
                              save_data=False, disp_messages=False, progress_bar=False,
                              )
    xray_dyn_mag.energy = np.r_[700, 800]*u.eV
    xray_dyn_mag.qz = np.r_[0.01:5:0.02]/u.nm
    return xray_dyn_mag


@pytest.fixture(scope='module')
def strain_map_crystalline(structure_crystalline):
    L = structure_crystalline.get_number_of_layers()
    M = 10  # delay steps
    rng = np.random.default_rng(0)
    return 0.01*rng.random((M, L))


@pytest.fixture(scope='module')
def strain_map_mixed(structure):
    L = structure.get_number_of_layers()
    M = 1  # delay steps
    return 0.01*np.ones((M, L))


@pytest.fixture(scope='module')
def magnetization_map(structure):
    L = structure.get_number_of_layers()
    M = 1  # delay steps
    magnetization_map = np.zeros((M, L, 3))
    magnetization_map[:, :, 0] = 0.5  # amplitude
    magnetization_map[:, :, 2] = np.pi  # gamma
    return magnetization_map


@pytest.fixture(scope='module')
def ref_trans_matrix(xray_dyn_mag):
    A0, A0_phi, _, _, _, _, k_z_0 = xray_dyn_mag.get_atom_boundary_phase_matrix([], 0, 0)
    RT, *_ = xray_dyn_mag.calc_homogeneous_matrix(xray_dyn_mag.S, A0, A0_phi, k_z_0)
    return RT


@pytest.fixture(scope='module')
def polarizations():
    # 18 elliptical polarizations
    return [(alpha*u.deg, 0.2) for alpha in range(0, 180, 10)]


# benchmarks


# XrayKin


@pytest.mark.benchmark
def test_xray_kin_homogeneous_reflectivity(xray_kin):
    xray_kin.homogeneous_reflectivity()


# XrayDyn


@pytest.mark.benchmark
def test_xray_dyn_homogeneous_reflectivity(xray_dyn):
    xray_dyn.homogeneous_reflectivity()


@pytest.mark.benchmark
def test_xray_dyn_inhomogeneous_reflectivity(xray_dyn, strain_map_crystalline):
    xray_dyn.inhomogeneous_reflectivity(strain_map_crystalline)


@pytest.mark.benchmark
def test_xray_dyn_inhomogeneous_reflectivity_strain_vectors(xray_dyn, strain_map_crystalline):
    strain_vectors = [np.linspace(0, 0.01, 20)]*xray_dyn.S.get_number_of_unique_layers()
    xray_dyn.inhomogeneous_reflectivity(strain_map_crystalline, strain_vectors=strain_vectors)


# XrayDynMag


@pytest.mark.benchmark
def test_xray_dyn_mag_homogeneous_reflectivity(xray_dyn_mag):
    xray_dyn_mag.set_polarization(3, 0)
    xray_dyn_mag.homogeneous_reflectivity()


@pytest.mark.benchmark
def test_xray_dyn_mag_inhomogeneous_reflectivity(xray_dyn_mag, strain_map_mixed,
                                                 magnetization_map):
    xray_dyn_mag.set_polarization(3, 0)
    xray_dyn_mag.inhomogeneous_reflectivity(strain_map=strain_map_mixed,
                                            magnetization_map=magnetization_map)


@pytest.mark.benchmark
def test_xray_dyn_mag_homogeneous_reflectivity_elliptical(xray_dyn_mag, polarizations):
    xray_dyn_mag.set_polarization(5, 0, polarization_in=polarizations)
    xray_dyn_mag.homogeneous_reflectivity()


def test_xray_dyn_mag_calc_reflectivity_transmissivity_from_matrix(
        benchmark, ref_trans_matrix, polarizations):
    pol_in = XrayDynMag.calc_elliptical_polarization(polarizations)
    pol_out = np.array([], dtype=np.complex128)  # no analyzer
    benchmark(XrayDynMag.calc_reflectivity_transmissivity_from_matrix,
              ref_trans_matrix, pol_in, pol_out)
