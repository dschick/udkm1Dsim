#!/usr/bin/env python

import numpy as np
import pytest
from pint.testing import assert_allclose

from udkm1Dsim import XrayDyn, XrayKin, u

# fixtures


@pytest.fixture(scope="module")
def xray_kin(structure_crystalline, tmp_path_factory):
    return XrayKin(
        structure_crystalline,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def xray_dyn(structure_crystalline, tmp_path_factory):
    return XrayDyn(
        structure_crystalline,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def delays():
    return np.r_[-1:10:1] * u.ps


@pytest.fixture(scope="module")
def distances(structure_amorph):
    dists, _, _ = structure_amorph.get_distances_of_layers()
    return dists


@pytest.fixture(scope="module")
def temp_map(delays, distances):
    return 50 * np.ones([len(delays), len(distances)])


@pytest.fixture(scope="module")
def strain_map(delays, distances):
    return 0.01 * np.ones([len(delays), len(distances)])


# tests


# XrayKin


def test_xray_kin_update_experiment(xray_kin):
    # energy, wl, or k must be set first
    with pytest.raises(IndexError):
        xray_kin.theta = 45 * u.deg
    xray_kin.energy = 1000 * u.eV
    xray_kin.theta = 45 * u.deg
    assert_allclose(xray_kin.qz, np.array([[7.1669]]) / u.nm, rtol=1e-2)
    xray_kin.qz = 1 / u.nm
    assert_allclose(xray_kin.theta, np.array([[5.66]]) * u.deg, rtol=1e-2)
    assert_allclose(xray_kin.wl, np.array([1.24]) * u.nm, rtol=1e-2)
    assert_allclose(xray_kin.k, np.array([5.07]) / u.nm, rtol=1e-2)
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


def test_xray_kin_get_polarization_factor(xray_kin):
    xray_kin.get_polarization_factor(0)


def test_xray_kin_get_hash(xray_kin, strain_map):
    strain_vectors = {}
    xray_kin.get_hash(strain_vectors, strain_map=strain_map)


def test_xray_kin_get_uc_structure_factor(xray_kin, unit_cell_iron):
    # for q_z -> 0, the structure factor is the sum of the atomic form factors
    assert_allclose(
        xray_kin.get_uc_structure_factor(1000, 0, unit_cell_iron, strain=0.01),
        sum(atom.get_atomic_form_factor(1000) for atom, _, _ in unit_cell_iron.atoms),
        rtol=1e-9,
    )


def test_xray_kin_homogeneous_reflectivity(xray_kin):
    xray_kin.homogeneous_reflectivity()


# XrayDyn


def test_xray_dyn_repr(xray_dyn):
    xray_dyn.energy = 8000 * u.eV
    xray_dyn.theta = np.r_[1:10] * u.deg
    assert repr(xray_dyn).startswith("Dynamical X-Ray Diffraction simulation properties:")
    assert str(xray_dyn) == repr(xray_dyn)
    assert xray_dyn._repr_html_().startswith("<h2>Dynamical X-Ray Diffraction simulation</h2>")


def test_xray_dyn_get_hash(xray_dyn, strain_map, temp_map):
    strain_vectors = {}
    xray_dyn.get_hash(strain_vectors, strain_map=strain_map, temp_map=temp_map)


def test_xray_dyn_homogeneous_reflectivity(xray_dyn):
    xray_dyn.homogeneous_reflectivity()
    xray_dyn.homogeneous_reflectivity(strains=np.zeros([2]))
    xray_dyn.homogeneous_reflectivity(temps=300 * np.ones([2]))
    xray_dyn.homogeneous_reflectivity(strains=np.zeros([2]), temps=300 * np.ones([2]))
    with pytest.raises(IndexError):
        xray_dyn.homogeneous_reflectivity(strains=np.zeros([3]))
    with pytest.raises(IndexError):
        xray_dyn.homogeneous_reflectivity(temps=300 * np.ones([3]))
    with pytest.raises(IndexError):
        xray_dyn.homogeneous_reflectivity(temps=300 * np.ones([2, 2]))


def test_xray_dyn_inhomogeneous_reflectivity(xray_dyn, strain_map, temp_map):
    xray_dyn.energy = 8000 * u.eV
    xray_dyn.theta = np.r_[1:10] * u.deg
    xray_dyn.inhomogeneous_reflectivity(strain_map)
    xray_dyn.inhomogeneous_reflectivity(strain_map, temp_map=temp_map)
    strain_vectors = [np.r_[0:1], np.r_[0:1]]
    with pytest.warns(UserWarning):
        xray_dyn.inhomogeneous_reflectivity(
            strain_map, strain_vectors=strain_vectors, temp_map=temp_map
        )
    xray_dyn.inhomogeneous_reflectivity(strain_map, strain_vectors=strain_vectors)
    xray_dyn.force_recalc = False
    xray_dyn.inhomogeneous_reflectivity(strain_map, strain_vectors=strain_vectors)
    xray_dyn.force_recalc = True
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
