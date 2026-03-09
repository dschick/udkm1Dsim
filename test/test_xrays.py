#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Xray, XrayKin, XrayDyn, XrayDynMag
from udkm1Dsim import u
import numpy as np
import pytest
from pint.testing import assert_allclose


# fixtures


@pytest.fixture(scope='module')
def xray(structure, tmp_path_factory):
    return Xray(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                save_data=True, disp_messages=True, progress_bar=True,
                )


@pytest.fixture(scope='module')
def xray_kin(structure_crystalline, tmp_path_factory):
    return XrayKin(structure_crystalline, force_recalc=True,
                   cache_dir=tmp_path_factory.mktemp('cache'),
                   save_data=True, disp_messages=True, progress_bar=True,
                   )


@pytest.fixture(scope='module')
def xray_dyn(structure_crystalline, tmp_path_factory):
    return XrayDyn(structure_crystalline, force_recalc=True,
                   cache_dir=tmp_path_factory.mktemp('cache'),
                   save_data=True, disp_messages=True, progress_bar=True,
                   )


@pytest.fixture(scope='module')
def xray_dyn_mag(structure, tmp_path_factory):
    return XrayDynMag(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                      save_data=True, disp_messages=True, progress_bar=True,
                      )


@pytest.fixture(scope='module')
def delays():
    return np.r_[-1:10:1]*u.ps


@pytest.fixture(scope='module')
def distances(structure_amorph):
    dists, _, _ = structure_amorph.get_distances_of_layers()
    return dists


@pytest.fixture(scope='module')
def temp_map(delays, distances):
    return 50*np.ones([len(delays), len(distances), 1])


@pytest.fixture(scope='module')
def strain_map(delays, distances):
    return 50*np.ones([len(delays), len(distances), 1])


# tests


# Xray


def text_xray_set_incoming_polarization(xray):
    with pytest.raises(NotImplementedError):
        xray.set_incoming_polarization(1)


def text_xray_set_outgoing_polarization(xray):
    with pytest.raises(NotImplementedError):
        xray.set_outgoing_polarization(1)


# XrayKin


def text_xray_kin_str(xray_kin):
    xray_kin.__str__()


def test_xray_kin_update_experiment(xray_kin):
    # energy, wl, or k must be set first
    with pytest.raises(IndexError):
        xray_kin.theta = 45*u.deg
    xray_kin.energy = 1000*u.eV
    xray_kin.theta = 45*u.deg
    assert_allclose(xray_kin.qz, np.array([[7.1669]])/u.nm, rtol=1e-2)
    xray_kin.qz = 1/u.nm
    assert_allclose(xray_kin.theta, np.array([[5.66]])*u.deg, rtol=1e-2)
    assert_allclose(xray_kin.wl, np.array([1.24])*u.nm, rtol=1e-2)
    assert_allclose(xray_kin.k, np.array([5.07])/u.nm, rtol=1e-2)
    xray_kin.k = 1/u.nm
    xray_kin.wl = 1*u.nm


def text_xray_kin_set_polarization(xray_kin):
    xray_kin.set_polarization(0, 0)
    xray_kin.set_polarization(1, 1)
    xray_kin.set_polarization(2, 0)
    xray_kin.set_polarization(3, 0)
    xray_kin.set_polarization(4, 0)


def test_xray_kin_get_polarization_factor(xray_kin):
    xray_kin.get_polarization_factor(0)


def test_xray_kin_get_hash(xray_kin, strain_map):
    strain_vectors = {}
    xray_kin.get_hash(strain_vectors, strain_map=strain_map)


def test_xray_kin_get_uc_structure_factor(xray_kin, unit_cell_iron):
    assert_allclose(xray_kin.get_uc_structure_factor(1000, 1, unit_cell_iron, strain=0.01),
                    -15.52-11.72j, rtol=1e-2)


def test_xray_kin_homogeneous_reflectivity(xray_kin):
    xray_kin.homogeneous_reflectivity()
