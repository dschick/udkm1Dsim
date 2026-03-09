#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Phonon, PhononNum, PhononAna
from udkm1Dsim import u
import numpy as np
import pytest


# fixtures


@pytest.fixture(scope='module')
def phonon(structure, tmp_path_factory):
    return Phonon(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                  save_data=True, disp_messages=True, progress_bar=True,
                  )


@pytest.fixture(scope='module')
def phonon_num(structure, tmp_path_factory):
    return PhononNum(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                     save_data=True, disp_messages=True, progress_bar=True,
                     )


@pytest.fixture(scope='module')
def phonon_ana(structure, tmp_path_factory):
    return PhononAna(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                     save_data=True, disp_messages=True, progress_bar=True,
                     )


@pytest.fixture(scope='module')
def delays():
    return np.r_[-1:10:0.01]*u.ps


@pytest.fixture(scope='module')
def distances(structure):
    dists, _, _ = structure.get_distances_of_layers()
    return dists


@pytest.fixture(scope='module')
def temp_map(delays, distances):
    return np.zeros((len(delays), len(distances)))


@pytest.fixture(scope='module')
def strain_map(delays, distances):
    return np.zeros((len(delays), len(distances)))


# tests


# Phonon


def test_phonon_str(phonon):
    phonon.__str__()


def test_get_hash(phonon, delays, distances, temp_map):
    phonon.get_hash(delays, temp_map, temp_map)


def test_get_all_strains_per_unique_layer(phonon, strain_map):
    assert np.allclose(phonon.get_all_strains_per_unique_layer(strain_map), [])


def test_get_reduced_strains_per_unique_layer(phonon, strain_map):
    assert np.allclose(phonon.get_reduced_strains_per_unique_layer(strain_map), [])


def test_check_temp_maps(phonon, temp_map, delays):
    phonon.check_temp_maps(temp_map, temp_map, delays)


def test_calc_sticks_from_temp_map(phonon, temp_map):
    phonon.calc_sticks_from_temp_map(temp_map, temp_map)


# PhononNum


def test_phonon_num_str(phonon_num):
    phonon_num.__str__()


def test_phonon_num_get_strain_map(phonon_num, delays, temp_map):
    phonon_num.get_strain_map(delays, temp_map, temp_map)
    phonon_num.force_recalc = False
    phonon_num.get_strain_map(delays, temp_map, temp_map)


# PhononAna


def test_phonon_ana_str(phonon_ana):
    phonon_ana.__str__()


def test_phonon_ana_get_strain_map(phonon_ana, delays, temp_map):
    phonon_ana.get_strain_map(delays, temp_map, temp_map)
    phonon_ana.force_recalc = False
    phonon_ana.get_strain_map(delays, temp_map, temp_map)


def test_phonon_ana_get_energy_per_eigenmode(phonon_ana, delays, temp_map):
    _, A, B = phonon_ana.get_strain_map(delays, temp_map, temp_map)
    phonon_ana.get_energy_per_eigenmode(A, B)
