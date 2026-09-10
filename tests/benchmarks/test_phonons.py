#!/usr/bin/env python

import numpy as np
import pytest

from udkm1Dsim import PhononNum, u

# fixtures


@pytest.fixture(scope='module')
def phonon_num(structure, tmp_path_factory):
    return PhononNum(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                     save_data=True, disp_messages=True, progress_bar=True,
                     )


@pytest.fixture(scope='module')
def delays():
    return np.r_[-1:10:0.01]*u.ps


@pytest.fixture(scope='module')
def temp_map(structure, delays):
    L = structure.get_number_of_layers()
    M = len(delays)
    temp_map = np.ones((M, L))
    for i in range(temp_map.shape[0]):
        for j in range(temp_map.shape[1]):
            temp_map[i, j] = 300*np.exp(-i)*np.exp(-j)
    return temp_map


@pytest.fixture(scope='module')
def delta_temp_map(temp_map):
    diff_temp_map = np.zeros_like(temp_map)
    diff_temp_map[1:, :] = np.diff(temp_map, axis=0)
    return diff_temp_map


@pytest.fixture(scope='module')
def sticks(phonon_num, temp_map, delta_temp_map):
    sticks, _ = phonon_num.calc_sticks_from_temp_map(temp_map, delta_temp_map)
    return sticks


# benchmarks


# Phonon


@pytest.mark.benchmark
def test_calc_sticks_from_temp_map(phonon_num, temp_map, delta_temp_map):
    phonon_num.calc_sticks_from_temp_map(temp_map, delta_temp_map)


# PhononNum


def test_phonon_num_ode_func(benchmark, structure, delays, sticks):
    L = structure.get_number_of_layers()
    masses = structure.get_layer_property_vector('_mass_unit_area')
    spring_consts = structure.get_layer_property_vector('spring_const')
    damping = structure.get_layer_property_vector('_phonon_damping')
    force_from_heat = PhononNum.calc_force_from_heat(sticks, spring_consts)
    x0 = np.zeros([2*L])
    delays = delays.to('s').magnitude
    benchmark(PhononNum.ode_func,
              0,
              x0,
              delays,
              force_from_heat,
              damping,
              spring_consts,
              masses,
              L,
              None,
              None
              )


@pytest.mark.benchmark
def test_phonon_num_calc_strain_map(phonon_num, delays, temp_map, delta_temp_map):
    phonon_num.calc_strain_map(delays, temp_map, delta_temp_map)
