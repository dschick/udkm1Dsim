#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import PhononNum
import numpy as np
import pytest


# fixtures


@pytest.fixture(scope='module')
def phonon_num(structure, tmp_path_factory):
    return PhononNum(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                     save_data=True, disp_messages=True, progress_bar=True,
                     )


# benchmarks


def test_phonon_num_ode_func(benchmark, phonon_num):
    L = phonon_num.S.get_number_of_layers()
    masses = phonon_num.S.get_layer_property_vector('_mass_unit_area')
    spring_consts = phonon_num.S.get_layer_property_vector('spring_const')
    damping = phonon_num.S.get_layer_property_vector('_phonon_damping')
    sticks = np.ones((2, L))
    force_from_heat = PhononNum.calc_force_from_heat(sticks, spring_consts)
    x0 = np.zeros([2*L])
    benchmark(PhononNum.ode_func,
              0,
              x0,
              [1, 0],
              force_from_heat,
              damping,
              spring_consts,
              masses,
              L,
              None,
              None
              )


def test_phonon_num_calc_strain_map(benchmark, phonon_num):
    L = phonon_num.S.get_number_of_layers()
    delays = np.linspace(0, 1, 100)
    temp_map = np.ones((len(delays), L))
    delta_temp_map = np.zeros_like(temp_map)
    delta_temp_map[1, :] = 0.1
    benchmark(phonon_num.calc_strain_map, delays, temp_map, delta_temp_map)
