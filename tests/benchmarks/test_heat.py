#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Heat
from udkm1Dsim.helpers import finderb
import pytest
import numpy as np


# fixtures


@pytest.fixture(scope='module')
def heat(structure, tmp_path_factory):
    return Heat(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                save_data=True, disp_messages=True, progress_bar=True,
                )


# benchmarks


@pytest.mark.benchmark
def test_get_Lambert_Beer_absorption_profile(heat):
    heat.get_Lambert_Beer_absorption_profile()


@pytest.mark.benchmark
def test_get_multilayers_absorption_profile(heat):
    heat.get_multilayers_absorption_profile()


@pytest.mark.benchmark
def test_get_temperature_after_delta_excitation(heat):
    heat.get_temperature_after_delta_excitation(10, 300)


def test_odefunc(benchmark, heat):
    distances, _ = heat.S.interp_distance_at_interfaces(
        heat.intp_at_interface, False
        )
    N = len(distances)
    K = heat.S.num_sub_systems
    d_start, _, _ = heat.S.get_distances_of_layers(False)
    d_distances = np.diff(distances)
    indices = finderb(distances, d_start)
    dAdz = np.zeros_like(distances)
    fluence = 0
    delay_pump = 0
    pulse_width = 0

    densities = heat.S.get_layer_property_vector('_density')
    init_temp = heat.check_initial_temperature(300, distances)
    therm_conds = heat.S.get_layer_property_vector('therm_cond')
    heat_capacities = heat.S.get_layer_property_vector('heat_capacity')
    sub_system_couplings = heat.S.get_layer_property_vector('sub_system_coupling')

    benchmark(Heat.odefunc,
              0,
              init_temp,
              N,
              K,
              d_distances,
              d_start,
              therm_conds,
              heat_capacities,
              sub_system_couplings,
              densities[indices],
              indices,
              dAdz,
              fluence,
              delay_pump,
              pulse_width,
              0,
              0,
              0,
              0,
              None,
              None
              )
