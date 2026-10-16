#!/usr/bin/env python

import numpy as np
import pytest

from udkm1Dsim import Heat, u
from udkm1Dsim.helpers import finderb
from udkm1Dsim.simulations.solvers.heat_scipy import odefunc

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


def test_calc_energy_map(benchmark, heat):
    dists, _, _ = heat.S.get_distances_of_layers()
    temp_map = 1000 * np.ones((100, len(dists)))
    benchmark(heat.calc_energy_map, temp_map, 300)


def test_calc_energy_flux_map(benchmark, heat):
    delays = np.r_[-1:10:0.01] * u.ps
    dists, _, _ = heat.S.get_distances_of_layers()

    temp_map = 1000 * np.ones((100, len(dists)))
    delta_map = temp_map
    benchmark(heat.calc_energy_flux_map, temp_map, delta_map, delays)


def test_odefunc(benchmark, heat):
    distances, _ = heat.S.interp_distance_at_interfaces(heat.intp_at_interface, False)
    N = len(distances)
    K = heat.S.num_sub_systems
    d_start, _, _ = heat.S.get_distances_of_layers(False)
    d_distances = np.diff(distances)
    dAdz = np.zeros_like(distances)
    fluence = 0
    delay_pump = 0
    pulse_width = 0

    # prepare the inputs per unique layer as in Heat.calc_heat_diffusion
    _, unique_layers = heat.S.get_unique_layers()
    unique_indices = np.asarray(heat.S.get_layer_vectors()[0], dtype=int)
    indices = unique_indices[finderb(distances, d_start)]
    groups = [(idx, np.flatnonzero(indices == idx)) for idx in np.unique(indices)]
    therm_conds = [layer.thermal.therm_cond.get_functional() for layer in unique_layers]
    therm_cond_takes_vector = [
        layer.thermal.therm_cond.takes_temperature_vector for layer in unique_layers
    ]
    heat_capacities = [layer.thermal.heat_capacity.get_functional() for layer in unique_layers]
    sub_system_couplings = [
        layer.thermal.sub_system_coupling.get_functional() for layer in unique_layers
    ]
    densities = np.array([layer.structural.density.magnitude for layer in unique_layers])
    init_temp = heat.check_initial_temperature(300, distances)

    benchmark(
        odefunc,
        0,
        init_temp,
        N,
        K,
        d_distances,
        therm_conds,
        therm_cond_takes_vector,
        heat_capacities,
        sub_system_couplings,
        densities[indices],
        groups,
        dAdz,
        fluence,
        delay_pump,
        pulse_width,
        0,
        0,
        0,
        0,
        None,
        None,
    )
