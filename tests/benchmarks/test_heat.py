#!/usr/bin/env python

from copy import deepcopy

import numpy as np
import pytest

from udkm1Dsim import Dispersion, Heat, Layer, Structure, u
from udkm1Dsim.helpers import finderb
from udkm1Dsim.simulations.solvers.heat_scipy import odefunc

# fixtures


def simulation(S, tmp_path_factory, **kwargs):
    return Heat(
        S,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
        **kwargs,
    )


@pytest.fixture(scope="module")
def heat(structure, tmp_path_factory):
    return simulation(structure, tmp_path_factory)


def excited(S, tmp_path_factory, pulse_width, **kwargs):
    heat = simulation(S, tmp_path_factory, **kwargs)
    heat.excitation = {
        "fluence": [5] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [pulse_width] * u.ps,
    }
    return heat


@pytest.fixture(scope="module")
def heat_delta(structure_amorph, tmp_path_factory):
    """a delta excitation without heat diffusion, which neglects the pulse width"""
    return excited(structure_amorph, tmp_path_factory, 0)


@pytest.fixture(scope="module")
def heat_excited(structure_amorph, tmp_path_factory):
    return excited(structure_amorph, tmp_path_factory, 0.1, heat_diffusion=True)


@pytest.fixture(scope="module")
def heat_two_temperatures(tmp_path_factory):
    """a two-temperature model of electrons and phonons of a metal film with
    temperature-dependent heat capacity and thermal conductivity of the
    electrons on an insulating substrate"""
    metal = Layer(
        "metal",
        "metal",
        thickness=1 * u.nm,
        density=8900 * u.kg / u.m**3,
        heat_capacity=["0.08*T", 420],
        therm_cond=["90*T_0/T_1", 1],
        sub_system_coupling=["-4e17*(T_0-T_1)", "4e17*(T_0-T_1)"],
        lin_therm_exp=[0, 1.3e-5],
        dielectric=Dispersion.from_ref_index(2.2 + 4.9j),
    )
    substrate = Layer(
        "substrate",
        "substrate",
        thickness=1 * u.nm,
        density=2200 * u.kg / u.m**3,
        heat_capacity=[1, 740],
        therm_cond=[0, 1.4],
        sub_system_coupling=[0, 0],
        lin_therm_exp=[0, 5e-7],
        dielectric=Dispersion.from_ref_index(1.45),
    )
    S = Structure("two-temperature model")
    S.add_sub_structure(metal, 20)
    S.add_sub_structure(substrate, 50)
    return excited(S, tmp_path_factory, 0.1, heat_diffusion=True)


@pytest.fixture(scope="module")
def heat_thick_substrate(structure_amorph, amorphous_layer_iron, tmp_path_factory):
    """a substrate of a million layers, as e.g. a substrate of unit cells"""
    S = deepcopy(structure_amorph)
    S.add_substrate(amorphous_layer_iron, 1000000)
    return Heat(
        S,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )


@pytest.fixture(scope="module")
def heat_bandwidth(structure, tmp_path_factory):
    """a pulse with 7 x 7 nodes of its bandwidth and divergence"""
    heat = Heat(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )
    heat.excitation = {
        "fluence": [5] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0.1] * u.ps,
        "theta": 45 * u.deg,
        "bandwidth": 20 * u.nm,
        "divergence": 2 * u.deg,
    }
    return heat


# benchmarks


@pytest.mark.benchmark
def test_get_absorption_from_multilayer_bandwidth(heat_bandwidth):
    heat_bandwidth.get_absorption_from_multilayer()


@pytest.mark.benchmark
def test_get_absorption_from_lambert_beer(heat):
    heat.get_absorption_from_lambert_beer()


@pytest.mark.benchmark
def test_get_absorption_from_multilayer(heat):
    heat.get_absorption_from_multilayer()


@pytest.mark.benchmark
def test_get_absorption_from_multilayer_thick_substrate(heat_thick_substrate):
    heat_thick_substrate.get_absorption_from_multilayer()


@pytest.mark.benchmark
def test_get_temperature_after_delta_excitation(heat):
    heat.calc_temperature_after_delta_excitation(10, 300)


def test_calc_energy_map(benchmark, heat):
    dists, _, _ = heat.S.layer_distances
    temp_map = 1000 * np.ones((100, len(dists)))
    benchmark(heat.calc_energy_map, temp_map, 300)


def test_calc_energy_flux_map(benchmark, heat):
    delays = np.linspace(-1, 10, 100) * u.ps
    dists, _, _ = heat.S.layer_distances

    temp_map = 1000 * np.ones((100, len(dists)))
    delta_map = temp_map
    benchmark(heat.calc_energy_flux_map, temp_map, delta_map, delays)


def test_odefunc(benchmark, heat):
    distances, _ = heat.S.interp_distance_at_interfaces(heat.intp_at_interface, False)
    N = len(distances)
    K = heat.S.num_sub_systems
    d_start, _, _ = heat.S._layer_distances()
    d_distances = np.diff(distances)
    dAdz = np.zeros_like(distances)
    fluence = 0
    delay_pump = 0
    pulse_width = 0

    # prepare the inputs per unique layer as in Heat.calc_heat_diffusion
    _, unique_layers = heat.S.unique_layers
    unique_indices = np.asarray(heat.S.layer_vectors[0], dtype=int)
    indices = unique_indices[finderb(distances, d_start)]
    groups = [(idx, np.flatnonzero(indices == idx)) for idx in np.unique(indices)]
    therm_conds = [layer.thermal.therm_cond.functional for layer in unique_layers]
    therm_cond_takes_vector = [
        layer.thermal.therm_cond.takes_temperature_vector for layer in unique_layers
    ]
    heat_capacities = [layer.thermal.heat_capacity.functional for layer in unique_layers]
    sub_system_couplings = [
        layer.thermal.sub_system_coupling.functional for layer in unique_layers
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


@pytest.mark.benchmark
def test_calc_temp_map(heat_delta):
    heat_delta.calc_temp_map(np.r_[-1:10:0.1] * u.ps, 300)


@pytest.mark.benchmark
def test_calc_temp_map_heat_diffusion(heat_excited):
    heat_excited.calc_temp_map(np.r_[-1:10:0.1] * u.ps, 300)


@pytest.mark.benchmark
def test_calc_temp_map_two_temperatures(heat_two_temperatures):
    heat_two_temperatures.calc_temp_map(np.r_[-1:10:0.1] * u.ps, 300)
