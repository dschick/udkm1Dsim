#!/usr/bin/env python

import numpy as np
import pytest

from udkm1Dsim import PhononAna, PhononNum, u
from udkm1Dsim.simulations.solvers.phonon_scipy import force_from_heat as calc_force_from_heat
from udkm1Dsim.simulations.solvers.phonon_scipy import odefunc

# fixtures


@pytest.fixture(scope="module")
def phonon_num(structure, tmp_path_factory):
    # the same settings as the numba backend for a fair comparison of both
    return PhononNum(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )


@pytest.fixture(scope="module")
def phonon_num_numba(structure, tmp_path_factory, delays, temp_map, delta_temp_map):
    pytest.importorskip("numba")
    phonon_num = PhononNum(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
        backend="numba",
    )
    # compile the numba functions once, which is not benchmarked
    phonon_num.calc_strain_map(delays[:10], temp_map[:10], delta_temp_map[:10])
    return phonon_num


@pytest.fixture(scope="module")
def phonon_ana(structure, tmp_path_factory):
    return PhononAna(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )


@pytest.fixture(scope="module")
def delays():
    return np.r_[-1:10:0.01] * u.ps


@pytest.fixture(scope="module")
def temp_map(structure, delays):
    # step-like heating at t = 0 with an exponential profile of 20 nm
    _, _, distances = structure.layer_distances
    heated = np.exp(-distances.to("nm").magnitude / 20)
    return 300 + 100 * (delays.magnitude >= 0)[:, np.newaxis] * heated[np.newaxis, :]


@pytest.fixture(scope="module")
def delta_temp_map(temp_map):
    diff_temp_map = np.zeros_like(temp_map)
    diff_temp_map[1:, :] = np.diff(temp_map, axis=0)
    return diff_temp_map


@pytest.fixture(scope="module")
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
    L = structure.num_layers
    masses = structure.get_layer_property_vector("structural.mass_unit_area.magnitude")
    spring_consts = structure.get_layer_property_vector("elastic.spring_const.magnitude")
    damping = structure.get_layer_property_vector("elastic.phonon_damping.magnitude")
    force_from_heat = calc_force_from_heat(sticks, spring_consts)
    x0 = np.zeros([2 * L])
    delays = delays.to("s").magnitude
    benchmark(
        odefunc,
        0,
        x0,
        delays,
        force_from_heat,
        damping,
        spring_consts,
        masses,
        L,
        None,
        None,
    )


@pytest.mark.benchmark
def test_phonon_num_calc_strain_map(phonon_num, delays, temp_map, delta_temp_map):
    phonon_num.calc_strain_map(delays, temp_map, delta_temp_map)


@pytest.mark.benchmark
def test_phonon_num_calc_strain_map_numba(phonon_num_numba, delays, temp_map, delta_temp_map):
    phonon_num_numba.calc_strain_map(delays, temp_map, delta_temp_map)


# PhononAna


@pytest.mark.benchmark
def test_phonon_ana_calc_eigenproblem(phonon_ana):
    phonon_ana.calc_eigenproblem()


@pytest.mark.benchmark
def test_phonon_ana_calc_strain_map(phonon_ana, delays, temp_map, delta_temp_map):
    phonon_ana.calc_strain_map(delays, temp_map, delta_temp_map)
