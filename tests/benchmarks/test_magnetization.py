#!/usr/bin/env python

import numpy as np
import pytest
import scipy.constants as constants

from tests.conftest import magnetic_layer
from udkm1Dsim import LLB, LLG, Structure, iLLG, u

# fixtures


@pytest.fixture(scope="module")
def delays():
    return np.r_[-1:5:0.05] * u.ps


@pytest.fixture(scope="module")
def temp_map(structure_magnetic, delays):
    """heating below the Curie temperature, which decays into the film"""
    L = structure_magnetic.num_layers
    temp_map = 300 * np.ones((len(delays), L))
    temp_map[delays.magnitude >= 0, :] += 300 * np.exp(-np.r_[0:L] / 5)
    return temp_map


@pytest.fixture(scope="module")
def structure_ferrimagnetic():
    """a film with two antiparallel sublattices of an antiferromagnetic
    mean-field exchange"""
    exchange = constants.k * np.array([[600, -100], [-100, 400]])
    layer = magnetic_layer("ab", mf_exch_coupling=exchange * u.J)
    layer.magnetization = [(1, 0, 0), (-1, 0, 0)]
    S = Structure("ferrimagnetic film")
    S.add_sub_structure(layer, 10)
    return S


def simulation(cls, S):
    return cls(S, True, save_data=False, disp_messages=False, progress_bar=False)


B_EXT = np.r_[0, 0, 0.1]

# benchmarks


@pytest.mark.parametrize("cls", [LLG, iLLG, LLB], ids=lambda cls: cls.__name__)
def test_calc_magnetization_map(benchmark, cls, structure_magnetic, delays, temp_map):
    sim = simulation(cls, structure_magnetic)
    benchmark(sim.calc_magnetization_map, delays, temp_map=temp_map, B_ext=B_EXT)


def test_calc_mean_field_mag_map(benchmark, structure_magnetic, temp_map):
    sim = simulation(LLB, structure_magnetic)
    benchmark(sim.calc_mean_field_mag_map, temp_map)


def test_calc_equilibrium_magnetization(benchmark, structure_magnetic, temp_map):
    sim = simulation(LLB, structure_magnetic)
    benchmark(sim.calc_equilibrium_magnetization, B_ext=B_EXT, temp_map=temp_map[0])


def test_calc_magnetization_map_sublattices(benchmark, structure_ferrimagnetic, delays, temp_map):
    """the LLB of two exchange-coupled sublattices of the top 10 layers"""
    sim = simulation(LLB, structure_ferrimagnetic)
    benchmark(sim.calc_magnetization_map, delays, temp_map=temp_map[:, :10], B_ext=B_EXT)
