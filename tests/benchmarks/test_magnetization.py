#!/usr/bin/env python

import numpy as np
import pytest

import udkm1Dsim as ud
from udkm1Dsim import LLB, LLG, iLLG, u

# fixtures


@pytest.fixture(scope="module")
def structure_magnetic():
    """a Ni-like film with all parameters of the LLG, iLLG, and LLB"""
    atom = ud.Atom("Ni")
    atom.magnetization = (1, 0, 0)
    layer = ud.AmorphousLayer(
        "Ni",
        "Ni",
        thickness=1 * u.nm,
        density=8900 * u.kg / u.m**3,
        atom=atom,
        heat_capacity=444 * u.J / (u.kg * u.K),
        therm_cond=90 * u.W / (u.m * u.K),
        lin_therm_exp=1.3e-5,
        sub_system_coupling=0,
        eff_spin=0.5,
        curie_temp=630 * u.K,
        lamda=0.01,
        mag_moment=0.6 * u.bohr_magneton,
        mag_saturation=4.9e5 * u.A / u.m,
        gilbert_damping=0.1,
        inertial_time=1 * u.fs,
        aniso_exponent=2,
        anisotropy=[0, 0, 1e5] * u.J / u.m**3,
        exch_stiffness=1e-11 * u.J / u.m,
    )
    S = ud.Structure("Ni film")
    S.add_sub_structure(layer, 20)
    return S


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
