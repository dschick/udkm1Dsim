#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Heat
from udkm1Dsim import u
import numpy as np
import pytest
from pint.testing import assert_allclose


# fixtures


@pytest.fixture(scope='module')
def heat(structure, tmp_path_factory):
    return Heat(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                save_data=True, disp_messages=True, progress_bar=True,
                )


# tests


def test_heat_str(heat):
    heat.__str__()


def test_get_hash(heat):
    heat.get_hash(np.r_[0:10:1], 300)


def test_check_initial_temperature(heat):
    assert np.allclose(heat.check_initial_temperature(300), 300)
    with pytest.raises(ValueError):
        heat.check_initial_temperature(np.r_[300, 300])


def test_check_excitation(heat):
    heat.heat_diffusion = False
    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [0]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': True,
                       'backside': False}
    res, fluence, delay_pump, pulse_width = heat.check_excitation(np.r_[-10:10:1]*u.ps)
    assert np.allclose(fluence, [70.71067812])
    assert np.allclose(delay_pump, [0.])
    assert np.allclose(pulse_width, [0.])

    heat.heat_diffusion = True
    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [1]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': True,
                       'backside': False}
    res, fluence, delay_pump, pulse_width = heat.check_excitation(np.r_[-10:10:1]*u.ps)
    assert np.allclose(fluence, [70.71067812])
    assert np.allclose(delay_pump, [0.])
    assert np.allclose(pulse_width, [1.0e-12])


def test_get_absorption_profile(heat):
    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [0]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': False,
                       'backside': False}
    heat.get_absorption_profile()

    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [0]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': False,
                       'backside': True}
    heat.get_absorption_profile()

    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [0]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': True,
                       'backside': False}
    heat.get_absorption_profile()

    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [0]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': False,
                       'backside': False}
    heat.get_absorption_profile(distances=np.r_[0:10e-9:1e-10])


def test_get_temperature_after_delta_excitation(heat):
    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [0]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': True,
                       'backside': False}
    heat.get_temperature_after_delta_excitation(10, 300)
    heat.get_temperature_after_delta_excitation(10, 300, distances=np.r_[0:10e-9:1e-10])


def test_get_temp_map(heat):
    print('\n test_get_temp_map\n')
    delays = np.r_[-1:10:0.01]*u.ps
    init_temp = 300*u.K
    heat.backend = 'scipy'
    heat.heat_diffusion = False
    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [0]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': True,
                       'backside': False}
    heat.get_temp_map(delays, init_temp)

    # load data from cache
    heat.force_recalc = False
    heat.get_temp_map(delays, init_temp)

    heat.backend = 'matlab'
    heat.force_recalc = True
    heat.heat_diffusion = True
    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [0.1]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': True,
                       'backside': False}
    with pytest.raises(Warning):
        heat.get_temp_map(delays, init_temp)

    heat.backend = 'scipy'


@pytest.mark.skip(reason="takes too much time in CI")
def test_get_diffusion_temp_map(heat):
    print('\n test_get_temp_map\n')
    delays = np.r_[-1:10:0.01]*u.ps
    init_temp = 300*u.K
    heat.backend = 'scipy'
    heat.heat_diffusion = True
    heat.excitation = {'fluence': [10]*u.mJ/u.cm**2,
                       'delay_pump': [0]*u.ps,
                       'pulse_width': [0]*u.ps,
                       'wavelength': 800*u.nm,
                       'theta': 45*u.deg,
                       'multilayer_absorption': True,
                       'backside': False}
    heat.get_temp_map(delays, init_temp)


def test_ode_func(heat):
    t = np.r_[0:1]
    u = np.r_[0:2]
    N = 2
    K = 1
    d_x_grid = np.r_[0.5, 0.5]
    x = np.r_[0:2]
    thermal_conds = [[lambda T: 1], [lambda T: 1]]
    heat_capacities = [[lambda T: 1], [lambda T: 1]]
    sub_system_coupling = [[lambda T: 1], [lambda T: 1]]
    densities = np.r_[1:3]
    indices = np.r_[0:2]
    dAdz = np.r_[0:2]
    fluence = np.r_[1]
    delay_pump = np.r_[0]
    pulse_length = np.r_[1]
    bc_top_type = 'isolator'
    bc_top_value = np.r_[1]
    bc_bottom_type = 'isolator'
    bc_bottom_value = np.r_[1]
    pbar = None
    state = (0, 1)

    heat.odefunc(t, u, N, K, d_x_grid, x, thermal_conds, heat_capacities,
                 sub_system_coupling, densities, indices, dAdz, fluence,
                 delay_pump, pulse_length, bc_top_type, bc_top_value,
                 bc_bottom_type, bc_bottom_value, pbar, state)


def test_boundary_conditions(heat):
    heat.boundary_conditions = {'top_type': 'isolator', 'bottom_type': 'isolator'}
    assert heat.boundary_conditions == {'top_type': 'isolator', 'bottom_type': 'isolator'}

    heat.boundary_conditions = {'top_type': 'temperature', 'top_value': 300*u.K,
                                'bottom_type': 'temperature', 'bottom_value': 300*u.K}
    assert heat.boundary_conditions == {'top_type': 'temperature', 'top_value': 300*u.K,
                                        'bottom_type': 'temperature', 'bottom_value': 300*u.K}

    heat.boundary_conditions = {'top_type': 'flux', 'top_value': 1*u.W/u.m**2,
                                'bottom_type': 'flux', 'bottom_value': 1*u.W/u.m**2}
    assert heat.boundary_conditions == {'top_type': 'flux', 'top_value': 1*u.W/u.m**2,
                                        'bottom_type': 'flux', 'bottom_value': 1*u.W/u.m**2}

    with pytest.raises(ValueError):
        heat.boundary_conditions = {'top_type': 'test'}
    with pytest.raises(ValueError):
        heat.boundary_conditions = {'bottom_type': 'test'}
    with pytest.raises(ValueError):
        heat.boundary_conditions = {'top_value': [1, 2]*u.W/u.m**2}
    with pytest.raises(ValueError):
        heat.boundary_conditions = {'bottom_value': [1, 2]*u.W/u.m**2}
    with pytest.raises(ValueError):
        heat.boundary_conditions = 'test'


def test_distances(heat):
    heat.distances = np.r_[0:10e-9:1e-10]*u.nm
    assert_allclose(heat.distances, np.r_[0:10e-9:1e-10]*u.nm)
