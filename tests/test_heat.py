#!/usr/bin/env python

import numpy as np
import pytest
from pint.testing import assert_allclose

from udkm1Dsim import Heat, u
from udkm1Dsim.simulations.solvers import HAS_NUMBA

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


# tests


def test_heat_repr(heat):
    assert repr(heat).startswith("Heat simulation properties:")
    assert str(heat) == repr(heat)
    assert heat._repr_html_().startswith("<h2>Heat simulation</h2>")


def test_get_hash(heat):
    heat.get_hash(np.r_[0:10:1], 300)


def test_check_initial_temperature(heat):
    assert np.allclose(heat.check_initial_temperature(300), 300)
    with pytest.raises(ValueError):
        heat.check_initial_temperature(np.r_[300, 300])


def test_check_excitation(heat):
    heat.heat_diffusion = False
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": True,
        "backside": False,
    }
    res, fluence, delay_pump, pulse_width = heat.check_excitation(np.r_[-10:10:1] * u.ps)
    assert np.allclose(fluence, [70.71067812])
    assert np.allclose(delay_pump, [0.0])
    assert np.allclose(pulse_width, [0.0])

    heat.heat_diffusion = True
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [1] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "s",
        "theta": 45 * u.deg,
        "multilayer_absorption": True,
        "backside": False,
    }
    res, fluence, delay_pump, pulse_width = heat.check_excitation(np.r_[-10:10:1] * u.ps)
    assert np.allclose(fluence, [70.71067812])
    assert np.allclose(delay_pump, [0.0])
    assert np.allclose(pulse_width, [1.0e-12])

    with pytest.raises(Warning):
        heat.excitation = {
            "fluence": [10] * u.mJ / u.cm**2,
            "delay_pump": [0] * u.ps,
            "pulse_width": [1] * u.ps,
            "wavelength": 800 * u.nm,
            "polarization": "elliptical",
            "theta": 45 * u.deg,
            "multilayer_absorption": True,
            "backside": False,
        }


def test_get_absorption_profile(heat):
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": False,
        "backside": False,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": False,
        "backside": True,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": True,
        "backside": False,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": True,
        "backside": True,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "s",
        "theta": 45 * u.deg,
        "multilayer_absorption": True,
        "backside": False,
    }
    heat.get_absorption_profile()

    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": False,
        "backside": False,
    }
    heat.get_absorption_profile(distances=np.r_[0:10e-9:1e-10])


def test_get_temperature_after_delta_excitation(heat):
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": True,
        "backside": False,
    }
    heat.get_temperature_after_delta_excitation(10, 300)
    heat.get_temperature_after_delta_excitation(10, 300, distances=np.r_[0:10e-9:1e-10])


def test_get_temp_map(heat):
    delays = np.r_[-1:10:0.01] * u.ps
    init_temp = 300 * u.K
    heat.heat_diffusion = False
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": True,
        "backside": False,
    }
    heat.get_temp_map(delays, init_temp)

    # load data from cache
    heat.force_recalc = False
    heat.get_temp_map(delays, init_temp)


@pytest.mark.skip(reason="takes too much time in CI")
def test_get_diffusion_temp_map(heat):
    delays = np.r_[-1:10:0.01] * u.ps
    init_temp = 300 * u.K
    heat.heat_diffusion = True
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": True,
        "backside": False,
    }
    heat.get_temp_map(delays, init_temp)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")
def test_get_diffusion_temp_map_numba(heat):
    """the numba backend for the heat diffusion is not implemented, yet"""
    heat.heat_diffusion = True
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0.1] * u.ps,
        "wavelength": 800 * u.nm,
        "polarization": "p",
        "theta": 45 * u.deg,
        "multilayer_absorption": True,
        "backside": False,
    }
    heat.backend = "numba"
    try:
        # calc_temp_map, since the cached results do not depend on the backend
        with pytest.raises(NotImplementedError):
            heat.calc_temp_map(np.r_[-1:10:0.01] * u.ps, 300 * u.K)
    finally:
        heat.backend = "scipy"


def test_calc_energy_map(heat):
    dists, _, _ = heat.S.get_distances_of_layers()
    temp_map = np.zeros((10, len(dists)))
    heat.calc_energy_map(temp_map, 0)


def test_calc_energy_flux_map(heat):
    delays = np.r_[-1:10:0.01] * u.ps
    dists, _, _ = heat.S.get_distances_of_layers()

    temp_map = np.zeros((len(delays), len(dists)))
    delta_map = temp_map
    heat.calc_energy_flux_map(temp_map, delta_map, delays)


def test_boundary_conditions(heat):
    heat.boundary_conditions = {"top_type": "isolator", "bottom_type": "isolator"}
    assert heat.boundary_conditions == {"top_type": "isolator", "bottom_type": "isolator"}

    heat.boundary_conditions = {
        "top_type": "temperature",
        "top_value": 300 * u.K,
        "bottom_type": "temperature",
        "bottom_value": 300 * u.K,
    }
    assert heat.boundary_conditions == {
        "top_type": "temperature",
        "top_value": 300 * u.K,
        "bottom_type": "temperature",
        "bottom_value": 300 * u.K,
    }

    heat.boundary_conditions = {
        "top_type": "flux",
        "top_value": 1 * u.W / u.m**2,
        "bottom_type": "flux",
        "bottom_value": 1 * u.W / u.m**2,
    }
    assert heat.boundary_conditions == {
        "top_type": "flux",
        "top_value": 1 * u.W / u.m**2,
        "bottom_type": "flux",
        "bottom_value": 1 * u.W / u.m**2,
    }

    with pytest.raises(ValueError):
        heat.boundary_conditions = {"top_type": "test"}
    with pytest.raises(ValueError):
        heat.boundary_conditions = {"bottom_type": "test"}
    with pytest.raises(ValueError):
        heat.boundary_conditions = {"top_value": [1, 2] * u.W / u.m**2}
    with pytest.raises(ValueError):
        heat.boundary_conditions = {"bottom_value": [1, 2] * u.W / u.m**2}
    with pytest.raises(ValueError):
        heat.boundary_conditions = "test"


def test_distances(heat):
    heat.distances = np.r_[0:10e-9:1e-10] * u.nm
    assert_allclose(heat.distances, np.r_[0:10e-9:1e-10] * u.nm)


def test_backend(heat):
    assert heat.backend == "scipy"

    with pytest.raises(ValueError):
        heat.backend = "matlab"

    heat.backend = "numba"
    if HAS_NUMBA:
        assert heat.backend == "numba"
    else:
        # without numba being installed the backend falls back to scipy
        assert heat.backend == "scipy"

    heat.backend = "scipy"
    assert heat.backend == "scipy"


def test_get_hash_is_backend_independent(heat):
    """both backends must yield the same results, hence cached results must
    not be invalidated by a change of the backend"""
    hash_scipy = heat.get_hash(np.r_[0:10:1], 300)
    heat.backend = "numba"
    assert heat.get_hash(np.r_[0:10:1], 300) == hash_scipy
    heat.backend = "scipy"
