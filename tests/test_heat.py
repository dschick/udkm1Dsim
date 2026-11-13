#!/usr/bin/env python

import numpy as np
import pytest
from pint.testing import assert_allclose

from udkm1Dsim import Dispersion, Heat, Layer, Structure, u
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


@pytest.mark.parametrize("polarization", ["s", "p"])
@pytest.mark.parametrize("theta", [90, 60, 30])
def test_multilayer_absorption_conserves_energy(tmp_path, polarization, theta):
    # absorbed fraction from the absorption profile equals 1 - R - T
    def layer(id, n, thickness=1 * u.nm):
        return Layer(
            id, id, thickness=thickness, dielectric=Dispersion.from_ref_index(n), therm_cond=1
        )

    absorber = layer("absorber", 3.5 + 4.0j)
    S = Structure("stack")
    S.add_sub_structure(absorber, 20)
    S.add_sub_structure(layer("dielectric", 1.5 + 0.01j), 50)
    S.add_sub_structure(absorber, 10)
    S.add_substrate(layer("substrate", 3.7 + 0.01j, 0 * u.nm))
    heat = Heat(S, True, cache_dir=tmp_path, save_data=False, disp_messages=False)
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "wavelength": 800 * u.nm,
        "polarization": polarization,
        "theta": theta * u.deg,
        "multilayer_absorption": True,
    }
    distances = np.linspace(0, S.interface_distances[-1].to("m").magnitude, 20001)
    dAdz, _, R, T = heat.get_multilayers_absorption_profile(distances)
    assert np.isclose(np.trapezoid(dAdz, distances), 1 - R - T, rtol=1e-4)


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
    result = heat.get_temperature_after_delta_excitation(10, 300)
    final_temp, delta_temp = heat.calc_temperature_after_delta_excitation(10, 300)
    np.testing.assert_array_equal(result.final_temp, final_temp)
    # the temperature change is the difference to the initial temperature
    assert np.max(delta_temp) > 0
    np.testing.assert_allclose(delta_temp, final_temp - 300)
    assert result.dims("delta_temp") == ("distances", "subsystems")
    distances = np.r_[0:10e-9:1e-10]
    result = heat.get_temperature_after_delta_excitation(10, 300, distances=distances)
    np.testing.assert_array_equal(result.distances, distances)


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
    heat.force_recalc = True


def test_get_diffusion_temp_map(heat):
    """the heat diffuses from the excited surface into the depth, while the
    isolating boundaries conserve the absorbed energy"""
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
    heat.boundary_conditions = {"top_type": "isolator", "bottom_type": "isolator"}
    res = heat.get_temp_map(delays, init_temp)
    temp_map = res.temp_map
    after = delays.to("ps").magnitude >= 0
    # energy per area of the layers with the constant heat capacity of the fixtures
    thicknesses = heat.S.get_layer_property_vector("structural.thickness.magnitude")
    densities = heat.S.get_layer_property_vector("structural.density.magnitude")
    energy = (temp_map[after] - 300) @ (500 * densities * thicknesses)
    assert energy[0] > 0
    # up to the tolerance of the solver and the sampling of the layers
    np.testing.assert_allclose(energy, energy[0], rtol=1e-2)
    # the excited surface cools, while the deeper layers warm up
    excited = temp_map[after][0]
    assert temp_map[-1, 0] < excited[0]
    assert np.any(temp_map[-1] > excited + 1e-3)


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
    dists, _, _ = heat.S.layer_distances
    temp_map = np.zeros((10, len(dists)))
    heat.calc_energy_map(temp_map, 0)


def test_calc_energy_flux_map(heat):
    delays = np.r_[-1:10:0.01] * u.ps
    dists, _, _ = heat.S.layer_distances

    temp_map = np.zeros((len(delays), len(dists)))
    delta_map = temp_map
    heat.calc_energy_flux_map(temp_map, delta_map, delays)


def test_get_energy_maps_of_temp_map(heat):
    """the energy and energy flux maps of the result of get_temp_map equal the
    ones of its arrays"""
    delays = np.r_[-1:5:0.1] * u.ps
    heat.heat_diffusion = False
    heat.excitation = {
        "fluence": [10] * u.mJ / u.cm**2,
        "delay_pump": [0] * u.ps,
        "pulse_width": [0] * u.ps,
    }
    temp = heat.get_temp_map(delays, 300 * u.K)
    np.testing.assert_array_equal(temp.init_temp, heat.check_initial_temperature(300))
    energy = heat.get_energy_map(temp)
    reference = heat.calc_energy_map(temp.temp_map, 300 * u.K)
    np.testing.assert_array_equal(energy.energy_map, reference)
    assert np.max(energy.energy_map) > 0
    np.testing.assert_array_equal(energy.delays, temp.delays)
    flux = heat.get_energy_flux_map(temp)
    reference = heat.calc_energy_flux_map(temp.temp_map, temp.delta_temp_map, delays)
    for i, name in enumerate(("total", "coupling", "diffusion")):
        np.testing.assert_array_equal(flux[f"{name}_energy_flux_map"], reference[..., i])
    assert flux.unit("total_energy_flux_map") == "W"
    with pytest.raises(ValueError, match="init_temp must be given"):
        heat.get_energy_map(temp.temp_map)
    with pytest.raises(ValueError, match="delays must be given"):
        heat.get_energy_flux_map(temp.temp_map)


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
    heat.distances = np.r_[0:10:0.1] * u.nm
    assert_allclose(heat.distances, np.r_[0:10:0.1] * u.nm)
    assert_allclose(heat.distances.to("m"), np.r_[0:10e-9:1e-10] * u.m)
    heat.distances = np.array([]) * u.nm


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


def test_heat_inconsistent_num_sub_systems_raises(tmp_path):
    layer = Layer("a", "a", thickness=1 * u.nm, heat_capacity=[10, 20], therm_cond=1)
    S = Structure("s")
    S.add_sub_structure(layer, 10)
    heat = Heat(S, True, cache_dir=tmp_path, save_data=False, disp_messages=False)
    with pytest.raises(ValueError, match="'therm_cond' has 1"):
        heat.get_temp_map(np.r_[0:1:0.1] * u.ps, 300 * u.K)
