#!/usr/bin/env python

import numpy as np
import pytest

from udkm1Dsim import Layer, Phonon, PhononAna, PhononNum, Structure, u
from udkm1Dsim.simulations.solvers import HAS_NUMBA

# fixtures


@pytest.fixture(scope="module")
def phonon(structure, tmp_path_factory):
    return Phonon(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def phonon_num(structure, tmp_path_factory):
    return PhononNum(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def phonon_ana(structure, tmp_path_factory):
    return PhononAna(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def delays():
    return np.r_[-1:10:0.01] * u.ps


@pytest.fixture(scope="module")
def distances(structure):
    dists, _, _ = structure.get_distances_of_layers()
    return dists


@pytest.fixture(scope="module")
def temp_map(delays, distances):
    # 100 K step-like heating of the top 20 layers at t = 0
    t = delays.to("s").magnitude
    temp_map = 300 * np.ones((len(delays), len(distances)))
    temp_map[np.ix_(t >= 0, np.r_[0:20])] += 100
    return temp_map


@pytest.fixture(scope="module")
def delta_temp_map(temp_map):
    delta_temp_map = np.zeros_like(temp_map)
    delta_temp_map[1:] = np.diff(temp_map, axis=0)
    return delta_temp_map


@pytest.fixture(scope="module")
def strain_map(delays, distances):
    return np.zeros((len(delays), len(distances)))


# tests


# Phonon


def test_phonon_repr(phonon):
    assert repr(phonon).startswith("Phonon simulation properties:")
    assert str(phonon) == repr(phonon)
    assert phonon._repr_html_().startswith("<h2>Phonon simulation</h2>")


def test_get_hash(phonon, delays, distances, temp_map, delta_temp_map):
    phonon.get_hash(delays, temp_map, delta_temp_map)


def test_get_all_strains_per_unique_layer(phonon, strain_map):
    assert np.allclose(phonon.get_all_strains_per_unique_layer(strain_map), [])


def test_get_reduced_strains_per_unique_layer(phonon, strain_map):
    assert np.allclose(phonon.get_reduced_strains_per_unique_layer(strain_map), [])


def test_check_temp_maps(phonon, temp_map, delays, delta_temp_map):
    phonon.check_temp_maps(temp_map, delta_temp_map, delays)


def test_calc_sticks_from_temp_map(phonon, temp_map, delta_temp_map):
    phonon.calc_sticks_from_temp_map(temp_map, delta_temp_map)


# PhononNum


def test_phonon_num_repr(phonon_num):
    assert repr(phonon_num).startswith("Numerical Phonon simulation properties:")
    assert str(phonon_num) == repr(phonon_num)
    assert phonon_num._repr_html_().startswith("<h2>Numerical Phonon simulation</h2>")


def test_phonon_num_get_strain_map(phonon_num, delays, temp_map, delta_temp_map):
    phonon_num.get_strain_map(delays, temp_map, delta_temp_map)
    phonon_num.force_recalc = False
    phonon_num.get_strain_map(delays, temp_map, delta_temp_map)
    phonon_num.force_recalc = True


def test_phonon_num_backend(phonon_num):
    assert phonon_num.backend == "scipy"

    with pytest.raises(ValueError):
        phonon_num.backend = "matlab"

    phonon_num.backend = "numba"
    if HAS_NUMBA:
        assert phonon_num.backend == "numba"
    else:
        # without numba being installed the backend falls back to scipy
        assert phonon_num.backend == "scipy"

    phonon_num.backend = "scipy"
    assert phonon_num.backend == "scipy"


def test_phonon_num_get_hash_is_backend_independent(phonon_num, delays, temp_map, delta_temp_map):
    """both backends must yield the same results, hence cached results must
    not be invalidated by a change of the backend"""
    hash_scipy = phonon_num.get_hash(delays, temp_map, delta_temp_map)
    phonon_num.backend = "numba"
    assert phonon_num.get_hash(delays, temp_map, delta_temp_map) == hash_scipy
    phonon_num.backend = "scipy"


@pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")
def test_phonon_num_backends_agree(phonon_num, delays, temp_map, delta_temp_map):
    """the step-like heating of the top 20 layers launches the same strain
    wave with both backends"""
    ode_options = dict(phonon_num.ode_options)
    phonon_num.ode_options.update({"rtol": 1e-8, "atol": 1e-12})
    strain_maps = []
    try:
        for backend in ["scipy", "numba"]:
            phonon_num.backend = backend
            # calc_strain_map, since the cached results do not depend on the backend
            strain_maps.append(phonon_num.calc_strain_map(delays, temp_map, delta_temp_map)[0])
    finally:
        phonon_num.backend = "scipy"
        phonon_num.ode_options = ode_options

    assert np.all(np.isfinite(strain_maps[1]))
    # the heated layers expand by about 1e-3 and launch a strain wave
    assert np.max(strain_maps[0]) > 5e-4
    assert np.max(np.abs(strain_maps[0][:, 20:])) > 1e-4
    np.testing.assert_allclose(strain_maps[1], strain_maps[0], atol=1e-6)


# PhononAna


def test_phonon_ana_repr(phonon_ana):
    assert repr(phonon_ana).startswith("Analytical Phonon simulation properties:")
    assert str(phonon_ana) == repr(phonon_ana)
    assert phonon_ana._repr_html_().startswith("<h2>Analytical Phonon simulation</h2>")


def test_phonon_ana_get_strain_map(phonon_ana, delays, temp_map, delta_temp_map):
    phonon_ana.get_strain_map(delays, temp_map, delta_temp_map)
    phonon_ana.force_recalc = False
    phonon_ana.get_strain_map(delays, temp_map, delta_temp_map)


def test_phonon_ana_get_energy_per_eigenmode(phonon_ana, delays, temp_map, delta_temp_map):
    _, A, B = phonon_ana.get_strain_map(delays, temp_map, delta_temp_map)
    phonon_ana.get_energy_per_eigenmode(A, B)


@pytest.mark.parametrize("phonon_class", [PhononNum, PhononAna])
def test_phonon_inconsistent_num_sub_systems_raises(tmp_path, phonon_class):
    layer = Layer(
        "a",
        "a",
        thickness=1 * u.nm,
        density=1000 * u.kg / u.m**3,
        sound_vel=5 * u.nm / u.ps,
        heat_capacity=[10, 20],
        lin_therm_exp=1e-5,
    )
    S = Structure("s")
    S.add_sub_structure(layer, 10)
    phonon = phonon_class(S, True, cache_dir=tmp_path, save_data=False, disp_messages=False)
    delays = np.r_[0:1:0.1] * u.ps
    temp_map = 300 * np.ones([len(delays), S.get_number_of_layers(), 2])
    with pytest.raises(ValueError, match="'lin_therm_exp' has 1"):
        phonon.get_strain_map(delays, temp_map, np.zeros_like(temp_map))
