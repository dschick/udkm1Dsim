#!/usr/bin/env python

import numpy as np
import pytest

from tests.conftest import absorbing_stack
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
    return np.linspace(-1, 10, 221) * u.ps


@pytest.fixture(scope="module")
def distances(structure):
    dists, _, _ = structure.layer_distances
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


def test_get_hash(structure, delays, temp_map, delta_temp_map):
    """the hash follows the maps, only_heat, and the ode_options"""
    phonon_num = PhononNum(structure, True, save_data=False, disp_messages=False)
    hash = phonon_num.get_hash(delays, temp_map, delta_temp_map)
    assert hash == phonon_num.get_hash(delays, temp_map.copy(), delta_temp_map.copy())
    assert hash != phonon_num.get_hash(delays, temp_map + 1, delta_temp_map)
    phonon_num.only_heat = True
    assert hash != phonon_num.get_hash(delays, temp_map, delta_temp_map)
    assert phonon_num._parameters()["only_heat"] is True
    phonon_num.only_heat = False
    phonon_num.ode_options["rtol"] = 1e-4
    assert hash != phonon_num.get_hash(delays, temp_map, delta_temp_map)


def test_calc_all_strains_per_unique_layer(phonon, strain_map):
    """the sorted unique strains of all layers of each unique layer"""
    assert all(
        np.array_equal(s, [0]) for s in phonon.calc_all_strains_per_unique_layer(strain_map)
    )
    positions = phonon.S.positions_per_unique_layer
    strain_map = np.zeros([3, phonon.S.num_layers])
    first, second = list(positions.values())[:2]
    strain_map[0, first[0]] = 0.02
    strain_map[1, first[-1]] = -0.01
    strain_map[2, second[0]] = 0.03
    strains = phonon.calc_all_strains_per_unique_layer(strain_map)
    assert len(strains) == len(positions)
    np.testing.assert_array_equal(strains[0], [-0.01, 0, 0.02])
    np.testing.assert_array_equal(strains[1], [0, 0.03])


def test_calc_reduced_strains_per_unique_layer(phonon):
    """N equally spaced strains between the extrema of each unique layer and
    the zero strain"""
    positions = list(phonon.S.positions_per_unique_layer.values())
    strain_map = np.zeros([2, phonon.S.num_layers])
    strain_map[0, positions[0]] = 0.01
    strain_map[1, positions[0][0]] = 0.03
    strains = phonon.calc_reduced_strains_per_unique_layer(strain_map, N=3)
    np.testing.assert_allclose(strains[0], [0, 0.015, 0.03])
    np.testing.assert_array_equal(strains[1], [0])
    strains = phonon.calc_reduced_strains_per_unique_layer(strain_map, N=[5] + [1] * 3)
    assert len(strains[0]) == 5
    with pytest.raises(ValueError, match="dimension of N"):
        phonon.calc_reduced_strains_per_unique_layer(strain_map, N=[2, 3])


def test_check_temp_maps(phonon, temp_map, delays, delta_temp_map):
    checked = phonon.check_temp_maps(temp_map, delta_temp_map, delays)
    assert checked[0] is temp_map and checked[1] is delta_temp_map
    # a single delta_temp_map of the first delay is expanded
    _, expanded = phonon.check_temp_maps(temp_map, delta_temp_map[[20]], delays)
    np.testing.assert_array_equal(expanded[0], delta_temp_map[20])
    assert np.all(expanded[1:] == 0)
    with pytest.raises(ValueError, match="dimension M x L"):
        phonon.check_temp_maps(temp_map, delta_temp_map[:, :-1], delays)
    with pytest.raises(ValueError, match="same size"):
        phonon.check_temp_maps(temp_map[:, :-1], delta_temp_map, delays)


def test_calc_sticks_from_temp_map(phonon, temp_map, delta_temp_map):
    """the sticks of a constant linear thermal expansion alpha are
    d (exp(alpha dT) - 1) of the layer thickness d"""
    sticks, sticks_sub_systems = phonon.calc_sticks_from_temp_map(temp_map, delta_temp_map)
    thicknesses = phonon.S.get_layer_property_vector("structural.thickness.magnitude")
    np.testing.assert_allclose(
        sticks[-1, :20], thicknesses[:20] * (np.exp(1e-5 * 100) - 1), rtol=1e-12
    )
    assert np.all(sticks[-1, 20:] == 0)
    assert np.all(sticks[0] == 0)
    np.testing.assert_array_equal(sticks_sub_systems[..., 0], sticks)


@pytest.mark.parametrize("phonon_class", [PhononNum, PhononAna])
def test_only_heat_is_the_quasi_static_strain(
    phonon_class, structure_freestanding, delays, tmp_path
):
    """without coherent phonons, the strain follows the temperature by the
    linear thermal expansion as exp(alpha dT) - 1"""
    phonon = phonon_class(
        structure_freestanding,
        True,
        cache_dir=tmp_path,
        save_data=False,
        disp_messages=False,
        only_heat=True,
    )
    temp_map, delta_temp_map = _heating(structure_freestanding, delays)
    strain_map = phonon.calc_strain_map(delays, temp_map, delta_temp_map)[0]
    t = delays.to("s").magnitude
    np.testing.assert_allclose(strain_map[t < 0], 0)
    np.testing.assert_allclose(strain_map[-1, :10], np.exp(1e-5 * 150) - 1, rtol=1e-12)
    np.testing.assert_allclose(strain_map[-1, 10:20], np.exp(1e-5 * 100) - 1, rtol=1e-12)
    np.testing.assert_allclose(strain_map[-1, 20:], 0)


# PhononNum


def test_phonon_num_default_tolerances(structure):
    """the absolute tolerance is set for the shifts of the layers of a few
    picometers"""
    pnum = PhononNum(structure, True, save_data=False)
    assert pnum.ode_options["atol"] == 1e-12
    assert pnum.ode_options["rtol"] == 1e-3


def test_phonon_num_repr(phonon_num):
    assert repr(phonon_num).startswith("Numerical Phonon simulation properties:")
    assert str(phonon_num) == repr(phonon_num)
    assert phonon_num._repr_html_().startswith("<h2>Numerical Phonon simulation</h2>")


def test_phonon_num_get_strain_map(phonon_num, delays, temp_map, delta_temp_map):
    result = phonon_num.get_strain_map(delays, temp_map, delta_temp_map)
    phonon_num.force_recalc = False
    try:
        cached = phonon_num.get_strain_map(delays, temp_map, delta_temp_map)
    finally:
        phonon_num.force_recalc = True
    np.testing.assert_array_equal(cached.strain_map, result.strain_map)
    np.testing.assert_array_equal(
        result.strain_map, phonon_num.calc_strain_map(delays, temp_map, delta_temp_map)[0]
    )


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


# PhononAna


def test_phonon_ana_repr(phonon_ana):
    assert repr(phonon_ana).startswith("Analytical Phonon simulation properties:")
    assert str(phonon_ana) == repr(phonon_ana)
    assert phonon_ana._repr_html_().startswith("<h2>Analytical Phonon simulation</h2>")


def test_phonon_ana_get_strain_map(structure_freestanding, delays, tmp_path):
    phonon_ana = PhononAna(
        structure_freestanding, True, cache_dir=tmp_path, save_data=True, disp_messages=False
    )
    temp_map, delta_temp_map = _heating(structure_freestanding, delays)
    result = phonon_ana.get_strain_map(delays, temp_map, delta_temp_map)
    phonon_ana.force_recalc = False
    try:
        cached = phonon_ana.get_strain_map(delays, temp_map, delta_temp_map)
    finally:
        phonon_ana.force_recalc = True
    for name in ("strain_map", "coefficients_A", "coefficients_B"):
        np.testing.assert_array_equal(cached[name], result[name])


def test_phonon_ana_get_energy_per_eigenmode(structure_freestanding, delays, tmp_path):
    """the energies of the eigenmodes are constant without a change of the
    temperature, i.e. after the last heating at 3 ps"""
    phonon_ana = PhononAna(
        structure_freestanding, True, cache_dir=tmp_path, save_data=True, disp_messages=False
    )
    strain = phonon_ana.get_strain_map(delays, *_heating(structure_freestanding, delays))
    omega, E = phonon_ana.calc_energy_per_eigenmode(strain.coefficients_A, strain.coefficients_B)
    t = delays.to("s").magnitude
    after = t > 3.1e-12
    np.testing.assert_allclose(E[after], np.broadcast_to(E[after][0], E[after].shape), rtol=1e-8)
    assert np.all(E[t < 0] == 0)
    # the second heating changes the energies
    before = (t > 0.1e-12) & (t < 2.9e-12)
    assert not np.allclose(E[after][0], E[before][0], rtol=1e-3, atol=0)
    result = phonon_ana.get_energy_per_eigenmode(strain)
    np.testing.assert_array_equal(result.energy_per_eigenmode, E)
    np.testing.assert_array_equal(result.eigenfrequencies, omega)
    np.testing.assert_array_equal(result.delays, strain.delays)
    assert result.E is result.energy_per_eigenmode and result.omega is result.eigenfrequencies
    assert strain.A is strain.coefficients_A and strain.B is strain.coefficients_B


@pytest.mark.parametrize("phonon_class", [PhononNum, PhononAna])
def test_phonon_without_density_raises(tmp_path, phonon_class):
    """instead of divisions by zero masses, see #125"""
    S = absorbing_stack()
    phonon = phonon_class(S, True, cache_dir=tmp_path, save_data=False, disp_messages=False)
    delays = np.r_[0:1:0.1] * u.ps
    temp_map = 300 * np.ones([len(delays), S.num_layers])
    with pytest.raises(ValueError, match="positive density"):
        phonon.get_strain_map(delays, temp_map, np.zeros_like(temp_map))


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
    temp_map = 300 * np.ones([len(delays), S.num_layers, 2])
    with pytest.raises(ValueError, match="'lin_therm_exp' has 1"):
        phonon.get_strain_map(delays, temp_map, np.zeros_like(temp_map))


def test_check_temp_maps_expands_single_delta_temp_map_of_sub_systems(tmp_path):
    """a temperature difference map of a single delay is expanded to all M
    delays for K > 1 subsystems, also for M != K"""
    layer = Layer(
        "a",
        "a",
        thickness=1 * u.nm,
        density=1000 * u.kg / u.m**3,
        sound_vel=5 * u.nm / u.ps,
        heat_capacity=[10, 20],
        lin_therm_exp=[1e-5, 1e-5],
    )
    S = Structure("s")
    S.add_sub_structure(layer, 10)
    phonon = Phonon(S, True, cache_dir=tmp_path, save_data=False, disp_messages=False)
    delays = np.r_[0:1:0.1] * u.ps
    temp_map = 300 * np.ones([len(delays), S.num_layers, 2])
    delta = np.ones([1, S.num_layers, 2])
    _, delta_temp_map = phonon.check_temp_maps(temp_map, delta, delays)
    assert delta_temp_map.shape == temp_map.shape
    assert np.all(delta_temp_map[0] == 1) and np.all(delta_temp_map[1:] == 0)


# PhononAna vs. PhononNum
#
# The analytical solution is exact for the harmonic linear chain without
# damping and serves as reference for the numerical solvers.


def _heating(S, delays, two_steps=True):
    """step-like heating of the top 20 layers at t = 0 and, optionally, of
    the top 10 layers at t = 3 ps to test the re-initialization of the
    analytical solution at each temperature change"""
    t = delays.to("s").magnitude
    temp_map = 300 * np.ones((len(t), S.num_layers))
    temp_map[np.ix_(t >= 0, np.r_[0:20])] += 100
    if two_steps:
        temp_map[np.ix_(t >= 3e-12, np.r_[0:10])] += 50
    delta_temp_map = np.zeros_like(temp_map)
    delta_temp_map[1:] = np.diff(temp_map, axis=0)
    return temp_map, delta_temp_map


# the freestanding layer and the substrate of the crystalline structure with
# its acoustic mismatch are thin enough to reflect the strain wave at their
# backside within the delays
@pytest.fixture(scope="module", params=["structure_freestanding", "structure_crystalline"])
def reference_structure(request):
    """the reference structures without phonon damping, which the analytical
    solution neglects"""
    S = request.getfixturevalue(request.param)
    layers = S.unique_layers[1]
    dampings = [layer.phonon_damping for layer in layers]
    for layer in layers:
        layer.phonon_damping = 0 * u.kg / u.s
    yield S
    for layer, damping in zip(layers, dampings):
        layer.phonon_damping = damping


@pytest.fixture(scope="module")
def reference_strain_map(reference_structure, delays, tmp_path_factory):
    phonon_ana = PhononAna(
        reference_structure,
        True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=False,
        disp_messages=False,
    )
    strain_map, _, _ = phonon_ana.calc_strain_map(delays, *_heating(reference_structure, delays))
    return strain_map


def test_phonon_ana_free_backside(structure_freestanding, delays, tmp_path):
    """the strain wave is reflected at the free backside of a freestanding
    layer with an inverted sign, closing #186"""
    phonon_ana = PhononAna(
        structure_freestanding, True, cache_dir=tmp_path, save_data=False, disp_messages=False
    )
    strain_map, _, _ = phonon_ana.calc_strain_map(
        delays, *_heating(structure_freestanding, delays, two_steps=False)
    )
    eps = np.exp(1e-5 * 100) - 1  # quasi-static strain of the 20 heated layers
    t = delays.to("s").magnitude
    # the total expansion of the 1 nm layers is 20 eps, before the strain wave
    # reaches the backside of the 50 nm layer at (50 - 20) nm / 6 nm/ps = 5 ps
    i = np.argmin(np.abs(t - 4e-12))
    assert np.sum(strain_map[i]) == pytest.approx(20 * eps, rel=1e-2)
    # its compressive half is reflected as expansion at the free backside,
    # while a clamped backside would reflect it as compression, and the heated
    # layers relax to the quasi-static strain
    i = np.argmin(np.abs(t - 8e-12))
    assert np.mean(strain_map[i, 40:49]) > 0.5 * eps
    assert np.sum(strain_map[i]) > 35 * eps
    assert np.mean(strain_map[i, :10]) == pytest.approx(eps, rel=1e-2)


def test_phonon_ana_zero_mode(reference_structure, tmp_path):
    """the free chain has exactly one zero-frequency mode for its translation"""
    phonon_ana = PhononAna(
        reference_structure, True, cache_dir=tmp_path, save_data=False, disp_messages=False
    )
    res = phonon_ana.get_eigenproblem()
    Xi, omega = res.eigenvectors, res.eigenfrequencies
    np.testing.assert_array_equal(phonon_ana.calc_eigenproblem()[1], omega)
    assert res.dims("eigenvectors") == ("distances", "eigenmodes")
    assert np.all(np.isreal(omega)) and np.all(omega >= 0)
    assert np.sum(omega == 0) == 1
    # the eigenvectors are orthonormal with respect to the masses
    masses = reference_structure.get_layer_property_vector("structural.mass_unit_area.magnitude")
    np.testing.assert_allclose(Xi.T @ (masses[:, None] * Xi), np.eye(len(masses)), atol=1e-10)


@pytest.mark.parametrize(
    "backend",
    [
        "scipy",
        pytest.param(
            "numba", marks=pytest.mark.skipif(not HAS_NUMBA, reason="numba is not installed")
        ),
    ],
)
def test_phonon_num_matches_ana(
    reference_structure, reference_strain_map, delays, backend, tmp_path
):
    """the numerical solvers reproduce the analytical solution"""
    phonon_num = PhononNum(
        reference_structure,
        True,
        cache_dir=tmp_path,
        save_data=False,
        disp_messages=False,
        backend=backend,
    )
    phonon_num.ode_options.update({"rtol": 1e-8, "atol": 1e-20})
    strain_map, _, _ = phonon_num.calc_strain_map(delays, *_heating(reference_structure, delays))

    max_strain = np.max(np.abs(reference_strain_map))
    assert max_strain > 1e-4
    np.testing.assert_allclose(strain_map, reference_strain_map, rtol=0, atol=1e-5 * max_strain)


def test_phonon_num_negative_damping_raises(structure_freestanding, delays, tmp_path):
    layer = structure_freestanding.get_layer_handle(0)
    damping = layer.phonon_damping
    layer.phonon_damping = -1e-12 * u.kg / u.s
    phonon_num = PhononNum(
        structure_freestanding, True, cache_dir=tmp_path, save_data=False, disp_messages=False
    )
    try:
        with pytest.raises(ValueError, match="must not be negative"):
            phonon_num.calc_strain_map(delays, *_heating(structure_freestanding, delays))
    finally:
        layer.phonon_damping = damping
