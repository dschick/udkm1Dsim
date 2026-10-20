#!/usr/bin/env python

import copy
import warnings

import numpy as np
import pytest

from udkm1Dsim import LLB, Magnetization, u

# fixtures


@pytest.fixture(scope="module")
def magnetization(structure, tmp_path_factory):
    return Magnetization(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def llb(structure_amorph, tmp_path_factory):
    return LLB(
        structure_amorph,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def delays():
    return np.r_[-1:10:1] * u.ps


@pytest.fixture(scope="module")
def distances(structure_amorph):
    dists, _, _ = structure_amorph.get_distances_of_layers()
    return dists


@pytest.fixture(scope="module")
def temp_map(delays, distances):
    return 50 * np.ones([len(delays), len(distances), 1])


# tests


# Magnetization


def test_magnetization_repr(magnetization):
    assert repr(magnetization).startswith("Magnetization simulation properties:")
    assert str(magnetization) == repr(magnetization)
    assert magnetization._repr_html_().startswith("<h2>Magnetization simulation</h2>")


def test_magnetization_get_hash(magnetization, delays, temp_map):
    magnetization.get_hash(delays=delays, temp_map=temp_map)


def test_magnetization_check_initial_magnetization(magnetization):
    # must fail as UnitCells are not yet supported, see #129
    with pytest.raises(TypeError):
        magnetization.check_initial_magnetization([])


# LLB


def test_llb_repr(llb):
    assert repr(llb).startswith(
        "Landau-Lifshitz-Bloch Magnetization Dynamics simulation properties:"
    )
    assert str(llb) == repr(llb)
    assert llb._repr_html_().startswith(
        "<h2>Landau-Lifshitz-Bloch Magnetization Dynamics simulation</h2>"
    )


def test_llb_check_initial_magnetization(llb, distances):
    llb.check_initial_magnetization(np.array([0, 0, 0]), distances=distances)
    llb.check_initial_magnetization(np.zeros([len(distances), 3]), distances=distances)
    with pytest.raises(ValueError):
        llb.check_initial_magnetization(np.zeros([3, len(distances)]), distances=distances)


def test_llb_get_magnetization(llb, temp_map, delays):
    llb.get_magnetization_map(delays, temp_map=temp_map)
    llb.force_recalc = False
    llb.get_magnetization_map(delays, temp_map=temp_map)


@pytest.mark.parametrize(
    "temp, below_tc",
    [(50, True), (99.9, True), (99.97, True), (100, False), (100.02, False), (150, False)],
)
def test_llb_mean_field_mag_map_at_curie_temp(llb, delays, distances, temp, below_tc):
    # all layers of the fixture have a Curie temperature of 100 K
    temp_map = temp * np.ones([len(delays), len(distances)])
    temp_map[0, :] = 50  # at least one temperature below T_C is required
    mf_mag_map = llb.get_mean_field_mag_map(temp_map)
    assert np.all(mf_mag_map[0, :] > 0)
    if below_tc:
        assert np.all(mf_mag_map[1:, :] > 0)
    else:
        assert np.all(mf_mag_map[1:, :] == 0)


def test_llb_mean_field_mag_map_above_curie_temp_raises(llb, delays, distances):
    with pytest.raises(IndexError):
        llb.get_mean_field_mag_map(150 * np.ones([len(delays), len(distances)]))


def test_llb_get_magnetization_just_below_curie_temp(llb, delays, distances):
    llb = copy.deepcopy(llb)  # do not alter the module-scoped fixture
    llb.force_recalc = True
    llb.save_data = False
    temp_map = 99.97 * np.ones([len(delays), len(distances), 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        magnetization_map = llb.get_magnetization_map(delays, temp_map=temp_map)
    assert np.all(np.isfinite(magnetization_map))
