#!/usr/bin/env python

import copy

import numpy as np
import pint
import pytest
from pint.testing import assert_allclose

from udkm1Dsim import Structure

u = pint.get_application_registry()
u.formatter.default_format = ".4g~P"


# tests


def test_structure_str(structure):
    structure.__str__()


def test_thickness(structure):
    assert_allclose(structure.thickness, 150 * u.nm, rtol=1e-2)


# test_visualize() fails on certain virtual machine due to TCL error


def test_structure_get_hash(structure):
    structure.get_hash()


@pytest.mark.parametrize(
    "fixture_name", ["layer", "vacuum", "amorphous_layer", "unit_cell", "structure"]
)
def test_structure_add_sub_structure(request, fixture_name, structure):
    substructure = request.getfixturevalue(fixture_name)
    struc_copy = copy.deepcopy(structure)
    if isinstance(substructure, Structure):
        with pytest.warns(UserWarning):
            struc_copy.add_sub_structure(substructure, 10)
    else:
        struc_copy.add_sub_structure(substructure, 10)


@pytest.mark.parametrize(
    "fixture_name", ["layer", "vacuum", "amorphous_layer", "unit_cell", "structure"]
)
def test_structure_add_superstrate(request, fixture_name, structure):
    substructure = request.getfixturevalue(fixture_name)
    struc_copy = copy.deepcopy(structure)
    if isinstance(substructure, Structure):
        with pytest.raises(TypeError):
            struc_copy.add_superstrate(substructure)
    else:
        struc_copy.add_superstrate(substructure)


@pytest.mark.parametrize(
    "fixture_name", ["layer", "vacuum", "amorphous_layer", "unit_cell", "structure"]
)
def test_structure_add_substrate(request, fixture_name, structure):
    substructure = request.getfixturevalue(fixture_name)
    struc_copy = copy.deepcopy(structure)
    if isinstance(substructure, Structure):
        with pytest.raises(TypeError):
            struc_copy.add_substrate(substructure)
    else:
        struc_copy.add_substrate(substructure)


def test_get_number_of_sub_structures(structure):
    assert structure.get_number_of_sub_structures() == 5


def test_get_number_of_layers(structure):
    assert structure.get_number_of_layers() == 280


def test_get_number_of_unique_layers(structure):
    assert structure.get_number_of_unique_layers() == 4


def test_get_unique_layers(structure):
    layer_ids, _ = structure.get_unique_layers()
    assert layer_ids == ["amorphous_layer_Fe", "amorphous_layer_O", "unit_cell_Fe", "unit_cell_O"]


def test_get_layer_vectors(structure):
    structure.get_layer_vectors()


def test_get_all_positions_per_unique_layer(structure):
    pos = structure.get_all_positions_per_unique_layer()
    assert np.allclose(pos["amorphous_layer_Fe"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_get_distances_of_layers(structure):
    d_start, _, _ = structure.get_distances_of_layers()
    assert_allclose(np.sum(d_start), 2.22e-05 * u.m, rtol=1e-2)


def test_get_distances_of_interfaces(structure):
    assert_allclose(np.sum(structure.get_distances_of_interfaces()), 3.68e-6 * u.m, rtol=1e-2)


def test_interp_distance_at_interfaces(structure):
    structure.interp_distance_at_interfaces(N=11)


def test_get_layer_property_vector(structure):
    structure.get_layer_property_vector("density")
    structure.get_layer_property_vector("_density")
    structure.get_layer_property_vector("heat_capacity")
    structure.get_layer_property_vector("spring_const")


def test_get_layer_property_vector_mag_moment_in_si(structure):
    structure = copy.deepcopy(structure)  # do not alter the module-scoped fixtures
    layer = structure.get_layer_handle(0)
    layer.mag_moment = 2 * u.bohr_magneton
    for name in ["_mag_moment", "magnetic.mag_moment.magnitude"]:
        mag_moments = structure.get_layer_property_vector(name)
        assert np.isclose(mag_moments[0], 1.855e-23, rtol=1e-3)


@pytest.mark.parametrize(
    "types, prop, value",
    [
        ("heat", "heat_capacity", 500 * u.J / u.kg / u.K),
        ("heat", "thickness", 3 * u.nm),
        ("phonon", "phonon_damping", 1e-12 * u.kg / u.s),
        ("xray", "roughness", 1 * u.nm),
        ("optical", "opt_ref_index", 1.5 + 0.1j),
        ("magnetic", "curie_temp", 500 * u.K),
        (["xray", "magnetic"], "mag_moment", 2 * u.bohr_magneton),
    ],
)
def test_structure_get_hash_depends_on_properties(structure, types, prop, value):
    structure = copy.deepcopy(structure)  # do not alter the module-scoped fixtures
    old_hash = structure.get_hash(types=types)
    assert structure.get_hash(types=types) == old_hash
    setattr(structure.get_layer_handle(0), prop, value)
    assert structure.get_hash(types=types) != old_hash


def test_get_layer_property_vector_padding(structure):
    """layers without higher order spring constants get zeros for them"""
    _, layers = structure.get_unique_layers()
    anharmonic = layers[2]
    anharmonic_pos = structure.get_all_positions_per_unique_layer()[anharmonic.id]
    harmonic_pos = np.setdiff1d(np.r_[0 : structure.get_number_of_layers()], anharmonic_pos)
    anharmonic.set_ho_spring_constants([-7e11, 1e20])
    try:
        spring_consts = structure.get_layer_property_vector("_spring_const")
    finally:
        # reset to the harmonic spring constant for the following tests
        anharmonic.set_ho_spring_constants([])

    assert spring_consts.shape == (structure.get_number_of_layers(), 3)
    assert np.all(spring_consts[harmonic_pos, 1:] == 0)
    assert np.allclose(spring_consts[anharmonic_pos, 0], anharmonic.spring_const[0].magnitude)
    assert np.allclose(spring_consts[anharmonic_pos, 1:], [-7e11, 1e20])


def test_get_layer_handle(structure):
    structure.get_layer_handle(0)


def test_reverse(structure):
    structure.reverse()
