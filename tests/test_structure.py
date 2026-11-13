#!/usr/bin/env python

import copy

import numpy as np
import pint
import pytest
from pint.testing import assert_allclose

from udkm1Dsim import Dispersion, Layer, Structure

u = pint.get_application_registry()
u.formatter.default_format = ".4g~P"


# tests


def test_structure_repr(structure):
    assert repr(structure).startswith("Structure properties:")
    assert str(structure) == repr(structure)
    html = structure._repr_html_()
    assert html.startswith("<h2>Structure: structure mixed</h2>")
    # the header is styled as the headers of the other tables
    assert "<thead><tr><th>times</th>" in html


def test_thickness(structure):
    assert_allclose(structure.thickness, 150 * u.nm, rtol=1e-2)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_structure_visualize(structure, agg_backend):
    structure.visualize(show=False)
    structure.visualize(unit="um", block=False)


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


def test_num_sub_structures(structure):
    assert structure.num_sub_structures == 5


def test_num_layers(structure):
    assert structure.num_layers == 280


def test_num_unique_layers(structure):
    assert structure.num_unique_layers == 4


def test_unique_layers(structure):
    layer_ids, _ = structure.unique_layers
    assert layer_ids == ["amorphous_layer_Fe", "amorphous_layer_O", "unit_cell_Fe", "unit_cell_O"]


def test_layer_vectors(structure, unit_cell_iron, unit_cell_oxygen):
    indices, layer_ids, handles = structure.layer_vectors
    assert len(indices) == len(layer_ids) == len(handles) == structure.num_layers
    # 10 + 10 + 20 layers before the 20 repetitions of the nested double layer
    assert np.array_equal(indices[:40], [0] * 10 + [1] * 10 + [2] * 20)
    assert np.array_equal(indices[40:], ([2] * 5 + [3] * 7) * 20)
    assert layer_ids[40:52] == ["unit_cell_Fe"] * 5 + ["unit_cell_O"] * 7
    assert handles[40] is unit_cell_iron
    assert handles[-1] is unit_cell_oxygen
    assert [structure.get_layer_handle(i) for i in [0, 45, -1]] == [
        handles[0],
        handles[45],
        handles[-1],
    ]


def test_positions_per_unique_layer(structure):
    pos = structure.positions_per_unique_layer
    assert np.allclose(pos["amorphous_layer_Fe"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_layer_distances(structure):
    d_start, _, _ = structure.layer_distances
    assert_allclose(np.sum(d_start), 2.22e-05 * u.m, rtol=1e-2)


def test_layer_distances_are_consistent(structure):
    d_start, d_end, d_mid = structure.layer_distances
    assert_allclose(d_end[-1], structure.thickness)
    assert_allclose(d_start[1:], d_end[:-1])
    assert_allclose(d_mid, (d_start + d_end) / 2)
    np.testing.assert_allclose(structure._layer_distances()[2], d_mid.to("m").magnitude)


def test_interface_distances(structure):
    interfaces = structure.interface_distances
    assert_allclose(np.sum(interfaces), 3.68e-6 * u.m, rtol=1e-2)
    # the surface, the 2 interfaces of the amorphous layers, the 20 Fe/O and
    # 19 O/Fe interfaces of the double layers, whose first one continues the
    # Fe unit cells, and the bottom
    assert len(interfaces) == 1 + 2 + 20 + 19 + 1
    assert interfaces[0] == 0 * u.m
    assert_allclose(interfaces[-1], structure.thickness)
    np.testing.assert_allclose(structure._interface_distances(), interfaces.to("m").magnitude)


@pytest.mark.parametrize(
    "prop",
    [
        "num_sub_structures",
        "num_layers",
        "num_unique_layers",
        "unique_layers",
        "layer_vectors",
        "positions_per_unique_layer",
        "layer_distances",
        "interface_distances",
        "thickness",
    ],
)
def test_structure_properties_are_read_only(structure, prop):
    with pytest.raises(AttributeError):
        setattr(structure, prop, 1)


def test_interp_distance_at_interfaces(structure):
    distances, original = structure.interp_distance_at_interfaces(N=11)
    _, _, d_mid = structure.layer_distances
    # the distances of the layers are kept
    assert_allclose(distances[original], d_mid)
    # an even number of points is increased to the next odd number
    even, _ = structure.interp_distance_at_interfaces(N=10, units=False)
    np.testing.assert_allclose(even, distances.to("m").magnitude)


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
        ("heat", "heat_capacity", 700 * u.J / u.kg / u.K),  # differs from the fixtures
        ("heat", "thickness", 3 * u.nm),
        ("phonon", "phonon_damping", 2e-12 * u.kg / u.s),
        ("xray", "roughness", 1 * u.nm),
        ("optical", "dielectric", Dispersion.from_ref_index(1.5 + 0.1j)),
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
    _, layers = structure.unique_layers
    anharmonic = layers[2]
    anharmonic_pos = structure.positions_per_unique_layer[anharmonic.id]
    harmonic_pos = np.setdiff1d(np.r_[0 : structure.num_layers], anharmonic_pos)
    anharmonic.set_ho_spring_constants([-7e11, 1e20])
    try:
        spring_consts = structure.get_layer_property_vector("_spring_const")
    finally:
        # reset to the harmonic spring constant for the following tests
        anharmonic.set_ho_spring_constants([])

    assert spring_consts.shape == (structure.num_layers, 3)
    assert np.all(spring_consts[harmonic_pos, 1:] == 0)
    assert np.allclose(spring_consts[anharmonic_pos, 0], anharmonic.spring_const[0].magnitude)
    assert np.allclose(spring_consts[anharmonic_pos, 1:], [-7e11, 1e20])


def test_get_layer_handle(structure):
    structure.get_layer_handle(0)


def test_reverse(structure):
    structure.reverse()


def test_structure_num_sub_systems_follows_layers():
    layer = Layer("a", "a", thickness=1 * u.nm, heat_capacity=10, therm_cond=1)
    S = Structure("s")
    assert S.num_sub_systems == 1  # empty structure
    S.add_sub_structure(layer, 2)
    assert S.num_sub_systems == 1
    # changes of the layers after adding them to the structure are considered
    layer.heat_capacity = [10, 20]
    assert S.num_sub_systems == 2
    with pytest.raises(AttributeError):
        S.num_sub_systems = 2


def test_structure_check_num_sub_systems():
    a = Layer("a", "a", thickness=1 * u.nm, heat_capacity=[10, 20], therm_cond=[1, 2])
    b = Layer("b", "b", thickness=1 * u.nm, heat_capacity=[10, 20], therm_cond=1)
    S = Structure("s")
    S.add_sub_structure(a, 2)
    D = Structure("double layer")  # nested structures are checked, too
    D.add_sub_structure(b, 2)
    S.add_sub_structure(D, 2)
    assert S.num_sub_systems == 2
    S.check_num_sub_systems("heat_capacity")
    with pytest.raises(ValueError, match="layer 'b': 'therm_cond' has 1"):
        S.check_num_sub_systems("heat_capacity", "therm_cond")


def test_get_hash_does_not_depend_on_symbols_and_labels(structure):
    S = copy.deepcopy(structure)  # do not alter the module-scoped fixture
    hashes = [S.get_hash(types="heat"), S.get_hash(types="phonon")]
    for layer in S.unique_layers[1]:
        layer.thermal.heat_capacity.symbol = "C"
        layer.structural.thickness.symbol = "t"
        layer.thermal.sub_system_labels = ["x"]
    assert [S.get_hash(types="heat"), S.get_hash(types="phonon")] == hashes
