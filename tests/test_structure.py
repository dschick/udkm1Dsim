#!/usr/bin/env python

import copy

import numpy as np
import pytest
from pint.testing import assert_allclose

from udkm1Dsim import Dispersion, Layer, Structure, u

# tests


def test_structure_repr(structure):
    assert repr(structure).startswith("Structure properties:")
    assert str(structure) == repr(structure)
    html = structure._repr_html_()
    assert html.startswith("<h2>Structure: structure mixed</h2>")
    # the header is styled as the headers of the other tables
    assert "<thead><tr><th>times</th>" in html


def test_thickness(structure):
    # 20 amorphous layers of 1 nm and 20 + 20 x 12 unit cells of 0.5 nm
    assert_allclose(structure.thickness, 150 * u.nm, rtol=1e-12)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_structure_visualize(structure, agg_backend):
    structure.visualize(show=False)
    structure.visualize(unit="um", block=False)
    structure.visualize(show=False, superstrate_substrate=False)


def test_structure_get_hash(structure):
    hash = structure.get_hash()
    assert isinstance(hash, str) and len(hash) == 32
    assert copy.deepcopy(structure).get_hash() == hash
    assert structure.get_hash(types="heat") != structure.get_hash(types="phonon")


@pytest.mark.parametrize(
    "fixture_name", ["layer", "vacuum", "amorphous_layer", "unit_cell", "structure"]
)
def test_structure_add_sub_structure(request, fixture_name, structure):
    substructure = request.getfixturevalue(fixture_name)
    struc_copy = copy.deepcopy(structure)
    if isinstance(substructure, Structure):
        with pytest.warns(UserWarning, match="substrate of the sub_structure is ignored"):
            struc_copy.add_sub_structure(substructure, 10)
        added = 10 * substructure.num_layers
    else:
        struc_copy.add_sub_structure(substructure, 10)
        added = 10
    assert struc_copy.num_layers == structure.num_layers + added
    assert struc_copy.sub_structures[-1] == [substructure, 10]
    with pytest.raises(TypeError, match="no possible sub structure"):
        struc_copy.add_sub_structure("layer", 10)


@pytest.mark.parametrize(
    "fixture_name", ["layer", "vacuum", "amorphous_layer", "unit_cell", "structure"]
)
def test_structure_add_superstrate(request, fixture_name, structure):
    substructure = request.getfixturevalue(fixture_name)
    struc_copy = copy.deepcopy(structure)
    if isinstance(substructure, Structure):
        with pytest.raises(TypeError, match="no possible superstrate"):
            struc_copy.add_superstrate(substructure)
    else:
        struc_copy.add_superstrate(substructure)
        assert struc_copy.superstrate == [substructure, 1]
        # the superstrate is no layer of the structure
        assert struc_copy.num_layers == structure.num_layers


@pytest.mark.parametrize(
    "fixture_name", ["layer", "vacuum", "amorphous_layer", "unit_cell", "structure"]
)
def test_structure_add_substrate(request, fixture_name, structure):
    substructure = request.getfixturevalue(fixture_name)
    struc_copy = copy.deepcopy(structure)
    if isinstance(substructure, Structure):
        with pytest.raises(TypeError, match="no possible substrate"):
            struc_copy.add_substrate(substructure)
    else:
        struc_copy.add_substrate(substructure, 3)
        assert struc_copy.substrate == [substructure, 3]
        assert struc_copy.num_layers == structure.num_layers


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


def test_layer_tree(
    structure, amorphous_layer_iron, amorphous_layer_oxygen, unit_cell_iron, unit_cell_oxygen
):
    """the nodes of the sub-structures with their positions in the vectors of
    all sub-structures and of all layers"""
    tree = structure.layer_tree
    assert [node.sub_structure for node in tree[:3]] == [
        amorphous_layer_iron,
        amorphous_layer_oxygen,
        unit_cell_iron,
    ]
    assert [node.repetitions for node in tree] == [10, 10, 20, 20]
    assert [node.layer_indices for node in tree] == [
        slice(0, 10),
        slice(10, 20),
        slice(20, 40),
        slice(40, 280),
    ]
    assert [node.sub_structure_indices for node in tree] == [
        slice(0, 1),
        slice(1, 2),
        slice(2, 3),
        slice(3, 5),
    ]
    assert [node.is_layer for node in tree] == [True, True, True, False]
    assert [node.period for node in tree] == [1, 1, 1, 12]
    assert all(node.children == () for node in tree[:3])
    # the children of the first repetition of the double layer
    fe, ox = tree[3].children
    assert (fe.sub_structure, ox.sub_structure) == (unit_cell_iron, unit_cell_oxygen)
    assert (fe.layer_indices, ox.layer_indices) == (slice(40, 45), slice(45, 52))
    assert (fe.sub_structure_indices, ox.sub_structure_indices) == (slice(3, 4), slice(4, 5))
    # the layers of each node are its layers in the layer vectors
    handles = structure.layer_vectors[2]
    for node in tree[:3] + [fe, ox]:
        assert all(h is node.sub_structure for h in handles[node.layer_indices][: node.period])


def test_layer_tree_of_shared_sub_structure(unit_cell_iron, unit_cell_oxygen):
    """the same nested structure at two positions has two nodes with their
    own positions, and the tree follows changes of the sub-structures"""
    DL = Structure("double layer")
    DL.add_sub_structure(unit_cell_iron, 2)
    DL.add_sub_structure(unit_cell_oxygen, 3)
    S = Structure("shared")
    S.add_sub_structure(DL, 2)
    S.add_sub_structure(unit_cell_iron, 4)
    S.add_sub_structure(DL, 1)
    first, _, last = S.layer_tree
    assert first.layer_indices == slice(0, 10) and last.layer_indices == slice(14, 19)
    assert [c.layer_indices for c in last.children] == [slice(14, 16), slice(16, 19)]
    assert [c.sub_structure_indices for c in last.children] == [slice(3, 4), slice(4, 5)]
    DL.set_repetitions(0, 1)
    assert S.layer_tree[-1].layer_indices == slice(12, 16)
    assert S.num_layers == 16
    assert S.num_sub_structures == 5


def test_insert_remove_and_set_repetitions(structure, unit_cell_oxygen):
    """convenience methods to change the sub_structures by their index"""
    S = copy.deepcopy(structure)
    num_layers = S.num_layers
    S.insert_sub_structure(1, unit_cell_oxygen, 3)
    assert S.sub_structures[1] == [unit_cell_oxygen, 3]
    assert S.num_layers == num_layers + 3
    assert S.layer_vectors[1][10:13] == [unit_cell_oxygen.id] * 3
    assert S.remove_sub_structure(1) == (unit_cell_oxygen, 3)
    assert S.layer_vectors[1] == structure.layer_vectors[1]
    # e.g. the number of unit cells of a layer, which zero removes from the layers
    S.set_repetitions(0, np.int64(4))
    assert S.sub_structures[0][1] == 4 and isinstance(S.sub_structures[0][1], int)
    assert S.num_layers == num_layers - 6
    S.set_repetitions(-1, 0)
    assert S.num_layers == num_layers - 6 - 240
    assert S.num_sub_structures == structure.num_sub_structures
    layer, N = S.remove_sub_structure(-1)
    assert isinstance(layer, Structure) and N == 0
    assert S.num_sub_structures == structure.num_sub_structures - 2


@pytest.mark.parametrize(
    "N, error, match",
    [
        (2.0, TypeError, "integer number of repetitions"),
        ("2", TypeError, "integer number of repetitions"),
        (True, TypeError, "not a boolean"),
        (-1, ValueError, "non-negative"),
    ],
)
def test_repetitions_are_validated(structure, unit_cell_oxygen, N, error, match):
    S = copy.deepcopy(structure)
    with pytest.raises(error, match=match):
        S.set_repetitions(0, N)
    with pytest.raises(error, match=match):
        S.add_sub_structure(unit_cell_oxygen, N)
    with pytest.raises(error, match=match):
        S.insert_sub_structure(0, unit_cell_oxygen, N)
    with pytest.raises(error, match=match):
        S.add_substrate(unit_cell_oxygen, N)
    assert S.layer_vectors[1] == structure.layer_vectors[1]


def test_structure_cannot_contain_itself(structure):
    S = copy.deepcopy(structure)
    with pytest.raises(ValueError, match="cannot contain itself"):
        S.add_sub_structure(S)
    nested = S.sub_structures[-1][0]
    with pytest.raises(ValueError, match="cannot contain itself"):
        nested.insert_sub_structure(0, S)


def test_positions_per_unique_layer(structure):
    pos = structure.positions_per_unique_layer
    assert np.allclose(pos["amorphous_layer_Fe"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_layer_distances(structure):
    d_start, _, _ = structure.layer_distances
    # 1 nm amorphous layers followed by 0.5 nm unit cells
    assert_allclose(d_start[[0, 1, 19, 20, 21, 40, -1]], [0, 1, 19, 20, 20.5, 30, 149.5] * u.nm)


def test_layer_distances_are_consistent(structure):
    d_start, d_end, d_mid = structure.layer_distances
    assert_allclose(d_end[-1], structure.thickness)
    assert_allclose(d_start[1:], d_end[:-1])
    assert_allclose(d_mid, (d_start + d_end) / 2)
    np.testing.assert_allclose(structure._layer_distances()[2], d_mid.to("m").magnitude)


def test_interface_distances(structure):
    interfaces = structure.interface_distances
    # the Fe/O interfaces of the double layers start after 20 nm + 10 nm + 2.5 nm
    assert_allclose(interfaces[:5], [0, 10, 20, 32.5, 36] * u.nm)
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
    """the property of all layers in their order, as Quantity or magnitude"""
    density = structure.get_layer_property_vector("density")
    assert len(density) == structure.num_layers
    assert_allclose(density[:20], 5000 * u.kg / u.m**3)
    np.testing.assert_allclose(
        structure.get_layer_property_vector("_density"), density.to("kg/m**3").magnitude
    )
    thickness = structure.get_layer_property_vector("structural.thickness.magnitude")
    np.testing.assert_allclose(thickness[18:22], [1e-9, 1e-9, 5e-10, 5e-10])
    # the functions of the temperature of each sub-system
    heat_capacity = structure.get_layer_property_vector("heat_capacity")
    assert [c[0](300) for c in (heat_capacity[0], heat_capacity[-1])] == [500, 500]
    spring_const = structure.get_layer_property_vector("spring_const")
    assert len(spring_const) == structure.num_layers
    with pytest.raises(AttributeError):
        structure.get_layer_property_vector("structural.not_a_parameter")


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


@pytest.mark.parametrize("types", ["heat", "xray", "optical", ["xray", "magnetic", "optical"]])
def test_structure_get_hash_depends_on_superstrate_and_substrate(
    structure, amorphous_layer_oxygen, types
):
    S = copy.deepcopy(structure)  # do not alter the module-scoped fixtures
    hashes = {S.get_hash(types=types)}
    S.add_substrate(S.substrate[0], 50)
    hashes.add(S.get_hash(types=types))
    S.substrate[0].roughness = 2 * u.nm
    S.substrate[0].dielectric = Dispersion.from_ref_index(3.9 + 0.02j)
    S.substrate[0].density = 2000 * u.kg / u.m**3
    hashes.add(S.get_hash(types=types))
    S.add_superstrate(copy.deepcopy(amorphous_layer_oxygen))
    hashes.add(S.get_hash(types=types))
    assert len(hashes) == 4


def test_structure_get_hash_depends_on_atoms(structure_crystalline, atom_oxygen):
    """the atoms of a unit cell and their positions, ionicities, and form
    factors enter the hash of X-ray simulations"""
    S = copy.deepcopy(structure_crystalline)  # do not alter the module-scoped fixtures
    unit_cell = S.get_layer_handle(0)
    hashes = {S.get_hash(types="xray")}
    unit_cell.atoms[0][0].ionicity = 3
    hashes.add(S.get_hash(types="xray"))
    unit_cell.atoms[0][0].atomic_form_factor_coeff = (
        unit_cell.atoms[0][0].atomic_form_factor_coeff * 1.01
    )
    hashes.add(S.get_hash(types="xray"))
    unit_cell.add_atom(copy.deepcopy(atom_oxygen), 0.5)
    hashes.add(S.get_hash(types="xray"))
    atom, position, position_str = unit_cell.atoms[-1]
    unit_cell.atoms[-1] = [atom, position, "0.6*(1+s)"]
    hashes.add(S.get_hash(types="xray"))
    assert len(hashes) == 5


def test_structure_get_hash_depends_on_direction_of_zero_magnetization(structure):
    """the angles of a zero magnetization are kept, e.g. as initial direction"""
    S = copy.deepcopy(structure)  # do not alter the module-scoped fixtures
    magnetization = S.get_layer_handle(0).magnetic.magnetization
    magnetization.set({"r": 0, "phi": 0 * u.deg, "gamma": 0 * u.deg})
    old_hash = S.get_hash(types="magnetic")
    magnetization.phi = 90 * u.deg
    assert S.get_hash(types="magnetic") != old_hash


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


def test_get_layer_handle(structure, amorphous_layer_iron, unit_cell_oxygen):
    assert structure.get_layer_handle(0) is amorphous_layer_iron
    assert structure.get_layer_handle(-1) is unit_cell_oxygen


def test_reverse(structure):
    """the reversed structure has the layers in reversed order and swaps the
    superstrate and the substrate"""
    reversed_structure = structure.reverse()
    indices, ids, _ = structure.layer_vectors
    _, reversed_ids, _ = reversed_structure.layer_vectors
    assert reversed_ids == ids[::-1]
    assert reversed_structure.num_layers == structure.num_layers
    assert_allclose(reversed_structure.thickness, structure.thickness)
    # of copies of the layers
    assert reversed_structure.substrate[0].id == structure.superstrate[0].id
    assert reversed_structure.superstrate[0].id == structure.substrate[0].id
    assert reversed_structure.superstrate[1] == structure.substrate[1]
    # the original structure is kept
    assert structure.layer_vectors[1] == ids


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
