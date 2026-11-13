#!/usr/bin/env python

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from udkm1Dsim import (
    AmorphousLayer,
    Atom,
    AtomMixed,
    Dielectric,
    Dispersion,
    Layer,
    Structure,
    UnitCell,
    Vacuum,
    u,
)
from udkm1Dsim.structures import yaml_io
from udkm1Dsim.structures.dielectric import _OptPenDepth

HASH_TYPES = ("heat", "phonon", "xray", "optical", "magnetic")
FE_CF = Path(__file__).parent / "data" / "Fe.cf"
K_B = 1.380649e-23
EXCHANGE = np.array([[3 * K_B * 500, -K_B * 100], [-K_B * 100, 3 * K_B * 300]])


def round_trip(S, **kwargs):
    """Structure read back from its YAML text, which is checked on writing."""
    text = S.to_yaml(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        read = Structure.from_yaml(text)
    return text, read


def assert_same_structure(S, read):
    """Same hashes, YAML, and order of the layers."""
    for types in HASH_TYPES:
        assert S.get_hash(types=types) == read.get_hash(types=types), types
    assert read.to_yaml() == S.to_yaml()
    assert read.layer_vectors[1] == S.layer_vectors[1]
    assert read.layer_vectors[0].tolist() == S.layer_vectors[0].tolist()
    # new layers
    assert all(a is not b for a, b in zip(read.layer_vectors[2], S.layer_vectors[2]))


def single(layer, name="sample"):
    S = Structure(name)
    S.add_sub_structure(layer, 2)
    return S


def substrate():
    return Layer("substrate", "substrate", thickness=10 * u.nm, density=2000 * u.kg / u.m**3)


# structures of the conftest


@pytest.mark.parametrize("name", ["structure", "structure_amorph", "structure_crystalline"])
def test_round_trip_of_fixtures(name, request):
    S = request.getfixturevalue(name)
    text, read = round_trip(S)
    assert_same_structure(S, read)
    assert yaml.safe_load(text)["format_version"] == yaml_io.FORMAT_VERSION
    assert read.name == S.name
    assert read.num_layers == S.num_layers
    assert read.substrate[1] == S.substrate[1]


def test_round_trip_of_layers(layer, amorphous_layer, unit_cell):
    S = Structure("layers")
    for sub_structure in (layer, amorphous_layer, unit_cell):
        S.add_sub_structure(sub_structure, 3)
    _, read = round_trip(S)
    assert_same_structure(S, read)
    for original, copy in zip(S.unique_layers[1], read.unique_layers[1]):
        assert type(copy) is type(original)
        assert repr(copy) == repr(original)


def test_shared_objects_are_written_once(structure):
    data = yaml_io.structure_to_dict(structure)
    # the double layer and the unit cells of iron are only written once
    assert list(data["structures"]) == ["double layer"]
    assert set(data["layers"]) == {
        "amorphous_layer_Fe",
        "amorphous_layer_O",
        "unit_cell_Fe",
        "unit_cell_O",
    }
    # the copies of the atoms in the dispersions of the layers share their key
    assert set(data["atoms"]) == {"Fe", "O"}
    # the oxygen of the conftest has no Cromer-Mann coefficients for its ionicity
    with pytest.warns(UserWarning, match="closest ionicity"):
        read = yaml_io.structure_from_dict(data)
    handles = read.layer_vectors[2]
    assert handles[20] is handles[40]
    assert read.substrate[0] is handles[0]


def test_different_objects_with_the_same_id_get_different_keys():
    first = Layer("layer", "first", thickness=1 * u.nm)
    second = Layer("layer", "second", thickness=2 * u.nm)
    S = Structure("same ids")
    S.add_sub_structure(first)
    S.add_substrate(second)
    data = yaml_io.structure_to_dict(S)
    assert data["layers"]["layer_2"]["id"] == "layer"
    read = yaml_io.structure_from_dict(data)
    assert read.substrate[0].id == "layer"
    assert read.substrate[0].name == "second"


def test_default_parameters_are_not_written():
    S = single(Layer("layer", "layer", thickness=1 * u.nm))
    data = yaml_io.structure_to_dict(S)
    assert data["layers"]["layer"] == {
        "type": "Layer",
        "name": "layer",
        "structural": {"thickness": "1.0 nm"},
    }
    assert "atoms" not in data and "structures" not in data
    # the default vacuum superstrate and substrate
    assert set(data["structure"]) == {"name", "sub_structures"}


def test_readable_and_exact_units():
    layer = Layer(
        "layer",
        "layer",
        thickness=3.905 * u.angstrom,
        density=5000 * u.kg / u.m**3,
        roughness=1e-10 * u.m,
        sound_vel=6.312 * u.nm / u.ps,
        therm_cond=5.72 * u.W / (u.m * u.K),
        heat_capacity="455.2 + 0.112*T - 2.1935e6/T**2",
    )
    data = yaml_io.structure_to_dict(single(layer))["layers"]["layer"]
    assert data["structural"] == {
        "thickness": "3.905 Å",
        "density": "5000.0 kg/m³",
        "roughness": "1.0 Å",
    }
    assert data["elastic"] == {"sound_vel": "6.312 nm/ps"}
    assert data["thermal"]["therm_cond"] == "5.72 W/K/m"
    assert data["thermal"]["heat_capacity"] == "0.112*T + 455.2 - 2193500.0/T**2"


def test_hand_written_yaml(tmp_path):
    text = """
format_version: 1
atoms:
  O: {symbol: O}
  Sr: {symbol: Sr}
  Ru: {symbol: Ru}
layers:
  SRO:
    type: UnitCell
    name: Strontium Ruthenate
    lattice: {a_axis: 3.905 Å, b_axis: 3.905 Å, c_axis: 0.394897 nm}
    thermal:
      therm_cond: 5.72 W/(m K)
      lin_therm_exp: 1e-5
      heat_capacity: 455.2 + 0.112*T - 2.1935e6/T**2
    elastic: {sound_vel: 6.312 nm/ps}
    optical:
      dielectric: {ref_index: 2.44+4.32j}
    atoms:
      - [O, 0]
      - [Sr, 0]
      - [O, 0.5]
      - [O, 0.5]
      - [Ru, 0.5]
  substrate:
    structural: {thickness: 1 µm, density: 5.1 g/cm**3}
structure:
  name: SRO
  sub_structures:
    - [SRO, 100]
  substrate: [substrate, 10]
"""
    filename = tmp_path / "sample.yml"
    filename.write_text(text, encoding="utf-8")
    S = Structure.from_yaml(filename)
    assert S.num_layers == 100
    SRO = S.unique_layers[1][0]
    assert SRO.num_atoms == 5
    assert SRO.c_axis.to("angstrom").magnitude == pytest.approx(3.94897)
    assert SRO.therm_cond[0](300) == pytest.approx(5.72)
    assert SRO.lin_therm_exp[0](300) == pytest.approx(1e-5)
    assert SRO.heat_capacity[0](300) == pytest.approx(455.2 + 0.112 * 300 - 2.1935e6 / 300**2)
    assert SRO.sound_vel.to("nm/ps").magnitude == pytest.approx(6.312)
    assert SRO.get_ref_index(1.55) == pytest.approx(2.44 + 4.32j)
    layer, N = S.substrate
    assert N == 10 and type(layer) is Layer
    assert layer.density.to("g/cm**3").magnitude == pytest.approx(5.1)
    # also from the path as str and from the text
    for read in (Structure.from_yaml(str(filename)), Structure.from_yaml(text)):
        for types in HASH_TYPES:
            assert read.get_hash(types=types) == S.get_hash(types=types)


# atoms


def test_atoms():
    fe = Atom("Fe", id="Fe1", magnetization={"r": 1, "phi": 90 * u.deg, "gamma": 45 * u.deg})
    o = Atom("O", atomic_form_factor_source="henke")
    with pytest.warns(UserWarning, match="closest ionicity"):
        o.ionicity = 1
    ti = Atom("Ti", atomic_form_factor_path=FE_CF)
    ti.name = "Titanium of iron"
    sr = Atom("Sr")
    sr.atomic_form_factor_coeff = np.array([[10.0, 30.0, 1.0], [20000.0, 38.0, 0.1]])
    sr.magnetic_form_factor_coeff = np.array([[10.0, 0.0, 1.0], [20000.0, 0.0, 0.1]])
    sr.cromer_mann_coeff = np.r_[38, 0, np.arange(9.0)]
    mixed = AtomMixed("FeO", id="mixed", name="iron oxide")
    mixed.add_atom(fe, 0.5)
    mixed.add_atom(o, 0.5)
    mixed.magnetization = (0, 0, 0.5)
    uc = UnitCell("uc", "unit cell", 4 * u.angstrom)
    for i, atom in enumerate([fe, o, ti, sr, mixed]):
        uc.add_atom(atom, i / 5)
    uc.add_atom(o, "0.3*(1+s) + 0.01*s**2")
    uc.set_dielectric_from_atoms()
    S = single(uc)
    _, read = round_trip(S)
    assert_same_structure(S, read)
    atoms = {atom.id: atom for atom, _, _ in read.unique_layers[1][0].atoms}
    assert atoms["Fe1"].magnetization.magnitude == pytest.approx([1 / 2**0.5, 1 / 2**0.5, 0])
    assert atoms["O"].form_factor_source() == "Henke" and atoms["O"].ionicity == 1
    assert atoms["Ti"].name == "Titanium of iron"
    np.testing.assert_array_equal(
        atoms["Ti"].atomic_form_factor_coeff, ti.atomic_form_factor_coeff
    )
    assert atoms["Sr"].form_factor_source() == "custom"
    assert atoms["Sr"].form_factor_source(magnetic=True) == "custom"
    np.testing.assert_array_equal(atoms["Sr"].cromer_mann_coeff, sr.cromer_mann_coeff)
    assert atoms["mixed"].name == "iron oxide"
    assert atoms["mixed"].atomic.mass.magnitude == mixed.atomic.mass.magnitude
    positions = [position for _, _, position in read.unique_layers[1][0].atoms]
    assert "0.3*(1+s) + 0.01*s**2" in positions


# magnetization


def test_magnetization_of_layers():
    zero = Layer(
        "zero", "zero", thickness=1 * u.nm, magnetization={"r": 0, "phi": 30, "gamma": 60}
    )
    reversed_ = Layer(
        "reversed",
        "reversed",
        thickness=1 * u.nm,
        magnetization={"r": -0.5, "phi": 90 * u.deg, "gamma": 0 * u.deg},
    )
    cartesian = Layer("cartesian", "cartesian", thickness=1 * u.nm, magnetization=(1, 2, 3))
    fe = Atom("Fe", magnetization=(0, 0, 1))
    amorphous = AmorphousLayer("amorphous", "amorphous", 1 * u.nm, 7000 * u.kg / u.m**3, atom=fe)
    # the layer differs from its atom
    amorphous.magnetization = (1, 0, 0)
    S = Structure("magnetization")
    for layer in (zero, reversed_, cartesian, amorphous):
        S.add_sub_structure(layer)
    _, read = round_trip(S)
    assert_same_structure(S, read)
    for original, copy in zip(S.unique_layers[1], read.unique_layers[1]):
        np.testing.assert_array_equal(
            copy.magnetic.magnetization.polar_rad, original.magnetic.magnetization.polar_rad
        )


def test_sublattices():
    dy, fe = Atom("Dy", magnetization=(0, 0, 1)), Atom("Fe", magnetization=(0, 0, -1))
    mixed = AtomMixed("DyFe")
    mixed.add_atom(dy, 0.3)
    mixed.add_atom(fe, 0.7)
    layer = AmorphousLayer(
        "ferrimagnet",
        "ferrimagnet",
        10 * u.nm,
        8000 * u.kg / u.m**3,
        atom=mixed,
        eff_spin=[np.inf, np.inf],
        lamda=[0.01, 0.02],
        anisotropy=[[0, 0, 1], [0, 0, 2]] * u.J / u.m**3,
        mf_exch_coupling=EXCHANGE * u.J,
    )
    explicit = Layer(
        "explicit",
        "explicit",
        thickness=1 * u.nm,
        eff_spin=[0.5, 0.5],
        mf_exch_coupling=EXCHANGE * u.J,
        concentration=[0.4, 0.6],
        magnetization=[(0, 0, 1), {"r": -0.5, "phi": 0, "gamma": 0}],
    )
    S = Structure("sublattices")
    S.add_sub_structure(layer)
    S.add_sub_structure(explicit)
    _, read = round_trip(S)
    assert_same_structure(S, read)
    magnetic = read.unique_layers[1][0].magnetic
    assert magnetic.num_sublattices == magnetic.check_sublattices() == 2
    assert magnetic.concentration.magnitude.tolist() == [0.3, 0.7]
    assert "curie_temp" in magnetic._derived
    magnetic = read.unique_layers[1][1].magnetic
    assert magnetic._concentration_explicit
    assert [m.r.magnitude for m in magnetic.magnetization] == [1, -0.5]


# thermal and elastic parameters


def test_sub_systems_and_higher_orders():
    layer = Layer(
        "layer",
        "layer",
        thickness=1 * u.nm,
        density=5000 * u.kg / u.m**3,
        sound_vel=5 * u.nm / u.ps,
        heat_capacity=["0.1*T", 450 * u.J / u.kg / u.K, "2e3"],
        therm_cond=[1, "10*T_0/T_1", 0.5],
        sub_system_coupling=["1e18*(T_1-T_0)", "-1e18*(T_1-T_0)", 0],
        sub_system_labels=["e", "p", "s"],
    )
    layer.set_ho_spring_constants([1e3, 2e4])
    S = single(layer)
    _, read = round_trip(S)
    assert_same_structure(S, read)
    copy = read.unique_layers[1][0]
    assert copy.thermal.sub_system_labels == ["e", "p", "s"]
    assert copy.num_sub_systems == 3
    np.testing.assert_array_equal(copy.spring_const.magnitude, layer.spring_const.magnitude)


# dispersions and dielectric tensors


def layer_with(dielectric, id="layer"):
    return Layer(id, id, thickness=1 * u.nm, density=5000 * u.kg / u.m**3, dielectric=dielectric)


def dispersions(tmp_path):
    """All kinds of dispersions and their combinations by their name."""
    nk = tmp_path / "data" / "test.nk"
    nk.parent.mkdir()
    nk.write_text("# wl n k\n0.2 2.0 0.1\n0.4 2.2 0.2\n0.8 1.5 0.0\n2.0 1.4 0.0\n")
    fe = Atom("Fe")
    with pytest.warns(UserWarning):
        mass_attenuation = Dispersion.from_mass_attenuation(
            [100.0, 1000.0, 10000.0], [1e5, 1e4, 1e2]
        )
    table = Dispersion.from_array([1, 2, 3] * u.eV, [[2, 2.1], [2.2, 2.3], [2.4, 2.5]], kind="n")
    wavelengths = Dispersion.from_array([800, 400] * u.nm, [2 + 1j, 3 + 2j])
    return {
        "ref_index": Dispersion.from_ref_index(2.44 + 4.32j),
        "uniaxial": Dispersion.from_ref_index([2 + 0.1j, 2.2 + 0.2j]),
        "tensor": Dispersion.from_array(None, np.diag([2, 3, 4]) + 0.1j, kind="eps"),
        "chi": Dispersion.from_array(None, -1e-5 + 1e-6j, kind="chi"),
        "table": table,
        "wavelengths": wavelengths,
        "nk_file": Dispersion.from_file(nk),
        "form_factors_file": Dispersion.from_form_factors(FE_CF, 55.845),
        "form_factors_array": Dispersion.from_array(
            [10, 20000] * u.eV, [26 + 1j, 26 + 0.1j], kind="f", molar_mass=55.845 * u.g / u.mol
        ),
        "atoms": Dispersion.from_array(None, 1.5) | _atoms_dispersion(fe),
        "opt_pen_depth": Dispersion.from_opt_pen_depth(20 * u.nm, n=1.5),
        "opt_pen_depth_base": _OptPenDepth(10 * u.nm, base=wavelengths),
        "mass_attenuation": mass_attenuation | Dispersion.from_ref_index(1.0),
        "sum": Dispersion.from_ref_index(2) + Dispersion.from_array(None, 0.1j, kind="chi"),
        "difference": Dispersion.from_ref_index(2) - Dispersion.from_array(None, 0.1, kind="chi"),
        "scaled": 0.5 * Dispersion.from_ref_index(2),
        "scaled_complex": (0.5 + 0.1j) * Dispersion.from_ref_index(2),
        "negative": -Dispersion.from_array(None, 0.1, kind="chi") + Dispersion.from_ref_index(3),
    }


def _atoms_dispersion(atom):
    layer = AmorphousLayer("atoms", "atoms", 1 * u.nm, 7000 * u.kg / u.m**3, atom=atom)
    return layer.dispersion_from_atoms()


def test_all_dispersions(tmp_path):
    S = Structure("dispersions")
    for name, dispersion in dispersions(tmp_path).items():
        S.add_sub_structure(layer_with(dispersion, name))
    filename = tmp_path / "yaml" / "sample.yml"
    filename.parent.mkdir()
    S.to_yaml(filename)
    data = yaml.safe_load(filename.read_text(encoding="utf-8"))
    # paths relative to the YAML file
    assert data["layers"]["nk_file"]["optical"]["dielectric"] == {"file": "../data/test.nk"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        read = Structure.from_yaml(filename)
    for types in HASH_TYPES:
        assert S.get_hash(types=types) == read.get_hash(types=types)
    energies = {"form_factors_file": 1000, "form_factors_array": 1000, "mass_attenuation": 500}
    for original, copy in zip(S.unique_layers[1], read.unique_layers[1]):
        energy = energies.get(original.id, 2)
        np.testing.assert_array_equal(
            copy.get_dielectric_tensor(energy), original.get_dielectric_tensor(energy)
        )


def has_key(data, key):
    """True if a nested dict or list has a dict with the key."""
    if isinstance(data, dict):
        return key in data or any(has_key(v, key) for v in data.values())
    if isinstance(data, list):
        return any(has_key(v, key) for v in data)
    return False


def test_embedded_dispersions(tmp_path):
    S = Structure("embedded")
    for name, dispersion in dispersions(tmp_path).items():
        S.add_sub_structure(layer_with(dispersion, name))
    fe = Atom("Fe", atomic_form_factor_path=FE_CF)
    S.add_sub_structure(AmorphousLayer("Fe", "Fe", 1 * u.nm, 7000 * u.kg / u.m**3, atom=fe))
    data = yaml.safe_load(S.to_yaml(embed=True))
    assert not has_key(data["layers"], "file") and not has_key(data["atoms"], "file")
    assert data["layers"]["nk_file"]["optical"]["dielectric"]["table"]["origin"] == "test.nk"
    _, read = round_trip(S, embed=True)
    for types in HASH_TYPES:
        assert S.get_hash(types=types) == read.get_hash(types=types)
    assert read.unique_layers[1][-1].atom.form_factor_source() == "Fe.cf"


def test_dielectric_tensor():
    eps0 = Dispersion.from_ref_index([2 + 0.1j, 2.2 + 0.2j])
    dielectric = Dielectric(
        eps0=eps0,
        eps_mag=Dispersion.from_array(None, 0.01j, kind="voigt"),
        strain_model="photoelastic",
        deps_dstrain=[0.1, 0.2, 0.3 + 0.01j],
        euler_angles=(10, 20.5 * u.deg, 0.1 * u.rad),
    )
    dielectric.display_energy = 800 * u.nm
    deps = Dielectric(eps0=eps0, strain_model="photoelastic", deps_dstrain=eps0)
    S = Structure("dielectric")
    S.add_sub_structure(layer_with(dielectric, "full"))
    S.add_sub_structure(layer_with(deps, "deps"))
    S.add_sub_structure(layer_with(Dielectric(eps0=eps0), "density"))
    _, read = round_trip(S)
    assert_same_structure(S, read)
    full = read.unique_layers[1][0].optical.dielectric
    assert full.display_energy == 800 * u.nm
    np.testing.assert_array_equal(full._euler_angles, dielectric._euler_angles)
    np.testing.assert_array_equal(
        full.get_tensor(1.5, strain=0.01, magnetization=(0, 0, 1)),
        dielectric.get_tensor(1.5, strain=0.01, magnetization=(0, 0, 1)),
    )
    assert read.unique_layers[1][2].optical.dielectric.strain_model == "density"


def test_refractiveindex():
    pytest.importorskip("refractiveindex")
    dispersion = Dispersion.from_refractiveindex("main", "Cr", "Johnson", num=200)
    S = single(layer_with(dispersion))
    data = yaml_io.structure_to_dict(S)
    written = data["layers"]["layer"]["optical"]["dielectric"]
    assert written["refractiveindex"] == {
        "shelf": "main",
        "book": "Cr",
        "page": "Johnson",
        "num": 200,
    }
    _, read = round_trip(S)
    assert_same_structure(S, read)
    written["version"] = "other"
    with pytest.warns(UserWarning, match="version other"):
        yaml_io.structure_from_dict(data)


# substrates, superstrates, and nested structures


def test_nested_structures_substrate_and_superstrate():
    a = Layer("a", "a", thickness=1 * u.nm)
    b = Layer("b", "b", thickness=2 * u.nm)
    inner = Structure("inner")
    inner.add_sub_structure(a, 2)
    inner.add_sub_structure(b, 1)
    outer = Structure("outer")
    outer.add_sub_structure(inner, 3)
    outer.add_sub_structure(b, 2)
    S = Structure("sample")
    S.add_sub_structure(outer, 2)
    S.add_sub_structure(inner, 1)
    S.add_superstrate(Vacuum(thickness=2 * u.nm))
    S.add_substrate(substrate(), 100)
    data = yaml_io.structure_to_dict(S)
    assert list(data["structures"]) == ["inner", "outer"]
    assert data["structure"]["superstrate"] == "vacuum"
    assert data["structure"]["substrate"] == ["substrate", 100]
    _, read = round_trip(S)
    assert_same_structure(S, read)
    assert read.superstrate[0].thickness.to("nm").magnitude == pytest.approx(2)
    assert type(read.superstrate[0]) is Vacuum
    assert read.num_layers == S.num_layers


def test_empty_structure():
    _, read = round_trip(Structure("empty"))
    assert read.name == "empty" and read.sub_structures == []


# errors


def test_unsupported_layer_raises():
    class Custom(Layer):
        pass

    with pytest.raises(TypeError, match="Custom"):
        single(Custom("custom", "custom", thickness=1 * u.nm)).to_yaml()


def test_unsupported_dispersion_raises():
    class Custom(Dispersion):
        scalar = True

        def _eps(self, energy, density, vacuum, eps_xx):
            return np.full(energy.shape, 2.0 + 0j)

    with pytest.raises(TypeError, match="Custom"):
        single(layer_with(Custom())).to_yaml()


def test_check_raises_for_differences(monkeypatch):
    S = single(Layer("layer", "layer", thickness=1 * u.nm))
    original = yaml_io.structure_from_dict

    def lossy(data, base_dir=None):
        read = original(data, base_dir)
        read.unique_layers[1][0].thickness = 2 * u.nm
        return read

    monkeypatch.setattr(yaml_io, "structure_from_dict", lossy)
    with pytest.raises(ValueError, match="thickness"):
        S.to_yaml()
    # without the check
    assert "thickness: 1.0 nm" in S.to_yaml(check=False)


@pytest.mark.parametrize(
    "data, match",
    [
        ({"structure": {}}, "format_version"),
        ({"format_version": 99, "structure": {}}, "newer"),
        ({"format_version": 1}, "missing"),
        ({"format_version": 1, "structure": {}, "other": 1}, "Unknown sections"),
        (
            {"format_version": 1, "structure": {"sub_structures": [["a", 1]]}},
            "'a' is not defined",
        ),
        (
            {
                "format_version": 1,
                "layers": {"a": {"thickness": "1 nm"}},
                "structure": {"sub_structures": [["a", 1]]},
            },
            "Unknown entries",
        ),
        (
            {
                "format_version": 1,
                "layers": {"a": {"structural": {"unknown": 1}}},
                "structure": {"sub_structures": [["a", 1]]},
            },
            "Unknown parameter 'unknown'",
        ),
        (
            {
                "format_version": 1,
                "layers": {"a": {"type": "Other"}},
                "structure": {"sub_structures": [["a", 1]]},
            },
            "Unknown type 'Other'",
        ),
        (
            {
                "format_version": 1,
                "layers": {"a": {"optical": {"dielectric": {"unknown": 1}}}},
                "structure": {"sub_structures": [["a", 1]]},
            },
            "Unknown dispersion",
        ),
    ],
)
def test_invalid_yaml_raises(data, match):
    with pytest.raises(ValueError, match=match):
        yaml_io.structure_from_dict(data)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Structure.from_yaml("missing.yml")
