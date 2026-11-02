#!/usr/bin/env python


import copy
from pathlib import Path

import numpy as np
import pytest
from pint.testing import assert_allclose, assert_equal

from udkm1Dsim import (
    Atom,
    AtomMixed,
    AtomMixedParameters,
    AtomParameters,
    Layer,
    VectorParameter,
    u,
)

# tests


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("atom_iron", "Fe"), ("atom_dysprosium", "Dy"), ("atom_mixed", "DyFe")],
)
def test_atom_symbol(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.symbol == expected


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("atom_iron", "Fe"), ("atom_dysprosium", "Dy"), ("atom_mixed", "DyFe")],
)
def test_atom_id(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.id == expected


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("atom_iron", "Iron"), ("atom_dysprosium", "Dysprosium"), ("atom_mixed", "DyFe")],
)
def test_atom_name(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.name == expected


@pytest.mark.parametrize(
    "fixture_name, expected", [("atom_iron", 0), ("atom_dysprosium", 1), ("atom_mixed", 0.4)]
)
def test_atom_ionicity(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.ionicity == expected


@pytest.mark.parametrize(
    "fixture_name, expected", [("atom_iron", 26), ("atom_dysprosium", 66), ("atom_mixed", 42)]
)
def test_atom_atomic_number_z(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.atomic_number_z == expected


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("atom_iron", 55.845), ("atom_dysprosium", 162.5), ("atom_mixed", 98.507)],
)
def test_atom_mass_number_a(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.mass_number_a == expected


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("atom_iron", 9.273e-26 * u.kg),
        ("atom_dysprosium", 2.698e-25 * u.kg),
        ("atom_mixed", 1.636e-25 * u.kg),
    ],
)
def test_atom_mass(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert_allclose(atom.mass, expected, rtol=1e-3)


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("atom_iron", [20.83314, 4.7737, 3.0272]), ("atom_dysprosium", [14.92335, 10.9656, 0.46755])],
)
def test_atom_atomic_form_factor_coeff(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert np.allclose(atom.atomic_form_factor_coeff[10], np.array(expected))


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        (
            "atom_iron",
            [26.0, 0.0, 11.7695, 7.3573, 3.5222, 2.3045, 4.7611, 0.3072, 15.3535, 76.8805, 1.0369],
        ),
        (
            "atom_dysprosium",
            [
                66.0,
                0.0,
                26.507,
                17.6383,
                14.5596,
                2.96577,
                2.1802,
                0.202172,
                12.1899,
                111.874,
                4.29728,
            ],
        ),
    ],
)
def test_atom_cromer_mann_coeff(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert np.allclose(atom.cromer_mann_coeff, np.array(expected))


@pytest.mark.parametrize(
    "fixture_name, expected",
    [("atom_iron", [5.0148e02, 1.6077e-02, 0.0000e00]), ("atom_dysprosium", [5000.0, 0.0, 0.0])],
)
def test_atom_magnetic_form_factor_coeff(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert np.allclose(atom.magnetic_form_factor_coeff[4], np.array(expected))


# atom_mixed must read atomic form factors from an external file and is hence excluded here
@pytest.mark.parametrize("fixture_name", ["atom_iron", "atom_dysprosium"])
def test_atom_read_atomic_form_factor_coeff(request, fixture_name):
    atom = request.getfixturevalue(fixture_name)
    atom.read_atomic_form_factor_coeff(source="chantler")
    atom.read_atomic_form_factor_coeff(source="henke")
    with pytest.raises(ValueError):
        atom.read_atomic_form_factor_coeff(source="wrong")


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("atom_iron", np.complex128(3.84 - 1.85j)),
        ("atom_dysprosium", np.complex128(36.00 - 11.25j)),
        ("atom_mixed", np.complex128(16.71 - 5.61j)),
    ],
)
def test_atom_get_atomic_form_factor(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.get_atomic_form_factor(700) == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("atom_iron", np.complex128(-33.9 - 1.9j)),
        ("atom_dysprosium", np.complex128(-56.5 - 11.3j)),
        ("atom_mixed", np.complex128(-43.0 - 5.6j)),
    ],
)
def test_atom_get_cm_atomic_form_factor(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.get_cm_atomic_form_factor(700, 1) == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("fixture_name", ["atom_iron", "atom_dysprosium", "atom_mixed"])
def test_atom_read_magnetic_form_factor_coeff(request, fixture_name):
    atom = request.getfixturevalue(fixture_name)
    atom.read_magnetic_form_factor_coeff()


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("atom_iron", np.complex128(1.11 - 0.01j)),
        ("atom_dysprosium", np.complex128(0j)),
        ("atom_mixed", np.complex128(0.67 - 0.01j)),
    ],
)
def test_atom_get_magnetic_form_factor(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.get_magnetic_form_factor(700) == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("atom_iron", [0.5, 0 * u.deg, 180 * u.deg]),
        ("atom_dysprosium", [0, 0 * u.deg, 0 * u.deg]),
        ("atom_mixed", [0, 0 * u.deg, 0 * u.deg]),
    ],
)
def test_atom_magnetization(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.mag_amplitude == expected[0]
    assert_equal(atom.mag_phi, expected[1])
    assert_equal(atom.mag_gamma, expected[2])


@pytest.mark.parametrize(
    "fixture_name, cls_name",
    [("atom_iron", "Atom"), ("atom_dysprosium", "Atom"), ("atom_mixed", "AtomMixed")],
)
def test_atom_repr(request, fixture_name, cls_name):
    atom = request.getfixturevalue(fixture_name)
    text = repr(atom)
    assert text.startswith(f"{cls_name}: {atom.name}\nID: {atom.id}\nSymbol: {atom.symbol}\n")
    assert "Atom Parameters" not in text  # the table of the group without its title
    assert "mass_number" in text and "Symbol" in text
    assert str(atom) == repr(atom)
    html = atom._repr_html_()
    assert html.startswith(f"<h3>{cls_name}: {atom.name}</h3>")
    assert "<td>mass_number</td>" in html


@pytest.mark.parametrize("fixture_name", ["atom_iron", "atom_dysprosium", "atom_mixed"])
def test_atom_read_atomic_form_factor_coeff_from_file(request, fixture_name):
    atom = request.getfixturevalue(fixture_name)
    filename = Path.joinpath(Path(__file__).parent, "data", "Fe.cf")
    atom.read_atomic_form_factor_coeff(filename=filename)
    with pytest.raises(FileNotFoundError):
        atom.read_atomic_form_factor_coeff(filename="wrong")


@pytest.mark.parametrize("fixture_name", ["atom_iron", "atom_dysprosium", "atom_mixed"])
def test_atom_read_magnetic_form_factor_coeff_from_file(request, fixture_name):
    atom = request.getfixturevalue(fixture_name)
    # magnetic and electronic form factor files have the same format
    filename = Path.joinpath(Path(__file__).parent, "data", "Fe.cf")
    mag_coeff = atom.read_magnetic_form_factor_coeff(filename=filename)
    atom.magnetic_form_factor_coeff = mag_coeff
    atom.get_magnetic_form_factor(700)
    atom.read_magnetic_form_factor_coeff(filename="wrong")


# ============================================================================
# parameter groups of atoms
# ============================================================================


def test_atom_parameters(atom_iron):
    assert isinstance(atom_iron.atomic, AtomParameters)
    assert atom_iron.atomic.mass_number.magnitude == atom_iron.mass_number_a
    assert atom_iron.atomic.mass.quantity == atom_iron.mass
    assert atom_iron._mass == atom_iron.mass.magnitude
    symbols = {name: p.symbol for name, p in atom_iron.atomic._parameters().items()}
    assert symbols == {
        "atomic_number": "Z",
        "mass_number": "A",
        "mass": "m",
        "ionicity": "I",
        "magnetization": r"\mathbf{m}",
        "cromer_mann_a": "a",
        "cromer_mann_b": "b",
        "cromer_mann_c": "c",
    }


def test_atom_mass_is_derived_from_mass_number():
    atom = Atom("Fe")
    atom.mass_number_a = 2 * atom.mass_number_a
    assert_allclose(atom.mass, 2 * Atom("Fe").mass)
    with pytest.raises(AttributeError):
        atom.mass = 1 * u.kg


@pytest.mark.parametrize("cls", ["Atom", "AtomMixed", "Layer"])
def test_magnetization_of_atoms_and_layers_is_identical(cls):
    if cls == "Atom":
        obj = Atom("Fe")
    elif cls == "AtomMixed":
        obj = AtomMixed("FeCr")
    else:
        obj = Layer("Fe", "Fe")
    group = obj.magnetic if cls == "Layer" else obj.atomic
    assert isinstance(group.magnetization, VectorParameter)
    assert group.magnetization.symbol == r"\mathbf{m}"
    group.magnetization.polar = (1, 90 * u.deg, 270 * u.deg)
    # the gamma angle is given in -180 .. 180 deg
    assert_allclose(group.magnetization.polar[2], -90 * u.deg)
    assert np.allclose(obj.magnetization.magnitude, [0, -1, 0])
    with pytest.raises(ValueError):
        group.magnetization.polar = (-1, 0 * u.deg, 0 * u.deg)
    # the angles are given in deg in the tables
    rows = group.magnetization._table_rows("text")
    assert [row[0] for row in rows] == ["r", "φ", "γ"]
    assert [row[2] for row in rows] == ["", "deg", "deg"]


def test_atom_magnetization_attributes():
    atom = Atom("Fe", mag_amplitude=1, mag_phi=90 * u.deg, mag_gamma=45 * u.deg)
    assert atom.mag_amplitude == pytest.approx(1)
    assert_allclose(atom.mag_phi, 90 * u.deg)
    assert_allclose(atom.mag_gamma, 45 * u.deg)
    assert atom._mag_gamma == pytest.approx(np.pi / 4)
    atom.mag_gamma = 0 * u.deg
    assert np.allclose(atom.magnetization.magnitude, [1, 0, 0])


def test_atom_mixed_parameters(atom_mixed, atom_iron, atom_dysprosium):
    assert isinstance(atom_mixed.atomic, AtomMixedParameters)
    assert_allclose(atom_mixed.mass, 0.6 * atom_iron.mass + 0.4 * atom_dysprosium.mass)
    # the mass of a mixed atom is also available as float in kg
    assert atom_mixed._mass == atom_mixed.mass.magnitude


@pytest.mark.parametrize("attribute", ["atomic_number_z", "mass_number_a", "ionicity", "mass"])
def test_atom_mixed_derived_parameters_are_read_only(atom_mixed, attribute):
    with pytest.raises(AttributeError):
        setattr(atom_mixed, attribute, 1)


def test_atom_parameters_are_copied(atom_iron):
    atom = copy.deepcopy(atom_iron)
    atom.mag_amplitude = 0.1
    assert atom_iron.mag_amplitude != 0.1
    assert atom.atomic is not atom_iron.atomic


def test_atom_cromer_mann_parameters(atom_iron):
    p = atom_iron.atomic
    cm = atom_iron.cromer_mann_coeff
    # the coefficients as read from the parameter file, including Z and ionicity
    assert np.array_equal(cm[0:2], [26, 0])
    assert np.array_equal(p.cromer_mann_a.magnitude, cm[2:6])
    assert np.array_equal(p.cromer_mann_b.magnitude, cm[6:10])
    assert p.cromer_mann_c.magnitude == cm[10]
    assert p.cromer_mann_b.unit == u.angstrom**2


def test_atom_mixed_has_no_cromer_mann_parameters(atom_mixed):
    assert not hasattr(atom_mixed.atomic, "cromer_mann_a")
    assert "cromer_mann" not in repr(atom_mixed)
