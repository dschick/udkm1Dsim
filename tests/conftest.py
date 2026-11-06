#!/usr/bin/env python

import pytest

from udkm1Dsim import (
    AmorphousLayer,
    Atom,
    AtomMixed,
    Dispersion,
    Layer,
    Parameter,
    Structure,
    TemperatureParameter,
    UnitCell,
    Vacuum,
    VectorParameter,
    u,
)

# fixtures

# parameters


# refractive index of the layers for the optical excitation
REF_INDEX = 5 - 3j


def _dielectric_from_atoms_and_optics(layer):
    """Dielectric from the atoms in the X-ray range and a constant refractive
    index at all other energies, e.g. for the optical excitation."""
    layer.set_dielectric_from_atoms(fallback=Dispersion.from_ref_index(REF_INDEX))
    return layer


@pytest.fixture(scope="module")
def parameter():
    return Parameter("m", 0, "parameter")


# temperature paramter


@pytest.fixture(scope="module")
def temperature_parameter():
    return TemperatureParameter("J/kg/K", 0, "temperature_parameter")


# vector parameter


@pytest.fixture
def vector_parameter():
    return VectorParameter("T", name="B")


# atoms


@pytest.fixture(scope="module")
def atom_iron():
    atom_iron = Atom("Fe")
    atom_iron.magnetization = {"r": 0.5, "phi": 0 * u.deg, "gamma": 180 * u.deg}
    return atom_iron


@pytest.fixture(scope="module")
def atom_oxygen():
    atom_oxygen = Atom("O")
    # not tabulated for the Cromer-Mann coefficients, which fall back to ionicity 0
    with pytest.warns(UserWarning, match="closest ionicity 0"):
        atom_oxygen.ionicity = 1
    atom_oxygen.magnetization = {"r": 0.5, "phi": 0 * u.deg, "gamma": 180 * u.deg}
    return atom_oxygen


@pytest.fixture(scope="module")
def atom_strontium():
    atom_strontium = Atom("Sr")
    atom_strontium.magnetization = {"r": 0.5, "phi": 0 * u.deg, "gamma": 180 * u.deg}
    return atom_strontium


@pytest.fixture(scope="module")
def atom_titanium():
    atom_titanium = Atom("Ti")
    atom_titanium.magnetization = {"r": 0.5, "phi": 0 * u.deg, "gamma": 180 * u.deg}
    return atom_titanium


@pytest.fixture(scope="module")
def atom_dysprosium():
    atom_dysprosium = Atom("Dy")
    # not tabulated for the Cromer-Mann coefficients, which fall back to ionicity 0
    with pytest.warns(UserWarning, match="closest ionicity 0"):
        atom_dysprosium.ionicity = 1
    atom_dysprosium.magnetization = {"r": 0.0, "phi": 0.0 * u.deg, "gamma": 0.0 * u.deg}
    return atom_dysprosium


@pytest.fixture(scope="module")
def atom_mixed(atom_dysprosium, atom_iron):
    atom_mixed = AtomMixed("DyFe")
    atom_mixed.add_atom(atom_dysprosium, 0.4)
    atom_mixed.add_atom(atom_iron, 0.6)
    atom_mixed.magnetization = {"r": 0.0, "phi": 0.0 * u.deg, "gamma": 0.0 * u.deg}
    return atom_mixed


# layers


@pytest.fixture(scope="module")
def properties():
    props = {
        "roughness": 0.5 * u.angstrom,
        "deb_wal_fac": 1 * u.angstrom**2,
        "sound_vel": 6 * u.nm / u.ps,
        "phonon_damping": 1 * u.kg / u.s,
        "dielectric": Dispersion.from_ref_index(REF_INDEX),
        "therm_cond": 1 * u.W / (u.m * u.K),
        "lin_therm_exp": 1e-5,
        "heat_capacity": 10 * (u.J / u.kg / u.K),
        "sub_system_coupling": [0],
        "eff_spin": 1,
        "curie_temp": 100 * u.K,
        "lamda": 1,
        "mag_moment": 1 * u.bohr_magneton,
        "aniso_exponent": 3,
        "anisotropy": [1, 2, 3] * u.J / u.m**3,
        "exch_stiffness": 1 * u.J / u.m,
        "mag_saturation": 1 * u.J / u.T / u.m**3,
    }
    return props


@pytest.fixture(scope="module")
def layer(properties):
    return Layer(id="layer", name="base layer", **properties)


@pytest.fixture(scope="module")
def vacuum(properties):
    return Vacuum()


@pytest.fixture(scope="module")
def amorphous_layer(atom_iron, properties):
    return AmorphousLayer(
        id="amorphous_layer",
        name="amorphous layer",
        thickness=1 * u.nm,
        density=5000 * u.kg / u.m**3,
        atom=atom_iron,
        **properties,
    )


@pytest.fixture(scope="module")
def unit_cell(atom_strontium, atom_oxygen, atom_titanium, properties):
    uc = UnitCell(id="unit_cell", name="unit_cell", c_axis=5.0 * u.angstrom, **properties)
    uc.add_atom(atom_strontium, 0.0)
    uc.add_atom(atom_oxygen, 0.5)
    uc.add_atom(atom_titanium, 1.0)
    return uc


@pytest.fixture(scope="module")
def amorphous_layer_iron(atom_iron, properties):
    layer = AmorphousLayer(
        id="amorphous_layer_Fe",
        name="amorphous layer iron",
        thickness=1 * u.nm,
        density=5000 * u.kg / u.m**3,
        atom=atom_iron,
        **properties,
    )
    return _dielectric_from_atoms_and_optics(layer)


@pytest.fixture(scope="module")
def amorphous_layer_oxygen(atom_oxygen, properties):
    layer = AmorphousLayer(
        id="amorphous_layer_O",
        name="amorphous layer oxygen",
        thickness=1 * u.nm,
        density=5000 * u.kg / u.m**3,
        atom=atom_oxygen,
        **properties,
    )
    return _dielectric_from_atoms_and_optics(layer)


@pytest.fixture(scope="module")
def unit_cell_iron(atom_iron, properties):
    uc = UnitCell(id="unit_cell_Fe", name="unit cell iron", c_axis=5.0 * u.angstrom, **properties)
    uc.add_atom(atom_iron, 0.0)
    return uc


@pytest.fixture(scope="module")
def unit_cell_oxygen(atom_oxygen, properties):
    uc = UnitCell(id="unit_cell_O", name="unit cell oxygen", c_axis=5.0 * u.angstrom, **properties)
    uc.add_atom(atom_oxygen, 0.0)
    return uc


# structure


@pytest.fixture(scope="module")
def structure(amorphous_layer_iron, amorphous_layer_oxygen, unit_cell_iron, unit_cell_oxygen):
    S = Structure("structure mixed")
    S.add_sub_structure(amorphous_layer_iron, 10)
    S.add_sub_structure(amorphous_layer_oxygen, 10)
    S.add_sub_structure(unit_cell_iron, 20)

    DL = Structure("double layer")
    DL.add_sub_structure(unit_cell_iron, 5)
    DL.add_sub_structure(unit_cell_oxygen, 7)

    S.add_sub_structure(DL, 20)

    S.add_substrate(amorphous_layer_iron, 100)

    return S


@pytest.fixture(scope="module")
def structure_amorph(amorphous_layer_iron, amorphous_layer_oxygen):
    S = Structure("structure amorph")
    S.add_sub_structure(amorphous_layer_iron, 10)
    S.add_sub_structure(amorphous_layer_oxygen, 10)

    S.add_substrate(amorphous_layer_iron, 100)

    return S


@pytest.fixture(scope="module")
def structure_crystalline(unit_cell_iron, unit_cell_oxygen):
    S = Structure("structure crystalline")
    S.add_sub_structure(unit_cell_iron, 10)
    S.add_sub_structure(unit_cell_oxygen, 10)

    S.add_substrate(unit_cell_iron, 100)

    return S
