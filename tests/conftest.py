#!/usr/bin/env python

import pytest

from udkm1Dsim import (
    Atom,
    AtomMixed,
    Dielectric,
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


@pytest.fixture
def agg_backend():
    """Non-interactive backend of matplotlib for the visualizations, which
    neither needs a display nor Tcl/Tk."""
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    yield
    plt.close("all")


# parameters


# refractive index of the layers for the optical excitation
REF_INDEX = 5 + 3j  # n + i kappa with kappa > 0 for an absorbing layer


def _dielectric_from_atoms_and_optics(layer):
    """Dielectric from the atoms in the X-ray range and a constant refractive
    index at all other energies, e.g. for the optical excitation."""
    layer.set_dielectric_from_atoms(fallback=Dispersion.from_ref_index(REF_INDEX))
    return layer


@pytest.fixture
def parameter():
    return Parameter("m", 0, "parameter")


# temperature parameter


@pytest.fixture
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
        # small enough to keep PhononNum fast: the phonon damping between two
        # layers limits the time step of its solver by about 4 * damping / mass,
        # i.e. 4e12 1/s for the lightest layers of 1e-27 kg, which is well below
        # their highest phonon frequency of about 4e13 1/s
        "phonon_damping": 1e-15 * u.kg / u.s,
        "dielectric": Dispersion.from_ref_index(REF_INDEX),
        "therm_cond": 1 * u.W / (u.m * u.K),
        "lin_therm_exp": 1e-5,
        "heat_capacity": 500 * (u.J / u.kg / u.K),
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
    """an amorphous layer with the magnetization of the iron atom"""
    return Layer(
        id="amorphous_layer",
        name="amorphous layer",
        thickness=1 * u.nm,
        density=5000 * u.kg / u.m**3,
        magnetization=atom_iron.atomic.magnetization,
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
    """an amorphous layer of iron with the dielectric and magnetization of the atom"""
    return Layer(
        id="amorphous_layer_Fe",
        name="amorphous layer iron",
        thickness=1 * u.nm,
        density=5000 * u.kg / u.m**3,
        magnetization=atom_iron.atomic.magnetization,
        **{
            **properties,
            "dielectric": Dielectric.from_atoms(
                atom_iron, fallback=Dispersion.from_ref_index(REF_INDEX)
            ),
        },
    )


@pytest.fixture(scope="module")
def amorphous_layer_oxygen(atom_oxygen, properties):
    """an amorphous layer of oxygen with the dielectric and magnetization of the atom"""
    return Layer(
        id="amorphous_layer_O",
        name="amorphous layer oxygen",
        thickness=1 * u.nm,
        density=5000 * u.kg / u.m**3,
        magnetization=atom_oxygen.atomic.magnetization,
        **{
            **properties,
            "dielectric": Dielectric.from_atoms(
                atom_oxygen, fallback=Dispersion.from_ref_index(REF_INDEX)
            ),
        },
    )


@pytest.fixture(scope="module")
def unit_cell_iron(atom_iron, properties):
    """a unit cell of iron with the dielectric and magnetization of the atom"""
    uc = UnitCell(
        id="unit_cell_Fe",
        name="unit cell iron",
        c_axis=5.0 * u.angstrom,
        magnetization=atom_iron.atomic.magnetization,
        **properties,
    )
    uc.add_atom(atom_iron, 0.0)
    return _dielectric_from_atoms_and_optics(uc)


@pytest.fixture(scope="module")
def unit_cell_oxygen(atom_oxygen, properties):
    uc = UnitCell(id="unit_cell_O", name="unit cell oxygen", c_axis=5.0 * u.angstrom, **properties)
    uc.add_atom(atom_oxygen, 0.0)
    return _dielectric_from_atoms_and_optics(uc)


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


# stacks for the optical and X-ray scattering and the absorption

# refractive index of the substrate of the stacks of the multilayer formalisms
N_SUB = 1.5 + 0.1j


def substrate_layer(eps0=None, euler_angles=(0, 0, 0) * u.deg):
    """substrate of the refractive index N_SUB or a given dispersion"""
    sub = Layer("sub", "substrate", thickness=0 * u.nm, density=1000 * u.kg / u.m**3)
    eps0 = Dispersion.from_ref_index(N_SUB) if eps0 is None else eps0
    sub.optical.dielectric = Dielectric(eps0=eps0, euler_angles=euler_angles)
    return sub


def interface(substrate):
    """a single interface of vacuum and the substrate"""
    S = Structure("interface")
    S.add_sub_structure(Vacuum(thickness=0 * u.nm), 1)
    S.add_substrate(substrate)
    return S


def optical_stack(substrate=None, thickness=1):
    """non-magnetic optical stack of a metal and a dielectric, whose
    thicknesses are scaled by ``thickness``"""
    a = Layer("a", "metal", thickness=3 * thickness * u.nm, density=7874 * u.kg / u.m**3)
    a.optical.dielectric = Dispersion.from_ref_index(2.9 + 3.4j)
    b = Layer("b", "oxide", thickness=2 * thickness * u.nm, density=2500 * u.kg / u.m**3)
    b.optical.dielectric = Dispersion.from_ref_index(1.45)
    S = Structure("optical stack")
    S.add_sub_structure(a, 2)
    S.add_sub_structure(b, 3)
    S.add_sub_structure(a, 1)
    S.add_substrate(Vacuum() if substrate is None else substrate)
    return S


def magnetic_film(magnetization, num_layers=1, thickness=10 * u.nm, **kwargs):
    """Fe film with a gyrotropic dielectric at 800 nm of ``num_layers``
    layers of the ``thickness``"""
    fe = Layer("fe", "fe", thickness=thickness, density=7874 * u.kg / u.m**3, **kwargs)
    fe.optical.dielectric = Dielectric(
        eps0=Dispersion.from_ref_index(2.9 + 3.4j),
        eps_mag=Dispersion.from_array([0, 10], [0.6 + 0.2j] * 2, kind="eps"),
    )
    fe.magnetization = magnetization
    S = Structure("film")
    S.add_sub_structure(fe, num_layers)
    S.add_substrate(Vacuum())
    return S


def optical_layer(id, n, thickness=1 * u.nm):
    """layer of a constant refractive index for the optical excitation"""
    return Layer(
        id, id, thickness=thickness, dielectric=Dispersion.from_ref_index(n), therm_cond=1
    )


def absorbing_stack(substrate_index=3.7 + 0.01j):
    """stack of an absorber and a dielectric on an absorbing substrate"""
    absorber = optical_layer("absorber", 3.5 + 4.0j)
    S = Structure("stack")
    S.add_sub_structure(absorber, 20)
    S.add_sub_structure(optical_layer("dielectric", 1.5 + 0.01j), 50)
    S.add_sub_structure(absorber, 10)
    S.add_substrate(optical_layer("substrate", substrate_index, 0 * u.nm))
    return S


def fe_layer(thickness=2 * u.nm, **kwargs):
    """Fe layer with the dielectric of its atoms for soft and hard X-rays"""
    return Layer(
        "fe",
        "fe",
        thickness=thickness,
        density=7874 * u.kg / u.m**3,
        dielectric=Dielectric.from_atoms(Atom("Fe")),
        **kwargs,
    )


def oxide_layer():
    """O layer with the dielectric of its atoms"""
    return Layer(
        "ox",
        "ox",
        thickness=1.3 * u.nm,
        density=3000 * u.kg / u.m**3,
        dielectric=Dielectric.from_atoms(Atom("O")),
    )


@pytest.fixture(scope="module")
def magnetic_multilayer():
    """Fe/O multilayer at the Fe L3 edge with an in-plane magnetization"""
    fe = fe_layer(magnetization={"r": 1, "phi": 90 * u.deg, "gamma": 30 * u.deg})
    ox = oxide_layer()
    S = Structure("magnetic multilayer")
    S.add_sub_structure(fe, 3)
    S.add_sub_structure(ox, 2)
    S.add_sub_structure(fe, 2)
    S.add_substrate(ox)
    return S


@pytest.fixture(scope="module")
def periodic_multilayer():
    """5 periods of rough Fe and O on a substrate of O for X-rays"""
    ox = oxide_layer()
    S = Structure("multilayer")
    S.add_sub_structure(fe_layer(roughness=0.3 * u.nm), 1)
    S.add_sub_structure(ox, 1)
    ML = Structure("periods")
    ML.add_sub_structure(S, 5)
    ML.add_substrate(ox)
    return ML


# samples of the heat, phonon, and magnetization simulations


@pytest.fixture(scope="module")
def structure_freestanding(amorphous_layer_iron):
    """a freestanding layer of 50 nm, whose backside reflects strain waves"""
    S = Structure("freestanding layer")
    S.add_sub_structure(amorphous_layer_iron, 50)
    return S


def slab(therm_cond=10, num_layers=10, heat_capacity=500, **kwargs):
    """slab of 1 nm layers with constant thermal properties and the
    absorbing optical constants of a metal at 800 nm"""
    layer = Layer(
        "a",
        "a",
        thickness=1 * u.nm,
        density=1000 * u.kg / u.m**3,
        heat_capacity=heat_capacity,
        therm_cond=therm_cond,
        dielectric=Dispersion.from_ref_index(3.5 + 4.0j),
        **kwargs,
    )
    S = Structure("slab")
    S.add_sub_structure(layer, num_layers)
    return S


def magnetic_layer(id, **overrides):
    """a magnetic layer of 1 nm with sensible defaults, magnetized along x,
    unlike the fixtures above with a Curie temperature of 100 K"""
    props = {
        "heat_capacity": 1,
        "therm_cond": 0,
        "lin_therm_exp": 0,
        "sub_system_coupling": 0,
        "eff_spin": 0.5,
        "curie_temp": 630 * u.K,
        "lamda": 0.005,
        "mag_moment": 0.393 * u.bohr_magneton,
        "aniso_exponent": 2,
        "anisotropy": [0, 0, 0] * u.J / u.m**3,
        "exch_stiffness": 0 * u.J / u.m,
        "mag_saturation": 5e5 * u.J / u.T / u.m**3,
    }
    props.update(overrides)
    props.setdefault("magnetization", (1, 0, 0))
    density = props.pop("density", 7000 * u.kg / u.m**3)
    return Layer(id, id, thickness=1 * u.nm, density=density, **props)


@pytest.fixture(scope="module")
def structure_magnetic():
    """a Ni-like film with all parameters of the LLG, iLLG, and LLB"""
    layer = magnetic_layer(
        "Ni",
        density=8900 * u.kg / u.m**3,
        heat_capacity=444 * u.J / (u.kg * u.K),
        therm_cond=90 * u.W / (u.m * u.K),
        lin_therm_exp=1.3e-5,
        lamda=0.01,
        mag_moment=0.6 * u.bohr_magneton,
        mag_saturation=4.9e5 * u.A / u.m,
        gilbert_damping=0.1,
        inertial_time=1 * u.fs,
        anisotropy=[0, 0, 1e5] * u.J / u.m**3,
        exch_stiffness=1e-11 * u.J / u.m,
    )
    S = Structure("Ni film")
    S.add_sub_structure(layer, 20)
    return S
