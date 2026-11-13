#!/usr/bin/env python

import copy
import pickle
import warnings

import numpy as np
import pint
import pytest
import sympy as sp
from pint.testing import assert_allclose as assert_approx
from pint.testing import assert_equal

from udkm1Dsim import AmorphousLayer, Atom, AtomMixed, Dielectric, Dispersion, Layer, UnitCell, u
from udkm1Dsim.structures.parameter_groups import (
    ElasticParameters,
    LatticeParameters,
    MagneticParameters,
    StructuralParameters,
    ThermalParameters,
)
from udkm1Dsim.structures.parameters import Parameter, TemperatureParameter, VectorParameter

# tests

# Layer


def test_layer_id(layer):
    assert layer.id == "layer"


def test_layer_name(layer):
    assert layer.name == "base layer"


# ============================================================================
# Structural parameters
# ============================================================================


def test_layer_roughness(layer):
    assert_equal(layer.roughness, 0.5 * u.angstrom)


# ============================================================================
# Thermal parameters
# ============================================================================


def test_layer_therm_cond(layer):
    assert layer.therm_cond[0](300) == 1


def test_layer_therm_cond_expr(layer):
    assert float(layer.therm_cond_expr[0]) == pytest.approx(1.0, rel=1e-9, abs=0)


def test_layer_lin_therm_exp(layer):
    assert layer.lin_therm_exp[0](300) == 1e-5


def test_layer_lin_therm_exp_expr(layer):
    assert float(layer.lin_therm_exp_expr[0]) == pytest.approx(1e-05, rel=1e-9, abs=0)


def test_layer_int_lin_therm_exp(layer):
    assert layer.int_lin_therm_exp[0](300) == 3e-3


def test_layer_int_lin_therm_exp_expr(layer):
    T = sp.symbols("T")
    assert layer.int_lin_therm_exp_expr[0].equals(1.0e-5 * T)


def test_layer_nonintegratable_lin_therm_exp(layer):
    with pytest.warns(UserWarning):
        layer.lin_therm_exp = "1/abs(T-300)"
        # integration is only carried out when explicitly called
        layer.int_lin_therm_exp
    # reset to original value
    layer.lin_therm_exp = 1e-5


def test_layer_heat_capacity(layer):
    assert layer.heat_capacity[0](300) == 500


def test_layer_heat_capacity_expr(layer):
    assert float(layer.heat_capacity_expr[0]) == pytest.approx(500.0, rel=1e-9, abs=0)


def test_layer_int_heat_capacity(layer):
    assert layer.int_heat_capacity[0](300) == 150000.0


def test_layer_int_heat_capacity_expr(layer):
    T = sp.symbols("T")
    assert layer.int_heat_capacity_expr[0].equals(500.0 * T)


def test_layer_nonintegratable_heat_capacity(layer):
    with pytest.warns(UserWarning):
        layer.heat_capacity = "1/abs(T-300)"
        # integration is only carried out when explicitly called
        layer.int_heat_capacity
    # reset to original value
    layer.heat_capacity = 500 * (u.J / u.kg / u.K)


def test_layer_sub_system_coupling(layer):
    assert layer.sub_system_coupling[0](0) == 0


def test_layer_num_subsystems(layer):
    assert layer.num_sub_systems == 1


def test_layer_switch_num_sub_systems_silently(layer, capsys):
    layer = copy.deepcopy(layer)  # do not alter the module-scoped fixture
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # the parameters are inconsistent in between, which must not be reported
        layer.therm_cond = [0, 5 * u.W / (u.m * u.K)]
        assert layer.num_sub_systems == 2
        layer.heat_capacity = ["0.1*T", 500]
        layer.sub_system_coupling = ["5e17*(T_1-T_0)", "5e17*(T_0-T_1)"]
    assert capsys.readouterr().out == ""
    assert layer.num_sub_systems == 2


def test_layer_deb_wal_fac(layer):
    # asserat_equal does not work on because of last digit uncertainty
    assert_approx(layer.deb_wal_fac[0](300), 1e-20)


def test_layer_deb_wal_fac_expr(layer):
    assert float(layer.deb_wal_fac_expr[0]) == pytest.approx(1e-20, rel=1e-9, abs=0)


# ============================================================================
# Elastic parameters
# ============================================================================


def test_layer_set_deb_wal_fac(layer):
    layer = copy.deepcopy(layer)  # do not alter the module-scoped fixture
    layer.deb_wal_fac = 2e-20 * u.m**2
    assert_approx(layer.deb_wal_fac[0](300), 2e-20)


def test_layer_sound_vel(layer):
    assert_equal(layer.sound_vel, 6 * u.nm / u.ps)


def test_layer_phonon_damping(layer):
    assert_equal(layer.phonon_damping, 1e-15 * u.kg / u.s)


def test_layer_set_ho_spring_constants(layer):
    assert layer.spring_const == 0.0
    layer.set_ho_spring_constants([1.0, 2.0, 3.0])
    assert np.allclose(layer.spring_const, [0.0, 1.0, 2.0, 3.0] * u.kg / u.s**2)


# ============================================================================
# Optical parameters
# ============================================================================


def test_layer_removed_optical_inputs():
    for removed in ("opt_ref_index", "opt_pen_depth", "opt_ref_index_per_strain"):
        with pytest.raises(TypeError, match=f"'{removed}' was removed"):
            Layer("l", "l", **{removed: 1})
        with pytest.raises(TypeError, match=f"'{removed}' was removed"):
            Layer("l", "l", optical={removed: 1})


def test_layer_dielectric_from_ref_index(layer):
    assert isinstance(layer.dielectric, Dielectric)
    assert layer.dielectric.is_isotropic
    np.testing.assert_allclose(layer.get_ref_index(800 * u.nm), 5 + 3j)
    np.testing.assert_allclose(layer.get_dielectric_tensor(1.0)[0], (5 + 3j) ** 2 * np.eye(3))


# ============================================================================
# Magnetic parameters
# ============================================================================


def test_layer_eff_spin(layer):
    assert layer.eff_spin == 1


def test_layer_curie_temp(layer):
    assert_equal(layer.curie_temp, 100 * u.K)


def test_layer_mf_exch_coupling(layer):
    assert_approx(layer.mf_exch_coupling, 2.07e-21 * u.m**2 * u.kg / u.s**2, rtol=1e-2)


def test_layer_lamda(layer):
    assert layer.lamda == 1


def test_layer_mag_moment(layer):
    # asserat_equal does not work on because of last digit uncertainty
    assert_approx(layer.mag_moment, 1 * u.bohr_magneton, rtol=1e-2)


def test_layer_set_magnetic_parameters_after_creation(layer):
    layer = copy.deepcopy(layer)  # do not alter the module-scoped fixture
    layer.curie_temp = 200 * u.K
    layer.eff_spin = 2
    layer.anisotropy = 5 * u.J / u.m**3
    layer.exch_stiffness = [1, 2, 3] * u.J / u.m
    assert_approx(layer.mf_exch_coupling, 3 * 2 / 3 * u.boltzmann_constant * 200 * u.K)
    assert_equal(layer.anisotropy, [5, 5, 5] * u.J / u.m**3)
    assert_equal(layer.exch_stiffness, [1, 2, 3] * u.J / u.m)


def test_layer_aniso_exponent(layer):
    assert layer.aniso_exponent == 3


def test_layer_anisotropy(layer):
    assert_equal(layer.anisotropy, [1, 2, 3] * u.J / u.m**3)


def test_layer_exch_stiffness(layer):
    assert_equal(layer.exch_stiffness, [1, 1, 1] * u.J / u.m)


def test_layer_mag_saturation(layer):
    assert_equal(layer.mag_saturation, 1 * u.J / u.T / u.m**3)


def test_layer_exch_relax_rate_and_concentration(layer):
    layer = copy.deepcopy(layer)  # do not alter the module-scoped fixture
    layer.exch_relax_rate = 0.5
    layer.concentration = 0.25
    assert_approx(layer.exch_relax_rate, 0.5 * u.dimensionless)
    assert_approx(layer.concentration, 0.25 * u.dimensionless)
    assert layer.magnetic.exch_relax_rate.magnitude == 0.5
    assert layer.magnetic.concentration.magnitude == 0.25


def test_layer_dielectric_tensor_of_sublattices_needs_magnetization():
    layer = Layer(
        "AFM",
        "AFM",
        eff_spin=[0.5, 0.5],
        magnetization=[(1, 0, 0), (-1, 0, 0)],
        dielectric=Dispersion.from_ref_index(1.5),
    )
    with pytest.raises(NotImplementedError, match="several sublattices"):
        layer.get_dielectric_tensor(1.55 * u.eV)
    tensor = layer.get_dielectric_tensor(1.55 * u.eV, magnetization=(0, 0, 0))
    assert np.allclose(tensor, 1.5**2 * np.eye(3))


def test_layer_get_property_dict(layer):
    assert layer.get_property_dict(types="all") == vars(layer)
    for types in ["heat", "phonon", "xray", "optical", "magnetic"]:
        props = layer.get_property_dict(types=types)
        assert props, f"no properties for types='{types}'"
        assert all(isinstance(p, Parameter) for p in props.values())
    props = layer.get_property_dict(types=["heat", "magnetic"])
    assert props["structural.thickness"] is layer.structural.thickness
    assert props["magnetic.curie_temp"] is layer.magnetic.curie_temp


def test_layer_mag_moment_in_si(layer):
    assert layer.magnetic.mag_moment.unit == u.A * u.m**2
    assert np.isclose(layer.magnetic.mag_moment.magnitude, 9.274e-24, rtol=1e-3)


def test_layer_get_opt_pen_depth(layer):
    # from the refractive index
    assert_approx(layer.get_opt_pen_depth(800 * u.nm), 21.22 * u.nm, rtol=1e-3)
    layer = Layer("opt", "opt", dielectric=Dispersion.from_ref_index(2.0))
    assert np.isinf(layer.get_opt_pen_depth(800 * u.nm).magnitude)
    with pytest.raises(ValueError, match="Layer 'empty'.*not set"):
        Layer("empty", "empty").get_opt_pen_depth(800 * u.nm)


def test_layer_set_opt_pen_depth():
    # without a dielectric, the real part of the refractive index is 1
    layer = Layer("opt", "opt")
    with pytest.warns(UserWarning, match="only changes the extinction coefficient"):
        layer.set_opt_pen_depth(20 * u.nm)
    for wavelength in (400, 800) * u.nm:
        assert_approx(layer.get_opt_pen_depth(wavelength), 20 * u.nm)
    assert layer.get_ref_index(800 * u.nm).real == pytest.approx(1)
    # the real part of the dielectric is kept
    layer.optical.dielectric = Dispersion.from_ref_index(3 + 4j)
    with pytest.warns(UserWarning):
        layer.set_opt_pen_depth(10 * u.nm)
    assert layer.get_ref_index(800 * u.nm).real == pytest.approx(3)
    assert_approx(layer.get_opt_pen_depth(400 * u.nm), 10 * u.nm)
    # a new penetration depth replaces the former one
    with pytest.warns(UserWarning):
        layer.set_opt_pen_depth(5 * u.nm)
    assert_approx(layer.get_opt_pen_depth(800 * u.nm), 5 * u.nm)
    assert layer.get_ref_index(800 * u.nm).real == pytest.approx(3)
    assert "ζ = 5 nm" in str(layer.optical.dielectric)
    # anisotropic dielectrics have no single penetration depth
    layer.optical.dielectric = Dispersion.from_ref_index([2, 3])
    with pytest.raises(ValueError, match="isotropic"):
        layer.set_opt_pen_depth(5 * u.nm)


# ============================================================================
# Structural & elastic parameters and their dependencies
# ============================================================================


def test_layer_structural_parameters(amorphous_layer):
    assert_approx(amorphous_layer.thickness, 1 * u.nm)
    assert_approx(amorphous_layer.density, 5000 * u.kg / u.m**3)
    assert_approx(amorphous_layer.area, 1 * u.angstrom**2)
    assert_approx(amorphous_layer.volume, 1e-29 * u.m**3)
    assert_approx(amorphous_layer.mass, 5e-26 * u.kg)
    assert_approx(amorphous_layer.mass_unit_area, 5e-26 * u.kg)


def test_layer_spring_const_and_acoustic_impedance(amorphous_layer):
    # k = m / A * 1 Å² * (v / d)²
    assert_approx(amorphous_layer.spring_const, [1.8] * u.kg / u.s**2)
    # Z = rho * v
    assert_approx(amorphous_layer.acoustic_impedance, 3e7 * u.kg / u.m**2 / u.s)
    assert (
        amorphous_layer.acoustic_impedance == amorphous_layer.elastic.acoustic_impedance.quantity
    )


@pytest.mark.parametrize(
    "prop",
    [
        "mass",
        "mass_unit_area",
        "area",
        "volume",
        "spring_const",
        "acoustic_impedance",
        "num_sub_systems",
        "int_lin_therm_exp",
        "int_heat_capacity",
    ],
)
def test_layer_derived_parameters_are_read_only(layer, prop):
    with pytest.raises(AttributeError):
        setattr(layer, prop, 1)


@pytest.mark.parametrize(
    "prop, value, mass, spring_const, acoustic_impedance",
    [
        ("thickness", 2 * u.nm, 1e-25, 0.9, 3e7),
        ("density", 10000 * u.kg / u.m**3, 1e-25, 3.6, 6e7),
        ("sound_vel", 3 * u.nm / u.ps, 5e-26, 0.45, 1.5e7),
    ],
)
def test_layer_setters_update_depending_parameters(
    amorphous_layer, prop, value, mass, spring_const, acoustic_impedance
):
    layer = copy.deepcopy(amorphous_layer)  # do not alter the module-scoped fixture
    setattr(layer, prop, value)
    assert_approx(layer.mass, mass * u.kg)
    assert_approx(layer.spring_const, [spring_const] * u.kg / u.s**2)
    assert_approx(layer.acoustic_impedance, acoustic_impedance * u.kg / u.m**2 / u.s)


def test_layer_magnetization(layer):
    layer = copy.deepcopy(layer)  # do not alter the module-scoped fixture
    assert np.allclose(layer.magnetization.magnitude, [0, 0, 0])
    layer.magnetization = [1, 0, 0]
    assert np.allclose(layer.magnetic.magnetization.magnitude, [1, 0, 0])


# ============================================================================
# Representations
# ============================================================================


@pytest.mark.parametrize(
    "fixture_name, header",
    [
        ("layer", "Layer: base layer"),
        ("vacuum", "Vacuum layer of thickness: 1 nm"),
        ("amorphous_layer", "AmorphousLayer: amorphous layer"),
        ("unit_cell", "UnitCell: unit_cell"),
    ],
)
def test_layer_repr(request, fixture_name, header):
    layer = request.getfixturevalue(fixture_name)
    assert repr(layer).startswith(header)
    assert str(layer) == repr(layer)
    assert layer._repr_html_().startswith("<h2>")


def test_layer_repr_contains_parameter_groups(unit_cell):
    text = repr(unit_cell)
    html = unit_cell._repr_html_()
    for group in ["Structural", "Lattice", "Thermal", "Elastic", "Optical", "Magnetic"]:
        assert f"{group} Parameters" in text
        assert f"<h3>{group} Parameters</h3>" in html
    assert "<h3>Atoms</h3>" in html
    assert "Strontium" in text and "Strontium" in html


# ============================================================================
# Vacuum
# ============================================================================


def test_vacuum(vacuum):
    assert_approx(vacuum.thickness, 1 * u.nm)
    assert_approx(vacuum.density, 0 * u.kg / u.m**3)
    assert_approx(vacuum.mass, 0 * u.kg)
    assert vacuum.get_ref_index(800 * u.nm) == 1


# ============================================================================
# AmorphousLayer
# ============================================================================


def test_amorphous_layer_atom(amorphous_layer, atom_iron):
    assert amorphous_layer.atom is atom_iron


def test_amorphous_layer_without_atom():
    layer = AmorphousLayer("a", "a", 1 * u.nm, 1000 * u.kg / u.m**3)
    assert np.allclose(layer.magnetization.magnitude, [0, 0, 0])


def test_amorphous_layer_invalid_atom():
    with pytest.raises(TypeError):
        AmorphousLayer("a", "a", 1 * u.nm, 1000 * u.kg / u.m**3, atom="Fe")


# ============================================================================
# UnitCell
# ============================================================================


def test_unit_cell_lattice_parameters(unit_cell):
    assert_approx(unit_cell.a_axis, 5 * u.angstrom)
    assert_approx(unit_cell.b_axis, 5 * u.angstrom)
    assert_approx(unit_cell.c_axis, 5 * u.angstrom)
    assert_approx(unit_cell.thickness, 5 * u.angstrom)
    assert_approx(unit_cell.area, 25 * u.angstrom**2)
    assert_approx(unit_cell.volume, 125 * u.angstrom**3)


def test_unit_cell_mass_from_atoms(unit_cell, atom_strontium, atom_oxygen, atom_titanium):
    mass = atom_strontium.mass + atom_oxygen.mass + atom_titanium.mass
    assert_approx(unit_cell.mass, mass)
    assert_approx(unit_cell.density, mass / unit_cell.volume)
    assert_approx(unit_cell.mass_unit_area, mass * u.angstrom**2 / unit_cell.area)


def test_unit_cell_spring_const_and_acoustic_impedance(unit_cell):
    assert_approx(unit_cell.spring_const, [1.449] * u.kg / u.s**2, rtol=1e-3)
    assert_approx(
        unit_cell.elastic.acoustic_impedance.quantity,
        2.415e6 * u.kg / u.m**2 / u.s,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "axis, density_factor, spring_const_factor",
    [("a_axis", 0.5, 0.5), ("b_axis", 0.5, 0.5), ("c_axis", 0.5, 0.25)],
)
def test_unit_cell_axis_setter_keeps_mass(unit_cell, axis, density_factor, spring_const_factor):
    uc = copy.deepcopy(unit_cell)  # do not alter the module-scoped fixture
    setattr(uc, axis, 10 * u.angstrom)
    assert_approx(uc.mass, unit_cell.mass)
    assert_approx(uc.volume, 2 * unit_cell.volume)
    assert_approx(uc.density, density_factor * unit_cell.density)
    assert_approx(uc.spring_const, spring_const_factor * unit_cell.spring_const)


def test_unit_cell_atoms(unit_cell):
    assert unit_cell.num_atoms == 3
    assert unit_cell.atom_ids == ["Sr", "O", "Ti"]
    assert np.allclose(unit_cell.get_atom_positions(), [0, 0.5, 1])
    assert np.allclose(unit_cell.get_atom_positions(0.1), [0, 0.55, 1.1])


def test_unit_cell_without_atoms_has_no_dielectric_from_atoms():
    uc = UnitCell("empty", "empty", c_axis=1 * u.nm)
    assert uc.atom_ids == []
    with pytest.raises(TypeError, match="has no atoms"):
        uc.dielectric_from_atoms()


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_unit_cell_visualize(unit_cell, atom_oxygen, agg_backend):
    uc = copy.deepcopy(unit_cell)  # do not alter the module-scoped fixture
    # a repeated atom is labeled only once
    uc.add_atom(atom_oxygen, 0.75)
    uc.visualize(block=False)
    uc.visualize(block=False, strain=0.1)


def test_unit_cell_add_multiple_atoms(unit_cell, atom_oxygen):
    uc = copy.deepcopy(unit_cell)  # do not alter the module-scoped fixture
    uc.add_multiple_atoms(atom_oxygen, 0.5, 2)
    assert uc.num_atoms == 5
    assert uc.atom_ids == ["Sr", "O", "Ti"]  # unique ids
    assert np.allclose(uc.get_atom_positions(), [0, 0.5, 0.5, 0.5, 1])
    assert_approx(uc.mass, unit_cell.mass + 2 * atom_oxygen.mass)


def test_unit_cell_strain_dependent_position(atom_oxygen, atom_strontium):
    uc = UnitCell("uc", "uc", 5 * u.angstrom)
    uc.add_atom(atom_oxygen, "0.1*(s**2+1)")
    uc.add_atom(atom_strontium, 0.5)
    assert uc.atoms[0][2] == "0.1*(s**2+1)"
    assert np.allclose(uc.get_atom_positions(), [0.1, 0.5])
    assert np.allclose(uc.get_atom_positions(1), [0.2, 1.0])


def test_unit_cell_position_as_function_raises(atom_oxygen):
    uc = UnitCell("uc", "uc", 5 * u.angstrom)
    with pytest.raises(ValueError):
        uc.add_atom(atom_oxygen, lambda s: s)


# ============================================================================
# ParameterGroups as input and their connection to the layer
# ============================================================================


def _magnitudes(layer):
    """all parameter magnitudes of a layer by their path"""
    return {
        f"{group_name}.{name}": repr(np.asarray(p.magnitude).tolist())
        for group_name in layer._parameter_group_names
        for name, p in getattr(layer, group_name)._parameters().items()
        if name not in ("heat_capacity", "therm_cond")  # sympy expressions
    }


def test_layer_init_with_parameter_groups_or_dicts():
    single = Layer(
        "a",
        "a",
        thickness=2 * u.nm,
        density=5000 * u.kg / u.m**3,
        sound_vel=6 * u.nm / u.ps,
        heat_capacity=[10, 20],
    )
    groups = Layer(
        "a",
        "a",
        structural=StructuralParameters(thickness=2 * u.nm, density=5000 * u.kg / u.m**3),
        elastic=ElasticParameters(sound_vel=6 * u.nm / u.ps),
        thermal=ThermalParameters(heat_capacity=[10, 20]),
    )
    dicts = Layer(
        "a",
        "a",
        structural={"thickness": 2 * u.nm},
        density=5000 * u.kg / u.m**3,  # single parameters can complement a dict
        elastic={"sound_vel": 6 * u.nm / u.ps},
        thermal={"heat_capacity": [10, 20]},
    )
    assert _magnitudes(groups) == _magnitudes(single)
    assert _magnitudes(dicts) == _magnitudes(single)
    assert groups.num_sub_systems == 2


def test_layer_parameter_groups_are_copied():
    thermal = ThermalParameters(heat_capacity=[10, 20])
    elastic = ElasticParameters(sound_vel=6 * u.nm / u.ps)
    a = Layer("a", "a", thickness=1 * u.nm, density=1000 * u.kg / u.m**3, elastic=elastic)
    b = Layer("b", "b", thickness=1 * u.nm, density=1000 * u.kg / u.m**3, elastic=elastic)
    a.thermal = thermal
    assert a.elastic is not elastic and a.elastic is not b.elastic
    assert a.thermal is not thermal
    assert a.elastic._owner is a and b.elastic._owner is b
    assert elastic._owner is None and thermal._owner is None
    # changing a copy does not change the others
    a.sound_vel = 3 * u.nm / u.ps
    assert_approx(b.sound_vel, 6 * u.nm / u.ps)
    assert_approx(elastic.sound_vel.quantity, 6 * u.nm / u.ps)


@pytest.mark.parametrize(
    "path, value, spring_const",
    [
        ("structural.thickness", 2 * u.nm, 0.9),
        ("structural.density", 10000 * u.kg / u.m**3, 3.6),
        ("elastic.sound_vel", 3 * u.nm / u.ps, 0.45),
    ],
)
def test_layer_group_changes_update_layer(amorphous_layer, path, value, spring_const):
    layer = copy.deepcopy(amorphous_layer)  # do not alter the module-scoped fixture
    group_name, name = path.split(".")
    getattr(getattr(layer, group_name), name).quantity = value
    assert_approx(layer.spring_const, [spring_const] * u.kg / u.s**2)


def test_layer_replace_parameter_group(amorphous_layer):
    layer = copy.deepcopy(amorphous_layer)  # do not alter the module-scoped fixture
    layer.elastic = {"sound_vel": 3 * u.nm / u.ps}
    assert layer.elastic._owner is layer
    assert_approx(layer.spring_const, [0.45] * u.kg / u.s**2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thickness": 1 * u.nm, "structural": {"thickness": 2 * u.nm}},
        {"sound_vel": 1 * u.nm / u.ps, "elastic": ElasticParameters(sound_vel=2 * u.nm / u.ps)},
    ],
)
def test_layer_parameter_given_twice_raises(kwargs):
    with pytest.raises(ValueError, match="at the same time"):
        Layer("a", "a", **kwargs)


def test_layer_parameter_group_typo_raises():
    with pytest.raises(TypeError, match="deb_Wal_Fac"):
        Layer("a", "a", thermal={"deb_Wal_Fac": 1 * u.angstrom**2})


@pytest.mark.parametrize("thermal", [ElasticParameters(), [1, 2]])
def test_layer_parameter_group_wrong_type_raises(thermal):
    with pytest.raises(TypeError, match="must be ThermalParameters"):
        Layer("a", "a", thermal=thermal)


def test_layer_deepcopy_and_pickle_keep_owner(amorphous_layer):
    for layer in (copy.deepcopy(amorphous_layer), pickle.loads(pickle.dumps(amorphous_layer))):
        for group_name in layer._parameter_group_names:
            assert getattr(layer, group_name)._owner is layer
        assert_approx(layer.spring_const, amorphous_layer.spring_const)


def test_parameter_group_deepcopy_does_not_copy_layer(amorphous_layer):
    group = copy.deepcopy(amorphous_layer.thermal)
    assert group._owner is None
    # parameters stay connected to the copied group
    assert group.heat_capacity._caller is group


def test_amorphous_layer_init_with_structural_group():
    layer = AmorphousLayer(
        "a", "a", structural=StructuralParameters(thickness=2 * u.nm, density=1000 * u.kg / u.m**3)
    )
    assert_approx(layer.mass, 2e-26 * u.kg)


def test_unit_cell_init_with_lattice_group():
    lattice = LatticeParameters(c_axis=3 * u.angstrom)
    uc = UnitCell("uc", "uc", lattice=lattice, sound_vel=5 * u.nm / u.ps)
    assert_approx(uc.a_axis, 3 * u.angstrom)  # a and b default to c
    assert_approx(uc.b_axis, 3 * u.angstrom)
    assert_approx(uc.thickness, 3 * u.angstrom)
    assert_approx(uc.area, 9 * u.angstrom**2)


def test_unit_cell_lattice_group_changes_keep_mass(unit_cell):
    uc = copy.deepcopy(unit_cell)  # do not alter the module-scoped fixture
    uc.lattice.a_axis.quantity = 10 * u.angstrom
    assert_approx(uc.mass, unit_cell.mass)
    assert_approx(uc.area, 2 * unit_cell.area)
    assert_approx(uc.density, 0.5 * unit_cell.density)
    assert_approx(uc.spring_const, 0.5 * unit_cell.spring_const)


def test_unit_cell_thickness_is_c_axis(unit_cell):
    uc = copy.deepcopy(unit_cell)  # do not alter the module-scoped fixture
    uc.thickness = 10 * u.angstrom
    assert_approx(uc.c_axis, 10 * u.angstrom)
    assert_approx(uc.structural.thickness.quantity, 10 * u.angstrom)


def test_unit_cell_density_with_atoms_is_read_only(unit_cell):
    with pytest.raises(AttributeError):
        unit_cell.density = 1000 * u.kg / u.m**3
    # without atoms the density can be set
    uc = UnitCell("uc", "uc", 3 * u.angstrom)
    uc.density = 1000 * u.kg / u.m**3
    assert_approx(uc.mass, 1000 * u.kg / u.m**3 * uc.volume)
    uc.add_atom(Atom("Fe"), 0)
    assert_approx(uc.mass, Atom("Fe").mass)


@pytest.mark.parametrize(
    "kwargs",
    [{"thickness": 1 * u.nm}, {"area": 1 * u.nm**2}, {"structural": {"thickness": 1 * u.nm}}],
)
def test_unit_cell_structural_given_by_lattice_raises(kwargs):
    with pytest.raises(ValueError, match="given by its lattice"):
        UnitCell("uc", "uc", 3 * u.angstrom, **kwargs)


def test_unit_cell_without_c_axis_raises():
    with pytest.raises(TypeError, match="c_axis"):
        UnitCell("uc", "uc")


# ============================================================================
# symbols, sub-system labels, and table views
# ============================================================================


@pytest.mark.parametrize(
    "path, symbol",
    [
        ("structural.thickness", "d"),
        ("structural.density", r"\rho"),
        ("thermal.therm_cond", r"\kappa"),
        ("thermal.heat_capacity", "c"),
        ("elastic.spring_const", "k"),
        ("optical.dielectric", r"\varepsilon"),
        ("magnetic.aniso_exponent", r"\beta"),
        ("magnetic.exch_stiffness", r"A_\mathrm{ex}"),
    ],
)
def test_layer_default_symbols(layer, path, symbol):
    group_name, name = path.split(".")
    assert getattr(getattr(layer, group_name), name).symbol == symbol


def test_unit_cell_all_parameters_have_symbols(unit_cell):
    for group_name in unit_cell._parameter_group_names:
        for name, p in getattr(unit_cell, group_name)._parameters().items():
            assert p.symbol, f"{group_name}.{name} has no symbol"


def test_thermal_sub_system_labels():
    thermal = ThermalParameters(heat_capacity=["0.027*T", 449], sub_system_coupling=0)
    assert thermal.sub_system_labels == ["0", "1"]
    assert thermal._display_sub_system_labels() == ["0", "1"]
    thermal.sub_system_labels = ["e", "p"]
    assert str(thermal.heat_capacity).split("\n")[0] == "cₑ(Tₑ) = 0.027·Tₑ J/K/kg"
    # missing labels default to the index and spare labels are ignored
    thermal.sub_system_labels = ["e"]
    assert thermal.sub_system_labels == ["e", "1"]
    thermal.sub_system_labels = ["e", "p", "s"]
    assert thermal.sub_system_labels == ["e", "p"]
    thermal.sub_system_labels = None
    assert thermal.sub_system_labels == ["0", "1"]


def test_thermal_single_sub_system_is_not_labeled():
    thermal = ThermalParameters(heat_capacity=449)
    assert thermal._display_sub_system_labels() is None
    assert str(thermal.heat_capacity) == "c = 449 J/K/kg"
    thermal.sub_system_labels = ["p"]
    assert str(thermal.heat_capacity) == "cₚ = 449 J/K/kg"


def test_layer_sub_system_labels_as_keyword_argument():
    layer = Layer("a", "a", heat_capacity=[1, 2], sub_system_labels=["e", "p"])
    assert layer.thermal.sub_system_labels == ["e", "p"]
    copied = copy.deepcopy(layer)
    assert copied.thermal.sub_system_labels == ["e", "p"]


def test_parameter_group_table_columns_and_rows():
    thermal = ThermalParameters(heat_capacity=["0.027*T", 449], sub_system_labels=["e", "p"])
    text = repr(thermal)
    assert text.startswith("Thermal Parameters\n")
    header = text.split("\n")[2]
    assert [h.strip() for h in header.strip("║").split("║")] == [
        "Parameter",
        "Symbol",
        "Value",
        "Unit",
    ]
    # one row per sub-system, the name and unit only in the first one
    rows = [line for line in text.split("\n") if "cₑ" in line or "cₚ" in line]
    assert "heat_capacity" in rows[0] and "J/K/kg" in rows[0]
    assert "heat_capacity" not in rows[1] and "J/K/kg" not in rows[1]
    html = thermal._repr_html_()
    assert '<th style="text-align: center;">Symbol</th>' in html
    # the name and unit span the rows of the sub-systems
    assert '<td rowspan="2">heat_capacity</td>' in html
    assert '<td rowspan="2">J/(K kg)</td>' in html
    assert '<td style="text-align: center;"><i>c</i><sub><i>e</i></sub></td>' in html
    # the cells are not padded
    assert "  " not in html


def test_layer_symbol_is_editable(amorphous_layer):
    layer = copy.deepcopy(amorphous_layer)  # do not alter the module-scoped fixture
    layer.thermal.heat_capacity.symbol = "C"
    assert "C" in [row[0] for row in layer.thermal.heat_capacity._table_rows("text")]
    # the symbol is copied along with the group
    assert copy.deepcopy(layer).thermal.heat_capacity.symbol == "C"


# assignments to parameters of a group


def test_group_assignment_keeps_parameter_and_updates_depending():
    magnetic = MagneticParameters(eff_spin=0.5, curie_temp=600 * u.K)
    parameter = magnetic.curie_temp
    magnetic.curie_temp = 400 * u.K
    assert magnetic.curie_temp is parameter
    assert magnetic.curie_temp.symbol == r"T_\mathrm{C}"
    assert_approx(magnetic.mf_exch_coupling.quantity, u.boltzmann_constant * 400 * u.K)
    magnetic.anisotropy = [1, 2, 3]
    assert np.array_equal(magnetic.anisotropy.magnitude, [1, 2, 3])
    magnetic.anisotropy = 5 * u.J / u.m**3
    assert np.array_equal(magnetic.anisotropy.magnitude, [5, 5, 5])


def test_group_assignment_updates_owner(amorphous_layer):
    layer = copy.deepcopy(amorphous_layer)  # do not alter the module-scoped fixture
    layer.elastic.sound_vel = 3 * u.nm / u.ps
    assert_approx(layer.spring_const, [0.45] * u.kg / u.s**2)
    layer.thermal.heat_capacity = "0.1*T"
    assert isinstance(layer.thermal.heat_capacity, TemperatureParameter)
    assert str(layer.thermal.heat_capacity.magnitude[0]) == "0.1*T"


def test_group_assignment_of_vector_parameter():
    magnetic = MagneticParameters()
    magnetic.magnetization = {"r": -1, "phi": 90 * u.deg, "gamma": 0 * u.deg}
    assert isinstance(magnetic.magnetization, VectorParameter)
    assert np.allclose(magnetic.magnetization.magnitude, [-1, 0, 0])
    magnetic.magnetization = (0, 0, 1)
    assert np.allclose(magnetic.magnetization.magnitude, [0, 0, 1])


def test_group_assignment_of_derived_parameter_raises():
    with pytest.raises(AttributeError, match="derived"):
        StructuralParameters(thickness=1 * u.nm, density=1 * u.kg / u.m**3).mass = 1.0


def test_group_assignment_replaces_parameter():
    magnetic = MagneticParameters(eff_spin=0.5, curie_temp=600 * u.K)
    given = Parameter("mK", 500e3, symbol="T_c")
    magnetic.curie_temp = given
    assert magnetic.curie_temp is not given
    assert magnetic.curie_temp.unit == u.K
    assert np.isclose(magnetic.curie_temp.magnitude, 500)
    assert magnetic.curie_temp.symbol == "T_c"
    assert magnetic.curie_temp.name == "curie_temp"
    assert_approx(magnetic.mf_exch_coupling.quantity, u.boltzmann_constant * 500 * u.K)
    # the copy is connected to the group, the given parameter is not
    given.magnitude = 1
    assert np.isclose(magnetic.curie_temp.magnitude, 500)
    magnetic.curie_temp = 300 * u.K
    assert_approx(magnetic.mf_exch_coupling.quantity, u.boltzmann_constant * 300 * u.K)


def test_group_assignment_replaces_vector_parameter():
    magnetic = MagneticParameters()
    magnetic.magnetization = VectorParameter("", (0, 0, 2), symbol="M")
    assert magnetic.magnetization.symbol == "M"
    assert np.allclose(magnetic.magnetization.magnitude, [0, 0, 2])


@pytest.mark.parametrize(
    "parameter, error",
    [
        (Parameter("m", 1), pint.DimensionalityError),
        (TemperatureParameter("K", 1), TypeError),
    ],
)
def test_group_assignment_replaces_parameter_raises(parameter, error):
    with pytest.raises(error):
        MagneticParameters().curie_temp = parameter


def test_group_assignment_replaces_temperature_parameter():
    thermal = ThermalParameters(heat_capacity=100 * u.J / u.kg / u.K)
    thermal.heat_capacity = TemperatureParameter("J/kg/K", "0.1*T", symbol="C")
    assert thermal.heat_capacity.symbol == "C"
    with pytest.raises(ValueError, match="must be given in"):
        thermal.heat_capacity = TemperatureParameter("kJ/kg/K", 1)


# sublattices

K_B = 1.380649e-23
EXCHANGE = np.array([[3 * K_B * 500, -K_B * 100], [-K_B * 100, 3 * K_B * 300]])


def test_sublattice_parameters():
    """scalars hold for all sublattices, whose number is given by the
    parameter with the most values"""
    magnetic = MagneticParameters(
        eff_spin=[np.inf, np.inf],
        lamda=0.01,
        anisotropy=[[0, 0, 1], [0, 0, 2]] * u.J / u.m**3,
        mf_exch_coupling=EXCHANGE * u.J,
        magnetization=[(0, 0, 1), {"r": -0.5, "phi": 0, "gamma": 0}],
    )
    assert magnetic.num_sublattices == magnetic.check_sublattices() == 2
    assert np.array_equal(magnetic.sublattice_values("lamda"), [0.01, 0.01])
    assert np.array_equal(magnetic.sublattice_values("anisotropy"), [[0, 0, 1], [0, 0, 2]])
    assert np.array_equal(magnetic.sublattice_values("mf_exch_coupling"), EXCHANGE)
    assert [v.r.magnitude for v in magnetic.magnetization] == [1, -0.5]
    copied = copy.deepcopy(magnetic)
    assert all(v._caller is copied for v in copied.magnetization)
    assert "magnetization_2" in repr(magnetic)


def test_curie_temp_derived_from_exchange_matrix():
    """the Curie temperature of several sublattices is the ordering temperature
    of the exchange matrix, which is the input"""
    magnetic = MagneticParameters(eff_spin=[np.inf, np.inf], mf_exch_coupling=EXCHANGE * u.J)
    T_C = np.max(np.linalg.eigvalsh(EXCHANGE / 3)) / K_B
    assert np.isclose(magnetic.curie_temp.magnitude, T_C)
    magnetic.mf_exch_coupling = 2 * EXCHANGE * u.J
    assert np.isclose(magnetic.curie_temp.magnitude, 2 * T_C)
    with pytest.raises(AttributeError, match="derived from the mean-field exchange matrix"):
        magnetic.curie_temp = 400 * u.K


def test_curie_temp_and_exchange_switch_for_single_sublattice(layer):
    layer = copy.deepcopy(layer)  # do not alter the module-scoped fixture
    J = layer.mf_exch_coupling
    layer.mf_exch_coupling = 2 * J
    assert_approx(layer.curie_temp, 200 * u.K)
    layer.curie_temp = 100 * u.K
    assert_approx(layer.mf_exch_coupling, J)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"eff_spin": [1, 2], "lamda": [1, 2, 3], "mf_exch_coupling": EXCHANGE},
            "eff_spin has 2 values",
        ),
        ({"eff_spin": [1, 2], "curie_temp": 500 * u.K}, "explicit 2x2"),
        ({"eff_spin": [1, 2], "mf_exch_coupling": EXCHANGE, "magnetization": (0, 0, 1)}, "each"),
        ({"eff_spin": [1, 2, 3], "mf_exch_coupling": EXCHANGE}, "explicit 3x3"),
        (
            {"anisotropy": np.ones((3, 3)), "eff_spin": [1, 2], "mf_exch_coupling": EXCHANGE},
            "eff_spin has 2 values, but there are 3",
        ),
    ],
)
def test_inconsistent_sublattices_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MagneticParameters(**kwargs).check_sublattices()


def test_layer_sublattices_from_mixed_atom():
    """every constituent of a mixed atom forms a sublattice of the layer"""
    Gd = Atom("Gd", magnetization=(0, 0, -1))
    Fe = Atom("Fe", magnetization=(0, 0, 1))
    GdFe = AtomMixed("GdFe")
    GdFe.add_atom(Gd, 0.25)
    GdFe.add_atom(Fe, 0.75)
    layer = AmorphousLayer(
        "GdFe",
        "GdFe",
        thickness=1 * u.nm,
        density=7000 * u.kg / u.m**3,
        atom=GdFe,
        eff_spin=[3.5, 0.5],
        mf_exch_coupling=EXCHANGE * u.J,
    )
    assert layer.num_sublattices == 2
    assert_approx(layer.magnetization, [[0, 0, -1], [0, 0, 1]] * u.dimensionless)
    layer.magnetization = [(1, 0, 0), (-1, 0, 0)]
    assert_approx(layer.magnetization, [[1, 0, 0], [-1, 0, 0]] * u.dimensionless)
    # a single constituent per layer keeps the magnetization of the mixed atom
    single = AmorphousLayer("GdFe", "GdFe", thickness=1 * u.nm, atom=GdFe)
    assert single.num_sublattices == 1
