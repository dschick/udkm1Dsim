#!/usr/bin/env python

import copy
import warnings

import numpy as np
import pytest
import sympy as sp
from pint.testing import assert_allclose as assert_approx
from pint.testing import assert_equal

from udkm1Dsim import AmorphousLayer, UnitCell, u
from udkm1Dsim.structures.parameters import Parameter

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
    assert layer.heat_capacity[0](300) == 10


def test_layer_heat_capacity_expr(layer):
    assert float(layer.heat_capacity_expr[0]) == pytest.approx(10.0, rel=1e-9, abs=0)


def test_layer_int_heat_capacity(layer):
    assert layer.int_heat_capacity[0](300) == 3000.0


def test_layer_int_heat_capacity_expr(layer):
    T = sp.symbols("T")
    assert layer.int_heat_capacity_expr[0].equals(10.0 * T)


def test_layer_nonintegratable_heat_capacity(layer):
    with pytest.warns(UserWarning):
        layer.heat_capacity = "1/abs(T-300)"
        # integration is only carried out when explicitly called
        layer.int_heat_capacity
    # reset to original value
    layer.heat_capacity = 10 * (u.J / u.kg / u.K)


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


def test_layer_sound_vel(layer):
    assert_equal(layer.sound_vel, 6 * u.nm / u.ps)


def test_layer_phonon_damping(layer):
    assert_equal(layer.phonon_damping, 1 * u.kg / u.s)


def test_layer_set_ho_spring_constants(layer):
    assert layer.spring_const == 0.0
    layer.set_ho_spring_constants([1.0, 2.0, 3.0])
    assert np.allclose(layer.spring_const, [0.0, 1.0, 2.0, 3.0] * u.kg / u.s**2)


# ============================================================================
# Optical parameters
# ============================================================================


def test_layer_opt_pen_depth(layer):
    # asserat_equal does not work on because of last digit uncertainty
    assert_approx(layer.opt_pen_depth, 10 * u.nm)


def test_layer_opt_ref_index(layer):
    assert layer.opt_ref_index == 5 - 3j


def test_layer_opt_ref_index_per_strain(layer):
    assert layer.opt_ref_index_per_strain == 1 - 1j


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
    assert layer.aniso_exponent == 1


def test_layer_anisotropy(layer):
    assert_equal(layer.anisotropy, [1, 2, 3] * u.J / u.m**3)


def test_layer_exch_stiffness(layer):
    assert_equal(layer.exch_stiffness, [1, 1, 1] * u.J / u.m)


def test_layer_mag_saturation(layer):
    assert_equal(layer.mag_saturation, 1 * u.J / u.T / u.m**3)


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


def test_layer_set_opt_pen_depth_from_ref_index(layer):
    layer = copy.deepcopy(layer)  # do not alter the module-scoped fixture
    layer.set_opt_pen_depth_from_ref_index(800 * u.nm)
    assert_approx(layer.opt_pen_depth, 21.22 * u.nm, rtol=1e-2)
    layer.opt_ref_index = 2.0 + 0j
    layer.set_opt_pen_depth_from_ref_index(800 * u.nm)
    assert np.isinf(layer.optical.opt_pen_depth.magnitude)


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
    assert_approx(amorphous_layer.elastic.acoustic_impedance.quantity, 3e7 * u.kg / u.m**2 / u.s)


@pytest.mark.parametrize(
    "prop", ["mass", "mass_unit_area", "area", "volume", "spring_const", "num_sub_systems"]
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
    assert_approx(
        layer.elastic.acoustic_impedance.quantity, acoustic_impedance * u.kg / u.m**2 / u.s
    )


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
    assert vacuum.opt_ref_index == 1


# ============================================================================
# AmorphousLayer
# ============================================================================


def test_amorphous_layer_atom(amorphous_layer, atom_iron):
    assert amorphous_layer.atom is atom_iron


def test_amorphous_layer_magnetization_from_atom(amorphous_layer):
    amplitude, phi, gamma = amorphous_layer.magnetic.magnetization.polar
    assert np.isclose(amplitude, 0.5)
    assert_approx(phi, 0 * u.deg)
    assert_approx(gamma, 180 * u.deg)


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
    assert unit_cell.get_atom_ids() == ["Sr", "O", "Ti"]
    assert np.allclose(unit_cell.get_atom_positions(), [0, 0.5, 1])
    assert np.allclose(unit_cell.get_atom_positions(0.1), [0, 0.55, 1.1])


def test_unit_cell_add_multiple_atoms(unit_cell, atom_oxygen):
    uc = copy.deepcopy(unit_cell)  # do not alter the module-scoped fixture
    uc.add_multiple_atoms(atom_oxygen, 0.5, 2)
    assert uc.num_atoms == 5
    assert uc.get_atom_ids() == ["Sr", "O", "Ti"]  # unique ids
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
