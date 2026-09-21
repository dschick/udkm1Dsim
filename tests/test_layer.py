#!/usr/bin/env python

import numpy as np
import pytest
import sympy as sp
from pint.testing import assert_allclose as assert_approx
from pint.testing import assert_equal

from udkm1Dsim import u

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
    layer.elastic.set_ho_spring_constants([1.0, 2.0, 3.0])
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
    layer.get_property_dict(types="heat")
    layer.get_property_dict(types="phonon")
    layer.get_property_dict(types="xray")
    layer.get_property_dict(types="optical")
    layer.get_property_dict(types="magnetic")


# def test_layer_set_opt_pen_depth_from_ref_index(layer):
#     layer.set_opt_pen_depth_from_ref_index(800 * u.nm)
#     assert_approx(layer.opt_pen_depth, 21.22 * u.nm, rtol=1e-2)


# # Vacuum, AmorphousLayer & UnitCell (need atoms, mass, area to be set)


# @pytest.mark.parametrize("fixture_name", ["vacuum", "amorphous_layer", "unit_cell"])
# def test_layer_to_str(request, fixture_name):
#     layer = request.getfixturevalue(fixture_name)
#     layer.__str__()


# @pytest.mark.parametrize(
#     "fixture_name, expected",
#     [
#         ("vacuum", 1.0 * u.nm),
#         ("amorphous_layer", 1.0 * u.nm),
#         ("unit_cell", 5.0 * u.angstrom),
#     ],
# )
# def test_layer_thickness(request, fixture_name, expected):
#     layer = request.getfixturevalue(fixture_name)
#     assert_approx(layer.thickness, expected, rtol=1e-2)


# @pytest.mark.parametrize(
#     "fixture_name, expected",
#     [
#         ("vacuum", 0.0 * u.kg),
#         ("amorphous_layer", 5e-26 * u.kg),
#         ("unit_cell", 2.51e-25 * u.kg),
#     ],
# )
# def test_layer_mass(request, fixture_name, expected):
#     layer = request.getfixturevalue(fixture_name)
#     assert_approx(layer.mass, expected, rtol=1e-2)


# @pytest.mark.parametrize(
#     "fixture_name, expected",
#     [
#         ("vacuum", 1e-20 * u.m**2),
#         ("amorphous_layer", 1e-20 * u.m**2),
#         ("unit_cell", 2.5e-19 * u.m**2),
#     ],
# )
# def test_layer_area(request, fixture_name, expected):
#     layer = request.getfixturevalue(fixture_name)
#     assert_approx(layer.area, expected, rtol=1e-2)


# @pytest.mark.parametrize(
#     "fixture_name, expected",
#     [
#         ("vacuum", 1e-29 * u.m**3),
#         ("amorphous_layer", 1e-29 * u.m**3),
#         ("unit_cell", 1.25e-28 * u.m**3),
#     ],
# )
# def test_layer_volume(request, fixture_name, expected):
#     layer = request.getfixturevalue(fixture_name)
#     assert_approx(layer.volume, expected, rtol=1e-2)


# @pytest.mark.parametrize(
#     "fixture_name, expected",
#     [
#         ("amorphous_layer", 30000000.0 * u.kg**0.5 / u.m**2),
#         ("unit_cell", 2414871.22 * u.kg**0.5 / u.m**2),
#     ],
# )
# def test_layer_get_acoustic_impedance(request, fixture_name, expected):
#     layer = request.getfixturevalue(fixture_name)
#     assert_approx(layer.get_acoustic_impedance(), expected, rtol=1e-2)


# @pytest.mark.parametrize(
#     "fixture_name, expected",
#     [
#         ("amorphous_layer", 1.8),
#         ("unit_cell", 1.45),
#     ],
# )
# def test_layer_calc_spring_const(request, fixture_name, expected):
#     layer = request.getfixturevalue(fixture_name)
#     layer.calc_spring_const()
#     assert_approx(layer.spring_const[0], expected, rtol=1e-2)


# # AmorphousLayer


# def test_amorphous_layer_atom(amorphous_layer, atom_iron):
#     assert amorphous_layer.atom == atom_iron


# def test_amorphous_layer_magnetization(amorphous_layer):
#     assert amorphous_layer.magnetization["amplitude"] == 0.5
#     assert_equal(amorphous_layer.magnetization["phi"], 0 * u.deg)
#     assert_equal(amorphous_layer.magnetization["gamma"], 180 * u.deg)


# # UnitCell


# def test_unit_cell_crystal_axis(unit_cell):
#     assert_approx(unit_cell.a_axis, 5.0 * u.angstrom, rtol=1e-2)
#     assert_approx(unit_cell.b_axis, 5.0 * u.angstrom, rtol=1e-2)
#     assert_approx(unit_cell.c_axis, 5.0 * u.angstrom, rtol=1e-2)


# def test_unit_cell_number_atoms(unit_cell):
#     assert unit_cell.num_atoms == 3


# # test_visualize() fails on certain virtual machine due to TCL error


# def test_unit_cell_add_multiple_atoms(unit_cell, atom_oxygen):
#     unit_cell.add_multiple_atoms(atom_oxygen, 0.5, 2)


# def test_unit_cell_get_atom_ids(unit_cell):
#     assert unit_cell.get_atom_ids() == ["Sr", "O", "Ti"]


# def test_unit_cell_get_atom_positions(unit_cell):
#     assert np.allclose(unit_cell.get_atom_positions(), [0, 0.5, 0.5, 0.5, 1])
#     assert np.allclose(unit_cell.get_atom_positions(0.1), [0, 0.55, 0.55, 0.55, 1.1])
