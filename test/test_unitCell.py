#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Atom
from udkm1Dsim import UnitCell
from udkm1Dsim import u
u.default_format = '~P'


def test_unit_cell():
    Fe = Atom('Fe')
    uc = UnitCell('uc', 'Unit Cell', 2.86*u.angstrom, heat_capacity=10*(u.J/u.kg/u.K),
                  lin_therm_exp=1e-6/u.K, therm_cond=1*(u.W/u.m/u.K),
                  opt_pen_depth=11*u.nm, sound_vel=5*(u.nm/u.ps))
    uc.add_atom(Fe, 'lambda strain: 0*(strain+1)')
    uc.add_atom(Fe, 'lambda strain: 0.5*(strain+1)')

    assert uc.id == 'uc'
    assert uc.name == 'Unit Cell'
    assert uc.a_axis == 2.86*u.angstrom
    assert uc.b_axis == 2.86*u.angstrom
    assert uc.c_axis == 2.86*u.angstrom
    assert uc.heat_capacity[0](300) == 10
    assert uc.int_heat_capacity[0](300) == 3000
    assert uc.lin_therm_exp[0](300) == 1e-6
    assert uc.int_lin_therm_exp[0](300) == 0.0003
    assert uc.therm_cond[0](300) == 1
    assert uc.opt_pen_depth == 11*u.nm
    assert uc.sound_vel == 5*(u.nm/u.ps)
