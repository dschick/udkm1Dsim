#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Atom
from udkm1Dsim import UnitCell
from udkm1Dsim import Structure
from udkm1Dsim import u
u.default_format = '~P'


def test_structure():
    Dy = Atom('Dy')
    uc = UnitCell('uc', 'Unit Cell', 3.1*u.angstrom, heat_capacity=10*(u.J/u.kg/u.K),
                  lin_therm_exp=1e-6/u.K, therm_cond=1*(u.W/u.m/u.K),
                  opt_pen_depth=11*u.nm, sound_vel=5*(u.nm/u.ps))
    uc.add_atom(Dy, '0*(s+1)')
    uc.add_atom(Dy, '0.5*(s+1)')

    S = Structure('sample')
    assert S.name == 'sample'
