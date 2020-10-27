#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Atom, AtomMixed
import numpy as np
import pytest


def test_atom():
    Dy = Atom('Dy')
    assert Dy.symbol == 'Dy'
    assert Dy.id == 'Dy'
    assert Dy.ionicity == 0
    assert Dy.name == 'Dysprosium'
    assert Dy.atomic_number_z == 66
    assert Dy.mass_number_a == 162.5
    assert Dy.mass.magnitude == pytest.approx(2.698e-25)
    # check if python hash works the same on different systems
    assert np.array_equal(Dy.atomic_form_factor_coeff[10],
                          np.array([1.4923350000000001e+01, 1.09656e+01, 4.6755e-01]))
    assert np.array_equal(Dy.cromer_mann_coeff,
                          np.array([66.0, 0.0, 26.507, 17.6383, 14.5596, 2.96577, 2.1802, 0.202172,
                                    12.1899, 111.874, 4.29728]))

    Oxygen = Atom('O', id='myOxygen', ionicity=-1)
    assert Oxygen.symbol == 'O'
    assert Oxygen.id == 'myOxygen'
    assert Oxygen.ionicity == -1


def test_atom_mixed():
    DyTb = AtomMixed('DyTb')
    DyTb.add_atom(Atom('Dy', ID='Dy'), 0.4)
    DyTb.add_atom(Atom('Tb', ID='Tb'), 0.6)
    assert DyTb.name == 'DyTb'
    assert DyTb.id == 'DyTb'
    assert DyTb.atomic_number_z == 65.4
    assert DyTb.mass_number_a == 160.35518
    assert DyTb.mass.magnitude == pytest.approx(2.663e-25)
