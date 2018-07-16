#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsimpy.atoms import Atom, AtomMixed


def test_atom():
    Dy = Atom('Dy')
    assert Dy.name == 'Dysprosium'
    assert Dy.id == 'Dy'


def test_atom_mixed():
    FeCo = AtomMixed('FeCo')
    assert FeCo.name == 'FeCo'
    assert FeCo.id == 'FeCo'
