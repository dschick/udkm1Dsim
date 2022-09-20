#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import udkm1Dsim as ud


@pytest.fixture(scope='module')
def atom_oxygen():
    atom_oxygen = ud.Atom('O')
    return atom_oxygen


@pytest.fixture(scope='module')
def atom_dysprosium():
    atom_dysprosium = ud.Atom('Dy')
    return atom_dysprosium
