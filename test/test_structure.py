#!/usr/bin/env python
# -*- coding: utf-8 -*-
from udkm1Dsim import Atom
from udkm1Dsim import UnitCell, AmorphousLayer
from udkm1Dsim import Structure
from udkm1Dsim import u
u.default_format = '~P'
# import numpy as np
import pytest
# from pint.testing import assert_allclose as assert_approx
# from pint.testing import assert_equal


# fixtures


@pytest.fixture(scope='module')
def atom_iron():
    atom_iron = Atom('Fe')
    return atom_iron


@pytest.fixture(scope='module')
def atom_oxygen():
    atom_oxygen = Atom('O')
    atom_oxygen.ionicity = 1
    return atom_oxygen


@pytest.fixture(scope='module')
def amorphous_layer_iron(atom_iron):
    return AmorphousLayer(id='amorphous_layer_Fe', name='amorphous layer iron', thickness=1*u.nm,
                          density=5000*u.kg/u.m**3, atom=atom_iron)


@pytest.fixture(scope='module')
def amorphous_layer_oxygen(atom_oxygen):
    return AmorphousLayer(id='amorphous_layer_O', name='amorphous layer oxygen', thickness=1*u.nm,
                          density=5000*u.kg/u.m**3, atom=atom_oxygen)


@pytest.fixture(scope='module')
def unit_cell_iron(atom_iron, atom_oxygen):
    uc = UnitCell(id='unit_cell_Fe', name='unit cell iron', c_axis=5.0*u.angstrom)
    uc.add_atom(atom_iron, 0.0)
    return uc


@pytest.fixture(scope='module')
def unit_cell_oxygen(atom_oxygen):
    uc = UnitCell(id='unit_cell_O', name='unit cell oxygen', c_axis=5.0*u.angstrom)
    uc.add_atom(atom_oxygen, 0.0)
    return uc


# structure amorph, crystalline, mixed, with and w/o substrate
# I guess one could directly try with a mixed structure with substrate and sub_structure
@pytest.fixture(scope='module')
def structure_amorph(amorphous_layer_iron, amorphous_layer_oxygen):
    S = Structure('structure amorph')
    S.add_sub_structure(amorphous_layer_iron, 10)
    S.add_sub_structure(amorphous_layer_oxygen, 10)
    return S


# tests

# __str__

# visualize

def test_structure_get_hash(structure_amorph):
    structure_amorph.get_hash()


# add_sub_structure

# add_substrate

# get_number_of_sub_structures

# get_number_of_layers

# get_number_of_unique_layers

# get_thickness

# get_unique_layers

# get_layer_vectors

# get_all_positions_per_unique_layer

# get_distances_of_layers

# get_distances_of_interfaces

# interp_distance_at_interfaces

# get_layer_property_vector

# get_layer_handle

# reverse

# reverse_sub_structures
