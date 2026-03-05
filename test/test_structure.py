#!/usr/bin/env python
# -*- coding: utf-8 -*-
from udkm1Dsim import Atom
from udkm1Dsim import UnitCell, AmorphousLayer
from udkm1Dsim import Structure
from udkm1Dsim import u
u.default_format = '~P'
import numpy as np
import pytest
from pint.testing import assert_allclose


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


@pytest.fixture(scope='module')
def structure(amorphous_layer_iron, amorphous_layer_oxygen, unit_cell_iron, unit_cell_oxygen):
    S = Structure('structure amorph')
    S.add_sub_structure(amorphous_layer_iron, 10)
    S.add_sub_structure(amorphous_layer_oxygen, 10)
    S.add_sub_structure(unit_cell_iron, 20)

    DL = Structure('double layer')
    DL.add_sub_structure(unit_cell_iron, 5)
    DL.add_sub_structure(unit_cell_oxygen, 7)

    S.add_sub_structure(DL, 20)

    substrate = Structure('substrate')
    substrate.add_sub_structure(amorphous_layer_iron, 100)

    S.add_substrate(substrate)

    return S


# tests


def test_structure_str(structure):
    structure.__str__()


def test_structure_visualize(structure):
    structure.visualize(block=False)


def test_structure_get_hash(structure):
    assert structure.get_hash(types='heat') == 'b3a2d778b80f935f6aeb70c0428f83a8'
    assert structure.get_hash(types='phonon') == '7906ad75e7bd69cdf354ed3f7180d0f4'
    assert structure.get_hash(types='xray') == 'ff455714885d2d028e8efa36ca391d64'
    assert structure.get_hash(types='magnetic') == '5711dc2e9867eefa69c7fc230b216036'


def test_get_number_of_sub_structures(structure):
    assert structure.get_number_of_sub_structures() == 5


def test_get_number_of_layers(structure):
    assert structure.get_number_of_layers() == 280


def test_get_number_of_unique_layers(structure):
    assert structure.get_number_of_unique_layers() == 4


def test_get_thickness(structure):
    assert_allclose(structure.get_thickness(), 150*u.nm, rtol=1e-2)


def test_get_unique_layers(structure):
    layer_ids, _ = structure.get_unique_layers()
    assert layer_ids == ['amorphous_layer_Fe', 'amorphous_layer_O', 'unit_cell_Fe', 'unit_cell_O']


def test_get_layer_vectors(structure):
    structure.get_layer_vectors()


def test_get_all_positions_per_unique_layer(structure):
    pos = structure.get_all_positions_per_unique_layer()
    assert np.allclose(pos['amorphous_layer_Fe'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_get_distances_of_layers(structure):
    d_start, _, _ = structure.get_distances_of_layers()
    assert_allclose(np.sum(d_start), 2.22e-05*u.m, rtol=1e-2)


def test_get_distances_of_interfaces(structure):
    assert_allclose(np.sum(structure.get_distances_of_interfaces()), 3.68e-6*u.m, rtol=1e-2)


def test_interp_distance_at_interfaces(structure):
    structure.interp_distance_at_interfaces(N=11)


def test_get_layer_property_vector(structure):
    structure.get_layer_property_vector('density')
    structure.get_layer_property_vector('_density')
    structure.get_layer_property_vector('heat_capacity')
    structure.get_layer_property_vector('spring_const')


def test_get_layer_handle(structure):
    structure.get_layer_handle(0)


def test_reverse(structure):
    structure.reverse()
