#!/usr/bin/env python

import numpy as np
import pint
from pint.testing import assert_allclose

u = pint.get_application_registry()
u.formatter.default_format = '.4g~P'


# tests


def test_structure_str(structure):
    structure.__str__()


def test_thickness(structure):
    assert_allclose(structure.thickness, 150*u.nm, rtol=1e-2)


# test_visualize() fails on certain virtual machine due to TCL error


def test_structure_get_hash(structure):
    structure.get_hash()


def test_get_number_of_sub_structures(structure):
    assert structure.get_number_of_sub_structures() == 5


def test_get_number_of_layers(structure):
    assert structure.get_number_of_layers() == 280


def test_get_number_of_unique_layers(structure):
    assert structure.get_number_of_unique_layers() == 4


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
