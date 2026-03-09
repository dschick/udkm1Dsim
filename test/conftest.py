#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Atom, AtomMixed
from udkm1Dsim import Layer, AmorphousLayer, UnitCell
from udkm1Dsim import Structure
from udkm1Dsim import u
import pytest


# fixtures

# atoms

@pytest.fixture(scope='module')
def atom_iron():
    atom_iron = Atom('Fe')
    atom_iron.mag_amplitude = 0.5
    atom_iron.mag_phi = 0*u.deg
    atom_iron.mag_gamma = 180*u.deg
    return atom_iron


@pytest.fixture(scope='module')
def atom_oxygen():
    atom_oxygen = Atom('O')
    atom_oxygen.ionicity = 1
    atom_oxygen.mag_amplitude = 0.5
    atom_oxygen.mag_phi = 0*u.deg
    atom_oxygen.mag_gamma = 180*u.deg
    return atom_oxygen


@pytest.fixture(scope='module')
def atom_strontium():
    atom_strontium = Atom('Sr')
    atom_strontium.mag_amplitude = 0.5
    atom_strontium.mag_phi = 0*u.deg
    atom_strontium.mag_gamma = 180*u.deg
    return atom_strontium


@pytest.fixture(scope='module')
def atom_titanium():
    atom_titanium = Atom('Ti')
    atom_titanium.mag_amplitude = 0.5
    atom_titanium.mag_phi = 0*u.deg
    atom_titanium.mag_gamma = 180*u.deg
    return atom_titanium


@pytest.fixture(scope='module')
def atom_dysprosium():
    atom_dysprosium = Atom('Dy')
    atom_dysprosium.ionicity = 1
    atom_dysprosium.mag_amplitude = 0.
    atom_dysprosium.mag_phi = 0.*u.deg
    atom_dysprosium.mag_gamma = 0.*u.deg
    return atom_dysprosium


@pytest.fixture(scope='module')
def atom_mixed(atom_dysprosium, atom_iron):
    atom_mixed = AtomMixed('DyFe')
    atom_mixed.add_atom(atom_dysprosium, 0.4)
    atom_mixed.add_atom(atom_iron, 0.6)
    atom_mixed.mag_amplitude = 0.
    atom_mixed.mag_phi = 0.*u.deg
    atom_mixed.mag_gamma = 0.*u.deg
    return atom_mixed


# layers


@pytest.fixture(scope='module')
def properties():
    props = {'roughness': 0.5*u.angstrom,
             'deb_wal_fac': 1*u.angstrom**2,
             'sound_vel': 6*u.nm/u.ps,
             'phonon_damping': 1*u.kg/u.s,
             'opt_pen_depth': 10.0*u.nm,
             'opt_ref_index': 5-3j,
             'opt_ref_index_per_strain': 1-1j,
             'therm_cond': 1*u.W/(u.m*u.K),
             'lin_therm_exp': 1e-5,
             'heat_capacity': 10*(u.J/u.kg/u.K),
             'sub_system_coupling': [0],
             'eff_spin': 1,
             'curie_temp': 100*u.K,
             'lamda': 1,
             'mag_moment': 1*u.bohr_magneton,
             'aniso_exponent': 1,
             'anisotropy': [1, 2, 3]*u.J/u.m**3,
             'exch_stiffness': 1*u.J/u.m,
             'mag_saturation': 1*u.J/u.T/u.m**3,
             }
    return props


@pytest.fixture(scope='module')
def layer(properties):
    return Layer(id='layer', name='base layer', **properties)


@pytest.fixture(scope='module')
def amorphous_layer(atom_iron, properties):
    return AmorphousLayer(id='amorphous_layer', name='amorphous layer', thickness=1*u.nm,
                          density=5000*u.kg/u.m**3, atom=atom_iron, **properties)


@pytest.fixture(scope='module')
def unit_cell(atom_strontium, atom_oxygen, atom_titanium, properties):
    uc = UnitCell(id='unit_cell', name='unit_cell', c_axis=5.0*u.angstrom, **properties)
    uc.add_atom(atom_strontium, 0.0)
    uc.add_atom(atom_oxygen, 0.5)
    uc.add_atom(atom_titanium, 1.0)
    return uc


@pytest.fixture(scope='module')
def amorphous_layer_iron(atom_iron, properties):
    return AmorphousLayer(id='amorphous_layer_Fe', name='amorphous layer iron', thickness=1*u.nm,
                          density=5000*u.kg/u.m**3, atom=atom_iron, **properties)


@pytest.fixture(scope='module')
def amorphous_layer_oxygen(atom_oxygen, properties):
    return AmorphousLayer(id='amorphous_layer_O', name='amorphous layer oxygen', thickness=1*u.nm,
                          density=5000*u.kg/u.m**3, atom=atom_oxygen, **properties)


@pytest.fixture(scope='module')
def unit_cell_iron(atom_iron, properties):
    uc = UnitCell(id='unit_cell_Fe', name='unit cell iron', c_axis=5.0*u.angstrom, **properties)
    uc.add_atom(atom_iron, 0.0)
    return uc


@pytest.fixture(scope='module')
def unit_cell_oxygen(atom_oxygen, properties):
    uc = UnitCell(id='unit_cell_O', name='unit cell oxygen', c_axis=5.0*u.angstrom, **properties)
    uc.add_atom(atom_oxygen, 0.0)
    return uc


# structure


@pytest.fixture(scope='module')
def structure(amorphous_layer_iron, amorphous_layer_oxygen, unit_cell_iron, unit_cell_oxygen):
    S = Structure('structure mixed')
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


@pytest.fixture(scope='module')
def structure_amorph(amorphous_layer_iron, amorphous_layer_oxygen):
    S = Structure('structure amorph')
    S.add_sub_structure(amorphous_layer_iron, 10)
    S.add_sub_structure(amorphous_layer_oxygen, 10)

    substrate = Structure('substrate')
    substrate.add_sub_structure(amorphous_layer_iron, 100)

    S.add_substrate(substrate)

    return S


@pytest.fixture(scope='module')
def structure_crystalline(unit_cell_iron, unit_cell_oxygen):
    S = Structure('structure crystalline')
    S.add_sub_structure(unit_cell_iron, 20)
    S.add_sub_structure(unit_cell_oxygen, 20)

    substrate = Structure('substrate')
    substrate.add_sub_structure(unit_cell_iron, 100)

    S.add_substrate(substrate)

    return S
