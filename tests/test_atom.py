#!/usr/bin/env python


from pathlib import Path

import numpy as np
import pytest
from pint.testing import assert_allclose, assert_equal

from udkm1Dsim import u

# tests

@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', 'Fe'), ('atom_dysprosium', 'Dy'), ('atom_mixed', 'DyFe')])
def test_atom_symbol(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.symbol == expected


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', 'Fe'), ('atom_dysprosium', 'Dy'), ('atom_mixed', 'DyFe')])
def test_atom_id(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.id == expected


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', 'Iron'),
                          ('atom_dysprosium', 'Dysprosium'),
                          ('atom_mixed', 'DyFe')])
def test_atom_name(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.name == expected


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', 0), ('atom_dysprosium', 1), ('atom_mixed', 0.4)])
def test_atom_ionicity(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.ionicity == expected


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', 26), ('atom_dysprosium', 66), ('atom_mixed', 42)])
def test_atom_atomic_number_z(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.atomic_number_z == expected


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', 55.845),
                          ('atom_dysprosium', 162.5),
                          ('atom_mixed', 98.507)])
def test_atom_mass_number_a(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.mass_number_a == expected


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', 9.273e-26*u.kg),
                          ('atom_dysprosium', 2.698e-25*u.kg),
                          ('atom_mixed', 1.636e-25*u.kg)])
def test_atom_mass(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert_allclose(atom.mass, expected, rtol=1e-3)


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', [20.83314, 4.7737, 3.0272]),
                          ('atom_dysprosium', [14.92335, 10.9656,  0.46755])
                          ])
def test_atom_atomic_form_factor_coeff(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert np.allclose(atom.atomic_form_factor_coeff[10], np.array(expected))


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', [26., 0., 11.7695, 7.3573, 3.5222, 2.3045, 4.7611, 0.3072,
                                         15.3535, 76.8805, 1.0369]),
                          ('atom_dysprosium', [66., 0., 26.507, 17.6383, 14.5596, 2.96577, 2.1802,
                                               0.202172, 12.1899, 111.874, 4.29728])
                          ])
def test_atom_cromer_mann_coeff(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert np.allclose(atom.cromer_mann_coeff, np.array(expected))


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', [5.0148e+02, 1.6077e-02, 0.0000e+00]),
                          ('atom_dysprosium', [5000., 0., 0.])
                          ])
def test_atom_magnetic_form_factor_coeff(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert np.allclose(atom.magnetic_form_factor_coeff[4], np.array(expected))


# atom_mixed must read atomic form factors from an external file and is hence excluded here
@pytest.mark.parametrize('fixture_name', ['atom_iron', 'atom_dysprosium'])
def test_atom_read_atomic_form_factor_coeff(request, fixture_name):
    atom = request.getfixturevalue(fixture_name)
    atom.read_atomic_form_factor_coeff(source='chantler')
    atom.read_atomic_form_factor_coeff(source='henke')
    with pytest.raises(ValueError):
        atom.read_atomic_form_factor_coeff(source='wrong')


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', np.complex128(3.84-1.85j)),
                          ('atom_dysprosium', np.complex128(36.00-11.25j)),
                          ('atom_mixed', np.complex128(16.71-5.61j))])
def test_atom_get_atomic_form_factor(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.get_atomic_form_factor(700) == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', np.complex128(-33.9-1.9j)),
                          ('atom_dysprosium', np.complex128(-56.5-11.3j)),
                          ('atom_mixed', np.complex128(-43.0-5.6j))
                          ])
def test_atom_get_cm_atomic_form_factor(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.get_cm_atomic_form_factor(700, 1) == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize('fixture_name', ['atom_iron', 'atom_dysprosium', 'atom_mixed'])
def test_atom_read_magnetic_form_factor_coeff(request, fixture_name):
    atom = request.getfixturevalue(fixture_name)
    atom.read_magnetic_form_factor_coeff()


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', np.complex128(1.11-0.01j)),
                          ('atom_dysprosium', np.complex128(0j)),
                          ('atom_mixed', np.complex128(0.67-0.01j))])
def test_atom_get_magnetic_form_factor(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.get_magnetic_form_factor(700) == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize('fixture_name, expected',
                         [('atom_iron', [0.5, 0*u.deg, 180*u.deg]),
                          ('atom_dysprosium', [0, 0*u.deg, 0*u.deg]),
                          ('atom_mixed', [0, 0*u.deg, 0*u.deg])])
def test_atom_magnetization(request, fixture_name, expected):
    atom = request.getfixturevalue(fixture_name)
    assert atom.mag_amplitude == expected[0]
    assert_equal(atom.mag_phi, expected[1])
    assert_equal(atom.mag_gamma, expected[2])


@pytest.mark.parametrize('fixture_name', ['atom_iron', 'atom_dysprosium', 'atom_mixed'])
def test_atom_to_string(request, fixture_name):
    atom = request.getfixturevalue(fixture_name)
    atom.__str__()


@pytest.mark.parametrize('fixture_name', ['atom_iron', 'atom_dysprosium', 'atom_mixed'])
def test_atom_read_atomic_form_factor_coeff_from_file(request, fixture_name):
    atom = request.getfixturevalue(fixture_name)
    filename = Path.joinpath(Path(__file__).parent, 'data', 'Fe.cf')
    atom.read_atomic_form_factor_coeff(filename=filename)
    with pytest.raises(FileNotFoundError):
        atom.read_atomic_form_factor_coeff(filename='wrong')


@pytest.mark.parametrize('fixture_name', ['atom_iron', 'atom_dysprosium', 'atom_mixed'])
def test_atom_read_magnetic_form_factor_coeff_from_file(request, fixture_name):
    atom = request.getfixturevalue(fixture_name)
    # magnetic and electronic form factor files have the same format
    filename = Path.joinpath(Path(__file__).parent, 'data', 'Fe.cf')
    mag_coeff = atom.read_magnetic_form_factor_coeff(filename=filename)
    atom.magnetic_form_factor_coeff = mag_coeff
    atom.get_magnetic_form_factor(700)
    atom.read_magnetic_form_factor_coeff(filename='wrong')
