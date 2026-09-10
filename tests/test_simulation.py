#!/usr/bin/env python
import numpy as np
import pytest

from udkm1Dsim import Simulation

# fixtures


@pytest.fixture(scope='module')
def simulation(structure):
    return Simulation(structure, force_recalc=True, cache_dir='./',
                      save_data=True, disp_messages=True, progress_bar=True)


# tests


def test_simulation_str(simulation):
    simulation.__str__()


def test_disp_message(simulation, capsys):
    simulation.disp_messages = True
    simulation.disp_message('test message')
    captured = capsys.readouterr()
    assert captured.out == 'test message\n'
    simulation.disp_messages = False
    simulation.disp_message('do not show this')
    captured = capsys.readouterr()
    assert captured.out == ''


def test_simulation_save(simulation, tmp_path_factory):
    test_array = np.r_[1, 2, 3]
    filename = tmp_path_factory.mktemp("data") / "test"
    simulation.save(filename, {'test_array': test_array})
    with pytest.raises(TypeError):
        simulation.save(filename, test_array)


def test_conv_with_function(simulation):
    def handle(x):
        return np.exp(-x**2)

    x = np.r_[0:10:1]
    y = np.r_[0:10:1]
    assert np.allclose(simulation.conv_with_function(y, x, handle),
                       np.array([0.22840624, 1.01047185, 2.00006975, 3.00000006, 4.,
                                 4.99999943, 5.99937279, 6.90631129, 6.02811867, 6.02811867]))


def test_cache_dir(simulation):
    assert simulation.cache_dir == './'
