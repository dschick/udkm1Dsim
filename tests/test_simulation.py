#!/usr/bin/env python
import numpy as np
import pytest

from udkm1Dsim import LLB, Elzo, Heat, PhononAna, PhononNum, Result, Simulation, XrayDyn

# fixtures


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("cache")


@pytest.fixture(scope="module")
def simulation(structure, cache_dir):
    return Simulation(
        structure,
        force_recalc=True,
        cache_dir=cache_dir,
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


# tests


def test_simulation_repr(simulation):
    assert repr(simulation).startswith("Simulation properties:")
    assert str(simulation) == repr(simulation)
    assert simulation._repr_html_().startswith("<h2>Simulation</h2>")


def test_disp_message(simulation, capsys):
    simulation.disp_messages = True
    simulation.disp_message("test message")
    captured = capsys.readouterr()
    assert captured.out == "test message\n"
    simulation.disp_messages = False
    simulation.disp_message("do not show this")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_simulation_save(simulation, tmp_path_factory):
    test_array = np.r_[1, 2, 3]
    filename = tmp_path_factory.mktemp("data") / "test.h5"
    result = simulation.new_result("test", "hash")
    result.add("test_array", test_array)
    simulation.save(filename, result)
    loaded = Result.load(filename)
    assert np.all(loaded.test_array == test_array)
    assert loaded.attrs["hash"] == "hash"
    assert loaded.attrs["simulation"] == "Simulation"
    with pytest.raises(AttributeError):
        simulation.save(filename, test_array)


def test_cache_dir(simulation, cache_dir, structure):
    assert simulation.cache_dir == cache_dir
    # the current directory is the default
    assert Simulation(structure, True, save_data=False).cache_dir == "./"


@pytest.mark.parametrize(
    "cls, kwargs, message",
    [
        (Simulation, {"save": False}, r"'save' \(did you mean 'save_data'\?\)"),
        (Heat, {"heat_difusion": True}, r"'heat_difusion' \(did you mean 'heat_diffusion'\?\)"),
        (LLB, {"H_ext": 1}, r"LLB\(\) got an unexpected keyword argument 'H_ext'"),
        (PhononAna, {"backend": "numba"}, r"PhononAna\(\) got an unexpected keyword"),
        (XrayDyn, {"calc_type": "parallel"}, r"XrayDyn\(\) got an unexpected keyword"),
        (Elzo, {"num_workers": 4}, r"Elzo\(\) got an unexpected keyword"),
    ],
)
def test_simulation_rejects_unknown_keyword_arguments(structure, cls, kwargs, message):
    """misspelled keyword arguments would be ignored silently otherwise"""
    with pytest.raises(TypeError, match=message):
        cls(structure, True, **kwargs)


def test_simulation_keyword_arguments_of_sub_classes(structure):
    heat = Heat(structure, True, save_data=False, heat_diffusion=True, intp_at_interface=3)
    assert heat.heat_diffusion
    assert heat.intp_at_interface == 3
    phonon = PhononNum(structure, True, only_heat=True, backend="scipy")
    assert phonon.only_heat
    assert phonon.backend == "scipy"
