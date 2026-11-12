#!/usr/bin/env python

import numpy as np
import pytest

from udkm1Dsim import Result, u

# fixtures


@pytest.fixture(scope="module")
def result():
    """a result with the size of a typical temperature map"""
    rng = np.random.default_rng(0)
    result = Result("temperature map")
    result.add_coord("delays", np.linspace(-1, 100, 1000) * u.ps, "s")
    result.add_coord("distances", np.linspace(0, 200, 500) * u.nm, "m")
    result.add("temp_map", 300 + rng.random((1000, 500, 2)), ("delays", "distances"), "K")
    return result


# benchmarks


def test_result_save(benchmark, result, tmp_path):
    pytest.importorskip("h5py")
    benchmark(result.save, tmp_path / "result.h5")


def test_result_load(benchmark, result, tmp_path):
    pytest.importorskip("h5py")
    filename = tmp_path / "result.h5"
    result.save(filename)
    benchmark(Result.load, filename)
