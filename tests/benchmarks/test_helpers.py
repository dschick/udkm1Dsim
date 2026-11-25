#!/usr/bin/env python

import numpy as np
import pytest

from udkm1Dsim.helpers import (
    convert_cartesian_to_polar,
    convert_polar_to_cartesian,
    finderb,
    m_power_x,
    m_times_n,
    make_hash_md5,
)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


# the inputs are created outside of the benchmarked functions


@pytest.mark.benchmark
def test_make_hash_md5():
    make_hash_md5([{"item1": np.ones((10, 10, 10))}])


def test_m_power_x(benchmark, rng):
    benchmark(m_power_x, rng.random((25, 25, 4, 4)), 25)


def test_m_times_n(benchmark, rng):
    benchmark(m_times_n, rng.random((100, 10, 4, 4)), rng.random((100, 10, 4, 4)))


def test_finderb(benchmark, rng):
    benchmark(finderb, rng.random(100), np.linspace(0, 1, 10000))


def test_convert_polar_to_cartesian(benchmark, rng):
    benchmark(convert_polar_to_cartesian, rng.random((100, 100, 3)))


def test_convert_cartesian_to_polar(benchmark, rng):
    benchmark(convert_cartesian_to_polar, rng.random((100, 100, 3)))
