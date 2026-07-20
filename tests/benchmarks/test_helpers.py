#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim.helpers import m_power_x, m_times_n
from udkm1Dsim.helpers import finderb, make_hash_md5
from udkm1Dsim.helpers import convert_cartesian_to_polar, convert_polar_to_cartesian
import pytest
import numpy as np


@pytest.mark.benchmark
def test_make_hash_md5():
    make_hash_md5([{'item1': np.ones((10, 10, 10))}])


@pytest.mark.benchmark
def test_m_power_x():
    m_power_x(np.random.rand(25, 25, 4, 4), 25)


@pytest.mark.benchmark
def test_m_times_n():
    m_times_n(
        np.random.rand(100, 10, 4, 4),
        np.random.rand(100, 10, 4, 4)
        )


@pytest.mark.benchmark
def test_finderb():
    finderb(np.random.rand(100), np.linspace(0, 1, 10000))


@pytest.mark.benchmark
def test_convert_polar_to_cartesian():
    convert_polar_to_cartesian(np.random.rand(100, 100, 3))


@pytest.mark.benchmark
def test_convert_cartesian_to_polar():
    convert_cartesian_to_polar(np.random.rand(100, 100, 3))
