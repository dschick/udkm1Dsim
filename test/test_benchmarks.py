#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CodSpeed performance benchmarks for udkm1Dsim.

These benchmarks focus on the pure numerical routines that sit on the hot
path of the toolbox (matrix helpers, binary search, coordinate conversions)
as well as a couple of representative object-level workloads (creating atoms
and computing energy-dependent atomic form factors). They are intentionally
deterministic and CPU-bound so they measure well with CodSpeed's simulation
instrument.
"""

import numpy as np

from udkm1Dsim import Atom
from udkm1Dsim.helpers import (
    make_hash_md5,
    m_power_x,
    m_times_n,
    finderb,
    multi_gauss,
    convert_cartesian_to_polar,
    convert_polar_to_cartesian,
)


# helpers: numerical hot paths


def test_m_power_x(benchmark):
    m = np.reshape(np.arange(2 * 8 * 8, dtype=float), (2, 1, 8, 8))
    m += np.eye(8)
    benchmark(lambda: m_power_x(m.copy(), 5))


def test_m_times_n(benchmark):
    rng = np.random.default_rng(42)
    m = rng.random((50, 50, 3, 3))
    n = rng.random((50, 50, 3, 3))
    benchmark(m_times_n, m, n)


def test_finderb(benchmark):
    array = np.linspace(0, 1000, 10000)
    keys = np.linspace(0, 1000, 5000)
    benchmark(finderb, keys, array)


def test_multi_gauss(benchmark):
    x = np.linspace(-50, 50, 20000)
    s = [1, 2, 3, 4, 5]
    x0 = [-10, -5, 0, 5, 10]
    A = [1, 2, 3, 2, 1]
    benchmark(multi_gauss, x, s, x0, A)


def test_convert_cartesian_to_polar(benchmark):
    rng = np.random.default_rng(7)
    cartesian = rng.random((10000, 3))
    benchmark(convert_cartesian_to_polar, cartesian)


def test_convert_polar_to_cartesian(benchmark):
    rng = np.random.default_rng(7)
    polar = rng.random((10000, 3))
    benchmark(convert_polar_to_cartesian, polar)


def test_make_hash_md5(benchmark):
    obj = {'a': list(range(100)), 'b': {'nested': tuple(range(100))}, 'c': set(range(100))}
    benchmark(make_hash_md5, obj)


# object-level workloads


def test_atom_creation(benchmark):
    benchmark(Atom, 'Fe')


def test_atom_get_atomic_form_factor(benchmark):
    atom = Atom('Fe')
    energies = np.linspace(1000, 10000, 2000)
    benchmark(atom.get_atomic_form_factor, energies)
