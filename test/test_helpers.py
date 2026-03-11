#!/usr/bin/env python
# -*- coding: utf-8 -*-
from udkm1Dsim.helpers import make_hash_md5, m_power_x, m_times_n
from udkm1Dsim.helpers import finderb, multi_gauss
from udkm1Dsim.helpers import convert_cartesian_to_polar, convert_polar_to_cartesian
import numpy as np


def test_make_hash_md5():
    assert make_hash_md5('test') == '2f4a8dbd4cdc82139c47d0df78b540ac'
    assert make_hash_md5(123456) == 'e10adc3949ba59abbe56e057f20f883e'


def test_m_power_x():
    m = np.arange(4)
    m = np.reshape(m, (1, 1, 2, 2))
    result = np.zeros_like(m)
    result[0, 0, 0, 0] = 2
    result[0, 0, 0, 1] = 3
    result[0, 0, 1, 0] = 6
    result[0, 0, 1, 1] = 11
    assert np.allclose(m_power_x(m, 2), result)


def test_m_times_n():
    m = np.arange(4)
    m = np.reshape(m, (1, 1, 2, 2))
    n = 2*m
    result = np.zeros_like(m)
    result[0, 0, 0, 0] = 4
    result[0, 0, 0, 1] = 6
    result[0, 0, 1, 0] = 12
    result[0, 0, 1, 1] = 22
    assert np.allclose(m_times_n(m, n), result)


def test_finderb():
    assert np.allclose(finderb([1.1, 2.2, 3.3, 4.4, 5.5], np.array([1, 2, 3, 4, 5])),
                       [0, 1, 2, 3, 4])


def test_multi_gauss():
    assert np.allclose(multi_gauss(np.r_[-1:1:0.1]),
                       [0.05871483, 0.09943301, 0.15930558, 0.24146211, 0.34624587, 0.46971864,
                        0.60284907, 0.73197625, 0.84081992, 0.91374832, 0.93943728, 0.91374832,
                        0.84081992, 0.73197625, 0.60284907, 0.46971864, 0.34624587, 0.24146211,
                        0.15930558, 0.09943301])


def test_convert_cartesian_to_polar():
    assert np.allclose(convert_cartesian_to_polar(np.array([0., 0., 0.])),
                       np.array([0., 0., 0.]))
    assert np.allclose(convert_cartesian_to_polar(np.array([1., 0., 0.])),
                       np.array([1., 1.57079633, 0]))
    assert np.allclose(convert_cartesian_to_polar(np.array([0., 1., 0.])),
                       np.array([1., 1.57079633, 1.57079633]))
    assert np.allclose(convert_cartesian_to_polar(np.array([0., 0., 1.])),
                       np.array([1., 0., 0.]))
    assert np.allclose(convert_cartesian_to_polar(np.array([1., 1., 1.])),
                       np.array([1.73205081, 0.95531662, 0.78539816]))


def test_convert_polar_to_cartesian():
    assert np.allclose(convert_polar_to_cartesian(np.array([0., 0., 0.])),
                       np.array([0., 0., 0.]))
    assert np.allclose(convert_polar_to_cartesian(np.array([1., np.deg2rad(90), 0.])),
                       np.array([1., 0., 0.]))
    assert np.allclose(convert_polar_to_cartesian(np.array([1., 0., np.deg2rad(90)])),
                       np.array([0., 0., 1.]))
    assert np.allclose(convert_polar_to_cartesian(np.array([1., np.deg2rad(90), np.deg2rad(90)])),
                       np.array([0., 1., 0.]))
