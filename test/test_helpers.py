#!/usr/bin/env python
# -*- coding: utf-8 -*-
from udkm1Dsim.helpers import make_hash_md5
# , m_power_x, m_times_n
from udkm1Dsim.helpers import finderb
# , multi_gauss
# from udkm1Dsim.helpers import convert_cartesian_to_polar, convert_polar_to_cartesian
# from udkm1Dsim import u
# u.default_format = '~P'
import numpy as np
# import pytest
# from pint.testing import assert_allclose


def test_make_hash_md5():
    assert make_hash_md5('test') == '2f4a8dbd4cdc82139c47d0df78b540ac'


def test_finderb():
    assert np.allclose(finderb([1.1, 2.2, 3.3, 4.4, 5.5], np.array([1, 2, 3, 4, 5])),
                       [0, 1, 2, 3, 4])
