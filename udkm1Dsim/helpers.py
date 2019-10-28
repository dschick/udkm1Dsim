#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the udkm1Dsimpy module.
#
# udkm1Dsimpy is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2017 Daniel Schick

"""A :mod:`helpers` module """

__all__ = ['make_hash_md5', 'make_hashable', 'm_power_x',
           'm_times_n']

__docformat__ = "restructuredtext"

import hashlib
import numpy as np


def make_hash_md5(o):
    hasher = hashlib.md5()
    hasher.update(repr(make_hashable(o)).encode())
    return hasher.hexdigest()


def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))

    return o


def m_power_x(m, x):
    # apply exponent to each matrix in a numpy array
    if x > 1:
        for i in range(np.size(m, 2)):
            m[:, :, i] = np.linalg.matrix_power(m[:, :, i], x)
    return m


def m_times_n(m, x):
    return np.einsum("ijl,jkl->ikl", m, x)
