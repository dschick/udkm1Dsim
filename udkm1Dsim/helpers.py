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
           'm_times_n', 'finderb']

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
            for j in range(np.size(m, 3)):
                m[:, :, i, j] = np.linalg.matrix_power(m[:, :, i, j], x)
    return m


def m_times_n(m, n):
    return np.einsum("ijlm,jklm->iklm", m, n)


def finderb(key, vector):
    """finderb

    Binary search algorithm for sorted vector. Searches for the first
    index ``i`` of vector where ``key`` >= ``vector[i]``.
    ``key`` can be a scalar or a np.ndarray of keys.
    ``vector`` must be a sorted np.ndarray

    author: André Bojahr
    licence: BSD

    """
    key = np.array(key, ndmin=1)
    n = len(key)
    i = np.zeros([n], dtype=int)

    for m in range(n):
        i[m] = finderb_nest(key[m], vector)
    return i


def finderb_nest(key, vector):
    """finderb_nest

    nested sub-function of finderb

    """
    a = 0  # start of intervall
    b = len(vector)  # end of intervall

    # if the key is smaller than the first element of the
    # vector we return 1
    if key < vector[0]:
        return 1

    while (b-a) > 1:  # loop until the intervall is larger than 1
        c = int(np.floor((a+b)/2))  # center of intervall
        if key < vector[c]:
            # the key is in the left half-intervall
            b = c
        else:
            # the key is in the right half-intervall
            a = c

    return a
