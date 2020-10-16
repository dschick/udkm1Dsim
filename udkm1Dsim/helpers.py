#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
# Copyright (c) 2020 Daniel Schick
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

"""A :mod:`helpers` module """

__all__ = ['make_hash_md5', 'make_hashable', 'm_power_x',
           'm_times_n', 'finderb', 'multi_gauss']

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
        for i in range(np.size(m, 0)):
            for j in range(np.size(m, 1)):
                m[i, j, :, :] = np.linalg.matrix_power(m[i, j, :, :], x)
    return m


def m_times_n(m, n):
    return np.einsum("lmij,lmjk->lmik", m, n)


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
        return 0

    while (b-a) > 1:  # loop until the intervall is larger than 1
        c = int(np.floor((a+b)/2))  # center of intervall
        if key < vector[c]:
            # the key is in the left half-intervall
            b = c
        else:
            # the key is in the right half-intervall
            a = c

    return a


def multi_gauss(x, s=1, x0=0, A=1):
    """multi_gauss

    multiple gauss functions with width given as FWHM and area normalized
    to one and maximum of gauss at x0.

    """
    s = np.asarray(s)/(2*np.sqrt(2*np.log(2)))
    a = np.asarray(A)/np.sqrt(2*np.pi*s**2)  # normalize area to 1
    x0 = np.asarray(x0)

    y = np.zeros_like(x)
    for i in range(len(s)):
        y = y + a[i] * np.exp(-((x-x0[i])**2)/(2*s[i]**2))
    return y
