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

__all__ = ['make_hash_md5', 'make_hashable', 'm_power_x',
           'm_times_n', 'finderb', 'multi_gauss', 'convert_cartesian_to_polar',
           'convert_polar_to_cartesian']

__docformat__ = 'restructuredtext'

import hashlib
import numpy as np


def make_hash_md5(obj):
    """make_hash_md5


    Args:
        obj (any): anything that can be hashed.

    Returns:
        hash (str): hash from object.

    """
    hasher = hashlib.md5()
    hasher.update(repr(make_hashable(obj)).encode())
    return hasher.hexdigest()


def make_hashable(obj):
    """make_hashable

    Recursive calls to elements of tuples, lists, dicts, set, and frozensets.

    Args:
        obj (any): anything that can be hashed..

    Returns:
        obj (tuple): hashable object.

    """
    if isinstance(obj, (tuple, list)):
        return tuple((make_hashable(e) for e in obj))

    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))

    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in obj))

    return obj


def m_power_x(m, x):
    """m_power_x

    Apply ``numpy.linalg.matrix_power`` to last 2 dimensions of 4-dimensional
    input matrix.

    Args:
        m (ndarray[float, complex]): 4-dimensional input matrix.
        x (float): exponent.

    Returns:
        m (ndarray[float, complex]): resulting matrix.

    """
    if x > 1:
        for i in range(np.size(m, 0)):
            for j in range(np.size(m, 1)):
                m[i, j, :, :] = np.linalg.matrix_power(m[i, j, :, :], x)
    return m


def m_times_n(m, n):
    """m_times_n

    Matrix multiplication of last 2 dimensions for two 4-dimensional input
    matrices.

    Args:
        m (ndarray[float, complex]): 4-dimensional input matrix.
        n (ndarray[float, complex]): 4-dimensional input matrix.

    Returns:
        res (ndarray[float, complex]): 4-dimensional multiplication result.

    """
    return np.einsum("lmij,lmjk->lmik", m, n)


def finderb(key, array):
    """finderb

    Binary search algorithm for sorted array. Searches for the first index
    ``i`` of array where ``key`` >= ``array[i]``. ``key`` can be a scalar or
    a np.ndarray of keys. ``array`` must be a sorted np.ndarray.

    Author: André Bojahr.
    Licence: BSD.

    Args:
        key (float, ndarray[float]): single or multiple sorted keys.
        array (ndarray[float]): sorted array.

    Returns:
        i (ndarray[float]): position indices for each key in the array.

    """
    key = np.array(key, ndmin=1)
    n = len(key)
    i = np.zeros([n], dtype=int)

    for m in range(n):
        i[m] = finderb_nest(key[m], array)
    return i


def finderb_nest(key, array):
    """finderb_nest

    Nested sub-function of :func:`.finderb` for one single key.

    Author: André Bojahr.
    Licence: BSD.

    Args:
        key (float): single key.
        array (ndarray[float]): sorted array.

    Returns:
        a (float): position index of key in the array.

    """
    a = 0  # start of intervall
    b = len(array)  # end of intervall

    # if the key is smaller than the first element of the
    # vector we return 1
    if key < array[0]:
        return 0

    while (b-a) > 1:  # loop until the intervall is larger than 1
        c = int(np.floor((a+b)/2))  # center of intervall
        if key < array[c]:
            # the key is in the left half-intervall
            b = c
        else:
            # the key is in the right half-intervall
            a = c

    return a


def multi_gauss(x, s=[1], x0=[0], A=[1]):
    """multi_gauss

    Multiple gauss functions with width ``s`` given as FWHM and area normalized
    to input ``A`` and maximum of gauss at ``x0``.

    Args:
        x (ndarray[float]): argument of multi_gauss.
        s (ndarray[float], optional): FWHM of Gaussians. Defaults to 1.
        x0 (ndarray[float], optional): centers of Gaussians. Defaults to 0.
        A (ndarray[float], optional): amplitudes of Gaussians. Defaults to 1.

    Returns:
        y (ndarray[float]): multiple Gaussians.

    """
    s = np.asarray(s)/(2*np.sqrt(2*np.log(2)))
    a = np.asarray(A)/np.sqrt(2*np.pi*s**2)  # normalize area to 1
    x0 = np.asarray(x0)

    y = np.zeros_like(x)
    for i in range(len(s)):
        y = y + a[i] * np.exp(-((x-x0[i])**2)/(2*s[i]**2))
    return y


def convert_polar_to_cartesian(polar):
    r"""convert_polar_to_cartesian

    Convert a vector or field from polar coordinates
    :math:`(r, \phi, \gamma)` to cartesian coordinates :math:`(x, y, z)`:

    .. math::

        F_x & = r \sin(\phi)\cos(\gamma) \\
        F_y & = r \sin(\phi)\sin(\gamma) \\
        F_z & = r \cos(\phi)

    where :math:`r`, :math:`\phi`, :math:`\gamma` are the radius
    (amplitude), azimuthal, and polar angles of vector field
    :math:`\mathbf{F}`, respectively.

    Args:
        polar (ndarray[float]): vector of field to convert.

    Returns:
        cartesian (ndarray[float]): converted vector or field.

    """
    cartesian = np.zeros_like(polar)

    amplitudes = polar[..., 0]
    phis = polar[..., 1]
    gammas = polar[..., 2]
    cartesian[..., 0] = amplitudes*np.sin(phis)*np.cos(gammas)
    cartesian[..., 1] = amplitudes*np.sin(phis)*np.sin(gammas)
    cartesian[..., 2] = amplitudes*np.cos(phis)

    return cartesian


def convert_cartesian_to_polar(cartesian):
    r"""convert_cartesian_to_polar

    Convert a vector or field from cartesian coordinates :math:`(x, y, z)`
    to polar coordinates :math:`(r, \phi, \gamma)`:

    .. math::

        F_r & = \sqrt{F_x^2 + F_y^2+F_z^2}\\
        F_{\phi} & = \begin{cases}\\
        \arctan\left(\frac{F_y}{F_x} \right) & \mathrm{for}\ F_x > 0 \\
        \pi + \arctan\left(\frac{F_y}{F_x}\right)
        & \mathrm{for}\ F_x < 0 \ \mathrm{and}\ F_y \geq 0 \\
        \arctan\left(\frac{F_y}{F_x}\right) - \pi
        & \mathrm{for}\ F_x < 0 \ \mathrm{and}\ F_y < 0 \\
        0 & \mathrm{for}\ F_x = F_y = 0
        \end{cases} \\
        F_{\gamma} & = \arccos\left(\frac{F_z}{F_r} \right)

    where :math:`F_r`, :math:`F_{\phi}`, :math:`F_{\gamma}` are the radial
    (amplitude), azimuthal, and polar component of vector field
    :math:`\mathbf{F}`, respectively.

    Args:
        cartesian (ndarray[float]): vector of field to convert.

    Returns:
        polar (ndarray[float]): converted vector or field.

    """
    polar = np.zeros_like(cartesian)
    xs = cartesian[..., 0]
    ys = cartesian[..., 1]
    zs = cartesian[..., 2]
    amplitudes = np.sqrt(xs**2 + ys**2 + zs**2)
    mask = amplitudes != 0.  # mask for non-zero amplitudes
    polar[..., 0] = amplitudes
    polar[mask, 1] = np.arccos(np.divide(zs[mask], amplitudes[mask]))
    polar[..., 2] = np.arctan2(ys, xs)

    return polar
