#!/usr/bin/env python
import numpy as np
import pytest
from scipy.special import erf

import udkm1Dsim as ud
from udkm1Dsim.helpers import (
    RemovedAttribute,
    check_keys,
    conv_with_function,
    convert_cartesian_to_polar,
    convert_polar_to_cartesian,
    finderb,
    m_power_x,
    m_times_n,
    make_hash_md5,
    multi_gauss,
    to_angle,
)


def test_make_hash_md5():
    assert make_hash_md5("test") == "2f4a8dbd4cdc82139c47d0df78b540ac"
    assert make_hash_md5(123456) == "e10adc3949ba59abbe56e057f20f883e"


def test_make_hash_md5_depends_on_all_values_of_arrays():
    """the repr of an array shows only its corners and rounded values"""
    a = np.zeros((100, 100))
    b = a.copy()
    b[50, 50] = 1
    assert make_hash_md5([a]) != make_hash_md5([b])
    assert make_hash_md5(np.ones(3)) != make_hash_md5(np.ones(3) + 1e-12)
    assert make_hash_md5(np.ones(4)) != make_hash_md5(np.ones((2, 2)))
    assert make_hash_md5({"a": a}) == make_hash_md5({"a": a.copy()})
    assert make_hash_md5(np.ones(3) * ud.u.nm) != make_hash_md5(np.ones(3) * ud.u.m)
    # arrays of objects, e.g. ragged strain vectors, and sets
    ragged = np.array([np.r_[0, 1], np.r_[0]], dtype=object)
    changed = np.array([np.r_[0, 2], np.r_[0]], dtype=object)
    assert make_hash_md5(ragged) != make_hash_md5(changed)
    assert make_hash_md5({1, 2}) == make_hash_md5({2, 1})
    assert make_hash_md5({1, 2}) != make_hash_md5({1, 3})


def test_m_power_x():
    m = np.arange(4)
    m = np.reshape(m, (1, 1, 2, 2))
    result = np.zeros_like(m)
    result[0, 0, 0, 0] = 2
    result[0, 0, 0, 1] = 3
    result[0, 0, 1, 0] = 6
    result[0, 0, 1, 1] = 11
    assert np.allclose(m_power_x(m, 2), result)
    # the input is kept, e.g. for repeated calls
    np.testing.assert_array_equal(m, np.reshape(np.arange(4), (1, 1, 2, 2)))
    rng = np.random.default_rng(0)
    stack = rng.random((3, 2, 4, 4))
    powers = m_power_x(stack, 5)
    np.testing.assert_allclose(powers[2, 1], np.linalg.matrix_power(stack[2, 1], 5))


def test_m_times_n():
    m = np.arange(4)
    m = np.reshape(m, (1, 1, 2, 2))
    n = 2 * m
    result = np.zeros_like(m)
    result[0, 0, 0, 0] = 4
    result[0, 0, 0, 1] = 6
    result[0, 0, 1, 0] = 12
    result[0, 0, 1, 1] = 22
    assert np.allclose(m_times_n(m, n), result)


def test_finderb():
    assert np.allclose(
        finderb([1.1, 2.2, 3.3, 4.4, 5.5], np.array([1, 2, 3, 4, 5])), [0, 1, 2, 3, 4]
    )
    # keys below the first and above the last element and on the elements
    np.testing.assert_array_equal(finderb([0.5, 3, 9], np.arange(1, 6)), [0, 2, 4])


def test_multi_gauss():
    assert np.allclose(
        multi_gauss(np.r_[-1:1:0.1]),
        [
            0.05871483,
            0.09943301,
            0.15930558,
            0.24146211,
            0.34624587,
            0.46971864,
            0.60284907,
            0.73197625,
            0.84081992,
            0.91374832,
            0.93943728,
            0.91374832,
            0.84081992,
            0.73197625,
            0.60284907,
            0.46971864,
            0.34624587,
            0.24146211,
            0.15930558,
            0.09943301,
        ],
    )


def test_multi_gauss_of_several_gaussians():
    """each Gaussian has its own FWHM, center, and area"""
    x = np.linspace(-20, 20, 40001)
    y = multi_gauss(x, s=[1, 2], x0=[-3, 4], A=[1, 2])
    np.testing.assert_allclose(np.sum(y) * (x[1] - x[0]), 3, rtol=1e-6)
    # with the tail of 4e-5 of the second Gaussian
    np.testing.assert_allclose(np.sum(y[x < 0.5]) * (x[1] - x[0]), 1, rtol=1e-4)
    left, right = x < 0.5, x >= 0.5
    assert x[left][np.argmax(y[left])] == pytest.approx(-3)
    assert x[right][np.argmax(y[right])] == pytest.approx(4)
    # the FWHM of the second Gaussian is 2
    half = y[right] >= np.max(y[right]) / 2
    assert x[right][half][-1] - x[right][half][0] == pytest.approx(2, abs=2e-3)


def test_convert_cartesian_to_polar():
    assert np.allclose(
        convert_cartesian_to_polar(np.array([0.0, 0.0, 0.0])), np.array([0.0, 0.0, 0.0])
    )
    # also zero vectors with signed zero components have zero angles
    assert np.array_equal(
        convert_cartesian_to_polar(np.array([-0.0, -0.0, 0.0])), np.array([0.0, 0.0, 0.0])
    )
    assert np.allclose(
        convert_cartesian_to_polar(np.array([1.0, 0.0, 0.0])), np.array([1.0, 1.57079633, 0])
    )
    assert np.allclose(
        convert_cartesian_to_polar(np.array([0.0, 1.0, 0.0])),
        np.array([1.0, 1.57079633, 1.57079633]),
    )
    assert np.allclose(
        convert_cartesian_to_polar(np.array([0.0, 0.0, 1.0])), np.array([1.0, 0.0, 0.0])
    )
    assert np.allclose(
        convert_cartesian_to_polar(np.array([1.0, 1.0, 1.0])),
        np.array([1.73205081, 0.95531662, 0.78539816]),
    )
    # the other quadrants of gamma and the lower half space of phi
    np.testing.assert_allclose(
        convert_cartesian_to_polar(np.array([-1.0, 0.0, 0.0])), [1, np.pi / 2, np.pi]
    )
    np.testing.assert_allclose(
        convert_cartesian_to_polar(np.array([0.0, -1.0, 0.0])), [1, np.pi / 2, -np.pi / 2]
    )
    np.testing.assert_allclose(
        convert_cartesian_to_polar(np.array([0.0, 0.0, -1.0])), [1, np.pi, 0], atol=1e-15
    )
    np.testing.assert_allclose(
        convert_cartesian_to_polar(np.array([-1.0, -1.0, -1.0])),
        [np.sqrt(3), np.pi - 0.95531662, -3 * np.pi / 4],
    )


def test_convert_polar_and_cartesian_round_trip():
    """arrays of any shape with the components in the last dimension"""
    rng = np.random.default_rng(0)
    cartesian = rng.uniform(-1, 1, (4, 5, 3))
    polar = convert_cartesian_to_polar(cartesian)
    assert polar.shape == (4, 5, 3)
    np.testing.assert_allclose(convert_polar_to_cartesian(polar), cartesian, atol=1e-15)


def test_convert_polar_to_cartesian():
    assert np.allclose(
        convert_polar_to_cartesian(np.array([0.0, 0.0, 0.0])), np.array([0.0, 0.0, 0.0])
    )
    assert np.allclose(
        convert_polar_to_cartesian(np.array([1.0, np.deg2rad(90), 0.0])), np.array([1.0, 0.0, 0.0])
    )
    assert np.allclose(
        convert_polar_to_cartesian(np.array([1.0, 0.0, np.deg2rad(90)])), np.array([0.0, 0.0, 1.0])
    )
    assert np.allclose(
        convert_polar_to_cartesian(np.array([1.0, np.deg2rad(90), np.deg2rad(90)])),
        np.array([0.0, 1.0, 0.0]),
    )


def test_read_columns(tmp_path):
    from udkm1Dsim.helpers import read_columns

    whitespace = tmp_path / "data.txt"
    whitespace.write_text("# comment\nE f1 f2\n1 2 3\n4 5 6\n")
    comma = tmp_path / "data.csv"
    comma.write_text("E,f1,f2,extra\n1,2,3,9\n4,5,6,9\n")
    for filename in (whitespace, comma):
        np.testing.assert_allclose(read_columns(filename), [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="at least 4 columns"):
        read_columns(whitespace, columns=4)


def test_spectral_table():
    from udkm1Dsim import u
    from udkm1Dsim.helpers import SpectralTable

    table = SpectralTable([3, 1, 2], [3 + 3j, 1 + 1j, np.nan], source="test")
    np.testing.assert_allclose(table.energy, [1, 2, 3])
    assert table.energy_range == (1, 3)
    assert np.isnan(table([0.5, 1.5, 4])).all()
    # wavelengths are converted to photon energies
    table = SpectralTable([400, 800] * u.nm, [1, 2])
    np.testing.assert_allclose(table(1.55)[0], 2, rtol=1e-3)
    # columns with invalid entries and the sign of the imaginary part
    table = SpectralTable.from_columns(
        [[1, -9999, 1], [2, 3, 4], [3, 5, 6]], invalid_below=-9000, imaginary_sign=-1
    )
    assert table.energy_range == (2, 3)
    np.testing.assert_allclose(table([2.5]), [4 - 5j])
    assert "md5" in repr(table) and len(table) == 3


def test_conv_with_function():
    def handle(x):
        return np.exp(-(x**2))

    x = np.r_[0:10:1]
    y = np.r_[0:10:1]
    # symmetric about the center, including the last point, with the first
    # and last value held beyond the ends
    np.testing.assert_allclose(
        conv_with_function(y, x, handle),
        [
            0.22840624,
            1.01047185,
            2.00006975,
            3.00000006,
            4.0,
            5.0,
            5.99999994,
            6.99993025,
            7.98952815,
            8.77159376,
        ],
        rtol=1e-7,
    )


def test_conv_with_function_has_no_shift_and_keeps_the_ends():
    """a step on an even number of points is convolved to the error
    function at its position, and a constant stays constant at the ends"""
    sigma = 0.5

    def handle(x):
        return np.exp(-(x**2) / (2 * sigma**2))

    x = np.linspace(-10, 10, 100)
    step = conv_with_function((x > 0).astype(float), x, handle)
    np.testing.assert_allclose(step, 0.5 * (1 + erf(x / (np.sqrt(2) * sigma))), atol=2e-3)
    np.testing.assert_allclose(step + step[::-1], 1, atol=1e-12)
    np.testing.assert_allclose(conv_with_function(np.ones(100), x, handle), 1)
    # a non-equidistant grid is interpolated to its smallest spacing
    x = np.r_[0:5:0.5, 5:20:1.5]
    np.testing.assert_allclose(conv_with_function(np.full(len(x), 2.0), x, handle), 2)


def test_to_angle():
    u = ud.u
    np.testing.assert_allclose(to_angle(90 * u.deg), np.pi / 2)
    np.testing.assert_allclose(to_angle(np.pi * u.rad, "deg"), 180)
    np.testing.assert_allclose(to_angle(np.r_[0, 30] * u.deg, "deg"), [0, 30])
    np.testing.assert_allclose(to_angle([1 * u.mrad, 1 * u.deg], "mrad"), [1, np.pi * 1e3 / 180])
    # the unit of an inverse function is rad
    np.testing.assert_allclose(to_angle(np.arcsin(0.5 * u.dimensionless), "deg"), 30)
    # bare numbers, dimensionless and other quantities are no angles
    for value in (0, 0.5, np.r_[0.5], [0.5 * u.deg, 1], 0.5 * u.dimensionless, 1 * u.sr, 1 * u.m):
        with pytest.raises(TypeError, match="alpha must be given"):
            to_angle(value, name="alpha")


def test_check_keys():
    check_keys({"a": 1}, ["a", "b"], "f()")
    with pytest.raises(
        TypeError, match=r"^f\(\) got an unexpected keyword argument 'bb' \(did you mean 'b'\?\)!$"
    ):
        check_keys({"a": 1, "bb": 2}, ["a", "b"], "f()")
    with pytest.raises(
        TypeError, match=r"^d got unexpected keys 'bb' \(did you mean 'b'\?\), 'xyz'!$"
    ):
        check_keys(["xyz", "bb"], ["a", "b"], "d", kind="key")


def test_removed_attribute():
    """an assignment of a removed attribute would create an unused attribute"""

    class Owner:
        old = RemovedAttribute("use new instead")

    owner = Owner()
    message = r"'Owner.old' was removed in version 3.0, use new instead!"
    with pytest.raises(AttributeError, match=message):
        owner.old
    with pytest.raises(AttributeError, match=message):
        owner.old = 1
    with pytest.raises(AttributeError, match=message):
        del owner.old
    assert "old" not in vars(owner)
    assert not hasattr(owner, "old")
    assert isinstance(Owner.old, RemovedAttribute)
