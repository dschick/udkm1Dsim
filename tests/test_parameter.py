#!/usr/bin/env python

import numpy as np
import pytest
import sympy as sp
import pint
from pint.testing import assert_allclose as assert_approx
from pint.testing import assert_equal

from udkm1Dsim import u
from udkm1Dsim import VectorParameter

# tests

# Parameter

@pytest.mark.parametrize(
    "input, expected",
    [
        (1, 1.0),
        (1.0, 1.0),
        (1.0 * u.m, 1.0),
        (1.0 * u.nm, 1.0e-9),
    ],
)
def test_parameter(parameter, input, expected):
    parameter.quantity = input
    assert parameter.magnitude == expected


# TemperatureParameter


@pytest.mark.parametrize(
    "input, expected, expected_expr",
    [
        (1, 1.0, "1.0"),
        (1.0, 1.0, "1.0"),
        (1.0 * u.m, 1.0, "1.0"),
        (1.0 * u.nm, 1.0e-9, "1e-09"),
        ("1", 1.0, "1"),
        ("1.0", 1.0, "1.0"),
        ("1.0*T", 300.0, "1.0*T"),
        ("1.0*T_0", 300.0, "1.0*T_0"),
        ("lambda T: 1.0*T", 300.0, "1.0*T"),
    ],
)
def test_temperature_parameter(temperature_parameter, input, expected, expected_expr):
    temperature_parameter.quantity = input
    if isinstance(input, str) and "_" in input:
        assert temperature_parameter.functional[0]([300]) == expected
    else:
        assert temperature_parameter.functional[0](300) == expected
    assert temperature_parameter.quantity[0].equals(sp.sympify(expected_expr))


@pytest.mark.parametrize(
    "input, expected, expected_expr",
    [
        ([1, 2, 3], [1, 2, 3], ["1.0", "2.0", "3.0"]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], ["1.0", "2.0", "3.0"]),
        ([1.0 * u.m, 2.0 * u.m, 3.0 * u.m], [1.0, 2.0, 3.0], ["1.0", "2.0", "3.0"]),
        (
            [1.0 * u.nm, 2.0 * u.nm, 2.0 * u.nm],
            [1.0e-9, 2.0e-9, 2.0e-9],
            ["1e-09", "2e-09", "2e-09"],
        ),
        (["1", "2", "3"], [1.0, 2.0, 3.0], ["1", "2", "3"]),
        (["1.0", "2.0", "3.0"], [1.0, 2.0, 3.0], ["1.0", "2.0", "3.0"]),
        (["1.0*T", "2.0*T", "3.0*T"], [300.0, 600.0, 900.0], ["1.0*T", "2.0*T", "3.0*T"]),
        (
            ["1.0*T_0", "2.0*T_1", "3.0*T_2"],
            [300.0, 600.0, 900.0],
            ["1.0*T_0", "2.0*T_1", "3.0*T_2"],
        ),
        (
            ["lambda T: 1.0*T", "lambda T: 2.0*T", "lambda T: 3.0*T"],
            [300.0, 600.0, 900.0],
            ["1.0*T", "2.0*T", "3.0*T"],
        ),
    ],
)
def test_temperature_parameter_list(temperature_parameter, input, expected, expected_expr):
    temperature_parameter.quantity = input
    for i, (functional, expression) in enumerate(zip(temperature_parameter.functional, temperature_parameter.magnitude)):
        if isinstance(input[i], str) and "_" in input[i]:
            assert functional([300, 300, 300]) == expected[i]
        else:
            assert functional(300) == expected[i]
        assert expression.equals(sp.sympify(expected_expr[i]))


# VectorParameter


class DummyCaller:
    """Counts the notifications a Parameter sends to its owner."""

    def __init__(self):
        self.calls = 0

    def _update_depending(self):
        self.calls += 1


def test_vector_parameter_default(vector_parameter):
    np.testing.assert_array_equal(vector_parameter.magnitude, [0.0, 0.0, 0.0])
    assert vector_parameter.unit == u.T
    assert vector_parameter.name == "B"
    assert vector_parameter.angle_unit == u.deg


def test_vector_parameter_init():
    # integer input must not be truncated by the conversion helpers
    vector = VectorParameter("mT", (1, 1, 1), name="B")
    assert vector.magnitude.dtype == float
    np.testing.assert_allclose(vector.r.magnitude, np.sqrt(3))
    assert vector.r.units == u.mT


@pytest.mark.parametrize("attribute", ["magnitude", "quantity", "cartesian"])
@pytest.mark.parametrize(
    "input, expected",
    [
        ((1, 2, 3), [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        (np.array([1, 2, 3]), [1.0, 2.0, 3.0]),
        (np.array([-1.5, 0.0, 2.5]), [-1.5, 0.0, 2.5]),
        (u.Quantity([1.0, 2.0, 3.0], "T"), [1.0, 2.0, 3.0]),
        (u.Quantity([1.0, 2.0, 3.0], "mT"), [1.0e-3, 2.0e-3, 3.0e-3]),
    ],
)
def test_vector_parameter_cartesian(vector_parameter, attribute, input, expected):
    setattr(vector_parameter, attribute, input)
    assert vector_parameter.magnitude.dtype == float
    np.testing.assert_allclose(vector_parameter.magnitude, expected)
    assert_approx(vector_parameter.quantity, u.Quantity(expected, "T"))
    assert_approx(vector_parameter.cartesian, u.Quantity(expected, "T"))


@pytest.mark.parametrize(
    "input, error",
    [
        ((1, 2), ValueError),
        ((1, 2, 3, 4), ValueError),
        (np.ones((2, 3)), ValueError),
        (1.0, ValueError),
        (u.Quantity([1.0, 2.0], "T"), ValueError),
        (u.Quantity([1.0, 2.0, 3.0], "m"), pint.DimensionalityError),
        ("abc", TypeError),
        ((1, 2, "a"), TypeError),
        ((1 + 1j, 2, 3), TypeError),
        (None, TypeError),
    ],
)
def test_vector_parameter_invalid(vector_parameter, input, error):
    vector_parameter.magnitude = (1.0, 2.0, 3.0)
    with pytest.raises(error):
        vector_parameter.magnitude = input
    # a failed set leaves the old value untouched
    np.testing.assert_array_equal(vector_parameter.magnitude, [1.0, 2.0, 3.0])


def test_vector_parameter_magnitude_is_copy(vector_parameter):
    source = np.array([1.0, 2.0, 3.0])
    vector_parameter.magnitude = source
    source[0] = 99.0  # input is not aliased
    vector_parameter.magnitude[1] = 99.0  # returned array is a copy
    np.testing.assert_array_equal(vector_parameter.magnitude, [1.0, 2.0, 3.0])


# polar: (r, phi, gamma) with phi from +z and gamma in the xy-plane from +x


@pytest.mark.parametrize(
    "cartesian, expected",
    [
        ((0, 0, 0), (0.0, 0.0, 0.0)),  # zero vector
        ((0, 0, 1), (1.0, 0.0, 0.0)),
        ((0, 0, -2), (2.0, 180.0, 0.0)),
        ((1, 0, 0), (1.0, 90.0, 0.0)),
        ((-1, 0, 0), (1.0, 90.0, 180.0)),
        ((0, 1, 0), (1.0, 90.0, 90.0)),
        ((0, -1, 0), (1.0, 90.0, -90.0)),  # gamma in (-180, 180]
        ((1, 1, 0), (np.sqrt(2), 90.0, 45.0)),
        ((-1, -1, 0), (np.sqrt(2), 90.0, -135.0)),
        ((1, 1, 1), (np.sqrt(3), np.degrees(np.arccos(1 / np.sqrt(3))), 45.0)),
    ],
)
def test_vector_parameter_polar_getter(vector_parameter, cartesian, expected):
    vector_parameter.cartesian = cartesian
    r, phi, gamma = vector_parameter.polar
    assert r.units == u.T
    assert phi.units == u.deg
    assert gamma.units == u.deg
    np.testing.assert_allclose(
        [r.magnitude, phi.magnitude, gamma.magnitude], expected, atol=1e-12
    )
    # single components agree with the tuple
    assert_approx(vector_parameter.r, r)
    assert_approx(vector_parameter.phi, phi)
    assert_approx(vector_parameter.gamma, gamma)


@pytest.mark.parametrize(
    "polar, expected",
    [
        ((1, 0, 0), (0.0, 0.0, 1.0)),
        ((2, 90, 0), (2.0, 0.0, 0.0)),
        ((1, 90, 90), (0.0, 1.0, 0.0)),
        ((1, 180, 0), (0.0, 0.0, -1.0)),
        ((np.sqrt(2), 90, 45), (1.0, 1.0, 0.0)),
        ((0, 45, 30), (0.0, 0.0, 0.0)),
        # Quantities and unit conversion
        ((2.0 * u.T, 90 * u.deg, 0 * u.deg), (2.0, 0.0, 0.0)),
        ((2000.0 * u.mT, 90 * u.deg, 0 * u.deg), (2.0, 0.0, 0.0)),
        ((2.0 * u.T, np.pi / 2 * u.rad, np.pi / 2 * u.rad), (0.0, 2.0, 0.0)),
        # mixed: bare floats are r in T and angles in deg
        ((2.0, np.pi / 2 * u.rad, 90), (0.0, 2.0, 0.0)),
    ],
)
def test_vector_parameter_polar_setter(vector_parameter, polar, expected):
    vector_parameter.polar = polar
    np.testing.assert_allclose(vector_parameter.magnitude, expected, atol=1e-12)


def test_vector_parameter_angle_unit():
    vector = VectorParameter("T", (0.0, 1.0, 0.0), angle_unit="rad")
    r, phi, gamma = vector.polar
    assert phi.units == u.rad
    assert gamma.units == u.rad
    np.testing.assert_allclose([phi.magnitude, gamma.magnitude], [np.pi / 2, np.pi / 2])

    # bare floats are interpreted in the instance's angle_unit
    vector.polar = (1.0, np.pi / 2, 0.0)
    np.testing.assert_allclose(vector.magnitude, [1.0, 0.0, 0.0], atol=1e-12)

    # switching angle_unit only changes the returned units, not the vector
    vector.angle_unit = u.Unit("deg")
    assert vector.phi.units == u.deg
    np.testing.assert_allclose(vector.phi.magnitude, 90.0)
    np.testing.assert_allclose(vector.magnitude, [1.0, 0.0, 0.0], atol=1e-12)


def test_vector_parameter_set_r(vector_parameter):
    vector_parameter.magnitude = (0.0, 3.0, 4.0)  # r = 5
    vector_parameter.r = 10.0
    np.testing.assert_allclose(vector_parameter.magnitude, [0.0, 6.0, 8.0], atol=1e-6)
    vector_parameter.r = 2500.0 * u.mT  # direction is kept
    np.testing.assert_allclose(vector_parameter.magnitude, [0.0, 1.5, 2.0], atol=1e-6)


def test_vector_parameter_set_r_on_zero_vector(vector_parameter):
    vector_parameter.r = 2.0  # zero vector grows along +z
    np.testing.assert_allclose(vector_parameter.magnitude, [0.0, 0.0, 2.0], atol=1e-12)


@pytest.mark.parametrize(
    "attribute, value, expected",
    [
        ("phi", 90, [np.sqrt(1.5), np.sqrt(1.5), 0.0]),
        ("phi", np.pi / 2 * u.rad, [np.sqrt(1.5), np.sqrt(1.5), 0.0]),
        ("gamma", 0, [np.sqrt(2), 0.0, 1.0]),
        ("gamma", -45 * u.deg, [1.0, -1.0, 1.0]),
    ],
)
def test_vector_parameter_set_angle(vector_parameter, attribute, value, expected):
    vector_parameter.magnitude = (1.0, 1.0, 1.0)
    before = {name: getattr(vector_parameter, name) for name in ("r", "phi", "gamma")}
    setattr(vector_parameter, attribute, value)
    np.testing.assert_allclose(vector_parameter.magnitude, expected, atol=1e-12)
    # only the requested component changes
    for name in ("r", "phi", "gamma"):
        if name != attribute:
            assert_approx(getattr(vector_parameter, name), before[name])


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("r", -1.0),
        ("r", -1.0 * u.T),
        ("polar", (-1.0, 0.0, 0.0)),
        ("polar_rad", (-1.0, 0.0, 0.0)),
    ],
)
def test_vector_parameter_negative_r(vector_parameter, attribute, value):
    vector_parameter.magnitude = (1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        setattr(vector_parameter, attribute, value)
    np.testing.assert_array_equal(vector_parameter.magnitude, [1.0, 2.0, 3.0])


def test_vector_parameter_polar_rad(vector_parameter):
    vector_parameter.magnitude = (1.0, 1.0, 1.0)
    polar_rad = vector_parameter.polar_rad
    np.testing.assert_allclose(
        polar_rad, [np.sqrt(3), np.arccos(1 / np.sqrt(3)), np.pi / 4]
    )
    polar_rad[0] = 99.0  # fresh array, mutating it changes nothing
    np.testing.assert_allclose(vector_parameter.polar_rad[0], np.sqrt(3))

    vector_parameter.polar_rad = (2.0, np.pi / 2, np.pi / 2)
    np.testing.assert_allclose(vector_parameter.magnitude, [0.0, 2.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("input", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0), np.ones((2, 3))])
def test_vector_parameter_polar_rad_invalid(vector_parameter, input):
    with pytest.raises(ValueError):
        vector_parameter.polar_rad = input


def test_vector_parameter_from_polar():
    vector = VectorParameter.from_polar("T", 2.0, 90.0, 90.0, name="B")
    assert vector.name == "B"
    assert vector.unit == u.T
    np.testing.assert_allclose(vector.magnitude, [0.0, 2.0, 0.0], atol=1e-12)

    vector = VectorParameter.from_polar("T", 2.0, np.pi / 2, np.pi / 2, angle_unit="rad")
    np.testing.assert_allclose(vector.magnitude, [0.0, 2.0, 0.0], atol=1e-12)

    # r given in T is converted to the parameter's unit (mT)
    vector = VectorParameter.from_polar("mT", 1.0 * u.T, 90 * u.deg, 0 * u.deg)
    np.testing.assert_allclose(vector.magnitude, [1000.0, 0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize(
    "cartesian",
    [
        (1, 2, 3),
        (-1, 2, -3),
        (0.3, -0.2, 0.1),
        (0, 0, -5),
        (-4, 0, 0),
        (1e-3, -1e-3, 1e-3),
    ],
)
def test_vector_parameter_polar_roundtrip(cartesian):
    source = VectorParameter("T", cartesian)

    target = VectorParameter("T")
    target.polar = source.polar
    np.testing.assert_allclose(target.magnitude, source.magnitude, atol=1e-12)

    target = VectorParameter("T")
    target.polar_rad = source.polar_rad
    np.testing.assert_allclose(target.magnitude, source.magnitude, atol=1e-12)

    # the same physical vector in a different unit
    target = VectorParameter("mT")
    target.polar = source.polar
    np.testing.assert_allclose(target.magnitude, 1000.0 * source.magnitude, atol=1e-9)


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("magnitude", (1.0, 2.0, 3.0)),
        ("quantity", u.Quantity([1.0, 2.0, 3.0], "mT")),
        ("cartesian", (1.0, 2.0, 3.0)),
        ("polar", (1.0, 90.0, 0.0)),
        ("polar_rad", (1.0, 0.5, 0.5)),
        ("r", 2.0),
        ("phi", 45.0),
        ("gamma", 45.0),
    ],
)
def test_vector_parameter_notifies_caller(vector_parameter, attribute, value):
    vector_parameter.magnitude = (1.0, 1.0, 1.0)
    caller = DummyCaller()
    vector_parameter._caller = caller
    setattr(vector_parameter, attribute, value)
    assert caller.calls == 1  # exactly one notification per assignment


def test_vector_parameter_failed_set_does_not_notify(vector_parameter):
    vector_parameter.magnitude = (1.0, 2.0, 3.0)
    caller = DummyCaller()
    vector_parameter._caller = caller
    with pytest.raises(ValueError):
        vector_parameter.magnitude = (1.0, 2.0)
    with pytest.raises(ValueError):
        vector_parameter.polar = (-1.0, 0.0, 0.0)
    assert caller.calls == 0
    np.testing.assert_array_equal(vector_parameter.magnitude, [1.0, 2.0, 3.0])


def test_vector_parameter_repr(vector_parameter):
    vector_parameter.magnitude = (1.0, 2.0, 3.0)
    assert repr(vector_parameter).startswith("VectorParameter(B=[1.0, 2.0, 3.0]")
