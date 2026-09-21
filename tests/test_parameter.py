#!/usr/bin/env python

import numpy as np
import pytest
import sympy as sp
from pint.testing import assert_allclose as assert_approx
from pint.testing import assert_equal

from udkm1Dsim import u

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
