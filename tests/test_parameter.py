#!/usr/bin/env python

import pickle

import numpy as np
import pint
import pytest
import sympy as sp
from pint.testing import assert_allclose as assert_approx

from udkm1Dsim import Parameter, TemperatureParameter, VectorParameter, u
from udkm1Dsim.helpers import (
    add_latex_subscript,
    expression_to_text,
    format_number,
    latex_to_html,
    latex_to_text,
)

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
    for i, (functional, expression) in enumerate(
        zip(temperature_parameter.functional, temperature_parameter.magnitude)
    ):
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
    np.testing.assert_allclose([r.magnitude, phi.magnitude, gamma.magnitude], expected, atol=1e-12)
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
        ("r", -2.0),
        ("r", -2.0 * u.T),
        ("polar", (-2.0, 90.0, 0.0)),
        ("polar_rad", (-2.0, np.pi / 2, 0.0)),
    ],
)
def test_vector_parameter_negative_r(vector_parameter, attribute, value):
    vector_parameter.polar = (1.0, 90.0, 0.0)
    setattr(vector_parameter, attribute, value)
    # the negative radius reverses the direction and keeps the angles
    assert_approx(vector_parameter.r, -2.0 * u.T)
    assert_approx(vector_parameter.phi, 90 * u.deg)
    assert_approx(vector_parameter.gamma, 0 * u.deg)
    np.testing.assert_array_equal(vector_parameter.magnitude, [-2.0, 0.0, 0.0])


@pytest.mark.parametrize("r", [-5.0, -1.0, -0.1, 0.1, 1.0, 5.0, 1e3])
def test_vector_parameter_radius_keeps_angles(vector_parameter, r):
    vector_parameter.polar = (1.0, 30.0, 60.0)
    direction = vector_parameter.magnitude
    vector_parameter.r = r
    assert_approx(vector_parameter.phi, 30 * u.deg)
    assert_approx(vector_parameter.gamma, 60 * u.deg)
    np.testing.assert_allclose(vector_parameter.magnitude, r * direction)


@pytest.mark.parametrize(
    "polar, norm",
    [
        ((2.0, 90.0, 0.0), 2.0),
        ((-2.0, 90.0, 0.0), 2.0),
        ((0.0, 30.0, 60.0), 0.0),
    ],
)
def test_vector_parameter_norm(vector_parameter, polar, norm):
    vector_parameter.polar = polar
    # the norm is the non-negative length of the vector also for a negative r
    assert_approx(vector_parameter.norm, norm * u.T)
    assert_approx(vector_parameter.norm, np.linalg.norm(vector_parameter.quantity))


def test_vector_parameter_norm_of_cartesian(vector_parameter):
    vector_parameter.magnitude = (0.0, -3.0, 4.0)
    assert_approx(vector_parameter.norm, 5.0 * u.T)
    # the norm is read-only, the length is changed by r
    with pytest.raises(AttributeError):
        vector_parameter.norm = 1.0


def test_vector_parameter_cartesian_gives_non_negative_r(vector_parameter):
    vector_parameter.polar = (1.0, 90.0, 0.0)
    # the reversed vector is given by the angles, not by a negative radius
    vector_parameter.magnitude = (-1.0, 0.0, 0.0)
    assert_approx(vector_parameter.r, 1.0 * u.T)
    assert_approx(vector_parameter.phi, 90 * u.deg)
    assert_approx(vector_parameter.gamma, 180 * u.deg)


@pytest.mark.parametrize(
    "polar, cartesian",
    [
        ((1.0, 90.0, 0.0), [1.0, 0.0, 0.0]),
        ((1.0, 90.0, 90.0), [0.0, 1.0, 0.0]),
        ((1.0, 0.0, 45.0), [0.0, 0.0, 1.0]),
        ((-2.0, 180.0, 0.0), [0.0, 0.0, 2.0]),
        ((1e-20, 90.0, 180.0), [-1e-20, 0.0, 0.0]),
    ],
)
def test_vector_parameter_exact_cartesian(vector_parameter, polar, cartesian):
    # components, which are zero within the numerical precision, are zero
    vector_parameter.polar = polar
    np.testing.assert_array_equal(vector_parameter.magnitude, cartesian)
    assert repr(vector_parameter) == f"VectorParameter(B={cartesian} T)"


def test_vector_parameter_polar_rad(vector_parameter):
    vector_parameter.magnitude = (1.0, 1.0, 1.0)
    polar_rad = vector_parameter.polar_rad
    np.testing.assert_allclose(polar_rad, [np.sqrt(3), np.arccos(1 / np.sqrt(3)), np.pi / 4])
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
        vector_parameter.polar_rad = (1.0, 0.0)
    assert caller.calls == 0
    np.testing.assert_array_equal(vector_parameter.magnitude, [1.0, 2.0, 3.0])


def test_vector_parameter_repr(vector_parameter):
    vector_parameter.magnitude = (1.0, 2.0, 3.0)
    assert repr(vector_parameter).startswith("VectorParameter(B=[1.0, 2.0, 3.0]")


# Symbols and display


@pytest.mark.parametrize(
    "symbol, text, html",
    [
        ("d", "d", "<i>d</i>"),
        (r"\kappa", "κ", "<i>κ</i>"),
        ("c_e", "cₑ", "<i>c</i><sub><i>e</i></sub>"),
        ("c_{SRO}", "c_{SRO}", "<i>c</i><sub>SRO</sub>"),
        (r"T_\mathrm{C}", "T_C", "<i>T</i><sub>C</sub>"),
        (r"A_\mathrm{ex}", "Aₑₓ", "<i>A</i><sub>ex</sub>"),
        (r"\langle u^2 \rangle", "⟨u²⟩", "⟨<i>u</i><sup>2</sup>⟩"),
        (r"\mathbf{m}", "m", "<b>m</b>"),
        (r"\mathrm{d}n/\mathrm{d}\epsilon", "dn/dε", "d<i>n</i>/d<i>ε</i>"),
    ],
)
def test_symbol_formats(symbol, text, html):
    assert latex_to_text(symbol) == text
    assert latex_to_html(symbol) == html


@pytest.mark.parametrize(
    "symbol, subscript, expected",
    [("c", "e", "c_{e}"), ("M_s", "e", "M_{s,e}"), ("M_{s}", "0", "M_{s,0}")],
)
def test_add_latex_subscript(symbol, subscript, expected):
    assert add_latex_subscript(symbol, subscript) == expected


@pytest.mark.parametrize(
    "value, text, latex",
    [
        (5000.0, "5000", "5000"),
        (2e-8, "2×10⁻⁸", r"2 \times 10^{-8}"),
        (-1.5, "−1.5", "-1.5"),
        (2.44 + 4.32j, "2.44 + 4.32i", r"2.44 + 4.32\,i"),
        (2, "2", "2"),
        (np.array([1.0, 2e-8]), "[1, 2×10⁻⁸]", r"[1, 2 \times 10^{-8}]"),
    ],
)
def test_format_number(value, text, latex):
    assert format_number(value) == text
    assert format_number(value, "latex") == latex


def test_parameter_display():
    p = Parameter("m", 2e-8, name="thickness", symbol="d")
    assert str(p) == "d = 2×10⁻⁸ m"
    assert p._repr_latex_() == r"$d = 2 \times 10^{-8}\ \mathrm{m}$"
    assert p._table_rows("html") == [("<i>d</i>", "2×10<sup>−8</sup>", "m")]


def test_parameter_display_without_symbol_shows_name():
    p = Parameter("", 0.5, name="my_param")
    assert str(p) == "my_param = 0.5"
    assert p._repr_latex_() == r"$\mathrm{my\_param} = 0.5$"


def test_parameter_repr_does_not_depend_on_symbol():
    # the repr is used for the hash of simulation results
    p = Parameter("m", 2e-8, name="thickness", symbol="d")
    q = Parameter("m", 2e-8, name="thickness", symbol="t")
    assert repr(p) == repr(q) == "Parameter(thickness=2e-08 m)"


def test_temperature_parameter_display_single_sub_system():
    p = TemperatureParameter("J/(kg K)", "0.112*T + 455.2", name="heat_capacity", symbol="c")
    assert str(p) == "c(T) = 0.112·T + 455.2 J/K/kg"
    assert p._repr_latex_().startswith("$c(T) = 0.112 T + 455.2")


def test_temperature_parameter_display_constant():
    p = TemperatureParameter("W/(m K)", 94, name="therm_cond", symbol=r"\kappa")
    assert str(p) == "κ = 94 W/K/m"


def test_temperature_parameter_display_sub_systems():
    # without a group, the sub-systems are labeled by their index
    p = TemperatureParameter(
        "W/m**3", ["-4.2e17*(T_0-T_1)", "4.2e17*(T_0-T_1)"], name="coupling", symbol="G"
    )
    assert str(p).split("\n") == [
        "G₀(T₀, T₁) = −4.2×10¹⁷·T₀ + 4.2×10¹⁷·T₁ W/m³",
        "G₁(T₀, T₁) = 4.2×10¹⁷·T₀ − 4.2×10¹⁷·T₁ W/m³",
    ]
    latex = p._repr_latex_()
    assert latex.startswith(r"$\begin{aligned}G_{0}(T_{0}, T_{1}) &= ")
    assert latex.count(r" \\ ") == 1
    rows = p._table_rows("text")
    assert [row[0] for row in rows] == ["G₀", "G₁"]
    assert [row[2] for row in rows] == ["W/m³", ""]  # unit only in the first row


@pytest.mark.parametrize(
    "expression, text",
    [
        ("0.2*T - 100.0", "0.2·T − 100"),
        ("-1.5*T", "−1.5·T"),
        ("470 + 0.19*T - 7.08e6/T**2", "0.19·T + 470 − 7.08×10⁶/T²"),
        ("-4.2e17*(T_0-T_1)", "−4.2×10¹⁷·T₀ + 4.2×10¹⁷·T₁"),
        ("(T_0+T_1)**2", "(T₀ + T₁)²"),
        ("T**2.5", "T^2.5"),
        ("2.42e4*T**-1.114", "2.42×10⁴/T^1.114"),
        ("1 + abs(T - 500)**0.7", "|T − 500|^0.7 + 1"),
    ],
)
def test_expression_to_text(expression, text):
    assert expression_to_text(sp.sympify(expression)) == text


def test_vector_parameter_keeps_angles_of_zero_vector():
    v = VectorParameter("")
    v.polar = (0, 90 * u.deg, 270 * u.deg)
    assert_approx(v.phi, 90 * u.deg)
    assert_approx(v.gamma, -90 * u.deg)  # gamma is given in -180 .. 180 deg
    np.testing.assert_array_equal(v.magnitude, [0, 0, 0])
    v.r = 1  # the direction is restored
    assert np.allclose(v.magnitude, [0, -1, 0])
    # also cartesian zero vectors keep the angles
    v.magnitude = [0, 0, 0]
    assert_approx(v.phi, 90 * u.deg)
    assert_approx(v.gamma, -90 * u.deg)


def test_vector_parameter_keeps_angles_set_before_amplitude():
    v = VectorParameter("")
    v.phi = 90 * u.deg
    v.gamma = 45 * u.deg
    v.r = 2
    assert np.allclose(v.magnitude, [np.sqrt(2), np.sqrt(2), 0])


def test_vector_parameter_keeps_gamma_along_z_axis():
    v = VectorParameter("")
    v.polar = (1, 0 * u.deg, 120 * u.deg)
    assert_approx(v.gamma, 120 * u.deg)
    v.phi = 90 * u.deg
    assert np.allclose(v.magnitude, [-0.5, np.sqrt(3) / 2, 0])


def test_vector_parameter_keeps_cartesian_components():
    rng = np.random.default_rng(0)
    v = VectorParameter("")
    for scale in (1e-12, 1.0, 1e12):
        for _ in range(100):
            cartesian = rng.normal(size=3) * scale
            cartesian[rng.random(3) < 0.2] = 0
            v.magnitude = cartesian
            np.testing.assert_array_equal(v.magnitude, cartesian)
    # a change of the polar form calculates them from it
    v.magnitude = (1.0, 2.0, 3.0)
    v.r = 2 * v.r
    np.testing.assert_allclose(v.magnitude, [2.0, 4.0, 6.0])


def test_vector_parameter_pickle():
    v = VectorParameter("", name="m")
    v.polar = (-1, 90 * u.deg, 45 * u.deg)
    v = pickle.loads(pickle.dumps(v))
    assert np.allclose(v.polar_rad, [-1, np.pi / 2, np.pi / 4])


@pytest.mark.parametrize(
    "state, polar_rad",
    [
        # cartesian components only
        ({"_magnitude": np.array([1.0, 0.0, 0.0])}, [1, np.pi / 2, 0]),
        # with the last defined angles of a zero vector
        (
            {"_magnitude": np.zeros(3), "_angles": np.array([np.pi / 2, np.pi])},
            [0, np.pi / 2, np.pi],
        ),
        # with the last defined gamma of a vector along the z-axis
        ({"_magnitude": np.array([0.0, 0.0, 2.0]), "_angles": np.array([0, 1.0])}, [2, 0, 1.0]),
    ],
)
def test_vector_parameter_pickled_by_former_versions(state, polar_rad):
    v = VectorParameter("", name="m")
    former = {k: val for k, val in v.__dict__.items() if k != "_polar"}
    v = VectorParameter.__new__(VectorParameter)
    v.__setstate__({**former, **state})
    assert "_magnitude" not in v.__dict__
    assert "_angles" not in v.__dict__
    assert np.allclose(v.polar_rad, polar_rad)
    # the cartesian components are kept, e.g. for the hash of the structure
    np.testing.assert_array_equal(v.magnitude, state["_magnitude"])


# VectorParameter.set


@pytest.mark.parametrize(
    "value, polar_rad",
    [
        ((0.0, 2.0, 0.0), [2, np.pi / 2, np.pi / 2]),
        ({"r": -1.0, "phi": 90.0, "gamma": 0.0}, [-1, np.pi / 2, 0]),
        ({"r": -1.0}, [-1, np.pi / 4, np.pi / 2]),  # the angles are kept
        ({"phi": 0.5 * u.rad}, [1, 0.5, np.pi / 2]),
        ({"x": 1.0}, [np.sqrt(2), np.pi / 3, np.arctan(np.sqrt(0.5))]),  # y and z are kept
        ({"x": 0.0, "y": 0.0, "z": 3.0}, [3, 0, np.pi / 2]),  # gamma is kept
    ],
)
def test_vector_parameter_set(vector_parameter, value, polar_rad):
    vector_parameter.polar = (1.0, 45.0, 90.0)
    caller = DummyCaller()
    vector_parameter._caller = caller
    vector_parameter.set(value)
    np.testing.assert_allclose(vector_parameter.polar_rad, polar_rad, atol=1e-15)
    assert caller.calls == 1


def test_vector_parameter_set_from_vector_parameter():
    source = VectorParameter("mT", (1.0, 2.0, 3.0))
    target = VectorParameter("T", angle_unit="rad")
    target.set(source)
    # the cartesian components are kept and converted to the unit of the target
    np.testing.assert_array_equal(target.magnitude, [1e-3, 2e-3, 3e-3])
    assert target.angle_unit == u.rad
    # also a signed radius and undefined angles are copied
    source.polar = (-1.0, 0.0, 30.0)
    target.set(source)
    np.testing.assert_allclose(target.polar_rad, [-1e-3, 0, np.pi / 6])
    # the copy is independent of the source
    source.r = 2
    np.testing.assert_allclose(target.polar_rad, [-1e-3, 0, np.pi / 6])


def test_vector_parameter_init_by_dict():
    vector = VectorParameter("T", {"r": -1, "phi": 90, "gamma": 180}, name="B")
    np.testing.assert_array_equal(vector.magnitude, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(vector.polar_rad, [-1, np.pi / 2, np.pi])


@pytest.mark.parametrize("value", [{}, {"r": 1, "x": 0}, {"amplitude": 1}])
def test_vector_parameter_set_invalid_dict(vector_parameter, value):
    vector_parameter.magnitude = (1.0, 2.0, 3.0)
    caller = DummyCaller()
    vector_parameter._caller = caller
    with pytest.raises(ValueError):
        vector_parameter.set(value)
    np.testing.assert_array_equal(vector_parameter.magnitude, [1.0, 2.0, 3.0])
    assert caller.calls == 0
