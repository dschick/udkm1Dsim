#!/usr/bin/env python

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

__all__ = ["Parameter", "TemperatureParameter", "VectorParameter"]

__docformat__ = "restructuredtext"

import warnings
from inspect import isfunction

import numpy as np
import pint
from scipy.integrate import quad
from sympy import Symbol, integrate, lambdify, symarray, symbols, sympify

from ..helpers import (
    add_latex_subscript,
    convert_cartesian_to_polar,
    convert_polar_to_cartesian,
    expression_to_html,
    expression_to_latex,
    expression_to_text,
    format_number,
    latex_to_html,
    latex_to_text,
)

u = pint.get_application_registry()


class Parameter:
    r"""Parameter with a unit and a magnitude.

    The `symbol` of the parameter is given in LaTeX, e.g. ``r"\kappa"`` or
    ``"T_\mathrm{C}"``, and is used for its display by `print()` and
    `display()` together with its value and unit. Without a symbol, the
    name of the parameter is displayed.

    """

    def __init__(self, unit, magnitude=0.0, name="", symbol=""):
        self.unit = u.Unit(unit)
        self.name = name
        self.symbol = symbol
        self._caller = None
        self.magnitude = magnitude

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, value):
        if isinstance(value, u.Quantity):
            self._magnitude = value.to(self.unit).magnitude
        elif isinstance(value, (int, float, complex, np.ndarray)):
            self._magnitude = value
        else:
            raise TypeError(f"Cannot set Parameter '{self.name}' from type {type(value)}")

        if self._caller is not None:
            self._caller._update_depending()

    @property
    def quantity(self):
        return self._magnitude * self.unit

    @quantity.setter
    def quantity(self, value):
        self.magnitude = value

    def __repr__(self):
        magnitude = self.magnitude
        if isinstance(magnitude, np.ndarray):
            magnitude = magnitude.tolist()
        unit = f"{self.unit:~P}"
        unit = f" {unit}" if unit else ""
        return f"{type(self).__name__}({self.name}={magnitude}{unit})"

    # The repr is also used for the hash of the simulation results, which must
    # not depend on the symbol. Hence, the symbol is only used by the following
    # methods for the display of the parameter.

    def _latex_symbol(self):
        """LaTeX symbol of the parameter, which is its name if not set."""
        return self.symbol or r"\mathrm{" + self.name.replace("_", r"\_") + "}"

    @staticmethod
    def _format_symbol(symbol, mode):
        """Format a LaTeX symbol in the mode `text`, `html`, or `latex`."""
        if mode == "latex":
            return symbol
        return latex_to_html(symbol) if mode == "html" else latex_to_text(symbol)

    def _format_unit(self, mode):
        """Format the unit in the mode `text`, `html`, or `latex`."""
        return format(self.unit, {"text": "~P", "html": "~H", "latex": "~L"}[mode])

    def _entries(self, mode):
        """List of the symbol, the arguments, and the value of the parameter
        in the mode `text`, `html`, or `latex`, with one entry per sub-system."""
        return [
            (
                self._format_symbol(self._latex_symbol(), mode),
                "",
                format_number(self.magnitude, mode),
            )
        ]

    def _table_rows(self, mode):
        """Rows of the symbol, value, and unit of the parameter for the table
        view of its group in the mode `text` or `html`."""
        unit = self._format_unit(mode)
        return [
            (symbol, value, unit if i == 0 else "")
            for i, (symbol, _, value) in enumerate(self._entries(mode))
        ]

    def __str__(self):
        unit = self._format_unit("text")
        return "\n".join(
            f"{symbol}{arguments} = {value}" + (f" {unit}" if unit else "")
            for symbol, arguments, value in self._entries("text")
        )

    def _repr_latex_(self):
        unit = self._format_unit("latex")
        unit = rf"\ {unit}" if unit else ""
        lines = [
            f"{symbol}{arguments} = {value}{unit}"
            for symbol, arguments, value in self._entries("latex")
        ]
        if len(lines) == 1:
            return f"${lines[0]}$"
        lines = [line.replace(" = ", " &= ", 1) for line in lines]
        return r"$\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}$"


class VectorParameter(Parameter):
    """3D vector Parameter with a unit, settable in cartesian or polar form.

    `magnitude` holds the cartesian components (x, y, z) in `self.unit`.
    Polar form is (r, phi, gamma) as in convert_polar_to_cartesian:
    phi is the angle from +z, gamma the angle in the xy-plane from +x.
    """

    def __init__(self, unit, magnitude=(0.0, 0.0, 0.0), name="", angle_unit="deg", symbol=""):
        self.angle_unit = u.Unit(angle_unit)  # unit for bare-float angles
        super().__init__(unit, magnitude, name, symbol)

    @classmethod
    def from_polar(cls, unit, r, phi, gamma, name="", angle_unit="deg", symbol=""):
        vec = cls(unit, name=name, angle_unit=angle_unit, symbol=symbol)
        vec.polar = (r, phi, gamma)
        return vec

    @staticmethod
    def _to_magnitude(value, unit, default=None):
        """Float in `unit`; bare numbers are taken to be in `default` (or `unit`)."""
        if not isinstance(value, u.Quantity):
            value = value * (default if default is not None else unit)
        return value.to(unit).magnitude

    # ---- cartesian (magnitude / quantity) --------------------------------
    @property
    def magnitude(self):
        return self._magnitude.copy()  # copy: no silent in-place edits

    @magnitude.setter
    def magnitude(self, value):
        if isinstance(value, u.Quantity):
            value = value.to(self.unit).magnitude
        arr = np.asarray(value)
        if arr.dtype.kind not in "iuf":
            raise TypeError(f"Cannot set Parameter '{self.name}' from {type(value)}")
        if arr.shape != (3,):
            raise ValueError(f"Parameter '{self.name}' needs exactly 3 components")
        # float cast matters: the helpers use zeros_like and would truncate ints
        Parameter.magnitude.fset(self, arr.astype(float))

    @property
    def cartesian(self):
        """(x, y, z) as a pint Quantity."""
        return self.quantity

    @cartesian.setter
    def cartesian(self, value):
        self.magnitude = value

    # ---- polar (r, phi, gamma) -------------------------------------------
    def _polar_rad(self):
        """[r, phi, gamma] as floats, angles in rad."""
        return convert_cartesian_to_polar(self._magnitude)

    def _set_polar_rad(self, polar):
        if polar[0] < 0:
            raise ValueError("r must be >= 0")
        self.magnitude = convert_polar_to_cartesian(np.asarray(polar, dtype=float))

    def _set_component(self, index, value):
        polar = self._polar_rad()
        if index == 0:
            polar[0] = self._to_magnitude(value, self.unit)
        else:
            polar[index] = self._to_magnitude(value, u.rad, self.angle_unit)
        self._set_polar_rad(polar)

    @property
    def r(self):
        return self._polar_rad()[0] * self.unit

    @r.setter
    def r(self, value):
        self._set_component(0, value)

    @property
    def phi(self):
        return (self._polar_rad()[1] * u.rad).to(self.angle_unit)

    @phi.setter
    def phi(self, value):
        self._set_component(1, value)

    @property
    def gamma(self):
        return (self._polar_rad()[2] * u.rad).to(self.angle_unit)

    @gamma.setter
    def gamma(self, value):
        self._set_component(2, value)

    @property
    def polar(self):
        """(r, phi, gamma) as pint Quantities."""
        return self.r, self.phi, self.gamma

    @polar.setter
    def polar(self, value):
        r, phi, gamma = value
        self._set_polar_rad(
            [
                self._to_magnitude(r, self.unit),
                self._to_magnitude(phi, u.rad, self.angle_unit),
                self._to_magnitude(gamma, u.rad, self.angle_unit),
            ]
        )

    @property
    def polar_rad(self):
        """Polar coordinates [r, phi, gamma] as floats, r in the unit of the
        parameter and the angles in rad."""
        return self._polar_rad()  # fresh array each call, safe to mutate

    @polar_rad.setter
    def polar_rad(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"Parameter '{self.name}' needs (r, phi, gamma)")
        self._set_polar_rad(arr)


class TemperatureParameter(Parameter):
    """Parameter with a unit and a magnitude, which depends on temperature.

    For several sub-systems, their labels are displayed as subscripts of the
    symbol and the temperatures, e.g. :math:`c_e(T_e)`. The labels are given
    by the `sub_system_labels` of the :class:`.ThermalParameters` of the
    parameter and default to the index of the sub-system.

    """

    def __init__(self, unit, magnitude=0.0, name="", symbol=""):
        self._functional = []
        self._num_sub_systems = 0
        self._takes_temperature_vector = []
        self._integral = []
        self._integral_expr = []
        super().__init__(unit, magnitude=magnitude, name=name, symbol=symbol)

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, value):
        # here the sympy expression is stored
        self._magnitude = self.parse_input(value)
        # an expression takes the temperature vector of all sub-systems for
        # more than 1 variable or "_" in its name
        self._takes_temperature_vector = [
            len(expression.free_symbols) > 1 or any("_" in s.name for s in expression.free_symbols)
            for expression in self._magnitude
        ]
        # reset of dependent parameters on change
        self._functional = []
        self._integral = []
        self._integral_expr = []
        if self._caller is not None:
            self._caller._update_depending()

    @property
    def num_sub_systems(self):
        """int: number of sub-systems, i.e. the number of given expressions."""
        return self._num_sub_systems

    def _sub_system_labels(self):
        """Labels of the sub-systems, which are given by the group of the
        parameter, or None if the sub-systems are not labeled."""
        group_labels = getattr(self._caller, "_display_sub_system_labels", None)
        if group_labels is not None:
            return group_labels()
        if self.num_sub_systems > 1:
            return [str(i) for i in range(self.num_sub_systems)]
        return None

    def _entries(self, mode):
        labels = self._sub_system_labels()

        def label(i):
            return labels[i] if i < len(labels) else str(i)

        entries = []
        for i, expression in enumerate(self.magnitude):
            symbol = self._latex_symbol()
            # the temperatures as arguments by the index of their sub-system
            indices = {
                s: i if s.name == "T" else int(s.name.split("_")[1])
                for s in expression.free_symbols
            }
            if labels is not None:
                symbol = add_latex_subscript(symbol, label(i))
                # the temperatures are labeled as the sub-systems
                temperatures = {s: Symbol(f"T_{{{label(j)}}}") for s, j in indices.items()}
                expression = expression.subs(temperatures)
            else:
                temperatures = {s: s for s in indices}
            arguments = [temperatures[s] for s in sorted(indices, key=indices.get)]
            arguments = (
                "(" + ", ".join(self._format_symbol(s.name, mode) for s in arguments) + ")"
                if arguments
                else ""
            )
            printer = {
                "text": expression_to_text,
                "html": expression_to_html,
                "latex": expression_to_latex,
            }
            entries.append(
                (self._format_symbol(symbol, mode), arguments, printer[mode](expression))
            )
        return entries

    @property
    def takes_temperature_vector(self):
        """list[bool]: true if the functional of a sub-system must be called
        with the temperatures of all sub-systems instead of a scalar."""
        return self._takes_temperature_vector

    @property
    def functional(self):
        """list[@lambda]: one callable per expression in ``magnitude``.

        Scalar expressions are called with the temperature of their own
        sub-system, expressions that take the temperature vector with the
        temperatures of all sub-systems, which are unpacked along the first
        axis.

        """
        if self._functional == []:
            # vector arguments are unpacked by a nested argument list
            T_scalar = symbols("T")
            T_vector = [list(symarray("T", self._num_sub_systems))]
            self._functional = [
                lambdify(T_vector if takes_vector else T_scalar, expression, modules="numpy")
                for expression, takes_vector in zip(
                    self._magnitude, self._takes_temperature_vector
                )
            ]

        return self._functional

    @property
    def integral(self):
        if self._integral == []:
            # for now the dedicated logic as in functional is skipped
            self._integral_expr = []
            T = symbols("T")
            for functional, expression in zip(self.functional, self.magnitude):
                syms = sorted(expression.free_symbols, key=lambda s: s.name)
                if len(syms) == 0:
                    syms = ["T"]
                if len(syms) == 1:
                    T = symbols("T")
                else:
                    T = symarray("T", len(syms))

                try:
                    integral = integrate(expression, T)
                    self._integral.append(lambdify(T, integral, modules="numpy"))
                    self._integral_expr.append(integral)
                except Exception:
                    warnings.warn(
                        "\nSympy's analytical integration of the heat capacity "
                        "did not work.\n"
                        "Just do it numerically with scipy.integrate.quad"
                    )
                    self._integral.append(lambda T: quad(functional, 0, T, limit=10000)[0])
                    self._integral_expr.append(f"scipy.integrate.quad({str(expression)}, 0, T)[0]")

        return self._integral

    @property
    def integral_expr(self):
        if self._integral_expr == []:
            self.integral
        return self._integral_expr

    @property
    def quantity(self):
        return self.magnitude

    @quantity.setter
    def quantity(self, value):
        self.magnitude = value

    def parse_input(self, inputs):
        """parse_input

        Parses the input and create a list of function handle strings with T as
        argument. Inputs can be strings, floats, ints, or pint quantities.

        Args:
            inputs (list[str, int, float, Quantity]): list of strings, int, floats,
                or Pint quantities.

        Returns:
            (tuple):
            - *expressions (list[@expres])* - list of sympy expressions from sympify.

        """
        expressions = []
        # if the input is not a list, we convert it to one
        if not isinstance(inputs, list):
            inputs = [inputs]
        self._num_sub_systems = len(inputs)

        # traverse each list element and convert it to a function handle
        for input in inputs:
            # first create a string
            if isfunction(input):
                raise ValueError("Please use string representation of function!")
            elif isinstance(input, str):
                # backwards compatibility for direct lambda definition
                if ":" in input:
                    # strip lambda prefix
                    input = input.split(":")[1]
                # backwards compatibility for []-indexing
                input = input.replace("[", "_").replace("]", "")
            elif isinstance(input, (int, float)):
                input = str(input)
            elif isinstance(input, u.Quantity):
                input = str(input.to_base_units().magnitude)

            if "_" in input:
                # the temperature is input as a vector
                T = symarray("T", self._num_sub_systems)  # noqa: F841
            else:
                # the temperature is input as a scalar
                T = symbols("T")  # noqa: F841
            try:
                expressions.append(sympify(input))
            except Exception as e:
                print(
                    "String input for layer property "
                    + input
                    + " \
                    cannot be converted to function handle!"
                )
                print(e)

        return expressions
