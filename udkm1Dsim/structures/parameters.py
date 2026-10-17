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

__all__ = ["Parameter", "TemperatureParameter"]

__docformat__ = "restructuredtext"

import warnings
from inspect import isfunction

import numpy as np
import pint
from scipy.integrate import quad
from sympy import integrate, lambdify, symarray, symbols, sympify

from ..helpers import convert_cartesian_to_polar, convert_polar_to_cartesian

u = pint.get_application_registry()


class Parameter:
    """Parameter with a unit and a magnitude."""

    def __init__(self, unit, magnitude=0.0, name=""):
        self.unit = u.Unit(unit)
        self.name = name
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


class VectorParameter(Parameter):
    """3D vector Parameter with a unit, settable in cartesian or polar form.

    `magnitude` holds the cartesian components (x, y, z) in `self.unit`.
    Polar form is (r, phi, gamma) as in convert_polar_to_cartesian:
    phi is the angle from +z, gamma the angle in the xy-plane from +x.
    """

    def __init__(self, unit, magnitude=(0.0, 0.0, 0.0), name="", angle_unit="deg"):
        self.angle_unit = u.Unit(angle_unit)  # unit for bare-float angles
        super().__init__(unit, magnitude, name)

    @classmethod
    def from_polar(cls, unit, r, phi, gamma, name="", angle_unit="deg"):
        vec = cls(unit, name=name, angle_unit=angle_unit)
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
        """[r, phi, gamma] as floats: r in self.unit, angles in rad."""
        return self._polar_rad()  # fresh array each call, safe to mutate

    @polar_rad.setter
    def polar_rad(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"Parameter '{self.name}' needs (r, phi, gamma)")
        self._set_polar_rad(arr)


class TemperatureParameter(Parameter):
    """Parameter with a unit and a magnitude, which depends on temperature."""

    def __init__(self, unit, magnitude=0.0, name=""):
        self._functional = []
        self._num_sub_systems = 0
        self._takes_temperature_vector = []
        self._integral = []
        self._integral_expr = []
        super().__init__(unit, magnitude=magnitude, name=name)

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
