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

__all__ = [
    "Parameter",
    "TemperatureParameter"
]

__docformat__ = "restructuredtext"

import warnings
from inspect import isfunction

import numpy as np
import pint
from scipy.integrate import quad
from sympy import integrate, lambdify, symarray, symbols, sympify
from sympy.printing.numpy import NumPyPrinter

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
        return f"Parameter({self.name}={self.magnitude} {self.unit})"


class TemperatureParameter(Parameter):
    """Parameter with a unit and a magnitude, which depends on temperature."""

    def __init__(self, unit, magnitude=0.0, name=""):
        super().__init__(unit, magnitude=magnitude, name=name)
        self._functional = []
        self._integral = []
        self._integral_expr = []

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, value):
        self._magnitude = self.parse_input(value)
        # reset of dependent parameters on change
        self._functional = []
        self._integral = []
        self._integral_expr = []
        if self._caller is not None:
            self._caller._update_depending()

    @property
    def functional(self):
        if self._functional == []:
            for expression in self._magnitude:
                syms = sorted(expression.free_symbols, key=lambda s: s.name)
                if len(syms) == 0:
                    syms = ['T']

                if len(syms) == 1:
                    is_vector = False
                else:
                    is_vector = True

                if is_vector:
                    body = NumPyPrinter().doprint(expression)
                    unpack = "".join(f"    T_{i} = T[{i}]\n" for i in range(len(syms)))
                    src = f"def _f(T):\n{unpack}    return {body}\n"
                    ns = {'numpy': np}
                    exec(src, ns)
                    f = ns["_f"]
                    self._functional.append(f)
                else:
                    self._functional.append(lambdify(syms, expression, modules="numpy"))

        return self._functional

    @property
    def integral(self):
        if self._integral == []:
            self._integral_expr = []
            T = symbols("T")
            for functional, expression in zip(self.functional, self.magnitude):
                syms = sorted(expression.free_symbols, key=lambda s: s.name)
                if len(syms) == 0:
                    syms = ['T']
                if len(syms) == 1:
                    T = symbols("T")
                else:
                    T = symarray("T", len(syms))

                try:
                    integral = integrate(expression, T)
                    self._integral.append(lambdify(T, integral, modules='numpy'))
                    self._integral_expr.append(integral)
                except Exception:
                    warnings.warn('\nSympy\'s analytical integration of the heat capacity '
                                    'did not work.\n'
                                    'Just do it numerically with scipy.integrate.quad')
                    self._integral.append(lambda T: quad(functional, 0, T, limit=10000)[0])
                    self._integral_expr.append(f'scipy.integrate.quad({str(expression)}, 0, T)[0]')

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

    def __repr__(self):
        return f"Parameter({self.name}={self.magnitude} {self.unit})"

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
        k = len(inputs)

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

            if '_' in input:
                # the temperature is input as a vector
                T = symarray("T", k)  # noqa: F841
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
