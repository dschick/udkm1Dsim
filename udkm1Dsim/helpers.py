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
    "read_columns",
    "to_energy",
    "SpectralTable",
    "make_hash_md5",
    "make_hashable",
    "m_power_x",
    "m_times_n",
    "finderb",
    "multi_gauss",
    "convert_cartesian_to_polar",
    "convert_polar_to_cartesian",
    "latex_to_text",
    "latex_to_html",
    "add_latex_subscript",
    "format_number",
    "expression_to_text",
    "expression_to_html",
    "expression_to_latex",
]

__docformat__ = "restructuredtext"

import hashlib
import os
import re

import numpy as np
import pint
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty_symbology import greek_unicode
from sympy.printing.str import StrPrinter

u = pint.get_application_registry()


def read_columns(filename, delimiter=None, columns=3):
    """read_columns

    Read the first columns of a text file, e.g. of tabulated atomic form
    factors or refractive indices. Lines starting with ``#`` and lines
    without numbers, e.g. headers, are skipped.

    Args:
        filename (str): path of the file.
        delimiter (str, optional): column delimiter, by default white space or
            a comma.
        columns (int, optional): number of columns. Defaults to 3.

    Returns:
        data (ndarray[float]): columns of shape (N, columns).

    """
    if delimiter is None:
        with open(filename) as file:
            text = file.read()
        delimiter = "," if "," in text else None
    data = np.atleast_2d(np.genfromtxt(filename, comments="#", delimiter=delimiter))
    if data.shape[1] < columns:
        raise ValueError(f"{filename} must have at least {columns} columns!")
    # lines without a number in the first column, e.g. headers
    return data[~np.isnan(data[:, 0]), :columns]


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
        return tuple(make_hashable(e) for e in obj)

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

    while (b - a) > 1:  # loop until the intervall is larger than 1
        c = int(np.floor((a + b) / 2))  # center of intervall
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
    s = np.asarray(s) / (2 * np.sqrt(2 * np.log(2)))
    a = np.asarray(A) / np.sqrt(2 * np.pi * s**2)  # normalize area to 1
    x0 = np.asarray(x0)

    y = np.zeros_like(x)
    for i in range(len(s)):
        y = y + a[i] * np.exp(-((x - x0[i]) ** 2) / (2 * s[i] ** 2))
    return y


def convert_polar_to_cartesian(polar):
    r"""convert_polar_to_cartesian

    Convert a vector or field from polar coordinates
    :math:`(r, \phi, \gamma)` to cartesian coordinates :math:`(x, y, z)`:

    .. math::

        F_x & = r \sin(\phi)\cos(\gamma) \\
        F_y & = r \sin(\phi)\sin(\gamma) \\
        F_z & = r \cos(\phi)

    where :math:`r` is the radius (amplitude), :math:`\phi` the polar angle
    from the :math:`z`-axis, and :math:`\gamma` the azimuthal angle in the
    :math:`xy`-plane from the :math:`x`-axis of vector field
    :math:`\mathbf{F}`. A negative radius reverses the direction given by
    the angles.

    In case of multidimensional array as input, the relevant vector or
    field must be contained in its last dimension.

    Args:
        polar (ndarray[float]): vector of field to convert.

    Returns:
        cartesian (ndarray[float]): converted vector or field.

    """
    cartesian = np.zeros_like(polar)

    amplitudes = polar[..., 0]
    phis = polar[..., 1]
    gammas = polar[..., 2]
    cartesian[..., 0] = amplitudes * np.sin(phis) * np.cos(gammas)
    cartesian[..., 1] = amplitudes * np.sin(phis) * np.sin(gammas)
    cartesian[..., 2] = amplitudes * np.cos(phis)

    return cartesian


def convert_cartesian_to_polar(cartesian):
    r"""convert_cartesian_to_polar

    Convert a vector or field from cartesian coordinates :math:`(x, y, z)`
    to polar coordinates :math:`(r, \phi, \gamma)`:

    .. math::

        F_r & = \sqrt{F_x^2 + F_y^2+F_z^2}\\
        F_{\phi} & = \arccos\left(\frac{F_z}{F_r} \right) \\
        F_{\gamma} & = \begin{cases}\\
        \arctan\left(\frac{F_y}{F_x} \right) & \mathrm{for}\ F_x > 0 \\
        \pi + \arctan\left(\frac{F_y}{F_x}\right)
        & \mathrm{for}\ F_x < 0 \ \mathrm{and}\ F_y \geq 0 \\
        \arctan\left(\frac{F_y}{F_x}\right) - \pi
        & \mathrm{for}\ F_x < 0 \ \mathrm{and}\ F_y < 0 \\
        \pm\frac{\pi}{2} & \mathrm{for}\ F_x = 0 \ \mathrm{and}\ F_y \gtrless 0 \\
        0 & \mathrm{for}\ F_x = F_y = 0
        \end{cases}

    where :math:`F_r` is the radial (amplitude) component, :math:`F_{\phi}`
    the polar angle from the :math:`z`-axis, and :math:`F_{\gamma}` the
    azimuthal angle in the :math:`xy`-plane from the :math:`x`-axis of vector
    field :math:`\mathbf{F}`. Both angles are zero for a zero vector. The
    radius is never negative.

    In case of multidimensional array as input, the relevant vector or
    field must be contained in its last dimension.

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
    mask = amplitudes != 0.0  # mask for non-zero amplitudes
    polar[..., 0] = amplitudes
    polar[mask, 1] = np.arccos(np.divide(zs[mask], amplitudes[mask]))
    # the angles of zero vectors are zero, also for signed zero components
    polar[mask, 2] = np.arctan2(ys[mask], xs[mask])

    return polar


# ============================================================================
# formatting of symbols, numbers, and expressions as text, HTML, and LaTeX
# ============================================================================

_SUB = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))
_SUP = dict(zip("0123456789+-=()in", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿ"))
_COMMANDS = {
    "varepsilon": "ε",
    "langle": "⟨",
    "rangle": "⟩",
    "partial": "∂",
    "cdot": "·",
    "times": "×",
}
_TOKENS = re.compile(r"\\[A-Za-z]+|\\.|[{}_^]|\s+|.")


def _parse(tokens):
    """Parse LaTeX tokens into a list of nodes until the closing brace."""
    nodes = []
    while tokens and tokens[0] != "}":
        node = _parse_one(tokens)
        if node is not None:
            nodes.append(node)
    if tokens:
        tokens.pop(0)  # closing brace
    return nodes


def _parse_one(tokens):
    """Parse a single node, i.e. a group, command, script, or character."""
    token = tokens.pop(0)
    if token == "{":
        return ("group", _parse(tokens))
    if token in "_^":
        return (token, _argument(tokens))
    if token in ("\\mathrm", "\\mathbf"):
        return (token[1:], _argument(tokens))
    if token == "\\frac":
        return ("frac", _argument(tokens), _argument(tokens))
    if token.isspace():
        return None
    return ("char", token)


def _argument(tokens):
    """Parse the argument of a command or script, i.e. a single node."""
    while tokens and tokens[0].isspace():
        tokens.pop(0)
    if not tokens or tokens[0] == "}":
        return []
    node = _parse_one(tokens)
    return node[1] if node[0] == "group" else [node]


def _render(nodes, html, italic=True):
    """Render parsed nodes as HTML or plain text with unicode characters."""
    out = ""
    for node in nodes:
        kind = node[0]
        if kind == "char":
            char = node[1]
            if char.startswith("\\"):
                name = char[1:]
                char = greek_unicode.get(name, _COMMANDS.get(name, name))
            if html and italic and char.isalpha():
                char = f"<i>{char}</i>"
            out += char
        elif kind == "group":
            out += _render(node[1], html, italic)
        elif kind == "mathrm":
            out += _render(node[1], html, italic=False)
        elif kind == "mathbf":
            inner = _render(node[1], html, italic=False)
            out += f"<b>{inner}</b>" if html else inner
        elif kind == "frac":
            out += _render(node[1], html, italic) + "/" + _render(node[2], html, italic)
        else:  # sub- or superscript, which is upright if descriptive
            descriptive = len(_render(node[1], html=False)) > 1
            inner = _render(node[1], html, italic and not descriptive)
            if html:
                tag = "sub" if kind == "_" else "sup"
                out += f"<{tag}>{inner}</{tag}>"
            else:
                table = _SUB if kind == "_" else _SUP
                if all(c in table for c in inner):
                    out += "".join(table[c] for c in inner)
                else:
                    out += kind + (inner if len(inner) == 1 else "{" + inner + "}")
    return out


def latex_to_text(latex):
    r"""latex_to_text

    Convert a symbol given in a small subset of LaTeX to plain text with
    unicode characters, e.g. ``r"\kappa_e"`` to ``κₑ``. Supported are greek
    letters, sub- and superscripts, ``\mathrm``, ``\mathbf``, ``\langle``,
    ``\rangle``, ``\partial``, and ``\frac``. Sub- and superscripts without
    unicode characters are kept as ``_{...}`` and ``^{...}``.

    Args:
        latex (str): symbol in LaTeX.

    Returns:
        text (str): symbol as plain text.

    """
    return _render(_parse(_TOKENS.findall(latex)), html=False)


def latex_to_html(latex):
    r"""latex_to_html

    Convert a symbol given in a small subset of LaTeX to HTML, e.g.
    ``r"\kappa_e"`` to ``<i>κ</i><sub><i>e</i></sub>``, see
    :func:`latex_to_text` for the supported LaTeX. Letters are italic,
    except for descriptive sub- and superscripts of several characters
    and within ``\mathrm``.

    Args:
        latex (str): symbol in LaTeX.

    Returns:
        html (str): symbol as HTML.

    """
    return _render(_parse(_TOKENS.findall(latex)), html=True)


def add_latex_subscript(latex, subscript):
    """add_latex_subscript

    Add a subscript to a symbol given in LaTeX, which is merged with an
    existing trailing subscript, e.g. ``M_s`` and ``e`` result in ``M_{s,e}``.

    Args:
        latex (str): symbol in LaTeX.
        subscript (str): subscript to add.

    Returns:
        latex (str): symbol with subscript in LaTeX.

    """
    match = re.search(r"_(\{(?P<group>[^{}]*)\}|(?P<single>\\?[A-Za-z0-9]))$", latex)
    if match:
        existing = match.group("group") or match.group("single")
        return latex[: match.start()] + "_{" + existing + "," + subscript + "}"
    return latex + "_{" + subscript + "}"


def _format_real(value, mode):
    """Format a real number with 4 significant digits and a power of ten."""
    text = f"{value:.4g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    exponent = str(int(exponent))
    if mode == "latex":
        return rf"{mantissa} \times 10^{{{exponent}}}"
    if mode == "html":
        return f"{mantissa}×10<sup>{exponent}</sup>"
    return f"{mantissa}×10" + "".join(_SUP[c] for c in exponent)


def _format_number(value, mode):
    if isinstance(value, (np.ndarray, list)):
        # nested lists of matrices, e.g. the exchange matrix of sublattices
        values = value.tolist() if isinstance(value, np.ndarray) else value
        return "[" + ", ".join(_format_number(v, mode) for v in values) + "]"
    if isinstance(value, complex):
        real = _format_real(value.real, mode)
        sign = "-" if value.imag < 0 else "+"
        imag = _format_real(abs(value.imag), mode)
        return f"{real} {sign} {imag}i" if mode != "latex" else rf"{real} {sign} {imag}\,i"
    if isinstance(value, (bool, int, np.integer)):
        return str(value)
    return _format_real(float(value), mode)


def format_number(value, mode="text"):
    """format_number

    Format a number, complex number, or array with 4 significant digits and
    powers of ten as plain text with unicode characters, HTML, or LaTeX,
    e.g. ``2×10⁻⁸`` or ``2.44 + 4.32i``.

    Args:
        value (float, int, complex, ndarray): number to format.
        mode (str, optional): ``text``, ``html``, or ``latex``. Defaults to
            ``text``.

    Returns:
        formatted (str): formatted number.

    """
    text = _format_number(value, mode)
    return text if mode == "latex" else text.replace("-", "−")


class _TextPrinter(StrPrinter):
    """Prints sympy expressions as plain text with unicode characters or as
    HTML, e.g. with superscripts for powers and symbols with subscripts."""

    def __init__(self, html=False):
        super().__init__({"full_prec": False})
        self.html = html

    def _print_Float(self, expr):
        # with an ASCII minus, so that sympy recognizes negative numbers in sums,
        # which is replaced by a proper minus at the end
        return _format_number(float(expr), "html" if self.html else "text")

    def _print_Symbol(self, expr):
        return latex_to_html(expr.name) if self.html else latex_to_text(expr.name)

    def _print_Abs(self, expr):
        return f"|{self._print(expr.args[0])}|"

    def _print_Pow(self, expr, rational=False):
        base, exp = expr.as_base_exp()
        if exp.is_Number and exp.is_positive and not (exp.is_Rational and not exp.is_Integer):
            base_str = self._print(base)
            if not (base.is_Atom or base.func.__name__ == "Abs"):
                base_str = f"({base_str})"
            exp_str = self._print(exp)
            if self.html:
                return f"{base_str}<sup>{exp_str}</sup>"
            if all(c in _SUP for c in exp_str):
                return base_str + "".join(_SUP[c] for c in exp_str)
            return f"{base_str}^{exp_str}"
        return super()._print_Pow(expr, rational)

    def doprint(self, expr):
        text = super().doprint(expr)
        return text.replace("*", "·").replace("-", "−")


class _LatexPrinter(LatexPrinter):
    """Prints sympy expressions as LaTeX with 4 significant digits."""

    def _print_Float(self, expr):
        return format_number(float(expr), "latex")

    def _print_Symbol(self, expr, style="plain"):
        return expr.name  # the names of the symbols are given in LaTeX


def expression_to_text(expr):
    """expression_to_text

    Convert a sympy expression to plain text with unicode characters in a
    single line, e.g. ``0.19·Tₚ + 470 − 7.08×10⁶/Tₚ²``. Floats are given
    with 4 significant digits and the names of symbols in LaTeX.

    Args:
        expr (sympy.Expr): sympy expression.

    Returns:
        text (str): expression as plain text.

    """
    return _TextPrinter().doprint(expr)


def expression_to_html(expr):
    """expression_to_html

    Convert a sympy expression to HTML in a single line, see
    :func:`expression_to_text`.

    Args:
        expr (sympy.Expr): sympy expression.

    Returns:
        html (str): expression as HTML.

    """
    return _TextPrinter(html=True).doprint(expr)


def expression_to_latex(expr):
    """expression_to_latex

    Convert a sympy expression to LaTeX, where floats are given with 4
    significant digits and the names of symbols in LaTeX.

    Args:
        expr (sympy.Expr): sympy expression.

    Returns:
        latex (str): expression as LaTeX.

    """
    return _LatexPrinter().doprint(expr)


# ============================================================================
# spectral tables, e.g. of dispersions and atomic form factors
# ============================================================================


def to_energy(value):
    """to_energy

    Convert photon energies, wavelengths, frequencies, or wavenumbers given
    as pint Quantity to photon energies in eV. Numbers are taken in eV.

    Args:
        value (float, ndarray[float], Quantity): spectral value(s).

    Returns:
        energy (ndarray[float]): photon energies (eV).

    """
    if isinstance(value, u.Quantity):
        return np.asarray(value.to("eV", "sp").magnitude, dtype=float)
    return np.asarray(value, dtype=float)


def _md5(*arrays):
    """Short md5 digest of arrays for the representation of tabulated data."""
    hasher = hashlib.md5()
    for array in arrays:
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()[:8]


def interpolate_table(x, values, energy):
    """interpolate_table

    Linear interpolation of complex values, which are tabulated for the
    photon energies ``x`` (eV) in ascending order, e.g. of a
    :class:`Dispersion` or of the form factors of an :class:`.Atom`. The
    values are NaN outside of the table and between NaN entries of the table.

    Args:
        x (ndarray[float]): tabulated photon energies (eV) in ascending order.
        values (ndarray[complex]): tabulated values of shape (K,) or (K, ...).
        energy (ndarray[float]): photon energies (eV) of shape (M,).

    Returns:
        values (ndarray[complex]): interpolated values of shape (M,) or (M, ...).

    """
    values = np.asarray(values, dtype=complex)
    flat = values.reshape(len(x), -1)
    result = np.empty((len(energy), flat.shape[1]), dtype=complex)
    for i in range(flat.shape[1]):
        result[:, i] = np.interp(energy, x, flat[:, i].real, left=np.nan, right=np.nan) + 1j * (
            np.interp(energy, x, flat[:, i].imag, left=np.nan, right=np.nan)
        )
    return result.reshape((len(energy),) + values.shape[1:])


def defined_range(x, values):
    """Lowest and highest photon energy (eV) with defined, i.e. not NaN,
    tabulated values."""
    values = np.asarray(values, dtype=complex)
    valid = ~(np.isnan(values.real) | np.isnan(values.imag))
    if valid.ndim > 1:
        valid = valid.reshape(len(x), -1).all(axis=1)
    if not np.any(valid):
        return np.inf, -np.inf
    return float(np.min(x[valid])), float(np.max(x[valid]))


def undefined_error(description, energy, invalid, energy_ranges, spectral=None, hint=""):
    """ValueError for the photon energies (eV) for which a tabulated quantity
    is not defined. The requested and the defined ranges are given in the
    quantity and unit of the ``spectral`` input of the user."""
    invalid_energies = np.asarray(energy)[invalid]
    name, texts = _spectral_texts([np.min(invalid_energies), np.max(invalid_energies)], spectral)
    requested = (
        f"the {name} {texts[0]}"
        if texts[0] == texts[1]
        else f"the {name}s from {texts[0]} to {texts[1]}"
    )
    defined = _range_text(energy_ranges, spectral, separator="\nand ")
    message = f"{description}\nis not defined for {requested},\nbut only {defined}."
    return ValueError(message + (f"\n{hint}" if hint else ""))


def _format_energy(energy):
    """Short text of a photon energy (eV)."""
    if not np.isfinite(energy):
        return "∞"
    if energy >= 1000:
        return f"{energy / 1000:.4g} keV"
    return f"{energy:.4g} eV"


_SPECTRAL_NAMES = {
    "[length] ** 2 * [mass] / [time] ** 2": "photon energy",
    "[length]": "wavelength",
    "1 / [time]": "frequency",
    "1 / [length]": "wavenumber",
}


def _spectral_texts(energies, spectral=None):
    """Name of the spectral quantity and texts of photon energies (eV) in the
    quantity and unit of the ``spectral`` input of the user, e.g. as
    wavelengths in nm, sorted in ascending order. Without a pint Quantity as
    input, the photon energies are given in eV."""
    energies = np.asarray(energies, dtype=float)
    if not isinstance(spectral, u.Quantity) or not np.all(np.isfinite(energies)):
        return "photon energy", [_format_energy(e) for e in np.sort(energies)]
    unit = spectral.units
    name = _SPECTRAL_NAMES.get(str(spectral.dimensionality), "photon energy")
    with np.errstate(divide="ignore"):
        values = np.sort((energies * u.eV).to(unit, "sp").magnitude)
    return name, [f"{value:.4g} {unit:~P}" for value in values]


def _range_text(energy_ranges, spectral=None, separator=" and "):
    """Text of energy ranges (eV) in the quantity and unit of the ``spectral``
    input of the user, e.g. ``from 400 nm to 1000 nm and from ...``."""
    texts = []
    for energy_range in energy_ranges:
        _, (lo, hi) = _spectral_texts(energy_range, spectral)
        texts.append(f"from {lo} to {hi}")
    return separator.join(texts)


class SpectralTable:
    """SpectralTable

    Complex values, which are tabulated for photon energies in ascending
    order and interpolated linearly, e.g. of a dispersion or of atomic form
    factors. The values are NaN outside of the table and between invalid,
    i.e. NaN, entries of the table.

    Args:
        x (ndarray[float], Quantity): photon energies, where floats are taken in
            (eV), or wavelengths, frequencies, or wavenumbers as pint Quantity.
        values (ndarray[complex]): tabulated values of shape (N,) or (N, ...).
        source (str, optional): source of the table, e.g. its file name.

    """

    def __init__(self, x, values, source=""):
        energy = np.atleast_1d(to_energy(x))
        values = np.asarray(values, dtype=complex)
        if values.shape[:1] != energy.shape:
            raise ValueError(
                f"The first axis of the values {values.shape} must match the "
                f"{len(energy)} photon energies!"
            )
        order = np.argsort(energy)
        self.energy = energy[order]
        self.values = values[order]
        self.source = source

    @classmethod
    def from_columns(cls, data, x_unit="eV", imaginary_sign=1, invalid_below=None, source=""):
        """from_columns

        Table from the columns of the spectral axis and the real and
        imaginary part, e.g. read by :func:`read_columns`.

        Args:
            data (ndarray[float]): columns of shape (N, 3).
            x_unit (str, optional): unit of the spectral axis. Defaults to eV.
            imaginary_sign (int, optional): sign of the imaginary part, e.g. -1
                for atomic form factors :math:`f_1 - i f_2`. Defaults to 1.
            invalid_below (float, optional): real parts below are invalid,
                e.g. the -9999 of the tables of Henke et al.
            source (str, optional): source of the table.

        Returns:
            table (SpectralTable): table.

        """
        data = np.atleast_2d(np.asarray(data, dtype=float))
        data = data[~np.isnan(data[:, 0])]
        values = data[:, 1] + imaginary_sign * 1j * data[:, 2]
        invalid = np.isnan(data[:, 1]) | np.isnan(data[:, 2])
        if invalid_below is not None:
            invalid |= data[:, 1] <= invalid_below
        values[invalid] = np.nan
        return cls(data[:, 0] * u(x_unit), values, source=source)

    @classmethod
    def from_file(
        cls, filename, x_unit="eV", imaginary_sign=1, invalid_below=None, delimiter=None
    ):
        """from_file

        Table from a text file with the columns of the spectral axis and the
        real and imaginary part, see :func:`read_columns` and
        :meth:`from_columns`.

        Args:
            filename (str): path of the file.
            x_unit (str, optional): unit of the spectral axis. Defaults to eV.
            imaginary_sign (int, optional): sign of the imaginary part.
                Defaults to 1.
            invalid_below (float, optional): real parts below are invalid.
            delimiter (str, optional): column delimiter.

        Returns:
            table (SpectralTable): table.

        """
        return cls.from_columns(
            read_columns(filename, delimiter),
            x_unit=x_unit,
            imaginary_sign=imaginary_sign,
            invalid_below=invalid_below,
            source=os.path.basename(str(filename)),
        )

    def __len__(self):
        return len(self.energy)

    @property
    def energy_range(self):
        """tuple[float]: lowest and highest photon energy (eV) with defined
        values."""
        return defined_range(self.energy, self.values)

    @property
    def md5(self):
        """str: short md5 digest of the table, e.g. for the hash of simulations."""
        return _md5(self.energy, self.values)

    def __call__(self, energy):
        """Interpolated values of shape (M,) or (M, ...) for photon energies,
        where floats are taken in eV, or other spectral Quantities, which are
        NaN outside of the table."""
        return interpolate_table(self.energy, self.values, np.atleast_1d(to_energy(energy)))

    def __repr__(self):
        lo, hi = self.energy_range
        source = f", source={self.source}" if self.source else ""
        return (
            f"SpectralTable({len(self)} values from {lo:.4g} eV to {hi:.4g} eV{source}, "
            f"md5={self.md5})"
        )
