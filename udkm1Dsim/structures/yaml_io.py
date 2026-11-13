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

"""Saving and loading of structures as YAML.

A :class:`.Structure` is written by :meth:`.Structure.to_yaml` and read by
:meth:`.Structure.from_yaml`. The YAML file holds the sections

* ``format_version``: version of the format of the file,
* ``udkm1Dsim_version``: version of udkm1Dsim, which wrote the file,
* ``atoms``: all :class:`.Atom` and :class:`.AtomMixed` by their key,
* ``layers``: all :class:`.Layer` and its sub-classes by their key,
* ``structures``: all nested :class:`.Structure` by their key,
* ``structure``: the structure itself.

Atoms, layers, and nested structures are shared objects, which are defined
once and referenced by their key, which is their id, or the name of a nested
structure. Layers and nested structures share their keys. Only the parameters
are written, which differ from the defaults of their parameter group.
Quantities are written with their unit, e.g. ``5.72 W/K/m``, where a
readable unit is only used, if the value is read back exactly, so that the
hash of the structure is kept. Temperature-dependent parameters are written as
their formula strings in the unit of their parameter group.

"""

__all__ = [
    "FORMAT_VERSION",
    "structure_to_dict",
    "structure_from_dict",
    "structure_to_yaml",
    "structure_from_yaml",
]

__docformat__ = "restructuredtext"

import functools
import os
import warnings
from pathlib import PurePath

import numpy as np
import pint
import yaml
from sympy.printing.str import StrPrinter

from ..helpers import SpectralTable
from .atoms import Atom, AtomMixed
from .dielectric import (
    Dielectric,
    Dispersion,
    _Atoms,
    _Constant,
    _Fallback,
    _FormFactors,
    _MassAttenuation,
    _OptPenDepth,
    _Scaled,
    _Sum,
    _Table,
)
from .layers import AmorphousLayer, Layer, UnitCell, Vacuum
from .parameter_groups import LatticeParameters, MagneticParameters
from .parameters import Parameter, TemperatureParameter, VectorParameter
from .structure import Structure

u = pint.get_application_registry()
Q_ = u.Quantity

FORMAT_VERSION = 1

# the simulation types of Structure.get_hash(), which are checked on writing
_HASH_TYPES = ("heat", "phonon", "xray", "optical", "magnetic")

_LAYER_TYPES = {
    Layer: "Layer",
    AmorphousLayer: "AmorphousLayer",
    UnitCell: "UnitCell",
    Vacuum: "Vacuum",
}

# readable units, which are tried for quantities besides the unit of their parameter
_PREFERRED_UNITS = [
    u.Unit(unit)
    for unit in (
        "nm",
        "angstrom",
        "um",
        "pm",
        "angstrom**2",
        "nm**2",
        "g/cm**3",
        "nm/ps",
        "ps",
        "fs",
        "ns",
        "eV",
        "meV",
        "bohr_magneton",
    )
]

_DIELECTRIC_KEYS = ("eps0", "eps_mag", "strain_model", "deps_dstrain", "euler_angles")
_DIELECTRIC_INPUTS = _DIELECTRIC_KEYS + ("display_energy",)


# ============================================================================
# numbers, quantities, and expressions
# ============================================================================


class _ExactPrinter(StrPrinter):
    """Printer of sympy expressions, whose floats are read back exactly."""

    def _print_Float(self, expr):
        if expr._prec == 53:
            return repr(float(expr))
        return super()._print_Float(expr)


def _same(a, b):
    """Exact comparison by the text of the values, which distinguishes
    integers and floats, and -0.0 and 0.0, as the hash of the structure."""
    return repr(a) == repr(b)


def _number(value):
    """Python number of a number, also of a numpy number."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Cannot write the boolean {value} as a number!")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (complex, np.complexfloating)):
        return _complex(value)
    raise TypeError(f"Cannot write {value!r} of type {type(value).__name__} as a number!")


def _number_text(value):
    value = _number(value)
    return repr(value) if isinstance(value, float) else str(value)


def _parse_number(text):
    """int, float, or complex of a text, which raises a ValueError else."""
    text = text.strip()
    for parse in (int, float):
        try:
            return parse(text)
        except ValueError:
            pass
    return complex(text.replace(" ", ""))


def _complex(value):
    """A complex number as float without imaginary part or as text, e.g.
    ``2.44+4.32j``, as YAML has no complex numbers."""
    value = complex(value)
    if value.imag == 0 and not np.signbit(value.imag):
        return float(value.real)
    return repr(value).strip("()")


def _complex_array(values):
    """Nested lists of complex numbers, see :func:`_complex`."""
    values = np.asarray(values)
    if values.ndim == 0:
        return _complex(values.item())
    return [_complex_array(v) for v in values]


def _read_complex(value):
    if isinstance(value, str):
        return complex(value.replace(" ", ""))
    if isinstance(value, list):
        return [_read_complex(v) for v in value]
    return complex(value)


def _read_complex_array(value):
    return np.asarray(_read_complex(value), dtype=complex)


def _unit_text(unit):
    return f"{unit:~P}"


def _parse_quantity(text):
    """Quantity of a text, e.g. ``3.905 Å``, where the number is read exactly
    and only its unit is parsed by pint."""
    parts = text.strip().split(None, 1)
    if len(parts) == 2:
        try:
            return Q_(_parse_number(parts[0]), parts[1])
        except (ValueError, pint.errors.PintError):
            pass
    return Q_(text)


def _quantity_text(magnitude, unit):
    """Shortest text of a scalar quantity, which gives the same magnitude in
    the unit of its parameter on reading, as the unit of the parameter or a
    more readable unit, e.g. ``3.905 Å`` instead of ``3.905e-10 m``."""
    # the text of the number is part of the key of the cache, e.g. to tell
    # -0.0 from 0.0
    return _cached_quantity_text(_number_text(magnitude), magnitude, unit)


@functools.lru_cache(maxsize=4096)
def _cached_quantity_text(original, magnitude, unit):
    """:func:`_quantity_text` of the text ``original`` of the magnitude,
    which is cached, since the same quantities, e.g. the default
    parameters, are written many times."""
    quantity = Q_(magnitude, unit)
    candidates = [unit] + [
        p for p in _PREFERRED_UNITS if p != unit and p.dimensionality == unit.dimensionality
    ]
    best = f"{original} {_unit_text(unit)}"
    for candidate in candidates:
        try:
            converted = quantity.to(candidate).magnitude
        except (ValueError, TypeError, pint.errors.PintError):
            continue
        # without the rounding errors of the conversion, e.g. 6.312 nm/ps
        # instead of 6312.000000000001 m/s
        numbers = [converted]
        if isinstance(converted, float):
            numbers += [float(f"{converted:.{digits:d}g}") for digits in (15, 16)]
        for number in numbers:
            text = f"{_number_text(number)} {_unit_text(candidate)}"
            try:
                back = _parse_quantity(text).to(unit).magnitude
            except (ValueError, TypeError, pint.errors.PintError):
                continue
            if _number_text(back) == original and len(text) < len(best):
                best = text
    return best


def _write_quantity(magnitude, unit):
    """Value of a parameter, which is a number, a list, a text with the unit,
    e.g. ``3.905 Å``, or a dict of the value and unit for arrays."""
    if isinstance(magnitude, np.ndarray) and magnitude.ndim == 0:
        magnitude = magnitude.item()
    dimensionless = unit == u.dimensionless
    if isinstance(magnitude, np.ndarray):
        values = _plain(magnitude.tolist())
        return values if dimensionless else {"value": values, "unit": _unit_text(unit)}
    if dimensionless:
        return _number(magnitude)
    return _quantity_text(magnitude, unit)


def _read_quantity(value):
    """Input of a parameter from its value, see :func:`_write_quantity`."""
    if isinstance(value, dict):
        if set(value) != {"value", "unit"}:
            raise ValueError(f"An array must be given by its value and unit, not by {value}!")
        return Q_(np.asarray(value["value"]), value["unit"])
    if isinstance(value, str):
        try:
            return _parse_number(value)
        except ValueError:
            return _parse_quantity(value)
    return value


def _plain(value):
    """Nested lists of Python numbers from numpy types."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return _number(value)


def _write_expression(expression):
    if expression.is_Integer:
        return int(expression)
    if expression.is_Float and expression._prec == 53:
        return float(expression)
    return _ExactPrinter().doprint(expression)


def _write_temperature(parameter):
    """Formula strings of a temperature-dependent parameter in the unit of
    its parameter group, where constants are written with this unit, e.g.
    ``5.72 W/K/m``, if they are read back exactly."""
    values = []
    for expression in parameter.magnitude:
        value = _write_expression(expression)
        if isinstance(value, (int, float)) and parameter.unit != u.dimensionless:
            text = f"{_number_text(value)} {_unit_text(parameter.unit)}"
            if _same(_read_temperature(text, parameter.unit), value):
                value = text
        values.append(value)
    return values[0] if len(values) == 1 else values


def _read_temperature(value, unit):
    """Input of a temperature-dependent parameter, whose constants may be
    given with a unit, which is converted to the unit of the parameter."""
    if isinstance(value, list):
        return [_read_temperature(v, unit) for v in value]
    if isinstance(value, str):
        parts = value.strip().split(None, 1)
        if len(parts) == 2:
            try:
                quantity = Q_(_parse_number(parts[0]), parts[1])
                if quantity.is_compatible_with(unit):
                    return quantity.to(unit).magnitude
            except (ValueError, TypeError, pint.errors.PintError):
                pass
    return value


# ============================================================================
# vectors
# ============================================================================


def _read_vector(value, unit=""):
    """Fresh VectorParameter from cartesian components or a dict of polar
    or cartesian components, whose angles are given with their unit or as
    floats in deg."""
    vector = VectorParameter(unit)
    if isinstance(value, dict):
        vector.set({key: _read_quantity(v) for key, v in value.items()})
    else:
        vector.set(_read_quantity(value))
    return vector


def _write_vector(vector):
    """Cartesian components of a vector, or its polar components with the
    angles in deg or rad, whichever gives the same vector on reading."""
    if vector.unit != u.dimensionless:
        raise TypeError(f"Only dimensionless vectors can be written, not '{vector.name}'!")
    target = (vector.polar_rad.tolist(), vector.magnitude.tolist())

    def same(candidate):
        read = _read_vector(candidate)
        return _same((read.polar_rad.tolist(), read.magnitude.tolist()), target)

    cartesian = _plain(vector.magnitude.tolist())
    if same(cartesian):
        return cartesian
    r, phi, gamma = (float(v) for v in vector.polar_rad)
    for angle_unit in ("deg", "rad"):
        candidate = {
            "r": r,
            "phi": f"{float((phi * u.rad).to(angle_unit).magnitude)!r} {angle_unit}",
            "gamma": f"{float((gamma * u.rad).to(angle_unit).magnitude)!r} {angle_unit}",
        }
        if same(candidate):
            return candidate
    raise ValueError(f"The vector '{vector.name}' = {vector!r} cannot be written exactly!")


def _read_magnetization(value):
    """A VectorParameter or a list of VectorParameters, one per sublattice."""
    if isinstance(value, list) and value and all(isinstance(v, (list, dict)) for v in value):
        return [_read_vector(v) for v in value]
    return _read_vector(value)


def _apply_magnetization(group, value):
    """Set the magnetization exactly, keeping the VectorParameter of a single
    magnetization."""
    if isinstance(value, VectorParameter) and isinstance(group.magnetization, VectorParameter):
        group.magnetization.set(value)
    else:
        group.magnetization = value


# ============================================================================
# writing
# ============================================================================


class _Writer:
    """Writes the atoms, layers, and structures of a structure to dicts,
    where every object is written once and referenced by its key."""

    def __init__(self, base_dir=None, embed=False):
        self.base_dir = base_dir
        self.embed = embed
        self.atoms = {}
        self.layers = {}
        self.structures = {}
        self._keys = {}
        self._used = {"atoms": set(), "layers": set()}
        # the id and data of the atoms by their key, written with their id as key
        self._atom_data = {}
        # the objects are kept, so that their id() is not reused
        self._objects = []
        self._default_groups = {}
        self._default_vacuum = None

    def _reserve(self, obj, base, namespace):
        """Key of an object and true if it is new, where the keys of
        different objects with the same id get a suffix."""
        key = self._keys.get(id(obj))
        if key is not None:
            return key, False
        key, i = str(base), 2
        while key in self._used[namespace]:
            key = f"{base}_{i:d}"
            i += 1
        self._used[namespace].add(key)
        self._keys[id(obj)] = key
        self._objects.append(obj)
        return key, True

    def _path(self, path):
        """Path relative to the YAML file with forward slashes on all
        operating systems, or as given without file."""
        path = str(path)
        if self.base_dir is None:
            return path
        try:
            return PurePath(os.path.relpath(os.path.abspath(path), self.base_dir)).as_posix()
        except ValueError:
            # e.g. another drive on Windows
            return os.path.abspath(path)

    # atoms

    def atom(self, atom):
        """Key of an atom, where copies of an atom, e.g. in the dispersion of a
        layer, which copies its parameter groups, share the key of the atom."""
        if not isinstance(atom, Atom):
            raise TypeError(f"Cannot write {type(atom).__name__} as atom!")
        key = self._keys.get(id(atom))
        if key is not None:
            return key
        data = self._write_atom(atom, atom.id)
        for key, (atom_id, existing) in self._atom_data.items():
            if atom_id == atom.id and _same(existing, data):
                self._keys[id(atom)] = key
                self._objects.append(atom)
                return key
        key, _ = self._reserve(atom, atom.id, "atoms")
        self._atom_data[key] = (atom.id, data)
        self.atoms[key] = data if key == atom.id else self._write_atom(atom, key)
        return key

    def _write_atom(self, atom, key):
        mixed = isinstance(atom, AtomMixed)
        data = {"type": "AtomMixed"} if mixed else {}
        data["symbol"] = atom.symbol
        if atom.id != key:
            data["id"] = atom.id
        default_name = atom.symbol if mixed else _element_name(atom.symbol)
        if atom.name != default_name:
            data["name"] = atom.name
        if not mixed and not _same(atom.ionicity, 0):
            data["ionicity"] = _number(atom.ionicity)
        magnetization = _write_vector(atom.atomic.magnetization)
        if magnetization != _write_vector(VectorParameter("")):
            data["magnetization"] = magnetization
        for kind, magnetic in (("atomic", False), ("magnetic", True)):
            form_factors = self._write_form_factors(atom, magnetic)
            if form_factors is not None:
                data[f"{kind}_form_factors"] = form_factors
        if not mixed:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tabulated = atom.read_cromer_mann_coeff()
            if not _same(np.asarray(atom.cromer_mann_coeff).tolist(), tabulated.tolist()):
                data["cromer_mann_coeff"] = _plain(atom.cromer_mann_coeff.tolist())
        if mixed:
            data["atoms"] = [[self.atom(a), _number(fraction)] for a, fraction in atom.atoms]
        return data

    def _write_form_factors(self, atom, magnetic):
        """Source of the atomic or magnetic form factors, or None for the
        default ones."""
        kind = "magnetic" if magnetic else "atomic"
        source = getattr(atom, f"_{kind}_form_factor_source")
        path = getattr(atom, f"_{kind}_form_factor_path", "")
        coeff = getattr(atom, f"{kind}_form_factor_coeff")
        mixed = isinstance(atom, AtomMixed)
        if source != "custom" and path and not self.embed:
            return {"file": self._path(path)}
        if source != "custom" and not path:
            # the tables of the package
            default = "" if mixed else (f"{atom.symbol}.mf" if magnetic else "Chantler")
            if source == default:
                return None
            if not magnetic and not mixed and source == "Henke":
                return "henke"
        # custom coefficients, embedded files, or embedded tables read back
        table = None if coeff is None else _plain(np.asarray(coeff).tolist())
        return {"table": table, "source": source}

    # layers

    def layer(self, layer):
        if type(layer) not in _LAYER_TYPES:
            raise TypeError(
                f"Cannot write the layer '{getattr(layer, 'id', layer)}' of type "
                f"{type(layer).__name__}, only of the types {', '.join(_LAYER_TYPES.values())}!"
            )
        key, new = self._reserve(layer, layer.id, "layers")
        if new:
            self.layers[key] = self._write_layer(layer, key)
        return key

    def _write_layer(self, layer, key):
        data = {"type": _LAYER_TYPES[type(layer)]}
        if layer.id != key:
            data["id"] = layer.id
        data["name"] = layer.name
        for group_name in layer._parameter_group_names:
            group = getattr(layer, group_name)
            skip = ()
            if isinstance(layer, UnitCell) and group_name == "structural":
                # given by the lattice and the atoms
                skip = ("thickness", "area") + (("density",) if layer.num_atoms > 0 else ())
            values = self._write_group(group, skip)
            if values:
                data[group_name] = values
        if isinstance(layer, AmorphousLayer) and getattr(layer, "_atom", None) is not None:
            data["atom"] = self.atom(layer._atom)
        if isinstance(layer, UnitCell) and layer.num_atoms > 0:
            data["atoms"] = [
                [self.atom(atom), _write_position(position)] for atom, _, position in layer.atoms
            ]
        higher_orders = np.atleast_1d(layer.elastic.spring_const.magnitude)[1:]
        if higher_orders.size:
            data["ho_spring_constants"] = _plain(higher_orders.tolist())
        return data

    def _default_group(self, group_class):
        if group_class not in self._default_groups:
            self._default_groups[group_class] = group_class()
        return self._default_groups[group_class]

    def _write_group(self, group, skip=()):
        """Inputs of a parameter group, which differ from its defaults."""
        if isinstance(group, LatticeParameters):
            # the a- and b-axis default to the c-axis
            c_axis = self.value(group.c_axis)
            data = {}
            for name in ("a_axis", "b_axis"):
                value = self.value(getattr(group, name))
                if value != c_axis:
                    data[name] = value
            data["c_axis"] = c_axis
            return data
        default = self._default_group(type(group))
        data = {}
        for name in group._input_names():
            if name in skip or name in group._derived:
                continue
            if name == "sub_system_labels":
                value, default_value = group._sub_system_labels, default._sub_system_labels
            else:
                value = self.value(getattr(group, name), name)
                default_value = self.value(getattr(default, name), name)
            # e.g. explicit concentrations, which are derived by default
            explicit = name in default._derived
            if explicit or not _same(value, default_value):
                data[name] = value
        return data

    def value(self, parameter, name=""):
        """Value of a parameter."""
        if isinstance(parameter, list):
            # the magnetizations of several sublattices
            return [_write_vector(vector) for vector in parameter]
        if isinstance(parameter, Dielectric):
            return self._write_dielectric(parameter)
        if isinstance(parameter, TemperatureParameter):
            return _write_temperature(parameter)
        if isinstance(parameter, VectorParameter):
            return _write_vector(parameter)
        if isinstance(parameter, Parameter):
            return _write_quantity(parameter.magnitude, parameter.unit)
        raise TypeError(f"Cannot write the parameter '{name}' of type {type(parameter).__name__}!")

    # dielectric tensor

    def _write_dielectric(self, dielectric):
        data = {}
        if dielectric.eps0 is not None:
            data["eps0"] = self.dispersion(dielectric.eps0)
        if dielectric.eps_mag is not None:
            data["eps_mag"] = self.dispersion(dielectric.eps_mag)
        if dielectric.strain_model != "density":
            data["strain_model"] = dielectric.strain_model
        deps = dielectric.deps_dstrain
        if isinstance(deps, Dispersion):
            data["deps_dstrain"] = self.dispersion(deps)
        elif not _same(np.asarray(deps).tolist(), 0j):
            data["deps_dstrain"] = _complex_array(deps)
        if dielectric._euler_angles is not None:
            data["euler_angles"] = [_write_angle(angle) for angle in dielectric._euler_angles]
        energy = dielectric.display_energy
        if energy is not None:
            data["display_energy"] = (
                f"{_number_text(energy.magnitude)} {_unit_text(energy.units)}"
                if isinstance(energy, u.Quantity)
                else _number(energy)
            )
        if set(data) == {"eps0", "strain_model"} and data["strain_model"] == "none":
            # the same as setting the dielectric tensor by a dispersion
            return data["eps0"]
        return data

    def dispersion(self, dispersion):
        """Dict of a dispersion by its source, or by its tabulated values."""
        recipe = dispersion._recipe
        if recipe is not None and not self.embed:
            return self._write_recipe(recipe)
        kind = type(dispersion)
        if kind is _Constant:
            value = _complex_array(dispersion._input_values)
            if dispersion._input_kind == "n":
                return {"ref_index": value}
            return {"constant": value, "kind": dispersion._input_kind}
        if kind is _Table:
            table = {
                "kind": dispersion._kind,
                "input_kind": dispersion._input_kind,
                "energy": _plain(dispersion._energy.tolist()),
                "values": _complex_array(dispersion._values),
            }
            if dispersion._origin:
                table["origin"] = dispersion._origin
            return {"table": table}
        if kind is _FormFactors:
            table = dispersion._table
            return {
                "form_factors": {
                    "magnetic": dispersion._magnetic,
                    "molar_mass": _number(dispersion._molar_mass),
                    "energy": _plain(table.energy.tolist()),
                    "values": _complex_array(table.values),
                    "source": table.source,
                }
            }
        if kind is _Atoms:
            return {
                "atoms": [[self.atom(atom), _number(f)] for atom, f in dispersion._atoms],
                "magnetic": dispersion._magnetic,
            }
        if kind is _OptPenDepth:
            data = {
                "value": _quantity_text(dispersion._opt_pen_depth, u.m),
                "n": _number(dispersion._n),
            }
            if dispersion._base is not None:
                data["base"] = self.dispersion(dispersion._base)
            return {"opt_pen_depth": data}
        if kind is _MassAttenuation:
            return {
                "mass_attenuation": {
                    "energy": _plain(dispersion._energy.tolist()),
                    "values": _plain(dispersion._values.tolist()),
                }
            }
        if kind is _Fallback:
            return {"fallback": [self.dispersion(dispersion._a), self.dispersion(dispersion._b)]}
        if kind is _Sum:
            operation = "sum" if dispersion._sign > 0 else "difference"
            return {operation: [self.dispersion(dispersion._a), self.dispersion(dispersion._b)]}
        if kind is _Scaled:
            return {
                "scaled": {
                    "factor": _number(dispersion._factor),
                    "dispersion": self.dispersion(dispersion._a),
                }
            }
        raise TypeError(f"Cannot write the dispersion {dispersion!r} of type {kind.__name__}!")

    def _write_recipe(self, recipe):
        if "file" in recipe:
            data = {"file": self._path(recipe["file"])}
            for key, default in (("kind", "n"), ("x_unit", None), ("delimiter", None)):
                if recipe[key] != default:
                    data[key] = recipe[key]
            molar_mass = recipe["molar_mass"]
            if molar_mass is not None:
                data["molar_mass"] = (
                    _quantity_text(molar_mass.magnitude, molar_mass.units)
                    if isinstance(molar_mass, u.Quantity)
                    else _number(molar_mass)
                )
            return data
        page = dict(recipe["refractiveindex"])
        if page["num"] == 1000:
            del page["num"]
        return {"refractiveindex": page, "version": recipe["version"]}

    # structures

    def sub_structure(self, sub_structure):
        if isinstance(sub_structure, Structure):
            key, new = self._reserve(sub_structure, sub_structure.name, "layers")
            if new:
                self.structures[key] = self.structure(sub_structure, key)
            return key
        return self.layer(sub_structure)

    def structure(self, structure, key=None):
        # the name of a nested structure is its key, unless they differ
        data = {} if key == structure.name else {"name": structure.name}
        data["sub_structures"] = [
            [self.sub_structure(sub), _number(N)] for sub, N in structure.sub_structures
        ]
        superstrate, _ = structure.superstrate
        if not self._is_default_vacuum(superstrate):
            data["superstrate"] = self.layer(superstrate)
        substrate, N = structure.substrate
        if not (self._is_default_vacuum(substrate) and N == 1):
            data["substrate"] = [self.layer(substrate), _number(N)]
        return data

    def _is_default_vacuum(self, layer):
        if type(layer) is not Vacuum:
            return False
        if self._default_vacuum is None:
            self._default_vacuum = self._write_layer(Vacuum(), "vacuum")
        return _same(self._write_layer(layer, layer.id), self._default_vacuum)


def _write_angle(angle):
    """An angle (rad) as float in deg, if read back exactly, else in rad."""
    degree = float(np.rad2deg(angle))
    if float(np.deg2rad(degree)) == angle:
        return degree
    return f"{float(angle)!r} rad"


def _write_position(position):
    """Position of an atom in a unit cell as number for the strain-dependent
    position ``x*(1+s)`` of a number, else as its formula string."""
    suffix = "*(1+s)"
    if position.endswith(suffix):
        number = position[: -len(suffix)]
        try:
            value = _parse_number(number)
        except ValueError:
            return position
        if not isinstance(value, complex) and str(value) == number:
            return value
    return position


_ELEMENT_NAMES = {}


def _element_name(symbol):
    """Name of an element as given to an Atom."""
    if not _ELEMENT_NAMES:
        filename = os.path.join(os.path.dirname(__file__), "../parameters/elements.dat")
        symbols = np.genfromtxt(filename, dtype="U2", usecols=(0))
        names = np.genfromtxt(filename, dtype="U15", usecols=(1))
        _ELEMENT_NAMES.update(zip(symbols.tolist(), names.tolist()))
    return _ELEMENT_NAMES.get(symbol)


# ============================================================================
# reading
# ============================================================================


class _Reader:
    """Reads the atoms, layers, and structures from their dicts, where every
    object is created once on its first reference."""

    def __init__(self, data, base_dir=None):
        if not isinstance(data, dict):
            raise ValueError("A structure must be given as YAML mapping!")
        version = data.get("format_version")
        if not isinstance(version, int) or version < 1:
            raise ValueError(f"Unknown format_version {version!r} of the structure!")
        if version > FORMAT_VERSION:
            raise ValueError(
                f"The format_version {version} of the structure is newer than the "
                f"supported version {FORMAT_VERSION}, please update udkm1Dsim!"
            )
        unknown = set(data) - {
            "format_version",
            "udkm1Dsim_version",
            "atoms",
            "layers",
            "structures",
            "structure",
        }
        if unknown:
            raise ValueError(f"Unknown sections {sorted(unknown)} of the structure!")
        if "structure" not in data:
            raise ValueError("The structure is missing in its section 'structure'!")
        self.data = data
        self.base_dir = base_dir
        self._objects = {"atoms": {}, "layers": {}, "structures": {}}
        self._pending = set()

    def _section(self, name):
        section = self.data.get(name) or {}
        if not isinstance(section, dict):
            raise ValueError(f"The section '{name}' must be a mapping by the keys!")
        return section

    def _get(self, section, key, read):
        """Object of a key, which is created on its first reference."""
        objects = self._objects[section]
        if key in objects:
            return objects[key]
        data = self._section(section)
        if key not in data:
            raise ValueError(f"The key '{key}' is not defined in the section '{section}'!")
        if (section, key) in self._pending:
            raise ValueError(f"The key '{key}' of the section '{section}' references itself!")
        self._pending.add((section, key))
        try:
            obj = read(key, dict(data[key] or {}))
        except (ValueError, TypeError) as e:
            # the error with the section and key of the object, also for errors
            # with other arguments, e.g. of pint
            error = ValueError if isinstance(e, ValueError) else TypeError
            raise error(f"{section} '{key}': {e}") from e
        finally:
            self._pending.discard((section, key))
        objects[key] = obj
        return obj

    def _path(self, path):
        path = str(path)
        if self.base_dir is None or os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    @staticmethod
    def _check_empty(data, kind):
        if data:
            raise ValueError(f"Unknown entries {sorted(data)} of the {kind}!")

    # atoms

    def atom(self, key):
        return self._get("atoms", key, self._read_atom)

    def _read_atom(self, key, data):
        mixed = data.pop("type", "Atom")
        if mixed not in ("Atom", "AtomMixed"):
            raise ValueError(f"Unknown type '{mixed}' of an atom!")
        mixed = mixed == "AtomMixed"
        kwargs = {"id": data.pop("id", key)}
        if "magnetization" in data:
            kwargs["magnetization"] = _read_vector(data.pop("magnetization"))
        if "ionicity" in data:
            if mixed:
                raise ValueError("The ionicity of a mixed atom is given by its constituents!")
            kwargs["ionicity"] = data.pop("ionicity")
        tables = {}
        for kind in ("atomic", "magnetic"):
            source = data.pop(f"{kind}_form_factors", None)
            if source is None:
                continue
            if isinstance(source, str) and kind == "atomic" and not mixed:
                kwargs["atomic_form_factor_source"] = source
            elif isinstance(source, dict) and set(source) == {"file"}:
                kwargs[f"{kind}_form_factor_path"] = self._path(source["file"])
            elif isinstance(source, dict) and "table" in source:
                tables[kind] = source
            else:
                raise ValueError(f"Unknown {kind}_form_factors {source!r}!")
        name = data.pop("name", None)
        cromer_mann = data.pop("cromer_mann_coeff", None)
        constituents = data.pop("atoms", [])
        if constituents and not mixed:
            raise ValueError("Only a mixed atom has constituent atoms!")
        if cromer_mann is not None and mixed:
            raise ValueError("A mixed atom has no Cromer-Mann coefficients!")
        symbol = data.pop("symbol", None)
        if symbol is None:
            raise ValueError("The symbol of the atom is missing!")
        self._check_empty(data, "atom")

        if mixed:
            atom = AtomMixed(symbol, name=name if name is not None else symbol, **kwargs)
            for constituent, fraction in constituents:
                atom.add_atom(self.atom(constituent), fraction)
        else:
            atom = Atom(symbol, **kwargs)
            if name is not None:
                atom.name = name
            if cromer_mann is not None:
                atom.cromer_mann_coeff = cromer_mann
        for kind, table in tables.items():
            coeff = table["table"]
            setattr(
                atom,
                f"{kind}_form_factor_coeff",
                None if coeff is None else np.asarray(coeff, dtype=float),
            )
            setattr(atom, f"_{kind}_form_factor_source", table.get("source", "custom"))
        return atom

    # layers

    def sub_structure(self, key):
        in_layers = key in self._section("layers")
        in_structures = key in self._section("structures")
        if in_layers and in_structures:
            raise ValueError(f"The key '{key}' is defined as layer and as structure!")
        if in_structures:
            return self._get("structures", key, self._read_structure)
        return self.layer(key)

    def layer(self, key):
        return self._get("layers", key, self._read_layer)

    def _read_layer(self, key, data):
        type_name = data.pop("type", "Layer")
        classes = {name: cls for cls, name in _LAYER_TYPES.items()}
        if type_name not in classes:
            raise ValueError(f"Unknown type '{type_name}' of a layer!")
        cls = classes[type_name]
        layer_id = data.pop("id", key)
        name = data.pop("name", layer_id)
        groups = {}
        for group_name in cls._parameter_group_names:
            if group_name in data:
                group_class = getattr(cls, group_name).group_class
                groups[group_name] = self._read_group(group_class, data.pop(group_name))
        atom = data.pop("atom", None)
        atoms = data.pop("atoms", [])
        higher_orders = data.pop("ho_spring_constants", None)
        self._check_empty(data, "layer")
        if atom is not None and cls is not AmorphousLayer:
            raise ValueError("Only an AmorphousLayer has an atom!")
        if atoms and cls is not UnitCell:
            raise ValueError("Only a UnitCell has atoms!")

        if cls is Vacuum:
            layer = Vacuum()
            layer.id, layer.name = layer_id, name
            for group_name, values in groups.items():
                setattr(layer, group_name, values)
        elif cls is AmorphousLayer:
            atom = self.atom(atom) if atom is not None else []
            layer = AmorphousLayer(layer_id, name, atom=atom, **groups)
            # the atom sets the magnetization of the layer
            magnetization = groups.get("magnetic", {}).get("magnetization", VectorParameter(""))
            _apply_magnetization(layer.magnetic, magnetization)
        else:
            if cls is UnitCell and "lattice" not in groups:
                raise ValueError("The lattice of the UnitCell is missing!")
            layer = cls(layer_id, name, **groups)
        for atom_key, position in atoms:
            layer.add_atom(self.atom(atom_key), position)
        if higher_orders is not None:
            layer.set_ho_spring_constants(np.asarray(higher_orders))
        return layer

    def _read_group(self, group_class, data):
        """Inputs of a parameter group."""
        if not isinstance(data, dict):
            raise ValueError(f"The parameters of {group_class.__name__} must be a mapping!")
        names = group_class._input_names()
        default = group_class()
        values = {}
        for name, value in data.items():
            if name not in names:
                raise ValueError(f"Unknown parameter '{name}' of {group_class.__name__}!")
            target = getattr(default, name, None)
            if group_class is MagneticParameters and name == "magnetization":
                values[name] = _read_magnetization(value)
            elif isinstance(target, Dielectric):
                values[name] = self.dielectric(value)
            elif isinstance(target, TemperatureParameter):
                values[name] = _read_temperature(value, target.unit)
            elif name == "sub_system_labels":
                values[name] = value
            else:
                values[name] = _read_quantity(value)
        return values

    # dielectric tensor

    def dielectric(self, data):
        """A Dielectric of its dict, or a Dispersion for the dict of a
        dispersion, which sets the dielectric tensor without strain model."""
        if not isinstance(data, dict):
            raise ValueError(f"The dielectric must be given as mapping, not as {data!r}!")
        if data and not set(data) & set(_DIELECTRIC_INPUTS):
            return self.dispersion(data)
        data = dict(data)
        kwargs = {}
        for key in ("eps0", "eps_mag"):
            if key in data:
                kwargs[key] = self.dispersion(data.pop(key))
        if "strain_model" in data:
            kwargs["strain_model"] = data.pop("strain_model")
        if "deps_dstrain" in data:
            deps = data.pop("deps_dstrain")
            kwargs["deps_dstrain"] = (
                self.dispersion(deps) if isinstance(deps, dict) else _read_complex_array(deps)
            )
        if "euler_angles" in data:
            kwargs["euler_angles"] = [_read_quantity(a) for a in data.pop("euler_angles")]
        energy = data.pop("display_energy", None)
        self._check_empty(data, "dielectric")
        dielectric = Dielectric(**kwargs)
        if energy is not None:
            dielectric.display_energy = _read_quantity(energy)
        return dielectric

    def dispersion(self, data):
        if not isinstance(data, dict):
            raise ValueError(f"A dispersion must be given as mapping, not as {data!r}!")
        data = dict(data)
        if "ref_index" in data:
            value = data.pop("ref_index")
            self._check_empty(data, "dispersion")
            return Dispersion.from_ref_index(_read_complex_array(value))
        if "constant" in data:
            value = data.pop("constant")
            kind = data.pop("kind", "eps")
            self._check_empty(data, "dispersion")
            return Dispersion.from_array(None, _read_complex_array(value), kind=kind)
        if "file" in data:
            path = self._path(data.pop("file"))
            kwargs = {key: data.pop(key) for key in ("kind", "x_unit", "delimiter") if key in data}
            if "molar_mass" in data:
                kwargs["molar_mass"] = _read_quantity(data.pop("molar_mass"))
            self._check_empty(data, "dispersion")
            return Dispersion.from_file(path, **kwargs)
        if "refractiveindex" in data:
            page = dict(data.pop("refractiveindex"))
            version = data.pop("version", None)
            self._check_empty(data, "dispersion")
            dispersion = Dispersion.from_refractiveindex(**page)
            current = dispersion._recipe["version"]
            if version is not None and version != current:
                warnings.warn(
                    f"The page {page} was written for the version {version} of the database "
                    f"of refractiveindex.info, but the version {current} is used.",
                    stacklevel=2,
                )
            return dispersion
        if "atoms" in data:
            atoms = [(self.atom(key), fraction) for key, fraction in data.pop("atoms")]
            magnetic = data.pop("magnetic", False)
            self._check_empty(data, "dispersion")
            return _Atoms(atoms, magnetic=magnetic)
        if len(data) != 1:
            raise ValueError(f"Unknown dispersion {data!r}!")
        ((kind, value),) = data.items()
        if kind == "table":
            return self._read_table(dict(value))
        if kind == "form_factors":
            value = dict(value)
            table = SpectralTable(
                np.asarray(value.pop("energy"), dtype=float),
                _read_complex_array(value.pop("values")),
                source=value.pop("source", ""),
            )
            dispersion = _FormFactors(
                table, _read_quantity(value.pop("molar_mass")), value.pop("magnetic", False)
            )
            self._check_empty(value, "form factors")
            return dispersion
        if kind == "opt_pen_depth":
            value = dict(value)
            base = value.pop("base", None)
            dispersion = _OptPenDepth(
                _read_quantity(value.pop("value")),
                n=value.pop("n", 1.0),
                base=None if base is None else self.dispersion(base),
            )
            self._check_empty(value, "optical penetration depth")
            return dispersion
        if kind == "mass_attenuation":
            value = dict(value)
            dispersion = _MassAttenuation(
                np.asarray(value.pop("energy"), dtype=float),
                Q_(np.asarray(value.pop("values"), dtype=float), "m**2/kg"),
            )
            self._check_empty(value, "mass attenuation")
            return dispersion
        if kind in ("fallback", "sum", "difference"):
            a, b = (self.dispersion(v) for v in value)
            if kind == "fallback":
                return _Fallback(a, b)
            return _Sum(a, b, sign=1 if kind == "sum" else -1)
        if kind == "scaled":
            value = dict(value)
            factor = value.pop("factor")
            if isinstance(factor, str):
                factor = _parse_number(factor)
            dispersion = _Scaled(self.dispersion(value.pop("dispersion")), factor)
            self._check_empty(value, "scaled dispersion")
            return dispersion
        raise ValueError(f"Unknown dispersion '{kind}'!")

    @staticmethod
    def _read_table(table):
        energy = np.asarray(table.pop("energy"), dtype=float)
        values = _read_complex_array(table.pop("values"))
        kind = table.pop("kind", "eps")
        origin = table.pop("origin", "")
        if "input_kind" in table:
            # the stored values of a written table
            input_kind = table.pop("input_kind")
            if table:
                raise ValueError(f"Unknown entries {sorted(table)} of the table!")
            dispersion = _Table(energy, values, values.ndim == 1, kind, input_kind)
        else:
            if table:
                raise ValueError(f"Unknown entries {sorted(table)} of the table!")
            dispersion = Dispersion.from_array(energy, values, kind=kind)
        dispersion._origin = origin
        return dispersion

    # structures

    def _read_structure(self, key, data):
        structure = Structure(data.pop("name", key))
        for sub_key, N in data.pop("sub_structures", None) or []:
            structure.add_sub_structure(self.sub_structure(sub_key), N)
        if "superstrate" in data:
            structure.add_superstrate(self.layer(data.pop("superstrate")))
        if "substrate" in data:
            substrate = data.pop("substrate")
            if isinstance(substrate, list):
                key, N = substrate
            else:
                key, N = substrate, 1
            structure.add_substrate(self.layer(key), N)
        self._check_empty(data, "structure")
        return structure

    def structure(self):
        data = self.data["structure"]
        if not isinstance(data, dict):
            raise ValueError("The section 'structure' must be a mapping!")
        return self._read_structure("structure", dict(data))


# ============================================================================
# public functions
# ============================================================================


def structure_to_dict(structure, base_dir=None, embed=False):
    """structure_to_dict

    Dict of a structure as written to YAML, see :meth:`.Structure.to_yaml`.

    Args:
        structure (Structure): structure.
        base_dir (str, optional): directory to which the paths of files are
            written relative. Defaults to None for the paths as given.
        embed (bool, optional): write the tabulated values of files instead
            of their paths. Defaults to False.

    Returns:
        data (dict): dict of the structure.

    """
    from .. import __version__

    if not isinstance(structure, Structure):
        raise TypeError(f"Only a Structure can be written, not {type(structure).__name__}!")
    writer = _Writer(base_dir, embed)
    root = writer.structure(structure)
    data = {"format_version": FORMAT_VERSION, "udkm1Dsim_version": __version__}
    for name in ("atoms", "layers", "structures"):
        section = getattr(writer, name)
        if section:
            data[name] = section
    data["structure"] = root
    return data


def structure_from_dict(data, base_dir=None):
    """structure_from_dict

    Structure of its dict, see :meth:`.Structure.from_yaml`.

    Args:
        data (dict): dict of the structure.
        base_dir (str, optional): directory of relative paths of files.
            Defaults to None for the current directory.

    Returns:
        structure (Structure): structure.

    """
    return _Reader(data, base_dir).structure()


class _Dumper(yaml.SafeDumper):
    """Dumper, which writes lists of numbers in a single line."""


def _represent_list(dumper, data):
    flow = all(not isinstance(v, (list, dict)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def _dump(data):
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)


def _first_difference(a, b, path=""):
    """Path of the first difference of two dicts, compared by their text."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in list(a) + [k for k in b if k not in a]:
            if key not in a or key not in b:
                return f"{path}/{key}"
            difference = _first_difference(a[key], b[key], f"{path}/{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            difference = _first_difference(x, y, f"{path}[{i:d}]")
            if difference is not None:
                return difference
        return None
    return None if _same(a, b) else path or "/"


def _check_round_trip(structure, data, base_dir, embed):
    """Read the dict of the structure back and compare it and the hashes of
    the structure, so that nothing is lost on writing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        read = structure_from_dict(data, base_dir)
        written = structure_to_dict(read, base_dir, embed)
    difference = _first_difference(data, written)
    if difference is not None:
        raise ValueError(
            f"The structure '{structure.name}' cannot be written exactly to YAML, as "
            f"'{difference}' differs on reading it back!"
        )
    for types in _HASH_TYPES:
        if structure.get_hash(types=types) != read.get_hash(types=types):
            raise ValueError(
                f"The structure '{structure.name}' cannot be written exactly to YAML, as "
                f"its hash of the {types} parameters differs on reading it back!"
            )


def structure_to_yaml(structure, filename=None, embed=False, check=True):
    """structure_to_yaml

    See :meth:`.Structure.to_yaml`.

    """
    base_dir = None if filename is None else os.path.dirname(os.path.abspath(filename))
    data = structure_to_dict(structure, base_dir, embed)
    if check:
        _check_round_trip(structure, data, base_dir, embed)
    text = _dump(data)
    if filename is None:
        return text
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return None


def structure_from_yaml(source):
    """structure_from_yaml

    See :meth:`.Structure.from_yaml`.

    """
    base_dir = None
    if isinstance(source, os.PathLike) or (
        isinstance(source, str) and "\n" not in source and os.path.isfile(source)
    ):
        base_dir = os.path.dirname(os.path.abspath(source))
        with open(source, encoding="utf-8") as f:
            text = f.read()
    else:
        text = source
    data = yaml.safe_load(text)
    if isinstance(data, str) and "\n" not in source:
        raise FileNotFoundError(f"The YAML file '{source}' does not exist!")
    return structure_from_dict(data, base_dir)
