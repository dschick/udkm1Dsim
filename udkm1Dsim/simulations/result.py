#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2026 Daniel Schick
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

"""Results of the simulations with their coordinates, units, and metadata,
which can be saved to and loaded from HDF5 files."""

__all__ = ["Result", "get_entry", "find_source"]

__docformat__ = "restructuredtext"

import warnings

import numpy as np
import pint
from tabulate import tabulate

u = pint.get_application_registry()

#: minimum number of elements of a dataset, which is compressed in HDF5 files
_COMPRESSION_SIZE = 1024


class Result:
    r"""Result

    Dict-like container of the results of a simulation, which are accessed by
    their name, either as item ``result["temp_map"]`` or as attribute
    ``result.temp_map``. The data are plain numpy arrays. Their units are
    stored as metadata and :meth:`quantity` returns an entry as pint
    Quantity.

    The coordinates of the data, e.g. the ``delays`` or ``distances``, are
    accessed in the same way, while :attr:`data` and :attr:`coords` hold the
    data entries and the coordinates separately and :meth:`keys`,
    :meth:`values`, and :meth:`items` only refer to the data. The names of
    the entries must not be attributes of the class, e.g. ``data`` or
    ``save``. The names of the dimensions of
    each entry are given by :meth:`dims`.

    Data entries can have short :attr:`aliases`, e.g. ``R`` for the
    ``reflectivity``, which are accessed in the same way as their names, but
    are not listed by :meth:`keys` or :attr:`data`.

    A result cannot be unpacked as a tuple, since the order of its entries
    is not part of its meaning. The metadata in :attr:`attrs` hold the title,
    the simulation, the version of udkm1Dsim, the hash, the sample structure,
    and the parameters of the simulation.

    Args:
        title (str, optional): title of the result.
        attrs (dict, optional): metadata of the result.

    Attributes:
        attrs (dict): metadata of the result.

    """

    def __init__(self, title="", attrs=None):
        self._data = {}
        self._coords = {}
        self._dims = {}
        self._units = {}
        self._aliases = {}
        self.attrs = {"title": title}
        if attrs is not None:
            self.attrs.update(attrs)

    @staticmethod
    def _parse_value(value, unit):
        """the magnitude and unit of a value, which can be a Quantity"""
        if isinstance(value, u.Quantity):
            if unit:
                value = value.to(unit)
            else:
                unit = f"{value.units:~}"
            value = value.magnitude
        if isinstance(value, (list, tuple)) or np.isscalar(value):
            try:
                value = np.asarray(value)
            except ValueError:
                pass  # ragged lists are kept as they are
        return value, unit

    def _check_dims(self, name, value, dims):
        """dimension names of an entry, whose sizes must agree with the
        coordinates of the same name"""
        dims = tuple(dims)
        if not isinstance(value, np.ndarray):
            return ()  # e.g. lists of the matrices of sub-structures
        shape = np.shape(value)
        if len(dims) > value.ndim:
            raise ValueError(
                f"{name} has {value.ndim} dimensions, but {len(dims)} dims are given!"
            )
        for axis, dim in enumerate(dims):
            if dim in self._coords and self._coords[dim].ndim == 1:
                if np.size(self._coords[dim]) != shape[axis]:
                    raise ValueError(
                        f"the size {shape[axis]} of axis {axis} of {name} does not agree "
                        f"with the size {np.size(self._coords[dim])} of {dim}!"
                    )
        return dims

    @classmethod
    def _check_name(cls, name):
        """the name of an entry must be an identifier, which is not an
        attribute of the class, so that the entry is accessible as attribute"""
        if not str(name).isidentifier() or name.startswith("_"):
            raise ValueError(f"{name} must be an identifier, which does not start with _!")
        if hasattr(cls, name) or name == "attrs":
            raise ValueError(f"{name} is reserved for an attribute of Result!")

    def add(self, name, value, dims=(), unit="", aliases=()):
        """add

        Add a data entry to the result.

        Args:
            name (str): name of the entry.
            value (ndarray, Quantity, any): data, a Quantity is converted to
                its magnitude in its own or the given ``unit``.
            dims (tuple[str], optional): names of the first dimensions of the
                data, which refer to coordinates of the same name, if
                present.
            unit (str, optional): unit of the data.
            aliases (str, tuple[str], optional): short aliases of the entry.

        """
        self._check_name(name)
        if name in self._coords:
            raise KeyError(f"{name} is already a coordinate!")
        if name in self._aliases:
            raise KeyError(f"{name} is already an alias of {self._aliases[name]}!")
        aliases = (aliases,) if isinstance(aliases, str) else tuple(aliases)
        for alias in aliases:
            self._check_name(alias)
            if alias in self._data or alias in self._coords or alias == name:
                raise KeyError(f"the alias {alias} is already a name of the result!")
            if self._aliases.get(alias, name) != name:
                raise KeyError(f"{alias} is already an alias of {self._aliases[alias]}!")
        value, unit = self._parse_value(value, unit)
        self._dims[name] = self._check_dims(name, value, dims)
        self._data[name] = value
        self._units[name] = unit
        for alias in aliases:
            self._aliases[alias] = name

    def add_coord(self, name, value, unit="", dims=None):
        """add_coord

        Add a coordinate to the result.

        Args:
            name (str): name of the coordinate.
            value (ndarray, Quantity): coordinate, a Quantity is converted to
                its magnitude in its own or the given ``unit``.
            unit (str, optional): unit of the coordinate.
            dims (tuple[str], optional): names of the dimensions of the
                coordinate. Defaults to ``(name,)`` for a 1D coordinate.

        """
        self._check_name(name)
        if name in self._data or name in self._aliases:
            raise KeyError(f"{name} is already a data entry or alias!")
        value, unit = self._parse_value(value, unit)
        value = np.asarray(value)
        if dims is None:
            dims = (name,) if value.ndim == 1 else ()
        self._coords[name] = value
        self._dims[name] = tuple(dims)
        self._units[name] = unit

    def copy_coords(self, other, names=None):
        """copy_coords

        Copy coordinates of another result, e.g. of an upstream simulation.

        Args:
            other (Result): result to copy from.
            names (list[str], optional): names of the coordinates, defaults
                to all coordinates of ``other``.

        """
        for name in other.coords if names is None else names:
            if name in other.coords and name not in self._coords:
                self.add_coord(name, other[name], other.unit(name), other.dims(name))

    def select(self, *names):
        """select

        New result with the given data entries, all coordinates, and a copy
        of the metadata.

        Args:
            *names (str): names of the data entries.

        Returns:
            result (Result): result with the selected entries.

        """
        result = Result(attrs=dict(self.attrs))
        result.copy_coords(self)
        for name in names:
            name = self._resolve(name)
            result.add(
                name, self[name], self._dims[name], self._units[name], self.aliases_of(name)
            )
        return result

    @property
    def aliases(self):
        """dict[str, str]: aliases and the names of their data entries."""
        return dict(self._aliases)

    def aliases_of(self, name):
        """aliases of a data entry"""
        name = self._resolve(name)
        return tuple(alias for alias, target in self._aliases.items() if target == name)

    @property
    def data(self):
        """dict[str, ndarray]: data entries of the result."""
        return dict(self._data)

    @property
    def coords(self):
        """dict[str, ndarray]: coordinates of the result."""
        return dict(self._coords)

    def keys(self):
        """names of the data entries"""
        return self._data.keys()

    def values(self):
        """data entries"""
        return self._data.values()

    def items(self):
        """names and data entries"""
        return self._data.items()

    def get(self, name, default=None):
        """data entry or coordinate of the given name or the default"""
        return self[name] if name in self else default

    def dims(self, name):
        """names of the dimensions of an entry or coordinate"""
        return self._dims[self._resolve(name)]

    def unit(self, name):
        """unit of an entry or coordinate"""
        return self._units[self._resolve(name)]

    def quantity(self, name, unit=None):
        """quantity

        Entry or coordinate as pint Quantity in its unit or the given
        ``unit``.

        Args:
            name (str): name of the entry or coordinate.
            unit (str, optional): unit to convert to, e.g. ``"ps"`` for the
                ``delays``.

        Returns:
            quantity (Quantity): entry or coordinate with unit.

        """
        quantity = u.Quantity(self[name], self.unit(name) or "dimensionless")
        return quantity if unit is None else quantity.to(unit)

    def _resolve(self, name):
        """name of an entry, coordinate, or of the entry of an alias"""
        if name in self._data or name in self._coords:
            return name
        if name in self._aliases:
            return self._aliases[name]
        raise KeyError(f"{name} is not part of the result, which holds {self._names()}")

    def __getitem__(self, name):
        name = self._resolve(name)
        return self._data[name] if name in self._data else self._coords[name]

    def __getattr__(self, name):
        # only called if the attribute is not found the usual way
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(error.args[0]) from None

    def __contains__(self, name):
        return name in self._data or name in self._coords or name in self._aliases

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        raise TypeError(
            "a Result cannot be iterated or unpacked, access its entries by name, e.g. "
            f"result['{next(iter(self._data), 'name')}'], of {self._names()}"
        )

    def __dir__(self):
        return list(super().__dir__()) + self._names() + list(self._aliases)

    def _names(self):
        """names of all data entries and coordinates"""
        return list(self._data) + list(self._coords)

    def _rows(self):
        """[name, alias, kind, shape, dims, unit] rows of the table
        representation"""
        rows = []
        for kind, entries in (("data", self._data), ("coord", self._coords)):
            for name, value in entries.items():
                shape = np.shape(value) if isinstance(value, np.ndarray) else type(value).__name__
                aliases = ", ".join(self.aliases_of(name)) if kind == "data" else ""
                dims = ", ".join(self._dims[name])
                rows.append([name, aliases, kind, shape, dims, self._units[name]])
        return rows

    def _table_representation(self, style):
        return tabulate(
            self._rows(),
            headers=["name", "alias", "kind", "shape", "dims", "unit"],
            tablefmt=style,
        )

    def __repr__(self):
        class_str = f"Result: {self.attrs.get('title', '')}\n\n"
        return class_str + self._table_representation("rst")

    def _repr_html_(self):
        class_str = f"<h3>Result: {self.attrs.get('title', '')}</h3>"
        return class_str + self._table_representation("html")

    @staticmethod
    def _attr_value(value):
        """value of a metadata attribute, which can be stored in HDF5"""
        if isinstance(value, (str, bool, int, float, np.number, np.bool_)):
            return value
        return str(value)

    def save(self, filename):
        """save

        Save the result to a HDF5 file with the data in the group ``data``
        and the coordinates in the group ``coords``. The units and
        dimension names are attributes of the datasets and 1D coordinates
        are attached to the data as dimension scales. The metadata are
        attributes of the file and the parameters of the simulation are
        attributes of the group ``parameters``. Entries, which are no arrays,
        are skipped with a warning.

        Args:
            filename (str): name of the HDF5 file.

        """
        import h5py

        with h5py.File(filename, "w", track_order=True) as file:
            for key, value in self.attrs.items():
                if key == "parameters":
                    group = file.create_group("parameters", track_order=True)
                    for name, parameter in value.items():
                        group.attrs[name] = self._attr_value(parameter)
                else:
                    file.attrs[key] = self._attr_value(value)
            coords = file.create_group("coords", track_order=True)
            for name, value in self._coords.items():
                self._create_dataset(coords, name, value)
                if self._dims[name] == (name,):
                    coords[name].make_scale(name)
            data = file.create_group("data", track_order=True)
            for name, value in self._data.items():
                if not isinstance(value, np.ndarray) or value.dtype == object:
                    warnings.warn(f"{name} is no array and is not saved to the file.")
                    continue
                dataset = self._create_dataset(data, name, value)
                dataset.attrs["aliases"] = list(self.aliases_of(name))
                for axis, dim in enumerate(self._dims[name]):
                    dataset.dims[axis].label = dim
                    if dim in self._coords and self._dims[dim] == (dim,):
                        dataset.dims[axis].attach_scale(coords[dim])

    def _create_dataset(self, group, name, value):
        """dataset with its unit and dimension names as attributes"""
        compression = "gzip" if np.size(value) >= _COMPRESSION_SIZE else None
        dataset = group.create_dataset(name, data=value, compression=compression)
        dataset.attrs["units"] = self._units[name]
        dataset.attrs["dims"] = list(self._dims[name])
        return dataset

    @classmethod
    def load(cls, filename):
        """load

        Load a result from a HDF5 file, which was written by :meth:`save`.

        Args:
            filename (str): name of the HDF5 file.

        Returns:
            result (Result): loaded result.

        """
        import h5py

        def text(value):
            return value.decode() if isinstance(value, bytes) else value

        def dims(dataset):
            return tuple(text(dim) for dim in dataset.attrs.get("dims", []))

        result = cls()
        with h5py.File(filename, "r") as file:
            result.attrs = {key: text(value) for key, value in file.attrs.items()}
            if "parameters" in file:
                result.attrs["parameters"] = {
                    key: text(value) for key, value in file["parameters"].attrs.items()
                }
            for name, dataset in file["coords"].items():
                result.add_coord(name, dataset[()], text(dataset.attrs["units"]), dims(dataset))
            for name, dataset in file["data"].items():
                aliases = tuple(text(alias) for alias in dataset.attrs.get("aliases", []))
                result.add(name, dataset[()], dims(dataset), text(dataset.attrs["units"]), aliases)
        return result

    def to_xarray(self):
        """to_xarray

        Convert the result to a :class:`xarray.Dataset` with the units as
        attributes. Dimensions without name are named by the entry and the
        axis.

        Returns:
            dataset (xarray.Dataset): result as xarray dataset.

        """
        import xarray as xr

        def variable(name, value):
            dims = list(self._dims[name])
            dims += [f"{name}_dim_{axis}" for axis in range(len(dims), np.ndim(value))]
            return xr.Variable(dims, value, attrs={"units": self._units[name]})

        coords = {name: variable(name, value) for name, value in self._coords.items()}
        data = {
            name: variable(name, value)
            for name, value in self._data.items()
            if isinstance(value, np.ndarray) and value.dtype != object
        }
        attrs = {key: self._attr_value(value) for key, value in self.attrs.items()}
        return xr.Dataset(data, coords=coords, attrs=attrs)


def get_entry(value, name):
    """get_entry

    Entry of a result of an upstream simulation, which is passed on as input,
    or the value itself, if it is no :class:`Result`.

    Args:
        value (Result, any): result or value.
        name (str): name of the entry.

    Returns:
        value (any): entry or value.

    """
    return value[name] if isinstance(value, Result) else value


def find_source(*values):
    """find_source

    First :class:`Result` of the inputs of a simulation, e.g. of an upstream
    simulation, whose coordinates are passed on.

    Args:
        *values (Result, any): inputs of a simulation.

    Returns:
        source (Result, None): first result of the inputs or None.

    """
    return next((value for value in values if isinstance(value, Result)), None)
