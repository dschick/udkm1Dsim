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

__all__ = ["Simulation"]

__docformat__ = "restructuredtext"

import os
import warnings
from datetime import datetime, timezone
from time import time

import numpy as np
from tabulate import tabulate

from .result import Result


class Simulation:
    """Simulation

    Base class for all simulations.

    Handles the caching and some displaying option.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.

    Attributes:
        S (Structure): sample structure to calculate simulations on.
        force_recalc (boolean): force recalculation of results.
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.

    """

    _title = "Simulation"

    def __init__(self, S, force_recalc, **kwargs):
        self.S = S
        self.force_recalc = force_recalc
        self.save_data = kwargs.get("save_data", True)
        self.cache_dir = kwargs.get("cache_dir", "./")
        self.disp_messages = kwargs.get("disp_messages", True)
        self.progress_bar = kwargs.get("progress_bar", True)

    def _property_rows(self):
        """[parameter, value] rows of the simulation properties, which are
        extended by the sub-classes"""
        return [
            ["force recalc", self.force_recalc],
            ["cache directory", self.cache_dir],
            ["display messages", self.disp_messages],
            ["save data", self.save_data],
            ["progress bar", self.progress_bar],
        ]

    def _table_representation(self, style):
        return tabulate(
            self._property_rows(),
            headers=["parameter", "value"],
            tablefmt=style,
            colalign=("right",),
            floatfmt=(".2f", ".2f"),
        )

    def __repr__(self):
        """String representation of this class"""
        class_str = f"{self._title} properties:\n\n"
        class_str += self._table_representation("rst")
        class_str += "\n\nThis is the current structure for the simulations:\n\n"
        class_str += repr(self.S)
        return class_str

    def _repr_html_(self):
        """HTML representation of this class"""
        class_str = f"<h2>{self._title}</h2>"
        class_str += self._table_representation("html")
        class_str += self.S._repr_html_()
        return class_str

    def disp_message(self, message):
        """disp_message

        Wrapper to display messages for that class.

        Args:
            message (str): message to display.

        """
        if self.disp_messages:
            print(message)

    def save(self, full_filename, result, *args):
        """save

        Save a result to a HDF5 file, if :attr:`save_data` is true. The
        variable name for the message can be handed as variable argument.

        Args:
            full_filename (str): full file name to data file.
            result (Result): result to save.
            *args (str, optional): variable name for the message.

        """
        var_name = args[0] if len(args) == 1 else "_data_"
        if self.save_data:
            result.save(full_filename)
            filename = os.path.basename(full_filename)
            self.disp_message(f"{var_name:s} saved to file:\n\t {filename:s}")

    def new_result(self, title, hash=""):
        """new_result

        New :class:`~udkm1Dsim.simulations.result.Result` with the metadata
        of this simulation, i.e. its class, the version of udkm1Dsim, the
        time of creation, the ``hash``, the sample structure, and the
        parameters of the simulation.

        Args:
            title (str): title of the result.
            hash (str, optional): hash of the result.

        Returns:
            result (Result): empty result.

        """
        from .. import __version__

        return Result(
            title,
            {
                "simulation": type(self).__name__,
                "version": __version__,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hash": hash,
                "structure": repr(self.S),
                "parameters": {str(key): value for key, value in self._property_rows()},
            },
        )

    def new_map_result(
        self, title, delays=None, distances=None, source=None, num_delays=None, num_distances=None
    ):
        """new_map_result

        New :class:`~udkm1Dsim.simulations.result.Result` of maps, see
        :meth:`new_result`, with the coordinates ``delays`` (s) and
        ``distances`` (m), which are the given ones or the ones of an upstream
        ``source`` result, if their sizes agree with ``num_delays`` and
        ``num_distances``. Otherwise, the ``distances`` are the centers of the
        layers of the structure, if their number agrees.

        Args:
            title (str): title of the result.
            delays (ndarray[float, Quantity], optional): delays (s).
            distances (ndarray[float, Quantity], optional): distances (m).
            source (Result, optional): upstream result.
            num_delays (int, optional): number of delays of the maps.
            num_distances (int, optional): number of distances of the maps.

        Returns:
            result (Result): empty result with coordinates.

        """
        result = self.new_result(title)
        if delays is not None:
            result.add_coord("delays", delays, "s")
        if distances is not None:
            result.add_coord("distances", distances, "m")
        if isinstance(source, Result):
            for name, size in (("delays", num_delays), ("distances", num_distances)):
                if name in source.coords and (size is None or np.size(source[name]) == size):
                    result.copy_coords(source, [name])
        if "distances" not in result:
            layers = self.S._layer_distances()[2]
            if num_distances is None or np.size(layers) == num_distances:
                result.add_coord("distances", layers, "m")
        return result

    def _timed(self, name, calc, *args, **kwargs):
        """Result of ``calc(*args, **kwargs)`` with the messages of its
        calculation and elapsed time under the ``name``."""
        t1 = time()
        self.disp_message(f"Calculating _{name}_ ...")
        result = calc(*args, **kwargs)
        self.disp_message(f"Elapsed time for _{name}_: {time() - t1:f} s")
        return result

    def _cached_result(self, name, hash, calc, timed=True):
        """Result of ``calc()`` loaded from or saved to the HDF5 file
        ``{name}_{hash}.h5`` in the :attr:`cache_dir`, whose calculation is
        timed, unless ``calc`` displays its time itself."""
        filename = f"{name}_{hash}.h5"
        full_filename = os.path.abspath(os.path.join(self.cache_dir, filename))
        if os.path.exists(full_filename) and not self.force_recalc:
            try:
                result = Result.load(full_filename)
            except (OSError, KeyError):
                self.disp_message(f"_{name}_ cannot be loaded from file:\n\t{filename}")
            else:
                self.disp_message(f"_{name}_ loaded from file:\n\t{filename}")
                return result
        result = self._timed(name, calc) if timed else calc()
        result.attrs["hash"] = hash
        self.save(full_filename, result, f"_{name}_")
        return result

    @staticmethod
    def conv_with_function(y, x, handle):
        """conv_with_function

        Convolutes the array :math:`y(x)` with a function given by the handle
        on the argument array :math:`x`.

        Args:
            y (ndarray[float]): y data.
            x (ndarray[float]): x data.
            handle (@lamdba): convolution function.

        Returns:
            y_conv (ndarray[float]): convoluted data.

        """
        dx = np.min(np.diff(x))
        x_lin = np.r_[np.min(x) : np.max(x) : dx]
        y_lin = np.interp(x_lin, x, y)
        x0 = np.mean(x_lin)
        y_handle = handle(x_lin - x0)

        temp = np.convolve(y_lin, y_handle / y_handle.sum(), mode="same")

        y_conv = np.interp(x, x_lin, temp)
        # finally remove NaN entries due to the interpolation
        y_conv[np.isnan(y_conv)] = 0
        return y_conv

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir):
        import os.path as path

        if path.exists(cache_dir):
            self._cache_dir = cache_dir
        else:
            warnings.warn("Cache dir does not exist. Please create the path first.")
