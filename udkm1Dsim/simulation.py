#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the udkm1Dsimpy module.
#
# udkm1Dsimpy is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2017 Daniel Schick

"""A :mod:`Simulation` module """

__all__ = ["Simulation"]

__docformat__ = "restructuredtext"

from tabulate import tabulate
import numpy as np
import os


class Simulation:
    """Simulation

    Base class for all simulations. Handles the caching and some
    displaying option

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Keyword Args:
        cache_dir (str): path to cached data
        disp_messages (boolean): is true to display messages from within
           the simulations
        disp_calc_time (boolean): is true to display the duration of
           certain calculations
        progress_bar (boolean): enable tqdm progress bar

    Attributes:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results
        disp_messages (boolean): is true to display messages from within
           the simulations
        disp_calc_time (boolean): is true to display the duration of
           certain calculations
        progress_bar (boolean): enable tqdm progress bar

    """

    def __init__(self, S, force_recalc, **kwargs):
        self.S = S
        self.force_recalc = force_recalc
        self.cache_dir = kwargs.get('cache_dir', './')
        self.disp_messages = kwargs.get('disp_messages', True)
        self.save_data = kwargs.get('save_data', True)
        self.progress_bar = kwargs.get('progress_bar', True)

    def __str__(self, output=[]):
        """String representation of this class"""
        output = [['force recalc', self.force_recalc],
                  ['cache directory', self.cache_dir],
                  ['display messages', self.disp_messages],
                  ['save data', self.save_data],
                  ['progress bar', self.progress_bar]] + output

        class_str = 'This is the current structure for the simulations:\n\n'
        class_str += self.S.__str__()
        class_str += '\n\nDisplay properties:\n\n'
        class_str += tabulate(output, headers=['parameter', 'value'], tablefmt="rst",
                              colalign=('right',), floatfmt=('.2f', '.2f'))
        return class_str

    def disp_message(self, message):
        """wrapper to display messages for that class"""
        if self.disp_messages:
            print(message)

    def save(self, full_filename, data, *args):
        if len(args) == 1:
            var_name = args[0]
        else:
            var_name = '_data_'
        if self.save_data:
            np.save(full_filename, data)
            filename = os.path.basename(full_filename)
            self.disp_message('{:s} saved to file:\n\t {:s}'.format(var_name, filename))

    def conv_with_function(self, y, x, handle):
        """conv_with_function

        Convolutes the array :math;`y(x)` with a function given by the
        handle on the argument vector :math:`x`.

        """
        dx = np.min(np.diff(x))
        x_lin = np.r_[np.min(x):np.max(x):dx]
        y_lin = np.interp(x_lin, x, y)
        x0 = np.mean(x_lin)
        y_handle = handle(x_lin-x0)

        temp = np.convolve(y_lin, y_handle/y_handle.sum(), mode="same")

        y_conv = np.interp(x, x_lin, temp)
        # finaly remove NaN entries due to the interpolation
        y_conv[np.isnan(y_conv)] = 0
        return y_conv

    @property
    def cache_dir(self):
        """str: path to cached data"""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir):
        """set.cache_dir"""
        import os.path as path
        if path.exists(cache_dir):
            self._cache_dir = cache_dir
        else:
            print('Cache dir does not exist.\nPlease create the path first.')
