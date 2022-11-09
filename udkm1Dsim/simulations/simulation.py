#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

__all__ = ['Simulation']

__docformat__ = 'restructuredtext'

from tabulate import tabulate
import numpy as np
import os


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

    def __init__(self, S, force_recalc, **kwargs):
        self.S = S
        self.force_recalc = force_recalc
        self.save_data = kwargs.get('save_data', True)
        self.cache_dir = kwargs.get('cache_dir', './')
        self.disp_messages = kwargs.get('disp_messages', True)
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
        """disp_message

        Wrapper to display messages for that class.

        Args:
            message (str): message to display.

        """
        if self.disp_messages:
            print(message)

    def save(self, full_filename, data, *args):
        """save

        Save data to file. The variable name can be handed as variable argument.

        Args:
            full_filename (str): full file name to data file.
            data (ndarray): actual data to save.
            *args (str, optional): variable name within the data file.

        """
        if len(args) == 1:
            var_name = args[0]
        else:
            var_name = '_data_'
        if self.save_data:
            np.savez(full_filename, **data)
            filename = os.path.basename(full_filename)
            self.disp_message('{:s} saved to file:\n\t {:s}'.format(var_name, filename))

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
        x_lin = np.r_[np.min(x):np.max(x):dx]
        y_lin = np.interp(x_lin, x, y)
        x0 = np.mean(x_lin)
        y_handle = handle(x_lin-x0)

        temp = np.convolve(y_lin, y_handle/y_handle.sum(), mode="same")

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
            print('Cache dir does not exist.\nPlease create the path first.')
