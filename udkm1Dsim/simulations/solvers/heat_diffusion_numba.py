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

__all__ = ["HeatDiffusionNumba"]

__docformat__ = "restructuredtext"


import numpy as np


class HeatDiffusionNumba:
    @staticmethod
    def solve_problem(
        progress_bar,
        delays,
        N,
        K,
        init_temp,
        d_distances,
        d_start,
        therm_conds,
        heat_capacities,
        sub_system_couplings,
        densities,
        indices,
        dAdz,
        fluence,
        delay_pump,
        pulse_width,
        bc_top_type,
        bc_top_value,
        bc_bottom_type,
        bc_bottom_value,
        ode_options,
    ):
        M = len(delays)

        return np.zeros([M, N, K])

