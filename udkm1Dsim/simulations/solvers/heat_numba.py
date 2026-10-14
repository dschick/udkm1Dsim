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

__all__ = ["solve"]

__docformat__ = "restructuredtext"


import numpy as np


def solve(
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
    """solve

    Solves the 1D heat diffusion equation for all sub-systems on the given
    spatial grid and delays using numba-compiled functions.

    Args:
        progress_bar (bool): enable tqdm progress bar.
        delays (ndarray[float]): delays range of simulation [s].
        N (int): number of spatial grid points.
        K (int): number of sub-systems.
        init_temp (ndarray[float]): initial temperature profile of shape
            (N, K) [K].
        d_distances (ndarray[float]): derivative of spatial grid [m].
        d_start (ndarray[float]): start points of the actual layers [m].
        therm_conds (list[list[@lambda]]): T-dependent thermal conductivity
            function handles of all layers and sub-systems.
        heat_capacities (list[list[@lambda]]): T-dependent heat capacity
            function handles of all layers and sub-systems.
        sub_system_couplings (list[list[@lambda]]): T-dependent sub-system
            coupling function handles of all layers and sub-systems.
        densities (ndarray[float]): densities of all layers [kg/m³].
        indices (ndarray[int]): indices of actual layers in respect to
            interpolated spatial grid.
        dAdz (ndarray[float]): differential absorption profile [1/m].
        fluence (ndarray[float]): excitation fluences [J/m²].
        delay_pump (ndarray[float]): delays of the excitations [s].
        pulse_width (ndarray[float]): pulse widths of the excitations [s].
        bc_top_type (int): top boundary type.
        bc_top_value (ndarray[float]): top boundary value.
        bc_bottom_type (int): bottom boundary type.
        bc_bottom_value (ndarray[float]): bottom boundary value.
        ode_options (dict): options passed to the ode solver.

    Returns:
        temp_map (ndarray[float]): spatio-temporal temperature map of
        shape (M, N, K) with M delays.

    """
    M = len(delays)

    return np.zeros([M, N, K])
