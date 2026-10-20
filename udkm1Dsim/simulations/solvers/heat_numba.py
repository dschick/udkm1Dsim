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


def solve(
    init_temp,
    distances,
    delays,
    delay_pump,
    pulse_width,
    fluence,
    therm_conds,
    therm_cond_takes_vector,
    heat_capacities,
    sub_system_couplings,
    densities,
    indices,
    dAdz,
    boundary_conditions,
    ode_options,
    progress_bar=False,
    backend_options=None,
):
    r"""solve

    Solves the 1D heat diffusion equation for :math:`K` sub-systems on the
    given spatial grid with a ``numba``-compiled Crank-Nicolson stepper.

    Between two consecutive requested delays the solution is advanced with a
    fixed time step, sub-stepped so that no step exceeds ``max_step`` of
    ``backend_options``. If ``max_step`` is ``None`` it is a tenth of the
    shorter of the sub-system coupling time, see :func:`coupling_time_step`,
    and the shortest pump pulse width. The scheme is second order in time, so
    halving ``max_step`` reduces the time integration error by four.
    ``ode_options`` is not used by this kernel, which does its own time
    integration.

    Args:
        init_temp (ndarray[float]): initial spatial temperature profile on the
            spatial grid :math:`N \times K` [K].
        distances (ndarray[float]): spatial grid for calculation [m].
        delays (ndarray[float]): delays range of simulation [s].
        delay_pump (ndarray[float]): delays of the excitations [s].
        pulse_width (ndarray[float]): pulse widths of the excitations [s].
        fluence (ndarray[float]): excitation fluences [J/m²].
        therm_conds (list[list[@lambda]]): T-dependent thermal conductivity
            functions per layer and sub-system.
        therm_cond_takes_vector (list[list[boolean]]): true if the according
            thermal conductivity depends on the temperatures of several
            sub-systems.
        heat_capacities (list[list[@lambda]]): T-dependent heat capacity
            functions per layer and sub-system.
        sub_system_couplings (list[list[@lambda]]): T-dependent sub-system
            coupling functions per layer and sub-system.
        densities (ndarray[float]): density of the layers on the spatial grid
            [kg/m³].
        indices (ndarray[int]): indices of actual layers in respect to
            interpolated spatial grid.
        dAdz (ndarray[float]): differential absorption profile.
        boundary_conditions (dict{str, int, ndarray[float]}): boundary
            conditions of the top and bottom boundary.
        ode_options (dict): options of the scipy solve_ivp ode solver, unused
            by this kernel.
        progress_bar (boolean, optional): enable tqdm progress bar. Defaults
            to False.
        backend_options (dict, optional): options of the kernel - ``theta``,
            ``n_picard``, ``max_step`` and ``face_conductivity``, which is
            either ``harmonic`` or ``node``, the latter reproducing the
            spatial stencil of the ``scipy`` backend. Defaults to None.

    Returns:
        temp_map (ndarray[float]): spatio-temporal temperature map
        :math:`M \times N \times K` [K].

    """
    raise NotImplementedError("heat diffusion by Numba is not implemented, yet")
