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

__all__ = ["odefunc", "solve"]

__docformat__ = "restructuredtext"


import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from ...helpers import multi_gauss


def solve(init_temp, distances, delays, delay_pump, pulse_width, fluence,
          therm_conds, heat_capacities, sub_system_couplings,
          densities, indices, dAdz, boundary_conditions, ode_options,
          progress_bar=False):
    """solve

    Solves the 1D heat diffusion equation for all sub-systems on the given
    spatial grid and delays using scipy's ``solve_ivp`` with
    :meth:`odefunc` as right-hand side.

    Args:
        init_temp (ndarray[float]): initial temperature profile of shape
            (N, K) for N spatial grid points and K sub-systems [K].
        distances (ndarray[float]): spatial grid for calculation [m].
        delays (ndarray[float]): delays range of simulation [s].
        delay_pump (ndarray[float]): delays of the excitations [s].
        pulse_width (ndarray[float]): pulse widths of the excitations [s].
        fluence (ndarray[float]): excitation fluences [J/m²].
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
        boundary_conditions (dict): boundary conditions with keys
            ``top_type``, ``top_value``, ``bottom_type``, and
            ``bottom_value``.
        ode_options (dict): options passed to scipy's ``solve_ivp``.
        progress_bar (bool, optional): enable tqdm progress bar. Defaults
            to False.

    Returns:
        temp_map (ndarray[float]): spatio-temporal temperature map of
        shape (M, N, K) for M delays [K].

    """
    M = len(delays)
    N = np.size(init_temp, 0)
    K = np.size(init_temp, 1)
    d_distances = np.diff(distances)

    if progress_bar:  # with tqdm progressbar
        pbar = tqdm()
        pbar.set_description(f"Delay = {delays[0] * 1e12:.3f} ps")
        state = [delays[0], abs(delays[-1] - delays[0]) / 100]
    else:  # without progressbar
        pbar = None
        state = None

    sol = solve_ivp(
        odefunc,
        [delays[0], delays[-1]],
        np.reshape(init_temp, K * N, order="F"),
        args=(
            N,
            K,
            d_distances,
            therm_conds,
            heat_capacities,
            sub_system_couplings,
            densities[indices],
            indices,
            dAdz,
            fluence,
            delay_pump,
            pulse_width,
            boundary_conditions['top_type'],
            boundary_conditions['top_value'],
            boundary_conditions['bottom_type'],
            boundary_conditions['bottom_value'],
            pbar,
            state,
        ),
        t_eval=delays,
        **ode_options,
    )
    if pbar is not None:  # close tqdm progressbar if used
        pbar.close()

    return np.array(sol.y.T).reshape([M, N, K], order='F')


def odefunc(
    t,
    u,
    N,
    K,
    d_x_grid,
    thermal_conds,
    heat_capacities,
    sub_system_coupling,
    densities,
    indices,
    dAdz,
    fluence,
    delay_pump,
    pulse_length,
    bc_top_type,
    bc_top_value,
    bc_bottom_type,
    bc_bottom_value,
    pbar,
    state,
):
    """odefunc

    Ordinary differential equation that is solved for 1D heat diffusion.

    Args:
        t (float): internal time step of the ode solver [s].
        u (ndarray[float]): internal variable of the ode solver.
        N (int): number of spatial grid points.
        K (int): number of sub-systems.
        d_x_grid (ndarray[float]): derivative of spatial grid [m].
        thermal_conds (list[list[@lambda]]): T-dependent thermal
            conductivity function handles of all layers and sub-systems.
        heat_capacities (list[list[@lambda]]): T-dependent heat capacity
            function handles of all layers and sub-systems.
        sub_system_coupling (list[list[@lambda]]): T-dependent sub-system
            coupling function handles of all layers and sub-systems.
        densities (ndarray[float]): densities at each spatial grid point
            [kg/m³].
        indices (ndarray[int]): indices of actual layers in respect to
            interpolated spatial grid.
        dAdz (ndarray[float]): differential absorption profile [1/m].
        fluence (ndarray[float]): excitation fluences [J/m²].
        delay_pump (ndarray[float]): delays of the excitations [s].
        pulse_length (ndarray[float]): pulse widths of the excitations [s].
        bc_top_type (int): top boundary type.
        bc_top_value (ndarray[float]): top boundary value.
        bc_bottom_type (int): bottom boundary type.
        bc_bottom_value (ndarray[float]): bottom boundary value.
        pbar (tqdm, None): tqdm progressbar or None if disabled.
        state (list[float], None): last updated delay and update interval
            of the progress bar or None if disabled.

    Returns:
        dudt (ndarray[float]): temporal derivative of internal variable.

    """
    # state is a list containing last updated time t:
    # state = [last_t, dt]
    # I used a list because its values can be carried between function
    # calls throughout the ODE integration
    if pbar is not None:
        # set everything for the tqdm progressbar
        last_t, dt = state
        try:
            n = int((float(np.asarray(t).item()) - last_t) / dt)
        except ValueError:
            n = 0

        if n >= 1:
            pbar.update(n)
            pbar.set_description(f"Delay = {t * 1e12:.3f} ps")
            state[0] = t
        elif n < 0:
            state[0] = t

    # reshape input temperature
    u = np.array(u).reshape([N, K], order="F")
    # initialize arrays
    dudt = np.zeros([N, K])
    ks = np.zeros([N, K])
    cs = np.zeros([N, K])
    rhos = densities

    # calculate external source
    source = np.zeros([N, K])
    if np.any(fluence):
        source[:, 0] = dAdz * multi_gauss(t, s=pulse_length, x0=delay_pump, A=fluence)

    # calculate temperature-dependent parameters
    for ii in range(N):
        idx = indices[ii]
        for iii in range(K):
            try:
                # temperature argument should be scalar
                ks[ii, iii] = thermal_conds[idx][iii](u[ii, iii])
            except (IndexError, TypeError):
                # temperature argument should be a vector
                ks[ii, iii] = thermal_conds[idx][iii](u[ii, :])

            cs[ii, iii] = heat_capacities[idx][iii](u[ii, iii])
            source[ii, iii] = source[ii, iii] + sub_system_coupling[idx][iii](u[ii, :])

    # boundary conditions
    if bc_top_type == 1:  # temperature
        u[0, :] = bc_top_value
    elif bc_top_type == 2:  # flux
        dudt[0, :] = (
            (
                (ks[0, :] * (u[1, :] - u[0, :]) / d_x_grid[0] + bc_top_value) / d_x_grid[0]
                + source[0, :]
            )
            / cs[0, :]
            / rhos[0]
        )
    else:  # isolator
        dudt[0, :] = (
            (ks[0, :] * (u[1, :] - u[0, :]) / d_x_grid[0] ** 2 + source[0, :]) / cs[0, :] / rhos[0]
        )

    if bc_bottom_type == 1:  # temperature
        u[-1, :] = bc_bottom_value
    elif bc_bottom_type == 2:  # flux
        dudt[-1, :] = (
            (
                (bc_bottom_value - ks[-1, :] * (u[-1, :] - u[-2, :]) / d_x_grid[-1]) / d_x_grid[-1]
                + source[-1, :]
            )
            / cs[-1, :]
            / rhos[-1]
        )
    else:  # isolator
        dudt[-1, :] = (
            (ks[-1, :] * (u[-2, :] - u[-1, :]) / d_x_grid[-1] ** 2 + source[-1, :])
            / cs[-1, :]
            / rhos[-1]
        )

    # calculate derivative of the interior nodes
    d_lower = d_x_grid[:-1, np.newaxis]
    d_upper = d_x_grid[1:, np.newaxis]
    dudt[1:-1, :] = (
        (
            (
                ks[2:, :] * (u[2:, :] - u[1:-1, :]) / d_upper
                - ks[1:-1, :] * (u[1:-1, :] - u[:-2, :]) / d_lower
            )
            / ((d_upper + d_lower) / 2)
            + source[1:-1, :]
        )
        / cs[1:-1, :]
        / rhos[1:-1, np.newaxis]
    )

    return np.reshape(dudt, K * N, order="F")
