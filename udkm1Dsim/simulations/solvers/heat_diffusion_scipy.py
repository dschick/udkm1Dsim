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

__all__ = ["HeatDiffusionScipy"]

__docformat__ = "restructuredtext"


import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from ...helpers import multi_gauss


class HeatDiffusionScipy:
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
        if progress_bar:  # with tqdm progressbar
            pbar = tqdm()
            pbar.set_description(f"Delay = {delays[0] * 1e12:.3f} ps")
            state = [delays[0], abs(delays[-1] - delays[0]) / 100]
        else:  # without progressbar
            pbar = None
            state = None

        sol = solve_ivp(
            HeatDiffusionScipy.odefunc,
            [delays[0], delays[-1]],
            np.reshape(init_temp, K * N, order="F"),
            args=(
                N,
                K,
                d_distances,
                d_start,
                therm_conds,
                heat_capacities,
                sub_system_couplings,
                densities[indices],
                indices,
                dAdz,
                fluence,
                delay_pump,
                pulse_width,
                bc_top_type,
                bc_top_value,
                bc_bottom_type,
                bc_bottom_value,
                pbar,
                state,
            ),
            t_eval=delays,
            **ode_options,
        )
        if pbar is not None:  # close tqdm progressbar if used
            pbar.close()

        return sol.y.T

    @staticmethod
    def odefunc(
        t,
        u,
        N,
        K,
        d_x_grid,
        x,
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
            t (ndarray[float]): internal time steps of the ode solver.
            u (ndarray[float]): internal variable of the ode solver.
            N (int): number of spatial grid points.
            K (int): number of sub-systems.
            d_x_grid (ndarray[float]): derivative of spatial grid.
            x (ndarray[float]): start point of actual layers.
            thermal_conds (ndarray[@lambda]): T-dependent thermal conductivity
                function handles.
            heat_capacities (ndarray[@lambda]): T-dependent heat capacity
                function handles.
            sub_system_coupling (ndarray[@lambda]): T-dependent sub-system
                coupling.
            densities (ndarray[float]): density of layers.
            indices (ndarray[int]): indices of actual layers in respect to
                interpolated spatial grid.
            dAdz (ndarray[float]): differential absorption profile.
            fluence (ndarray[float]): excitation fluences.
            delay_pump (ndarray[float]): delay of excitations.
            pulse_length (ndarray[float]): pulse widths of excitations.
            bc_top_type (int): top boundary type.
            bc_top_value (ndarray[float]): top boundary value.
            bc_bottom_type (int): bottom boundary type.
            bc_bottom_value (ndarray[float]): bottom boundary value.
            pbar (tqdm): tqdm progressbar.
            state (list[float]): state variables for progress bar.

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
                (ks[0, :] * (u[1, :] - u[0, :]) / d_x_grid[0] ** 2 + source[0, :])
                / cs[0, :]
                / rhos[0]
            )

        if bc_bottom_type == 1:  # temperature
            u[-1, :] = bc_bottom_value
        elif bc_bottom_type == 2:  # flux
            dudt[-1, :] = (
                (
                    (bc_bottom_value - ks[-1, :] * (u[-1, :] - u[-2, :]) / d_x_grid[-1])
                    / d_x_grid[-1]
                    + source[-1, :]
                )
                / cs[-1, :]
                / rhos[-1]
            )
        else:  # isolator
            dudt[-1, :] = (
                (ks[-1, :] * (u[-1, :] - u[-2, :]) / d_x_grid[-1] ** 2 + source[-1, :])
                / cs[-1, :]
                / rhos[-1]
            )

        # calculate derivative
        for ii in range(1, N - 1):
            dudt[ii, :] = (
                (
                    (
                        ks[ii + 1, :] * (u[ii + 1, :] - u[ii, :]) / (d_x_grid[ii])
                        - ks[ii, :] * (u[ii, :] - u[ii - 1, :]) / (d_x_grid[ii - 1])
                    )
                    / ((d_x_grid[ii] + d_x_grid[ii - 1]) / 2)
                    + source[ii, :]
                )
                / cs[ii, :]
                / rhos[ii]
            )

        return np.reshape(dudt, K * N, order="F")
