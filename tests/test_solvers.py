#!/usr/bin/env python

import numpy as np

from udkm1Dsim.simulations.solvers.heat_scipy import odefunc

# tests


def test_ode_func():
    t = np.r_[0:1]
    u = np.r_[0:2]
    N = 2
    K = 1
    d_x_grid = np.r_[0.5, 0.5]
    thermal_conds = [[lambda T: 1], [lambda T: 1]]
    therm_cond_takes_vector = [[False], [False]]
    heat_capacities = [[lambda T: 1], [lambda T: 1]]
    sub_system_coupling = [[lambda T: 1], [lambda T: 1]]
    densities = np.r_[1:3]
    groups = [(0, np.r_[0]), (1, np.r_[1])]
    dAdz = np.r_[0:2]
    fluence = np.r_[1]
    delay_pump = np.r_[0]
    pulse_length = np.r_[1]
    bc_top_type = "isolator"
    bc_top_value = np.r_[1]
    bc_bottom_type = "isolator"
    bc_bottom_value = np.r_[1]
    pbar = None
    state = (0, 1)

    odefunc(
        t,
        u,
        N,
        K,
        d_x_grid,
        thermal_conds,
        therm_cond_takes_vector,
        heat_capacities,
        sub_system_coupling,
        densities,
        groups,
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
    )
