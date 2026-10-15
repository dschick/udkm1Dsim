#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np
import pytest

from udkm1Dsim.simulations.solvers import heat_scipy
from udkm1Dsim.simulations.solvers.heat_scipy import jacobian_sparsity, odefunc

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


def test_ode_func_groups():
    # two layers with different T-dependent properties, interleaved on the grid
    N = 6
    K = 2
    d_x_grid = np.full(N - 1, 1e-9)
    thermal_conds = [[lambda T: T[0] / T[1], lambda T: 2.0], [lambda T: 3.0, lambda T: 0.1 * T]]
    therm_cond_takes_vector = [[True, False], [False, False]]
    heat_capacities = [[lambda T: 0.1 * T, lambda T: 500], [lambda T: 0.2 * T, lambda T: 700]]
    sub_system_coupling = [
        [lambda T: -1e17 * (T[0] - T[1]), lambda T: 1e17 * (T[0] - T[1])],
        [lambda T: 0, lambda T: 0],
    ]
    densities = np.r_[5000, 5000, 3000, 3000, 5000, 3000]
    indices = np.r_[0, 0, 1, 1, 0, 1]
    groups = [(0, np.r_[0, 1, 4]), (1, np.r_[2, 3, 5])]
    u = np.r_[np.linspace(300, 800, N), np.linspace(300, 400, N)]
    args = (np.zeros(N), np.r_[0], np.r_[0], np.r_[1], 0, 0, 0, 0, None, None)

    dudt = odefunc(
        0,
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
        *args,
    )

    # reference: evaluate the material parameters per grid point
    T = u.reshape([N, K], order="F")
    ks = np.zeros([N, K])
    cs = np.zeros([N, K])
    source = np.zeros([N, K])
    for ii in range(N):
        idx = indices[ii]
        for iii in range(K):
            if therm_cond_takes_vector[idx][iii]:
                ks[ii, iii] = thermal_conds[idx][iii](T[ii, :])
            else:
                ks[ii, iii] = thermal_conds[idx][iii](T[ii, iii])
            cs[ii, iii] = heat_capacities[idx][iii](T[ii, iii])
            source[ii, iii] = sub_system_coupling[idx][iii](T[ii, :])
    dx = d_x_grid[0]
    ref = np.zeros([N, K])
    ref[0] = (ks[0] * (T[1] - T[0]) / dx**2 + source[0]) / cs[0] / densities[0]
    ref[-1] = (ks[-1] * (T[-2] - T[-1]) / dx**2 + source[-1]) / cs[-1] / densities[-1]
    for ii in range(1, N - 1):
        ref[ii] = (
            (
                (ks[ii + 1] * (T[ii + 1] - T[ii]) - ks[ii] * (T[ii] - T[ii - 1])) / dx**2
                + source[ii]
            )
            / cs[ii]
            / densities[ii]
        )

    np.testing.assert_allclose(dudt, ref.reshape(N * K, order="F"), rtol=1e-12)


def test_jacobian_sparsity():
    N = 5
    K = 3
    pattern = jacobian_sparsity(N, K).toarray() != 0
    assert pattern.shape == (N * K, N * K)
    for row in range(N * K):
        for col in range(N * K):
            # unknowns are ordered grid point first per sub-system
            assert pattern[row, col] == (abs(row % N - col % N) <= 1)
    assert jacobian_sparsity(1, 1).shape == (1, 1)


def test_solve_max_step_and_jac_sparsity(monkeypatch):
    captured = {}

    def fake_solve_ivp(fun, t_span, y0, args, t_eval, **options):
        captured.update(options)
        return SimpleNamespace(y=np.tile(y0, (len(t_eval), 1)).T)

    monkeypatch.setattr(heat_scipy, "solve_ivp", fake_solve_ivp)

    N = 4
    K = 2
    lambdas = [[lambda T: 1.0, lambda T: 1.0]]
    bc = {"top_type": 0, "top_value": 0, "bottom_type": 0, "bottom_value": 0}

    def run(ode_options, fluence, pulse_width):
        captured.clear()
        heat_scipy.solve(
            300 * np.ones([N, K]),
            np.linspace(0, 3e-9, N),
            np.r_[0:1e-12:11j],
            np.r_[0, 0.5e-12],
            pulse_width,
            fluence,
            lambdas,
            [[False, False]],
            lambdas,
            lambdas,
            np.r_[1000.0],
            np.zeros(N, dtype=int),
            np.zeros(N),
            bc,
            ode_options,
        )
        return captured

    # shortest pulse width with non-zero fluence limits the step
    opts = run({"method": "Radau"}, np.r_[10, 0], np.r_[100e-15, 10e-15])
    assert opts["max_step"] == pytest.approx(10e-15)
    assert opts["jac_sparsity"].shape == (N * K, N * K)
    # a smaller max_step of the user is kept
    opts = run({"method": "Radau", "max_step": 1e-15}, np.r_[10, 10], np.r_[100e-15, 50e-15])
    assert opts["max_step"] == 1e-15
    # no excitation keeps the options, explicit solvers get no sparsity pattern
    opts = run({"method": "RK45", "max_step": np.inf}, np.r_[0, 0], np.r_[100e-15, 50e-15])
    assert opts["max_step"] == np.inf
    assert "jac_sparsity" not in opts
    # a user-defined jacobian is not overwritten
    opts = run({"method": "BDF", "jac_sparsity": "user"}, np.r_[0, 0], np.r_[0, 0])
    assert opts["jac_sparsity"] == "user"
