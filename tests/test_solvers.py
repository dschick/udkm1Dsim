#!/usr/bin/env python

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.constants as constants
from scipy.optimize import fsolve

from udkm1Dsim.simulations.solvers import (
    heat_numba,
    heat_scipy,
    magnetization_scipy,
    mean_field,
    phonon_scipy,
)
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


def test_phonon_solve_max_step(monkeypatch):
    captured = {}

    def fake_solve_ivp(fun, t_span, y0, args, t_eval, **options):
        captured.update(options)
        return SimpleNamespace(y=np.tile(y0, (len(t_eval), 1)).T)

    monkeypatch.setattr(phonon_scipy, "solve_ivp", fake_solve_ivp)

    L = 4
    masses = np.full(L, 1e-25)
    spring_consts = np.full(L, 10.0)
    dt_stab = phonon_scipy.stability_time_step(spring_consts, masses)
    # highest eigenfrequency of the chain is bounded by sqrt(4 k / m)
    assert dt_stab == pytest.approx(1 / np.sqrt(4 * 10.0 / 1e-25))

    def run(ode_options):
        captured.clear()
        phonon_scipy.solve(
            np.zeros(2 * L),
            np.r_[0:1e-12:11j],
            np.zeros((L, 11)),
            np.zeros(L),
            spring_consts,
            masses,
            ode_options,
        )
        return captured

    # explicit solvers are limited to the stability time step
    assert run({"method": "RK23", "max_step": np.inf})["max_step"] == dt_stab
    assert run({"method": "RK45", "max_step": None})["max_step"] == dt_stab
    # a smaller max_step of the user is kept
    assert run({"method": "RK45", "max_step": dt_stab / 2})["max_step"] == dt_stab / 2
    # implicit solvers are not limited
    assert run({"method": "Radau", "max_step": np.inf})["max_step"] == np.inf


def test_heat_numba_solve_not_implemented():
    # same interface as the scipy solver, which the Heat class calls
    scipy_params = list(inspect.signature(heat_scipy.solve).parameters)
    numba_params = list(inspect.signature(heat_numba.solve).parameters)
    assert numba_params[: len(scipy_params)] == scipy_params

    with pytest.raises(NotImplementedError):
        heat_numba.solve(*(None,) * 15)


@pytest.mark.parametrize("eff_spin", [0.5, 1.0, 3.5])
def test_mean_field_magnetization(eff_spin):
    """the vectorized solver agrees with a root finder and has the right
    limits"""
    reduced_temps = np.r_[0.0, 0.1, 0.5, 0.9, 0.999, 1.0, 1.5]
    mf = mean_field.mean_field_magnetization(reduced_temps, eff_spin)
    assert mf[0] == 1.0
    assert np.all(np.diff(mf[:-1]) < 0)
    assert np.allclose(mf[5:], 0)
    a = 3 * eff_spin / (eff_spin + 1)
    for t, m in zip(reduced_temps[1:5], mf[1:5]):
        root = fsolve(lambda x: x - mean_field.brillouin(a * x / t, eff_spin), np.sqrt(1 - t))
        assert np.isclose(m, root.item(), atol=1e-8)


def test_mean_field_coupled():
    """the coupled solver reduces to the single one without coupling, solves
    the self-consistent equations, and a GdFeCo-like ferrimagnet has a
    compensation temperature below its ordering temperature"""
    # decoupled sublattices are two independent ferromagnets
    J = np.diag([4e-20, 1e-20])
    spins = np.r_[1.5, 3.5]
    temps = np.r_[100.0, 500.0, 900.0]
    coupled = mean_field.mean_field_magnetization_coupled(temps, J, spins)
    for a in range(2):
        tc = mean_field.ordering_temperature(J[a : a + 1, a : a + 1], spins[a : a + 1])
        single = mean_field.mean_field_magnetization(temps / tc, spins[a])
        assert np.allclose(coupled[:, a], single, atol=1e-8)
    assert np.allclose(mean_field.sublattice_signs(J, spins), [1, 1])

    # GdFeCo-like ferrimagnet (Atxitia et al. 2012, x = 0.25, z = 12)
    J = np.array([[4.05e-20, -3.27e-21], [-9.81e-21, 3.78e-21]])
    tc = mean_field.ordering_temperature(J, spins)
    factors = (spins + 1) / (3 * spins)
    assert np.isclose(tc, np.max(np.linalg.eigvals(factors[:, None] * J)).real / constants.k)
    assert np.allclose(mean_field.sublattice_signs(J, spins), [1, -1])
    temps = np.linspace(1, tc * 1.1, 400)
    m = mean_field.mean_field_magnetization_coupled(temps, J, spins)
    below = temps < tc
    assert np.all(m[below, 0] > 0) and np.all(m[below, 1] < 0)
    assert np.allclose(m[~below], 0)
    # self-consistency
    beta = 1 / (constants.k * temps[below])[:, None]
    residual = m[below] - mean_field.brillouin(beta * (m[below] @ J.T), spins)
    assert np.max(np.abs(residual)) < 1e-8
    # the rare-earth sublattice (large moment) dominates at low temperature
    # and decays faster, so the net magnetization changes sign
    net = 1.0 * m[below, 0] + 2.0 * m[below, 1]
    assert net[0] < 0 and net[-1] > 0


def test_langevin_is_classical_limit():
    x = np.r_[1e-6, 1e-4, 0.1, 1.0, 10.0]
    assert np.allclose(mean_field.langevin(x), mean_field.brillouin(x, 1e6), atol=1e-5)
    assert np.allclose(mean_field.dlangevin_dx(x), mean_field.dbrillouin_dx(x, 1e6), atol=1e-5)
    assert np.allclose(mean_field.brillouin(x, np.inf), mean_field.langevin(x))
    assert np.allclose(mean_field.dbrillouin_dx(x, np.inf), mean_field.dlangevin_dx(x))
    assert np.all(np.isfinite(mean_field.langevin(np.r_[0.0, 1e-12, 500.0])))
    # classical ordering temperature J/(3 k_B) and the Langevin mean field
    J = 3 * constants.k * 500.0
    assert np.isclose(mean_field.ordering_temperature([[J]], [np.inf]), 500.0)
    m = mean_field.mean_field_magnetization_coupled(np.r_[250.0], [[J]], [np.inf])[0, 0]
    assert np.isclose(m, mean_field.langevin(J * m / (constants.k * 250.0)), atol=1e-8)


def test_equilibrium_tables():
    """the equilibrium tables reproduce the Brillouin mean field of the
    layer's spin, their classical limit S -> inf is the Langevin one with
    Garanin's damping parameters and the quantum damping parameters of Nieves
    et al. have the right limits"""
    lam = np.r_[0.02]
    mu = np.r_[0.5 * 9.274e-24]
    grid = np.linspace(1, 1300, 1300)
    Tc = 650.0
    for S in (0.5, 4.5, 1e4, np.inf):
        J = np.array([[(3.0 if np.isinf(S) else 3 * S / (S + 1)) * constants.k * Tc]])
        tables = mean_field.equilibrium_tables(grid, J, [S], mu, lam, np.ones(1))
        me, a_par, a_perp, chi = tables[0]
        assert np.allclose(me, mean_field.mean_field_magnetization(grid / Tc, S), atol=1e-5)
        k_tc = np.searchsorted(grid, Tc)
        # T -> 0: no longitudinal damping; both parameters meet 2 lambda/3 at T_C
        assert a_par[0] < 1e-4
        assert np.isclose(a_par[k_tc - 1], a_perp[k_tc - 1], atol=1e-3)
        assert np.allclose(a_par[k_tc:], 2 * lam * grid[k_tc:] / (3 * Tc))
        assert np.allclose(a_perp[k_tc:], 2 * lam * grid[k_tc:] / (3 * Tc))
        # the susceptibility is positive, peaks at T_C (where it is capped)
        # and decays above it
        assert np.all(chi > 0)
        assert np.argmax(chi) in (k_tc - 1, k_tc)
        assert chi[k_tc] == mean_field.CHI_MAX
        assert chi[-1] < chi[k_tc + 100]
        below = grid < Tc
        zeta = J[0, 0] * me / (constants.k * grid)
        dB = mean_field.dbrillouin_dx(zeta, S)
        with np.errstate(divide="ignore"):
            reference = mu[0] * dB / (constants.k * grid - J[0, 0] * dB)
        assert np.allclose(chi[below], reference[below])
        if np.isinf(S):
            # Garanin's classical LLB
            assert np.allclose(a_par[below], 2 * lam * grid[below] / (3 * Tc))
            assert np.allclose(a_perp[below], lam * (1 - grid[below] / (3 * Tc)))
            classical = tables
        elif S == 1e4:
            large_spin = tables
        elif S == 0.5:
            # the quantum transverse damping vanishes at T -> 0
            assert a_perp[0] < 1e-4
    # S = 1e4 is the classical limit for T >> T_C/S (Nieves et al. 2014):
    # magnetization and damping above 50 K, the susceptibility also away from
    # its divergence at T_C
    warm = grid > 50
    assert np.allclose(large_spin[:, 0:3, warm], classical[:, 0:3, warm], rtol=1e-3, atol=1e-5)
    away = warm & (np.abs(grid - Tc) > 10)
    assert np.allclose(large_spin[:, 3, away], classical[:, 3, away], rtol=2e-2)

    # two coupled sublattices: the classical tables of the GdFeCo set of
    # Raposo et al. 2022 (T_C = 603 K, m_e(300 K) = 0.801/0.623)
    x = np.r_[0.75, 0.25]
    J = 12 * np.array([[x[0] * 2.59e-21, x[1] * -1.13e-21], [x[0] * -1.13e-21, x[1] * 1.35e-21]])
    mus = np.r_[1.92, 7.63] * 9.274e-24
    tables = mean_field.equilibrium_tables(
        grid, J, [np.inf, np.inf], mus, [0.02, 0.02], np.ones(2)
    )
    assert abs(mean_field.ordering_temperature(J, [np.inf, np.inf]) - 603) < 2
    k300 = np.argmin(np.abs(grid - 300))
    assert np.allclose(tables[:, 0, k300], [0.801, 0.623], atol=2e-3)
    # padding
    padded = mean_field.equilibrium_tables(
        grid, J, [np.inf, np.inf], mus, [0.02, 0.02], np.r_[1.0, 0.0]
    )
    assert np.allclose(padded[1, 0:3], 0) and np.allclose(padded[1, 3], 1)
    tc = mean_field.ordering_temperature(J[:1, :1], [np.inf])
    assert np.allclose(
        padded[0, 0], mean_field.mean_field_magnetization(grid / tc, np.inf), atol=1e-6
    )

    # the grid of the tables contains the ordering temperatures and is dense
    # around them
    tgrid = mean_field.temperature_grid(1500.0, [630.0, 0.0, 1024.0])
    assert np.all(np.diff(tgrid) > 0)
    assert 630.0 in tgrid and 1024.0 in tgrid
    assert tgrid[0] == 1.0 and tgrid[-1] <= 1500.0
    near = tgrid[np.abs(tgrid - 630.0) < 1.0]
    assert len(near) > 20


def test_brillouin_limits():
    x = np.r_[-1e-9, 0.0, 1e-9, 1e-3, 1.0, 50.0]
    for S in (0.5, 2.0):
        b = mean_field.brillouin(x, S)
        assert np.all(np.isfinite(b))
        assert np.isclose(b[1], 0.0)
        assert np.isclose(b[-1], 1.0)
        assert np.isclose(b[3], (S + 1) / (3 * S) * 1e-3, rtol=1e-3)
        dbdx = mean_field.dbrillouin_dx(x, S)
        assert np.all(np.isfinite(dbdx))
        assert np.isclose(dbdx[1], (S + 1) / (3 * S))
        assert np.isclose(dbdx[-1], 0.0)


@pytest.mark.parametrize("eff_spin", [0.5, 2.0, np.inf])
def test_cubic_coefficient_from_brillouin_expansion(eff_spin):
    # B_S(x) = a x - b x^3 + ... yields c_S = b / a^3
    x = 1e-2
    b_func = (
        mean_field.langevin if np.isinf(eff_spin) else lambda x: mean_field.brillouin(x, eff_spin)
    )
    a = 1 / 3 if np.isinf(eff_spin) else (eff_spin + 1) / (3 * eff_spin)
    b = (a * x - b_func(x)) / x**3
    c = magnetization_scipy._cubic_coefficients(np.array([eff_spin]))[0]
    assert np.isclose(c, b / a**3, rtol=1e-3)


@pytest.mark.parametrize("eff_spin", [0.5, np.inf])
def test_longitudinal_field_above_curie_temp(eff_spin):
    T_C, T, mu = 600.0, 700.0, 1e-23
    factor = 3.0 if np.isinf(eff_spin) else 3 * eff_spin / (eff_spin + 1)
    J = factor * constants.k * T_C
    chi = mu * T_C / (J * (T - T_C))
    mag = SimpleNamespace(
        eff_spin=np.array([[eff_spin]]),
        mf_exch_coupling=np.array([[[J]]]),
        mag_moment=np.array([[mu]]),
        active=np.ones((1, 1)),
    )
    m = np.array([[[0.0, 0.0, 0.3]]])
    m_norm = np.array([[0.3]])
    B = magnetization_scipy.longitudinal_field(m, m_norm, np.zeros((1, 1)), np.array([[chi]]), mag)
    c = magnetization_scipy._cubic_coefficients(np.array([eff_spin]))[0]
    expected = -1 / chi * (1 + c * T_C / (T - T_C) * 0.3**2) * m
    assert np.allclose(B, expected, rtol=1e-12)
