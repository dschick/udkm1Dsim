# #!/usr/bin/env python

# # The MIT License (MIT)
# # Copyright (c) 2020 Daniel Schick
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# # EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# # IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# # DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# # OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# # OR OTHER DEALINGS IN THE SOFTWARE.

# r"""Numba-accelerated Crank-Nicolson kernel for the N-temperature heat
# diffusion model.

# In contrast to the ``scipy`` backend this kernel does not only replace the
# evaluation of the right-hand side, it replaces the time integration as well.
# ``scipy``'s Radau spends almost all of its time on step control, on the
# finite-difference Jacobian and on a dense LU of :math:`N \times K` unknowns,
# which is what limits the ``scipy`` backend once the right-hand side itself is
# compiled. Here the equations

# .. math::

#     C_k(T_k)\, \frac{\partial T_k}{\partial t} = \frac{\partial}{\partial z}
#     \left( k_k(T) \frac{\partial T_k}{\partial z} \right) + S(z, t)\,\delta_{k0}
#     + G_k(T_1, ..., T_K)

# are advanced by the theta method - ``theta = 0.5`` is Crank-Nicolson,
# ``theta = 1`` backward Euler - on a fixed time step. Per step the
# temperature-dependent coefficients, including the heat capacity that
# multiplies the temperature increment, are frozen at the Picard midpoint of
# the current estimate of the new profile, which is the midpoint rule for the
# integral of :math:`C\,dT` and keeps the scheme second order in time. One
# tridiagonal system per sub-system is solved directly, so the cost per step is
# :math:`O(N K)` instead of the :math:`O((NK)^3)` of a dense LU.

# The sub-system coupling is linearised about the Picard midpoint,

# .. math:: G_k(T) \approx G_k(T^*) + \frac{\partial G_k}{\partial T_k}
#           \left( T_k - T^*_k \right),

# so that its derivative can be put on the diagonal. For a relaxing coupling
# that derivative is negative and the term therefore adds to the diagonal, which
# keeps the system diagonally dominant however stiff the coupling is - that is
# where the unconditional stability comes from. The derivative is not estimated
# numerically: it is obtained exactly by differentiating the ``sympy``
# expression of the coupling.

# The material properties are evaluated by calling the compiled expressions and
# are deliberately *not* sampled onto lookup tables: a table costs memory that
# grows with the number of sub-systems and it only reproduces the expressions it
# was built for, whereas the single ``sympy`` expression per property stays the
# one source of truth and supports whatever a user writes in a material file.
# ``numba`` does not accept a list of dispatchers as an argument of a nopython
# function, so the property evaluation of a whole sample structure is
# source-generated as a branch per distinct material and compiled together with
# the stepper into one function, which is cached per structure.

# The spatial discretization is a node-centred finite-volume scheme whose face
# conductivity is the width-weighted *harmonic* mean of the two neighbouring
# values. Two cells in series are two thermal resistances in series, so the
# harmonic mean is the correct face value, and it vanishes when either side
# does - an arithmetic mean would leak a spurious flux across an insulating
# interface. This differs from the ``scipy`` backend, which evaluates the
# conductivity at the node, so the two backends do not agree to round-off but to
# their respective discretization errors.

# """

# __all__ = ["solve", "build_stepper", "coupling_time_step"]

# __docformat__ = "restructuredtext"

# import numpy as np
# from sympy import IndexedBase, Integer, Symbol, diff, lambdify, symarray, symbols
# from sympy.printing.numpy import NumPyPrinter
# from tqdm.auto import tqdm

# from . import HAS_NUMBA

# if HAS_NUMBA:
#     from numba import njit
# else:  # pragma: no cover - the kernel is unusable without numba anyway

#     def njit(*args, **kwargs):
#         def wrap(func):
#             return func

#         return wrap


# # generated steppers, keyed by the source they were compiled from
# _stepper_cache = {}

# # boundary types, as integers so they pass cleanly into nopython code
# ISOLATOR, TEMPERATURE, FLUX = 0, 1, 2


# def solve(
#     progress_bar,
#     delays,
#     N,
#     K,
#     init_temp,
#     d_distances,
#     d_start,
#     therm_conds,
#     heat_capacities,
#     sub_system_couplings,
#     densities,
#     indices,
#     dAdz,
#     fluence,
#     delay_pump,
#     pulse_width,
#     bc_top_type,
#     bc_top_value,
#     bc_bottom_type,
#     bc_bottom_value,
#     ode_options,
#     backend_options=None,
# ):
#     r"""solve

#     Solves the 1D heat diffusion equation for :math:`K` sub-systems on the
#     given spatial grid with a ``numba``-compiled Crank-Nicolson stepper.

#     Between two consecutive requested delays the solution is advanced with a
#     fixed time step, sub-stepped so that no step exceeds ``max_step`` of
#     ``backend_options``. If ``max_step`` is ``None`` it is a tenth of the
#     shorter of the sub-system coupling time, see :func:`coupling_time_step`,
#     and the shortest pump pulse width. The scheme is second order in time, so
#     halving ``max_step`` reduces the time integration error by four.
#     ``ode_options`` is not used by this kernel, which does its own time
#     integration.

#     Args:
#         init_temp (ndarray[float]): initial spatial temperature profile on the
#             spatial grid :math:`N \times K` [K].
#         distances (ndarray[float]): spatial grid for calculation [m].
#         delays (ndarray[float]): delays range of simulation [s].
#         delay_pump (ndarray[float]): delays of the excitations [s].
#         pulse_width (ndarray[float]): pulse widths of the excitations [s].
#         fluence (ndarray[float]): excitation fluences [J/m²].
#         therm_conds (list[list[@lambda]]): T-dependent thermal conductivity
#             functions per layer and sub-system.
#         therm_cond_vector (list[list[boolean]]): true if the according thermal
#             conductivity depends on the temperatures of several sub-systems.
#         heat_capacities (list[list[@lambda]]): T-dependent heat capacity
#             functions per layer and sub-system.
#         sub_system_coupling (list[list[@lambda]]): T-dependent sub-system
#             coupling functions per layer and sub-system.
#         densities (ndarray[float]): density of the layers on the spatial grid
#             [kg/m³].
#         indices (ndarray[int]): indices of actual layers in respect to
#             interpolated spatial grid.
#         dAdz (ndarray[float]): differential absorption profile.
#         boundary_conditions (dict{str, int, ndarray[float]}): boundary
#             conditions of the top and bottom boundary.
#         ode_options (dict): options of the scipy solve_ivp ode solver, unused
#             by this kernel.
#         progress_bar (boolean, optional): enable tqdm progress bar. Defaults
#             to False.
#         backend_options (dict, optional): options of the kernel - ``theta``,
#             ``n_picard``, ``max_step`` and ``face_conductivity``, which is
#             either ``harmonic`` or ``node``, the latter reproducing the
#             spatial stencil of the ``scipy`` backend. Defaults to None.

#     Returns:
#         temp_map (ndarray[float]): spatio-temporal temperature map
#         :math:`M \times N \times K` [K].

#     """
#     if not HAS_NUMBA:
#         raise RuntimeError("The numba backend requires the optional numba package!")

#     options = {"theta": 0.5, "n_picard": 3, "max_step": None, "face_conductivity": "harmonic"}
#     options.update(backend_options or {})

#     delays = np.asarray(delays, dtype=float)
#     M = len(delays)
#     N = np.size(init_temp, 0)
#     K = np.size(init_temp, 1)

#     # node-centred finite volumes: hf are the node-to-node distances, dz the
#     # width each node stands for. The outermost half cells are kept at the
#     # full spacing, which is what the scipy stencil does at its boundary rows.
#     hf = np.diff(distances)
#     dz = np.empty(N)
#     dz[1:-1] = (hf[:-1] + hf[1:]) / 2
#     dz[0] = hf[0]
#     dz[-1] = hf[-1]

#     stepper, materials = build_stepper(
#         N,
#         K,
#         np.asarray(indices, dtype=np.int64),
#         therm_conds,
#         therm_cond_vector,
#         heat_capacities,
#         sub_system_coupling,
#     )

#     bc_top_value = _boundary_value(boundary_conditions["top_value"], K)
#     bc_bottom_value = _boundary_value(boundary_conditions["bottom_value"], K)

#     # T is kept as (K, N), so that one sub-system is contiguous
#     T = np.ascontiguousarray(np.asarray(init_temp, dtype=float).T)
#     temp_map = np.empty((M, K, N))
#     temp_map[0] = T

#     fluence = np.atleast_1d(np.asarray(fluence, dtype=float))
#     delay_pump = np.atleast_1d(np.asarray(delay_pump, dtype=float))
#     pulse_width = np.atleast_1d(np.asarray(pulse_width, dtype=float))
#     if fluence.size == 0:
#         fluence = np.zeros(1)
#         delay_pump = np.zeros(1)
#         pulse_width = np.ones(1)

#     max_step = options["max_step"]
#     if max_step is None:
#         # the sub-system coupling and the pump pulses set the fastest time
#         # scales of the problem, both are resolved with ten steps
#         max_step = coupling_time_step(
#             init_temp, indices, heat_capacities, sub_system_coupling, densities, K
#         )
#         widths = pulse_width[(fluence != 0) & (pulse_width > 0)]
#         if widths.size > 0:
#             max_step = min(max_step, np.min(widths) / 10)
#     if not np.isfinite(max_step):
#         max_step = None
#     pbar = tqdm(total=M - 1) if progress_bar else None

#     for m in range(1, M):
#         t0, t1 = delays[m - 1], delays[m]
#         span = t1 - t0
#         steps = 1 if max_step is None else max(1, int(np.ceil(abs(span) / max_step)))
#         stepper(
#             T,
#             t0,
#             span / steps,
#             steps,
#             dz,
#             hf,
#             densities,
#             materials,
#             dAdz,
#             fluence,
#             delay_pump,
#             pulse_width,
#             float(options["theta"]),
#             int(options["n_picard"]),
#             1 if options["face_conductivity"] == "node" else 0,
#             boundary_conditions["top_type"],
#             bc_top_value,
#             boundary_conditions["bottom_type"],
#             bc_bottom_value,
#         )
#         temp_map[m] = T
#         if pbar is not None:
#             pbar.update(1)
#             pbar.set_description(f"Delay = {t1 * 1e12:.3f} ps")

#     if pbar is not None:
#         pbar.close()

#     return np.ascontiguousarray(np.transpose(temp_map, (0, 2, 1)))


# @njit(cache=True)
# def thomas_solve(a, b, c, d, x, cs, ds):
#     """thomas_solve

#     In-place tridiagonal solve of :math:`A x = d`, where ``a`` is the
#     sub-diagonal, ``b`` the diagonal and ``c`` the super-diagonal. ``cs`` and
#     ``ds`` are scratch buffers, handed in so that a step allocates nothing.

#     Args:
#         a (ndarray[float]): sub-diagonal, :math:`n-1` elements.
#         b (ndarray[float]): diagonal, :math:`n` elements.
#         c (ndarray[float]): super-diagonal, :math:`n-1` elements.
#         d (ndarray[float]): right-hand side, :math:`n` elements.
#         x (ndarray[float]): solution, :math:`n` elements.
#         cs (ndarray[float]): scratch buffer, :math:`n` elements.
#         ds (ndarray[float]): scratch buffer, :math:`n` elements.

#     """
#     n = len(d)
#     if n == 1:
#         x[0] = d[0] / b[0]
#         return
#     cs[0] = c[0] / b[0]
#     ds[0] = d[0] / b[0]
#     for i in range(1, n - 1):
#         m = b[i] - a[i - 1] * cs[i - 1]
#         cs[i] = c[i] / m
#         ds[i] = (d[i] - a[i - 1] * ds[i - 1]) / m
#     ds[n - 1] = (d[n - 1] - a[n - 2] * ds[n - 2]) / (b[n - 1] - a[n - 2] * cs[n - 2])
#     x[n - 1] = ds[n - 1]
#     for i in range(n - 2, -1, -1):
#         x[i] = ds[i] - cs[i] * x[i + 1]


# def coupling_time_step(
#     init_temp, indices, heat_capacities, sub_system_coupling, densities, K, fraction=0.1
# ):
#     r"""coupling_time_step

#     Time step that resolves the sub-system coupling. The coupling
#     :math:`G_k(T)` relaxes the temperature of sub-system :math:`k` on the
#     time scale

#     .. math:: \tau_k = \frac{\rho\, c_k(T)}{|\partial G_k/\partial T_k|},

#     which is evaluated for all sub-systems at the initial temperature of every
#     grid point. The step is a ``fraction`` of the shortest of these times. For
#     an electron-phonon coupled metal at room temperature this is a few
#     femtoseconds, far below a typical delay spacing, and a Crank-Nicolson
#     step of the delay spacing would be off by percent. Without any coupling
#     the step is infinite, i.e. one step per delay.

#     Args:
#         init_temp (ndarray[float]): initial temperatures :math:`N \times K`.
#         indices (ndarray[int]): indices of actual layers in respect to
#             interpolated spatial grid.
#         heat_capacities (list[list[@lambda]]): compiled heat capacity
#             functions per layer and sub-system.
#         sub_system_coupling (list[list[@lambda]]): compiled sub-system
#             coupling functions per layer and sub-system.
#         densities (ndarray[float]): density of the layers on the spatial grid.
#         K (int): number of sub-systems.
#         fraction (float, optional): fraction of the shortest coupling time.
#             Defaults to 0.1.

#     Returns:
#         max_step (float): time step [s], ``inf`` without coupling.

#     """
#     init_temp = np.asarray(init_temp, dtype=float)
#     indices = np.asarray(indices, dtype=int)
#     densities = np.asarray(densities, dtype=float)
#     tau = np.inf
#     T = symbols("T")
#     for idx in np.unique(indices):
#         select = np.flatnonzero(indices == idx)
#         temps = init_temp[select, :]
#         for k in range(K):
#             coupling = sub_system_coupling[int(idx)][k]
#             if coupling.is_vector:
#                 symbol = symbols(f"T_{k:d}")
#                 derivative = lambdify(
#                     [symarray("T", coupling.num_sub_systems)],
#                     diff(coupling.expression, symbol),
#                     modules="numpy",
#                 )
#                 dG = derivative(temps.T)
#             else:
#                 derivative = lambdify(T, diff(coupling.expression, T), modules="numpy")
#                 dG = derivative(temps[:, k])
#             dG = np.abs(np.broadcast_to(np.asarray(dG, dtype=float), (len(select),)))
#             capacity = lambdify(T, heat_capacities[int(idx)][k].expression, modules="numpy")
#             C = densities[select] * np.broadcast_to(
#                 np.asarray(capacity(temps[:, k]), dtype=float), (len(select),)
#             )
#             coupled = dG > 0
#             if np.any(coupled):
#                 tau = min(tau, np.min(C[coupled] / dG[coupled]))

#     return fraction * tau


# def build_stepper(
#     N, K, indices, therm_conds, therm_cond_vector, heat_capacities, sub_system_coupling
# ):
#     """build_stepper

#     Source-generates and compiles the Crank-Nicolson stepper of one sample
#     structure. The temperature-dependent material parameters of all distinct
#     materials are written out as a branch per material, so that the compiled
#     function neither calls back into Python nor receives function objects as
#     arguments. Results are cached, since the compilation is far more expensive
#     than a single step.

#     Args:
#         N (int): number of spatial grid points.
#         K (int): number of sub-systems.
#         indices (ndarray[int]): indices of actual layers in respect to
#             interpolated spatial grid.
#         therm_conds (list[list[@lambda]]): compiled thermal conductivity
#             functions per layer and sub-system.
#         therm_cond_vector (list[list[boolean]]): true if the according thermal
#             conductivity depends on the temperatures of several sub-systems.
#         heat_capacities (list[list[@lambda]]): compiled heat capacity
#             functions per layer and sub-system.
#         sub_system_coupling (list[list[@lambda]]): compiled sub-system
#             coupling functions per layer and sub-system.

#     Returns:
#         (tuple):
#         - *stepper (@lambda)* - compiled stepper of the heat diffusion
#           equation.
#         - *materials (ndarray[int])* - index of the set of material
#           expressions of each spatial grid point.

#     """
#     # several positions in the stack share the same material, so the branches
#     # are generated per distinct set of expressions and not per layer position
#     signatures = {}
#     materials = np.empty(len(indices), dtype=np.int64)
#     layers = []
#     for i, idx in enumerate(indices):
#         idx = int(idx)
#         signature = tuple(
#             str(prop[idx][iii].expression)
#             for prop in (therm_conds, heat_capacities, sub_system_coupling)
#             for iii in range(K)
#         )
#         if signature not in signatures:
#             signatures[signature] = len(layers)
#             layers.append(idx)
#         materials[i] = signatures[signature]

#     source = _generate_source(
#         N, K, layers, therm_conds, therm_cond_vector, heat_capacities, sub_system_coupling
#     )

#     if source not in _stepper_cache:
#         # the NumPyPrinter writes functions such as exp() as numpy.exp()
#         namespace = {"np": np, "numpy": np, "thomas_solve": thomas_solve}
#         exec(compile(source, "<heat_numba_stepper>", "exec"), namespace)
#         _stepper_cache[source] = njit(cache=False, error_model="numpy")(
#             namespace["_heat_diffusion_stepper"]
#         )

#     return _stepper_cache[source], materials


# def _generate_source(
#     N, K, layers, therm_conds, therm_cond_vector, heat_capacities, sub_system_coupling
# ):
#     """_generate_source

#     Returns the source code of the Crank-Nicolson stepper of one sample
#     structure.

#     Args:
#         N (int): number of spatial grid points.
#         K (int): number of sub-systems.
#         layers (list[int]): one representative layer index per distinct set of
#             material expressions.
#         therm_conds (list[list[@lambda]]): compiled thermal conductivity
#             functions per layer and sub-system.
#         therm_cond_vector (list[list[boolean]]): true if the according thermal
#             conductivity depends on the temperatures of several sub-systems.
#         heat_capacities (list[list[@lambda]]): compiled heat capacity
#             functions per layer and sub-system.
#         sub_system_coupling (list[list[@lambda]]): compiled sub-system
#             coupling functions per layer and sub-system.

#     Returns:
#         source (str): source code of the stepper.

#     """
#     lines = []
#     for n, idx in enumerate(layers):
#         lines.append("{:s} materials[i] == {:d}:".format("if" if n == 0 else "elif", n))
#         for k in range(K):
#             capacity = _print(heat_capacities[idx][k], k, False)
#             conductivity = _print(therm_conds[idx][k], k, therm_cond_vector[idx][k])
#             coupling = _print(sub_system_coupling[idx][k], k, True)
#             derivative = _print(sub_system_coupling[idx][k], k, True, derivative=k)
#             lines.append(f"    C[{k:d}, i] = rho[i]*({capacity:s})")
#             lines.append(f"    kap[{k:d}, i] = {conductivity:s}")
#             lines.append(f"    fk[{k:d}, i] = {coupling:s}")
#             lines.append(f"    dfk[{k:d}, i] = {derivative:s}")

#     def indent(width):
#         pad = " " * width
#         return "\n".join(pad + line for line in lines)

#     return _TEMPLATE.format(N=N, K=K, properties_inner=indent(16))


# def _print(func, k, vector, derivative=None):
#     """_print

#     Returns the source code of a single material property, with the
#     temperatures of the sub-systems replaced by the according elements of the
#     temperature array. If ``derivative`` is given, the derivative of the
#     property with respect to the temperature of that sub-system is printed
#     instead of the property itself.

#     Args:
#         func (@lambda): compiled material property carrying its expression.
#         k (int): index of the sub-system the property belongs to.
#         vector (boolean): true if the property depends on the temperatures of
#             several sub-systems.
#         derivative (int, optional): sub-system to differentiate for. Defaults
#             to None.

#     Returns:
#         source (str): source code of the material property.

#     """
#     T = IndexedBase("Ts")
#     i = Symbol("i", integer=True)
#     expression = func.expression
#     if vector and func.is_vector:
#         symbol = symbols(f"T_{k:d}")
#         substitutions = {
#             symbols(f"T_{j:d}"): T[Integer(j), i] for j in range(func.num_sub_systems)
#         }
#     else:
#         symbol = symbols("T")
#         substitutions = {symbol: T[Integer(k), i]}

#     if derivative is not None:
#         expression = diff(expression, symbol)

#     return NumPyPrinter().doprint(expression.subs(substitutions))


# def _boundary_value(value, K):
#     """_boundary_value

#     Returns a boundary value as an array of :math:`K` elements, as required by
#     the compiled stepper.

#     Args:
#         value (ndarray[float]): boundary value.
#         K (int): number of sub-systems.

#     Returns:
#         value (ndarray[float]): boundary value for each sub-system.

#     """
#     value = np.atleast_1d(np.asarray(value, dtype=float))
#     if value.size != K:
#         value = np.zeros(K)

#     return value


# _TEMPLATE = '''
# def _heat_diffusion_stepper(T, t_start, dt, nsteps, dz, hf, rho, materials,
#                             dAdz, fluence, delay_pump, pulse_width,
#                             theta, n_picard, face_mode, bc_top_type, bc_top_value,
#                             bc_bottom_type, bc_bottom_value):
#     """advance the (K, N) temperature field by nsteps steps of size dt"""
#     N = {N}
#     K = {K}
#     C = np.empty((K, N))
#     kap = np.empty((K, N))
#     fk = np.empty((K, N))
#     dfk = np.empty((K, N))
#     Ts = np.empty((K, N))
#     Tnew = np.empty((K, N))
#     kface = np.empty(N-1)
#     a = np.empty(N-1)
#     b = np.empty(N)
#     c = np.empty(N-1)
#     d = np.empty(N)
#     x = np.empty(N)
#     cs = np.empty(N)
#     ds = np.empty(N)
#     src_old = np.empty(N)
#     src_new = np.empty(N)

#     for step in range(nsteps):
#         t_old = t_start + step*dt
#         t_new = t_old + dt

#         # external source of the pump pulses, area-normalized exactly as
#         # helpers.multi_gauss does it for the scipy backend
#         amp_old = 0.0
#         amp_new = 0.0
#         for p in range(len(pulse_width)):
#             if fluence[p] == 0.0:
#                 continue
#             s = pulse_width[p]/(2*np.sqrt(2*np.log(2)))
#             amplitude = fluence[p]/np.sqrt(2*np.pi*s**2)
#             amp_old += amplitude*np.exp(-((t_old-delay_pump[p])**2)/(2*s**2))
#             amp_new += amplitude*np.exp(-((t_new-delay_pump[p])**2)/(2*s**2))
#         for i in range(N):
#             src_old[i] = dAdz[i]*amp_old
#             src_new[i] = dAdz[i]*amp_new

#         for kk in range(K):
#             for i in range(N):
#                 Tnew[kk, i] = T[kk, i]

#         for picard in range(n_picard):
#             # freeze the coefficients at the Picard midpoint
#             for kk in range(K):
#                 for i in range(N):
#                     Ts[kk, i] = 0.5*(T[kk, i] + Tnew[kk, i])
#             for i in range(N):
# {properties_inner}

#             for k in range(K):
#                 # width-weighted harmonic mean of the face conductivities
#                 conducts = False
#                 for i in range(N-1):
#                     kl = kap[k, i]
#                     kr = kap[k, i+1]
#                     if kl <= 0.0 or kr <= 0.0:
#                         kface[i] = 0.0
#                     else:
#                         kface[i] = 2.0*kl*kr/(kl + kr)
#                     if kl > 0.0 or kr > 0.0:
#                         conducts = True

#                 for i in range(N):
#                     if i < N-1:
#                         if face_mode == 1:
#                             # the scipy stencil takes the conductivity of the
#                             # downstream node for a face - except in its very
#                             # first row, where it takes the node itself
#                             ke = kap[k, 0] if i == 0 else kap[k, i+1]
#                         else:
#                             ke = kface[i]
#                         ge = theta*dt*ke/(hf[i]*dz[i]*C[k, i])
#                     else:
#                         ge = 0.0
#                     if i > 0:
#                         kw = kap[k, i] if face_mode == 1 else kface[i-1]
#                         gw = theta*dt*kw/(hf[i-1]*dz[i]*C[k, i])
#                     else:
#                         gw = 0.0

#                     # implicit half: new-level fluxes and the linearised coupling
#                     b[i] = 1.0 + gw + ge - theta*dt*dfk[k, i]/C[k, i]
#                     if i < N-1:
#                         c[i] = -ge
#                     if i > 0:
#                         a[i-1] = -gw

#                     # explicit half
#                     source_new = src_new[i] if k == 0 else 0.0
#                     source_old = src_old[i] if k == 0 else 0.0
#                     if i == 0 and bc_top_type == 2:
#                         source_new += bc_top_value[k]/dz[0]
#                         source_old += bc_top_value[k]/dz[0]
#                     if i == N-1 and bc_bottom_type == 2:
#                         source_new += bc_bottom_value[k]/dz[N-1]
#                         source_old += bc_bottom_value[k]/dz[N-1]

#                     # the heat capacity of the midpoint multiplies the whole
#                     # temperature increment - the midpoint rule for the
#                     # integral of C dT, which keeps the scheme second order.
#                     # The coupling is taken at the theta point
#                     # theta*T_new + (1-theta)*T, linearised around the Picard
#                     # midpoint Ts. For theta = 0.5 the linearisation vanishes
#                     # at convergence, so antisymmetric couplings such as
#                     # G(T_0-T_1) exchange energy without creating any.
#                     cpl = fk[k, i] - dfk[k, i]*(Ts[k, i] - (1.0 - theta)*T[k, i])
#                     d[i] = T[k, i] + dt*(theta*source_new + cpl)/C[k, i]
#                     if theta < 1.0:
#                         d[i] += dt*(1.0 - theta)*source_old/C[k, i]
#                         oe = (1.0 - theta)/theta
#                         if i < N-1:
#                             d[i] += oe*ge*(T[k, i+1] - T[k, i])
#                         if i > 0:
#                             d[i] += oe*gw*(T[k, i-1] - T[k, i])

#                 if bc_top_type == 1:
#                     b[0] = 1.0
#                     c[0] = 0.0
#                     d[0] = bc_top_value[k]
#                 if bc_bottom_type == 1:
#                     b[N-1] = 1.0
#                     a[N-2] = 0.0
#                     d[N-1] = bc_bottom_value[k]

#                 if conducts:
#                     thomas_solve(a, b, c, d, x, cs, ds)
#                 else:
#                     for i in range(N):
#                         x[i] = d[i]/b[i]
#                 for i in range(N):
#                     Tnew[k, i] = x[i]

#         for kk in range(K):
#             for i in range(N):
#                 T[kk, i] = Tnew[kk, i]
# '''
