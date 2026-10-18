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

__all__ = ["Phonon", "PhononNum", "PhononAna"]

__docformat__ = "restructuredtext"

import warnings
from os import path
from time import time

import numpy as np
from tqdm.auto import trange

from ..helpers import make_hash_md5
from .simulation import Simulation
from .solvers import HAS_NUMBA, phonon_scipy


class Phonon(Simulation):
    """Phonon

    Base class for phonon simulations in a linear chain of masses and springs.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expansion without
            coherent phonon dynamics.

    Attributes:
        S (Structure): sample structure to calculate simulations on.
        force_recalc (boolean): force recalculation of results.
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expansion without
            coherent phonon dynamics.

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.only_heat = kwargs.get("only_heat", False)

    def __str__(self, output=[]):
        """String representation of this class"""

        output = [
            ["only heat", self.only_heat],
        ] + output

        class_str = "Phonon simulation properties:\n\n"
        class_str += super().__str__(output)

        return class_str

    def get_hash(self, delays, temp_map, delta_temp_map, **kwargs):
        """get_hash

        Calculates an unique hash given by the ``delays``, and ``temp_map``
        and ``delta_temp_map`` as well as the sample structure hash for
        relevant lattice parameters.

        Args:
            delays (ndarray[float]): delay grid for the simulation.
            temp_map (ndarray[float]): spatio-temporal temperature profile.
            delta_temp_map (ndarray[float]): spatio-temporal temperature
                difference profile.
            **kwargs (float, optional): optional parameters.

        Returns:
            hash (str): unique hash.

        """
        param = [delays, self.only_heat]

        if np.size(temp_map) > 1e6:
            temp_map = temp_map.flatten()[0:1000000]
            delta_temp_map = delta_temp_map.flatten()[0:1000000]
        param.append(temp_map)
        param.append(delta_temp_map)

        for value in kwargs.values():
            param.append(value)

        return self.S.get_hash(types="phonon") + "_" + make_hash_md5(param)

    def get_all_strains_per_unique_layer(self, strain_map):
        """get_all_strains_per_unique_layer

        Determines all values of the strain per unique layer.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.

        Returns:
            strains (list[ndarray[float]]: all strains per unique layer.

        """
        # get the position indices of all unique layers in the sample structure
        positions = self.S.get_all_positions_per_unique_layer()
        strains = []

        for value in positions.values():
            strains.append(np.sort(np.unique(strain_map[:, value].flatten())))

        return strains

    def get_reduced_strains_per_unique_layer(self, strain_map, N=100):
        """get_reduced_strains_per_unique_layer

        Calculates all strains per unique layer that are given by the
        input ``strain_map``, but with a reduced number. The reduction is done
        by equally spacing the strains between the ``min`` and ``max`` strain
        with a given number :math:`N`, which can be also an array of the
        :math:`len(N) = L`, where :math:`L` is the number of unique layers.

        Args:
            strain_map (ndarray[float]): spatio-temporal strain profile.
            N (int, optional): number of reduced strains. Defaults to 100.

        Returns:
            strains (list[ndarray[float]]: reduced strains per unique layer.

        """
        # initialize
        all_strains = self.get_all_strains_per_unique_layer(strain_map)
        L = len(all_strains)  # Nb. of unique layers
        strains = []

        # normalize N to a 1D integer array of length L
        if np.size(N) == 1:
            N = np.full(L, int(N))
        else:
            N = np.asarray(N).ravel()
            if N.size != L:
                raise ValueError(
                    "The dimension of N must be either 1 or the number "
                    f"of unique layers ({L:d}) the structure!"
                )
            N = N.astype(int)

        for i, value in enumerate(all_strains):
            min_strain = np.min(value)
            max_strain = np.max(value)

            strains.append(
                np.sort(np.unique(np.r_[0, np.linspace(min_strain, max_strain, int(N[i]))]))
            )

        return strains

    def check_temp_maps(self, temp_map, delta_temp_map, delays):
        """check_temp_maps

        Check temperature profiles for correct dimensions.

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature profile.
            delta_temp_map (ndarray[float]): spatio-temporal temperature
                difference profile.
            delays (ndarray[float]): delay grid for the simulation.

        Returns:
            (tuple):
            - *temp_map (ndarray[float])* - checked spatio-temporal temperature
              profile.
            - *delta_temp_map (ndarray[float])* - checked spatio-temporal
              differential temperature profile.

        """
        M = len(delays)
        L = self.S.get_number_of_layers()
        K = self.S.num_sub_systems

        # check size of delta_temp_map
        if K == 1:
            if np.shape(delta_temp_map) == (1, L):
                temp = delta_temp_map
                delta_temp_map = np.zeros([M, L])
                delta_temp_map[0, :] = temp
            elif (np.size(delta_temp_map, 0) != M) or (np.size(delta_temp_map, 1) != L):
                raise ValueError(
                    "The given temperature difference map does not have the "
                    "dimension M x L, where M is the number of delay steps "
                    "and L the number of layers!"
                )
        else:
            if np.shape(delta_temp_map) == (1, L, K):
                temp = delta_temp_map
                delta_temp_map = np.zeros([K, L, K])
                delta_temp_map[0, :, :] = temp
            elif (
                (np.size(delta_temp_map, 0) != M)
                or (np.size(delta_temp_map, 1) != L)
                or (np.size(delta_temp_map, 2) != K)
            ):
                raise ValueError(
                    "The given temperature difference map does not have the "
                    "dimension M x L x K, where M is the number of delay steps "
                    "and L the number of layers and K is the number of subsystems!"
                )

        if np.shape(temp_map) != np.shape(delta_temp_map):
            raise ValueError(
                "The temperature map does not have the same size as the "
                "temperature difference map!"
            )

        return temp_map, delta_temp_map

    def calc_sticks_from_temp_map(self, temp_map, delta_temp_map):
        r"""calc_sticks_from_temp_map

        Calculates the sticks to insert into the layer springs which model the
        external force (thermal stress). The length of :math:`l_i` of the
        :math:`i`-th spacer stick is calculated from the temperature-dependent
        linear thermal expansion :math:`\alpha(T)` of the layer:

        .. math::

            \alpha(T) = \frac{1}{L} \frac{d L}{d T}

        which results after integration in

        .. math::

            l = \Delta L = L_1 \exp(A(T_2) - A(T_1)) - L_1

        where :math:`A(T)` is the integrated lin. therm. expansion coefficient
        in respect to the temperature :math:`T`. The indices 1 and 2 indicate
        the initial and final state.

        Args:
            temp_map (ndarray[float]): spatio-temporal temperature profile.
            delta_temp_map (ndarray[float]): spatio-temporal temperature
                difference profile.

        Returns:
            (tuple):
            - *sticks (ndarray[float])* - summed spacer sticks.
            - *sticks_sub_systems (ndarray[float])* - spacer sticks per
              sub-system.

        """
        M = np.size(temp_map, 0)  # nb of delay steps
        L = self.S.get_number_of_layers()
        K = self.S.num_sub_systems

        temp_map = np.reshape(temp_map, [M, L, K])
        delta_temp_map = np.reshape(delta_temp_map, [M, L, K])

        thicknesses = self.S.get_layer_property_vector("structural.thickness.magnitude")
        int_lin_therm_exps = self.S.get_layer_property_vector("thermal.lin_therm_exp.integral")

        # evaluated initial integrated linear thermal expansion from T1 to T2
        int_alpha_T0 = np.zeros([L, K])
        # evaluated integrated linear thermal expansion from T1 to T2
        int_alpha_T = np.zeros([L, K])
        sticks = np.zeros([M, L])  # the sticks inserted in the unit cells
        sticks_sub_systems = np.zeros([M, L, K])  # the sticks for each thermodynamic subsystem

        # calculate initial integrated linear thermal expansion from T1 to T2
        # traverse subsystems
        for ii in range(L):
            for iii in range(K):
                int_alpha_T0[ii, iii] = int_lin_therm_exps[ii][iii](
                    temp_map[0, ii, iii] - delta_temp_map[0, ii, iii]
                )

        # calculate sticks for all subsystems for all delay steps
        # traverse time
        for i in range(M):
            if np.any(delta_temp_map[i, :]):  # there is a temperature change
                # Calculate new sticks from the integrated linear thermal
                # expansion from initial temperature to current temperature for
                # each subsystem
                # traverse subsystems
                for ii in range(L):
                    for iii in range(K):
                        int_alpha_T[ii, iii] = int_lin_therm_exps[ii][iii](temp_map[i, ii, iii])

                # calculate the length of the sticks of each subsystem and sum
                # them up
                sticks_sub_systems[i, :, :] = (
                    np.tile(thicknesses, (K, 1)).T * np.exp(int_alpha_T - int_alpha_T0)
                    - np.tile(thicknesses, (K, 1)).T
                )
                sticks[i, :] = np.sum(sticks_sub_systems[i, :, :], 1)
            else:  # no temperature change, so keep the current sticks
                if i > 0:
                    sticks_sub_systems[i, :, :] = sticks_sub_systems[i - 1, :, :]
                    sticks[i, :] = sticks[i - 1, :]
        return sticks, sticks_sub_systems


class PhononNum(Phonon):
    """PhononNum

    Numerical model to simulate coherent acoustic phonons following
    :cite:t:`bojahr2012`.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expansion without
            coherent phonon dynamics.
        backend (str): ode solver backend - either default scipy or numba.

    Attributes:
        S (Structure): sample structure to calculate simulations on.
        force_recalc (boolean): force recalculation of results.
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expansion without
            coherent phonon dynamics.
        backend (str): ode solver backend - either default scipy or numba.
        ode_options (dict): options for scipy solve_ivp ode solver. The numba
            backend uses ``rtol``, ``atol``, ``max_step`` and ``first_step``.

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.backend = kwargs.get("backend", "scipy")
        self.ode_options = {
            "method": "RK45",
            "first_step": None,
            "max_step": np.inf,
            "rtol": 1e-3,
            "atol": 1e-6,
        }

    def __str__(self, output=[]):
        """String representation of this class"""

        output = [
            ["backend", self.backend],
        ] + output

        class_str = "Numerical Phonon simulation properties:\n\n"
        class_str += super().__str__(output)

        return class_str

    def get_strain_map(self, delays, temp_map, delta_temp_map):
        """get_strain_map

        Returns a strain profile for the sample structure for given temperature
        profile. The result can be saved using an unique hash of the sample
        and the simulation parameters in order to reuse it.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            temp_map (ndarray[float]): spatio-temporal temperature map.
            delta_temp_map (ndarray[float]): spatio-temporal differential
            temperature map.

        Returns:
            strain_map (ndarray[float]): spatio-temporal strain profile.

        """
        filename = "strain_map_num_" + self.get_hash(delays, temp_map, delta_temp_map) + ".npz"
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            strain_map = tmp["strain_map"]
            self.disp_message("_strain_map_ loaded from file:\n\t" + filename)
        else:
            # file does not exist so calculate and save
            strain_map, _, _ = self.calc_strain_map(delays, temp_map, delta_temp_map)
            self.save(full_filename, {"strain_map": strain_map}, "_strain_map_num_")
        return strain_map

    def calc_strain_map(self, delays, temp_map, delta_temp_map):
        r"""calc_strain_map

        Calculates the ``strain_map`` of the sample structure for a given
        ``temp_map`` and ``delta_temp_map`` and ``delay`` array. Further
        details are given in :cite:t:`bojahr2012`. The coupled differential
        equations are solved for each oscillator in a linear chain of masses
        and springs:

        .. math::

            m_i\ddot{x}_i = & -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1}) \\
            & + m_i\gamma_i(\dot{x}_i-\dot{x}_{i-1}) + F_i^{heat}(t)

        where :math:`x_i(t) = z_{i}(t) - z_i^0` is the shift of each layer.
        :math:`m_i` is the mass and :math:`k_i = m_i\, v_i^2/c_i^2` is the
        spring constant of each layer. Furthermore an empirical damping term
        :math:`F_i^{damp} = \gamma_i(\dot{x}_i-\dot{x}_{i-1})` is introduced
        and the external force (thermal stress) :math:`F_i^{heat}(t)`.
        The thermal stresses are modelled as spacer sticks which are
        calculated from the linear thermal expansion coefficients. The
        equation of motion can be reformulated as:

        .. math::

            m_i\ddot{x}_i = F_i^{spring} + F_i^{damp} + F_i^{heat}(t)

        The numerical solution also allows for non-harmonic inter-atomic
        potentials of up to the order :math:`M`. Accordingly
        :math:`k_i = (k_i^1 \ldots k_i^{M-1})` can be an array accounting for
        higher orders of the potential which is in the harmonic case purely
        quadratic (:math:`k_i = k_i^1`). The resulting force from the
        displacement of the springs

        .. math::

            F_i^{spring} = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1})

        includes:

        .. math::

            k_i(x_i-x_{i-1}) = \sum_{j=1}^{M-1} k_i^j (x_i-x_{i-1})^j

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            temp_map (ndarray[float]): spatio-temporal temperature map.
            delta_temp_map (ndarray[float]): spatio-temporal differential
                temperature map.

        Returns:
            (tuple):
            - *strain_map (ndarray[float])* - spatio-temporal strain profile.
            - *sticks_sub_systems (ndarray[float])* - spacer sticks per
              sub-system.
            - *velocities (ndarray[float])* - spatio-temporal velocity profile.

        """
        t1 = time()

        # initialize
        L = self.S.get_number_of_layers()
        thicknesses = self.S.get_layer_property_vector("structural.thickness.magnitude")
        x0 = np.zeros([2 * L])  # initial condition for the shift of the layers

        try:
            delays = delays.to("s").magnitude
        except AttributeError:
            pass

        # check temp_maps
        [temp_map, delta_temp_map] = self.check_temp_maps(temp_map, delta_temp_map, delays)

        # calculate the sticks due to heat expansion first for all delay steps
        self.disp_message("Calculating linear thermal expansion ...")
        sticks, sticks_sub_systems = self.calc_sticks_from_temp_map(temp_map, delta_temp_map)

        if self.only_heat:
            # no coherent dynamics so calculate the strain directly
            strain_map = sticks / np.tile(thicknesses, [np.size(sticks, 0), 1])
            velocities = np.zeros_like(strain_map)  # this is quasi-static
        else:
            # include coherent dynamics
            self.disp_message("Calculating coherent dynamics with ODE solver ...")

            L = self.S.get_number_of_layers()
            masses = self.S.get_layer_property_vector("structural.mass_unit_area.magnitude")
            thicknesses = self.S.get_layer_property_vector("structural.thickness.magnitude")
            spring_consts = self.S.get_layer_property_vector("elastic.spring_const.magnitude")
            damping = self.S.get_layer_property_vector("elastic.phonon_damping.magnitude")
            force_from_heat = phonon_scipy.force_from_heat(sticks, spring_consts)

            if self.backend == "numba":
                from .solvers.phonon_numba import solve
            else:
                from .solvers.phonon_scipy import solve

            X = solve(
                x0,
                delays,
                force_from_heat,
                damping,
                spring_consts,
                masses,
                self.ode_options,
                self.progress_bar,
            )

            # calculate the strain_map as the second spacial derivative
            # of the layer shift x(t). The result of the ode solver
            # contains x(t) = X(:,1:N) and v(t) = X(:,N+1:end) the
            # positions and velocities of the layers, respectively.
            temp = np.diff(X[:, 0:L], 1, 1)
            strain_map = np.zeros([temp.shape[0], temp.shape[1] + 1])
            strain_map[:, :-1] = temp
            strain_map = strain_map / np.tile(thicknesses[:], [np.size(temp, 0), 1])
            velocities = X[:, L:]
        self.disp_message(f"Elapsed time for _strain_map_: {time() - t1:f} s")
        return strain_map, sticks_sub_systems, velocities

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if backend not in ["scipy", "numba"]:
            raise ValueError("Backend must be either 'scipy' or 'numba'!")

        if backend == "numba" and not HAS_NUMBA:
            warnings.warn("Backend 'numba' is not installed. Fallback to default 'scipy'.")
            backend = "scipy"

        self._backend = backend


class PhononAna(Phonon):
    """PhononAna

    Analytical model to simulate coherent acoustic phonons following
    :cite:t:`herzog2012`.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expansion without
            coherent phonon dynamics.

    Attributes:
        S (Structure): sample structure to calculate simulations on.
        force_recalc (boolean): force recalculation of results.
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.
        only_heat (boolean): true when including only thermal expansion without
            coherent phonon dynamics.

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self, output=[]):
        """String representation of this class"""

        class_str = "Analytical Phonon simulation properties:\n\n"
        class_str += super().__str__()

        return class_str

    def get_strain_map(self, delays, temp_map, delta_temp_map):
        """get_strain_map

        Returns a strain profile for the sample structure for given temperature
        profile. The result can be saved using an unique hash of the sample
        and the simulation parameters in order to reuse it.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            temp_map (ndarray[float]): spatio-temporal temperature map.
            delta_temp_map (ndarray[float]): spatio-temporal differential
            temperature map.

        Returns:
            (tuple):
            - *strain_map (ndarray[float])* - spatio-temporal strain profile.
            - *A (ndarray[float])* - coefficient vector A of general solution.
            - *B (ndarray[float])* - coefficient vector B of general solution.

        """
        filename = "strain_map_ana_" + self.get_hash(delays, temp_map, delta_temp_map) + ".npz"
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            strain_map = tmp["strain_map"]
            A = tmp["A"]
            B = tmp["B"]
            self.disp_message("_strain_map_ loaded from file:\n\t" + filename)
        else:
            # file does not exist so calculate and save
            strain_map, A, B = self.calc_strain_map(delays, temp_map, delta_temp_map)
            self.save(
                full_filename, {"strain_map": strain_map, "A": A, "B": B}, "_strain_map_ana_"
            )
        return strain_map, A, B

    def calc_strain_map(self, delays, temp_map, delta_temp_map):
        r"""calc_strain_map

        Calculates the ``strain_map`` of the sample structure for a given
        ``temp_map`` and ``delta_temp_map`` and ``delay`` array. Further
        details are given in :cite:t:`herzog2012`. Within the linear chain
        of :math:`N` masses (:math:`m_i`) at position :math:`z_i` coupled
        with spring constants :math:`k_i` one can formulate the differential
        equation of motion as follow:

        .. math::

            m_i\ddot{x}_i = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1}) + F_i^{heat}(t)

        Since we only consider nearest-neighbor interaction one can write:

        .. math::

            \ddot{x}_i = \sum_{n=1}^N \kappa_{i,n} x_n = \Delta_i(t)

        Here :math:`x_i(t) = z_i(t)-z_i^0` is the shift of each layer,
        :math:`F_i^{heat}(t)` is the external force (thermal stress) of each
        layer and :math:`\kappa_{i,i} = -(k_i + k_{i+1})/m_i`, and
        :math:`\kappa_{i,i+1} = \kappa_{i+1,i} = k_{i+1}/m_i`.

        :math:`k_i = m_i\, v_i^2/c_i^2` is the spring constant and :math:`c_i`
        and :math:`v_i` are the thickness and longitudinal sound velocity of
        each layer respectively.
        One can rewrite the homogeneous differential equation in matrix
        form to obtain the general solution

        .. math::

            \frac{d^2}{dt^2} X = K \, X

        Here :math:`X = (x_1 \ldots x_N)` and :math:`K` is the
        tri-diagonal matrix of :math:`\kappa` which is real and symmetric.
        The differential equation can be solved with the ansatz:

        .. math::

         X(t) = \sum_j \Xi_j \, (A_j \cos(\omega_j \, t) + B_j \sin(\omega_j \, t))

        where :math:`\Xi_j = (\xi_1^j \ldots \xi_N^j)` are the eigenvectors of
        the matrix :math:`K`. Thus by solving the Eigenproblem for :math:`K` one
        gets the eigenvecotrs :math:`\Xi_j` and the eigenfrequencies
        :math:`\omega_j`. From the initial conditions

        .. math::

            X(0) = \sum_j \Xi_j \, A_j = \Xi \, A \qquad V(0) = \dot{X}(0)
                 = \sum_j \Xi_j \, \omega_j\, B_j = \Xi \, \omega \, B

        one can determine the real coefficient vectors :math:`A` and :math:`B` in
        order to calculate :math:`X(t)` and :math:`V(t)` using the ansatz:

        .. math::

            A = \Xi \setminus X(0) \qquad B = (\Xi \setminus V(0)) / \omega

        The external force is implemented as spacer sticks which are
        inserted into the springs and hence the layers have a new
        equilibrium positions :math:`z_i(\infty) = z_i^\infty`.
        Thus we can do a coordination transformation:

        .. math::

            z_i(t) = z_i^0 + x_i(t) = z_i^\infty + x_i^\infty(t)

        and

        .. math::

            x_i^\infty(t) = z_i^0 - z_i^\infty + x_i(t)

        with the initial condition :math:`x_i(0) = 0` it becomes

        .. math::

            x_i^\infty(0) = z_i^0 - z_i^\infty = \sum_{j = i+1}^N l_j

        :math:`x_i^\infty(0)` is the new initial condition after the excitation
        where :math:`l_i` is the length of the :math:`i`-th spacer stick.
        The spacer sticks are calculated from the temperature change and the
        linear thermal expansion coefficients.
        The actual strain :math:`\epsilon_i(t)` of each layer is calculates
        as follows:

        .. math::

            \epsilon_i(t) = [ \Delta x_i(t) + l_i) ] / c_i

        with :math:`\Delta x_i = x_i - x_{i-1}`. The stick :math:`l_i` have
        to be added here, because :math:`x_i` has been transformed into the new
        coordinate system :math:`x_i^\infty`.

        Args:
            delays (ndarray[Quantity]): delays range of simulation [s].
            temp_map (ndarray[float]): spatio-temporal temperature map.
            delta_temp_map (ndarray[float]): spatio-temporal differential
              temperature map.

        Returns:
            (tuple):
            - *strain_map (ndarray[float])* - spatio-temporal strain profile.
            - *A (ndarray[float])* - coefficient vector A of general solution.
            - *B (ndarray[float])* - coefficient vector B of general solution.

        """
        t1 = time()

        # initialize
        L = self.S.get_number_of_layers()
        M = len(delays)

        try:
            delays = delays.to("s").magnitude
        except AttributeError:
            pass

        delay0 = delays[0]  # initial delay
        thicknesses = self.S.get_layer_property_vector("structural.thickness.magnitude")
        X = np.zeros([M, L])  # shifts of the layers
        V = np.zeros_like(X)  # velocities of the layers
        A = np.zeros_like(X)  # coefficient vector for eigenwert solution
        B = np.zeros_like(X)  # coefficient vector for eigenwert solution
        strain_map = np.zeros_like(X)  # the restulting strain map

        # check temp_maps
        [temp_map, delta_temp_map] = self.check_temp_maps(temp_map, delta_temp_map, delays)

        # calculate the sticks due to heat expansion first for all delay steps
        self.disp_message("Calculating linear thermal expansion ...")
        sticks, _ = self.calc_sticks_from_temp_map(temp_map, delta_temp_map)

        if self.only_heat:
            # no coherent dynamics so calculate the strain directly
            strain_map = sticks / np.tile(thicknesses, [np.size(sticks, 0), 1])
        else:
            # solve the eigenproblem for the structure to obtains the
            # eigenvectors X_i and eigenfreqeuencies omega for the L
            # coupled differential equations
            Xi, omega = self.solve_eigenproblem()
            # calculate the actual strain map with the solution of the
            # eigenproblem and the external force (sticks, thermal stress)
            self.disp_message("Calculating _strain_map_ ...")
            if self.progress_bar:
                iterator = trange(M, desc="Progress", leave=True)
            else:
                iterator = range(M)
            for i in iterator:
                dt = delays[i] - delay0  # this is the time step
                # calculate the current shift X and velocity V of all
                # layers using the ansatz
                X[i, :] = np.real(
                    np.dot(Xi, (A[i, :].T * np.cos(omega * dt) + B[i, :].T * np.sin(omega * dt)))
                )
                V[i, :] = np.real(
                    np.dot(
                        Xi,
                        (
                            omega
                            * (-A[i, :].T * np.sin(omega * dt) + B[i, :].T * np.cos(omega * dt))
                        ),
                    )
                )
                # remember the velocities and shifts as ic for the next
                # time step
                X0 = X[i, :].T
                V0 = V[i, :].T
                # the strain can only be calculated for L-1 layers, so
                # we neglect the last one
                if i > 0:
                    strain_map[i, 0:-1] = (np.diff(X[i, :]) + sticks[i - 1, 0:-1]) / thicknesses[
                        0:-1
                    ].T
                else:
                    # initial sticks are zero
                    strain_map[i, 0:-1] = np.diff(X[i, :]) / thicknesses[0:-1].T
                # calculate everything for the next step
                if i < (M - 1):  # check, if there is a next step
                    if np.any(delta_temp_map[i, :]):  # there is a temperature change
                        delay0 = delays[i]  # set new initial delay
                        # determining the shifts due to inserted sticks
                        # as new initial conditions
                        if i > 0:
                            temp = np.flipud(
                                np.cumsum(np.flipud(sticks[i, :].T - sticks[i - 1, :].T))
                            )
                        else:
                            # initial sticks are zero
                            temp = np.flipud(np.cumsum(np.flipud(sticks[i, :].T)))
                        X0 = X0 + temp
                        # determining the coefficient vectors A and B of
                        # the general solution of X(t) using the initial
                        # conditions X0 and V0
                        A[i + 1, :] = np.real(np.linalg.solve(Xi, X0))
                        B[i + 1, :] = np.real((np.linalg.solve(Xi, V0) / omega).T)
                    else:
                        # no temperature change, so keep the current As,
                        # Bs, and sticks
                        A[i + 1, :] = A[i, :]
                        B[i + 1, :] = B[i, :]

        self.disp_message(f"Elapsed time for _strain_map_: {time() - t1:f} s")

        return strain_map, A, B

    def solve_eigenproblem(self):
        r"""solve_eigenproblem

        Creates the real and symmetric :math:`K` matrix (:math:`L \times L`) of
        spring constants :math:`k_i` and masses :math:`m_i` and calculates the
        eigenvectors :math:`\Xi_j` and eigenfrequencies :math:`\omega_j` for the
        matrix which are used to calculate the ``strain_map`` of the structure.
        If the result has been save to file, load it from there.

        Returns:
            (tuple):
            - *Xi (ndarray[float])* - eigenvectors.
            - *omega (ndarray[float])* - eigenfrequencies.

        """
        # create the file name to look for
        filename = "eigenvalues_" + self.S.get_hash(types="phonon") + ".npz"
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            tmp = np.load(full_filename)
            Xi = tmp["Xi"]
            omega = tmp["omega"]
            self.disp_message("_eigen_values_ loaded from file:\n\t" + filename)
        else:
            # file does not exist so calculate and save
            t1 = time()
            self.disp_message("Calculating _eigen_values_ ...")
            # initialize
            L = self.S.get_number_of_layers()
            K = np.zeros([L, L])  # initializing three-diagonal springs-masses matrix.
            # initializing a vector for eigenfrequencies
            omega = np.zeros([L, 1], dtype=np.complex128)

            masses = self.S.get_layer_property_vector("structural.mass_unit_area.magnitude")
            spring_consts = self.S.get_layer_property_vector("elastic.spring_const.magnitude")
            spring_consts = np.hstack((0, spring_consts))  # set the first spring free

            for i in range(L):  # defining main diagonal
                K[i, i] = -(spring_consts[i] + spring_consts[i + 1]) / masses[i]

            # defining the two other diagonals - nearest neighbor interaction
            for i in range(1, L):
                K[i, i - 1] = spring_consts[i] / masses[i]
                K[i - 1, i] = spring_consts[i] / masses[i - 1]

            # determining the eigenvectors and the eigenvalues
            lambd, Xi = np.linalg.eig(K)

            # calculate the eigenfrequencies from the eigenvalues
            omega = np.sqrt(-lambd)

            self.disp_message(f"Elapsed time for _eigen_values_: {time() - t1:f} s")
            # save the result to file
            self.save(full_filename, {"Xi": Xi, "omega": omega}, "_eigen_values_")

        return Xi, omega

    def get_energy_per_eigenmode(self, A, B):
        r"""get_energy_per_eigenmode

        Returns the sorted energy per Eigenmode of the coherent phonons of
        the 1D sample.

        The total energy of the linear chain is

        .. math::

            E = \frac{1}{2}\left(\dot{X}^T M \dot{X} + X^T K X\right)

        which, expanded in the eigenbasis :math:`X(t) = \sum_j \Xi_j\,
        (A_j\cos(\omega_j t) + B_j\sin(\omega_j t))`, decomposes exactly
        into a sum of independent per-mode energies (see :cite:t:`herzog2012`):

        .. math::

            E_j & = \frac{1}{2} (A^2_j + B^2_j)\, \omega_j^2\,
                  \left(\Xi_j^T M \Xi_j\right) \\
                & = \frac{1}{2} (A^2_j + B^2_j)\, \omega_j^2\,
                  \sum_{i=1}^N m_i\,(\xi_i^j)^2

        Here :math:`M = \mathrm{diag}(m_1, \ldots, m_N)` is the diagonal
        matrix of the (mass-per-unit-area) layer masses :math:`m_i`, and
        :math:`\Xi_j = (\xi_1^j \ldots \xi_N^j)` is the :math:`j`-th
        eigenvector.

        Args:
            A (ndarray[float]): coefficient vector A of general solution.
            B (ndarray[float]): coefficient vector B of general solution.

        Returns:
            (tuple):
            - *omega (ndarray[float])* - eigenfrequencies in Hz.
            - *E (ndarray[float])* - energy per eigenmode in J.

        """
        # initialize
        L = self.S.get_number_of_layers()
        M = A.shape[0]  # nb of delays
        E = np.zeros([M, L])
        masses = self.S.get_layer_property_vector("structural.mass_unit_area.magnitude")

        # get the eigenVectors and eigenFrequencies
        Xi, omega = self.solve_eigenproblem()

        # sort the frequencies
        idx = np.argsort(omega)

        # generalized mass per mode: diag(Xi.T @ M @ Xi)
        M_tilde = np.sum(masses[:, None] * Xi**2, axis=0)

        # traverse time
        for i in range(M):
            # calculate the energy for the jth mode
            E[i, :] = np.real(0.5 * (A[i, :].T ** 2 + B[i, :].T ** 2) * omega**2 * M_tilde)

        return np.real(omega[idx]), E[:, idx]
