#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the udkm1Dsimpy module.
#
# udkm1Dsimpy is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2017 Daniel Schick

"""A :mod:`PhononNum` module """

__all__ = ["PhononNum"]

__docformat__ = "restructuredtext"

import numpy as np
from .phonon import Phonon
from os import path
from time import time


class PhononNum(Phonon):
    """PhononNum

    Base class for numerical phonon simulatuons.

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Keyword Args:
        only_heat (boolean): true when including only thermal expanison without
            coherent phonon dynamics

    Attributes:
        S (object): sample to do simulations with
        only_heat (boolean): force recalculation of results

    References:

        .. [5] A. Bojahr, M. Herzog, D. Schick, I. Vrejoiu, & M. Bargheer
           (2012). `Calibrated real-time detection of nonlinearly propagating
           strain waves. Phys. Rev. B, 86(14), 144306.
           <http://www.doi.org/10.1103/PhysRevB.86.144306>`

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.only_heat = kwargs.get('only_heat', False)

    def __str__(self, output=[]):
        """String representation of this class"""

        class_str = 'Numerical Phonon simulation properties:\n\n'
        class_str += super().__str__()

        return class_str

    def get_strain_map(self, delays, temp_map, delta_temp_map):
        """get_strain_map

        Returns a strain profile for the sample structure for given temperature
        profile. The result can be saved using an unique hash of the sample
        and the simulation parameters in order to reuse it.

        """
        filename = 'strain_map_num_' \
                   + self.get_hash(delays, temp_map, delta_temp_map) \
                   + '.npy'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            strain_map = np.load(full_filename)
            self.disp_message('_strain_map_ loaded from file:\n\t' + filename)
        else:
            # file does not exist so calculate and save
            strain_map, sticks_sub_systems, velocities = \
                self.calc_strain_map(delays, temp_map, delta_temp_map)
            self.save(full_filename, [strain_map], '_strain_map_num_')
        return strain_map

    def calc_strain_map(self, delays, temp_map, delta_temp_map):
        """calc_strain_map

        Calculates the _strain_map_ of the sample structure for a given
        _temp_map_ and _delta_temp_map_ and _delay_ vector. Further details
        are given in Ref. [5]. The coupled differential equations are solved
        for each oscillator in a linear chain of masses and springs:

       .. math::

            m_i\ddot{x}_i = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1})
            + m_i\gamma_i(\dot{x}_i-\dot{x}_{i-1})
            + F_i^{heat}(t)

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
        :math:`k_i = (k_i^1 \ldots k_i^{M-1})` can be a vector accounting for
        higher orders of the potential which is in the harmonic case purely
        quadratic (:math:`k_i = k_i^1`). The resulting force from the
        displacement of the springs

        .. math::

            F_i^{spring} = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1})

        includes:

        .. math::

        k_i(x_i-x_{i-1}) = \sum_{j=1}^{M-1} k_i^j (x_i-x_{i-1})^j

        """
        t1 = time()

        # initialize
        L = self.S.get_number_of_layers()
        thicknesses = self.S.get_layer_property_vector('_thickness')
        x0 = np.zeros([2*L, 1])  # initial condition for the shift of the layers

        try:
            delays = delays.to('s').magnitude
        except AttributeError:
            pass

        # check temp_maps
        [temp_map, delta_temp_map] = self.check_temp_maps(temp_map, delta_temp_map, delays)

        # calculate the sticks due to heat expansion first for all delay steps
        self.disp_message('Calculating linear thermal expansion ...')
        sticks, sticks_sub_systems = self.calc_sticks_from_temp_map(temp_map, delta_temp_map)

        if self.only_heat:
            # no coherent dynamics so calculate the strain directly
            strain_map = sticks/np.tile(thicknesses, [np.size(sticks, 0), 1])
            velocities = np.zeros_like(strain_map)  # this is quasi-static
        else:
            # include coherent dynamics
            self.disp_message('Calculating coherent dynamics with ODE solver ...')

            # calculate the strainMap as the second spacial derivative
            # of the layer shift x(t). The result of the ode solver
            # contains x(t) = X(:,1:N) and v(t) = X(:,N+1:end) the
            # positions and velocities of the layers, respectively.
            temp = np.diff(X[:, 0:L], 0, 2)
            temp[:, :+1] = 0
            strain_map = temp/np.tile(thicknesses, np.size(temp, 0), 1)
            velocities = X[:, L+1:]
        self.disp_message('Elapsed time for _strain_map_:'
                          ' {:f} s'.format(time()-t1))
        return strain_map, sticks_sub_systems, velocities

    def __del__(self):
        # stop matlab engine if exists
        if self.matlab_engine != []:
            self.matlab_engine.quit()
