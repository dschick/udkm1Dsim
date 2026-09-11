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

__all__ = [
    'StructuralParameters',
    'LatticeParameters',
    'ThermalParameters',
    'ElasticParameters',
    'OpticalParameters',
    'MagneticParameters',
]

__docformat__ = 'restructuredtext'

from dataclasses import dataclass

import pint

u = pint.get_application_registry()
Q_ = u.Quantity


@dataclass
class StructuralParameters:
    """Structural parameters of a layer.

    thickness (float): thickness of the layer [m].
    mass (float): mass of the layer [kg].
    mass_unit_area (float): mass of layer normalized to unit area of 1 Å² [kg].
    density (float): density of the layer [kg/m³].
    area (float): area of layer [m²].
    volume (float): volume of layer [m³].
    roughness (float): gaussian width of the top roughness of a layer [m].

    """
    thickness: float | None = None
    density: float | None = None
    roughness: float | None = None
    area: float | None = None


@dataclass
class LatticeParameters:
    """Lattice parameters of a unit cell.

    a_axis (float): lattice parameter a [m].
    b_axis (float): lattice parameter b [m].
    c_axis (float): lattice parameter c [m].
    """
    a_axis: float | None = None
    b_axis: float | None = None
    c_axis: float | None = None


@dataclass
class ThermalParameters:
    """Thermal parameters of a layer.

    therm_cond (list[@lambda]): list of T-dependent thermal conductivity
            [W/(m K)].
    lin_therm_exp (list[@lambda]): list of T-dependent linear thermal
        expansion coefficient (relative).
    int_lin_therm_exp (list[@lambda]): list of T-dependent integrated
        linear thermal expansion coefficient.
    heat_capacity (list[@lambda]): list of T-dependent heat capacity
        function [J/(kg K)].
    int_heat_capacity (list[@lambda]): list of T-dependent integrated heat
        capacity function.
    sub_system_coupling (list[@lambda]): list of coupling functions of
        different subsystems [W/m³].
    num_sub_systems (int): number of subsystems for heat and phonons
        (electrons, lattice, spins, ...).

    """

    therm_cond: float | None = None
    heat_capacity: float | None = None
    lin_therm_exp: float | None = None
    int_lin_therm_exp: float | None = None
    int_heat_capacity: float | None = None
    sub_system_coupling: float | None = None
    num_sub_systems: int = 1


@dataclass
class ElasticParameters:
    """Elastic and phonon parameters of a layer.

    sound_vel (float): longitudinal sound velocity in the layer [m/s].
    spring_const (ndarray[float]): spring constant of the layer [kg/s²]
        and higher orders.
    phonon_damping (float): damping constant of phonon propagation [kg/s].

    """

    sound_vel: float | None = None
    phonon_damping: float | None = None
    spring_const: float | None = None


@dataclass
class OpticalParameters:
    """Optical adn X-ray parameters of a layer.

    opt_pen_depth (float): optical penetration depth of the layer [m].
    opt_ref_index (ndarray[float]): optical refractive index - real
        and imagenary part :math:`n + i\kappa`.
    opt_ref_index_per_strain (ndarray[float]): optical refractive
        index change per strain - real and imagenary part
        :math:`\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}`.
    deb_wal_fac (list[@lambda]): list of T-dependent Debye-Waller factors
                `\langle u^2\rangle` [m²].

    """

    opt_pen_depth: float | None = None
    opt_ref_index: complex | None = None
    opt_ref_index_per_strain: complex | None = None
    deb_wal_fac: float | None = None


@dataclass
class MagneticParameters:
    """Magnetic parameters of a layer.

    eff_spin (float): effective spin.
    curie_temp (float): Curie temperature [K].
    mf_exch_coupling (float): mean field exchange coupling constant [m²kg/s²].
    lamda (float): intrinsic coupling to bath parameter.
    mag_moment (float): atomic magnetic moment [mu_Bohr].
    aniso_exponent(ndarray[float]): exponent of T-dependence uniaxial
        anisotropy.
    anisotropy (ndarray[float]): anisotropy at T=0 K [J/m³] as x,y,z component vector.
    exch_stiffness (float): exchange stiffness at T=0 K [J/m].
    mag_saturation (float): saturation magnetization at 0 K [J/T/m³].
    magnetization (dict[float]): magnetization amplitude, phi and
        gamma angle inherited from the atom.

    """

    eff_spin: float | None = None
    curie_temp: float | None = None
    mf_exch_coupling: float | None = None
    lamda: float | None = None
    mag_moment: float | None = None
    aniso_exponent: float | None = None
    anisotropy: float | None = None
    exch_stiffness: float | None = None
    mag_saturation: float | None = None
    magnetization: float | None = None
