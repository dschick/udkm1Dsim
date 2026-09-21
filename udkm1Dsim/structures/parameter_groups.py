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
    "ParameterGroup",
    "StructuralParameters",
    "LatticeParameters",
    "ThermalParameters",
    "ElasticParameters",
    "OpticalParameters",
    "MagneticParameters",
]

__docformat__ = "restructuredtext"

import re

import numpy as np
import pint
import scipy.constants as constants
from tabulate import tabulate

from .parameters import Parameter, TemperatureParameter, VectorParameter

u = pint.get_application_registry()


class ParameterGroup:
    """A group of Parameters with units."""

    def _table_representation(self, style="double_grid"):
        rows = [[name, p.quantity] for name, p in vars(self).items()]
        return tabulate(
            rows, headers=["Parameter", "Value"], tablefmt=style, colalign=("right", "right")
        )

    def _pretty_class_name(self):
        return "".join(
            f"{word} "
            for word in re.sub(
                "([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", self.__class__.__name__)
            ).split()
        )

    def __repr__(self):
        class_str = self._pretty_class_name() + "\n"
        class_str += self._table_representation()
        return class_str

    def _repr_html_(self):
        return f"<h3>{self._pretty_class_name()}</h3>" + self._table_representation(style="html")


class StructuralParameters(ParameterGroup):
    """Structural parameters of a layer.

    thickness (float): thickness of the layer [m].
    density (float): density of the layer [kg/m³].
    area (float): area of layer [m²].
    roughness (float): gaussian width of the top roughness of a layer [m].
    mass (float): mass of the layer [kg].
    mass_unit_area (float): mass of layer normalized to unit area of 1 Å² [kg].
    volume (float): volume of layer [m³].

    """

    def __init__(self, thickness=0.0, density=0.0, area=1.0 * u.angstrom**2, roughness=0.0):
        self.thickness = Parameter("m", thickness)
        self.density = Parameter("kg/m**3", density)
        self.area = Parameter("m**2", area)
        self.roughness = Parameter("m", roughness)
        self.mass = Parameter("kg", 0.0)
        self.mass_unit_area = Parameter("kg", 0.0)
        self.volume = Parameter("m**3", 0.0)

        self._update_depending()

        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name
            if name in ["thickness", "density", "area"]:
                p._caller = self

    def _update_depending(self):
        self.volume.magnitude = self.thickness.magnitude * self.area.magnitude
        self.mass.magnitude = self.density.magnitude * self.volume.magnitude
        try:
            self.mass_unit_area.quantity = (
                self.mass.quantity / self.area.quantity * 1.0 * u.angstrom**2
            )
        except ZeroDivisionError:
            self.mass_unit_area.quantity = 0 * u.kg


class LatticeParameters(ParameterGroup):
    """Lattice parameters of a unit cell.

    a_axis (float): lattice parameter a [m].
    b_axis (float): lattice parameter b [m].
    c_axis (float): lattice parameter c [m].
    """

    def __init__(self, a_axis=0.0, b_axis=0.0, c_axis=0.0):
        self.a_axis = Parameter("m", a_axis)
        self.b_axis = Parameter("m", b_axis)
        self.c_axis = Parameter("m", c_axis)

        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name


class ThermalParameters(ParameterGroup):
    """Thermal parameters of a layer.

    therm_cond (list[@lambda]): list of T-dependent thermal conductivity
            [W/(m K)].
    lin_therm_exp (list[@lambda]): list of T-dependent linear thermal
        expansion coefficient (relative).
    heat_capacity (list[@lambda]): list of T-dependent heat capacity
        function [J/(kg K)].
    sub_system_coupling (list[@lambda]): list of coupling functions of
        different subsystems [W/m³].
    deb_wal_fac (list[@lambda]): list of T-dependent Debye-Waller factors
                    `\\langle u^2\rangle` [m²].
    num_sub_systems (int): number of subsystems for heat and phonons
        (electrons, lattice, spins, ...).

    """

    def __init__(
        self,
        therm_cond=0.0,
        heat_capacity=0.0,
        lin_therm_exp=0.0,
        sub_system_coupling=0.0,
        deb_wal_fac=0.0,
    ):
        self.therm_cond = TemperatureParameter("W/(m K)", therm_cond)
        self.heat_capacity = TemperatureParameter("J/(kg K)", heat_capacity)
        self.lin_therm_exp = TemperatureParameter("", lin_therm_exp)
        self.sub_system_coupling = TemperatureParameter("W/m**3", sub_system_coupling)
        self.deb_wal_fac = TemperatureParameter("m**2", deb_wal_fac)
        self.num_sub_systems = Parameter("", 0)

        self._update_depending()

        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name
            if isinstance(p, TemperatureParameter):
                p._caller = self

    def _update_depending(self):
        K = self.num_sub_systems.magnitude
        current_num_sub_systems = []
        for _, p in vars(self).items():
            if isinstance(p, TemperatureParameter):
                current_num_sub_systems.append(p._num_sub_systems)

        max_num = max(current_num_sub_systems)
        if max_num != K:
            self.num_sub_systems.magnitude = max_num
            if K > 0:
                print(f"'num_sub_systems' has been updated from {K} to {max_num}.")

        if len(set(current_num_sub_systems)) != 1:
            print(
                "'num_sub_systems' is not consistent for all "
                "'ThermalParameters' for this layer including\n"
                "'heat_capacity', 'therm_cond', 'lin_therm_exp', "
                "'sub_system_coupling', and 'deb_wal_fac'!"
            )


class ElasticParameters(ParameterGroup):
    """Elastic and phonon parameters of a layer.

    sound_vel (float): longitudinal sound velocity in the layer [m/s].
    phonon_damping (float): damping constant of phonon propagation [kg/s].
    spring_const (ndarray[float]): spring constant of the layer [kg/s²]
        and higher orders.
    acoustic_impedance (float): acoustic impedence of a layer [kg/m/s].

    """

    def __init__(self, sound_vel=0.0, phonon_damping=0.0, spring_const=0.0):
        self.sound_vel = Parameter("m/s", sound_vel)
        self.phonon_damping = Parameter("kg/s", phonon_damping)
        self.spring_const = Parameter("kg/s**2", np.array([0.0]))
        self.acoustic_impedance = Parameter("kg/m/s", 0.0)

        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name

    def calc_spring_const(self, mass_unit_area, thickness):
        r"""calc_spring_const

        Calculates the spring constant of the layer from the mass per unit area,
        sound velocity and thickness

        .. math:: k = m \, \left(\frac{v}{c}\right)^2

        """
        try:
            self.spring_const.magnitude[0] = (
                mass_unit_area * (self.sound_vel.magnitude / thickness) ** 2
            )
        except (ZeroDivisionError, AttributeError):
            # no mass set, yet
            self.spring_const.magnitude[0] = 0

    def calc_acoustic_impedance(self, mass, area):
        """calc_acoustic_impedance

        Calculates the acoustic impedance.

        Returns:
            Z (float): acoustic impedance.

        """
        self.acoustic_impedance.magnitude = np.sqrt(
            self.spring_const.magnitude[0] * mass / area**2
        )

    def set_ho_spring_constants(self, HO):
        """set_ho_spring_constants

        Set the higher orders of the spring constant for anharmonic
        phonon simulations.

        Args:
            HO (ndarray[float]): higher order spring constants.

        """
        # reset old higher order spring constants
        self.spring_const.magnitude = np.delete(
            self.spring_const.magnitude, np.r_[1 : len(self.spring_const.magnitude)]
        )
        self.spring_const.magnitude = np.hstack((self.spring_const.magnitude, HO))


class OpticalParameters(ParameterGroup):
    r"""Optical adn X-ray parameters of a layer.

    opt_pen_depth (float): optical penetration depth of the layer [m].
    opt_ref_index (ndarray[float]): optical refractive index - real
        and imagenary part :math:`n + i\kappa`.
    opt_ref_index_per_strain (ndarray[float]): optical refractive
        index change per strain - real and imagenary part
        :math:`\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}`.

    """

    def __init__(
        self, opt_pen_depth=0.0, opt_ref_index=0.0 + 0.0j, opt_ref_index_per_strain=0.0 + 0.0j
    ):
        self.opt_pen_depth = Parameter("m", opt_pen_depth)
        self.opt_ref_index = Parameter("", opt_ref_index)
        self.opt_ref_index_per_strain = Parameter("", opt_ref_index_per_strain)

        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name

    # def set_opt_pen_depth_from_ref_index(self, wavelength):
    #     """set_opt_pen_depth_from_ref_index
    #
    #     Set the optical penetration depth from the optical referactive index
    #     for a given wavelength.
    #
    #     Args:
    #         wavelength (Quantity): wavelength as Pint Quantitiy.
    #
    #     """
    #     if np.imag(self.opt_ref_index) == 0:
    #         self.opt_pen_depth = Q_(np.inf, u.m)
    #     else:
    #         self.opt_pen_depth = wavelength/(4*np.pi*np.abs(np.imag(self.opt_ref_index)))


class MagneticParameters(ParameterGroup):
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

    def __init__(
        self,
        eff_spin=0.0,
        curie_temp=0.0,
        lamda=0.0,
        mag_moment=0.0,
        aniso_exponent=0.0,
        anisotropy=np.array([0.0, 0.0, 0.0]),
        exch_stiffness=np.array([0.0, 0.0, 0.0]),
        mag_saturation=0.0,
        magnetization=np.array([0.0, 0.0, 0.0]),
    ):
        self.eff_spin = Parameter("", eff_spin)
        self.curie_temp = Parameter("K", curie_temp)
        self.mf_exch_coupling = Parameter("m**2kg/s**2", 0.0)
        self.lamda = Parameter("", lamda)
        self.mag_moment = Parameter("bohr_magneton", mag_moment)
        self.aniso_exponent = Parameter("", aniso_exponent)
        self.anisotropy = Parameter("J/m**3", anisotropy)
        self.exch_stiffness = Parameter("J/m", exch_stiffness)
        self.mag_saturation = Parameter("J/T/m**3", mag_saturation)
        self.magnetization = VectorParameter("", magnetization)

        self._update_depending()

        # automatically set the name of the parameters
        for name, p in vars(self).items():
            p.name = name
            if name in ["eff_spin", "curie_temp", "exch_stiffness", "anisotropy"]:
                p._caller = self

    def _update_depending(self):
        self.calc_mf_exchange_coupling()
        self.check_array(self.exch_stiffness, "Exchange stiffness")
        self.check_array(self.anisotropy, "Anisotropy")

    def calc_mf_exchange_coupling(self):
        r"""calc_mf_exchange_coupling

        Calculate the mean-field exchange coupling constant

        .. math:: J = \frac{3}{S_{eff}+1} k_B T_C

        """
        try:
            self.mf_exch_coupling.magnitude = (
                3
                * self.eff_spin.magnitude
                / (self.eff_spin.magnitude + 1)
                * constants.k
                * self.curie_temp.magnitude
            )
        except AttributeError:
            # on initialization self.curie_temp
            self.mf_exch_coupling.magnitude = 0

    def check_array(self, parameter, name):
        unit = parameter.unit
        value = np.asarray(parameter.magnitude)

        if value.ndim == 0:
            value = np.full(3, value)
        elif value.shape == (1,):
            value = np.full(3, value[0])
        elif value.shape != (3,):
            raise ValueError(
                f"{name} must be a scalar or a vector of length 3!"
            )

        parameter.quantity = u.Quantity(value, unit)
