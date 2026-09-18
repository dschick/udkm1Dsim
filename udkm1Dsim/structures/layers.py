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

__all__ = ["Layer", "Vacuum", "AmorphousLayer", "UnitCell"]

__docformat__ = "restructuredtext"

from inspect import isfunction

import numpy as np
import pint
from sympy import lambdify, symbols
from tabulate import tabulate

from udkm1Dsim.structures.parameters import (
    ElasticParameters,
    MagneticParameters,
    OpticalParameters,
    StructuralParameters,
    ThermalParameters,
)

from .atoms import Atom, AtomMixed

u = pint.get_application_registry()
Q_ = u.Quantity


class Layer:
    r"""Layer

    A layer consists of structural, thermal, elastic, optical, and magnetic properties.
    These properties are organized into dedicated parameter groups:

    * :class:`StructuralParameters`
    * :class:`ThermalParameters`
    * :class:`ElasticParameters`
    * :class:`OpticalParameters`
    * :class:`MagneticParameters`

    The parameter groups are accessible through the corresponding
    attributes of the layer.

    Args:
        id (str): id of the layer.
        name (str): name of the layer.

    Attributes:
        id (str): id of the layer.
        name (str): name of the layer.

    """

    def __init__(self, id, name, **kwargs):
        self.id = id
        self.name = name

        self.structural = StructuralParameters()
        self.thermal = ThermalParameters()
        self.elastic = ElasticParameters()
        self.optical = OpticalParameters()
        self.magnetic = MagneticParameters()

        # structural parameters
        self.thickness = kwargs.get("thickness", 0.0 * u.nm)
        self.roughness = kwargs.get("roughness", 0.0 * u.nm)
        self.density = kwargs.get("density", 0.0 * u.kg / u.m**3)

        # thermal parameters
        self.heat_capacity = kwargs.get("heat_capacity", 0.0 * u.J / u.kg / u.K)
        self.therm_cond = kwargs.get("therm_cond", 0.0 * u.W / u.m / u.K)
        self.lin_therm_exp = kwargs.get("lin_therm_exp", 0.0)
        self.sub_system_coupling = kwargs.get("sub_system_coupling", 0.0 * u.W / u.m**3)
        self.num_sub_systems = 1

        # if len(self.heat_capacity) == len(self.therm_cond) \
        #         == len(self.lin_therm_exp) == len(self.sub_system_coupling):
        #     self.num_sub_systems = len(self.heat_capacity)
        # else:
        #     raise ValueError('Heat capacity, thermal conductivity, linear '
        #                      'thermal expansion, and subsystem coupling have not '
        #                      'the same number of elements!')

        # elastic parameters
        self.sound_vel = kwargs.get("sound_vel", 0.0 * u.m / u.s)
        self.phonon_damping = kwargs.get("phonon_damping", 0.0 * u.kg / u.s)

        # optical parameters
        self.deb_wal_fac = kwargs.get("deb_wal_fac", 0.0 * u.angstrom**2)
        self.opt_pen_depth = kwargs.get("opt_pen_depth", 0.0 * u.nm)
        self.opt_ref_index = kwargs.get("opt_ref_index", 0.0 + 0.0j)
        self.opt_ref_index_per_strain = kwargs.get("opt_ref_index_per_strain", 0.0 + 0.0j)

        # magnetic parameters
        self.eff_spin = kwargs.get("eff_spin", 0.0)
        self.curie_temp = kwargs.get("curie_temp", 0.0 * u.K)
        self.lamda = kwargs.get("lamda", 0.0)
        self.mag_moment = kwargs.get("mag_moment", 0.0 * u.bohr_magneton)
        self.aniso_exponent = kwargs.get("aniso_exponent", 0.0)
        self.anisotropy = kwargs.get("anisotropy", [0.0, 0.0, 0.0] * u.J / u.m**3)
        self.exch_stiffness = kwargs.get("exch_stiffness", 0.0 * u.J / u.m)
        self.mag_saturation = kwargs.get("mag_saturation", 0.0 * u.J / u.T / u.m**3)
        self.magnetization = kwargs.get("magnetization", np.array([0.0, 0.0, 0.0]))

    def __repr__(self):
        """String representation of this class"""
        class_str = f"Layer: {self.name}\nID: {self.id}\n" + "=" * 30 + "\n"
        class_str += self.structural.__repr__() + "\n"
        class_str += self.thermal.__repr__() + "\n"
        class_str += self.elastic.__repr__() + "\n"
        class_str += self.optical.__repr__() + "\n"
        class_str += self.magnetic.__repr__() + "\n"

        return class_str

    def _repr_html_(self):
        """HTML representation of this class"""

        class_str = f"<h2>Layer: {self.name}</h2><b>ID:</b> <i>{self.id}</i><br>"
        class_str += self.structural._repr_html_()
        class_str += self.thermal._repr_html_()
        class_str += self.elastic._repr_html_()
        class_str += self.optical._repr_html_()
        class_str += self.magnetic._repr_html_()

        return class_str

    # def get_property_dict(self, **kwargs):
    #     """get_property_dict

    #     Returns a dictionary with all parameters. objects or dicts and
    #     objects are converted to strings. if a type is given, only these
    #     properties are returned.

    #     Args:
    #         **kwargs (list[str]): types of requested properties.

    #     Returns:
    #         R (dict): dictionary with requested properties.

    #     """
    #     # initialize input parser and define defaults and validators
    #     properties_by_types = {'heat': ['_thickness', '_mass_unit_area', '_density',
    #                                     '_opt_pen_depth', 'opt_ref_index',
    #                                     'therm_cond_str', 'heat_capacity_str',
    #                                     'int_heat_capacity_str', 'sub_system_coupling_str',
    #                                     'num_sub_systems'],
    #                          'phonon': ['num_sub_systems', 'int_lin_therm_exp_str', '_thickness',
    #                                       '_mass_unit_area', 'spring_const', '_phonon_damping'],
    #                            'xray': ['num_atoms', '_area', '_mass', 'deb_wal_fac_str',
    #                                     '_thickness'],
    #                            'optical': ['_c_axis', '_opt_pen_depth', 'opt_ref_index',
    #                                        'opt_ref_index_per_strain'],
    #                            'magnetic': ['_thickness', 'magnetization', 'eff_spin',
    #                                         '_curie_temp', '_aniso_exponents', '_anisotropy',
    #                                         '_exch_stiffness', '_mag_saturation', 'lamda'],
    #                            }

    #     types = (kwargs.get('types', 'all'))
    #     if type(types) is not list:
    #         types = [types]
    #     attrs = vars(self)
    #     R = {}
    #     for t in types:
    #         # define the property names by the given type
    #         if t == 'all':
    #             return attrs
    #         else:
    #             S = dict((key, value) for key, value in attrs.items()
    #                      if key in properties_by_types[t])
    #             R.update(S)

    #     return R

    # ============================================================================
    # Structural parameters
    # ============================================================================

    @property
    def thickness(self):
        return self.structural.thickness.quantity

    @thickness.setter
    def thickness(self, value):
        self.structural.thickness.quantity = value
        self.elastic.calc_spring_const(
            self.structural.mass_unit_area.magnitude, self.structural.thickness.magnitude
        )
        self.elastic.calc_acoustic_impedance(
            self.structural.mass.magnitude, self.structural.area.magnitude
        )

    @property
    def mass(self):
        return self.structural.mass.quantity

    @mass.setter
    def mass(self, value):
        raise AttributeError(
            "'mass' is derived from density and volume and cannot be set directly."
        )

    @property
    def mass_unit_area(self):
        return self.structural.mass_unit_area.quantity

    @mass_unit_area.setter
    def mass_unit_area(self, value):
        raise AttributeError(
            "'mass_unit_area' is derived from 'density and volume normalized to area "
            "and cannot be set directly. 'area' is fixed to 1.0 angstrom² for Layer"
        )

    @property
    def density(self):
        return self.structural.density.quantity

    @density.setter
    def density(self, value):
        self.structural.density.quantity = value
        self.elastic.calc_spring_const(
            self.structural.mass_unit_area.magnitude, self.structural.thickness.magnitude
        )

    @property
    def area(self):
        return self.structural.area.quantity

    @area.setter
    def area(self, value):
        raise AttributeError("'area' is fixed to 1.0 angstrom² for Layer")

    @property
    def volume(self):
        return self.structural.volume.quantity

    @volume.setter
    def volume(self, value):
        raise AttributeError(
            "'volume' is derived from 'thickness' and 'area' and cannot be set directly."
        )

    @property
    def roughness(self):
        return self.structural.roughness.quantity

    @roughness.setter
    def roughness(self, value):
        self.structural.roughness.quantity = value

    # ============================================================================
    # Thermal parameters
    # ============================================================================

    @property
    def therm_cond(self):
        return self.thermal.therm_cond.quantity

    @therm_cond.setter
    def therm_cond(self, value):
        self.thermal.therm_cond.quantity = value

    @property
    def heat_capacity(self):
        return self.thermal.heat_capacity.quantity

    @heat_capacity.setter
    def heat_capacity(self, value):
        self.thermal.heat_capacity.quantity = value

    @property
    def lin_therm_exp(self):
        return self.thermal.lin_therm_exp.quantity

    @lin_therm_exp.setter
    def lin_therm_exp(self, value):
        self.thermal.lin_therm_exp.quantity = value

    @property
    def int_lin_therm_exp(self):
        return self.thermal.int_lin_therm_exp.quantity

    @int_lin_therm_exp.setter
    def int_lin_therm_exp(self, value):
        raise AttributeError(
            "'int_lin_therm_exp' is automatically derived from lin_therm_exp. "
            "To set explicitly, modify 'Layer.thermal.int_lin_therm_exp' instead."
        )

    @property
    def int_heat_capacity(self):
        return self.thermal.int_heat_capacity.quantity

    @int_heat_capacity.setter
    def int_heat_capacity(self, value):
        raise AttributeError(
            "'int_heat_capacity' is automatically derived from heat_capacity. "
            "To set explicitly, modify 'Layer.thermal.int_heat_capacity' instead."
        )

    @property
    def sub_system_coupling(self):
        return self.thermal.sub_system_coupling.quantity

    @sub_system_coupling.setter
    def sub_system_coupling(self, value):
        self.thermal.sub_system_coupling.quantity = value

    @property
    def num_sub_systems(self):
        return self.thermal.num_sub_systems.quantity

    @num_sub_systems.setter
    def num_sub_systems(self, value):
        self.thermal.num_sub_systems.quantity = value

    # ============================================================================
    # Elastic parameters
    # ============================================================================

    @property
    def sound_vel(self):
        return self.elastic.sound_vel.quantity

    @sound_vel.setter
    def sound_vel(self, value):
        self.elastic.sound_vel.quantity = value
        self.elastic.calc_spring_const(
            self.structural.mass_unit_area.magnitude, self.structural.thickness.magnitude
        )
        self.elastic.calc_acoustic_impedance(
            self.structural.mass.magnitude, self.structural.area.magnitude
        )

    @property
    def spring_const(self):
        return self.elastic.spring_const.quantity

    @spring_const.setter
    def spring_const(self, value):
        raise AttributeError(
            "'spring_const' is automatically derived from sound_vel, mass, and thickness."
        )

    @property
    def phonon_damping(self):
        return self.elastic.phonon_damping.quantity

    @phonon_damping.setter
    def phonon_damping(self, value):
        self.elastic.phonon_damping.quantity = value

    # ============================================================================
    # Optical parameters
    # ============================================================================

    @property
    def opt_pen_depth(self):
        return self.optical.opt_pen_depth.quantity

    @opt_pen_depth.setter
    def opt_pen_depth(self, value):
        self.optical.opt_pen_depth.quantity = value

    @property
    def opt_ref_index(self):
        return self.optical.opt_ref_index.quantity

    @opt_ref_index.setter
    def opt_ref_index(self, value):
        self.optical.opt_ref_index.quantity = value

    @property
    def opt_ref_index_per_strain(self):
        return self.optical.opt_ref_index_per_strain.quantity

    @opt_ref_index_per_strain.setter
    def opt_ref_index_per_strain(self, value):
        self.optical.opt_ref_index_per_strain.quantity = value

    @property
    def deb_wal_fac(self):
        return self.optical.deb_wal_fac.quantity

    @deb_wal_fac.setter
    def deb_wal_fac(self, value):
        self.optical.deb_wal_fac.quantity = value

    # ============================================================================
    # Magnetic parameters
    # ============================================================================

    @property
    def eff_spin(self):
        return self.magnetic.eff_spin.quantity

    @eff_spin.setter
    def eff_spin(self, value):
        self.magnetic.eff_spin.quantity = value

    @property
    def curie_temp(self):
        return self.magnetic.curie_temp.quantity

    @curie_temp.setter
    def curie_temp(self, value):
        self.magnetic.curie_temp.quantity = value

    @property
    def mf_exch_coupling(self):
        return self.magnetic.mf_exch_coupling.quantity

    @mf_exch_coupling.setter
    def mf_exch_coupling(self, value):
        raise AttributeError(
            "'mf_exch_coupling' is derived from other 'eff_spin' and 'curie_temp' "
            " and cannot be set directly."
        )

    @property
    def lamda(self):
        return self.magnetic.lamda.quantity

    @lamda.setter
    def lamda(self, value):
        self.magnetic.lamda.quantity = value

    @property
    def mag_moment(self):
        return self.magnetic.mag_moment.quantity

    @mag_moment.setter
    def mag_moment(self, value):
        self.magnetic.mag_moment.quantity = value

    @property
    def aniso_exponent(self):
        return self.magnetic.aniso_exponent.quantity

    @aniso_exponent.setter
    def aniso_exponent(self, value):
        self.magnetic.aniso_exponent.quantity = value

    @property
    def anisotropy(self):
        return self.magnetic.anisotropy.quantity

    @anisotropy.setter
    def anisotropy(self, value):
        self.magnetic.anisotropy.quantity = value

    # @anisotropy.setter
    # def anisotropy(self, anisotropy):
    #     self._anisotropy = np.zeros(3)
    #     try:
    #         if len(anisotropy) == 3:
    #             self._anisotropy = anisotropy.to_base_units().magnitude
    #         else:
    #             warnings.warn('Anisotropy must be a scalar or vector of length 3!')
    #     except TypeError:
    #         self._anisotropy[0] = anisotropy.to_base_units().magnitude

    @property
    def exch_stiffness(self):
        return self.magnetic.exch_stiffness.quantity

    @exch_stiffness.setter
    def exch_stiffness(self, value):
        self.magnetic.exch_stiffness.quantity = value

    # @exch_stiffness.setter
    #     def exch_stiffness(self, exch_stiffness):
    #         self._exch_stiffness = np.zeros(3)
    #         try:
    #             if len(exch_stiffness) == 3:
    #                 self._exch_stiffness = exch_stiffness.to_base_units().magnitude
    #             else:
    #                 warnings.warn('Exchange stiffness must be a scalar or vector of length 3!')
    #         except TypeError:
    #             self._exch_stiffness[:] = exch_stiffness.to_base_units().magnitude

    @property
    def mag_saturation(self):
        return self.magnetic.mag_saturation.quantity

    @mag_saturation.setter
    def mag_saturation(self, value):
        self.magnetic.mag_saturation.quantity = value

    @property
    def magnetization(self):
        return self.magnetic.magnetization.quantity

    @magnetization.setter
    def magnetization(self, value):
        self.magnetic.magnetization.quantity = value

    # @property
    # def magnetization(self):
    #     return {'amplitude': self._magnetization['amplitude'],
    #             'phi': Q_(self._magnetization['phi'], u.rad).to('deg'),
    #             'gamma': Q_(self._magnetization['gamma'], u.rad).to('deg')
    #             }

    # @magnetization.setter
    # def magnetization(self, magnetization):
    #     self._magnetization = {'amplitude': magnetization['amplitude'],
    #                            'phi': magnetization['phi'].to_base_units().magnitude,
    #                            'gamma': magnetization['gamma'].to_base_units().magnitude
    #                            }


class Vacuum(Layer):
    def __init__(self, thickness=1 * u.nm, **kwargs):
        self.thickness = thickness
        self.density = 0.0 * u.kg / u.m**3
        self.area = 1.0 * u.angstrom**2  # set as unit area
        self.volume = self.area * self.thickness
        self.mass = 0 * u.kg
        self.mass_unit_area = self.mass
        super().__init__("vacuum", "vacuum", opt_ref_index=1 + 0.0j)

    def __str__(self):
        """String representation of this class"""
        return f"Vacuum layer of thickness: {self.thickness:.4g~P}"


class AmorphousLayer(Layer):
    r"""AmorphousLayer

    Representation of amorphous layers containing an Atom or AtomMixed.

    Args:
        id (str): id of the layer.
        name (str): name of layer.
        thickness (float): thickness of the layer.
        density (float): density of the layer.

    Keyword Args:
        atom (object): Atom or AtomMixed in the layer.
        roughness (float): gaussian width of the top roughness of a layer.
        deb_wal_fac (float): Debye Waller factor.
        sound_vel (float): sound velocity.
        phonon_damping (float): phonon damping.
        roughness (float): gaussian width of the top roughness of a layer.
        opt_pen_depth (float): optical penetration depth.
        opt_ref_index (float): refractive index.
        opt_ref_index_per_strain (float): change of refractive index per
           strain.
        heat_capacity (float): heat capacity.
        therm_cond (float): thermal conductivity.
        lin_therm_exp (float): linear thermal expansion.
        sub_system_coupling (float): sub-system coupling.

    Attributes:
        id (str): id of the layer.
        name (str): name of the layer.
        thickness (float): thickness of the layer [m].
        mass (float): mass of the layer [kg].
        mass_unit_area (float): mass of layer normalized to unit area of 1 Å² [kg].
        density (float): density of the layer [kg/m³].
        area (float): area of layer [m²].
        volume (float): volume of layer [m³].
        roughness (float): gaussian width of the top roughness of a layer [m].
        deb_wal_fac (list[@lambda]): list of T-dependent Debye-Waller factors
            `\langle u^2\rangle` [m²].
        sound_vel (float): longitudinal sound velocity in the layer [m/s].
        spring_const (ndarray[float]): spring constant of the layer [kg/s²]
            and higher orders.
        phonon_damping (float): damping constant of phonon propagation [kg/s].
        opt_pen_depth (float): optical penetration depth of the layer [m].
        opt_ref_index (ndarray[float]): optical refractive index - real
           and imagenary part :math:`n + i\kappa`.
        opt_ref_index_per_strain (ndarray[float]): optical refractive
           index change per strain - real and imagenary part
           :math:`\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}`.
        therm_cond (list[@lambda]): list of HANDLES T-dependent thermal
           conductivity [W/(m K)].
        lin_therm_exp (list[@lambda]): list of T-dependent linear thermal
           expansion coefficient (relative).
        int_lin_therm_exp (list[@lambda]): list of T-dependent integrated
           linear thermal expansion coefficient.
        heat_capacity (list[@lambda]): list of T-dependent heat capacity
           function [J/(kg K)].
        int_heat_capacity (list[@lambda]): list of T-dependent integrated heat
           capacity function.
        sub_system_coupling (list[@lambda]): list of of coupling functions of
           different subsystems [W/m³].
        num_sub_systems (int): number of subsystems for heat and phonons
           (electrons, lattice, spins, ...).
        eff_spin (float): effective spin.
        curie_temp (float): Curie temperature [K].
        mf_exch_coupling (float): mean field exchange coupling constant [m²kg/s²].
        lamda (float): intrinsic coupling to bath parameter.
        mag_moment (float): atomic magnetic moment [mu_Bohr].
        aniso_exponent(ndarray[float]): exponent of T-dependence uniaxial
            anisotropy.
        anisotropy (float): anisotropy at T=0 K [J/m³] as x,y,z component vector.
        exch_stiffness (float): exchange stiffness at T=0 K [J/m].
        mag_saturation (float): saturation magnetization at 0 K [J/T/m³].
        magnetization (dict[float]): magnetization amplitude, phi and
           gamma angle inherited from the atom.
        atom (object): Atom or AtomMixed in the layer.

    """

    def __init__(self, id, name, thickness, density, **kwargs):
        self.thickness = thickness
        self.density = density
        self.area = 1.0 * u.angstrom**2  # set as unit area
        self.volume = self.area * self.thickness
        self.mass = self.density * self.volume
        self.mass_unit_area = self.mass
        self.atom = kwargs.get("atom", [])
        super().__init__(id, name, **kwargs)

    def __str__(self):
        """String representation of this class"""
        output = [
            ["id", self.id],
            ["name", self.name],
            ["thickness", f"{self.thickness:.4g~P}"],
        ]
        output += super().__str__()

        try:
            output += [
                ["atom", self.atom.name],
                ["magnetization", ""],
                ["amplitude", self.magnetization["amplitude"]],
                ["phi [°]", "{:.4g~P}".format(self.magnetization["phi"].to("deg"))],
                ["gamma [°]", "{:.4g~P}".format(self.magnetization["gamma"].to("deg"))],
            ]
        except AttributeError:
            output += [["no atom set", ""]]

        class_str = "Amorphous layer with the following properties\n\n"
        class_str += tabulate(
            output,
            headers=["parameter", "value"],
            tablefmt="rst",
            colalign=("right",),
            floatfmt=(".2f", ".2f"),
        )
        return class_str

    @property
    def atom(self):
        return self._atom

    @atom.setter
    def atom(self, atom):
        if atom == []:  # no atom is set
            self.magnetization = {
                "amplitude": 0,
                "phi": 0 * u.deg,
                "gamma": 0 * u.deg,
            }
            return

        if not isinstance(atom, (Atom, AtomMixed)):
            raise TypeError(
                "Class "
                + type(atom).__name__
                + " is no possible atom of an amorphous layer. "
                + "Only Atom and AtomMixed are allowed!"
            )
        self._atom = atom
        self.magnetization = {
            "amplitude": atom.mag_amplitude,
            "phi": atom.mag_phi,
            "gamma": atom.mag_gamma,
        }

    @property
    def magnetization(self):
        return {
            "amplitude": self._magnetization["amplitude"],
            "phi": Q_(self._magnetization["phi"], u.rad).to("deg"),
            "gamma": Q_(self._magnetization["gamma"], u.rad).to("deg"),
        }

    @magnetization.setter
    def magnetization(self, magnetization):
        self._magnetization = {
            "amplitude": magnetization["amplitude"],
            "phi": magnetization["phi"].to_base_units().magnitude,
            "gamma": magnetization["gamma"].to_base_units().magnitude,
        }


class UnitCell(Layer):
    r"""UnitCell

    Representation of unit cells made of one or multiple Atom or AtomMixed
    instances at defined positions.

    Args:
        id (str): id of the UnitCell.
        name (str): name of the UnitCell.
        c_axis (float): c-axis of the UnitCell.

    Keyword Args:
        a_axis (float): a-axis of the UnitCell.
        b_axis (float): b-axis of the UnitCell.
        deb_wal_fac (float): Debye Waller factor.
        sound_vel (float): sound velocity.
        phonon_damping (float): phonon damping.
        roughness (float): gaussian width of the top roughness of a layer.
        opt_pen_depth (float): optical penetration depth.
        opt_ref_index (float): refractive index.
        opt_ref_index_per_strain (float): change of refractive index per
           strain.
        heat_capacity (float): heat capacity.
        therm_cond (float): thermal conductivity.
        lin_therm_exp (float): linear thermal expansion.
        sub_system_coupling (float): sub-system coupling.

    Attributes:
        id (str): id of the layer.
        name (str): name of the layer.
        c_axis (float): out-of-plane c-axis [m].
        a_axis (float): in-plane a-axis [m].
        b_axis (float): in-plane b-axis [m].
        thickness (float): thickness of the layer [m].
        mass (float): mass of the layer [kg].
        mass_unit_area (float): mass of layer normalized to unit area of 1 Å² [kg].
        density (float): density of the layer [kg/m³].
        area (float): area of layer [m²].
        volume (float): volume of layer [m³].
        roughness (float): gaussian width of the top roughness of a layer [m].
        deb_wal_fac (list[@lambda]): list of T-dependent Debye-Waller factors
            `\langle u^2\rangle` [m²].
        sound_vel (float): longitudinal sound velocity in the layer [m/s].
        spring_const (ndarray[float]): spring constant of the layer [kg/s²]
            and higher orders.
        phonon_damping (float): damping constant of phonon propagation [kg/s].
        opt_pen_depth (float): optical penetration depth of the layer [m].
        opt_ref_index (ndarray[float]): optical refractive index - real
           and imagenary part :math:`n + i\kappa`.
        opt_ref_index_per_strain (ndarray[float]): optical refractive
           index change per strain - real and imagenary part
           :math:`\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}`.
        therm_cond (list[@lambda]): list of HANDLES T-dependent thermal
           conductivity [W/(m K)].
        lin_therm_exp (list[@lambda]): list of T-dependent linear thermal
           expansion coefficient (relative).
        int_lin_therm_exp (list[@lambda]): list of T-dependent integrated
           linear thermal expansion coefficient.
        heat_capacity (list[@lambda]): list of T-dependent heat capacity
           function [J/(kg K)].
        int_heat_capacity (list[@lambda]): list of T-dependent integrated heat
           capacity function.
        sub_system_coupling (list[@lambda]): list of of coupling functions of
           different subsystems [W/m³].
        num_sub_systems (int): number of subsystems for heat and phonons
           (electrons, lattice, spins, ...).
        atoms (list[atom, @lambda]): list of atoms and function handle
           for strain dependent displacement.
        num_atoms (int): number of atoms in unit cell.
        eff_spin (float): effective spin.
        curie_temp (float): Curie temperature [K].
        mf_exch_coupling (float): mean field exchange coupling constant [m²kg/s²].
        lamda (float): intrinsic coupling to bath parameter.
        mag_moment (float): atomic magnetic moment [mu_Bohr].
        aniso_exponent(ndarray[float]): exponent of T-dependence uniaxial
            anisotropy.
        anisotropy (float): anisotropy at T=0 K [J/m³] as x,y,z component vector.
        exch_stiffness (float): exchange stiffness at T=0 K [J/m].
        mag_saturation (float): saturation magnetization at 0 K [J/T/m³].
        magnetization (list[float]): magnetization amplitudes, phi, and
           gamma angle of each atom in the unit cell.

    """

    def __init__(self, id, name, c_axis, **kwargs):
        self.c_axis = c_axis
        self.thickness = c_axis
        self.a_axis = kwargs.get("a_axis", self.c_axis)
        self.b_axis = kwargs.get("b_axis", self.a_axis)
        self.mass = 0.0 * u.kg
        self.mass_unit_area = 0.0 * u.kg
        self.density = 0.0 * u.kg / u.m**2

        super().__init__(id, name, **kwargs)

        self.area = self.a_axis * self.b_axis
        self.volume = self.area * self.c_axis
        self.atoms = []
        self.num_atoms = 0
        self.magnetization = []

    def __str__(self):
        """String representation of this class"""
        output = [
            ["id", self.id],
            ["name", self.name],
            ["a-axis", "{:.4g~P}".format(self.a_axis.to("nm"))],
            ["b-axis", "{:.4g~P}".format(self.b_axis.to("nm"))],
            ["c-axis", "{:.4g~P}".format(self.c_axis.to("nm"))],
            ["area", "{:.4g~P}".format(self.area.to("nm**2"))],
            ["volume", "{:.4g~P}".format(self.volume.to("nm**3"))],
            ["mass", "{:.4g~P}".format(self.mass.to("kg"))],
            ["mass per unit area", f"{self.mass_unit_area:.4g~P}"],
        ]
        output += super().__str__()

        class_str = "Unit Cell with the following properties\n\n"
        class_str += tabulate(
            output,
            headers=["parameter", "value"],
            tablefmt="rst",
            colalign=("right",),
            floatfmt=(".2f", ".2f"),
        )
        class_str += "\n\n" + str(self.num_atoms) + " Constituents:\n"

        atoms_str = []
        for i in range(self.num_atoms):
            atoms_str.append(
                [
                    self.atoms[i][0].name,
                    f"{self.atoms[i][1](0):0.2f}",
                    self.atoms[i][2],
                    "",
                    self.atoms[i][0].mag_amplitude,
                    self.atoms[i][0].mag_phi.to("deg").magnitude,
                    self.atoms[i][0].mag_gamma.to("deg").magnitude,
                ]
            )
        class_str += tabulate(
            atoms_str,
            headers=[
                "atom",
                "position",
                "position function",
                "magn.",
                "amplitude",
                "phi [°]",
                "gamma [°]",
            ],
            tablefmt="rst",
        )
        return class_str

    def visualize(self, block=True, **kwargs):
        """visualize

        Allows for 3D presentation of unit cell by allow for a & b
        coordinate of atoms.

        Todo:
            use the avogadro project as plugin
        Todo:
            create unit cell from CIF file e.g. by xrayutilities plugin.
        Todo:
            visualize magnetization per atom

        Args:
            **kwargs (str): strain for manipulating unit cell visualization.

        """
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        strain = kwargs.get("strain", 0)

        colors = [colormaps["Set3"](x) for x in np.linspace(0, 1, self.num_atoms)]
        atom_ids = self.get_atom_ids()

        plt.figure()
        atoms_plotted = np.zeros_like(atom_ids)
        for j in range(self.num_atoms):
            if not atoms_plotted[atom_ids.index(self.atoms[j][0].id)]:
                label = self.atoms[j][0].id
                atoms_plotted[atom_ids.index(self.atoms[j][0].id)] = True
                plt.plot(
                    1 + j,
                    self.atoms[j][1](strain),
                    "o",
                    markersize=10,
                    markeredgecolor=[0, 0, 0],
                    markerfacecolor=colors[atom_ids.index(self.atoms[j][0].id)],
                    label=label,
                )
            else:
                label = "_nolegend_"
                plt.plot(
                    1 + j,
                    self.atoms[j][1](strain),
                    "o",
                    markersize=10,
                    markeredgecolor=[0, 0, 0],
                    markerfacecolor=colors[atom_ids.index(self.atoms[j][0].id)],
                    label=label,
                )

        plt.axis([0.1, self.num_atoms + 0.9, -0.1, (1.1 + strain)])
        plt.grid(True)

        plt.title(f"Strain: {strain:0.2f}%")
        plt.ylabel("relative Position")
        plt.xlabel("# Atoms")
        plt.legend()
        plt.show(block=block)

    def add_atom(self, atom, position):
        r"""add_atom

        Adds an AtomBase/AtomMixed at a relative position of the unit cell.

        Sort the list of atoms by the position at zero strain.

        Update the mass, density and spring constant of the unit cell
        automatically:

        .. math:: \kappa = m \cdot (v_s / c)^2

        Args:
            atom (Atom, AtomMixed): Atom or AtomMixed added to unit cell.
            position (float): relative position within unit cel [0 .. 1].

        """
        s = symbols("s")
        position_str = ""
        # test the input type of the position
        if isfunction(position):
            raise ValueError("Please use string representation of function!")
        elif isinstance(position, str):
            try:
                # backwards compatibility for direct lambda definition
                if ":" in position:
                    # strip lambda prefix
                    position = position.split(":")[1]
                position_str = position.strip()
                position = lambdify(s, position, modules="numpy")
            except Exception as e:
                print(
                    "String input for unit cell property "
                    + position
                    + " \
                    cannot be converted to function handle!"
                )
                print(e)
        elif isinstance(position, (int, float)):
            position_str = str(position) + "*(1+s)"
            position = lambdify(s, position_str, modules="numpy")
        else:
            raise ValueError(
                "Atom position input has to be a scalar, or string"
                "which can be converted into a lambda function!"
            )

        # add the atom at the end of the array
        self.atoms.append([atom, position, position_str])
        # sort list of atoms by position at zero strain
        self.atoms.sort(key=lambda x: x[1](0))
        # increase the number of atoms
        self.num_atoms = self.num_atoms + 1

        self.magnetization.append([atom.mag_amplitude, atom.mag_phi, atom.mag_gamma])

        self.mass = 0 * u.kg
        for i in range(self.num_atoms):
            self.mass = self.mass + self.atoms[i][0].mass

        self.density = self.mass / self.volume
        # set mass per unit area
        self.mass_unit_area = self.mass * 1 * u.angstrom**2 / self.area
        self.calc_spring_const()

    def add_multiple_atoms(self, atom, position, Nb):
        """add_multiple_atoms

        Adds multiple AtomBase/AtomMixed at a relative position of the
        unit cell.

        Args:
            atom (Atom, AtomMixed): Atom or AtomMixed added to unit cell.
            position (float): relative position within unit cel [0 .. 1].
            Nb (int): repetition of atoms.

        """
        for _ in range(int(Nb)):
            self.add_atom(atom, position)

    def get_atom_ids(self):
        """get_atom_ids

        Provides a list of atom ids within the unit cell.

        Returns:
            ids (list[str]): list of atom ids within unit cell

        """
        ids = []
        for i in range(self.num_atoms):
            if self.atoms[i][0].id not in ids:
                ids.append(self.atoms[i][0].id)

        return ids

    def get_atom_positions(self, *args):
        """get_atom_positions

        Calculates the relative positions of the atoms in the unit cell

        Returns:
            res (ndarray[float]): relative postion of the atoms within the unit
            cell.

        """
        if args:
            strain = float(np.asarray(args[0]).item())
        else:
            strain = 0.0

        res = np.zeros([self.num_atoms])
        for i, atom in enumerate(self.atoms):
            res[i] = atom[1](strain)

        return res

    @property
    def a_axis(self):
        return Q_(self._a_axis, u.meter).to("nm")

    @a_axis.setter
    def a_axis(self, a_axis):
        self._a_axis = a_axis.to_base_units().magnitude

    @property
    def b_axis(self):
        return Q_(self._b_axis, u.meter).to("nm")

    @b_axis.setter
    def b_axis(self, b_axis):
        self._b_axis = b_axis.to_base_units().magnitude

    @property
    def c_axis(self):
        return Q_(self._c_axis, u.meter).to("nm")

    @c_axis.setter
    def c_axis(self, c_axis):
        self._c_axis = c_axis.to_base_units().magnitude
