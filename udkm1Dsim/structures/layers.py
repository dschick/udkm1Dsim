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

from udkm1Dsim.structures.parameter_groups import (
    ElasticParameters,
    LatticeParameters,
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

        self.structural = StructuralParameters(
            thickness=kwargs.get("thickness", 0.0 * u.nm),
            roughness=kwargs.get("roughness", 0.0 * u.nm),
            density=kwargs.get("density", 0.0 * u.kg / u.m**3),
        )
        self.thermal = ThermalParameters(
            heat_capacity=kwargs.get("heat_capacity", 0.0 * u.J / u.kg / u.K),
            therm_cond=kwargs.get("therm_cond", 0.0 * u.W / u.m / u.K),
            lin_therm_exp=kwargs.get("lin_therm_exp", 0.0),
            sub_system_coupling=kwargs.get("sub_system_coupling", 0.0 * u.W / u.m**3),
            deb_wal_fac=kwargs.get("deb_wal_fac", 0.0 * u.angstrom**2),
        )
        self.elastic = ElasticParameters(
            sound_vel=kwargs.get("sound_vel", 0.0 * u.m / u.s),
            phonon_damping=kwargs.get("phonon_damping", 0.0 * u.kg / u.s),
        )
        self.optical = OpticalParameters(
            opt_pen_depth=kwargs.get("opt_pen_depth", 0.0 * u.nm),
            opt_ref_index=kwargs.get("opt_ref_index", 0.0 + 0.0j),
            opt_ref_index_per_strain=kwargs.get("opt_ref_index_per_strain", 0.0 + 0.0j),
        )
        self.magnetic = MagneticParameters(
            eff_spin=kwargs.get("eff_spin", 0.0),
            curie_temp=kwargs.get("curie_temp", 0.0 * u.K),
            lamda=kwargs.get("lamda", 0.0),
            mag_moment=kwargs.get("mag_moment", 0.0 * u.bohr_magneton),
            aniso_exponent=kwargs.get("aniso_exponent", 0.0),
            anisotropy=kwargs.get("anisotropy", [0.0, 0.0, 0.0] * u.J / u.m**3),
            exch_stiffness=kwargs.get("exch_stiffness", 0.0 * u.J / u.m),
            mag_saturation=kwargs.get("mag_saturation", 0.0 * u.J / u.T / u.m**3),
            magnetization=kwargs.get("magnetization", np.array([0.0, 0.0, 0.0])),
        )

        # calc depending parameters across ParameterGroups
        self.elastic.calc_spring_const(
            self.structural.mass_unit_area.magnitude, self.structural.thickness.magnitude
        )
        self.elastic.calc_acoustic_impedance(
            self.structural.mass.magnitude, self.structural.area.magnitude
        )

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

    def get_property_dict(self, **kwargs):
        """get_property_dict

        Returns a dictionary with all parameters. objects or dicts and
        objects are converted to strings. if a type is given, only these
        properties are returned.

        Args:
            **kwargs (list[str]): types of requested properties.

        Returns:
            R (dict): dictionary with requested properties.

        """
        # initialize input parser and define defaults and validators
        properties_by_types = {
            "heat": [
                "_thickness",
                "_mass_unit_area",
                "_density",
                "_opt_pen_depth",
                "opt_ref_index",
                "therm_cond_str",
                "heat_capacity_str",
                "int_heat_capacity_str",
                "sub_system_coupling_str",
                "num_sub_systems",
            ],
            "phonon": [
                "num_sub_systems",
                "int_lin_therm_exp_str",
                "_thickness",
                "_mass_unit_area",
                "spring_const",
                "_phonon_damping",
            ],
            "xray": ["num_atoms", "_area", "_mass", "deb_wal_fac_str", "_thickness"],
            "optical": ["_c_axis", "_opt_pen_depth", "opt_ref_index", "opt_ref_index_per_strain"],
            "magnetic": [
                "_thickness",
                "magnetization",
                "eff_spin",
                "_curie_temp",
                "_aniso_exponents",
                "_anisotropy",
                "_exch_stiffness",
                "_mag_saturation",
                "lamda",
            ],
        }

        types = kwargs.get("types", "all")
        if type(types) is not list:
            types = [types]
        attrs = vars(self)
        R = {}
        for t in types:
            # define the property names by the given type
            if t == "all":
                return attrs
            else:
                S = dict(
                    (key, value) for key, value in attrs.items() if key in properties_by_types[t]
                )
                R.update(S)

        return R

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
        return self.thermal.therm_cond.functional

    @therm_cond.setter
    def therm_cond(self, value):
        self.thermal.therm_cond.quantity = value

    @property
    def therm_cond_expr(self):
        return self.thermal.therm_cond.quantity

    @property
    def heat_capacity(self):
        return self.thermal.heat_capacity.functional

    @heat_capacity.setter
    def heat_capacity(self, value):
        self.thermal.heat_capacity.quantity = value

    @property
    def heat_capacity_expr(self):
        return self.thermal.heat_capacity.quantity

    @property
    def lin_therm_exp(self):
        return self.thermal.lin_therm_exp.functional

    @lin_therm_exp.setter
    def lin_therm_exp(self, value):
        self.thermal.lin_therm_exp.quantity = value

    @property
    def lin_therm_exp_expr(self):
        return self.thermal.lin_therm_exp.quantity

    @property
    def int_lin_therm_exp(self):
        return self.thermal.lin_therm_exp.integral

    @int_lin_therm_exp.setter
    def int_lin_therm_exp(self, value):
        raise AttributeError(
            "'int_lin_therm_exp' is automatically derived from lin_therm_exp. "
            "To set explicitly, modify 'Layer.thermal.int_lin_therm_exp' instead."
        )

    @property
    def int_lin_therm_exp_expr(self):
        return self.thermal.lin_therm_exp.integral_expr

    @property
    def int_heat_capacity(self):
        return self.thermal.heat_capacity.integral

    @int_heat_capacity.setter
    def int_heat_capacity(self, value):
        raise AttributeError(
            "'int_heat_capacity' is automatically derived from heat_capacity. "
            "To set explicitly, modify 'Layer.thermal.int_heat_capacity' instead."
        )

    @property
    def int_heat_capacity_expr(self):
        return self.thermal.heat_capacity.integral_expr

    @property
    def sub_system_coupling(self):
        return self.thermal.sub_system_coupling.functional

    @sub_system_coupling.setter
    def sub_system_coupling(self, value):
        self.thermal.sub_system_coupling.quantity = value

    @property
    def deb_wal_fac(self):
        return self.thermal.deb_wal_fac.functional

    @deb_wal_fac.setter
    def deb_wal_fac(self, value):
        self.thermal.deb_wal_fac.quantity = value

    @property
    def deb_wal_fac_expr(self):
        return self.thermal.deb_wal_fac.quantity

    @property
    def num_sub_systems(self):
        return self.thermal.num_sub_systems.magnitude

    @num_sub_systems.setter
    def num_sub_systems(self, value):
        self.thermal.num_sub_systems.magnitude = value

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

        super().__init__("vacuum", "vacuum", opt_ref_index=1 + 0.0j)
        self.thickness = thickness
        self.density = 0.0 * u.kg / u.m**3

    def __str__(self):
        """String representation of this class"""
        return f"Vacuum layer of thickness: {self.thickness:.4g~P}"


class AmorphousLayer(Layer):
    r"""AmorphousLayer

    Representation of amorphous layers but kept for backwards compatibility.

    Use :class:`Layer` instead.

    """
    def __init__(self, id, name, thickness, density, **kwargs):
        super().__init__(id, name, thickness=thickness, density=density, **kwargs)


class UnitCell(Layer):
    r"""Layer

        Representation of unit cells made of one or multiple Atom or AtomMixed
        instances at defined positions.
        A unit cell consists of structural, lattice, thermal, elastic, optical, and magnetic properties.
        These properties are organized into dedicated parameter groups:

        * :class:`StructuralParameters`
        * :class:`LatticeParameters`
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

    def __init__(self, id, name, c_axis, **kwargs):
        super().__init__(id, name, **kwargs)

        self.lattice = LatticeParameters(
            a_axis=kwargs.get("a_axis", 0.0 * u.angstrom),
            b_axis=kwargs.get("b_axis", 0.0 * u.angstrom),
            c_axis=kwargs.get("c_axis", 0.0 * u.angstrom),
        )

        self.thickness = c_axis

        self.structural.area.quantity = self.a_axis * self.b_axis
        self.structural.volume.quantity = self.area * self.c_axis
        self.atoms = []
        self.num_atoms = 0

    def __repr__(self):
        """String representation of this class"""
        class_str = f"UnitCell: {self.name}\nID: {self.id}\n" + "=" * 30 + "\n"
        class_str += self.structural.__repr__() + "\n"
        class_str += self.lattice.__repr__() + "\n"
        class_str += self.thermal.__repr__() + "\n"
        class_str += self.elastic.__repr__() + "\n"
        class_str += self.optical.__repr__() + "\n"
        class_str += self.magnetic.__repr__() + "\n"

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
            tablefmt="double_grid",
        )

        return class_str

    def _repr_html_(self):
        """HTML representation of this class"""

        class_str = f"<h2>UnitCell: {self.name}</h2><b>ID:</b> <i>{self.id}</i><br>"
        class_str += self.structural._repr_html_()
        class_str += self.lattice._repr_html_()
        class_str += self.thermal._repr_html_()
        class_str += self.elastic._repr_html_()
        class_str += self.optical._repr_html_()
        class_str += self.magnetic._repr_html_()

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
        class_str += "<h3>Atoms</h3>" + tabulate(
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
            tablefmt="html",
        )

        return class_str

    # def __str__(self):
    #     """String representation of this class"""
    #     output = [
    #         ["id", self.id],
    #         ["name", self.name],
    #         ["a-axis", "{:.4g~P}".format(self.a_axis.to("nm"))],
    #         ["b-axis", "{:.4g~P}".format(self.b_axis.to("nm"))],
    #         ["c-axis", "{:.4g~P}".format(self.c_axis.to("nm"))],
    #         ["area", "{:.4g~P}".format(self.area.to("nm**2"))],
    #         ["volume", "{:.4g~P}".format(self.volume.to("nm**3"))],
    #         ["mass", "{:.4g~P}".format(self.mass.to("kg"))],
    #         ["mass per unit area", f"{self.mass_unit_area:.4g~P}"],
    #     ]
    #     output += super().__str__()

    #     class_str = "Unit Cell with the following properties\n\n"
    #     class_str += tabulate(
    #         output,
    #         headers=["parameter", "value"],
    #         tablefmt="rst",
    #         colalign=("right",),
    #         floatfmt=(".2f", ".2f"),
    #     )
    #     class_str += "\n\n" + str(self.num_atoms) + " Constituents:\n"

    #     atoms_str = []
    #     for i in range(self.num_atoms):
    #         atoms_str.append(
    #             [
    #                 self.atoms[i][0].name,
    #                 f"{self.atoms[i][1](0):0.2f}",
    #                 self.atoms[i][2],
    #                 "",
    #                 self.atoms[i][0].mag_amplitude,
    #                 self.atoms[i][0].mag_phi.to("deg").magnitude,
    #                 self.atoms[i][0].mag_gamma.to("deg").magnitude,
    #             ]
    #         )
    #     class_str += tabulate(
    #         atoms_str,
    #         headers=[
    #             "atom",
    #             "position",
    #             "position function",
    #             "magn.",
    #             "amplitude",
    #             "phi [°]",
    #             "gamma [°]",
    #         ],
    #         tablefmt="rst",
    #     )
    #     return class_str

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

        # drop automatic magnetization calculation for unitcell
        # self.magnetization.append([atom.mag_amplitude, atom.mag_phi, atom.mag_gamma])

        self.structural.mass.quantity = 0 * u.kg
        for i in range(self.num_atoms):
            self.structural.mass.quantity = self.structural.mass.quantity + self.atoms[i][0].mass

        self.density = self.mass / self.volume
        # set mass per unit area
        self.structural.mass_unit_area.quantity = self.mass* 1 * u.angstrom**2 / self.area
        self.elastic.calc_spring_const(
            self.structural.mass_unit_area.magnitude, self.structural.thickness.magnitude
        )

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

    # ============================================================================
    # Lattice parameters
    # ============================================================================

    @property
    def a_axis(self):
        return self.lattice.a_axis.quantity

    @a_axis.setter
    def a_axis(self, value):
        self.lattice.a_axis.quantity = value
        self.structural.area.quantity = self.a_axis * self.b_axis
        self.elastic.calc_spring_const(
            self.structural.mass_unit_area.magnitude, self.structural.thickness.magnitude
        )
        self.elastic.calc_acoustic_impedance(
            self.structural.mass.magnitude, self.structural.area.magnitude
        )

    @property
    def b_axis(self):
        return self.lattice.b_axis.quantity

    @b_axis.setter
    def b_axis(self, value):
        self.lattice.b_axis.quantity = value
        self.structural.area.quantity = self.a_axis * self.b_axis
        self.elastic.calc_spring_const(
            self.structural.mass_unit_area.magnitude, self.structural.thickness.magnitude
        )
        self.elastic.calc_acoustic_impedance(
            self.structural.mass.magnitude, self.structural.area.magnitude
        )

    @property
    def c_axis(self):
        return self.lattice.c_axis.quantity

    @c_axis.setter
    def c_axis(self, value):
        self.thickness = value
        self.lattice.c_axis.quantity = value
        self.elastic.calc_spring_const(
            self.structural.mass_unit_area.magnitude, self.structural.thickness.magnitude
        )

