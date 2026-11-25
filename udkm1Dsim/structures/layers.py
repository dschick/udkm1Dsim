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

__all__ = ["Layer", "Vacuum", "UnitCell"]

__docformat__ = "restructuredtext"

import copy
import warnings
from difflib import get_close_matches
from inspect import isfunction
from operator import attrgetter

import numpy as np
import pint
from sympy import lambdify, symbols
from tabulate import tabulate

from udkm1Dsim.structures.parameter_groups import (
    ElasticParameters,
    LatticeParameters,
    MagneticParameters,
    OpticalParameters,
    ParameterGroup,
    StructuralParameters,
    ThermalParameters,
)

from .dielectric import (
    _ATOMIC_SUSCEPTIBILITY_FACTOR,
    Dielectric,
    Dispersion,
    _Atoms,
    _cartesian_magnetizations,
    _OptPenDepth,
    to_energy,
)

u = pint.get_application_registry()
Q_ = u.Quantity


# removed optical inputs and their replacements
_REMOVED_OPTICAL_INPUTS = {
    "opt_ref_index": "dielectric=Dispersion.from_ref_index(n)",
    "opt_pen_depth": (
        "dielectric=Dispersion.from_opt_pen_depth(zeta) or layer.set_opt_pen_depth(zeta)"
    ),
    "opt_ref_index_per_strain": "the strain_model of the Dielectric",
}


class _GroupAttribute:
    """Parameter group of a :class:`Layer`.

    An assigned group is copied, so that it can be reused for other layers, and
    a dict is passed to the constructor of the group. The group is connected to
    the layer, which updates its parameters depending on several groups.

    """

    def __init__(self, group_class):
        self.group_class = group_class

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return obj.__dict__[self.name]
        except KeyError:
            raise AttributeError(self.name) from None

    def __set__(self, obj, value):
        if isinstance(value, dict):
            group = self.group_class(**value)
        elif isinstance(value, self.group_class):
            group = copy.deepcopy(value)
        else:
            raise TypeError(
                f"'{self.name}' must be {self.group_class.__name__} or a dict of its "
                f"parameters, not {type(value).__name__}!"
            )
        group._owner = obj
        obj.__dict__[self.name] = group
        obj._update_depending()


class Layer:
    r"""Layer

    Generic, homogeneous slab of any material, e.g. an amorphous,
    polycrystalline, or single-crystalline solid, a liquid, or a gas, which
    is given only by its physical properties without any assumption on its
    internal structure. Only X-ray diffraction by :class:`.XrayKin` and
    :class:`.XrayDyn` requires the positions of the atoms of a
    :class:`UnitCell`.

    A layer consists of structural, thermal, elastic, optical, and magnetic properties.
    These properties are organized into dedicated parameter groups:

    * :class:`.StructuralParameters`
    * :class:`.ThermalParameters`
    * :class:`.ElasticParameters`
    * :class:`.OpticalParameters`
    * :class:`.MagneticParameters`

    The parameter groups are accessible through the corresponding
    attributes of the layer, e.g. `layer.elastic`, whose attributes are the
    :class:`.Parameter` objects with their name, symbol, value, and unit, e.g.
    `layer.elastic.sound_vel`. In contrast, the shortcuts of the layer, e.g.
    `layer.sound_vel`, return only their values for calculations, i.e. a pint
    Quantity, a number, or, for temperature-dependent parameters, the list of
    their functions.

    The parameter groups can be given on initialization either as
    parameter group, as dict, or as single keyword arguments:

    .. code-block:: python

        thermal = ThermalParameters(heat_capacity=455 * u.J / u.kg / u.K)
        layer = Layer("SRO", "SRO", thermal=thermal, elastic={"sound_vel": 6.3 * u.nm / u.ps})
        layer = Layer("SRO", "SRO", heat_capacity=455 * u.J / u.kg / u.K)

    Parameter groups are copied on assignment, so they can be reused for
    several layers, but changing the original group does not change the layers.

    Args:
        id (str): id of the layer.
        name (str): name of the layer.

    Keyword Args:
        structural (StructuralParameters, dict): structural parameters.
        thermal (ThermalParameters, dict): thermal parameters.
        elastic (ElasticParameters, dict): elastic parameters.
        optical (OpticalParameters, dict): optical parameters.
        magnetic (MagneticParameters, dict): magnetic parameters.
        **kwargs: single parameters of any of the groups, e.g. `thickness`.

    Attributes:
        id (str): id of the layer.
        name (str): name of the layer.

    """

    structural = _GroupAttribute(StructuralParameters)
    thermal = _GroupAttribute(ThermalParameters)
    elastic = _GroupAttribute(ElasticParameters)
    optical = _GroupAttribute(OpticalParameters)
    magnetic = _GroupAttribute(MagneticParameters)

    _parameter_group_names = ("structural", "thermal", "elastic", "optical", "magnetic")
    _updating = False

    def __init__(self, id, name, **kwargs):
        self.id = id
        self.name = name
        optical = kwargs.get("optical")
        for removed, replacement in _REMOVED_OPTICAL_INPUTS.items():
            if removed in kwargs or (isinstance(optical, dict) and removed in optical):
                raise TypeError(f"'{removed}' was removed, use {replacement} instead!")
        self._check_inputs(kwargs)

        # the parameters depending on several groups are updated once all are set
        self._updating = True
        for group_name in self._parameter_group_names:
            setattr(self, group_name, self._group_input(group_name, kwargs))
        self._updating = False
        self._update_depending()

    def __repr__(self):
        """String representation of this class"""
        class_str = f"{type(self).__name__}: {self.name}\nID: {self.id}\n" + "=" * 30 + "\n"
        for group_name in self._parameter_group_names:
            class_str += repr(getattr(self, group_name)) + "\n"

        return class_str

    def _repr_html_(self):
        """HTML representation of this class"""
        class_str = f"<h2>{type(self).__name__}: {self.name}</h2><b>ID:</b> <i>{self.id}</i><br>"
        for group_name in self._parameter_group_names:
            class_str += getattr(self, group_name)._repr_html_()

        return class_str

    def dielectric_from_atoms(self, fallback=None):
        """dielectric_from_atoms

        Dielectric of the layer from its atoms with the non-magnetic
        dispersion from the atomic form factors, the gyrotropic dispersion
        from the magnetic form factors, if tabulated, and the strain model
        ``density``, see :meth:`.Dielectric.from_atoms`. It is not set as
        dielectric of the layer, e.g. to change it before, see
        :meth:`set_dielectric_from_atoms`.

        Args:
            fallback (Dispersion, optional): non-magnetic dispersion, which
                is used outside of the energy range of the atomic form factors,
                e.g. a refractive index in the optical range.

        Returns:
            dielectric (Dielectric): dielectric from the atoms.

        """
        return Dielectric.from_atoms(self._atoms_and_fractions(), fallback=fallback)

    def dispersion_from_atoms(self, magnetic=False):
        r"""dispersion_from_atoms

        Dispersion of the layer from the atomic or magnetic form factors of
        its atoms, see :meth:`.Atom.get_susceptibility`. It is evaluated for
        the actual density of the layer and is only defined in the energy
        range of the tabulated form factors, e.g. in the X-ray range.

        The dispersion is never set automatically, but must be assigned to
        the dielectric tensor of the layer, e.g. combined with an optical table:

        .. code-block:: python

            layer.optical.dielectric = (
                Dispersion.from_file("Fe.nk") | layer.dispersion_from_atoms()
            )

        Args:
            magnetic (bool, optional): gyrotropic dispersion from the magnetic
                form factors instead of the non-magnetic one from the atomic
                form factors. Defaults to False.

        Returns:
            dispersion (Dispersion): dispersion from the atoms.

        """
        return _Atoms(self._atoms_and_fractions(), magnetic=magnetic)

    def get_absorption_coefficient(self, energy, strain=0.0):
        """get_absorption_coefficient

        Attenuation coefficient of the intensity of the layer, i.e. the
        inverse of its optical penetration depth, see
        :meth:`.Dielectric.get_absorption_coefficient`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain. Defaults to 0.

        Returns:
            mu (Quantity): attenuation coefficient.

        """
        return self._dielectric_output("get_absorption_coefficient", energy, strain)

    def get_critical_angle(self, energy, strain=0.0):
        """get_critical_angle

        Critical angle of the total external reflection of the layer from
        vacuum, see :meth:`.Dielectric.get_critical_angle`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain. Defaults to 0.

        Returns:
            theta_c (Quantity): critical angle.

        """
        return self._dielectric_output("get_critical_angle", energy, strain)

    def get_delta_beta(self, energy, strain=0.0):
        """get_delta_beta

        Real and imaginary part of the refractive index
        :math:`n = 1 - \\delta + i\\beta` of the layer, see
        :meth:`.Dielectric.get_delta_beta`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain. Defaults to 0.

        Returns:
            (tuple):
            - *delta (float, ndarray[float])* - decrement of the real part.
            - *beta (float, ndarray[float])* - imaginary part.

        """
        return self._dielectric_output("get_delta_beta", energy, strain)

    def get_delta_beta_mag(self, energy, strain=0.0):
        """get_delta_beta_mag

        Magnetic optical constants :math:`\\Delta\\delta` and
        :math:`\\Delta\\beta` of the layer per unit magnetization, see
        :meth:`.Dielectric.get_delta_beta_mag`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain. Defaults to 0.

        Returns:
            (tuple):
            - *delta_mag (float, ndarray[float])* - magnetic decrement of the
              real part.
            - *beta_mag (float, ndarray[float])* - magnetic imaginary part.

        """
        return self._dielectric_output("get_delta_beta_mag", energy, strain)

    def get_dielectric_tensor(self, energy, strain=0.0, magnetization=None, polar=False):
        """get_dielectric_tensor

        Dielectric tensor of the layer for its density, see
        :meth:`.Dielectric.get_tensor`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain. Defaults to 0.
            magnetization (tuple[float, Quantity], VectorParameter, optional):
                cartesian magnetization ``(x, y, z)``, or for ``polar=True``
                its amplitude and the angles :math:`\\phi` and
                :math:`\\gamma` as angle Quantities, e.g.
                ``(1, 90 * u.deg, 0 * u.deg)``, or a list of the
                magnetizations of all sublattices. Defaults to the
                magnetization of the layer. For several sublattices, the
                ``gyrotropy`` of the dielectric gives their coupling, see
                :class:`.Dielectric`.
            polar (bool, optional): the magnetization is given in polar
                coordinates. Defaults to False.

        Returns:
            eps (ndarray[complex]): dielectric tensor of shape (M, 3, 3).

        """
        try:
            magnetization = self._magnetization_for_dielectric(magnetization, polar)
        except ValueError as e:
            raise ValueError(f"Layer '{self.id}': {e}") from None
        polar = False
        try:
            return self.optical.dielectric.get_tensor(
                energy,
                strain,
                magnetization,
                density=self.structural.density.magnitude,
                polar=polar,
            )
        except ValueError as e:
            raise ValueError(f"Layer '{self.id}': {e}") from None

    def get_form_factor(self, energy, strain=0.0):
        r"""get_form_factor

        Effective atomic form factor :math:`f = f_1 - i f_2` per atom of a
        layer, which follows from its dielectric function, e.g. also from
        tabulated optical constants, by

        .. math:: \varepsilon = 1 - \frac{r_e \lambda^2}{\pi} n_a f

        with the number density :math:`n_a = \rho N_A / M` of its atoms for
        their mean ``molar_mass`` :math:`M`, which is given by the atoms of a
        :class:`UnitCell`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain. Defaults to 0.

        Returns:
            f (complex, ndarray[complex]): effective atomic form factor.

        """
        molar_mass = self.structural.molar_mass.magnitude
        if molar_mass <= 0:
            raise ValueError(
                f"The {type(self).__name__} '{self.id}' has no molar_mass for the number "
                "density of its atoms!"
            )
        mass_number = molar_mass * 1000  # (g/mol)
        n = self._dielectric_output("get_ref_index", energy, strain)
        energy = to_energy(energy)
        density = self.structural.density.magnitude / (1 + strain)
        molar_density = density / 1000 / mass_number
        return -(n**2 - 1) * energy**2 / (_ATOMIC_SUSCEPTIBILITY_FACTOR * molar_density)

    def get_mass_attenuation_coefficient(self, energy, strain=0.0):
        """get_mass_attenuation_coefficient

        Mass attenuation coefficient of the layer for its density, see
        :meth:`.Dielectric.get_mass_attenuation_coefficient`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities.
            strain (float, optional): strain. Defaults to 0.

        Returns:
            mu_rho (Quantity): mass attenuation coefficient.

        """
        return self._dielectric_output("get_mass_attenuation_coefficient", energy, strain)

    def get_opt_pen_depth(self, energy, strain=0.0):
        """get_opt_pen_depth

        Optical penetration depth or absorption length of the intensity of
        the layer, see :meth:`.Dielectric.get_opt_pen_depth`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities, e.g.
                the wavelength.
            strain (float, optional): strain. Defaults to 0.

        Returns:
            zeta (Quantity): optical penetration depth.

        """
        return self._dielectric_output("get_opt_pen_depth", energy, strain)

    def get_property_dict(self, **kwargs):
        """get_property_dict

        Returns a dictionary with all parameters. If a type is given, only
        the parameters relevant for this type of simulation are returned.
        Their keys are the paths of the parameters, e.g.
        `structural.thickness`, and their values are the
        :class:`.Parameter` objects, whose `repr` contains their name, value,
        and unit, as required for :meth:`.Structure.get_hash`.

        Args:
            **kwargs (list[str]): types of requested properties.

        Returns:
            R (dict): dictionary with requested properties.

        """
        properties_by_types = {
            "heat": [
                "structural.thickness",
                "structural.mass_unit_area",
                "structural.density",
                "optical.dielectric",
                "thermal.therm_cond",
                "thermal.heat_capacity",
                "thermal.sub_system_coupling",
                "thermal.num_sub_systems",
            ],
            "phonon": [
                "thermal.num_sub_systems",
                "thermal.lin_therm_exp",
                "structural.thickness",
                "structural.mass_unit_area",
                "elastic.spring_const",
                "elastic.phonon_damping",
            ],
            "xray": [
                "num_atoms",
                "structural.area",
                "structural.mass",
                "structural.density",
                "structural.roughness",
                "structural.thickness",
                "thermal.deb_wal_fac",
            ],
            "optical": [
                "lattice.c_axis",
                "structural.density",
                "optical.dielectric",
            ],
            "magnetic": [
                "structural.thickness",
                "magnetic.magnetization",
                "magnetic.eff_spin",
                "magnetic.curie_temp",
                "magnetic.mf_exch_coupling",
                "magnetic.mag_moment",
                "magnetic.aniso_exponent",
                "magnetic.anisotropy",
                "magnetic.exch_stiffness",
                "magnetic.mag_saturation",
                "magnetic.lamda",
                "magnetic.gilbert_damping",
                "magnetic.gyromagnetic_ratio",
                "magnetic.inertial_time",
                "magnetic.magnetoelastic_coupling",
                "magnetic.exch_relax_rate",
                "magnetic.concentration",
            ],
        }

        types = kwargs.get("types", "all")
        if type(types) is not list:
            types = [types]
        R = {}
        for t in types:
            # define the property names by the given type
            if t == "all":
                return vars(self)
            for path in properties_by_types[t]:
                try:
                    R[path] = attrgetter(path)(self)
                except AttributeError:
                    # e.g. num_atoms and lattice exist only for UnitCell
                    continue

        return R

    def _hash_params(self, **kwargs):
        """Inputs of :meth:`.Structure.get_hash` of the layer, i.e. its
        properties of the given types, see :meth:`get_property_dict`, and for
        the type ``'xray'`` the hashes of its atoms, see :meth:`_atoms_hash`."""
        param = [self.get_property_dict(**kwargs)]
        types = kwargs.get("types", "all")
        if "xray" in (types if isinstance(types, list) else [types]):
            param.append(self._atoms_hash())
        return param

    def _atoms_hash(self):
        """Hashes of the atoms of the layer, see :attr:`.Atom.hash`,
        e.g. with their positions in a :class:`UnitCell`."""
        return []

    def get_ref_index(self, energy, strain=0.0):
        """get_ref_index

        Complex refractive index :math:`n = n' + i\\kappa` of the layer with
        an isotropic dielectric function, see :meth:`.Dielectric.get_ref_index`.

        Args:
            energy (float, ndarray[float], Quantity): photon energies, where
                floats are taken in eV, or other spectral Quantities, e.g.
                the wavelength.
            strain (float, optional): strain. Defaults to 0.

        Returns:
            n (complex, ndarray[complex]): refractive index.

        """
        return self._dielectric_output("get_ref_index", energy, strain)

    def set_dielectric_from_atoms(self, fallback=None):
        """set_dielectric_from_atoms

        Set the dielectric tensor of the layer from its atoms, see
        :meth:`dielectric_from_atoms`, e.g. with a refractive index in the
        optical range for the excitation:

        .. code-block:: python

            layer.set_dielectric_from_atoms()
            layer.set_dielectric_from_atoms(fallback=Dispersion.from_ref_index(2.9 + 3.4j))

        Args:
            fallback (Dispersion, optional): non-magnetic dispersion, which
                is used outside of the energy range of the atomic form factors.

        """
        self.optical.dielectric = self.dielectric_from_atoms(fallback=fallback)

    def set_ho_spring_constants(self, HO):
        """set_ho_spring_constants

        Set the higher orders of the spring constant for anharmonic
        phonon simulations.

        Args:
            HO (ndarray[float]): higher order spring constants.

        """
        self.elastic.set_ho_spring_constants(HO)

    def set_opt_pen_depth(self, opt_pen_depth):
        r"""set_opt_pen_depth

        Set the optical penetration depth of the intensity :math:`\zeta` of
        the layer at all wavelengths :math:`\lambda` by its extinction
        coefficient :math:`\kappa = \lambda / (4 \pi \zeta)`, see
        :meth:`.Dispersion.from_opt_pen_depth`.

        Only :math:`\kappa` is changed, while the real part :math:`n'` of the
        refractive index of the dielectric function is kept, which is 1 at energies
        without a dielectric function. Hence, a warning is raised, as this is probably
        only useful for the Lambert-Beer absorption. The magnetic part and the
        strain model of the dielectric tensor are kept.

        Args:
            opt_pen_depth (float, Quantity): optical penetration depth, where
                floats are taken in m.

        """
        dielectric = self.optical.dielectric
        if dielectric.is_set and not dielectric.is_isotropic:
            raise ValueError(
                f"Layer '{self.id}': the optical penetration depth requires an isotropic "
                "dielectric!"
            )
        warnings.warn(
            f"The optical penetration depth of the layer '{self.id}' only changes the "
            "extinction coefficient kappa, but not the real part n' of its refractive "
            "index, so that it is probably only useful for the Lambert-Beer absorption.",
            stacklevel=2,
        )
        base = dielectric.eps0
        if isinstance(base, _OptPenDepth):
            # replace a former penetration depth
            base = base._base
        if base is None:
            dielectric.eps0 = _OptPenDepth(opt_pen_depth)
        else:
            dielectric.eps0 = _OptPenDepth(opt_pen_depth, base=base)

    # ============================================================================
    # Structural parameters
    # ============================================================================

    @property
    def thickness(self):
        """Thickness :math:`d` of the layer as Quantity, see :class:`.StructuralParameters`."""
        return self.structural.thickness.quantity

    @thickness.setter
    def thickness(self, value):
        self.structural.thickness = value

    @property
    def mass(self):
        """Read-only mass :math:`m` of the layer as Quantity, derived from its density and volume,
        see :class:`.StructuralParameters`.
        """
        return self.structural.mass.quantity

    @mass.setter
    def mass(self, value):
        raise AttributeError(
            "'mass' is derived from density and volume and cannot be set directly."
        )

    @property
    def mass_unit_area(self):
        """Read-only mass :math:`m_A` of the layer normalized to a unit area of 1 Å² as Quantity,
        see :class:`.StructuralParameters`.
        """
        return self.structural.mass_unit_area.quantity

    @mass_unit_area.setter
    def mass_unit_area(self, value):
        raise AttributeError(
            "'mass_unit_area' is derived from 'density and volume normalized to area "
            "and cannot be set directly. 'area' is fixed to 1.0 angstrom² for Layer"
        )

    @property
    def density(self):
        r"""Mass density :math:`\rho` as Quantity, see :class:`.StructuralParameters`."""
        return self.structural.density.quantity

    @density.setter
    def density(self, value):
        self.structural.density = value

    @property
    def area(self):
        """Area :math:`A` of the layer as Quantity, see :class:`.StructuralParameters`."""
        return self.structural.area.quantity

    @area.setter
    def area(self, value):
        raise AttributeError("'area' is fixed to 1.0 angstrom² for Layer")

    @property
    def volume(self):
        """Read-only volume :math:`V` as Quantity, see :class:`.StructuralParameters`."""
        return self.structural.volume.quantity

    @volume.setter
    def volume(self, value):
        raise AttributeError(
            "'volume' is derived from 'thickness' and 'area' and cannot be set directly."
        )

    @property
    def roughness(self):
        r"""Gaussian width :math:`\sigma` of the top roughness of the layer as Quantity, see
        :class:`.StructuralParameters`.
        """
        return self.structural.roughness.quantity

    @roughness.setter
    def roughness(self, value):
        self.structural.roughness = value

    # ============================================================================
    # Thermal parameters
    # ============================================================================

    @property
    def therm_cond(self):
        r"""Temperature-dependent thermal conductivities :math:`\kappa` per subsystem as functions,
        see :class:`.ThermalParameters`.
        """
        return self.thermal.therm_cond.functional

    @therm_cond.setter
    def therm_cond(self, value):
        self.thermal.therm_cond = value

    @property
    def therm_cond_expr(self):
        """Sympy expressions of the thermal conductivities per subsystem in (W/(m K))."""
        return self.thermal.therm_cond.quantity

    @property
    def heat_capacity(self):
        """Temperature-dependent heat capacities :math:`c` per subsystem as functions, see
        :class:`.ThermalParameters`.
        """
        return self.thermal.heat_capacity.functional

    @heat_capacity.setter
    def heat_capacity(self, value):
        self.thermal.heat_capacity = value

    @property
    def heat_capacity_expr(self):
        """Sympy expressions of the heat capacities per subsystem in (J/(kg K))."""
        return self.thermal.heat_capacity.quantity

    @property
    def lin_therm_exp(self):
        r"""Temperature-dependent linear thermal expansion coefficients :math:`\alpha` per
        subsystem as functions, see :class:`.ThermalParameters`.
        """
        return self.thermal.lin_therm_exp.functional

    @lin_therm_exp.setter
    def lin_therm_exp(self, value):
        self.thermal.lin_therm_exp = value

    @property
    def lin_therm_exp_expr(self):
        """Sympy expressions of the linear thermal expansion coefficients per subsystem in
        (1/K)."""
        return self.thermal.lin_therm_exp.quantity

    @property
    def int_lin_therm_exp(self):
        """Integrals of the linear thermal expansion coefficients over the temperature as
        functions.
        """
        return self.thermal.lin_therm_exp.integral

    @int_lin_therm_exp.setter
    def int_lin_therm_exp(self, value):
        raise AttributeError(
            "'int_lin_therm_exp' is automatically derived from lin_therm_exp. "
            "To set explicitly, modify 'Layer.thermal.int_lin_therm_exp' instead."
        )

    @property
    def int_lin_therm_exp_expr(self):
        """Expressions of the integrals of the linear thermal expansion coefficients."""
        return self.thermal.lin_therm_exp.integral_expr

    @property
    def int_heat_capacity(self):
        """Integrals of the heat capacities over the temperature as functions."""
        return self.thermal.heat_capacity.integral

    @int_heat_capacity.setter
    def int_heat_capacity(self, value):
        raise AttributeError(
            "'int_heat_capacity' is automatically derived from heat_capacity. "
            "To set explicitly, modify 'Layer.thermal.int_heat_capacity' instead."
        )

    @property
    def int_heat_capacity_expr(self):
        """Expressions of the integrals of the heat capacities."""
        return self.thermal.heat_capacity.integral_expr

    @property
    def sub_system_coupling(self):
        """Coupling functions :math:`G` of the subsystems, see :class:`.ThermalParameters`."""
        return self.thermal.sub_system_coupling.functional

    @sub_system_coupling.setter
    def sub_system_coupling(self, value):
        self.thermal.sub_system_coupling = value

    @property
    def deb_wal_fac(self):
        r"""Temperature-dependent Debye-Waller factors :math:`\langle u^2\rangle` per subsystem as
        functions, see :class:`.ThermalParameters`.
        """
        return self.thermal.deb_wal_fac.functional

    @deb_wal_fac.setter
    def deb_wal_fac(self, value):
        self.thermal.deb_wal_fac = value

    @property
    def deb_wal_fac_expr(self):
        """Sympy expressions of the Debye-Waller factors per subsystem in (m²)."""
        return self.thermal.deb_wal_fac.quantity

    @property
    def num_sub_systems(self):
        """Number of subsystems :math:`N` for heat and phonons, see :class:`.ThermalParameters`."""
        return self.thermal.num_sub_systems.magnitude

    @num_sub_systems.setter
    def num_sub_systems(self, value):
        raise AttributeError(
            "'num_sub_systems' is derived from the temperature-dependent thermal "
            "parameters and cannot be set directly."
        )

    # ============================================================================
    # Elastic parameters
    # ============================================================================

    @property
    def sound_vel(self):
        """Longitudinal sound velocity :math:`v` of the layer as Quantity, see
        :class:`.ElasticParameters`.
        """
        return self.elastic.sound_vel.quantity

    @sound_vel.setter
    def sound_vel(self, value):
        self.elastic.sound_vel = value

    @property
    def spring_const(self):
        """Spring constants :math:`k` of the layer as Quantity, see :class:`.ElasticParameters`."""
        return self.elastic.spring_const.quantity

    @spring_const.setter
    def spring_const(self, value):
        raise AttributeError(
            "'spring_const' is automatically derived from sound_vel, mass, and thickness."
        )

    @property
    def acoustic_impedance(self):
        """Acoustic impedance :math:`Z` as Quantity, see :class:`.ElasticParameters`."""
        return self.elastic.acoustic_impedance.quantity

    @acoustic_impedance.setter
    def acoustic_impedance(self, value):
        raise AttributeError(
            "'acoustic_impedance' is automatically derived from spring_const, mass, and area."
        )

    @property
    def phonon_damping(self):
        r"""Damping constant :math:`\gamma` of the phonon propagation as Quantity, see
        :class:`.ElasticParameters`.
        """
        return self.elastic.phonon_damping.quantity

    @phonon_damping.setter
    def phonon_damping(self, value):
        self.elastic.phonon_damping = value

    # ============================================================================
    # Optical parameters
    # ============================================================================

    @property
    def dielectric(self):
        r"""Dielectric function :math:`\varepsilon`, see :class:`.OpticalParameters`."""
        return self.optical.dielectric

    @dielectric.setter
    def dielectric(self, value):
        self.optical.dielectric = value

    # ============================================================================
    # Magnetic parameters
    # ============================================================================

    @property
    def eff_spin(self):
        """Effective spin :math:`S` per sublattice, see :class:`.MagneticParameters`."""
        return self.magnetic.eff_spin.quantity

    @eff_spin.setter
    def eff_spin(self, value):
        self.magnetic.eff_spin = value

    @property
    def curie_temp(self):
        r"""Curie temperature :math:`T_\mathrm{C}`, see :class:`.MagneticParameters`."""
        return self.magnetic.curie_temp.quantity

    @curie_temp.setter
    def curie_temp(self, value):
        self.magnetic.curie_temp = value

    @property
    def mf_exch_coupling(self):
        """Mean-field exchange coupling constant :math:`J` as Quantity, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.mf_exch_coupling.quantity

    @mf_exch_coupling.setter
    def mf_exch_coupling(self, value):
        self.magnetic.mf_exch_coupling = value

    @property
    def lamda(self):
        r"""Intrinsic coupling to the bath :math:`\lambda` per sublattice, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.lamda.quantity

    @lamda.setter
    def lamda(self, value):
        self.magnetic.lamda = value

    @property
    def mag_moment(self):
        r"""Atomic magnetic moment :math:`\mu` per sublattice as Quantity, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.mag_moment.quantity

    @mag_moment.setter
    def mag_moment(self, value):
        self.magnetic.mag_moment = value

    @property
    def aniso_exponent(self):
        r"""Exponent :math:`\beta` of the temperature dependence of the anisotropy per sublattice,
        see :class:`.MagneticParameters`.
        """
        return self.magnetic.aniso_exponent.quantity

    @aniso_exponent.setter
    def aniso_exponent(self, value):
        self.magnetic.aniso_exponent = value

    @property
    def anisotropy(self):
        """Anisotropy :math:`K` at 0 K as Quantity of its x, y, and z components, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.anisotropy.quantity

    @anisotropy.setter
    def anisotropy(self, value):
        self.magnetic.anisotropy = value

    @property
    def exch_stiffness(self):
        r"""Exchange stiffness :math:`A_\mathrm{ex}` at 0 K as Quantity, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.exch_stiffness.quantity

    @exch_stiffness.setter
    def exch_stiffness(self, value):
        self.magnetic.exch_stiffness = value

    @property
    def mag_saturation(self):
        """Saturation magnetization :math:`M_s` at 0 K per sublattice as Quantity, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.mag_saturation.quantity

    @mag_saturation.setter
    def mag_saturation(self, value):
        self.magnetic.mag_saturation = value

    @property
    def magnetization(self):
        """The cartesian magnetization, as :math:`P \\times 3` array for
        several sublattices."""
        magnetization = self.magnetic.magnetization
        if isinstance(magnetization, list):
            return np.array([vector.magnitude for vector in magnetization]) * u.dimensionless
        return magnetization.quantity

    @magnetization.setter
    def magnetization(self, value):
        self.magnetic.magnetization = value

    @property
    def gilbert_damping(self):
        r"""Gilbert damping :math:`\alpha` of the LLG per sublattice, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.gilbert_damping.quantity

    @gilbert_damping.setter
    def gilbert_damping(self, value):
        self.magnetic.gilbert_damping = value

    @property
    def gyromagnetic_ratio(self):
        r"""Gyromagnetic ratio :math:`\gamma` including its negative sign as Quantity, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.gyromagnetic_ratio.quantity

    @gyromagnetic_ratio.setter
    def gyromagnetic_ratio(self, value):
        self.magnetic.gyromagnetic_ratio = value

    @property
    def inertial_time(self):
        r"""Inertial relaxation time :math:`\tau` of the inertial LLG per sublattice as Quantity,
        see :class:`.MagneticParameters`.
        """
        return self.magnetic.inertial_time.quantity

    @inertial_time.setter
    def inertial_time(self, value):
        self.magnetic.inertial_time = value

    @property
    def magnetoelastic_coupling(self):
        """Magnetoelastic coupling constant :math:`B_1` per sublattice as Quantity, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.magnetoelastic_coupling.quantity

    @magnetoelastic_coupling.setter
    def magnetoelastic_coupling(self, value):
        self.magnetic.magnetoelastic_coupling = value

    @property
    def exch_relax_rate(self):
        r"""Exchange-relaxation rate :math:`\lambda_\mathrm{ex}` of the extended LLB, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.exch_relax_rate.quantity

    @exch_relax_rate.setter
    def exch_relax_rate(self, value):
        self.magnetic.exch_relax_rate = value

    @property
    def concentration(self):
        """Concentration :math:`x` per sublattice for the exchange relaxation, see
        :class:`.MagneticParameters`.
        """
        return self.magnetic.concentration.quantity

    @concentration.setter
    def concentration(self, value):
        self.magnetic.concentration = value

    @property
    def num_sublattices(self):
        """int: number of magnetic sublattices, see
        :attr:`.MagneticParameters.num_sublattices`."""
        return self.magnetic.num_sublattices

    # ============================================================================
    # Private methods
    # ============================================================================

    # keyword arguments of a sub-class besides the parameter groups
    _extra_inputs = ()

    def _check_inputs(self, kwargs):
        """Raise for keyword arguments, which are neither a parameter group,
        nor one of their parameters, e.g. misspelled ones, which would be
        ignored otherwise, as Python does for an explicit signature, with
        the closest known names as hint."""
        known = set(self._parameter_group_names) | set(self._extra_inputs)
        for group_name in self._parameter_group_names:
            known.update(getattr(type(self), group_name).group_class._input_names())
        unknown = sorted(set(kwargs) - known)
        if not unknown:
            return
        names = []
        for key in unknown:
            matches = get_close_matches(key, known, n=1)
            names.append(f"{key!r}" + (f" (did you mean {matches[0]!r}?)" if matches else ""))
        plural = "s" if len(unknown) > 1 else ""
        raise TypeError(
            f"{type(self).__name__}() got {'an ' if not plural else ''}unexpected keyword "
            f"argument{plural} {', '.join(names)}, which {'are' if plural else 'is'} neither a "
            f"parameter group ({', '.join(self._parameter_group_names)}) nor one of their "
            "parameters!"
        )

    def _group_input(self, group_name, kwargs):
        """Input of a parameter group from the keyword arguments, either as
        parameter group or as dict of the group and its single parameters."""
        group_class = getattr(type(self), group_name).group_class
        single = {key: kwargs[key] for key in group_class._input_names() if key in kwargs}
        group = kwargs.get(group_name)
        if group is None:
            return single
        if isinstance(group, dict):
            twice = sorted(set(group) & set(single))
        elif isinstance(group, ParameterGroup):
            twice = sorted(single)
        else:
            return group  # raises a TypeError on assignment
        if twice:
            raise ValueError(
                f"{', '.join(repr(key) for key in twice)} must not be given as keyword "
                f"argument and in '{group_name}' at the same time!"
            )
        return {**group, **single} if isinstance(group, dict) else group

    def _update_depending(self):
        """Called on changes of the parameter groups to update the parameters
        depending on several groups."""
        if self._updating:
            return
        self._updating = True
        try:
            self._update_layer()
        finally:
            self._updating = False

    def _update_layer(self):
        """Update the parameters depending on several groups."""
        self._update_molar_mass()
        self.magnetic._set_number_density(self.structural.number_density.magnitude)
        self._update_elastic()

    def _update_molar_mass(self):
        """Update the molar mass, which is given by the atoms of a layer with
        atoms and kept otherwise."""
        try:
            atoms = self._atoms_and_fractions()
        except TypeError:  # a layer without atoms
            return
        total = sum(fraction for _, fraction in atoms)
        mass = sum((fraction * atom.mass for atom, fraction in atoms), 0 * u.kg) / total
        self.structural.molar_mass.quantity = mass * u.N_A

    def _update_elastic(self):
        """Recalculate the spring constant and acoustic impedance, which
        depend on the structural and elastic parameters."""
        self.elastic.calc_spring_const(
            self.structural.mass_unit_area.magnitude, self.structural.thickness.magnitude
        )
        self.elastic.calc_acoustic_impedance(
            self.structural.mass.magnitude, self.structural.area.magnitude
        )

    def _atoms_and_fractions(self):
        """Atoms of the layer and their number fractions, see
        :meth:`dispersion_from_atoms`."""
        raise TypeError(
            f"The {type(self).__name__} '{self.id}' has no atoms. Set its dielectric tensor "
            "by a Dispersion, e.g. Dispersion.from_ref_index(n)!"
        )

    def _magnetization_for_dielectric(self, magnetization=None, polar=False):
        """Cartesian magnetization of the dielectric tensor of shape (3,), or
        (P, 3) for the gyrotropy ``sublattices``, from the given one, a net or
        one per sublattice, or of the layer, see :class:`.Dielectric`."""
        if magnetization is None:
            magnetization = self.magnetic.sublattice_magnetizations()
            polar = False
        magnetization = _cartesian_magnetizations(magnetization, polar)
        P = self.magnetic.num_sublattices
        if magnetization.ndim == 2 and P == 1:
            # sublattice maps of several layers are padded to the most sublattices
            if np.any(magnetization[1:]):
                raise ValueError(
                    f"The layer has a single sublattice, but {len(magnetization):d} "
                    "magnetizations!"
                )
            magnetization = magnetization[0]
        dielectric = self.optical.dielectric
        if P == 1 or not dielectric._eps_mag_parts() or not np.any(magnetization):
            # a single sublattice, or no gyrotropic part
            return magnetization if magnetization.ndim == 1 else np.zeros(3)
        gyrotropy = dielectric.gyrotropy
        if gyrotropy is None:
            raise ValueError(
                f"The layer has {P:d} magnetic sublattices, whose coupling to the gyrotropic "
                "part of the dielectric must be chosen by its gyrotropy, either "
                'layer.optical.dielectric.gyrotropy = "net" for the net magnetization or '
                '"sublattices" for the magnetization of every sublattice!'
            )
        if magnetization.ndim == 1:
            if gyrotropy == "sublattices":
                raise ValueError(
                    'The gyrotropy "sublattices" requires the magnetizations of all '
                    f"{P:d} sublattices, e.g. a sublattice_magnetization_map, not a net one!"
                )
            return magnetization
        if len(magnetization) < P:
            raise ValueError(
                f"The layer has {P:d} sublattices, but {len(magnetization):d} magnetizations!"
            )
        magnetization = magnetization[:P]
        if gyrotropy == "sublattices":
            return magnetization
        return self._net_weights() @ magnetization

    def _net_weights(self):
        """weights of the sublattices for the net magnetization, their
        saturation magnetizations normalized to their sum, or equal weights
        without saturation magnetization, as for the net magnetization map"""
        weights = self.magnetic.sublattice_values("mag_saturation")
        total = np.sum(weights)
        if total <= 0:
            weights = np.ones_like(weights)
            total = np.sum(weights)
        return weights / total

    def _dielectric_output(self, method, energy, strain):
        """Output of the dielectric tensor of the layer for its density, where errors
        are given with the id of the layer, and a scalar for a scalar energy."""
        dielectric = self.optical.dielectric
        try:
            if not dielectric.is_set:
                # raises the error of the dielectric tensor
                dielectric.get_tensor(energy)
            result = getattr(dielectric, method)(
                energy, strain=strain, density=self.structural.density.magnitude
            )
        except ValueError as e:
            raise ValueError(f"Layer '{self.id}': {e}") from None
        if np.ndim(energy) != 0:
            return result
        if isinstance(result, tuple):
            return tuple(value[0] for value in result)
        return result[0]


class Vacuum(Layer):
    """Vacuum

    Layer of vacuum with a refractive index of 1 and zero density, e.g. as the
    default superstrate and substrate of a :class:`.Structure`.

    Args:
        thickness (Quantity, optional): thickness of the layer. Defaults to
            1 nm.

    """

    def __init__(self, thickness=1 * u.nm, **kwargs):

        super().__init__("vacuum", "vacuum", dielectric=Dispersion.from_ref_index(1))
        self.thickness = thickness
        self.density = 0.0 * u.kg / u.m**3

    def __repr__(self):
        """String representation of this class"""
        return f"Vacuum layer of thickness: {self.structural.thickness.quantity.to('nm'):.4g~P}"

    def _repr_html_(self):
        """HTML representation of this class"""
        return (
            f"<h2>Vacuum</h2><b>Thickness:</b> {self.structural.thickness.quantity.to('nm'):.4g~P}"
        )


class UnitCell(Layer):
    r"""Layer

    Representation of unit cells made of one or multiple Atom or AtomMixed
    instances at defined positions.
    A unit cell consists of structural, lattice, thermal, elastic, optical,
    and magnetic properties. These properties are organized into dedicated
    parameter groups:

    * :class:`.StructuralParameters`
    * :class:`.LatticeParameters`
    * :class:`.ThermalParameters`
    * :class:`.ElasticParameters`
    * :class:`.OpticalParameters`
    * :class:`.MagneticParameters`

    The parameter groups are accessible through the corresponding
    attributes of the layer, see :class:`Layer` for their initialization.
    The thickness and area of a unit cell are given by its lattice and its
    mass by its atoms, from which its density follows.

    Args:
        id (str): id of the layer.
        name (str): name of the layer.
        c_axis (Quantity): c-axis of the unit cell, can be omitted if given
            in `lattice`.

    Keyword Args:
        lattice (LatticeParameters, dict): lattice parameters.
        **kwargs: see :class:`Layer`.

    Attributes:
        id (str): id of the layer.
        name (str): name of the layer.

    """

    lattice = _GroupAttribute(LatticeParameters)

    _parameter_group_names = (
        "structural",
        "lattice",
        "thermal",
        "elastic",
        "optical",
        "magnetic",
    )

    def __init__(self, id, name, c_axis=None, **kwargs):
        if c_axis is not None:
            kwargs["c_axis"] = c_axis
        elif "lattice" not in kwargs:
            raise TypeError("UnitCell requires a c_axis or lattice parameters!")
        structural = kwargs.get("structural")
        for key in ("thickness", "area"):
            if key in kwargs or (isinstance(structural, dict) and key in structural):
                raise ValueError(f"The {key} of a UnitCell is given by its lattice!")

        self.atoms = []
        self.num_atoms = 0
        super().__init__(id, name, **kwargs)

    def _atoms_table(self, style):
        """Table of all atoms in the unit cell in the given tabulate style"""
        atoms_str = []
        for atom, position, position_function in self.atoms:
            mag_amplitude, mag_phi, mag_gamma = atom.atomic.magnetization.polar
            atoms_str.append(
                [
                    atom.name,
                    f"{position(0):0.2f}",
                    position_function,
                    "",
                    mag_amplitude.magnitude,
                    mag_phi.to("deg").magnitude,
                    mag_gamma.to("deg").magnitude,
                ]
            )
        return tabulate(
            atoms_str,
            headers=[
                "atom",
                "position",
                "position function",
                "magn.",
                "amplitude",
                "phi (°)",
                "gamma (°)",
            ],
            tablefmt=style,
        )

    def _atoms_hash(self):
        """Hashes of the atoms, see :attr:`.Atom.hash`, with their
        position functions in the unit cell."""
        return [(atom.hash, position_str) for atom, _, position_str in self.atoms]

    def __repr__(self):
        """String representation of this class"""
        return super().__repr__() + self._atoms_table("double_grid")

    def _repr_html_(self):
        """HTML representation of this class"""
        return super()._repr_html_() + "<h3>Atoms</h3>" + self._atoms_table("html")

    def add_atom(self, atom, position):
        r"""add_atom

        Adds an AtomBase/AtomMixed at a relative position of the unit cell.

        Sort the list of atoms by the position at zero strain.

        Update the mass, density and spring constant of the unit cell
        automatically:

        .. math:: k = m \cdot (v / d)^2

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
        # the unit cell follows later changes of the atom, e.g. of a mixed atom
        atom._register_owner(self)

        # drop automatic magnetization calculation for unitcell
        # self.magnetization.append([atom.mag_amplitude, atom.mag_phi, atom.mag_gamma])

        self._update_depending()

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

    def get_atom_positions(self, *args):
        """get_atom_positions

        Calculates the relative positions of the atoms in the unit cell.

        Args:
            *args (float): optional strain of the unit cell. Defaults to 0.

        Returns:
            res (ndarray[float]): relative position of the atoms within the
            unit cell.

        """
        if args:
            strain = float(np.asarray(args[0]).item())
        else:
            strain = 0.0

        res = np.zeros([self.num_atoms])
        for i, atom in enumerate(self.atoms):
            res[i] = atom[1](strain)

        return res

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
            block (bool, optional): block the execution until the figure is
                closed, see :func:`matplotlib.pyplot.show`. Defaults to True.
            **kwargs (float): ``strain`` of the unit cell for the
                visualization.

        """
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        strain = kwargs.get("strain", 0)

        colors = [colormaps["Set3"](x) for x in np.linspace(0, 1, self.num_atoms)]
        atom_ids = self.atom_ids

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

    @property
    def atom_ids(self):
        """list[str]: unique ids of the atoms within the unit cell."""
        ids = []
        for i in range(self.num_atoms):
            if self.atoms[i][0].id not in ids:
                ids.append(self.atoms[i][0].id)

        return ids

    # ============================================================================
    # Lattice parameters
    # ============================================================================

    @property
    def a_axis(self):
        """Lattice parameter :math:`a` as Quantity, see :class:`.LatticeParameters`."""
        return self.lattice.a_axis.quantity

    @a_axis.setter
    def a_axis(self, value):
        self.lattice.a_axis = value

    @property
    def b_axis(self):
        """Lattice parameter :math:`b` as Quantity, see :class:`.LatticeParameters`."""
        return self.lattice.b_axis.quantity

    @b_axis.setter
    def b_axis(self, value):
        self.lattice.b_axis = value

    @property
    def c_axis(self):
        """Lattice parameter :math:`c` as Quantity, see :class:`.LatticeParameters`."""
        return self.lattice.c_axis.quantity

    @c_axis.setter
    def c_axis(self, value):
        self.lattice.c_axis = value

    @property
    def thickness(self):
        """Thickness of the unit cell as Quantity, which is its c-axis."""
        return self.structural.thickness.quantity

    @thickness.setter
    def thickness(self, value):
        # the thickness of a unit cell is its c-axis
        self.lattice.c_axis = value

    @property
    def density(self):
        """Mass density of the unit cell as Quantity, derived from its atoms and volume, unless the
        unit cell has no atoms.
        """
        return self.structural.density.quantity

    @density.setter
    def density(self, value):
        if self.num_atoms > 0:
            raise AttributeError(
                "The density of a UnitCell with atoms is given by their mass and its volume."
            )
        self.structural.density = value

    # ============================================================================
    # Private methods
    # ============================================================================

    def _atom_changed(self):
        """Called on changes of an atom of the unit cell, e.g. of the
        constituents of a mixed atom, to update its mass and density."""
        self._update_depending()

    def __setstate__(self, state):
        # a copied or unpickled unit cell follows the changes of its atoms
        self.__dict__.update(state)
        for atom, _, _ in self.atoms:
            atom._register_owner(self)

    def _update_layer(self):
        """The thickness and area of a unit cell are given by its lattice and
        its mass by its atoms. Hence, the density follows from the mass and the
        volume. Without any atoms, the density is kept and the mass follows."""
        structural = self.structural
        structural.thickness.quantity = self.lattice.c_axis.quantity
        structural.area.quantity = self.lattice.a_axis.quantity * self.lattice.b_axis.quantity
        if self.num_atoms > 0:
            mass = sum((atom.mass for atom, _, _ in self.atoms), 0 * u.kg)
            structural.density.quantity = mass / structural.volume.quantity
        super()._update_layer()

    def _atoms_and_fractions(self):
        """All atoms of the unit cell with equal number fractions, which
        gives the susceptibility of the unit cell for its density."""
        if self.num_atoms == 0:
            return super()._atoms_and_fractions()
        return [(atom, 1 / self.num_atoms) for atom, _, _ in self.atoms]
