import pint

from .simulations.heat import Heat
from .simulations.magnetization import LLB, LLG, Magnetization, iLLG
from .simulations.phonons import Phonon, PhononAna, PhononNum
from .simulations.simulation import Simulation
from .simulations.xrays import Xray, XrayDyn, XrayDynMag, XrayKin
from .structures.atoms import Atom, AtomMixed
from .structures.dielectric import Dielectric, Dispersion
from .structures.layers import AmorphousLayer, Layer, UnitCell, Vacuum
from .structures.parameter_groups import (
    AtomMixedParameters,
    AtomParameters,
    ElasticParameters,
    LatticeParameters,
    MagneticParameters,
    OpticalParameters,
    StructuralParameters,
    ThermalParameters,
)
from .structures.parameters import Parameter, TemperatureParameter, VectorParameter
from .structures.structure import Structure

u = pint.get_application_registry()
u.formatter.default_format = ".4g~P"
Q_ = u.Quantity

__all__ = [
    "Atom",
    "AtomMixed",
    "AtomMixedParameters",
    "AtomParameters",
    "ElasticParameters",
    "LatticeParameters",
    "MagneticParameters",
    "OpticalParameters",
    "StructuralParameters",
    "ThermalParameters",
    "Layer",
    "Vacuum",
    "AmorphousLayer",
    "UnitCell",
    "Dielectric",
    "Dispersion",
    "Structure",
    "Simulation",
    "Heat",
    "Parameter",
    "Phonon",
    "PhononNum",
    "PhononAna",
    "Magnetization",
    "LLG",
    "iLLG",
    "LLB",
    "TemperatureParameter",
    "VectorParameter",
    "Xray",
    "XrayKin",
    "XrayDyn",
    "XrayDynMag",
    "u",
    "Q_",
]

__version__ = "2.6.1"
