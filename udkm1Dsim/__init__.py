import pint

from .simulations.heat import Heat
from .simulations.magnetization import LLB, LLG, Magnetization, iLLG
from .simulations.parameter_groups import ExperimentParameters
from .simulations.phonons import Phonon, PhononAna, PhononNum
from .simulations.scattering import (
    BerremanElzo,
    BerremanStepanov,
    Elzo,
    MultilayerScattering,
    Parratt,
    Scattering,
    Stepanov,
    XrayDyn,
    XrayKin,
)
from .simulations.simulation import Simulation
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
    "ExperimentParameters",
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
    "Scattering",
    "XrayKin",
    "XrayDyn",
    "MultilayerScattering",
    "Parratt",
    "Elzo",
    "Stepanov",
    "BerremanElzo",
    "BerremanStepanov",
    "u",
    "Q_",
]

__version__ = "3.0.0a2"
