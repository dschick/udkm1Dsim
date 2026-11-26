import pint

from .simulations.heat import Heat
from .simulations.magnetization import LLB, LLG, Magnetization, iLLG
from .simulations.parameter_groups import ExperimentParameters, InstrumentParameters
from .simulations.phonons import Phonon, PhononAna, PhononNum
from .simulations.result import Result
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
from .structures.layers import Layer, UnitCell, Vacuum
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
    "InstrumentParameters",
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
    "UnitCell",
    "Dielectric",
    "Dispersion",
    "Structure",
    "Simulation",
    "Result",
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

__version__ = "3.0.0a7"

_REMOVED = {
    "AmorphousLayer": (
        "AmorphousLayer was removed in version 3.0. Use Layer with the molar_mass of its atoms, "
        "the magnetization of the layer, and dielectric=Dielectric.from_atoms(atom) instead, "
        "see the structure example!"
    ),
    "XrayDynMag": "XrayDynMag was removed in version 3.0. Use Elzo instead!",
    "Xray": "Xray was removed in version 3.0. Use Scattering as base class instead!",
}


def __getattr__(name):
    """hints for removed classes"""
    if name in _REMOVED:
        raise AttributeError(_REMOVED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
