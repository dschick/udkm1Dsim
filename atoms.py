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

import numpy as np
import os
import scipy.constants as constants

class atom(object):
    """atom

    The atom class is the smallest structural unit of which one can build
    larger structures. It holds real physical properties of atoms defined in
    the attrubutes section can return parameters and data necessary for
    different simulation types.

    Attributes:
        symbol (str)                 : symbol of the element
        ID (str)                     :
            identifier of the atom, may be different from symbol and/or name
        name (str)                   : name of the element (generic)
        atomicNumberZ (int)          : Z atomic number
        massNumberA (float)          : A atomic mass number
        ionicity (int)               : ionicity of the atom
        mass (float)                 : mass of the atom [kg]
        atomicFormFactorCoeff (ndarray[float]) :
            atomic form factor coefficients for energy-dependent atomic
            form factor
        cromerMannCoeff (ndarray[float])       :
            cromer-mann coefficients for angular-dependent atomic form factor
    """

    def __init__(self, symbol, **kwargs):
        """Initialize the class, set all file names and load the spec file.

        Args:
            name (str)                  : Name of the spec file.
            filePath (str)              : Base path of the spec and HDF5 files.
            specFileExt (Optional[str]) : File extension of the spec file,
                                          default is none.

        """
        self.symbol   = symbol
        self.ID       = kwargs.get('ID', symbol)
        self.ionicity = kwargs.get('ionicity', 0)

        try:
            filename = os.path.join(os.path.dirname(__file__), 'parameters/elements/elements.dat')
            symbols  = np.genfromtxt(filename, dtype='U2', usecols = (0))
            elements = np.genfromtxt(filename, dtype='U15,i8,f8', usecols=(1,2,3))
            [rowIdx] = np.where(symbols == self.symbol)
            element = elements[rowIdx[0]]
        except Exception as e:
            print('Cannot load element specific data from elements data file!')
            print(e)

        self.name                   = element[0]
        self.atomicNumberZ          = element[1]
        self.massNumberA            = element[2]
        self.mass                   = self.massNumberA * constants.atomic_mass
        self.atomicFormFactorCoeff  = self.readAtomicFormFactorCoeff()
        self.cromerMannCoeff        = self.readCromerMannCoeff()

    def __str__(self):
        """String representation of this class

        """
        classStr  = 'Atom with the following properties\n'
        classStr += 'ID                 : {:s}\n'.format(self.ID)
        classStr += 'symbol             : {:s}\n'.format(self.symbol)
        classStr += 'name               : {:s}\n'.format(self.name)
        classStr += 'atomic number Z    : {:3.2f}\n'.format(self.atomicNumberZ)
        classStr += 'mass number   A    : {:3.2f} u\n'.format(self.massNumberA)
        classStr += 'mass               : {:3.2e} kg\n'.format(self.mass)
        classStr += 'ionicity           : {:3.2f}\n'.format(self.ionicity)
        classStr += 'Cromer Mann coeff  : {:s}\n'.format(np.array_str(self.cromerMannCoeff))
        return(classStr)

    def readAtomicFormFactorCoeff(self):
        """readAtomicFormFactorCoeff

        The atomic form factor $f$ in dependence from the energy $E$ is
        read from a parameter file given by Ref. [3].
        """
        filename = os.path.join(os.path.dirname(__file__), 'parameters/atomicFormFactors/{:s}.nff'.format(self.symbol.lower()))
        try:
            f = np.genfromtxt(filename, skip_header=1)
        except Exception as e:
            print('File {:s} not found!\nMake sure the path /parameters/atomicFormFactors/ is in your search path!', filename)
            print(e)

        return f

    def getAtomicFormFactor(self, E):
        """getAtomicFormFactor

        Returns the complex atomic form factor $f(E)=f_1-\i f_2$ for the
        energy $E$ [J].
        """
        E = E/constants.elementary_charge; # convert energy from [J] in [eV]
        # interpolate the real and imaginary part in dependence of E
        f1 = np.interp(E, self.atomicFormFactorCoeff[:,0],self.atomicFormFactorCoeff[:,1]);
        f2 = np.interp(E, self.atomicFormFactorCoeff[:,0],self.atomicFormFactorCoeff[:,2]);
        # Convention of Ref. [2] (p. 11, footnote) is a negative $f_2$
        return f1 - f2*1j;

    def readCromerMannCoeff(self):
        """readCromerMannCoeff

        The Cromer-Mann coefficients (Ref. [1]) are read from a parameter file and
        are returned in the following order:

        $$ a_1\; a_2\; a_3\; a_4\; b_1\; b_2\; b_3\; b_4\; c $$
        """
        filename = os.path.join(os.path.dirname(__file__), 'parameters/atomicFormFactors/cromermann.txt')
        try:
            cm = np.genfromtxt(filename, skip_header=1, usecols=(1,2,3,4,5,6,7,8,9,10,11))
        except Exception as e:
            print('File {:s} not found!\nMake sure the path /parameters/atomicFormFactors/ is in your search path!', filename)
            print(e)

        return cm[(cm[:,0] == self.atomicNumberZ) & (cm[:,1] == self.ionicity)][0]

    def getCMAtomicFormFactor(self, E, qz):
        """getAtomicFormFactor

        Returns the atomic form factor $f$ in dependence of the energy
        $E$ [J] and the $z$-component of the scattering vector $q_z$
        [m^-1] (Ref. [1]).
        Since the CM coefficients are fitted for $q_z$ in [Ang^-1]
        we have to convert it before!
        """
        qz = qz/constants.angstrom**-1; # qz in [Ang^-1]
        # See Ref. [2] (p. 235).
        #
        # $$f(q_z,E) = f_{CM}(q_z) + \delta f_1(E) -\i f_2(E)$$
        #
        # $f_{CM}(q_z)$ is given in Ref. 1:
        #
        # $$f_{CM}(q_z) = \sum(a_i \, \exp(-b_i \, (q_z/4\pi)^2))+ c$$
        # print(np.exp(-self.cromerMannCoeff[4:7].T * (qz/(4*np.pi))**2))
        f_CM = np.dot(self.cromerMannCoeff[0:3] ,
                np.exp(np.dot(-self.cromerMannCoeff[4:7], (qz/(4*np.pi))**2))) \
                + self.cromerMannCoeff[8];

        # $\delta f_1(E)$ is the dispersion correction:
        #
        # $$ \delta f_1(E) = f_1(E) - \left(\sum^4_i(a_i) + c\right)$$
        #
        # Thus:
        #
        # $$ f(q_z,E) = \sum(a_i \, \exp(b_i \, q_z/2\pi)) + c + f_1(E)-\i f_2(E) - \left(\sum(a_i) + c\right) $$
        #
        # $$ f(q_z,E) = \sum(a_i \, \exp(b_i \, q_z/2\pi)) + f_1(E) -\i f_2(E) - \sum(a_i) $$
        return f_CM + self.getAtomicFormFactor(E) \
            - (np.sum(self.cromerMannCoeff[0:3]) + self.cromerMannCoeff[8]);

class atomMixed(atom):
    """mixed atom

    The atomMixed class is sub class of atomBase and enables mixed atoms for
    certain alloys and stochiometric mixtures. All properties of the included
    sub-atoms of class atomBase are averaged and weighted with their
    stochiometric ratio

    Attributes:
        symbol (str)                 : symbol of the element
        ID (str)                     :
            identifier of the atom, may be different from symbol and/or name
        name (str)                   : name of the element (generic)
        atomicNumberZ (int)          : Z atomic number
        massNumberA (float)          : A atomic mass number
        ionicity (int)               : ionicity of the atom
        mass (float)                 : mass of the atom [kg]
        atomicFormFactorCoeff (ndarray[float]) :
            atomic form factor coefficients for energy-dependent atomic
            form factor
        cromerMannCoeff (ndarray[float])       :
            cromer-mann coefficients for angular-dependent atomic form factor
    """

    def __init__(self, symbol, **kwargs):
        """Initialize the class, set all file names and load the spec file.

        Args:
            name (str)                  : Name of the spec file.
            filePath (str)              : Base path of the spec and HDF5 files.
            specFileExt (Optional[str]) : File extension of the spec file,
                                          default is none.

        """
        self.symbol         = symbol
        self.ID             = kwargs.get('ID', symbol)
        self.name           = kwargs.get('name', symbol)
        self.ionicity       = 0
        self.atomicNumberZ  = 0
        self.massNumberA    = 0
        self.mass           = 0
        self.atoms          = []
        self.numAtoms       = 0
        self.cromerMannCoeff= np.array([])

    def __str__(self):
        """String representation of this class

        """
        classStr  = super().__str__()
        classStr  += '{:d} Constituents:\n'.format(self.numAtoms)
        for i in range(self.numAtoms):
            classStr  += '\t {:s} \t {:3.2f}%\n'.format(self.atoms[i][0].name, self.atoms[i][1]*100)

        return(classStr)


    def addAtom(self, atom,fraction):
        """addAtom

        Add a atomBase instance with its stochiometric fraction to the
        atomMixed instance.
        """
        self.atoms.append([atom, fraction])
        self.numAtoms = self.numAtoms + 1
        # calculate the mixed atomic properties of the atomMixed
        # instance
        self.atomicNumberZ    = self.atomicNumberZ     + fraction * atom.atomicNumberZ;
        self.massNumberA      = self.massNumberA       + fraction * atom.massNumberA;
        self.mass             = self.mass              + fraction * atom.mass;
        self.ionicity         = self.ionicity          + fraction * atom.ionicity;

    def getAtomicFormFactor(self, E):
        """getAtomicFormFactor

        Returns the mixed energy dependent atomic form factor.
        """
        f = 0
        for i in range(self.numAtoms):
            f += self.atoms[i][0].getAtomicFormFactor(E) * self.atoms[i][1]

        return f

    def getCMAtomicFormFactor(self,E,qz):
        """getCMAtomicFormFactor

        Returns the mixed energy and angle dependent atomic form factor.
        """
        f = 0
        for i in range(self.numAtoms):
            f += self.atoms[i][0].getCMAtomicFormFactor(E,qz) * self.atoms[i][1]

        return f

# References
#
# # D. T. Cromer & J. B. Mann (1968). _X-ray scattering factors computed from
# numerical Hartree–Fock wave functions_. Acta Crystallographica Section A,
# 24(2), 321–324. doi:10.1107/S0567739468000550
# # J. Als-Nielson, & D. McMorrow (2001). _Elements of Modern X-Ray
# Physics_. New York: John Wiley & Sons, Ltd. doi:10.1002/9781119998365
# # B. L. Henke, E. M. Gullikson & J. C. Davis (1993). _X-Ray Interactions:
# Photoabsorption, Scattering, Transmission, and Reflection at
# E = 50-30,000 eV, Z = 1-92_. Atomic Data and Nuclear Data Tables, 54(2),
# 181–342. doi:10.1006/adnd.1993.1013
