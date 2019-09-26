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

"""A :mod:`Structure` module """

__all__ = ["Structure"]

__docformat__ = "restructuredtext"

import itertools
import numpy as np
from .unitCell import UnitCell
from . import u


class Structure:
    """Structure
    The structure class can hold various sub_structures.
    Each sub_structure can be either a layer of N unitCell objects or a structure by itself.
    Thus it is possible to recursively build up 1D structures.
    """

    def __init__(self, name):
        """
        Properties (SetAccess=public,GetAccess=public)
        name                % STRING name of sample
        sub_structures = []; % CELL ARRAY of structures in sample
        substrate           % OBJECT HANDLE structure of the substrate
        num_sub_systems = 1;  % INTEGER number of subsystems for heat and phonons
        (electronic, lattice, spins, ...)
        """
        self.name = name
        self.num_sub_systems = 1
        self.sub_structures = []
        self.substrate = []

    def addsub_structure(self, sub_structure, N):
        """Add a sub_structure of N unitCells or N structures to the structure."""

        # check of the sub_structure is an instance of the unitCell of structure class

        if not isinstance(sub_structure, UnitCell, Structure):
            raise ValueError('Class ' + type(sub_structure).__name__ +
                             ' is no possible sub structure. Only UnitCell and'
                             'Structure classes are allowed!')

        # if a structure is added as a sub_structure, the sub_structure can not have a substrate
        if isinstance(sub_structure, Structure):
            if sub_structure.substrate:
                raise ValueError('No substrate in sub_structure allowed!')

        # check the number of subsystems of the sub_structure

        if (self.num_sub_systems > 1) and not (sub_structure.num_sub_systems ==
                                               self.num_sub_systems):
            raise ValueError('The number of subsystems in each sub_structure must be the same!')
        else:
            self.num_sub_systems = sub_structure.num_sub_systems

        # add a sub_structure of N repetitions to the structure with
        self.sub_structures.append([sub_structure, N])

    def addSubstrate(self, sub_structure):
        """Add a structure as static substrate to the structure"""

        if not isinstance(sub_structure, Structure):
            raise ValueError('Class ' + type(sub_structure).__name__ +
                             ' is no possible substrate. Only structure class is allowed!')

        self.substrate = sub_structure

    def getNumberOfsub_structures(self):
        """Returns the number of all sub structures.
        This methods does not return the number of all unitCells in the structure,
        see getNumberOfUnitCells()."""

        N = 0
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], UnitCell):
                N = N + 1
            else:
                N = N + self.sub_structures[i][0].getNumberOfsub_structures()
        return N

    def getNumberOfUnitCells(self):
        """Returns the number of all unitCells in the structure."""
        N = 0
        # traverse the substructres
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], UnitCell):
                N = N + self.sub_structures[i][1]
            else:
                # its a sturcture, so call the method recursively
                N = N + self.sub_structures[i][0].getNumberOfUnitCells()*self.sub_structures[i][1]

        return N

    def getNumberOfUniqueUnitCells(self):
        """Returns the number of unique unitCells in the structure."""

        N = len(self.getUniqueUnitCells()[0])
        return N

    def getLength(self):
        """Returns the length from surface to bottom of the structure"""

        pass

    def getUniqueUnitCells(self):
        """Returns a cell array of ids and handles of all unique UnitCell instances in the
        structure.
        The uniqueness is determined by the handle of each unitCell instance."""

        uc_ids = []
        uc_handles = []
        # traverse the sub_structures
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], UnitCell):
                # its a UnitCell
                uc_id = self.sub_structures[i][0].id
                if not uc_ids:
                    # the cell array is empty at the beginning so add
                    # the first unitCell
                    uc_ids = uc_ids + [uc_id]
                    uc_handles = uc_handles + [self.sub_structures[i][0]]
                else:
                    # the cell array is not empty so check if the id is
                    # already in the ucs id vector
                    if uc_id not in uc_ids:
                        # if id not in list, so add it
                        uc_ids = uc_ids + [uc_id]
                        uc_handles = uc_handles + [self.sub_structures[i][0]]
            else:
                # its a sub_structure
                if not uc_ids:
                    # the cell array is empty at the beginning so call
                    # the method recursively and add the result to the
                    # ucs array
                    uc_ids = self.sub_structures[i][0].getUniqueUnitCells()[0]
                    uc_handles = self.sub_structures[i][0].getUniqueUnitCells()[1]
                else:
                    # the cell array is not empty so check if the ids
                    # from the recursive call are already in the ucs id
                    # vector.
                    temp1 = self.sub_structures[i][0].getUniqueUnitCells()[0]
                    temp2 = self.sub_structures[i][0].getUniqueUnitCells()[1]
                    for j, temp in enumerate(temp1):
                        # check all ids from recursive call
                        if temp1[j] not in uc_ids:
                            # ids not in list, so add them
                            uc_ids = uc_ids + [temp1[j]]
                            uc_handles = uc_handles + [temp2[j]]

        return uc_ids, uc_handles

    def getUnitCellVectors(self, *args):
        """Returns three vectors with the numeric index of all unit cells in a structure given by
        the getUniqueUnitCells() method and addidionally vectors with the ids and Handles of the
        corresponding unitCell instances.
        The list and order of the unique unitCells can be either handed as an input parameter or
        is requested at the beginning."""

        indices = []
        uc_ids = []
        uc_handles = []
        # if no ucs (UniqueUnitCells) are given, we have to get them
        if len(args) < 1:
            ucs = self.getUniqueUnitCells()
        else:
            ucs = args[0]
        # traverse the substructres
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], UnitCell):
                # its a UnitCell
                # find the index of the current UC id in the unique
                # unitCell vector
                Index = ucs[0].index(self.sub_structures[i][0].id)
                # add the index N times to the indices vector
                indices = np.append(indices, Index*np.ones(self.sub_structures[i][1]))
                # create a cell array of N unitCell ids and add them to
                # the ids cell array
                temp1 = list(itertools.repeat(self.sub_structures[i][0].id,
                                              self.sub_structures[i][1]))
                uc_ids = uc_ids + list(temp1)
                # create a cell array of N unitCell handles and add them to
                # the Handles cell array
                temp2 = list(itertools.repeat(self.sub_structures[i][0], self.sub_structures[i][1]))
                uc_handles = uc_handles + list(temp2)
            else:
                # its a structure
                # make a recursive call and hand in the same unique
                # unit cell vector as we used before
                [temp1, temp2, temp3] = self.sub_structures[i][0].getUnitCellVectors(ucs)
                temp11 = []
                temp22 = []
                temp33 = []
                # concat the temporary arrays N times
                for j in range(self.sub_structures[i][1]):
                    temp11 = temp11 + list(temp1)
                    temp22 = temp22 + list(temp2)
                    temp33 = temp33 + list(temp3)
                # add the temporary arrays to the outputs
                indices = np.append(indices, temp11)
                uc_ids = uc_ids + list(temp22)
                uc_handles = uc_handles + list(temp33)
        return indices, uc_ids, uc_handles

    def getAllPositionsPerUniqueUnitCell(self):
        """Returns a cell array with one vector of position indices for each unique unitCell
        in the structure."""

        ucs = self.getUniqueUnitCells()
        indices = self.getUnitCellVectors()[0]
        pos = {}  # Dictionary used instead of array
        for i, uc in enumerate(ucs[0]):
            pos[ucs[0][i]] = list(np.where(indices == i))
        # Each element accessible through Unit cell id
        return pos

    def getDistancesOfUnitCells(self):
        """Returns a vector of the distance from the surface for each unit cell starting at 0
        (dStart) and starting at the end of the first UC (dEnd) and from the center of each UC
        (dMid)."""

        c_axes = self.getUnitCellPropertyVector(types='cAxis')
        d_end = np.cumsum(c_axes)
        d_start = np.hstack([[0], d_end[0:-1]])
        d_mid = (d_start + c_axes)/2
        return d_start, d_end, d_mid

    def getUnitCellPropertyVector(self, **kwargs):
        """Returns a vector for a property of all unitCells in the structure.
        The property is determined by the propertyName and returns a scalar value or a function
        handle."""

        types = kwargs.get('types')

        # get the Handle to all unitCells in the Structure
        handles = self.getUnitCellVectors()[2]

        if callable(getattr(handles[0], types)):
            prop = np.zeros([self.getNumberOfUnitCells()])
            for i in range(self.getNumberOfUnitCells()):
                prop[i] = getattr(handles[i], types)
        elif type(getattr(handles[0], types)) is list:
            # Prop = np.zeros([self.getNumberOfUnitCells(),len(getattr(Handles[0],types))])
            # Prop[]
            prop = {}
            for i in range(self.getNumberOfUnitCells()):
                # Prop = Prop + getattr(Handles[i],types)
                prop[i] = getattr(handles[i], types)
        else:
            # ucs = self.getUniqueUnitCells()
            prop = np.zeros([self.getNumberOfUnitCells()])

            # traverse all unitCells
            for i in range(self.getNumberOfUnitCells()):
                temp = getattr(handles[i], types)
                prop[i] = temp

        return prop

    def getUnitCellHandle(self, i):
        """Returns the handle to the unitCell at position i in the structure."""

        handles = self.getUnitCellVectors()[2]
        handle = handles[i]
        return handle
