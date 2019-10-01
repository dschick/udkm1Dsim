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


class Structure:
    """Structure

    The structure class can hold various sub_structures. Each
    sub_structure can be either a layer of N unitCell objects or a
    structure by itself.
    Thus it is possible to recursively build up 1D structures.

    Args:
        name (str): name of the sample

    Attributes:
        name (str): name of sample
        sub_structures (list): list of structures in sample
        substrate (object): structure of the substrate
        num_sub_systems (int): number of subsystems for heat and phonons
           (electronic, lattice, spins, ...)

    """

    def __init__(self, name):
        self.name = name
        self.num_sub_systems = 1
        self.sub_structures = []
        self.substrate = []

    def __str__(self):
        """String representation of this class"""
        return "hello world"
#        % check for the numbers of tabs added at the beginning
#            if nargin < 2
#                tabs = 0;
#            end%if
#            tabstr = '';
#            for i = 1:tabs
#                tabstr = strcat(tabstr, '\t');
#            end%for
#            % plot the properties
#            fprintf([tabstr 'Structure properties:\n']);
#            fprintf([tabstr 'Name  : %s\n'], obj.name);
#            fprintf([tabstr 'Length: %0.2f nm\n'], obj.getLength()/units.nm);
#            fprintf([tabstr '----\n']);
#            % traverse all substructures
#            for i = 1:size(obj.substructures,1)
#                if isa(obj.substructures{i,1},'unitCell')
#                    % the substructure is an unitCell
#                    fprintf([tabstr '%d times %s: %0.2f nm\n'], obj.substructures{i,2},...
#                        obj.substructures{i,1}.name,
#                        obj.substructures{i,2}*obj.substructures{i,1}.cAxis/units.nm);
#                else
#                    % the substructure is a structure instance by itself
#                    % call the display() method recursively
#                    fprintf([tabstr 'SubStructure %d times:\n'],...
#                        obj.substructures{i,2});
#                    obj.substructures{i,1}.displayTab(tabs+1);
#                end%if
#            end%for
#            fprintf([tabstr '----\n']);
#            % check for a substrate
#            if isa(obj.substrate,'structure')
#                fprintf([tabstr 'Substrate:\n']);
#                fprintf([tabstr '----\n']);
#                fprintf([tabstr '%d times %s: %0.2f nm\n'], obj.substrate.substructures{1,2},...
#                    obj.substrate.substructures{1,1}.name, ...
#                    obj.substrate.substructures{1,2}*obj.substrate.substructures{1,1}.cAxis/units.nm);
#            end%if
#        output = [['id', self.id],
#                  ['name', self.name],
#                  ['a-axis', '{:.4~P}'.format(self.a_axis)],
#                  ['b-axis', '{:.4~P}'.format(self.b_axis)],
#                  ['c-axis', '{:.4~P}'.format(self.c_axis)],
#                  ['area', '{:.4~P}'.format(self.area.to('angstrom**2'))],
#                  ['volume', '{:.4~P}'.format(self.volume.to('angstrom**3'))],
#                  ['mass', '{:.4~P}'.format(self.mass)],
#                  ['density', '{:.4~P}'.format(self.density.to('kg/meter**3'))],
#                  ['Debye Waller Factor', self.deb_wal_fac.to('meter**2')],
#                  ['sound velocity', '{:.4~P}'.format(self.sound_vel.to('meter/s'))],
#                  ['spring constant', self.spring_const * u.kg/u.s**2],
#                  ['phonon damping', self.phonon_damping.to('kg/s')],
#                  ['opt. pen. depth', self.opt_pen_depth.to('nm')],
#                  ['opt. refractive index', self.opt_ref_index],
#                  ['opt. ref. index/strain', self.opt_ref_index_per_strain],
#                  ['thermal conduct.', ' W/(m K)\n'.join(self.therm_cond_str) + ' W/(m K)'],
#                  ['linear thermal expansion', '\n'.join(self.lin_therm_exp_str)],
#                  ['heat capacity', ' J/(kg K)\n'.join(self.heat_capacity_str) + ' J/(kg K)'],
#                  ['subsystem coupling', ' W/m³\n'.join(self.sub_system_coupling_str) + ' W/m³']]
#
#        class_str = 'Unit Cell with the following properties\n\n'
#        class_str += tabulate(output, headers=['parameter', 'value'], tablefmt="rst",
#                              colalign=('right',), floatfmt=('.2f', '.2f'))
#        class_str += '\n\n' + str(self.num_atoms) + ' Constituents:\n'
#
#        atoms_str = []
#        for i in range(self.num_atoms):
#            atoms_str.append([self.atoms[i][0].name,
#                              '{:0.2f}'.format(self.atoms[i][1](0)),
#                              self.atoms[i][2]])
#        class_str += tabulate(atoms_str, headers=['atom', 'position', 'position function'],
#                              tablefmt="rst")

    def add_sub_structure(self, sub_structure, N):
        """add_sub_structure

        Add a sub_structure of N unitCells or N structures to the
        structure.

        Args:
            sub_structure (UnitCell, Structure): unit cell or structure
               to add as sub structure
            N (int): number or repetitions
        """

        # check of the sub_structure is an instance of the unitCell of
        # structure class
        if not isinstance(sub_structure, (UnitCell, Structure)):
            raise ValueError('Class '
                             + type(sub_structure).__name__
                             + ' is no possible sub structure.'
                             + 'Only UnitCell and'
                             + 'Structure classes are allowed!')

        # if a structure is added as a sub_structure, the sub_structure
        # can not have a substrate
        if isinstance(sub_structure, Structure):
            if sub_structure.substrate:
                raise ValueError('No substrate in sub_structure allowed!')

        # check the number of subsystems of the sub_structure
        if ((self.num_sub_systems > 1)
           and not (sub_structure.num_sub_systems == self.num_sub_systems)):

            raise ValueError('The number of subsystems in each sub_structure'
                             'must be the same!')
        else:
            self.num_sub_systems = sub_structure.num_sub_systems

        # add a sub_structure of N repetitions to the structure with
        self.sub_structures.append([sub_structure, N])

    def add_substrate(self, sub_structure):
        """add_substrate

        Add a structure as static substrate to the structure

        Args:
            sub_structure (Structure): substrate structure
        """
        if not isinstance(sub_structure, Structure):
            raise ValueError('Class '
                             + type(sub_structure).__name__
                             + ' is no possible substrate. '
                             + 'Only structure class is allowed!')

        self.substrate = sub_structure

    def get_number_of_sub_structures(self):
        """get_number_of_sub_structures

        Returns the number of all sub structures.
        This methods does not return the number of all unitCells in the
        structure, see get_number_of_unit_cells().

        """
        N = 0
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], UnitCell):
                N = N + 1
            else:
                N = N + self.sub_structures[i][0].getNumberOfsub_structures()
        return N

    def get_number_of_unit_cells(self):
        """get_number_of_unit_cells

        Returns the number of all unitCells in the structure.

        """
        N = 0
        # traverse the substructres
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], UnitCell):
                N = N + self.sub_structures[i][1]
            else:
                # its a sturcture, so call the method recursively
                N = N + self.sub_structures[i][0].getNumberOfUnitCells()*self.sub_structures[i][1]

        return N

    def get_number_of_unique_unit_cells(self):
        """get_number_of_unique_unit_cells

        Returns the number of unique unitCells in the structure.

        """
        N = len(self.get_unique_unit_cells()[0])
        return N

    def get_length(self):
        """get_length

        Returns the length from surface to bottom of the structure

        """
        pass

    def get_unique_unit_cells(self):
        """get_unique_unit_cells

        Returns a list of ids and handles of all unique UnitCell
        instances in the structure.
        The uniqueness is determined by the handle of each unitCell
        instance.

        """

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
                    uc_ids = self.sub_structures[i][0].get_unique_unit_cells()[0]
                    uc_handles = self.sub_structures[i][0].get_unique_unit_cells()[1]
                else:
                    # the cell array is not empty so check if the ids
                    # from the recursive call are already in the ucs id
                    # vector.
                    temp1 = self.sub_structures[i][0].get_unique_unit_cells()[0]
                    temp2 = self.sub_structures[i][0].get_unique_unit_cells()[1]
                    for j, temp in enumerate(temp1):
                        # check all ids from recursive call
                        if temp1[j] not in uc_ids:
                            # ids not in list, so add them
                            uc_ids = uc_ids + [temp1[j]]
                            uc_handles = uc_handles + [temp2[j]]

        return uc_ids, uc_handles

    def get_unit_cell_vectors(self, *args):
        """get_unit_cell_vectors

        Returns three lists with the numeric index of all unit cells
        in a structure given by the get_unique_unit_cells() method and
        addidionally vectors with the ids and Handles of the
        corresponding unitCell instances.
        The list and order of the unique unitCells can be either handed
        as an input parameter or is requested at the beginning.

        Args:
            ucs (Optional[list]): list of unique unit cells including
               ids and handles

        """
        indices = []
        uc_ids = []
        uc_handles = []
        # if no ucs (UniqueUnitCells) are given, we have to get them
        if len(args) < 1:
            ucs = self.get_unique_unit_cells()
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
                temp2 = list(itertools.repeat(self.sub_structures[i][0],
                                              self.sub_structures[i][1]))
                uc_handles = uc_handles + list(temp2)
            else:
                # its a structure
                # make a recursive call and hand in the same unique
                # unit cell vector as we used before
                [temp1, temp2, temp3] = self.sub_structures[i][0].get_unit_cell_vectors(ucs)
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

    def get_all_positions_per_unique_unit_cell(self):
        """get_all_positions_per_unique_unit_cell

        Returns a list with one vector of position indices for
        each unique unitCell in the structure.

        """
        ucs = self.get_unique_unit_cells()
        indices = self.get_unit_cell_vectors()[0]
        pos = {}  # Dictionary used instead of array
        for i, uc in enumerate(ucs[0]):
            pos[ucs[0][i]] = list(np.where(indices == i))
        # Each element accessible through Unit cell id
        return pos

    def get_distances_of_unit_cells(self):
        """get_distances_of_unit_cells

        Returns a vector of the distance from the surface for each unit
        cell starting at 0 (dStart) and starting at the end of the first
        unit cell (dEnd) and from the center of each unit cell (dMid).

        """
        c_axes = self.get_unit_cell_property_vector('_c_axis')
        d_end = np.cumsum(c_axes)
        d_start = np.hstack([[0], d_end[0:-1]])
        d_mid = (d_start + c_axes)/2
        return d_start, d_end, d_mid

    def get_unit_cell_property_vector(self, property_name):
        """get_unit_cell_property_vector

        Returns a vector for a property of all unitCells in the
        structure. The property is determined by the propertyName and
        returns a scalar value or a function handle.

        Args:
            property_name (str): type of property to return as vector

        """
        # get the Handle to all unitCells in the Structure
        handles = self.get_unit_cell_vectors()[2]

        if callable(getattr(handles[0], property_name)):
            # it's a function
            print("it's a function")
            prop = np.zeros([self.get_number_of_unit_cells()])
            for i in range(self.get_number_of_unit_cells()):
                prop[i] = getattr(handles[i], property_name)
        elif ((type(getattr(handles[0], property_name)) is list) or
                (type(getattr(handles[0], property_name)) is str)):
            # it's a list if functions or str
            prop = []
            for i in range(self.get_number_of_unit_cells()):
                # Prop = Prop + getattr(Handles[i],types)
                prop.append(getattr(handles[i], property_name))
        else:
            # its a number or array
            ucs = self.get_unique_unit_cells()
            temp = np.zeros([len(ucs[0]), 1])
            for i, uc in enumerate(ucs[1]):
                temp[i] = len(getattr(uc, property_name))
            prop = np.zeros([self.get_number_of_unit_cells(), int(np.max(temp))])
            del temp
            # traverse all unitCells
            for i in range(self.get_number_of_unit_cells()):
                temp = getattr(handles[i], property_name)
                prop[i, 0:len(temp)] = temp

        return prop

    def get_unit_cell_handle(self, i):
        """get_unit_cell_handle

        Returns the handle to the unitCell at position i in the
        structure.

        """
        handles = self.get_unit_cell_vectors()[2]
        handle = handles[i]
        return handle
