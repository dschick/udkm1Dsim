#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

__all__ = ['Structure']

__docformat__ = 'restructuredtext'

from .layers import Vacuum, Layer
from .. import u, Q_
from ..helpers import make_hash_md5, finderb
import itertools
import numpy as np
import warnings


class Structure:
    """Structure

    Structure (sample) which holds various `sub_structures`.

    Each `sub_structure` can be either of type :class:`Layer` and its sub-classes,
    such as :class:`UnitCell` or :class:`AmorphousLayer`, or a :class:`Structure`
    by itself. It is possible to recursively build up samples.

    Args:
        name (str): name of the sample.

    Attributes:
        name (str): name of sample.
        thickness (float): thickness of the structure [m].
        sub_structures (list[Layer, Structure]): list of sub-structures in sample.
        superstrate (Layer, Structure): Layer or its sub-classes, or Structure forming
            the superstrate.
        substrate (Layer, Structure): Layer or its sub-classes, or Structure forming the substrate.
        num_sub_systems (int): number of subsystems for heat and phonons
            (electronic, lattice, spins, ...).

    """

    def __init__(self, name):
        self.name = name
        self.num_sub_systems = 1
        self.sub_structures = []
        self.superstrate = Vacuum()
        self.substrate = Vacuum()

    def __str__(self, tabs=0, recursive=False):
        """String representation of this class"""
        tab_str = tabs*'\t'

        class_str = tab_str + 'Structure properties:\n\n'
        class_str += tab_str + 'Name   : {:s}\n'.format(self.name)

        if len(self.sub_structures) == 0:
            class_str += tab_str + 'Structure is empty\n----\n'
            return class_str

        class_str += tab_str + 'Thickness : {:.4g~P}\n'.format(self.thickness.to('nm'))
        class_str += tab_str + '----\n'
        # traverse all substructures
        for sub_structure in self.sub_structures:
            if isinstance(sub_structure[0], Layer):
                # the substructure is a Layer or sub-class
                class_str += tab_str + '{:d} times {:s}: {:.4g~P}\n'.format(
                        sub_structure[1],
                        sub_structure[0].name,
                        sub_structure[1]*sub_structure[0].thickness.to('nm'))
            else:
                # the substructure is a Structure instance by itself
                # call the display() method recursively
                class_str += tab_str + 'sub-structure {:d} times:\n'.format(
                       sub_structure[1])
                class_str += sub_structure[0].__str__(tabs+1, recursive=True)
        class_str += tab_str + '----\n'
        # do not print substrate sub sub-structures
        if not recursive:
            class_str += tab_str + 'Superstrate (semi-infinite):\n'
            class_str += tab_str + '----\n'
            # check for a superstrate
            if isinstance(self.superstrate, Structure):
                class_str += tab_str + '{:d} times {:s}: {:.4g~P}\n'.format(
                        self.superstrate.sub_structures[0][1],
                        self.superstrate.sub_structures[0][0].name,
                        self.superstrate.sub_structures[0][1]
                        * self.superstrate.sub_structures[0][0].thickness.to('nm'))
            elif isinstance(self.superstrate, Layer):
                class_str += tab_str + '{:s} \n'.format(self.superstrate.name)
            else:
                warnings.warn('There should be a superstrate present!')
            class_str += tab_str + '----\n'
            class_str += tab_str + 'Substrate (semi-infinite):\n'
            class_str += tab_str + '----\n'
            # check for a substrate
            if isinstance(self.substrate, Structure):
                class_str += tab_str + '{:d} times {:s}: {:.4g~P}\n'.format(
                        self.substrate.sub_structures[0][1],
                        self.substrate.sub_structures[0][0].name,
                        self.substrate.sub_structures[0][1]
                        * self.substrate.sub_structures[0][0].thickness.to('nm'))
            elif isinstance(self.substrate, Layer):
                class_str += tab_str + '{:s} \n'.format(self.substrate.name)
            else:
                warnings.warn('There should be a substrate present!')
        return class_str

    def visualize(self, block=True, unit='nm', fig_size=[20, 1], cmap='Set3', linewidth=0.1,
                  show=True):
        """visualize

        Simple visualization of the :class:`Structure`.

        Args:
            unit (str): SI unit of the distance of the Structure. Defaults to
                'nm'.
            fig_size (list[float]): figure size of the visualization plot.
                Defaults to [20, 1].
            cmap (str): Matplotlib colormap for colors of Layer.
            linewidth (float): line width of the patches.
            show (boolean): show visualization plot at the end.

        """
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib import colormaps

        _, d_end, _ = self.get_distances_of_layers(True)  # distance vector of all layers
        layer_interfaces = np.append(0, d_end.to(unit).magnitude)  # Append zero at the start
        thickness = np.max(layer_interfaces)

        layer_ids = self.get_unique_layers()[0]
        N = len(layer_ids)  # number of unique layers

        cm = colormaps[cmap]
        colortable = {}
        for i in range(N):
            colortable[layer_ids[i]] = cm(i/(N-1))

        plt.figure(figsize=fig_size)
        ax = plt.axes()

        for i, name in enumerate(self.get_layer_vectors()[1]):
            col = colortable.get(name, 'k')
            rect = patches.Rectangle((layer_interfaces[i], 0), np.diff(layer_interfaces)[i], 1,
                                     linewidth=linewidth, facecolor=col, edgecolor='k')
            ax.add_patch(rect)

        plt.xlim(0, thickness)
        plt.ylim(0, 1)
        plt.xlabel('Distance ({:s})'.format(unit))
        plt.yticks([], [])

        # add labels for legend
        for layer_id, col in colortable.items():
            plt.plot(0, 0, color=col, label=layer_id)

        leg = plt.legend(bbox_to_anchor=(0., 1.08, 1, .102), frameon=False, ncol=8)

        for line in leg.get_lines():
            line.set_linewidth(8.0)

        if show:
            plt.show(block=block)

    def get_hash(self, **kwargs):
        """get_hash

        Create an unique hash from all layer IDs in the correct order in the
        Structure as well as the corresponding material properties which are
        given by the `kwargs`.

        `types='all'` is problematic, as function handles will be include,
        which will always change on recreation. Following errors from the
        GitHub test-suite, hashes do not seem to match across different OS
        and/or python versions.

        Args:
            **kwargs (list[str]): types of requested properties..

        Returns:
            hash (str): unique hash.

        """
        param = []
        layers = self.get_unique_layers()
        for layer in layers[1]:
            param.append(layer.get_property_dict(**kwargs))

        _, IDs, _ = self.get_layer_vectors()
        param.append(IDs)
        return make_hash_md5(param)

    def add_sub_structure(self, sub_structure, N=1):
        """add_sub_structure

        Add :math:`N` sub_structures of type :class:`Layer` or its sub-classes,
        or :class:`Structure`.

        Args:
            sub_structure (Layer, Structure):
               Layer or its sub-classes, or Structure to add as sub-structure.
            N (int): number or repetitions.

        """
        # check of the sub_structure is of type Layer
        # or its sub-classes, or Structure
        if not isinstance(sub_structure, (Layer, Structure)):
            raise TypeError('Class '
                            + type(sub_structure).__name__
                            + ' is no possible sub structure. '
                            + 'Only Layer, its sub-classes, and '
                            + 'Structure classes are allowed!')

        # if a Structure is added as a sub_structure, the sub_structure's
        # substrate is ignored (Vacuum default is ignored)
        if isinstance(sub_structure, Structure):
            if not isinstance(sub_structure.substrate, (Vacuum)):
                warnings.warn('The substrate of the sub_structure is ignored.')

        # check the number of subsystems of the sub_structure
        if ((self.num_sub_systems > 1)
           and not (sub_structure.num_sub_systems == self.num_sub_systems)):

            raise ValueError('The number of subsystems in each sub_structure'
                             'must be the same!')
        else:
            self.num_sub_systems = sub_structure.num_sub_systems

        # add a sub_structure of N repetitions to the structure with
        self.sub_structures.append([sub_structure, N])

    def add_superstrate(self, sub_structure):
        """add_superstrate

        Add :class:`Layer` or its sub-classes, or :class:`Structure` as static superstrate
        to the sample.

        Args:
            sub_structure (Layer, Structure): superstrate Layer or its sub-classes, or Structure.

        """
        if not isinstance(sub_structure, (Layer, Structure)):
            raise TypeError('Class '
                            + type(sub_structure).__name__
                            + ' is no possible superstrate. '
                            + 'Only Layer or its sub-classes, or Structure '
                            + 'class is allowed!')

        self.substrate = sub_structure

    def add_substrate(self, sub_structure):
        """add_substrate

        Add :class:`Layer` or its sub-classes, or :class:`Structure` as static substrate
        to the sample.

        Args:
            sub_structure (Layer, Structure): substrate Layer or its sub-classes, or Structure.

        """
        if not isinstance(sub_structure, (Layer, Structure)):
            raise TypeError('Class '
                            + type(sub_structure).__name__
                            + ' is no possible substrate. '
                            + 'Only Layer or its sub-classes, or Structure '
                            + 'class is allowed!')

        self.substrate = sub_structure

    def get_number_of_sub_structures(self):
        """get_number_of_sub_structures

        This methods does not return the number of all :class:`Layer` in the
        :class:`Structure`, see :meth:`.get_number_of_layers`.

        Returns:
            N (int): number of all sub structures.

        """
        N = 0
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], Layer):
                N = N + 1
            else:
                N = N + self.sub_structures[i][0].get_number_of_sub_structures()
        return N

    def get_number_of_layers(self):
        """get_number_of_layers

        Determines the number of all :class:`Layer` in the :class:`Structure`.

        Returns:
            L (int): number of all Layer in the Structure.

        """
        L = 0
        # traverse the substructures
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], Layer):
                L = L + self.sub_structures[i][1]
            else:
                # its a structure, so call the method recursively
                L = L + self.sub_structures[i][0].get_number_of_layers() \
                    * self.sub_structures[i][1]

        return L

    def get_number_of_unique_layers(self):
        """get_number_of_unique_layers

        Determines the number of unique :class:`Layer` in the :class:`Structure`.

        Returns:
            N (int): number of unique layers in the Structure.

        """
        N = len(self.get_unique_layers()[0])
        return N

    def get_unique_layers(self):
        """get_unique_layers

        The uniqueness is determined by the handle of each :class:`Layer`.

        Returns:
            (tuple):
            - *layer_ids (list[str])* - ids of all unique Layer instances in
              the Structure.
            - *layer_handles (list[Layer, Structure])* -
              handles of all unique Layer instances in the Structure.

        """
        layer_ids = []
        layer_handles = []
        # traverse the sub_structures
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], Layer):
                # its a AmorphousLayer or UnitCell
                layer_id = self.sub_structures[i][0].id
                if not layer_ids:
                    # the list is empty at the beginning so add
                    # the first layer
                    layer_ids = layer_ids + [layer_id]
                    layer_handles = layer_handles + [self.sub_structures[i][0]]
                else:
                    # the list is not empty so check if the id is
                    # already in the layers id list
                    if layer_id not in layer_ids:
                        # if id not in list, so add it
                        layer_ids = layer_ids + [layer_id]
                        layer_handles = layer_handles + [self.sub_structures[i][0]]
            else:
                # its a sub_structure
                if not layer_ids:
                    # the list is empty at the beginning so call
                    # the method recursively and add the result to the
                    # layers list
                    layer_ids = self.sub_structures[i][0].get_unique_layers()[0]
                    layer_handles = self.sub_structures[i][0].get_unique_layers()[1]
                else:
                    # the list is not empty so check if the ids
                    # from the recursive call are already in the layers id
                    # list.
                    temp1 = self.sub_structures[i][0].get_unique_layers()[0]
                    temp2 = self.sub_structures[i][0].get_unique_layers()[1]
                    for j in range(len(temp1)):
                        # check all ids from recursive call
                        if temp1[j] not in layer_ids:
                            # ids not in list, so add them
                            layer_ids = layer_ids + [temp1[j]]
                            layer_handles = layer_handles + [temp2[j]]

        return layer_ids, layer_handles

    def get_layer_vectors(self, *args):
        """get_layer_vectors

        Returns three lists with the numeric index of all :class:`Layer`
        in a :class:`Structure` given by the :meth:`get_unique_layers()`
        method and additionally vectors with the ids and Handles of the
        corresponding layer instances.
        The list and order of the unique :class:`Layer` can be either handed
        as an input parameter or is requested at the beginning.

        Args:
            layers (Optional[list]): list of unique Layer including
               ids and handles

        Returns:
            (tuple):
            - *indices (list[int])* - numeric index of all Layer in a
              Structure.
            - *layer_ids (list[str])* - ids of all unique Layer instances in
              the Structure.
            - *layer_handles (list[Layer, Structure])* -
              handles of all unique Layers instances in the Structure.

        """
        indices = []
        layer_ids = []
        layer_handles = []
        # if no layers (unique layers) are given, we have to get them
        if len(args) < 1:
            layers = self.get_unique_layers()
        else:
            layers = args[0]
        # traverse the substructres
        for i in range(len(self.sub_structures)):
            if isinstance(self.sub_structures[i][0], Layer):
                # its a Layer or sub-class
                # find the index of the current layer id in the unique
                # layer list
                Index = layers[0].index(self.sub_structures[i][0].id)
                # add the index N times to the indices vector
                indices = np.append(indices, Index*np.ones(self.sub_structures[i][1]))
                # create a list of N layer ids and add them to
                # the ids list
                temp1 = list(itertools.repeat(self.sub_structures[i][0].id,
                                              self.sub_structures[i][1]))
                layer_ids = layer_ids + list(temp1)
                # create a list of N layer handles and add them to
                # the Handles list
                temp2 = list(itertools.repeat(self.sub_structures[i][0],
                                              self.sub_structures[i][1]))
                layer_handles = layer_handles + list(temp2)
            else:
                # its a structure
                # make a recursive call and hand in the same unique
                # layer vector as we used before
                [temp1, temp2, temp3] = self.sub_structures[i][0].get_layer_vectors(layers)
                temp11 = []
                temp22 = []
                temp33 = []
                # concat the temporary arrays N times
                for _ in range(self.sub_structures[i][1]):
                    temp11 = temp11 + list(temp1)
                    temp22 = temp22 + list(temp2)
                    temp33 = temp33 + list(temp3)
                # add the temporary arrays to the outputs
                indices = np.append(indices, temp11)
                layer_ids = layer_ids + list(temp22)
                layer_handles = layer_handles + list(temp33)
        return indices, layer_ids, layer_handles

    def get_all_positions_per_unique_layer(self):
        """get_all_positions_per_unique_layer

        Determines the position indices for each unique :class:`Layer` in the :class:`Structure`.

        Returns:
            pos (dict{ndarray[int]}): position indices for each unique Layer in
            the Structure.

        """
        layers = self.get_unique_layers()
        indices = self.get_layer_vectors()[0]
        pos = {}  # Dictionary used instead of array
        for i, layer in enumerate(layers[0]):
            pos[layer] = np.flatnonzero(indices == i)
        # Each element accessible through layer id
        return pos

    def get_distances_of_layers(self, units=True):
        """get_distances_of_layers

        Returns a vector of the distance from the surface for each :class:`Layer`
        starting at 0 (`d_start`) and starting at the end of the first
        layer (`d_end`) and from the center of each layer (`d_mid`).

        Args:
            units (boolean, optional): whether units should be returned or not.
                Defaults to True.

        Returns:
            (tuple):
            - *d_start (ndarray[float, Quantity])* - distances from the surface
              of each Layer starting at 0.
            - *d_end (ndarray[float, Quantity])* - distances from the bottom
              of each Layer.
            - *d_mid (ndarray[float, Quantity])*: distance from the middle of
              each Layer.

        """
        thickness = self.get_layer_property_vector('_thickness')
        d_end = np.cumsum(thickness)
        d_start = np.hstack([[0], d_end[0:-1]])
        d_mid = (d_start + thickness/2)
        if units:
            return Q_(d_start, u.m), Q_(d_end, u.m), Q_(d_mid, u.m)
        else:
            return d_start, d_end, d_mid

    def get_distances_of_interfaces(self, units=True):
        """get_distances_of_interfaces

        Calculates the distances of the interafaces of the :class:`Structure`.

        Args:
            units (boolean, optional): whether units should be returned or not.
                Defaults to True.

        Returns:
            res (ndarray[float, Quantity]): distances from the surface of each
            interface of the Structure.

        """

        d_start, d_end, _ = self.get_distances_of_layers(False)
        indices = np.r_[1, np.diff(self.get_layer_vectors()[0])]
        res = np.append(d_start[np.nonzero(indices)], d_end[-1])
        if units:
            return Q_(res, u.m)
        else:
            return res

    def interp_distance_at_interfaces(self, N, units=True):
        """ interp_distance_at_interfaces

        Interpolates the distances at the :class:`Layer` interfaces by an odd number
        :math:`N`.

        Args:
            N (int): number of point of interpolation at interface
            units (boolean, optional): whether units should be returned or not.
                Defaults to True.

        Returns:
            (tuple):
            - *dist_interp (ndarray[float, Quantity])* - distance array of the
              middle of each Layer interpolated by an odd number :math:`N` at
              the interfaces
            - *original_indicies (ndarray[int])* - indicies of the original
              distances in the interpolated array

        """

        [d_start, d_end, d_mid] = self.get_distances_of_layers(False)
        # these are the distances of the interfaces
        dist_intf = self.get_distances_of_interfaces(False)
        # start with the distances of the centers of the layers
        dist_interp = d_mid

        N = int(N)  # make N an integer
        if N % 2 == 0:
            # odd numbers are required
            N += 1

        # traverse all distances
        for z in dist_intf:
            inda = finderb(z, d_start)  # this is the index of a layer after the interface
            indb = inda-1  # this is the index of a layer before the interface

            # interpolate linearly N new distances at the interface
            if indb == -1:  # this is the surface interface
                dist_interp = np.append(dist_interp, np.linspace(0, d_mid[inda], int(2+(N-1)/2)))
            elif inda >= (len(d_mid)-1):  # this is the bottom interface
                dist_interp = np.append(dist_interp,
                                        np.linspace(d_mid[inda], d_end[-1], int(2+(N-1)/2)))
            else:  # this is a surface inside the structure
                dist_interp = np.append(dist_interp, np.linspace(d_mid[indb], d_mid[inda], 2+N))

        dist_interp = np.unique(np.sort(dist_interp))  # sort and unify the distances
        # these are the indicies of the original distances in the interpolated new array
        original_indicies = finderb(d_mid, dist_interp)
        if units:
            return Q_(dist_interp, u.m), original_indicies
        else:
            return dist_interp, original_indicies

    def get_layer_property_vector(self, property_name):
        """get_layer_property_vector

        Returns a vector for a property of all :class:`Layer` in the
        Structure. The property is determined by the `property_name` and
        returns a scalar value or a function handle.

        Args:
            property_name (str): name of property to return as array

        Returns:
            prop (ndarray[float, @lambda]): array of a property for all Layer
            in the Structure.

        """
        # get the Handle to all Layer in the Structure
        handles = self.get_layer_vectors()[2]

        if callable(getattr(handles[0], property_name)):
            # it's a function
            prop = np.zeros([self.get_number_of_layers()])
            for i in range(self.get_number_of_layers()):
                prop[i] = getattr(handles[i], property_name)
        elif ((type(getattr(handles[0], property_name)) is list) or
                (type(getattr(handles[0], property_name)) is str) or
                (type(getattr(handles[0], property_name)) is dict)):
            # it's a list of functions or str
            prop = []
            for i in range(self.get_number_of_layers()):
                # Prop = Prop + getattr(Handles[i],types)
                prop.append(getattr(handles[i], property_name))
        elif type(getattr(handles[0], property_name)) is Q_:
            # its a pint quantity
            unit = getattr(handles[0], property_name).units
            prop = np.empty([self.get_number_of_layers()])
            for i in range(self.get_number_of_layers()):
                prop[i] = getattr(handles[i], property_name).magnitude
            prop *= unit
        else:
            # its a number or array
            layers = self.get_unique_layers()
            temp = np.zeros([len(layers[0]), 1])
            set_dtype = float
            for i, layer in enumerate(layers[1]):
                value = getattr(layer, property_name)
                if np.iscomplexobj(value):
                    set_dtype = np.complex128
                try:
                    temp[i] = len(value)
                except TypeError:
                    temp[i] = 1
            max_dim = int(np.max(temp))
            if max_dim > 1:
                prop = np.empty([self.get_number_of_layers(), max_dim], dtype=set_dtype)
            else:
                prop = np.empty([self.get_number_of_layers()], dtype=set_dtype)
            del temp
            # traverse all layers
            for i in range(self.get_number_of_layers()):
                temp = getattr(handles[i], property_name)
                if np.iscomplexobj(temp):
                    prop.dtype = np.complex128
                if max_dim > 1:
                    prop[i, :] = temp
                else:
                    prop[i] = np.asarray(temp).item()

        return prop

    def get_numel_of_layer_property(self, property_name):
        """get_numel_of_layer_property

        Returns the number of elements (numel) of a property within
        the :class:`Structure`.
        For most properties this is 1, but for temperature-dependent
        properties it should return the number of sub-systems.

        The method raises an `IndexError` if the dimension of the property
        is not equal for all :class:`Layer`.

        Args:
            property_name (str): name of property to be checked

        Returns:
            numel (float): number of elements of property.

        """
        numel = np.array([len(prop) for prop in self.get_layer_property_vector(property_name)])
        if np.all(numel == numel[0]):
            numel = numel[0]
        else:
            raise IndexError('Property {:s} has not the same number of elements '
                             '(num_sub_systems) across the whole sample '
                             'Structure.'.format(property_name))
        return numel

    def get_layer_handle(self, i):
        """get_layer_handle

        Returns the handle to a :class:`Layer` at a given position index.

        Args:
            i (int): index of the Layer to return.

        Returns:
            handle (Layer): handle to the Layer at position `i` in the Structure.

        """
        handles = self.get_layer_vectors()[2]
        return handles[i]

    def reverse(self):
        """reverse

        Returns a reversed :class:`Structure` also reversing all nested `sub_structure`.

        Returns:
            reversed (Structure): reversed Structure.

        """
        from copy import deepcopy

        reversed = deepcopy(self)

        # handle superstrate and substrate
        reversed.superstrate, reversed.substrate = reversed.substrate, reversed.superstrate

        if isinstance(reversed.superstrate, Structure):
            reversed.superstrate = self.reverse_sub_structures(reversed.superstrate)
        if isinstance(reversed.substrate, Structure):
            reversed.substrate = self.reverse_sub_structures(reversed.substrate)

        return self.reverse_sub_structures(reversed)

    def reverse_sub_structures(self, structure):
        """reverse_sub_structures

        Reverse a :class:`Structure` and recursively call itself if a
        `sub_structure` is a :class:`Structure` itself.

        Args:
            structure (Structure): Structure to be reversed.

        Returns:
            structure (Structure): reversed Structure.

        """
        # reverse the list of sub_structures
        structure.sub_structures.reverse()
        for (sub_structure, N) in structure.sub_structures:
            if isinstance(sub_structure, Structure):
                # recursive call
                self.reverse_sub_structures(sub_structure)
            else:
                pass
        return structure

    @property
    def thickness(self):
        _, d_end, _ = self.get_distances_of_layers(True)
        return d_end[-1]
