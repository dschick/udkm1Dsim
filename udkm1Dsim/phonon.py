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

"""A :mod:`Phonon` module """

__all__ = ["Phonon"]

__docformat__ = "restructuredtext"

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp2d
from time import time
from os import path
from .simulation import Simulation
from . import u, Q_
from .helpers import make_hash_md5, finderb
import warnings


class Phonon(Simulation):
    """Phonon

    Base class for phonon simulatuons.

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Keyword Args:
        only_heat (boolean): true when including only thermal expanison without
            coherent phonon dynamics

    Attributes:
        S (object): sample to do simulations with
        only_heat (boolean): force recalculation of results
        heat_diffusion (boolean): true when including only thermal expanison without
            coherent phonon dynamics
        matlab_engine (module): MATLAB to Python API engine required for
            calculating heat diffusion

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.only_heat = kwargs.get('only_heat', False)
        self.matlab_engine = []

    def __str__(self, output=[]):
        """String representation of this class"""

        output = [['only heat', self.heat_diffusion],
                  ] + output

        class_str = 'Phonon simulation properties:\n\n'
        class_str += super().__str__(output)

        return class_str

    def get_hash(self, delays, temp_map, delta_temp_map, **kwargs):
        """get_hash

        Returns an unique hash given by the delays, and temp- and delta_temp_map
        as well as the sample structure hash for relevant thermal parameters.

        """
        param = [delays, self.only_heat]
        
        if np.size(temp_map) > 1e6:
            temp_map = temp_map.flatten()[0:1000000]
            delta_temp_map = delta_temp_map.flatten()[0:1000000]
        param.append(temp_map)            
        param.append(delta_temp_map)

        for key, value in kwargs.items():
            param.append(value)

        return self.S.get_hash(types='phonon') + '_' + make_hash_md5(param)

    def get_all_strains_per_unique_layer(self, strain_map):
        """get_all_strains_per_unique_layer
        
        Returns a dict with all strains per unique layer that
        are given by the input _strain_map_.
        
        """
        # get the position indices of all unique layers in the sample structure
        positions = self.S.get_all_positions_per_unique_layer()
        strains = {}
        
        for key, value in positions.items():
            strains[key] = np.sort(np.unique(strain_map[:, value].flatten()))

        return strains

    def get_reduced_strains_per_unique_layer(self, strain_map, N=100):
        """ get_reduced_strains_per_unique_layer

        Returns a cell array with all strains per unique unit cell that
        are given by the input _strainMap_, BUT with a reduced number. The
        reduction is done by equally spacing the strains between the min
        and max strain with a given number $N$.
        $N$ can be also a vector of the length($N$) = $M$, where $M$ is 
        the number of unique unit cells.

        """

        # initialize
        all_strains = self.get_all_strains_per_unique_layer(strain_map)
        M = len(all_strains) # Nb. of unique layers
        strains = {}        
        
        if np.size(N) == 1:
            N = N*np.ones([M, 1])                
        elif np.size(N) != M:
            raise ValueError('The dimension of N must be either 1 or the number '
                             'of unique layers the structure!')
        
        
        for i, (key, value) in enumerate(all_strains.items()):
            min_strain = np.min(value)
            max_strain = np.max(value)
            strains[key] = np.sort(np.unique(np.r_[0, np.linspace(min_strain, max_strain, int(N[i]))]))

        return strains

"""        
        %% checkTempMaps
        % Returns the corrected _deltaTempMap_ for the _strainMap_ 
        % calculation and checks _tempMap_ and _deltaTempMap_ for the 
        % correct dimensions.
        function [tempMap, deltaTempMap] = checkTempMaps(obj,tempMap,deltaTempMap,time)
            N = obj.S.getNumberOfUnitCells; % nb of unit cells
            M = length(time);               % nb of time steps
            K = obj.S.numSubSystems;        % nb of subsystems
            
            % check size of deltaTempMap
            if K == 1
                if isequal(size(deltaTempMap),[1 N])
                    temp                = deltaTempMap;
                    deltaTempMap        = zeros(M,N);
                    deltaTempMap(1,:)   = temp;
                elseif size(deltaTempMap,1) ~= M || size(deltaTempMap,2) ~= N
                    error('The given temperature difference map does not have the dimension M x N, where M is the number of time steps and N the number of unitCells!');
                end%if
            else
                if isequal(size(deltaTempMap),[1 N K])
                    temp                = deltaTempMap;
                    deltaTempMap        = zeros(M,N,K);
                    deltaTempMap(1,:,:) = temp;
                elseif size(deltaTempMap,1) ~= M || size(deltaTempMap,2) ~= N || size(deltaTempMap,3) ~= K
                    error('The given temperature difference map does not have the dimension M x N, where M is the number of time steps and N the number of unitCells and K is the number of subsystems!');
                end%if
            end%if
            
            if size(tempMap) ~= size(deltaTempMap)
                error('The temperature map does not have the same size as the temperature difference map!');
            end%if
        end%function
        
        %% calcSticksFromTempMap
        % Calculates the sticks to insert into the unit cell springs which
        % model the external force (thermal stress). The length of $l_i$ 
        % of the i-th spacer stick is calculated from the
        % temperature-dependent linear thermal expansion $\alpha(T)$ of 
        % the unit cell:
        %
        % $$ \alpha(T) = \frac{1}{L} \frac{d L}{d T} $$
        % 
        % which results after integration in
        %
        % $$ l = \Delta L = L_1 \exp(A(T_2) - A(T_1)) - L_1 $$
        %
        % where $A(T)$ is the integrated lin. therm. expansion
        % coefficient in respect to the temperature $T$. The indices 1 and
        % 2 indicate the initial and final state.
        function [sticks, sticksSubSystems] = calcSticksFromTempMap(obj,tempMap,deltaTempMap)
            N = obj.S.getNumberOfUnitCells; % nb of unit cells
            K = obj.S.numSubSystems;        % nb of subsystems
            M = size(tempMap,1);            % nb of time steps
            
            cAxises         = obj.S.getUnitCellPropertyVector('cAxis');            
            intLinThermExps = obj.S.getUnitCellPropertyVector('intLinThermExp'); % integrated linear thermal expansion function
            intAlphaT0      = zeros(N,K); % evaluated initial integrated linear thermal expansion from T1 to T2
            intAlphaT       = zeros(N,K); % evaluated integrated linear thermal expansion from T1 to T2
            sticks          = zeros(M,N); % the sticks inserted in the unit cells
            sticksSubSystems= zeros(M,N,K); % the sticks for each thermodynamic subsystem
            
            % calculate initial integrated linear thermal expansion from T1 to T2
            % traverse subsystems
            for j=1:K
                intAlphaT0(:,j) = cellfun(@feval,intLinThermExps(:,j),num2cell(squeeze(tempMap(1,:,j))'-squeeze(deltaTempMap(1,:,j))'));
            end%for
            
            % calculate sticks for all subsytsems for all time steps
            % traverse time
            for i=1:M
                if find(deltaTempMap(i,:)) % there is a temperature change
                    % Calculate new sticks from the integrated linear 
                    % thermal expansion from initial temperature to 
                    % current temperature for each subsystem                        
                    % traverse subsystems
                    for j=1:K
                        intAlphaT(:,j) = cellfun(@feval,intLinThermExps(:,j),num2cell(squeeze(tempMap(i,:,j))'));
                    end%for

                    % calculate the length of the sticks of each subsystem and sum
                    % them up 
                    sticksSubSystems(i,:,:) = repmat(cAxises,1,K) .*exp(intAlphaT-intAlphaT0)-repmat(cAxises,1,K);
                    sticks(i,:)             = sum(sticksSubSystems(i,:,:),3)';
                else % no temperature change, so keep the current sticks
                    if i > 1
                        sticksSubSystems(i,:,:) = sticksSubSystems(i-1,:,:);
                        sticks(i,:)             = sticks(i-1,:);
                    end%if
                end%if
            end%for            
        end%function
"""