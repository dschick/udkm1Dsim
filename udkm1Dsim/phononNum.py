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

"""A :mod:`PhononNum` module """

__all__ = ["PhononNum"]

__docformat__ = "restructuredtext"

import numpy as np
from .phonon import Phonon
from os import path


class PhononNum(Phonon):
    """PhononNum

    Base class for numerical phonon simulatuons.

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Keyword Args:
        only_heat (boolean): true when including only thermal expanison without
            coherent phonon dynamics

    Attributes:
        S (object): sample to do simulations with
        only_heat (boolean): force recalculation of results
        matlab_engine (module): MATLAB to Python API engine required for
            calculating heat diffusion

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)
        self.only_heat = kwargs.get('only_he"at', False)
        self.matlab_engine = []

    def __str__(self, output=[]):
        """String representation of this class"""

        class_str = 'Numerical Phonon simulation properties:\n\n'
        class_str += super().__str__()

        return class_str


    def get_strain_map(self, delays, temp_map, delta_temp_map):
        """get_strain_map

        Returns a strain profile for the sample structure for given temperature
        profile. The result can be saved using an unique hash of the sample
        and the simulation parameters in order to reuse it.

        """
        filename = 'strain_map_num_' \
                   + self.get_hash(delays, temp_map, delta_temp_map) \
                   + '.npy'
        full_filename = path.abspath(path.join(self.cache_dir, filename))
        if path.exists(full_filename) and not self.force_recalc:
            # found something so load it
            strainMap, sticksSubSystems, velocities = np.load(full_filename)
            self.disp_message('_strain_map_ loaded from file:\n\t' + filename)
        else:
            # file does not exist so calculate and save
            strainMap, sticksSubSystems, velocities = \
                self.calc_strain_map(delays, temp_map, delta_temp_map)
            self.save(full_filename, [strainMap, sticksSubSystems, velocities], '_strain_map_num_')
        return strainMap, sticksSubSystems, velocities

"""       
        %% calcStrainMap
        % Calculates the _strainMap_ of the sample structure for a given 
        % _tempMap_ and _deltaTempMap_ and _time_ vector. Further details 
        % are given in Ref. [1]. We solve the coupled differential 
        % equations for each oscillator in a linear chain of masses and 
        % spring:
        %
        % $$ m_i\ddot{x}_i = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1})
        %  + m_i\gamma_i(\dot{x}_i-\dot{x}_{i-1}) 
        %  + F_i^{heat}(t) $$
        %
        % where $x_i(t) = z_{i}(t) - z_i^0$ is the shift of each unit cell. 
        % $m_i$ is the mass and $k_i = m_i\, v_i^2/c_i^2$ is the spring 
        % constant of each unit cell. Furthermore we introduce an empirical 
        % damping term $F_i^{damp} = \gamma_i(\dot{x}_i-\dot{x}_{i-1})$ and 
        % the external force (thermal stress) $F_i^{heat}(t)$. 
        % The thermal stresses are modelled as spacer sticks which are 
        % calculated from the linear thermal expansion coefficients. The 
        % equation of motion can be reformulated as:
        %
        % $$ m_i\ddot{x}_i = F_i^{spring} + F_i^{damp} + F_i^{heat}(t) $$ 
        %
        % The numerical solution also allows for non-harmonic
        % inter-atomic potentials of up to the order $M$. Accordingly 
        % $k_i = (k_i^1 \ldots k_i^{M-1})$ can be a vector
        % accounting for higher orders of the potential which is in the
        % harmonic case purely quadratic ($k_i = k_i^1$). The resulting 
        % force from the displacement of the springs 
        % 
        % $$F_i^{spring} = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1})$$
        %
        % includes:
        %
        % $$ k_i(x_i-x_{i-1}) = \sum_{j=1}^{M-1} k_i^j (x_i-x_{i-1})^j $$
        %
        function [strainMap, sticksSubSystems, velocities] = calcStrainMap(obj,time,tempMap,deltaTempMap)
            tic
            % initialize
            N       = obj.S.getNumberOfUnitCells; % nb of unit cells
            cAxises = obj.S.getUnitCellPropertyVector('cAxis');
            x0      = zeros(2*N,1); % initial condition for the shift of the unit cells
            
            % check tempMaps
            [tempMap, deltaTempMap] = obj.checkTempMaps(tempMap,deltaTempMap,time);
            
            % calculate the sticks due to heat expansion first for all time
            % steps   
            obj.dispMessage('Calculating linear thermal expansion ...');
            [sticks, sticksSubSystems] = obj.calcSticksFromTempMap(tempMap,deltaTempMap);
            
            if obj.onlyheat
                % no coherent dynamics so calculate the strain directly
                strainMap  = sticks./repmat(cAxises',size(sticks,1),1);
                velocities = zeros(size(strainMap)); % this is quasi-static
            else
                % include coherent dynamics
                obj.dispMessage('Calculating coherent dynamics with ODE solver ...')
                % define the ode which has to be solved (t- and x-grid)
                odefunc = @(t,X) obj.odeFunc(t,X,time,sticks);
                % apply MATLAB's ode-solver and input also temporal grid
                % (time) on which the result is extrapolated to and the
                % initial conditions x0 and the odeOptions
                [~,X] = ode23(odefunc,time,x0,odeset(obj.odeOptions));
                
                % calculate the strainMap as the second spacial derivative
                % of the unit cell shift x(t). The result of the ode solver
                % contains x(t) = X(:,1:N) and v(t) = X(:,N+1:end) the
                % positions and velocities of the unit cells, respectively.
                temp            = diff(X(:,1:N),1,2);
                temp(:,end+1)   = 0;
                strainMap       = temp./repmat(cAxises',size(temp,1),1);
                velocities      = X(:,N+1:end);
            end%if
            obj.dispMessage('Elapsed time for _strainMap_:',toc);
        end%function
        
        %% odeFunc
        % Provides the according ode function for the ode solver which
        % has to be solved. The ode function has the input $t$ and $X(t)$
        % and calculates the temporal derivative $\dot{X}(t)$ where the
        % vector 
        %
        % $$X(t) = [x(t) \; \dot{x}(t)] \quad \mbox{and } \quad
        % \dot{X}(t) = [\dot{x}(t) \; \ddot{x}(t)]$$
        %
        % $x(t)$ is the actual shift of each unit cell. Further inputs are
        % the actual time gride _time_ and the _sticks_ from the linear
        % thermal expansion.
        function Xprime = odeFunc(obj,t,X,time,sticks)
            persistent forceFromHeat masses N;
            if isempty(forceFromHeat)
                % calculate the external force according to the spacer 
                % sticks in advance
                forceFromHeat   = obj.calcFromForceHeat(sticks);
                masses          = obj.S.getUnitCellPropertyVector('mass');
                N               = obj.S.getNumberOfUnitCells; % nb of unit cells
            end%if
            
            x = X(1:N);
            v = X(N+1:end);
            
            % the output must be a column vector
            Xprime = zeros(2*N,1);
                        
            % accelerations = derivative of velocities
            Xprime(N+1:end) = ...
                (obj.calcForceFromDamping(v) ...
               + obj.calcForceFromSpring([diff(x);0],[0;diff(x)])...
               + forceFromHeat(:,finderb(t,time)))./masses;
            
           % velocities = derivative of positions
            Xprime(1:N) = v;
        end%function
        
        %% calcFromForceHeat
        % Calculates the force acting on each mass due to the heat 
        % expansion, which is modelled by spacer sticks.
        function F = calcFromForceHeat(obj,sticks)
            [M N] = size(sticks);
            F = zeros(N,M);
            % traverse time
            for i = 1:M;
                F(:,i) = -obj.calcForceFromSpring([sticks(i,1:N-1)';0],[0;sticks(i,1:N-1)']);
            end%for
        end%function
        
        %% calcForceFromSpring
        % Calculates the force $F_i^{spring}$ acting on each mass due to 
        % the displacement between the left and right site of that mass.
        %
        % $$F_i^{spring} = -k_i(x_i-x_{i-1})-k_{i+1}(x_i-x_{i+1})$$
        %
        % We introduce-higher order inter-atomic potentials by
        %
        % $$ k_i(x_i-x_{i-1}) = \sum_{j=1}^{M-1} k_i^j (x_i-x_{i-1})^j $$
        %
        % where $M-1$ is the order of the spring constants
        function F = calcForceFromSpring(obj,delX1,delX2)
            persistent springOrder coeff1 coeff2;
            if isempty(springOrder)
                springConsts = obj.S.getUnitCellPropertyVector('springConst');
                springOrder  = size(springConsts,2);
                coeff1 = [-springConsts(1:end-1,:); zeros(1,springOrder)];
                coeff2 = [zeros(1,springOrder); -springConsts(1:end-1,:)];
            end%if
            
            temp1 = zeros(length(delX1),springOrder);
            temp2 = zeros(length(delX1),springOrder);
            for i = 1:springOrder
                temp1(:,i) = delX1.^i;
                temp2(:,i) = delX2.^i;
            end%for
            
            F = sum(coeff2.*temp2,2)-sum(coeff1.*temp1,2);
        end%function
        
        %% calcForceFromDamping
        % Calculates the force acting on each mass in a linear spring due
        % to damping ($\gamma_i$) according to the shift velocity
        % difference $v_{i}-v_{i-1}$ with $v_i(t) = \dot{x}_i(t)$:
        %
        % $$F_i^{damp} = \gamma_i(\dot{x}_i-\dot{x}_{i-1})$$
        %
        function F = calcForceFromDamping(obj,v)
            persistent damping masses;
            if isempty(damping)
                damping = obj.S.getUnitCellPropertyVector('phononDamping');
                masses  = obj.S.getUnitCellPropertyVector('mass');
            end%if
            
            F = masses.*damping.*[diff(v,1); 0];            
        end%function        
    end%methods
end%classdef

%% References
%
% # A. Bojahr, M. Herzog, D. Schick, I. Vrejoiu, & M. Bargheer (2012). 
% _Calibrated real-time detection of nonlinearly propagating strain waves_. 
% Physical Review B, 86(14), 144306. doi:10.1103/PhysRevB.86.144306
"""
