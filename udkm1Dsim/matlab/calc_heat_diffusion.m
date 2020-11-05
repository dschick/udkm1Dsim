% The MIT License (MIT)
% Copyright (c) 2020 Daniel Schick
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
% OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
% OR OTHER DEALINGS IN THE SOFTWARE.

function temp_map = calc_heat_diffusion(K, init_temp, d_start, distances, ...
                    fluence, pulse_width, delay_pump, dalpha_dz, delays, ...
                    therm_cond, heat_capacity, density, sub_system_coupling, ...
                    left_type, left_value, right_type, right_value, ode_options) 

    % Convert Python lambda functions to MATLAB function handles
    therm_cond = lambda_to_handle(therm_cond, K);
    heat_capacity = lambda_to_handle(heat_capacity, K);
    sub_system_coupling = lambda_to_handle(sub_system_coupling, K);
               
    % This is the initial condition function that is necessary to
    % solve the heat equation with the pde solver, it returns the
    % initial temperature at each distance z.
    ic = @(z)(init_temp(finderb(z, distances), :)');
    % Create a  source if the fluence and pulse_width are non-zero
    if ~isempty(fluence) && sum(pulse_width) > 0
        source = generate_source(dalpha_dz, fluence, delay_pump, pulse_width, distances-distances(1));
    else
        source = [];
    end
    
    % Here we let the pde solver of matlab solve the differential
    % equation with the inital condition _ic_, the boundary
    % condition _obj.pde_boundary_conditions()_ and the pde function 
    % _obj.pde_heat_equation()_.
    temp_map =  pdepe(0,@(z,t,T,dTdz)(pde_heat_equation(z, t, T, dTdz, source, ...
                                                        K, d_start, therm_cond, ...
                                                        heat_capacity, density, ...
                                                        sub_system_coupling)),...
                   ic, @(zl, Tl, zr, Tr, t)(pde_boundary_conditions(zl, Tl, zr, Tr, ...
                                                                    t, K, left_type, left_value, ...
                                                                    right_type, right_value)), ...
                   distances, delays, odeset(ode_options));
end


%% generateSource
% Generate function handle of gaussians in time and absorption 
% profile $d\alpha/dz$ (Lambert-Beer' law) in space for use as a 
% source term in heat diffusion pdepe. The source term is engery 
% per second and volume [W/m^3]:
%
% $$ S(z,t) = \frac{\partial^2 E}{A\ \partial z \, \partial t} $$
% 
% with $A$ as unit cell area.
% For a Gaussian temporal profile we can substitute:
%
% $$ \frac{\partial^2 E}{\partial z \, \partial t} = \frac{d\alpha}{dz} \, E_0 \, \mbox{gauss}(t) $$
%
% where $\mbox{gauss}(t)$ is a normalized Gaussian function.
% Thus we get:
%
% $$ S(z,t) = \frac{d\alpha}{dz} \, F \, \mbox{gauss}(t) $$
%
% with $F$ as fluence [J/m^2].
function source = generate_source(dalpha_dz, fluence, delay_pump, pulse_width, distances)
    source   = @(t, z)(dalpha_dz(finderb(z, distances))*mgauss(t, pulse_width, delay_pump, fluence, 'widthType', 'FWHM', 'normalize', true));
end       

%% pde_heat_equation
% This is the definition of the differential heat equation that is
% solved by matlab's pde solver. In addition to $z,t,T,dT/dz$, we hand
% the vector of distances and handles to all unitCells to the
% function to save CPU time.
% matlab solves the following equation:
%
% $$ C\left(z,t,T,\frac{\partial T}{\partial z}\right) \frac{\partial T}{\partial t}  
%   = \frac{\partial}{\partial z} \left( f\left(z,t,T,\frac{\partial T}{\partial z}\right)\right) 
%   + s\left(x,t,T,\frac{\partial T}{\partial z}\right) $$
%
% which translates as following for the 1D heat equation:
%
% $$ C = c(T)\, \rho $$
%
% $$ f = k(T)\, \frac{\partial T}{\partial z} $$
%
% $$ s = G(T) + S(z,t) $$
%
% where $G$ is the subsystem-coupling-term and $S$ is the external 
% source term. The units used in the simulation are SI [W/m^3].
function [C, f, s] = pde_heat_equation(z, t, T, dTdz, source, K, d_start, therm_conds, heat_capacities, densities, sub_system_couplings, varargin)
    % find the first unitCell that is close to the current z
    % position that is given by the solver

    index = finderb(z, d_start);
    therm_cond = therm_conds(index, :)';
    heat_capacity = heat_capacities(index, :)';
    density = densities(index);
    sub_system_coupling = sub_system_couplings(index, :)';
    
    k = cellfun(@feval, therm_cond, num2cell(T));            
    % these are the parameters of the differential equation as they
    % are defined in matlab for the pdesolver

    vecSource = zeros(K,1);
    if isa(source,'function_handle')
        vecSource(1) = source(t, z);
    end            
    C = cellfun(@feval, heat_capacity, num2cell(T)).*density;
    f = k.*dTdz;
    s = cellfun(@feval, sub_system_coupling, repmat({T}, K, 1)) + vecSource;
end%function

%% pde_boundary_conditions
% This is the boundary condition function as it is required by the
% pde solver. For the left and right side the following equation
% has to be fulfilled:
%
% $$ p(z,t,T) + q(z,T) \, f\left(z,t,T, \frac{\partial T}{\partial z}\right) = 0 $$
%
function [pl,ql,pr,qr] = pde_boundary_conditions(zl, Tl, zr, Tr, t, K, left_type, left_value, right_type, right_value, varargin)
    persistent dim;
    if isempty(dim)        
        dim = [K, 1];
    end

    % check the type of the left boundary condition
    switch left_type
        case 2 % temperature
            pl = Tl - left_value';
            ql = zeros(dim);
        case 3 % flux
            pl = left_value';
            ql = ones(dim);                    
        otherwise % isolator
            pl = zeros(dim);
            if verLessThan('matlab', '8.3')
                ql = ones(size(dim));
            else                        
                ql = ones(dim);
            end%if
    end%switch

    % check the type of the right boundary condition
    switch right_type
        case 2 % temperature
            pr = Tr - right_value';
            qr = zeros(dim);
        case 3 % flux
            pr = -right_value';
            qr = ones(dim);                    
        otherwise % isolator
            pr = zeros(dim);
            if verLessThan('matlab', '8.3')
                qr = ones(size(dim));
            else                        
                qr = ones(dim);
            end%if
    end%switch
end%function

%% finderb
% Binary search algorithm for sorted lists.
% Searches for the first index _i_ of list where _key_ >= _list(i)_.
% _key_ can be a scalar or a vector of keys. 
% _list_ must be a sorted vector.
% author: André Bojahr
% licence: BSD
function i = finderb(key,list)
    n = length(key);
    i = zeros(1,n);

    if n > 500000 % if t is too long, we parallize it
        parfor (m = 1:n ,4)
            i(m) = finderb2(key(m),list);
        end%parfor
    else
        for m = 1:n
            i(m) = finderb2(key(m),list);
        end%for
    end%if
end%function

%% nested subfunction
function i = finderb2(key,list)
    a = 1;              % start of intervall
    b = length(list);   % end of intervall    
    
    % if the key is smaller than the first element of the
    % list we return 1
    if key < list(1)
        i = 1;
        return;
    end%if
    
    while (b-a) > 1 % loop until the intervall is larger than 1
        c = floor((a+b)/2); % center of intervall
        if key < list(c)
            % the key is in the left half-intervall
            b = c;
        else
            % the key is in the right half-intervall
            a = c;
        end%if        
    end%while
    
    i = a;
end%function

function y = mgauss(x,varargin)
    % initialize input parser and define defaults and validators
    p = inputParser;
    p.addRequired('x'                   , @isnumeric);
    p.addOptional('s'           , 1     , @isnumeric);
    p.addOptional('x0'          , 0     , @isnumeric);
    p.addOptional('A'           , 1     , @isnumeric);
    p.addOptional('widthType' , 'std' , @(x)(find(strcmp(x,{'std', 'FWHM', 'var', 'HWHM'}))));
    p.addOptional('normalize' , true  , @islogical);
    % parse the input
    p.parse(x,varargin{:});
    % assign parser results to object properties
    
    switch p.Results.widthType
        case 'FWHM' % Full Width at Half Maximum
            s = p.Results.s/(2*sqrt(2*log(2)));
        case 'var' % variance
            s = sqrt(p.Results.s);
        case 'HWHM' % Half Width at Half Maximum
            s = p.Results.s/(sqrt(2*log(2)));
        otherwise % standard derivation
            s = p.Results.s;
    end
        
    if p.Results.normalize == true
        a = p.Results.A./sqrt(2*pi*s.^2); % normalize area to 1
    else
        a = p.Results.A.*ones(size(s)); % normalize amplitude to 1
    end
    
    x0 = p.Results.x0.*ones(size(s));
        
    y = zeros(1,length(x));
    for i = 1:length(s)
        y = y + a(i) * exp(-((x-x0(i)).^2)./(2*s(i)^2));    
    end
end

function handle = lambda_to_handle(lambda, K)
    handle = cell(length(lambda), K);
    for i = 1:length(lambda)
        temp = lambda{i}';
        for j = 1:K
            func_str = strcat(strrep(strrep(strrep(temp{j}, 'lambda T: ', '@(T)('), '[' , '('), ']', '+1)'), ')');
            func_str = strrep(func_str, '**', '.^');
            func_str = strrep(func_str, '*', '.*');
            func_str = strrep(func_str, '/', './');
            handle{i, j} = str2func(func_str);
        end
    end
end