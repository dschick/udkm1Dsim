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
from inspect import isfunction
from sympy import integrate, Symbol
from sympy.utilities.lambdify import lambdify

class unitCell(object):
    """unitCell

    The unitCell class hold different structural properties of real physical
    unit cells and also an array of atoms at different postions in the unit
    cell.

    ID (str)                        : ID of the unit cell
    name (str)                      : name of the unit cell
    atoms (list[atom, @lambda])     : list of atoms and funtion handle for
                                    strain dependent displacement
    numAtoms (int)                  : number of atoms in unit cell
    aAxis (float)                   : in-plane a-axis [m]
    bAxis (float)                   : in-plane b-axis [m]
    cAxis (float)                   : out-of-plane c-axis [m]
    area  (float)                   : area of epitaxial unit cells
                                      need for normation for correct intensities) [m^2]
    volume (float)                  : volume of unit cell [m^3]
    mass (float)                    : mass of unit cell normalized to an area of 1 Ang^2 [kg]
    density (float)                 : density of the unitCell [kg/m^3]
    debWalFac (float)               : Debye Waller factor <u>^2 [m^2]
    soundVel (float)                : sound velocity in the unit cell [m/s]
    springConst (ndarray[float])    : spring constant of the unit cell [kg/s^2] and higher orders
    phononDamping (float)           : damping constant of phonon propagation [kg/s]
    optPenDepth (float)             : penetration depth for pump always for 1st subsystem
                                    light in the unit cell [m]
    optRefIndex (ndarray[float])    : optical refractive index - real and imagenary part $n + i\kappa$
    optRefIndexPerStrain (ndarray[float])   :
            optical refractive index change per strain -
            real and imagenary part %\frac{d n}{d \eta} + i\frac{d \kappa}{d \eta}$
    thermCond (list[@lambda])               :
            list of HANDLES T-dependent thermal conductivity [W/(m K)]
    linThermExp (list[@lambda])             :
            list of HANDLES T-dependent linear thermal expansion coefficient (relative)
    intLinThermExp (list[@lambda])          :
            list of HANDLES T-dependent integrated linear thermal expansion coefficient
    heatCapacity (list[@lambda])            :
            list of HANDLES T-dependent heat capacity function [J/(kg K)]
    intHeatCapacity (list[@lambda])         :
            list of HANDLES T-dependent integrated heat capacity function
    subSystemCoupling (list[@lambda])       :
            list of HANDLES of coupling functions of different subsystems [W/m^3]
    numSubSystems (int)                     :
            number of subsystems for heat and phonons (electrons, lattice, spins, ...)
    """

    def __init__(self, ID, name, cAxis, **kwargs):
        # % initialize input parser and define defaults and validators
        # p = inputParser;
        # p.addRequired('ID'                      , @ischar);
        # p.addRequired('name'                    , @ischar);
        # p.addRequired('cAxis'                   , @isnumeric);
        # p.addParamValue('aAxis'                 , cAxis , @isnumeric);
        # p.addParamValue('bAxis'                 , cAxis , @isnumeric);
        # p.addParamValue('debWalFac'             , 0     , @isnumeric);
        # p.addParamValue('soundVel'              , 0     , @isnumeric);
        # p.addParamValue('phononDamping'         , 0     , @isnumeric);
        # p.addParamValue('optPenDepth'           , 0     , @isnumeric);
        # p.addParamValue('optRefIndex'           , [0,0] , @(x) (isnumeric(x) && numel(x) == 2));
        # p.addParamValue('optRefIndexPerStrain'  , [0,0] , @(x) (isnumeric(x) && numel(x) == 2));
        # p.addParamValue('thermCond'             , 0     , @(x)(isnumeric(x) || isa(x,'function_handle') || ischar(x) || iscell(x)));
        # p.addParamValue('linThermExp'           , 0     , @(x)(isnumeric(x) || isa(x,'function_handle') || ischar(x) || iscell(x)));
        # p.addParamValue('heatCapacity'          , 0     , @(x)(isnumeric(x) || isa(x,'function_handle') || ischar(x) || iscell(x)));
        # p.addParamValue('subSystemCoupling'     , 0     , @(x)(isnumeric(x) || isa(x,'function_handle') || ischar(x) || iscell(x)));
        # % parse the input
        # p.parse(ID,name,cAxis,varargin{:});
        # % assign parser results to object properties
        self.ID = ID
        self.name = name
        self.cAxis = cAxis
        self.aAxis = kwargs.get('aAxis', self.cAxis)
        self.bAxis = kwargs.get('bAxis', self.aAxis)
        self.atoms          = []
        self.numAtoms       = 0
        self.mass           = 0
        self.density        = 0
        self.springConst    = np.array([0])
        self.debWalFac               = kwargs.get('debWalFac', 0)
        self.soundVel                = kwargs.get('soundVel', 0)
        self.phononDamping           = kwargs.get('phononDamping', 0)
        self.optPenDepth             = kwargs.get('optPenDepth', 0)
        self.optRefIndex             = kwargs.get('optRefIndex', 0)
        self.optRefIndexPerStrain    = kwargs.get('optRefIndexPerStrain', 0)
        self.heatCapacity, self.heatCapacityStr \
                                     = self.checkCellArrayInput(kwargs.get('heatCapacity', 0))
        self.thermCond, self.thermCondStr \
                                     = self.checkCellArrayInput(kwargs.get('thermCond', 0))
        self.linThermExp, self.linThermExpStr \
                                     = self.checkCellArrayInput(kwargs.get('linThermExp', 0))
        self.subSystemCoupling, self.subSystemCouplingStr \
                                     = self.checkCellArrayInput(kwargs.get('subSystemCoupling', 0))

        if len(self.heatCapacity) == len(self.thermCond) \
            and len(self.heatCapacity) == len(self.linThermExp) \
            and len(self.heatCapacity) == len(self.subSystemCoupling):
            self.numSubSystems = len(self.heatCapacity)
        else:
            raise ValueError('Heat capacity, thermal conductivity, linear \
                thermal expansion and subsystem coupling have not the same number of elements!')

        # calculate the area of the unit cell
        self.area           = self.aAxis * self.bAxis
        self.volume         = self.area * self.cAxis

    def __str__(self):
        """String representation of this class

        """
        classStr  = 'Unit Cell with the following properties\n'
        classStr += 'ID                     : {:s}\n'.format(self.ID)
        classStr += 'name                   : {:s}\n'.format(self.name)
        classStr += 'a-axis                 : {:3.2f} Å\n'.format(self.aAxis/1e-10)
        classStr += 'b-axis                 : {:3.2f} Å\n'.format(self.bAxis/1e-10)
        classStr += 'c-axis                 : {:3.2f} Å\n'.format(self.cAxis/1e-10)
        classStr += 'area                   : {:3.2f} Å²\n'.format(self.area/1e-20)
        classStr += 'volume                 : {:3.2f} Å³\n'.format(self.volume/1e-30)
        classStr += 'mass                   : {:3.2e} kg\n'.format(self.mass)
        classStr += 'density                : {:3.2e} kg/m³\n'.format(self.density)
        classStr += 'Debye Waller Factor    : {:3.2f} m²\n'.format(self.debWalFac)
        classStr += 'sound velocity         : {:3.2f} nm/ps\n'.format(self.soundVel/1e-9*1e-12)
        classStr += 'spring constant        : {:s} kg/s²\n'.format(np.array_str(self.springConst))
        classStr += 'phonon damping         : {:3.2f} kg/s\n'.format(self.phononDamping)
        classStr += 'opt. pen. depth        : {:3.2f} nm\n'.format(self.optPenDepth/1e-9)
        classStr += 'opt. refractive index  : {:3.2f}\n'.format(self.optRefIndex)
        classStr += 'opt. ref. index/strain : {:3.2f}\n'.format(self.optRefIndexPerStrain)
        classStr += 'thermal conduct. [W/m K]       :\n'
        for func in self.thermCondStr:
            classStr += '\t\t\t {:s}\n'.format(func)
        classStr += 'linear thermal expansion [1/K] :\n'
        for func in self.linThermExpStr:
            classStr += '\t\t\t {:s}\n'.format(func)
        classStr += 'heat capacity [J/kg K]         :\n'
        for func in self.heatCapacityStr:
            classStr += '\t\t\t {:s}\n'.format(func)
        classStr += 'subsystem coupling [W/m^3]     :\n'
        for func in self.subSystemCouplingStr:
            classStr += '\t\t\t {:s}\n'.format(func)
        # display the constituents
        classStr += str(self.numAtoms) + ' Constituents:\n'
        for i in range(self.numAtoms):
            classStr += '{:s} \t {:0.2f} \t {:s}\n'.format(self.atoms[i][0].name, self.atoms[i][1](0), self.atoms[i][2])

        return(classStr)

#
#     %% visualize
#     % plots the atoms in the unitCell for a given strain. You can input
#     % a figure handle.
#     function visualize(obj,varargin)
#         % initialize input parser and define defaults and validators
#         p = inputParser;
#         p.addRequired('obj'     , @(x)isa(x,'unitCell'));
#         p.addParamValue('strain', 0     , @isnumeric);
#         p.addParamValue('pause' , 0.05  , @isnumeric);
#         p.addParamValue('handle', ''    , @ishandle);
#         % parse the input
#         p.parse(obj,varargin{:});
#         % assign parser results to object properties
#         if isempty(p.Results.handle)
#             h = figure;
#         else
#             h = p.Results.handle;
#         end
#         strain = p.Results.strain;
#         figure(h);
#         colors = colormap(lines(obj.numAtoms));
#         atomIDs = obj.getAtomIDs();
#         atomsPlotted = zeros(size(atomIDs));
#         for i = 1:length(strain)
#             for j = 1:obj.numAtoms
#                 l = plot(1+0*j,obj.atoms{j,2}(strain(i)),'Marker', 'o', 'MarkerSize', 5, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', colors(strcmp(obj.atoms{j,1}.ID, atomIDs),:), 'LineStyle', 'none');
#                 % check if atom has already been plotted
#                 if atomsPlotted(strcmp(obj.atoms{j,1}.ID, atomIDs))
#                     % do not show the atom in the legend
#                     hasbehavior(l,'legend',false);
#                 else
#                     % set that the atom had been plotted
#                     atomsPlotted(strcmp(obj.atoms{j,1}.ID, atomIDs)) = true;
#                 end%if
#                 hold on;
#             end%for
# %                 axis([0.1 obj.numAtoms+0.9 -0.1 (1.1+max(strain))]); grid on; box on;
#             title(sprintf('Strain: %.2f%%',strain(i)), 'FontSize', 18);
#             ylabel('relative Position');
#             xlabel('# Atoms');
#             hold off
#             pause(p.Results.pause)
#         end%for
#         legend(atomIDs,'Location','NorthWest');
#     end%function

    def getPropertyStruct(self, **kwargs):
        """getParameterStruct

        Returns a struct with all parameters. objects or cell arrays and
        objects are converted to strings. if a type is given, only these
        properties are returned.
        """
        # initialize input parser and define defaults and validators
        types = ['all', 'heat', 'phonon', 'XRD', 'optical']
        propertiesByTypes = {
                'heat'     : ['cAxis', 'area', 'volume', 'optPenDepth', 'thermCondStr', 'heatCapacityStr', 'intHeatCapacityStr', 'subSystemCouplingStr', 'numSubSystems'],
                'phonon'   : ['numSubSystems', 'intLinThermExpStr', 'cAxis', 'mass', 'springConst', 'phononDamping'],
                'XRD'      : ['numAtoms', 'atoms', 'area', 'debWalFac', 'cAxis'],
                'optical'  : ['cAxis', 'optPenDepth', 'optRefIndex', 'optRefIndexPerStrain'],
                }

        types = kwargs.get('types', 'all')
        attrs = vars(self)
        # define the property names by the given type
        if types == 'all':
            S = attrs
        else:
            S = dict((key, value) for key, value in attrs.items() if key in propertiesByTypes[types])

        return S

    def checkCellArrayInput(self, inputs):
        """ checkCellArrayInput

        Checks the input for inputs which are cell arrays of function
        handles, such as the heat capacity which is a cell array of N
        function handles.
        """
        output      = []
        outputStrs  = []
        # if the input is not a list, we convert it to one
        if not isinstance(inputs, list):
            inputs = [inputs]
        # traverse each list element and convert it to a function handle
        for input in inputs:
            if isfunction(input):
                raise ValueError('Please use string representation of function!')
                output.append(input)
                outputStrs.append('no str representation available')
            elif isinstance(input, str):
                try:
                    output.append(eval(input))
                    outputStrs.append(input)
                except Exception as e:
                    print('String input for unit cell property ' + input + ' \
                        cannot be converted to function handle!')
                    print(e)
            elif isinstance(input, (int, float)):
                output.append(eval('lambda T: {:f}'.format(input)))
                outputStrs.append('lambda T: {:f}'.format(input))
            else:
                raise ValueError('Unit cell property input has to be a single or \
                cell array of numerics, function handles or strings which can be \
                converted into a function handle!')

        return(output, outputStrs)

    @property
    def intHeatCapacity(self):
        """get intHeatCapacity

        Returns the anti-derrivative of the temperature-dependent heat
        $c(T)$ capacity function. If the _intHeatCapacity_ property is
        not set, the symbolic integration is performed.
        """

        if hasattr(self, '_intHeatCapacity') and isinstance(self._intHeatCapacity, list):
            h = self._intHeatCapacity
        else:
            self._intHeatCapacity = []
            self.intHeatCapacityStr = []
            try:
                T = Symbol('T')
                for i, hcs in enumerate(self.heatCapacityStr):
                    integral = integrate(hcs.split(':')[1], T)
                    self._intHeatCapacity.append(lambdify(T, integral))
                    self.intHeatCapacityStr.append('lambda T : ' + str(integral))

            except Exception as e:
                print('The MATLAB Symbolic Math Toolbox is not installed. \
                Please set the analytical anti-derivative of the heat capacity \
                of your unit cells as anonymous function of the temperature \
                T by typing UC.intHeatCapacity = @(T)(c(T)); \
                where UC is the name of the unit cell object.')
                print(e)

        return(self._intHeatCapacity)

    @intHeatCapacity.setter
    def intHeatCapacity(self, intHeatCapacity):
        """set intHeatCapacity

        Set the integrated heat capacity manually when no Smybolic Math
        Toolbox is installed.
        """
        self._intHeatCapacity, self.intHeatCapacityStr = self.checkCellArrayInput(intHeatCapacity)

    @property
    def intLinThermExp(self):
        """get intLinThermExp

        Returns the anti-derrivative of theintegrated temperature-dependent
        linear thermal expansion function. If the __intLinThermExp__
        property is not set, the symbolic integration is performed.
        """

        if hasattr(self, '_intLinThermExp') and isinstance(self._intLinThermExp, list):
            h = self._intLinThermExp
        else:
            self._intLinThermExp = []
            self.intLinThermExpStr = []
            try:
                T = Symbol('T')
                for i, ltes in enumerate(self.linThermExpStr):
                    integral = integrate(ltes.split(':')[1], T)
                    self._intLinThermExp.append(lambdify(T, integral))
                    self.intLinThermExpStr.append('lambda T : ' + str(integral))

            except Exception as e:
                print('The MATLAB Symbolic Math Toolbox is not installed. \
                Please set the analytical anti-derivative of the heat capacity \
                of your unit cells as anonymous function of the temperature \
                T by typing UC.intHeatCapacity = @(T)(c(T)); \
                where UC is the name of the unit cell object.')
                print(e)

        return(self._intLinThermExp)

    @intLinThermExp.setter
    def intLinThermExp(self, intLinThermExp):
        """set intLinThermExp

        Set the integrated linear thermal expansion coefficient manually
        when no Smybolic Math Toolbox is installed.
        """
        self._intLinThermExp, self.intLinThermExpStr = self.checkCellArrayInput(intLinThermExp)


    #
    # %% getIntLinThermExp
    # % Returns the anti-derrivative of theintegrated temperature-dependent
    # % linear thermal expansion function. If the _intHeatCapacity_
    # % property is not set, the symbolic integration is performed.
    # function h = get.intLinThermExp(obj)
    #     if iscell(obj.intLinThermExp)
    #         h = obj.intLinThermExp;
    #     elseif exist('syms')
    #         % symbolic math toolbox is installed
    #         syms T;
    #         h = cell(length(obj.linThermExp),1);
    #         for i=1:length(obj.linThermExp)
    #             fstr = strrep(func2str(obj.linThermExp{i}),'@(T)','');
    #             fstr = strrep(fstr,'.*','*');
    #             fstr = strrep(fstr,'./','/');
    #             fstr = strrep(fstr,'.^','^');
    #             h{i} = str2func(['@(T)(' vectorize(int(sym(fstr),'T')) ')']);
    #         end%for
    #         obj.intLinThermExp = h;
    #         clear T;
    #     else
    #         error('The MATLAB Symbolic Math Toolbox is not installed. Please set the analytical anti-derivative of the linear thermal expansion coefficient of your unit cells as anonymous function of the temperature T by typing UC.intLinThermExp = @(T)(a(T)); where UC is the name of the unit cell object.');
    #     end%if
    # end%function
    #
    # %% setIntLinThermExp
    # % Set the integrated linear thermal expansion coefficient manually
    # % when no Smybolic Math Toolbox is installed.
    # function set.intLinThermExp(obj,value)
    #     obj.intLinThermExp = obj.checkCellArrayInput(value);
    # end%function

    def addAtom(self, atom, position):
        """ addAtom
        Adds an atomBase/atomMixed at a relative position of the unit
        cell.
        """

        positionStr = ''
        # test the input type of the position
        if isfunction(position):
            raise ValueError('Please use string representation of function!')
            pass
        elif isinstance(position, str):
            try:
                positionStr = position
                position = eval(position)
            except Exception as e:
                print('String input for unit cell property ' + position + ' \
                    cannot be converted to function handle!')
                print(e)
        elif isinstance(position, (int, float)):
            positionStr = 'lambda strain: {:e}*(strain+1)'.format(position)
            position = eval(positionStr);
        else:
            raise ValueError('Atom position input has to be a scalar, function \
            handle or string which can be converted into a function handle!')

        # add the atom at the end of the array
        self.atoms.append([atom, position, positionStr])
        # increase the number of atoms
        self.numAtoms = self.numAtoms + 1
        # Update the mass, density and spring constant of the unit cell
        # automatically:
        #
        # $$ \kappa = m \cdot (v_s / c)^2 $$

        self.mass = 0;
        for i  in range(self.numAtoms):
            self.mass = self.mass + self.atoms[i][0].mass

        self.density     = self.mass / self.volume
        # self.mass        = self.mass * 1*units.ang^2 / self.area;
        self.calcSpringConst()

    def addMultipleAtoms(self, atom, position, Nb):
        """addMultipleAtoms

        Adds multiple atomBase/atomMixed at a relative position of the unit
        cell.
        """
        for i in range(Nb):
           self.addAtom(atom,position)

    def calcSpringConst(self):
        """ calcSpringConst

        Calculates the spring constant of the unit cell from the mass,
        sound velocity and c-axis

        $$ k = m \, \left(\frac{v}{c}\right)^2 $$
        """
        self.springConst[0] = self.mass * (self.soundVel/self.cAxis)**2

    def getAcousticImpedance(self):
        """getAcousticImpedance
        """
        Z = np.sqrt(self.springConst[0] * self.mass/self.area**2)
        return(Z)

    @property
    def soundVel(self):
        return self._soundVel

    @soundVel.setter
    def soundVel(self, soundVel):
        """set.soundVel
        If the sound velocity is set, the spring constant is
        (re)calculated.
        """
        self._soundVel = soundVel
        self.calcSpringConst()

    def setHOspringConstants(self, HO):
        """setHOspringConstants

        Set the higher orders of the spring constant for anharmonic
        phonon simulations.
        """
        # check if HO is column vector and transpose it in this case
        if HO.shape[0] > 1:
            HO = HO.T

        self.springConst = np.delete(self.springConst, np.r_[1:len(self.springConst)]) # reset old higher order spring constants
        self.springConst = np.hstack((self.springConst, HO))

    def getAtomIDs(self):
        """getAtomIDs

        Returns a cell array of all atom IDs in the unit cell.
        """

        IDs = []
        for i in range(self.numAtoms):
            if not self.atoms[i][0].ID in IDs:
                IDs.append(self.atoms[i][0].ID)

        return IDs

    def getAtomPositions(self, *args):
        """getAtomPositions

        Returns a vector of all relative postion of the atoms in the unit
        cell.
        """

        if args:
            strain = args
        else:
            strain = 0

        # strains = strain*list(range(self.numAtoms))

        res = np.zeros([self.numAtoms])
        for i, atom in enumerate(self.atoms):
            res[i] = atom[1](strain)

        return res
