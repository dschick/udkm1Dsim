




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

import more_itertools
import itertools

class structure(object):
    
    
    """structure
    The structure class can hold various substructures. Each substructure can be either a layer of N unitCell 
    objects or a structure by itself. Thus it is possible to recursively build up 1D structures.
    """
    
    def __init__(self, name, **kwargs):
        """
        Properties (SetAccess=public,GetAccess=public)
        name                % STRING name of sample
        substructures = []; % CELL ARRAY of structures in sample
        substrate           % OBJECT HANDLE structure of the substrate
        numSubSystems = 1;  % INTEGER number of subsystems for heat and phonons (electronic, lattice, spins, ...) 
        """
        self.name             =  name
        self.numSubSystems    =  1
        self.substructures    =  []
        self.substrate        =  []
   
        
    
    def addSubStructure(self,subStructure,N):
        
        """Add a substructure of N unitCells or N structures to the structure."""
        
        #check of the substructure is an instance of the unitCell of structure class
        
        if not (isinstance(subStructure,unitCell) or isinstance(subStructure,structure)):
            raise ValueError('Class '+type(subStructure).__name__+' is no possible sub structure.Only unitCell and structure classes are allowed!')
        pass
    
        # if a structure is added as a substructure, the substructure can not have a substrate!   
        
        if isinstance(subStructure,structure):
            if (subStructure.substrate):
                raise ValueError('No substrate in substructure allowed!')
                
        # check the number of subsystems of the substructure   
            
        if ((self.numSubSystems>1) and not(subStructure.numSubSystems == self.numSubSystems)):
            raise ValueError('The number of subsystems in each substructure must be the same!')
        else:
            self.numSubSystems = subStructure.numSubSystems
        
        #add a substructure of N repetitions to the structure with
        self.substructures.append([subStructure, N])
         
    
    

    def addSubstrate(self,subStructure):
        
        """Add a structure as static substrate to the structure"""
        
        if not isinstance(subStructure,structure):
            raise ValueError('Class '+type(subStructure).__name__+' is no possible substrate. Only structure class is allowed!')
            
        self.substrate = subStructure
    
    def getNumberOfSubStructures(self):
        
        """Returns the number of all sub structures. 
        This methods does not return the number of all unitCells in the structure, see getNumberOfUnitCells()."""
        
        N = 0
        for i in range(len(self.substructures)):
            if isinstance(self.substructures[i][0],unitCell):
                N = N + 1
        
            else:
                N = N + self.substructures[i][0].getNumberOfSubStructures()
        return N  
    
    def getNumberOfUnitCells(self):
        
        """Returns the number of all unitCells in the structure."""
        N = 0
        #%traverse the substructres
        for i in range(len(self.substructures)):
            if isinstance(self.substructures[i][0],unitCell):
                N = N + self.substructures[i][1]
            else:
                #% its a sturcture, so call the method recursively
                N = N + self.substructures[i][0].getNumberOfUnitCells()*self.substructures[i][1]
                    
        return N  
    
    
    def getNumberOfUniqueUnitCells(self):
        
        """Returns the number of unique unitCells in the structure."""
        
        N = len(self.getUniqueUnitCells()[0])
        return N
    
    def getLength(self):
        
        """Returns the length from surface to bottom of the structure"""
        
        pass
    

    
    def getUniqueUnitCells(self):
        
        """Returns a cell array of IDs and handles of all unique unitCell instances in the structure.
        The uniqueness is determined by the handle of each unitCell instance."""
        
        UCIDs = []
        UCHandles = []
        #traverse the substructures
        for i in range(len(self.substructures)):
            if isinstance(self.substructures[i][0],unitCell):
                #its a UnitCell
                ID = self.substructures[i][0].ID
                if not UCIDs:
                    #the cell array is empty at the beginning so add
                    #the first unitCell
                    UCIDs = UCIDs + [ID]
                    UCHandles = UCHandles + [S.substructures[i][0]]
                else:
                    #the cell array is not empty so check if the ID is
                    #already in the UCs ID vector
                    if ID not in UCIDs:
                        #if ID not in list, so add it
                        UCIDs = UCIDs + [ID]
                        UCHandles = UCHandles + [S.substructures[i][0]]
            else:
                #its a substructure
                if not UCIDs:
                    #the cell array is empty at the beginning so call
                    #the method recursively and add the result to the
                    #UCs array
                    UCIDs = self.substructures[i][0].getUniqueUnitCells()[0]
                    UCHandles = self.substructures[i][0].getUniqueUnitCells()[1]
                else:
                    #the cell array is not empty so check if the IDs
                    #from the recursive call are already in the UCs ID
                    #vector.
                    temp1 = self.substructures[i][0].getUniqueUnitCells()[0]
                    temp2 = self.substructures[i][0].getUniqueUnitCells()[1]
                    for j in range(len(temp1)):
                        #check all IDs from recursive call
                        if temp1[j] not in UCIDs:
                            #IDs not in list, so add them
                            UCIDs = UCIDs + [temp1[j]]
                            UCHandles = UCHandles + [temp2[j]] 
                            
        return UCIDs,UCHandles
    
    

    def getUnitCellVectors(self,*args):
        
        """Returns three vectors with the numeric index of all unit cells in a structure given by the getUniqueUnitCells() method and addidionally vectors with the IDs and Handles of the corresponding unitCell instances. 
        The list and order of the unique unitCells can be either handed as an input parameter or is requested at the beginning."""
        

        Indices     =  []
        UCIDs       =  []
        UCHandles   =  []
        # if no UCs (UniqueUnitCells) are given, we have to get them
        if (len(args)<1):
            UCs = self.getUniqueUnitCells()
        else:
            UCs = args[0]
        # traverse the substructres
        for i in range(len(self.substructures)):
            if isinstance(self.substructures[i][0],unitCell):
            #its a UnitCell
            #find the index of the current UC ID in the unique
            #unitCell vector
                Index = UCs[0].index(self.substructures[i][0].ID)
                #add the index N times to the Indices vector
                Indices = np.append(Indices,Index*np.ones(self.substructures[i][1]))
                #create a cell array of N unitCell IDs and add them to
                #the IDs cell array
                temp1 = list(itertools.repeat(self.substructures[i][0].ID, self.substructures[i][1]))
                UCIDs = UCIDs + list(temp1)
                #% create a cell array of N unitCell handles and add them to
                #the Handles cell array
                temp2 = list(itertools.repeat(self.substructures[i][0], self.substructures[i][1]))
                UCHandles = UCHandles + list(temp2)
            else:
                #its a structure
                #make a recursive call and hand in the same unique
                #unit cell vector as we used before
                [temp1, temp2, temp3] =  self.substructures[i][0].getUnitCellVectors(UCs)
                temp11 = []
                temp22 = []
                temp33 = []
                # concat the temporary arrays N times
                for j in range(self.substructures[i][1]):
                    temp11 = temp11 + list(temp1)
                    temp22 = temp22 + list(temp2)
                    temp33 = temp33 + list(temp3)
                #add the temporary arrays to the outputs
                Indices = np.append(Indices,temp11)
                UCIDs = UCIDs + list(temp22)
                UCHandles = UCHandles + list(temp33)
        return Indices, UCIDs, UCHandles
    
    
    def getAllPositionsPerUniqueUnitCell(self):
        
            """Returns a cell array with one vector of position indices for each unique unitCell in the structure."""
            
            UCs     = self.getUniqueUnitCells()
            Indices = self.getUnitCellVectors()[0]
            Pos     = {} # Dictionary used instead of array
            for i in range(len(UCs[0])):
                Pos[UCs[0][i]] = list(np.where(Indices == i))
            #Each element accessible through Unit cell ID
            return Pos
    
    
    def getDistancesOfUnitCells(self):
        
        """Returns a vector of the distance from the surface for each unit cell starting at 0 (dStart) 
        and starting at the end of the first UC (dEnd) and from the center of each UC (dMid)."""
        
        cAxes = self.getUnitCellPropertyVector(types = 'cAxis')
        dEnd  = np.cumsum(cAxes)
        dStart= np.hstack([[0],dEnd[0:-1]])
        dMid  = (dStart + cAxes)/2
        return dStart,dEnd,dMid 
    
    
   def getUnitCellPropertyVector(self,**kwargs):
        
   
        """Returns a vector for a property of all unitCells in the structure.
        The property is determined by the propertyName and returns a scalar value or a function handle."""
        
        types = kwargs.get('types')
        

        #get the Handle to all unitCells in the Structure
        Handles = self.getUnitCellVectors()[2]

        
        if callable(getattr(Handles[0],types)):
            Prop = np.zeros([self.getNumberOfUnitCells()])
            for i in range(self.getNumberOfUnitCells()):
                Prop[i] = getattr(Handles[i],types)
        
        elif type(getattr(Handles[0],types)) is list:
            #Prop = np.zeros([self.getNumberOfUnitCells(),len(getattr(Handles[0],types))])
            #Prop[]
            Prop = {} 
            for i in range(self.getNumberOfUnitCells()):
                #Prop = Prop + getattr(Handles[i],types)
                Prop[i] =  getattr(Handles[i],types)
        else:
            
            UCs = self.getUniqueUnitCells()
            Prop = np.zeros([self.getNumberOfUnitCells()])
        
            #% traverse all unitCells
            
            for i in range(self.getNumberOfUnitCells()):
                temp = getattr(Handles[i],types)
                Prop[i] = temp
                
        return Prop    
    
    
    def getUnitCellHandle(self,i):
        
        """Returns the handle to the unitCell at position i in the structure."""
        
        Handles = self.getUnitCellVectors()[2]
        Handle  = Handles[i]
        return Handle
    
    
