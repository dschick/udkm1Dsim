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

"""A :mod:`XrayDynMag` module """

__all__ = ["XrayDynMag"]

__docformat__ = "restructuredtext"

import numpy as np
import scipy.constants as constants
from .xray import Xray
# from .unitCell import UnitCell
# from time import time
# from os import path
# from tqdm import trange
# from .helpers import make_hash_md5, m_power_x, m_times_n, finderb

r_0 = constants.physical_constants['classical electron radius'][0]


class XrayDynMag(Xray):
    """XrayDynMag

    Dynamical magnetic Xray simulations adapted from Elzo et.al. [4]_.
    Initially realized in `Project Dyna
    <http://neel.cnrs.fr/spip.php?rubrique1008>`_

    Original copyright notice:

    *Copyright Institut Neel, CNRS, Grenoble, France*

    **Project Collaborators:**

    - Stéphane Grenier, stephane.grenier@neel.cnrs.fr
    - Marta Elzo (PhD, 2009-2012)
    - Nicolas Jaouen Sextants beamline, Synchrotron Soleil,
      nicolas.jaouen@synchrotron-soleil.fr
    - Emmanuelle Jal (PhD, 2010-2013) now at `LCPMR CNRS, Paris
      <https://lcpmr.cnrs.fr/content/emmanuelle-jal>`_
    - Jean-Marc Tonnerre, jean-marc.tonnerre@neel.cnrs.fr
    - Ingrid Hallsteinsen - Padraic Shaffer’s group - Berkeley Nat. Lab.

    **Questions to:**

    - Stéphane Grenier, stephane.grenier@neel.cnrs.fr

    Args:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results

    Attributes:
        S (object): sample to do simulations with
        force_recalc (boolean): force recalculation of results
        polarization (float): polarization state
        last_atom_ref_trans_matrices (list): remember last result of
           atom ref_trans_matrices to speed up calculation

    References:

        .. [4] M. Elzo, E. Jal, O. Bunau, S. Grenier, Y. Joly, A. Y.
           Ramos, H. C. N. Tolentino, J. M. Tonnerre, and N. Jaouen,
           `J. Magn. Magn. Mater. 324, 105 (2012).
           <http://www.doi.org/10.1016/j.jmmm.2011.07.019>`_

    """

    def __init__(self, S, force_recalc, **kwargs):
        super().__init__(S, force_recalc, **kwargs)

    def __str__(self):
        """String representation of this class"""
        class_str = 'Dynamical magnetic X-Ray Diffraction simulation ' \
                    'properties:\n\n'
        class_str += super().__str__()
        return class_str

    def get_atom_ref_trans_matrix(self, atom, area, deb_wal_fac):
        """get_atom_ref_trans_matrix

        Returns the reflection-transmission matrix of an atom from
        Elzo algorithim:

        """
        M = len(self._energy)  # number of energies
        N = np.shape(self._qz)[1]  # number of q_z

        eps = np.zeros([3, 3, M, N], dtype=np.cfloat)
        u = np.zeros(3, dtype=np.float)
        mag = np.zeros([M, N], dtype=np.float)

        alphay = np.cos(self._theta)
        alphaz = np.zeros([M, N], dtype=np.cfloat)
        kz = np.zeros([M, N], dtype=np.cfloat)

        A = np.zeros([4, 4, M, N], dtype=np.cfloat)

        #wave_k = 5.0677e-4 * _Sim.EnergyRange

        for k in range(M):
            energy = self._energy[k]
            qz = self._qz[k, :]
            theta = self._theta[k, :]
            
    
            mag = _Sim.factor * _Sample[i].density[1] * _Sample[i].MMS[1] * _Sim.SF_mf[i]
    
            u   = [ np.sin(np.deg2rad(_Sample[i].phi[1])) * np.cos(np.deg2rad(_Sample[i].gamma[1])),
                    np.sin(np.deg2rad(_Sample[i].phi[1])) * np.sin(np.deg2rad(_Sample[i].gamma[1])),
                    np.cos(np.deg2rad(_Sample[i].phi[1]))]
    
            eps0 = 1 - _Sim.factor * _Sample[i].density[1] * _Sim.SF_cf[i]
            eps[:,0,0] =  eps0
            eps[:,0,1] = -1j * u[2] * mag
            eps[:,0,2] =  1j * u[1] * mag
            eps[:,1,0] = -eps[:,0,1]
            eps[:,1,1] =  eps0
            eps[:,1,2] = -1j * u[0] * mag
            eps[:,2,0] = -eps[:,0,2]
            eps[:,2,1] = -eps[:,1,2]
            eps[:,2,2] =  eps0
    
            alphay = np.cos(_Sim.AngleRangeRad) / np.sqrt(eps[:,0,0])
            alphaz = np.sqrt(1 - alphay**2)
    
            kz[i,:] = wave_k * np.sqrt(eps[:,0,0]) * alphaz          
    
            n_right_down = np.sqrt( eps[:,0,0] - 1j * eps[:,0,2] * alphay - 1j * eps[:,0,1] * alphaz )
            n_left_down  = np.sqrt( eps[:,0,0] + 1j * eps[:,0,2] * alphay + 1j * eps[:,0,1] * alphaz )
            n_right_up   = np.sqrt( eps[:,0,0] - 1j * eps[:,0,2] * alphay + 1j * eps[:,0,1] * alphaz )
            n_left_up    = np.sqrt( eps[:,0,0] + 1j * eps[:,0,2] * alphay - 1j * eps[:,0,1] * alphaz )
    
            alphay_right_down  = np.cos(_Sim.AngleRangeRad)/n_right_down
            alphaz_right_down  = np.sqrt(1-alphay_right_down**2)
            alphay_left_down   = np.cos(_Sim.AngleRangeRad)/n_left_down
            alphaz_left_down   = np.sqrt(1-alphay_left_down**2) 
            alphay_right_up    = np.cos(_Sim.AngleRangeRad)/n_right_up
            alphaz_right_up    = np.sqrt(1-alphay_right_up**2)
            alphay_left_up     = np.cos(_Sim.AngleRangeRad)/n_left_up
            alphaz_left_up     = np.sqrt(1-alphay_left_up**2)
         
            A[i,:,0,0] = -1 - 1j * eps[:,0,1] * alphaz_right_down  - 1j * eps[:,0,2] * alphay_right_down
            A[i,:,0,1] =  1 - 1j * eps[:,0,1] * alphaz_left_down   - 1j * eps[:,0,2] * alphay_left_down
            A[i,:,0,2] = -1 + 1j * eps[:,0,1] * alphaz_right_up    - 1j * eps[:,0,2] * alphay_right_up
            A[i,:,0,3] =  1 + 1j * eps[:,0,1] * alphaz_left_up     - 1j * eps[:,0,2] * alphay_left_up
    
            A[i,:,1,0] =  1j * alphaz_right_down - eps[:,0,1]  - 1j * eps[:,1,2] * alphay_right_down 
            A[i,:,1,1] =  1j * alphaz_left_down  + eps[:,0,1]  - 1j * eps[:,1,2] * alphay_left_down
            A[i,:,1,2] = -1j * alphaz_right_up   - eps[:,0,1]  - 1j * eps[:,1,2] * alphay_right_up
            A[i,:,1,3] = -1j * alphaz_left_up    + eps[:,0,1]  - 1j * eps[:,1,2] * alphay_left_up
    
            A[i,:,2,0] = -1j * n_right_down * A[i,:,0,0]
            A[i,:,2,1] =  1j * n_left_down  * A[i,:,0,1]
            A[i,:,2,2] = -1j * n_right_up   * A[i,:,0,2]
            A[i,:,2,3] =  1j * n_left_up    * A[i,:,0,3]
    
            A[i,:,3,0] = - alphaz_right_down * n_right_down * A[i,:,0,0]
            A[i,:,3,1] = - alphaz_left_down  * n_left_down  * A[i,:,0,1] 
            A[i,:,3,2] =   alphaz_right_up   * n_right_up   * A[i,:,0,2] 
            A[i,:,3,3] =   alphaz_left_up    * n_left_up    * A[i,:,0,3]
        
            APhi[i,:,0,0] = -1 + 1j * eps[:,0,1] * alphaz_left_down  + 1j * eps[:,0,2] * alphay_left_down
            APhi[i,:,0,1] =  1 + 1j * eps[:,0,1] * alphaz_right_down + 1j * eps[:,0,2] * alphay_right_down
            APhi[i,:,0,2] = -1 - 1j * eps[:,0,1] * alphaz_left_up    + 1j * eps[:,0,2] * alphay_left_up
            APhi[i,:,0,3] =  1 - 1j * eps[:,0,1] * alphaz_right_up   + 1j * eps[:,0,2] * alphay_right_up
    
            APhi[i,:,1,0] =  1j * alphaz_left_down + eps[:,0,1]  + 1j * eps[:,1,2] * alphay_left_down 
            APhi[i,:,1,1] =  1j * alphaz_right_down  - eps[:,0,1]  + 1j * eps[:,1,2] * alphay_right_down
            APhi[i,:,1,2] = -1j * alphaz_left_up   + eps[:,0,1]  + 1j * eps[:,1,2] * alphay_left_up
            APhi[i,:,1,3] = -1j * alphaz_right_up    - eps[:,0,1]  + 1j * eps[:,1,2] * alphay_right_up
     
            APhi[i,:,2,0] =  1j * n_left_down * APhi[i,:,0,0]
            APhi[i,:,2,1] = -1j * n_right_down  * APhi[i,:,0,1]
            APhi[i,:,2,2] =  1j * n_left_up   * APhi[i,:,0,2]
            APhi[i,:,2,3] = -1j * n_right_up    * APhi[i,:,0,3]
    
            APhi[i,:,3,0] = - alphaz_left_down * n_left_down * APhi[i,:,0,0]
            APhi[i,:,3,1] = - alphaz_right_down  * n_right_down  * APhi[i,:,0,1] 
            APhi[i,:,3,2] =   alphaz_left_up   * n_right_up   * APhi[i,:,0,2] 
            APhi[i,:,3,3] =   alphaz_right_up    * n_right_up    * APhi[i,:,0,3]
    
            A[i,:]    =    A[i,:] / (np.sqrt(2) * eps[:,0,0].reshape((len(_Sim.XRange),1,1))) 
            APhi[i,:] = APhi[i,:] / (np.sqrt(2) * eps[:,0,0].reshape((len(_Sim.XRange),1,1)))
    
            phase = wave_k * _Sample[i].thick[1]
    
            P[i,:,0,0] = np.exp( 1j * phase * n_right_down * alphaz_right_down)
            P[i,:,1,1] = np.exp( 1j * phase * n_left_down  * alphaz_left_down)
            P[i,:,2,2] = np.exp(-1j * phase * n_right_up   * alphaz_right_up)
            P[i,:,3,3] = np.exp(-1j * phase * n_left_up    * alphaz_left_up)
            PPhi[i,:,0,0] = P[i,:,1,1]
            PPhi[i,:,1,1] = P[i,:,0,0]
            PPhi[i,:,2,2] = P[i,:,3,3]
            PPhi[i,:,3,3] = P[i,:,2,2]

    def F0F1SampleLoop(_Sample, _Sim):  # _SF to be added
        
        eps=np.zeros((len(_Sim.XRange),3,3), dtype=np.cfloat)
        u = np.zeros(3, dtype=np.float)
        mag=np.zeros(len(_Sim.XRange), dtype=np.float)
    
        alphay = np.cos(_Sim.AngleRangeRad)
        alphaz = np.zeros(len(_Sim.AngleRangeRad), dtype=np.cfloat)  
        kz =np.zeros((len(_Sample), len(_Sim.XRange)), dtype=np.cfloat) # one array of _sample arrays with xrange elements
        
        A = np.zeros((len(_Sample), len(_Sim.XRange), 4, 4), dtype=np.cfloat)
        APhi = np.zeros((len(_Sample), len(_Sim.XRange), 4, 4), dtype=np.cfloat)
        P = np.zeros((len(_Sample), len(_Sim.XRange), 4, 4), dtype=np.cfloat)
        PPhi = np.zeros((len(_Sample), len(_Sim.XRange), 4, 4), dtype=np.cfloat)   
    
        wave_k = 5.0677e-4 * _Sim.EnergyRange
    
        for i in range(len(_Sample)): # i=0 is first layer, ... i=len(_sample) is substrate
    
            mag = _Sim.factor * _Sample[i].density[1] * _Sample[i].MMS[1] * _Sim.SF_mf[i]
    
            u   = [ np.sin(np.deg2rad(_Sample[i].phi[1])) * np.cos(np.deg2rad(_Sample[i].gamma[1])),
                    np.sin(np.deg2rad(_Sample[i].phi[1])) * np.sin(np.deg2rad(_Sample[i].gamma[1])),
                    np.cos(np.deg2rad(_Sample[i].phi[1]))]
    
            eps0 = 1 - _Sim.factor * _Sample[i].density[1] * _Sim.SF_cf[i]
            eps[:,0,0] =  eps0
            eps[:,0,1] = -1j * u[2] * mag
            eps[:,0,2] =  1j * u[1] * mag
            eps[:,1,0] = -eps[:,0,1]
            eps[:,1,1] =  eps0
            eps[:,1,2] = -1j * u[0] * mag
            eps[:,2,0] = -eps[:,0,2]
            eps[:,2,1] = -eps[:,1,2]
            eps[:,2,2] =  eps0
    
            alphay = np.cos(_Sim.AngleRangeRad) / np.sqrt(eps[:,0,0])
            alphaz = np.sqrt(1 - alphay**2)
    
            kz[i,:] = wave_k * np.sqrt(eps[:,0,0]) * alphaz          
    
            n_right_down = np.sqrt( eps[:,0,0] - 1j * eps[:,0,2] * alphay - 1j * eps[:,0,1] * alphaz )
            n_left_down  = np.sqrt( eps[:,0,0] + 1j * eps[:,0,2] * alphay + 1j * eps[:,0,1] * alphaz )
            n_right_up   = np.sqrt( eps[:,0,0] - 1j * eps[:,0,2] * alphay + 1j * eps[:,0,1] * alphaz )
            n_left_up    = np.sqrt( eps[:,0,0] + 1j * eps[:,0,2] * alphay - 1j * eps[:,0,1] * alphaz )
    
            alphay_right_down  = np.cos(_Sim.AngleRangeRad)/n_right_down
            alphaz_right_down  = np.sqrt(1-alphay_right_down**2)
            alphay_left_down   = np.cos(_Sim.AngleRangeRad)/n_left_down
            alphaz_left_down   = np.sqrt(1-alphay_left_down**2) 
            alphay_right_up    = np.cos(_Sim.AngleRangeRad)/n_right_up
            alphaz_right_up    = np.sqrt(1-alphay_right_up**2)
            alphay_left_up     = np.cos(_Sim.AngleRangeRad)/n_left_up
            alphaz_left_up     = np.sqrt(1-alphay_left_up**2)
         
            A[i,:,0,0] = -1 - 1j * eps[:,0,1] * alphaz_right_down  - 1j * eps[:,0,2] * alphay_right_down
            A[i,:,0,1] =  1 - 1j * eps[:,0,1] * alphaz_left_down   - 1j * eps[:,0,2] * alphay_left_down
            A[i,:,0,2] = -1 + 1j * eps[:,0,1] * alphaz_right_up    - 1j * eps[:,0,2] * alphay_right_up
            A[i,:,0,3] =  1 + 1j * eps[:,0,1] * alphaz_left_up     - 1j * eps[:,0,2] * alphay_left_up
    
            A[i,:,1,0] =  1j * alphaz_right_down - eps[:,0,1]  - 1j * eps[:,1,2] * alphay_right_down 
            A[i,:,1,1] =  1j * alphaz_left_down  + eps[:,0,1]  - 1j * eps[:,1,2] * alphay_left_down
            A[i,:,1,2] = -1j * alphaz_right_up   - eps[:,0,1]  - 1j * eps[:,1,2] * alphay_right_up
            A[i,:,1,3] = -1j * alphaz_left_up    + eps[:,0,1]  - 1j * eps[:,1,2] * alphay_left_up
    
            A[i,:,2,0] = -1j * n_right_down * A[i,:,0,0]
            A[i,:,2,1] =  1j * n_left_down  * A[i,:,0,1]
            A[i,:,2,2] = -1j * n_right_up   * A[i,:,0,2]
            A[i,:,2,3] =  1j * n_left_up    * A[i,:,0,3]
    
            A[i,:,3,0] = - alphaz_right_down * n_right_down * A[i,:,0,0]
            A[i,:,3,1] = - alphaz_left_down  * n_left_down  * A[i,:,0,1] 
            A[i,:,3,2] =   alphaz_right_up   * n_right_up   * A[i,:,0,2] 
            A[i,:,3,3] =   alphaz_left_up    * n_left_up    * A[i,:,0,3]
        
            APhi[i,:,0,0] = -1 + 1j * eps[:,0,1] * alphaz_left_down  + 1j * eps[:,0,2] * alphay_left_down
            APhi[i,:,0,1] =  1 + 1j * eps[:,0,1] * alphaz_right_down + 1j * eps[:,0,2] * alphay_right_down
            APhi[i,:,0,2] = -1 - 1j * eps[:,0,1] * alphaz_left_up    + 1j * eps[:,0,2] * alphay_left_up
            APhi[i,:,0,3] =  1 - 1j * eps[:,0,1] * alphaz_right_up   + 1j * eps[:,0,2] * alphay_right_up
    
            APhi[i,:,1,0] =  1j * alphaz_left_down + eps[:,0,1]  + 1j * eps[:,1,2] * alphay_left_down 
            APhi[i,:,1,1] =  1j * alphaz_right_down  - eps[:,0,1]  + 1j * eps[:,1,2] * alphay_right_down
            APhi[i,:,1,2] = -1j * alphaz_left_up   + eps[:,0,1]  + 1j * eps[:,1,2] * alphay_left_up
            APhi[i,:,1,3] = -1j * alphaz_right_up    - eps[:,0,1]  + 1j * eps[:,1,2] * alphay_right_up
     
            APhi[i,:,2,0] =  1j * n_left_down * APhi[i,:,0,0]
            APhi[i,:,2,1] = -1j * n_right_down  * APhi[i,:,0,1]
            APhi[i,:,2,2] =  1j * n_left_up   * APhi[i,:,0,2]
            APhi[i,:,2,3] = -1j * n_right_up    * APhi[i,:,0,3]
    
            APhi[i,:,3,0] = - alphaz_left_down * n_left_down * APhi[i,:,0,0]
            APhi[i,:,3,1] = - alphaz_right_down  * n_right_down  * APhi[i,:,0,1] 
            APhi[i,:,3,2] =   alphaz_left_up   * n_right_up   * APhi[i,:,0,2] 
            APhi[i,:,3,3] =   alphaz_right_up    * n_right_up    * APhi[i,:,0,3]
    
            A[i,:]    =    A[i,:] / (np.sqrt(2) * eps[:,0,0].reshape((len(_Sim.XRange),1,1))) 
            APhi[i,:] = APhi[i,:] / (np.sqrt(2) * eps[:,0,0].reshape((len(_Sim.XRange),1,1)))
    
            phase = wave_k * _Sample[i].thick[1]
    
            P[i,:,0,0] = np.exp( 1j * phase * n_right_down * alphaz_right_down)
            P[i,:,1,1] = np.exp( 1j * phase * n_left_down  * alphaz_left_down)
            P[i,:,2,2] = np.exp(-1j * phase * n_right_up   * alphaz_right_up)
            P[i,:,3,3] = np.exp(-1j * phase * n_left_up    * alphaz_left_up)
            PPhi[i,:,0,0] = P[i,:,1,1]
            PPhi[i,:,1,1] = P[i,:,0,0]
            PPhi[i,:,2,2] = P[i,:,3,3]
            PPhi[i,:,3,3] = P[i,:,2,2]
        
        return (A, APhi, P, PPhi, alphaz, kz, wave_k)