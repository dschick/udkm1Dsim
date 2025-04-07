# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:44:23 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u


class Al_1TM:
    def __init__(self):

        self.Al = ud.Atom('Al')

        self.prop = {}
        self.prop['crystal_struc'] = 'fcc'
        self.prop['c_axis'] = 1*u.angstrom  # periodictable.com -- fcc geometry: 3.524*np.sqrt(3)/3
        self.prop['a_axis'] = 1*u.angstrom  # adjust density
        self.prop['b_axis'] = 1*u.angstrom  # adjust density

        self.prop['density'] = 2700*u.kg/(u.m**3)
        self.prop['deb_wal_fac'] = 0*u.m**2

        self.prop['sound_vel'] = 5.1*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s

        self.prop['lin_therm_exp'] = 23.1e-6  # calculated: exp_c_axis*(1+2*c_13/c_33)
        self.prop['heat_capacity'] = 897*u.J/(u.kg * u.K)  # ph: 10.1016/0022-3697(81)90174-8
        self.prop['therm_cond'] = 235*u.W/(u.m * u.K)  # 10.1016/S0301-0104(99)00330-4
        self.prop['opt_pen_depth'] = np.inf*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 0.49 + 4.84j  # (800nm, thick film) Werner, J. Phys. Chem. Ref. Data 38 (2009)
