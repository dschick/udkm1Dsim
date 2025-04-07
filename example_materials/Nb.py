# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:44:23 2021

@author: matte
"""

import numpy as np
import udkm1Dsim as ud
u = ud.u


class Nb_1TM:
    def __init__(self):

        self.Nb = ud.Atom('Nb')

        self.prop = {}
        #self.prop['crystal_struc'] = 'fcc'
        self.prop['c_axis'] = 4.667*u.angstrom  # periodictable.com -- fcc geometry: 3.524*np.sqrt(3)/3
        self.prop['a_axis'] = 4.667*u.angstrom  # adjust density
        self.prop['b_axis'] = 3.3*u.angstrom  # adjust density
        #self.prop['molar_mass'] = 58.69*u.g
        self.prop['density'] = 8580*u.kg/(u.m**3)
        self.prop['deb_wal_fac'] = 0*u.m**2
        #self.prop['elastic_c11'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c12'] = 128e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c13'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c22'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c23'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c33'] = 351e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['sound_vel'] = 5.160*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        #self.prop['exp_c_axis'] = 12.7e-6  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        #self.prop['exp_a_axis'] = 12.7e-6  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        #self.prop['exp_b_axis'] = 12.7e-6  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['lin_therm_exp'] = 6.89e-6  # calculated: exp_c_axis*(1+2*c_13/c_33)
        #self.prop['Grun_c_axis'] = 1.8  # 10.1063/1.2902170
        #self.prop['Grun_a_axis'] = 1.8  # 10.1063/1.2902170
        #self.prop['Grun_b_axis'] = 1.8  # 10.1063/1.2902170
        self.prop['heat_capacity'] = 260*u.J/(u.kg * u.K)  # ph: 10.1016/0022-3697(81)90174-8
        self.prop['therm_cond'] = 53.3*u.W/(u.m * u.K)  # 10.1016/S0301-0104(99)00330-4
        self.prop['opt_pen_depth'] = np.inf*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 2.3344 + 3.2409j  # (800nm, thick film) Werner, J. Phys. Chem. Ref. Data 38 (2009)
        #self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Nb = ud.UnitCell(name, 'Nb', caxis, **prop)
        Nb.add_atom(self.Nb,0)
        Nb.add_atom(self.Nb,0)
        Nb.add_atom(self.Nb,0.5)
        Nb.add_atom(self.Nb,0.5)
        return Nb