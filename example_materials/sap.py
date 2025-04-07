# -*- coding: utf-8 -*-

import numpy as np
import udkm1Dsim as ud
u = ud.u


class sap_1TM:
    def __init__(self):

        self.Al = ud.Atom('Al')
        self.O = ud.Atom('O')

        self.prop = {}
        #self.prop['crystal_struc'] = 'fcc'
        self.prop['c_axis'] = 4.758*u.angstrom  # periodictable.com -- fcc geometry: 3.524*np.sqrt(3)/3
        self.prop['a_axis'] = 8.2411*u.angstrom  # adjust density
        self.prop['b_axis'] = 12.8042*u.angstrom  # adjust density
        #self.prop['molar_mass'] = 58.69*u.g
        self.prop['density'] = 3980*u.kg/(u.m**3)
        self.prop['deb_wal_fac'] = 0*u.m**2
        #self.prop['elastic_c11'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c12'] = 128e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c13'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c22'] = 327e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c23'] = 103e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        #self.prop['elastic_c33'] = 351e9*u.kg/(u.m*u.s**2)  # 10.1063/1.1702218: c11=253, c12=152, c44=124
        self.prop['sound_vel'] = 11.2*u.nm/u.ps  # calculated -- np.sqrt(c33/density)
        self.prop['phonon_damping'] = 0*u.kg/u.s
        #self.prop['exp_c_axis'] = 12.7e-6  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        #self.prop['exp_a_axis'] = 12.7e-6  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        #self.prop['exp_b_axis'] = 12.7e-6  # calculated: Grun*heat_cap*dens/(c_13+c_23+c_33)
        self.prop['lin_therm_exp'] = 5.38*1e-6  # calculated: exp_c_axis*(1+2*c_13/c_33)
        #self.prop['Grun_c_axis'] = 1.8  # 10.1063/1.2902170
        #self.prop['Grun_a_axis'] = 1.8  # 10.1063/1.2902170
        #self.prop['Grun_b_axis'] = 1.8  # 10.1063/1.2902170
        self.prop['heat_capacity'] = 790*u.J/(u.kg * u.K)  # ph: 10.1016/0022-3697(81)90174-8
        self.prop['therm_cond'] = 40*u.W/(u.m * u.K)  # 10.1016/S0301-0104(99)00330-4
        self.prop['opt_pen_depth'] = np.inf*u.nm  # (800nm) estimation
        self.prop['opt_ref_index'] = 1.76 + 0.0j  # (800nm, thick film) Werner, J. Phys. Chem. Ref. Data 38 (2009)
        #self.prop['opt_ref_index_per_strain'] = 0+0j

    def createUnitCell(self, name, caxis, prop):
        Sap = ud.UnitCell(name, 'Sap', caxis, **prop)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.O,0.153015)
        Sap.add_atom(self.O,0.153015)
        Sap.add_atom(self.O,0.194)
        Sap.add_atom(self.O,0.30603)
        Sap.add_atom(self.O,0.346995)
        Sap.add_atom(self.O,0.346995)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.O,0.653)
        Sap.add_atom(self.O,0.653)
        Sap.add_atom(self.O,0.69395)
        Sap.add_atom(self.O,0.806)
        Sap.add_atom(self.O,0.847)
        Sap.add_atom(self.O,0.847)

        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.O,0.153015)
        Sap.add_atom(self.O,0.153015)
        Sap.add_atom(self.O,0.194)
        Sap.add_atom(self.O,0.30603)
        Sap.add_atom(self.O,0.346995)
        Sap.add_atom(self.O,0.346995)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.O,0.653)
        Sap.add_atom(self.O,0.653)
        Sap.add_atom(self.O,0.69395)
        Sap.add_atom(self.O,0.806)
        Sap.add_atom(self.O,0.847)
        Sap.add_atom(self.O,0.847)

        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.Al,0)
        Sap.add_atom(self.O,0.153015)
        Sap.add_atom(self.O,0.153015)
        Sap.add_atom(self.O,0.194)
        Sap.add_atom(self.O,0.30603)
        Sap.add_atom(self.O,0.346995)
        Sap.add_atom(self.O,0.346995)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.Al,0.5)
        Sap.add_atom(self.O,0.653)
        Sap.add_atom(self.O,0.653)
        Sap.add_atom(self.O,0.69395)
        Sap.add_atom(self.O,0.806)
        Sap.add_atom(self.O,0.847)
        Sap.add_atom(self.O,0.847)
        return Sap