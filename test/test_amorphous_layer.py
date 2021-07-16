#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Atom
from udkm1Dsim import AmorphousLayer
from udkm1Dsim import u
u.default_format = '~P'
import numpy as np


def test_amorphous_layer():
    Fe = Atom('Fe')
    al = AmorphousLayer('al', 'Amorphous Layer', thickness=2.86*u.angstrom, density=10*u.g/u.cm**3,
                        atom=Fe, heat_capacity=10*(u.J/u.kg/u.K), lin_therm_exp=1e-6/u.K,
                        therm_cond=1*(u.W/u.m/u.K), opt_pen_depth=11*u.nm, sound_vel=5*(u.nm/u.ps))

    assert al.id == 'al'
    assert al.name == 'Amorphous Layer'
    assert al.thickness == 2.86*u.angstrom
    assert al.heat_capacity[0](300) == 10
    assert al.int_heat_capacity[0](300) == 3000
    assert al.lin_therm_exp[0](300) == 1e-6
    assert al.int_lin_therm_exp[0](300) == 0.0003
    assert al.therm_cond[0](300) == 1
    assert al.opt_pen_depth == 11*u.nm
    assert al.sound_vel == 5*(u.nm/u.ps)
    # test temperature-dependent parameters for float input
    al.num_sub_systems = 2
    al.heat_capacity = [10, 1000]
    al.therm_cond = [10, 1000]
    al.lin_therm_exp = [10, 1000]
    assert al.heat_capacity_str == ['10', '1000']
    assert al.heat_capacity[0](300) == 10
    assert al.heat_capacity[1](300) == 1000
    assert al.int_heat_capacity_str == ['10*T', '1000*T']
    assert al.int_heat_capacity[0](300) == 3000
    assert al.int_heat_capacity[1](300) == 300000
    assert al.therm_cond_str == ['10', '1000']
    assert al.therm_cond[0](300) == 10
    assert al.therm_cond[1](300) == 1000
    assert al.lin_therm_exp_str == ['10', '1000']
    assert al.lin_therm_exp[0](300) == 10
    assert al.lin_therm_exp[1](300) == 1000
    assert al.int_lin_therm_exp_str == ['10*T', '1000*T']
    assert al.int_lin_therm_exp[0](300) == 3000
    assert al.int_lin_therm_exp[1](300) == 300000
    # test temperature-dependent parameters for str function input
    al.heat_capacity = ['10*T', 'exp(300-T)+300']
    al.therm_cond = ['10*T', 'exp(300-T)+300']
    al.lin_therm_exp = ['10*T', 'exp(300-T)+300']
    assert al.heat_capacity_str == ['10*T', 'exp(300-T)+300']
    assert al.heat_capacity[0](300) == 3000
    assert al.heat_capacity[1](300) == 301
    assert al.int_heat_capacity_str == ['5*T**2', '300*T - exp(300 - T)']
    assert al.int_heat_capacity[0](300) == 450000
    assert al.int_heat_capacity[1](300) == 89999.0
    assert al.therm_cond_str == ['10*T', 'exp(300-T)+300']
    assert al.therm_cond[0](300) == 3000
    assert al.therm_cond[1](300) == 301
    assert al.lin_therm_exp_str == ['10*T', 'exp(300-T)+300']
    assert al.lin_therm_exp[0](300) == 3000
    assert al.lin_therm_exp[1](300) == 301
    assert al.int_lin_therm_exp_str == ['5*T**2', '300*T - exp(300 - T)']
    assert al.int_lin_therm_exp[0](300) == 450000
    assert al.int_lin_therm_exp[1](300) == 89999.0
    # check backward compatibility
    al.heat_capacity = ['lambda T: 10*T', 'lambda T: exp(300-T)+300']
    assert al.heat_capacity_str == ['10*T', 'exp(300-T)+300']
    assert al.heat_capacity[0](300) == 3000
    assert al.heat_capacity[1](300) == 301
    assert al.int_heat_capacity_str == ['5*T**2', '300*T - exp(300 - T)']
    assert al.int_heat_capacity[0](300) == 450000
    assert al.int_heat_capacity[1](300) == 89999.0
    # check subsystem temperatures
    al.therm_cond = ['10*T_0 + 30*T_1', 'exp(300-T_1)+300']
    assert al.therm_cond[0](np.array([300, 300])) == 12000
    assert al.therm_cond[1](np.array([300, 300])) == 301
    al.sub_system_coupling = ['500*(T_0-T_1)', 'lambda T: -500*(T[0]-T[1])']
    assert al.sub_system_coupling[0](np.array([301, 300])) == 500
    assert al.sub_system_coupling[1](np.array([301, 300])) == -500
