#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udkm1Dsim import Magnetization, LLB
from udkm1Dsim import u
import numpy as np
import pytest


# fixtures


@pytest.fixture(scope='module')
def magnetization(structure, tmp_path_factory):
    return Magnetization(structure, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
                         save_data=True, disp_messages=True, progress_bar=True,
                         )


@pytest.fixture(scope='module')
def llb(structure_amorph, tmp_path_factory):
    return LLB(structure_amorph, force_recalc=True, cache_dir=tmp_path_factory.mktemp('cache'),
               save_data=True, disp_messages=True, progress_bar=True,
               )


@pytest.fixture(scope='module')
def delays():
    return np.r_[-1:10:1]*u.ps


@pytest.fixture(scope='module')
def distances(structure_amorph):
    dists, _, _ = structure_amorph.get_distances_of_layers()
    return dists


@pytest.fixture(scope='module')
def temp_map(delays, distances):
    return 50*np.ones([len(delays), len(distances), 1])


# tests


# Magnetization


def test_magnetization_str(magnetization):
    magnetization.__str__()


def test_magnetization_get_hash(magnetization, delays, temp_map):
    magnetization.get_hash(delays=delays, temp_map=temp_map)


def test_magnetization_check_initial_magnetization(magnetization):
    # must fail as UnitCells are not yet supported, see #129
    with pytest.raises(TypeError):
        magnetization.check_initial_magnetization([])


# LLB


def test_llb_str(llb):
    llb.__str__()


def test_llb_check_initial_magnetization(llb, distances):
    llb.check_initial_magnetization(np.array([0, 0, 0]), distances=distances)
    llb.check_initial_magnetization(np.zeros([len(distances), 3]), distances=distances)
    with pytest.raises(ValueError):
        llb.check_initial_magnetization(np.zeros([3, len(distances)]),
                                        distances=distances)


def test_llb_get_magnetization(llb, temp_map, delays):
    llb.get_magnetization_map(delays, temp_map=temp_map)
    llb.force_recalc = False
    llb.get_magnetization_map(delays, temp_map=temp_map)
