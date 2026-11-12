#!/usr/bin/env python

import copy
import warnings
from types import SimpleNamespace

import numpy as np
import pint
import pytest

import udkm1Dsim as ud
from udkm1Dsim import LLB, LLG, Magnetization, Result, iLLG, u
from udkm1Dsim.helpers import convert_polar_to_cartesian
from udkm1Dsim.simulations.solvers import magnetization_scipy, mean_field

# cartesian coordinates of a unit vector along x
ALONG_X = np.array([1.0, 0.0, 0.0])

# fixtures


@pytest.fixture(scope="module")
def magnetization(structure, tmp_path_factory):
    return Magnetization(
        structure,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def llb(structure_amorph, tmp_path_factory):
    return LLB(
        structure_amorph,
        force_recalc=True,
        cache_dir=tmp_path_factory.mktemp("cache"),
        save_data=True,
        disp_messages=True,
        progress_bar=True,
    )


@pytest.fixture(scope="module")
def delays():
    return np.r_[-1:10:1] * u.ps


@pytest.fixture(scope="module")
def distances(structure_amorph):
    dists, _, _ = structure_amorph.get_distances_of_layers()
    return dists


@pytest.fixture(scope="module")
def temp_map(delays, distances):
    return 50 * np.ones([len(delays), len(distances), 1])


# helpers for the physics tests


def make_layer(id, **overrides):
    """a magnetic amorphous layer with sensible defaults, magnetized along x"""
    props = {
        "heat_capacity": 1,
        "therm_cond": 0,
        "lin_therm_exp": 0,
        "sub_system_coupling": 0,
        "eff_spin": 0.5,
        "curie_temp": 630 * u.K,
        "lamda": 0.005,
        "mag_moment": 0.393 * u.bohr_magneton,
        "aniso_exponent": 2,
        "anisotropy": [0, 0, 0] * u.J / u.m**3,
        "exch_stiffness": 0 * u.J / u.m,
        "mag_saturation": 5e5 * u.J / u.T / u.m**3,
    }
    props.update(overrides)
    atom = ud.Atom("Fe")
    atom.magnetization = (1, 0, 0)
    return ud.AmorphousLayer(
        id, id, thickness=1 * u.nm, density=7000 * u.kg / u.m**3, atom=atom, **props
    )


def make_structure(*layers):
    S = ud.Structure("test")
    for layer, n in layers:
        S.add_sub_structure(layer, n)
    return S


def simulation(cls, S, tmp_path):
    sim = cls(
        S,
        force_recalc=True,
        cache_dir=tmp_path,
        save_data=False,
        disp_messages=False,
        progress_bar=False,
    )
    sim.ode_options.update({"rtol": 1e-8, "atol": 1e-10})
    return sim


# tests


# Magnetization


def test_magnetization_repr(magnetization):
    assert repr(magnetization).startswith("Magnetization simulation properties:")
    assert str(magnetization) == repr(magnetization)
    assert magnetization._repr_html_().startswith("<h2>Magnetization simulation</h2>")


def test_magnetization_get_hash_distinguishes_simulations(structure_amorph, delays, temp_map):
    """the cache must not hand a result of one simulation type or
    temperature sub-system to another"""
    llb = LLB(structure_amorph, True, save_data=False, disp_messages=False)
    magnetization = Magnetization(structure_amorph, True, save_data=False, disp_messages=False)
    hash = llb.get_hash(delays=delays, temp_map=temp_map)
    assert hash != magnetization.get_hash(delays=delays, temp_map=temp_map)
    llb.temp_sub_system = 1
    assert hash != llb.get_hash(delays=delays, temp_map=temp_map)
    llg = LLG(structure_amorph, True, save_data=False, disp_messages=False)
    illg = iLLG(structure_amorph, True, save_data=False, disp_messages=False)
    hash = llg.get_hash(delays=delays, temp_map=temp_map)
    assert hash != illg.get_hash(delays=delays, temp_map=temp_map)
    llg.mag_sub_system = 1
    assert hash != llg.get_hash(delays=delays, temp_map=temp_map)


def test_magnetization_check_initial_magnetization(magnetization):
    # must fail as UnitCells are not yet supported, see #129
    with pytest.raises(TypeError):
        magnetization.check_initial_magnetization([])


# LLB


def test_llb_repr(llb):
    assert repr(llb).startswith(
        "Landau-Lifshitz-Bloch Magnetization Dynamics simulation properties:"
    )
    assert str(llb) == repr(llb)
    assert llb._repr_html_().startswith(
        "<h2>Landau-Lifshitz-Bloch Magnetization Dynamics simulation</h2>"
    )


def test_llb_check_initial_magnetization(llb, distances):
    llb.check_initial_magnetization(np.array([0, 0, 0]), distances=distances)
    llb.check_initial_magnetization(np.zeros([len(distances), 3]), distances=distances)
    with pytest.raises(ValueError):
        llb.check_initial_magnetization(np.zeros([3, len(distances)]), distances=distances)


def test_llb_get_magnetization(llb, temp_map, delays):
    llb.get_magnetization_map(delays, temp_map=temp_map).magnetization_map
    llb.force_recalc = False
    llb.get_magnetization_map(delays, temp_map=temp_map).magnetization_map
    llb.force_recalc = True


@pytest.mark.parametrize(
    "temp, below_tc",
    [(50, True), (99.9, True), (99.97, True), (100, False), (100.02, False), (150, False)],
)
def test_llb_mean_field_mag_map_at_curie_temp(llb, delays, distances, temp, below_tc):
    # all layers of the fixture have a Curie temperature of 100 K
    temp_map = temp * np.ones([len(delays), len(distances)])
    temp_map[0, :] = 50  # at least one temperature below T_C is required
    mf_mag_map = llb.get_mean_field_mag_map(temp_map).mean_field_magnetization_map
    assert np.all(mf_mag_map[0, :] > 0)
    if below_tc:
        assert np.all(mf_mag_map[1:, :] > 0)
    else:
        assert np.all(mf_mag_map[1:, :] == 0)


def test_llb_mean_field_mag_map_above_curie_temp(llb, delays, distances):
    # no temperature below T_C is required anymore
    mf_mag_map = llb.get_mean_field_mag_map(
        150 * np.ones([len(delays), len(distances)])
    ).mean_field_magnetization_map
    assert np.all(mf_mag_map == 0)


def test_llb_get_magnetization_just_below_curie_temp(llb, delays, distances):
    llb = copy.deepcopy(llb)  # do not alter the module-scoped fixture
    llb.force_recalc = True
    llb.save_data = False
    temp_map = 99.97 * np.ones([len(delays), len(distances), 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        magnetization_map = llb.get_magnetization_map(delays, temp_map=temp_map).magnetization_map
    assert np.all(np.isfinite(magnetization_map))


def test_llb_input_validation(llb, temp_map, delays, distances):
    with pytest.raises(TypeError):
        llb.get_magnetization_map(delays, temp_map=list(temp_map)).magnetization_map
    with pytest.raises(ValueError):
        llb.get_magnetization_map(delays, temp_map=temp_map, B_ext=np.zeros(2)).magnetization_map
    with pytest.raises(ValueError):
        llb.get_magnetization_map(
            delays, temp_map=temp_map, init_magnetization=np.zeros((2, 3))
        ).magnetization_map
    with pytest.raises(ValueError):
        llb.get_magnetization_map(delays, temp_map=temp_map[:, 0:3, :]).magnetization_map
    # an Nx3 initial magnetization and a per-delay external field are valid
    init_mag = np.tile(ALONG_X, (len(distances), 1))
    B_ext = np.tile(np.r_[0, 0, 1.0], (len(delays), 1))
    llb.get_magnetization_map(
        delays, temp_map=temp_map, init_magnetization=init_mag, B_ext=B_ext
    ).magnetization_map


def test_llb_mean_field_mag_map(llb, temp_map):
    """the mean-field magnetization is 1 at T = 0, 0 above T_C and solves the
    self-consistent equation in between"""
    mf = llb.calc_mean_field_mag_map(temp_map[:, :, 0])
    assert mf.shape == temp_map.shape[0:2]
    assert np.all(mf > 0)
    assert np.allclose(mf, mf[0, 0])
    layer = llb.S.get_unique_layers()[1][0]
    Tc = layer.curie_temp.magnitude
    temps = np.r_[0, 0.5 * Tc, 0.999 * Tc, Tc, 2 * Tc]
    mf = llb.calc_mean_field_mag_map(np.tile(temps[:, np.newaxis], (1, mf.shape[1])))
    assert np.allclose(mf[0], 1)
    assert np.all(mf[1] > mf[2])
    assert mf[2, 0] > 0
    assert np.allclose(mf[3:], 0)
    # the quantum equilibrium curve of the layer's effective spin
    reference = mean_field.mean_field_magnetization(temps / Tc, layer.eff_spin.magnitude)
    assert np.allclose(mf[:, 0], reference, atol=2e-3)


def test_interlayer_exchange_field_components(tmp_path):
    """the exchange field of three layers is component-wise the scaled
    difference of the neighbouring magnetizations"""
    A = 1e-11
    S = make_structure((make_layer("a", exch_stiffness=A * u.J / u.m), 3))
    llb = simulation(LLB, S, tmp_path)
    is_magnetic = np.ones(3, dtype=bool)
    mag = magnetization_scipy.stack_magnetic_parameters(llb.calc_magnetic_parameters(is_magnetic))
    thicknesses = S.get_layer_property_vector("structural.thickness.magnitude")
    m = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])[:, np.newaxis, :]
    B = magnetization_scipy.interlayer_exchange_field(
        m, mag, llb.calc_interlayer_exchange_stiffnesses(is_magnetic), thicknesses
    )
    es = 2 / (mag.mag_saturation[:, 0] * thicknesses**2)
    assert np.allclose(B[0, 0], es[0] * A * (m[1, 0] - m[0, 0]))
    assert np.allclose(B[1, 0], es[1] * A * ((m[0, 0] - m[1, 0]) + (m[2, 0] - m[1, 0])))
    assert np.allclose(B[2, 0], es[2] * A * (m[1, 0] - m[2, 0]))


def test_no_exchange_across_spacer(tmp_path):
    """magnetic layers separated by a non-magnetic spacer are not coupled"""
    A = 1e-11 * u.J / u.m
    spacer = make_layer("s", mag_saturation=0 * u.J / u.T / u.m**3, curie_temp=0 * u.K)
    S = make_structure(
        (make_layer("a", exch_stiffness=A), 1),
        (spacer, 1),
        (make_layer("b", exch_stiffness=A), 1),
    )
    llb = simulation(LLB, S, tmp_path)
    A_mag = llb.calc_interlayer_exchange_stiffnesses(np.r_[True, False, True])
    assert np.allclose(A_mag, 0)
    llb.S = make_structure((make_layer("a", exch_stiffness=A), 2))
    A_mag = llb.calc_interlayer_exchange_stiffnesses(np.r_[True, True])
    assert np.allclose(A_mag, [[0, 1e-11], [1e-11, 0]])


def test_gyromagnetic_ratio_default():
    """the gyromagnetic ratio defaults to the one of a free electron"""
    layer = make_layer("m")
    assert np.isclose(layer.gyromagnetic_ratio.to("rad/(s*T)").magnitude, -1.76085963e11)
    mag = magnetization_scipy.stack_magnetic_parameters([layer.magnetic])
    assert np.allclose(mag.gyromagnetic_ratio, magnetization_scipy.GAMMA_E)


@pytest.mark.parametrize("model", [LLG, LLB])
def test_gyromagnetic_ratio_precession(model, tmp_path):
    """without damping the magnetization precesses at the Larmor frequency
    gamma B of the gyromagnetic ratio of the layer"""
    gamma = -2 * np.pi * 30.8e9  # rad/(s T)
    S = make_structure(
        (make_layer("m", lamda=0.0, gilbert_damping=0.0, gyromagnetic_ratio=gamma), 1)
    )
    sim = simulation(model, S, tmp_path)
    delays = np.r_[0:100:0.5] * u.ps
    B = 0.5
    temp_map = 300 * np.ones((len(delays), 1, 1))
    m = sim.get_magnetization_map(
        delays, temp_map=temp_map, B_ext=np.r_[0, 0, B]
    ).magnetization_map[:, 0, :]
    t = delays.to("s").magnitude
    omega = gamma * B
    norm = np.linalg.norm(m[0])
    assert np.allclose(m[:, 0], norm * np.cos(omega * t), atol=1e-5)
    assert np.allclose(m[:, 1], -norm * np.sin(omega * t), atol=1e-5)
    assert np.allclose(np.linalg.norm(m, axis=-1), norm)


@pytest.mark.parametrize("eff_spin", [0.5, 3.5, np.inf])
def test_llb_relaxes_to_mean_field(eff_spin, tmp_path):
    """the LLB norm relaxes to the mean-field magnetization of the
    temperature, the quantum one for a finite spin and the classical Langevin
    one for an infinite spin"""
    S = make_structure((make_layer("m", lamda=0.1, eff_spin=eff_spin), 1))
    llb = simulation(LLB, S, tmp_path)
    delays = np.r_[0:20:0.2] * u.ps
    temp_map = 400 * np.ones((len(delays), 1, 1))
    mag_map = llb.get_magnetization_map(
        delays, temp_map=temp_map, init_magnetization=ALONG_X, B_ext=np.r_[0, 0, 0.01]
    ).magnetization_map
    mf = llb.calc_mean_field_mag_map(temp_map[:, :, 0])
    reference = mean_field.mean_field_magnetization(400 / 630, eff_spin)
    assert np.isclose(mf[-1, 0], reference, atol=1e-3)
    assert np.isclose(np.linalg.norm(mag_map[-1, 0]), mf[-1, 0], rtol=1e-2)


def test_llb_starts_in_equilibrium(tmp_path):
    """without an explicit initial magnetization the layers start in the
    equilibrium of their initial temperature and stay there"""
    S = make_structure((make_layer("m", lamda=0.1), 3))
    llb = simulation(LLB, S, tmp_path)
    delays = np.r_[0:10:0.5] * u.ps
    temp_map = 300 * np.ones((len(delays), 3, 1))
    m = llb.get_magnetization_map(delays, temp_map=temp_map).magnetization_map
    mf = llb.calc_mean_field_mag_map(temp_map[:, :, 0])
    assert np.allclose(np.linalg.norm(m, axis=-1), mf, atol=1e-6)
    # the direction is the one of the layers
    assert np.allclose(m[:, :, 0], mf, atol=1e-6)


def test_llb_survives_curie_temperature(tmp_path):
    """the LLB stays finite when a layer is heated above and cooled below its
    Curie temperature, demagnetizes at the peak and recovers"""
    S = make_structure((make_layer("m", lamda=0.1), 2))
    delays = np.r_[0:3:0.02] * u.ps
    t = delays.to("s").magnitude
    tc = 630
    temps = (
        300
        + 1.5 * tc * np.exp(-(((t - 0.5e-12) / 0.3e-12) ** 2)) * (t > 0.3e-12)
        + 0.5 * tc * np.exp(-(t - 0.8e-12) / 1e-12) * (t > 0.8e-12)
    )
    temp_map = temps[:, np.newaxis] * np.ones((1, 2))
    llb = simulation(LLB, S, tmp_path)
    mag_map = llb.get_magnetization_map(delays, temp_map=temp_map, polar=True).magnetization_map
    assert np.all(np.isfinite(mag_map))
    assert mag_map[np.argmax(temps), 0, 0] < 0.2
    assert mag_map[-1, 0, 0] > 0.3


def test_llb_warns_on_demagnetization_collapse(tmp_path):
    """a magnetization that starts at zero below T_C stays at the fixed point
    m = 0, which is reported"""
    S = make_structure((make_layer("m", lamda=0.1), 1))
    llb = simulation(LLB, S, tmp_path)
    delays = np.r_[0:1:0.1] * u.ps
    temp_map = 300 * np.ones((len(delays), 1, 1))
    with pytest.warns(UserWarning, match="m = 0 is an exact fixed point"):
        llb.get_magnetization_map(
            delays, temp_map=temp_map, init_magnetization=np.zeros(3)
        ).magnetization_map


# LLG and iLLG


def test_magnetoelastic_field():
    """the magnetoelastic field is an anisotropy along z scaled by the strain"""
    mag = SimpleNamespace(
        magnetoelastic_coupling=np.array([[2e6]]),
        mag_saturation=np.array([[5e5]]),
        aniso_exponent=np.array([[3.0]]),
    )
    m = np.array([[[0.3, 0.4, 0.5]]])
    B = magnetization_scipy.magnetoelastic_field(m, np.array([[0.8]]), np.array([1e-3]), mag)
    assert np.allclose(B[..., :2], 0)
    assert np.isclose(B[0, 0, 2], -2 * 2e6 * 1e-3 / 5e5 * 0.8 * 0.5)


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_llg_damped_precession(alpha, tmp_path):
    """the LLG precesses around the external field at the Larmor frequency
    reduced by 1/(1+alpha^2) and relaxes towards it with
    tan(phi/2) = tan(phi_0/2) exp(-alpha |gamma| B t / (1+alpha^2))"""
    S = make_structure((make_layer("m", gilbert_damping=alpha), 1))
    llg = simulation(LLG, S, tmp_path)
    delays = np.r_[0:100:0.5] * u.ps
    B = 0.5
    mag_map = llg.get_magnetization_map(delays, B_ext=np.r_[0, 0, B], polar=True).magnetization_map
    t = delays.to("s").magnitude
    omega = magnetization_scipy.GAMMA_E * B / (1 + alpha**2)
    phi = 2 * np.arctan(np.exp(-alpha * abs(omega) * t))
    gamma = np.mod(-omega * t + np.pi, 2 * np.pi) - np.pi
    assert np.allclose(mag_map[:, 0, 0], 1)
    assert np.allclose(mag_map[:, 0, 1], phi, atol=1e-5)
    assert np.allclose(np.exp(1j * mag_map[:, 0, 2]), np.exp(1j * gamma), atol=1e-4)


def test_llg_magnetization_amplitude(tmp_path):
    """the amplitude is given as mag_amplitude_map, taken from the
    mag_sub_system or the mean field of the temp_map, or kept at the initial
    amplitude"""
    S = make_structure((make_layer("m", gilbert_damping=0.1), 2))
    llg = simulation(LLG, S, tmp_path)
    delays = np.r_[0:5:0.5] * u.ps
    M = len(delays)
    amplitude_map = np.linspace(0.2, 0.9, M)[:, np.newaxis] * np.ones((1, 2))
    mag_map = llg.get_magnetization_map(
        delays, mag_amplitude_map=amplitude_map, polar=True
    ).magnetization_map
    assert np.allclose(mag_map[:, :, 0], amplitude_map)

    temp_map = np.stack((np.full((M, 2), 400.0), amplitude_map), axis=2)
    mag_map = llg.get_magnetization_map(delays, temp_map=temp_map, polar=True).magnetization_map
    assert np.allclose(
        mag_map[:, :, 0],
        llg.get_mean_field_mag_map(temp_map[:, :, 0]).mean_field_magnetization_map,
    )
    llg.mag_sub_system = 1
    mag_map = llg.get_magnetization_map(delays, temp_map=temp_map, polar=True).magnetization_map
    assert np.allclose(mag_map[:, :, 0], amplitude_map)

    # a negative amplitude of the polar form reverses the direction
    init_mag = {"r": [0.7, -0.7], "phi": 90 * u.deg, "gamma": 0}
    m = llg.get_magnetization_map(delays, init_magnetization=init_mag).magnetization_map
    assert np.allclose(m[:, 0], [0.7, 0, 0])
    assert np.allclose(m[:, 1], [-0.7, 0, 0])
    # but a negative prescribed amplitude is not allowed
    with pytest.raises(ValueError, match="must not be negative"):
        llg.get_magnetization_map(delays, mag_amplitude_map=-amplitude_map).magnetization_map


def test_llg_non_magnetic_layers(tmp_path):
    """layers without a saturation magnetization are not magnetic"""
    spacer = make_layer("s", mag_saturation=0 * u.J / u.T / u.m**3)
    S = make_structure((make_layer("m", gilbert_damping=0.1), 1), (spacer, 1))
    llg = simulation(LLG, S, tmp_path)
    mag_map = llg.get_magnetization_map(
        np.r_[0:5:1] * u.ps, B_ext=np.r_[0, 0, 1], polar=True
    ).magnetization_map
    assert np.allclose(mag_map[:, 1], 0)
    assert np.all(mag_map[1:, 0, 1] < np.pi / 2)


@pytest.mark.parametrize("cls", [LLG, LLB])
def test_strain_acts_as_anisotropy(cls, tmp_path):
    """the magnetoelastic field of a strain is the anisotropy K_z = -B_1 strain"""
    K, strain = 2e5, 1e-3
    delays = np.r_[0:50:1] * u.ps
    temp_map = np.full((len(delays), 1), 300.0)
    init_mag = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    kwargs = {"temp_map": temp_map} if cls is LLB else {}
    S = make_structure(
        (make_layer("a", gilbert_damping=0.1, anisotropy=[0, 0, K] * u.J / u.m**3), 1)
    )
    reference = (
        simulation(cls, S, tmp_path)
        .get_magnetization_map(delays, init_magnetization=init_mag, **kwargs)
        .magnetization_map
    )
    S = make_structure(
        (
            make_layer(
                "b", gilbert_damping=0.1, magnetoelastic_coupling=-K / strain * u.J / u.m**3
            ),
            1,
        )
    )
    mag_map = (
        simulation(cls, S, tmp_path)
        .get_magnetization_map(
            delays,
            init_magnetization=init_mag,
            strain_map=np.full((len(delays), 1), strain),
            **kwargs,
        )
        .magnetization_map
    )
    assert not np.allclose(reference[-1], reference[0], atol=1e-3)
    assert np.allclose(mag_map, reference, atol=1e-6)


def test_illg_requires_damping_and_inertia(tmp_path):
    S = make_structure((make_layer("m", gilbert_damping=0.1), 1))
    illg = simulation(iLLG, S, tmp_path)
    with pytest.raises(ValueError, match="inertial_time"):
        illg.get_magnetization_map(np.r_[0:2:1] * u.ps).magnetization_map


def test_illg_approaches_llg(tmp_path):
    """for a short inertial time the iLLG follows the LLG, starting with the
    velocity of the LLG"""
    S = make_structure((make_layer("m", gilbert_damping=0.2, inertial_time=1 * u.fs), 1))
    delays = np.r_[0:20:0.5] * u.ps
    B_ext = np.r_[0, 0, 1]
    llg = simulation(LLG, S, tmp_path)
    illg = simulation(iLLG, S, tmp_path)
    assert illg.ode_options["method"] == "LSODA"
    reference = llg.get_magnetization_map(delays, B_ext=B_ext).magnetization_map
    m = illg.get_magnetization_map(delays, B_ext=B_ext).magnetization_map
    assert np.allclose(np.sum(m**2, axis=2), 1)
    assert np.allclose(m, reference, atol=1e-2)


def test_illg_initial_velocity(tmp_path):
    """the default initial velocity is the one of the LLG, a given one is
    used as is"""
    S = make_structure((make_layer("m", gilbert_damping=0.2, inertial_time=10 * u.fs), 2))
    illg = simulation(iLLG, S, tmp_path)
    is_magnetic = np.ones(2, dtype=bool)
    delays = np.r_[0:1:0.5] * 1e-12
    inputs = magnetization_scipy.prepare_inputs(delays, 2, np.r_[0, 0, 1])
    n = np.tile([1.0, 0, 0], (2, 1, 1))
    state = illg.calc_initial_state(n, inputs, is_magnetic)
    assert state.shape == (2, 2, 1, 3)
    assert np.allclose(state[0], n)
    omega = magnetization_scipy.GAMMA_E / 1.04
    # precession along y and damping towards the field along z
    assert np.allclose(state[1, :, 0], [0, -omega, -0.2 * omega])
    state = illg.calc_initial_state(n, inputs, is_magnetic, init_velocity=np.r_[0, 1e9, 0])
    assert np.allclose(state[1, :, 0], [0, 1e9, 0])
    with pytest.raises(ValueError, match="init_velocity"):
        illg.calc_initial_state(n, inputs, is_magnetic, init_velocity=np.zeros((3, 3)))


def test_illg_free_nutation(tmp_path):
    """without a field an initial velocity v_0 decays on the inertial time tau
    while it rotates at 1/(alpha tau), which displaces the unit vector by
    v_0 tau alpha / (alpha + i) in the plane perpendicular to it"""
    alpha, tau, v0 = 0.5, 10e-15, 1e11
    S = make_structure((make_layer("m", gilbert_damping=alpha, inertial_time=tau * u.s), 1))
    illg = simulation(iLLG, S, tmp_path)
    delays = np.r_[0:1:0.01] * u.ps
    m = illg.get_magnetization_map(delays, init_velocity=np.r_[0, v0, 0]).magnetization_map[-1, 0]
    displacement = v0 * tau * alpha / (alpha + 1j)
    assert np.isclose(m[1] + 1j * m[2], displacement, rtol=1e-3)


@pytest.mark.parametrize("cls", [LLB, LLG, iLLG])
def test_no_magnetic_layers(cls, tmp_path):
    spacer = make_layer("s", mag_saturation=0 * u.J / u.T / u.m**3, curie_temp=0 * u.K)
    sim = simulation(cls, make_structure((spacer, 2)), tmp_path)
    delays = np.r_[0:5:1] * u.ps
    mag_map = sim.get_magnetization_map(
        delays, temp_map=np.full((len(delays), 2), 300.0)
    ).magnetization_map
    assert np.allclose(mag_map, 0)


@pytest.mark.parametrize("cls", [LLB, LLG, iLLG])
def test_single_delay(cls, tmp_path):
    """a single delay returns the initial state"""
    S = make_structure((make_layer("m", gilbert_damping=0.1, inertial_time=1 * u.fs), 1))
    sim = simulation(cls, S, tmp_path)
    mag_map = sim.get_magnetization_map(
        np.r_[0] * u.ps, temp_map=np.full((1, 1), 300.0)
    ).magnetization_map
    assert mag_map.shape == (1, 1, 3)
    # the magnetization of the layer along x
    assert np.allclose(mag_map[0, 0, 1:], [0, 0])
    assert mag_map[0, 0, 0] > 0


def test_llb_without_saturation_magnetization(tmp_path):
    """the fields scaled by 1/M_s vanish for M_s = 0 instead of diverging"""
    S = make_structure((make_layer("m", mag_saturation=0 * u.J / u.T / u.m**3), 1))
    llb = simulation(LLB, S, tmp_path)
    delays = np.r_[0:5:1] * u.ps
    mag_map = llb.get_magnetization_map(
        delays, temp_map=np.full((len(delays), 1), 300.0)
    ).magnetization_map
    assert np.all(np.isfinite(mag_map))


def test_llb_requires_eff_spin(tmp_path):
    llb = simulation(LLB, make_structure((make_layer("m", eff_spin=0), 1)), tmp_path)
    with pytest.raises(ValueError, match="eff_spin"):
        llb.get_magnetization_map(
            np.r_[0:2:1] * u.ps, temp_map=np.full((2, 1), 300.0)
        ).magnetization_map


def test_llg_rejects_unknown_keyword(tmp_path):
    llg = simulation(LLG, make_structure((make_layer("m"), 1)), tmp_path)
    with pytest.raises(TypeError, match="init_velocty"):
        llg.get_magnetization_map(
            np.r_[0:2:1] * u.ps, init_velocty=np.r_[0, 1, 0]
        ).magnetization_map


@pytest.mark.parametrize("eff_spin", [0.5, 3.5])
def test_llb_pumped_at_low_temperature(eff_spin, tmp_path):
    """at low temperatures the tiny susceptibility makes the longitudinal field
    huge, which must not spoil the torques, and the solver must not step over
    the excitation of the magnetization in equilibrium with the default
    tolerances"""
    S = make_structure((make_layer("m", eff_spin=eff_spin), 1))
    llb = LLB(S, True, save_data=False, disp_messages=False)
    delays = np.r_[0:50:0.1] * u.ps
    temp_map = np.full((len(delays), 1), 5.0)
    temp_map[20:, 0] += 400 * np.exp(-np.arange(len(delays) - 20) * 0.1 / 5)
    mag_map = llb.get_magnetization_map(delays, temp_map=temp_map, polar=True).magnetization_map
    assert np.all(np.isfinite(mag_map))
    assert mag_map[:, 0, 0].min() < 0.97 * mag_map[0, 0, 0]
    assert np.all(mag_map[:, 0, 0] <= mag_map[0, 0, 0] + 1e-9)
    # the direction is untouched without any field
    assert np.allclose(mag_map[:, 0, 1:], [np.pi / 2, 0])


@pytest.mark.parametrize(
    "props",
    [
        {"anisotropy": [0, 0, 1e5] * u.J / u.m**3},
        {"magnetoelastic_coupling": 1e6 * u.J / u.m**3},
    ],
)
def test_llb_requires_aniso_exponent_of_two(props, tmp_path):
    """the anisotropy and magnetoelastic fields diverge at T_C for beta < 2"""
    llb = simulation(
        LLB, make_structure((make_layer("m", aniso_exponent=1, **props), 1)), tmp_path
    )
    with pytest.raises(ValueError, match="aniso_exponent"):
        llb.get_magnetization_map(
            np.r_[0:2:1] * u.ps, temp_map=np.full((2, 1), 300.0)
        ).magnetization_map


# sublattices

K_B = 1.380649e-23


def mf_exchange(eff_spin, curie_temp):
    """mean-field exchange constant of a single sublattice"""
    factor = 3.0 if np.isinf(eff_spin) else 3 * eff_spin / (eff_spin + 1)
    return factor * K_B * curie_temp


def make_sublattice_layer(id, magnetizations, mf_exch_coupling, **overrides):
    """a magnetic amorphous layer with several sublattices"""
    layer = make_layer(id, mf_exch_coupling=mf_exch_coupling * u.J, **overrides)
    layer.magnetization = magnetizations
    return layer


def sublattice_llb(S, tmp_path, temp_map, delays, **kwargs):
    llb = simulation(LLB, S, tmp_path)
    return (
        llb.get_magnetization_map(delays, temp_map=temp_map, **kwargs).magnetization_map,
        llb.get_sublattice_magnetization_map(
            delays, temp_map=temp_map, **kwargs
        ).sublattice_magnetization_map,
    )


def pulse(delays, T0=300.0, dT=200.0, L=1):
    """a temperature pulse, which stays below the Curie temperature"""
    t = delays.to("ps").magnitude
    temps = T0 + dT * np.exp(-np.clip(t - 1, 0, None) / 3) * (t >= 1)
    return np.repeat(temps[:, np.newaxis], L, axis=1)


def test_uncoupled_sublattices_are_independent_layers(tmp_path):
    """sublattices without coupling evolve as separate layers"""
    J_a, J_b = mf_exchange(0.5, 630), mf_exchange(3.5, 1000)
    delays = np.r_[0:10:0.1] * u.ps
    kwargs = {"lamda": [0.01, 0.02], "eff_spin": [0.5, 3.5], "gilbert_damping": [0.05, 0.1]}
    layer = make_sublattice_layer("ab", [(1, 0, 0), (0, 1, 0)], np.diag([J_a, J_b]), **kwargs)
    reference = make_structure(
        (make_layer("a", lamda=0.01, eff_spin=0.5, curie_temp=630 * u.K, gilbert_damping=0.05), 1),
        (make_layer("b", lamda=0.02, eff_spin=3.5, curie_temp=1000 * u.K, gilbert_damping=0.1), 1),
    )
    reference.get_layer_vectors()[2][1].magnetization = (0, 1, 0)
    B_ext = np.r_[0, 0, 0.5]
    temp_map = pulse(delays)
    # LLB
    _, sub = sublattice_llb(make_structure((layer, 1)), tmp_path, temp_map, delays, B_ext=B_ext)
    ref_llb = simulation(LLB, reference, tmp_path)
    ref = ref_llb.get_magnetization_map(
        delays, temp_map=np.repeat(temp_map, 2, axis=1), B_ext=B_ext
    ).magnetization_map
    assert sub.shape == (len(delays), 1, 2, 3)
    assert np.allclose(sub[:, 0], ref)
    # LLG
    llg = simulation(LLG, make_structure((layer, 1)), tmp_path)
    sub = llg.get_sublattice_magnetization_map(delays, B_ext=B_ext).sublattice_magnetization_map
    ref = (
        simulation(LLG, reference, tmp_path)
        .get_magnetization_map(delays, B_ext=B_ext)
        .magnetization_map
    )
    assert np.allclose(sub[:, 0], ref)


@pytest.mark.parametrize("sign", [1, -1])
def test_identical_coupled_sublattices_are_one_sublattice(sign, tmp_path):
    """two identical sublattices with the exchange J_1 within and +-J_2 between
    them evolve as a single sublattice with J_1 + J_2 below the ordering
    temperature, parallel for a ferromagnetic and antiparallel for an
    antiferromagnetic coupling"""
    J_1, J_2 = mf_exchange(0.5, 400), mf_exchange(0.5, 230)
    delays = np.r_[0:10:0.1] * u.ps
    temp_map = pulse(delays)
    layer = make_sublattice_layer(
        "ab", [(1, 0, 0), (sign, 0, 0)], np.array([[J_1, sign * J_2], [sign * J_2, J_1]])
    )
    assert np.isclose(layer.curie_temp.magnitude, 630)
    S = make_structure((layer, 1))
    net, sub = sublattice_llb(S, tmp_path, temp_map, delays)
    ref = (
        simulation(LLB, make_structure((make_layer("r"), 1)), tmp_path)
        .get_magnetization_map(delays, temp_map=temp_map)
        .magnetization_map
    )
    m = sub[:, 0]
    assert np.allclose(m[:, 0], ref[:, 0], atol=1e-6)
    assert np.allclose(m[:, 1], sign * m[:, 0], atol=1e-6)
    # the net magnetization of an antiferromagnet vanishes
    assert np.allclose(net[:, 0], ref[:, 0] if sign > 0 else 0, atol=1e-6)
    llb = simulation(LLB, S, tmp_path)
    mf_sub = llb.get_sublattice_mean_field_mag_map(
        temp_map
    ).sublattice_mean_field_magnetization_map
    assert np.allclose(mf_sub[:, 0, 0], mf_sub[:, 0, 1])
    assert np.allclose(
        llb.get_mean_field_mag_map(temp_map).mean_field_magnetization_map[:, 0],
        mf_sub[:, 0, 0] * (sign > 0),
    )


def test_ferrimagnet(tmp_path):
    """the net magnetization of a ferrimagnet is the sum of the sublattice
    magnetizations weighted by their saturation magnetizations, which relax
    to the coupled mean field"""
    J = np.array([[mf_exchange(3.5, 300), -2e-22], [-2e-22, mf_exchange(1, 800)]])
    Ms = [6e5, 1.2e6]
    layer = make_sublattice_layer(
        "GdFe",
        [(0, 0, -1), (0, 0, 1)],
        J,
        eff_spin=[3.5, 1],
        mag_saturation=Ms * u.J / u.T / u.m**3,
        mag_moment=[7.6, 2.2] * u.bohr_magneton,
        lamda=[0.05, 0.05],
    )
    delays = np.r_[0:50:0.5] * u.ps
    temp_map = np.full((len(delays), 1), 250.0)
    net, sub = sublattice_llb(make_structure((layer, 1)), tmp_path, temp_map, delays)
    m = sub[:, 0]
    expected = (Ms[0] * m[:, 0] + Ms[1] * m[:, 1]) / sum(Ms)
    assert np.allclose(net[:, 0], expected)
    me = np.abs(mean_field.mean_field_magnetization_coupled(250.0, J, [3.5, 1]))[0]
    assert np.allclose(np.linalg.norm(sub[-1, 0], axis=-1), me, atol=1e-6)
    assert m[-1, 0, 2] < 0 < m[-1, 1, 2]


def test_mixed_numbers_of_sublattices(tmp_path):
    """layers with fewer sublattices are padded, and only existing
    sublattices are coupled to the neighbouring layers"""
    J = mf_exchange(0.5, 630)
    A = 1e-12 * u.J / u.m
    two = make_sublattice_layer("ab", [(1, 0, 0), (0, 1, 0)], np.diag([J, J]), exch_stiffness=A)
    one = make_layer("c", exch_stiffness=A)
    S = make_structure((two, 1), (one, 1))
    llb = simulation(LLB, S, tmp_path)
    is_magnetic = np.ones(2, dtype=bool)
    mag = magnetization_scipy.stack_magnetic_parameters(llb.calc_magnetic_parameters(is_magnetic))
    assert np.array_equal(mag.active, [[1, 1], [1, 0]])
    m = np.array([[[1.0, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 0]]])
    thicknesses = S.get_layer_property_vector("structural.thickness.magnitude")
    B = magnetization_scipy.interlayer_exchange_field(
        m, mag, llb.calc_interlayer_exchange_stiffnesses(is_magnetic), thicknesses
    )
    assert np.allclose(B[0, 1], 0)
    assert np.allclose(B[1, 1], 0)
    assert not np.allclose(B[0, 0], 0)
    delays = np.r_[0:5:0.5] * u.ps
    sub = llb.get_sublattice_magnetization_map(
        delays, temp_map=np.full((len(delays), 2), 300.0)
    ).sublattice_magnetization_map
    assert sub.shape == (len(delays), 2, 2, 3)
    assert np.allclose(sub[:, 1, 1], 0)


def test_sublattice_input_validation(tmp_path):
    J = mf_exchange(0.5, 630)
    layer = make_sublattice_layer("ab", [(1, 0, 0), (-1, 0, 0)], np.diag([J, J]))
    llb = simulation(LLB, make_structure((layer, 3)), tmp_path)
    delays = np.r_[0:2:1] * u.ps
    temp_map = np.full((2, 3), 300.0)
    # a vector per layer is not accepted for several sublattices
    for init_mag in (ALONG_X, np.tile(ALONG_X, (3, 1))):
        with pytest.raises(ValueError, match="sublattices"):
            llb.get_magnetization_map(
                delays, temp_map=temp_map, init_magnetization=init_mag
            ).magnetization_map
    init_mag = np.array([ALONG_X, -ALONG_X])
    sub = llb.get_sublattice_magnetization_map(
        delays, temp_map=temp_map, init_magnetization=init_mag
    ).sublattice_magnetization_map
    assert np.allclose(sub[0, :, 1], -sub[0, :, 0])
    full = llb.get_sublattice_magnetization_map(
        delays, temp_map=temp_map, init_magnetization=np.tile(init_mag, (3, 1, 1))
    ).sublattice_magnetization_map
    assert np.allclose(full, sub)
    # inconsistent numbers of sublattices
    layer.magnetic.lamda = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="2 magnetizations, but 3 sublattices"):
        llb.get_magnetization_map(delays, temp_map=temp_map).magnetization_map


def test_llg_sublattice_amplitudes(tmp_path):
    """the amplitudes of the LLG are given per sublattice or taken from the
    coupled mean field"""
    J = np.array([[mf_exchange(0.5, 500), -1e-21], [-1e-21, mf_exchange(0.5, 700)]])
    layer = make_sublattice_layer("ab", [(1, 0, 0), (-1, 0, 0)], J, gilbert_damping=0.1)
    llg = simulation(LLG, make_structure((layer, 2)), tmp_path)
    delays = np.r_[0:5:0.5] * u.ps
    M = len(delays)
    amplitudes = np.stack([np.full((M, 2), 0.3), np.full((M, 2), 0.8)], axis=2)
    sub = llg.get_sublattice_magnetization_map(
        delays, mag_amplitude_map=amplitudes, polar=True
    ).sublattice_magnetization_map
    assert np.allclose(sub[..., 0], amplitudes)
    temp_map = np.full((M, 2), 400.0)
    sub = llg.get_sublattice_magnetization_map(
        delays, temp_map=temp_map, polar=True
    ).sublattice_magnetization_map
    assert np.allclose(
        sub[..., 0],
        llg.get_sublattice_mean_field_mag_map(temp_map).sublattice_mean_field_magnetization_map,
    )
    with pytest.raises(ValueError, match="M x L x P"):
        llg.get_magnetization_map(delays, mag_amplitude_map=np.ones((M, 2, 3))).magnetization_map


# equilibrium magnetization


def stoner_wohlfarth_angle(B, theta_H, B_K):
    """equilibrium angle from the easy axis z of a uniaxial anisotropy with
    the anisotropy field B_K in a field B at the angle theta_H from z"""
    from scipy.optimize import minimize_scalar

    def energy(theta):
        return -B_K / 2 * np.cos(theta) ** 2 - B * np.cos(theta - theta_H)

    return minimize_scalar(
        energy, bounds=(0, theta_H), method="bounded", options={"xatol": 1e-12}
    ).x


def field(B, theta_H):
    return B * np.array([np.sin(theta_H), 0, np.cos(theta_H)])


def test_parse_initial_magnetization():
    """the initial magnetization is cartesian by default and can be given in
    polar form by a dict or a VectorParameter, as the magnetization of the
    layers"""
    parse = Magnetization.parse_initial_magnetization
    expected = np.array([2.0, np.pi / 2, np.pi / 2])  # along y
    forms = [
        (0, 2, 0),
        np.array([0, 2, 0]) * u.dimensionless,
        {"x": 0, "y": 2, "z": 0},
        {"r": 2, "phi": 90, "gamma": 90},
        {"r": 2, "phi": np.pi / 2 * u.rad, "gamma": 90 * u.deg},
        ud.VectorParameter("", (0, 2, 0)),
    ]
    for form in forms:
        assert np.allclose(parse(form), expected)
    # a negative amplitude is kept in the polar form only
    assert np.allclose(parse({"r": -2, "phi": 90, "gamma": 90}), [-2, np.pi / 2, np.pi / 2])
    # arrays of vectors and lists of VectorParameters
    cartesian = np.array([[[1, 0, 0], [0, 0, -1]]] * 3)
    polar = parse(cartesian)
    assert polar.shape == (3, 2, 3)
    assert np.allclose(convert_polar_to_cartesian(polar), cartesian)
    dict_form = {"r": polar[..., 0], "phi": polar[..., 1] * u.rad, "gamma": polar[..., 2] * u.rad}
    assert np.allclose(parse(dict_form), polar)
    vectors = [ud.VectorParameter("", (1, 0, 0)), ud.VectorParameter("", (0, 0, -1))]
    assert np.allclose(parse(vectors), polar[0])
    assert np.allclose(parse([vectors] * 3), polar)
    # empty inputs use the magnetization of the layers
    assert len(parse(None)) == 0 and len(parse([])) == 0
    with pytest.raises(ValueError, match="cartesian components"):
        parse([1, 0])
    with pytest.raises(ValueError, match="dict"):
        parse({"r": 1, "phi": 0})


def test_init_magnetization_forms(tmp_path):
    """all forms of the initial magnetization give the same result and cache,
    and the former polar init_mag raises an error"""
    S = make_structure((make_layer("m", gilbert_damping=0.1), 2))
    llg = simulation(LLG, S, tmp_path)
    delays = np.r_[0:20:0.5] * u.ps
    B_ext = np.r_[0, 0, 0.5]
    reference = llg.get_magnetization_map(
        delays, init_magnetization=ALONG_X, B_ext=B_ext
    ).magnetization_map
    for form in (
        {"r": 1, "phi": 90 * u.deg, "gamma": 0 * u.deg},
        ud.VectorParameter("", tuple(ALONG_X)),
    ):
        mag_map = llg.get_magnetization_map(
            delays, init_magnetization=form, B_ext=B_ext
        ).magnetization_map
        assert np.allclose(mag_map, reference)

    def hash_of(value):
        return llg.get_hash(
            delays=delays, init_magnetization=llg.parse_initial_magnetization(value)
        )

    assert hash_of(ALONG_X) == hash_of({"r": 1, "phi": 90, "gamma": 0})
    with pytest.raises(TypeError, match="init_magnetization"):
        llg.get_magnetization_map(delays, init_mag=np.array([1, np.pi / 2, 0])).magnetization_map
    with pytest.raises(TypeError, match="init_magnetization"):
        llg.calc_equilibrium_magnetization(init_mag=np.array([1, np.pi / 2, 0]))


def test_b_ext_quantity(tmp_path):
    """B_ext can be given as quantity in any unit of the magnetic field, which
    gives the same result and cache as in tesla"""
    S = make_structure((make_layer("m", gilbert_damping=0.1), 2))
    llg = simulation(LLG, S, tmp_path)
    delays = np.r_[0:20:0.5] * u.ps
    reference = llg.get_magnetization_map(delays, B_ext=np.r_[0, 0.1, 0.5]).magnetization_map
    mag_map = llg.get_magnetization_map(delays, B_ext=np.r_[0, 100, 500] * u.mT).magnetization_map
    assert np.allclose(mag_map, reference)
    assert llg.get_hash(delays=delays, B_ext=llg.parse_field(np.r_[0, 100, 500] * u.mT)) == (
        llg.get_hash(delays=delays, B_ext=np.r_[0, 0.1, 0.5])
    )
    eq = llg.calc_equilibrium_magnetization(
        B_ext=np.r_[0, 0.1, 0.5] * u.T, init_magnetization=ALONG_X
    )
    assert np.allclose(eq, llg.calc_equilibrium_magnetization(np.r_[0, 0.1, 0.5], ALONG_X))
    with pytest.raises(pint.DimensionalityError):
        llg.get_magnetization_map(delays, B_ext=np.r_[0, 0, 1] * u.m).magnetization_map


NEAR_Z = np.array([np.sin(1e-3), 0.0, np.cos(1e-3)])


@pytest.mark.parametrize("B, theta_H", [(0.3, np.radians(60)), (1.2, np.radians(45))])
def test_llg_equilibrium_stoner_wohlfarth(B, theta_H, tmp_path):
    """the equilibrium of a uniaxial anisotropy in a tilted field"""
    K, Ms = 2e5, 5e5
    S = make_structure((make_layer("m", anisotropy=[0, 0, K] * u.J / u.m**3), 2))
    llg = simulation(LLG, S, tmp_path)
    eq = llg.calc_equilibrium_magnetization(
        B_ext=field(B, theta_H), init_magnetization=NEAR_Z, polar=True
    )
    assert eq.shape == (2, 1, 3)
    assert np.allclose(eq[:, 0, 0], 1)
    assert np.allclose(eq[:, 0, 1], stoner_wohlfarth_angle(B, theta_H, 2 * K / Ms), atol=1e-6)
    assert np.allclose(eq[:, 0, 2], 0, atol=1e-6)


def test_llb_equilibrium_stoner_wohlfarth(tmp_path):
    """the LLB starts with the mean-field amplitude and the anisotropy field
    is scaled by m_e**(beta - 1)"""
    K, Ms, T = 2e5, 5e5, 400.0
    B, theta_H = 0.3, np.radians(60)
    layer = make_layer("m", anisotropy=[0, 0, K] * u.J / u.m**3, aniso_exponent=3)
    llb = simulation(LLB, make_structure((layer, 1)), tmp_path)
    eq = llb.calc_equilibrium_magnetization(
        B_ext=field(B, theta_H), init_magnetization=NEAR_Z, temp_map=np.array([T]), polar=True
    )
    me = mean_field.mean_field_magnetization(T / 630, 0.5)
    assert np.isclose(eq[0, 0, 0], me, atol=1e-4)
    B_K = 2 * K / Ms * eq[0, 0, 0] ** 2
    assert np.isclose(eq[0, 0, 1], stoner_wohlfarth_angle(B, theta_H, B_K), atol=1e-6)


def test_equilibrium_of_canted_antiferromagnet(tmp_path):
    """two antiferromagnetically coupled sublattices cant towards a field
    perpendicular to them by sin(theta) = B / (2 B_E) with their exchange
    field B_E"""
    mu = 0.393 * 9.2740100783e-24
    B_E, B = 5.0, 2.0
    J = np.array([[mf_exchange(0.5, 630), -B_E * mu], [-B_E * mu, mf_exchange(0.5, 630)]])
    layer = make_sublattice_layer("ab", [(0, 0, 1), (0, 0, -1)], J)
    llg = simulation(LLG, make_structure((layer, 1)), tmp_path)
    init_mag = {"r": 1, "phi": [1e-3, np.pi - 1e-3] * u.rad, "gamma": 0}
    eq = llg.calc_equilibrium_magnetization(B_ext=np.r_[B, 0, 0], init_magnetization=init_mag)
    n = eq[0]
    sin_theta = B / (2 * B_E)
    assert np.allclose(n[:, 0], sin_theta, atol=1e-6)
    assert np.allclose(n[:, 2], [np.sqrt(1 - sin_theta**2), -np.sqrt(1 - sin_theta**2)], atol=1e-6)


@pytest.mark.parametrize("cls", [LLG, iLLG, LLB])
def test_start_in_equilibrium(cls, tmp_path):
    """the magnetization started in equilibrium stays put, whereas the one
    started along the easy axis precesses. The relaxation keeps the
    amplitude, so that the LLB still relaxes it by the tiny amount, by which
    the field and the anisotropy increase it beyond the mean field."""
    K = 2e5
    layer = make_layer(
        "m",
        anisotropy=[0, 0, K] * u.J / u.m**3,
        aniso_exponent=3,
        gilbert_damping=0.01,
        inertial_time=1 * u.fs,
    )
    sim = simulation(cls, make_structure((layer, 1)), tmp_path)
    if cls is iLLG:
        # tight tolerances would resolve the weakly damped nutation at 1/(alpha tau)
        sim.ode_options.update({"rtol": 1e-3, "atol": 1e-6})
    delays = np.r_[0:50:0.5] * u.ps
    kwargs = {"temp_map": np.full((len(delays), 1), 300.0)} if cls is LLB else {}
    B_ext = field(0.3, np.radians(60))
    m = sim.get_magnetization_map(
        delays, B_ext=B_ext, init_magnetization=NEAR_Z, equilibrate=True, **kwargs
    ).magnetization_map
    assert np.max(np.abs(m - m[0])) < (2e-4 if cls is LLB else 1e-5)
    m = sim.get_magnetization_map(
        delays, B_ext=B_ext, init_magnetization=NEAR_Z, **kwargs
    ).magnetization_map
    assert np.max(np.abs(m - m[0])) > 1e-2


def test_polar_and_cartesian_maps(tmp_path):
    """the maps are cartesian by default and polar for polar=True from the
    same cached results, and the cartesian equilibrium magnetization is a
    valid initial magnetization"""
    J = np.array([[mf_exchange(0.5, 500), -1e-21], [-1e-21, mf_exchange(0.5, 700)]])
    layer = make_sublattice_layer("ab", [(1, 0, 0), (-1, 0, 0)], J, gilbert_damping=0.1)
    llg = simulation(LLG, make_structure((layer, 2)), tmp_path)
    delays = np.r_[0:5:0.5] * u.ps
    B_ext = np.r_[0, 0.1, 0.5]
    for get_map, name in (
        (llg.get_magnetization_map, "magnetization_map"),
        (llg.get_sublattice_magnetization_map, "sublattice_magnetization_map"),
    ):
        cartesian = get_map(delays, B_ext=B_ext)
        polar = get_map(delays, B_ext=B_ext, polar=True)
        assert not cartesian.attrs["polar"] and polar.attrs["polar"]
        cartesian, polar = cartesian[name], polar[name]
        assert cartesian.shape == polar.shape
        assert np.allclose(convert_polar_to_cartesian(polar), cartesian)
        assert np.all(polar[..., 0] >= 0)
    eq = llg.calc_equilibrium_magnetization(B_ext=B_ext)
    assert np.allclose(
        convert_polar_to_cartesian(llg.calc_equilibrium_magnetization(B_ext=B_ext, polar=True)), eq
    )
    sub = llg.get_sublattice_magnetization_map(
        delays, B_ext=B_ext, init_magnetization=eq
    ).sublattice_magnetization_map
    assert np.allclose(sub[0], eq)


def test_equilibrium_magnetization_result(tmp_path):
    """the result of get_equilibrium_magnetization in cartesian and polar
    coordinates is a valid initial magnetization, as its array of
    calc_equilibrium_magnetization"""
    J = np.array([[mf_exchange(0.5, 500), -1e-21], [-1e-21, mf_exchange(0.5, 700)]])
    layer = make_sublattice_layer("ab", [(1, 0, 0), (-1, 0, 0)], J, gilbert_damping=0.1)
    llg = simulation(LLG, make_structure((layer, 2)), tmp_path)
    delays = np.r_[0:2:0.5] * u.ps
    B_ext = np.r_[0, 0.1, 0.5]
    eq = llg.calc_equilibrium_magnetization(B_ext=B_ext)
    cartesian = llg.get_equilibrium_magnetization(B_ext=B_ext)
    polar = llg.get_equilibrium_magnetization(B_ext=B_ext, polar=True)
    np.testing.assert_array_equal(cartesian.equilibrium_magnetization, eq)
    assert cartesian.dims("equilibrium_magnetization") == (
        "distances",
        "sublattices",
        "components",
    )
    assert polar.attrs["polar"] and not cartesian.attrs["polar"]
    reference = llg.get_sublattice_magnetization_map(delays, B_ext=B_ext, init_magnetization=eq)
    for init in (cartesian, polar):
        sub = llg.get_sublattice_magnetization_map(delays, B_ext=B_ext, init_magnetization=init)
        np.testing.assert_allclose(
            sub.sublattice_magnetization_map, reference.sublattice_magnetization_map, atol=1e-12
        )
    # also as initial magnetization of the equilibrium itself
    np.testing.assert_allclose(
        llg.calc_equilibrium_magnetization(B_ext=B_ext, init_magnetization=polar), eq, atol=1e-10
    )


def test_magnetization_map_is_cached(tmp_path):
    """the net and sublattice maps are cached together in a HDF5 file, from
    which both are loaded in cartesian and polar coordinates"""
    llg = LLG(
        make_structure((make_layer("m", gilbert_damping=0.1), 1)),
        False,
        cache_dir=str(tmp_path),
        disp_messages=False,
        progress_bar=False,
    )
    delays = np.r_[0:2:1] * u.ps
    reference = llg.get_magnetization_map(delays)
    filename = tmp_path / ("magnetization_map_" + llg.get_hash(delays=delays) + ".h5")
    assert filename.exists()
    cached = Result.load(filename)
    assert np.allclose(cached.magnetization_map, reference.magnetization_map)
    assert cached.sublattice_magnetization_map.shape == (2, 1, 1, 3)
    np.testing.assert_allclose(cached.delays, delays.to("s").magnitude)
    # an invalid cache file is recalculated
    filename.write_text("no HDF5 file")
    polar = llg.get_magnetization_map(delays, polar=True)
    assert np.allclose(
        convert_polar_to_cartesian(polar.magnetization_map), reference.magnetization_map
    )
    assert polar.dims("magnetization_map") == ("delays", "distances", "components")


def test_equilibrium_options(tmp_path):
    S = make_structure((make_layer("m", anisotropy=[0, 0, 2e5] * u.J / u.m**3), 1))
    llg = simulation(LLG, S, tmp_path)
    delays = np.r_[0:2:1] * u.ps
    B_ext = field(0.3, np.radians(60))
    hash = llg.get_hash(delays=delays, B_ext=B_ext, equilibrate=True)
    assert hash != llg.get_hash(delays=delays, B_ext=B_ext)
    llg.equilibrium_options["tol"] = 1e-3
    assert hash != llg.get_hash(delays=delays, B_ext=B_ext, equilibrate=True)
    llg.equilibrium_options.update({"tol": 1e-9, "max_time": 1e-15})
    with pytest.warns(UserWarning, match="did not reach its equilibrium"):
        llg.calc_equilibrium_magnetization(B_ext=B_ext, init_magnetization=NEAR_Z)


# exchange relaxation


def test_exchange_relaxation_torque():
    """the torque of Raposo et al. transfers angular momentum between the two
    sublattices of a layer and vanishes otherwise"""
    mag = SimpleNamespace(
        active=np.ones((2, 2)),
        exch_relax_rate=np.array([0.013, 0.0]),
        mag_moment=np.array([[2.0, 7.0], [2.0, 7.0]]) * 9.274e-24,
        concentration=np.array([[0.75, 0.25], [0.75, 0.25]]),
        gyromagnetic_ratio=np.full((2, 2), magnetization_scipy.GAMMA_E),
    )
    m_norm = np.array([[0.4, 0.6], [0.4, 0.6]])
    B_par = np.array([[[0, 0, 3.0], [0, 0, -1.0]], [[0, 0, 3.0], [0, 0, -1.0]]])
    alpha_par = np.array([[0.01, 0.02], [0.01, 0.02]])
    torque = magnetization_scipy.exchange_relaxation_torque(m_norm, B_par, alpha_par, mag)
    mu, x = mag.mag_moment[0], mag.concentration[0]
    base = abs(magnetization_scipy.GAMMA_E) * 0.013 * (x @ (mu * m_norm[0]))
    base /= mu[0] * m_norm[0, 0] * mu[1] * m_norm[0, 1]
    D = mu[0] * B_par[0, 0] - mu[1] * B_par[0, 1]
    assert np.allclose(torque[0, 0], base * 0.01 * D)
    assert np.allclose(torque[0, 1], -base * 0.02 * D)
    # no torque for a zero rate, a padded sublattice, or a single sublattice
    assert np.allclose(torque[1], 0)
    mag.active[0, 1] = 0
    assert np.allclose(
        magnetization_scipy.exchange_relaxation_torque(m_norm, B_par, alpha_par, mag), 0
    )
    single = SimpleNamespace(
        **{k: v[:, :1] if np.ndim(v) == 2 else v for k, v in vars(mag).items()}
    )
    torque = magnetization_scipy.exchange_relaxation_torque(
        m_norm[:, :1], B_par[:, :1], alpha_par[:, :1], single
    )
    assert np.allclose(torque, 0)


def test_concentration():
    """the concentrations are equal unless they are given, e.g. as the
    fractions of the constituents of a mixed atom"""
    J = np.diag([mf_exchange(0.5, 630)] * 2)
    magnetic = ud.MagneticParameters(eff_spin=[0.5, 0.5], mf_exch_coupling=J * u.J)
    assert np.allclose(magnetic.sublattice_values("concentration"), [0.5, 0.5])
    magnetic.concentration = [0.3, 0.7]
    assert np.allclose(magnetic.sublattice_values("concentration"), [0.3, 0.7])
    magnetic.concentration = [0.3, 0.3, 0.4]
    with pytest.raises(ValueError, match="but there are 3 sublattices"):
        magnetic.check_sublattices()
    Fe, Gd = ud.Atom("Fe", magnetization=(0, 0, 1)), ud.Atom("Gd", magnetization=(0, 0, -1))
    GdFe = ud.AtomMixed("GdFe")
    GdFe.add_atom(Fe, 0.75)
    GdFe.add_atom(Gd, 0.25)
    layer = make_layer("GdFe", eff_spin=[0.5, 0.5], mf_exch_coupling=J * u.J)
    layer.atom = GdFe
    assert np.allclose(layer.magnetic.sublattice_values("concentration"), [0.75, 0.25])


@pytest.fixture(scope="module")
def gdfeco():
    """a GdFeCo film of Raposo et al. 2022 and the electron temperature of
    their two-temperature model for two absorbed power densities"""
    x = np.array([0.75, 0.25])
    J_TM, J_RE, J_TMRE = 2.59e-21, 1.35e-21, -1.13e-21
    J0 = 12 * np.array([[x[0] * J_TM, x[1] * J_TMRE], [x[0] * J_TMRE, x[1] * J_RE]])
    mu = np.array([1.92, 7.63])
    Ms = x * mu * 9.2740100783e-24 / (0.74 * 0.35e-9**3)
    FeCo = ud.Atom("Fe", magnetization=(0, 0, 1))
    Gd = ud.Atom("Gd", magnetization=(0, 0, -1))
    GdFeCo = ud.AtomMixed("GdFeCo")
    GdFeCo.add_atom(FeCo, x[0])
    GdFeCo.add_atom(Gd, x[1])
    rho, gamma_e, C_l, g_el, tau_D, T_R = 8000, 1.8e5 / 300, 3.8e6, 7e17, 10e-12, 300
    layer = ud.AmorphousLayer(
        "GdFeCo",
        "GdFeCo",
        thickness=1 * u.nm,
        density=rho * u.kg / u.m**3,
        atom=GdFeCo,
        heat_capacity=[f"{gamma_e / rho:g}*T", C_l / rho * u.J / u.kg / u.K],
        therm_cond=[20 * u.W / (u.m * u.K), 1 * u.W / (u.m * u.K)],
        sub_system_coupling=[
            f"-{g_el:g}*(T_0-T_1) - {gamma_e / tau_D:g}*T_0*(T_0-{T_R:g})",
            f"{g_el:g}*(T_0-T_1)",
        ],
        lin_therm_exp=[0, 1e-5],
        dielectric=ud.Dispersion.from_opt_pen_depth(20 * u.nm),
        eff_spin=[np.inf, np.inf],
        mag_moment=mu * u.bohr_magneton,
        lamda=0.02,
        mf_exch_coupling=J0 * u.J,
        mag_saturation=Ms * u.J / u.T / u.m**3,
        anisotropy=[[0, 0, 1.87e6], [0, 0, 0.62e6]] * u.J / u.m**3,
        aniso_exponent=2,
        exch_relax_rate=0.013,
    )
    S = make_structure((layer, 4))
    heat = ud.Heat(S, True, save_data=False, disp_messages=False, progress_bar=False)
    heat.heat_diffusion = True
    delays = np.r_[-0.5:6:0.01] * u.ps
    temp_maps = []
    for fluence in (0.81, 1.62):
        heat.excitation = {
            "fluence": [fluence] * u.mJ / u.cm**2,
            "delay_pump": [0] * u.ps,
            "pulse_width": [50] * u.fs,
            "multilayer_absorption": False,
        }
        res = heat.get_temp_map(delays, T_R * u.K)
        temp_map = res.temp_map
        temp_maps.append(temp_map)
    return S, delays, temp_maps


def test_all_optical_switching(gdfeco, tmp_path):
    """the exchange relaxation switches GdFeCo above a threshold fluence via
    a transient ferromagnetic-like state, which the conventional LLB does
    not, see Raposo et al. 2022"""
    S, delays, (low, high) = gdfeco
    assert S.get_layer_vectors()[2][0].curie_temp.magnitude == pytest.approx(603, abs=1)
    llb = LLB(S, True, save_data=False, disp_messages=False, progress_bar=False)

    def m_z(temp_map, **kwargs):
        sub = llb.get_sublattice_magnetization_map(
            delays, temp_map=temp_map, **kwargs
        ).sublattice_magnetization_map
        return np.mean(sub[..., 2], axis=1)

    switched = m_z(high, exchange_relaxation=True)
    # both sublattices are reversed, the RE still recovers at the end
    assert switched[-1, 0] < -0.3 and switched[-1, 1] > 0.1
    assert np.any((switched[:, 0] < -0.01) & (switched[:, 1] < -0.01))
    assert m_z(low, exchange_relaxation=True)[-1, 0] > 0.3
    assert m_z(high)[-1, 0] > 0
