#!/usr/bin/env python

import copy
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import udkm1Dsim as ud
from udkm1Dsim import LLB, LLG, Magnetization, iLLG, u
from udkm1Dsim.helpers import convert_polar_to_cartesian
from udkm1Dsim.simulations.solvers import magnetization_scipy, mean_field

# polar coordinates of a unit vector along x
ALONG_X = np.array([1.0, np.pi / 2, 0.0])

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


def test_magnetization_get_hash(magnetization, delays, temp_map):
    magnetization.get_hash(delays=delays, temp_map=temp_map)


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
    llb.get_magnetization_map(delays, temp_map=temp_map)
    llb.force_recalc = False
    llb.get_magnetization_map(delays, temp_map=temp_map)


@pytest.mark.parametrize(
    "temp, below_tc",
    [(50, True), (99.9, True), (99.97, True), (100, False), (100.02, False), (150, False)],
)
def test_llb_mean_field_mag_map_at_curie_temp(llb, delays, distances, temp, below_tc):
    # all layers of the fixture have a Curie temperature of 100 K
    temp_map = temp * np.ones([len(delays), len(distances)])
    temp_map[0, :] = 50  # at least one temperature below T_C is required
    mf_mag_map = llb.get_mean_field_mag_map(temp_map)
    assert np.all(mf_mag_map[0, :] > 0)
    if below_tc:
        assert np.all(mf_mag_map[1:, :] > 0)
    else:
        assert np.all(mf_mag_map[1:, :] == 0)


def test_llb_mean_field_mag_map_above_curie_temp(llb, delays, distances):
    # no temperature below T_C is required anymore
    mf_mag_map = llb.get_mean_field_mag_map(150 * np.ones([len(delays), len(distances)]))
    assert np.all(mf_mag_map == 0)


def test_llb_get_magnetization_just_below_curie_temp(llb, delays, distances):
    llb = copy.deepcopy(llb)  # do not alter the module-scoped fixture
    llb.force_recalc = True
    llb.save_data = False
    temp_map = 99.97 * np.ones([len(delays), len(distances), 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        magnetization_map = llb.get_magnetization_map(delays, temp_map=temp_map)
    assert np.all(np.isfinite(magnetization_map))


def test_llb_input_validation(llb, temp_map, delays, distances):
    with pytest.raises(TypeError):
        llb.get_magnetization_map(delays, temp_map=list(temp_map))
    with pytest.raises(ValueError):
        llb.get_magnetization_map(delays, temp_map=temp_map, H_ext=np.zeros(2))
    with pytest.raises(ValueError):
        llb.get_magnetization_map(delays, temp_map=temp_map, init_mag=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        llb.get_magnetization_map(delays, temp_map=temp_map[:, 0:3, :])
    # an Nx3 initial magnetization and a per-delay external field are valid
    init_mag = np.tile(ALONG_X, (len(distances), 1))
    H_ext = np.tile(np.r_[0, 0, 1.0], (len(delays), 1))
    llb.get_magnetization_map(delays, temp_map=temp_map, init_mag=init_mag, H_ext=H_ext)


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
    mag = magnetization_scipy.stack_magnetic_parameters(llb.get_magnetic_parameters(is_magnetic))
    thicknesses = S.get_layer_property_vector("structural.thickness.magnitude")
    m = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])[:, np.newaxis, :]
    B = magnetization_scipy.interlayer_exchange_field(
        m, mag, llb.get_interlayer_exchange_stiffnesses(is_magnetic), thicknesses
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
    A_mag = llb.get_interlayer_exchange_stiffnesses(np.r_[True, False, True])
    assert np.allclose(A_mag, 0)
    llb.S = make_structure((make_layer("a", exch_stiffness=A), 2))
    A_mag = llb.get_interlayer_exchange_stiffnesses(np.r_[True, True])
    assert np.allclose(A_mag, [[0, 1e-11], [1e-11, 0]])


def test_llb_larmor_precession(tmp_path):
    """without damping the magnetization precesses around the external field
    at the Larmor frequency"""
    S = make_structure((make_layer("m", lamda=0.0), 1))
    llb = simulation(LLB, S, tmp_path)
    delays = np.r_[0:100:0.5] * u.ps
    B = 0.5
    temp_map = 300 * np.ones((len(delays), 1, 1))
    mag_map = llb.get_magnetization_map(delays, temp_map=temp_map, H_ext=np.r_[0, 0, B])
    m = convert_polar_to_cartesian(mag_map)[:, 0, :]
    t = delays.to("s").magnitude
    omega = magnetization_scipy.GAMMA_E * B
    norm = mag_map[0, 0, 0]
    assert np.allclose(m[:, 0], norm * np.cos(omega * t), atol=1e-5)
    assert np.allclose(m[:, 1], -norm * np.sin(omega * t), atol=1e-5)
    assert np.allclose(mag_map[:, 0, 0], norm)


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
        delays, temp_map=temp_map, init_mag=ALONG_X, H_ext=np.r_[0, 0, 0.01]
    )
    mf = llb.calc_mean_field_mag_map(temp_map[:, :, 0])
    reference = mean_field.mean_field_magnetization(400 / 630, eff_spin)
    assert np.isclose(mf[-1, 0], reference, atol=1e-3)
    assert np.isclose(mag_map[-1, 0, 0], mf[-1, 0], rtol=1e-2)


def test_llb_starts_in_equilibrium(tmp_path):
    """without an explicit initial magnetization the layers start in the
    equilibrium of their initial temperature and stay there"""
    S = make_structure((make_layer("m", lamda=0.1), 3))
    llb = simulation(LLB, S, tmp_path)
    delays = np.r_[0:10:0.5] * u.ps
    temp_map = 300 * np.ones((len(delays), 3, 1))
    mag_map = llb.get_magnetization_map(delays, temp_map=temp_map)
    mf = llb.calc_mean_field_mag_map(temp_map[:, :, 0])
    assert np.allclose(mag_map[:, :, 0], mf, atol=1e-6)
    # the direction is the one of the layers
    m = convert_polar_to_cartesian(mag_map)
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
    mag_map = llb.get_magnetization_map(delays, temp_map=temp_map)
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
        llb.get_magnetization_map(delays, temp_map=temp_map, init_mag=np.zeros(3))


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
    mag_map = llg.get_magnetization_map(delays, H_ext=np.r_[0, 0, B])
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
    mag_map = llg.get_magnetization_map(delays, mag_amplitude_map=amplitude_map)
    assert np.allclose(mag_map[:, :, 0], amplitude_map)

    temp_map = np.stack((np.full((M, 2), 400.0), amplitude_map), axis=2)
    mag_map = llg.get_magnetization_map(delays, temp_map=temp_map)
    assert np.allclose(mag_map[:, :, 0], llg.get_mean_field_mag_map(temp_map[:, :, 0]))
    llg.mag_sub_system = 1
    mag_map = llg.get_magnetization_map(delays, temp_map=temp_map)
    assert np.allclose(mag_map[:, :, 0], amplitude_map)

    # a negative amplitude reverses the direction
    init_mag = np.array([[0.7, np.pi / 2, 0.0], [-0.7, np.pi / 2, 0.0]])
    mag_map = llg.get_magnetization_map(delays, init_mag=init_mag)
    m = convert_polar_to_cartesian(mag_map)
    assert np.allclose(m[:, 0], [0.7, 0, 0])
    assert np.allclose(m[:, 1], [-0.7, 0, 0])
    # but a negative prescribed amplitude is not allowed
    with pytest.raises(ValueError, match="must not be negative"):
        llg.get_magnetization_map(delays, mag_amplitude_map=-amplitude_map)


def test_llg_non_magnetic_layers(tmp_path):
    """layers without a saturation magnetization are not magnetic"""
    spacer = make_layer("s", mag_saturation=0 * u.J / u.T / u.m**3)
    S = make_structure((make_layer("m", gilbert_damping=0.1), 1), (spacer, 1))
    llg = simulation(LLG, S, tmp_path)
    mag_map = llg.get_magnetization_map(np.r_[0:5:1] * u.ps, H_ext=np.r_[0, 0, 1])
    assert np.allclose(mag_map[:, 1], 0)
    assert np.all(mag_map[1:, 0, 1] < np.pi / 2)


@pytest.mark.parametrize("cls", [LLG, LLB])
def test_strain_acts_as_anisotropy(cls, tmp_path):
    """the magnetoelastic field of a strain is the anisotropy K_z = -B_1 strain"""
    K, strain = 2e5, 1e-3
    delays = np.r_[0:50:1] * u.ps
    temp_map = np.full((len(delays), 1), 300.0)
    init_mag = np.array([1.0, np.pi / 4, 0.0])
    kwargs = {"temp_map": temp_map} if cls is LLB else {}
    S = make_structure(
        (make_layer("a", gilbert_damping=0.1, anisotropy=[0, 0, K] * u.J / u.m**3), 1)
    )
    reference = simulation(cls, S, tmp_path).get_magnetization_map(
        delays, init_mag=init_mag, **kwargs
    )
    S = make_structure(
        (
            make_layer(
                "b", gilbert_damping=0.1, magnetoelastic_coupling=-K / strain * u.J / u.m**3
            ),
            1,
        )
    )
    mag_map = simulation(cls, S, tmp_path).get_magnetization_map(
        delays, init_mag=init_mag, strain_map=np.full((len(delays), 1), strain), **kwargs
    )
    assert not np.allclose(reference[-1], reference[0], atol=1e-3)
    assert np.allclose(mag_map, reference, atol=1e-6)


def test_illg_requires_damping_and_inertia(tmp_path):
    S = make_structure((make_layer("m", gilbert_damping=0.1), 1))
    illg = simulation(iLLG, S, tmp_path)
    with pytest.raises(ValueError, match="inertial_time"):
        illg.get_magnetization_map(np.r_[0:2:1] * u.ps)


def test_illg_approaches_llg(tmp_path):
    """for a short inertial time the iLLG follows the LLG, starting with the
    velocity of the LLG"""
    S = make_structure((make_layer("m", gilbert_damping=0.2, inertial_time=1 * u.fs), 1))
    delays = np.r_[0:20:0.5] * u.ps
    H_ext = np.r_[0, 0, 1]
    llg = simulation(LLG, S, tmp_path)
    illg = simulation(iLLG, S, tmp_path)
    assert illg.ode_options["method"] == "LSODA"
    reference = convert_polar_to_cartesian(llg.get_magnetization_map(delays, H_ext=H_ext))
    m = convert_polar_to_cartesian(illg.get_magnetization_map(delays, H_ext=H_ext))
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
    state = illg.get_initial_state(n, inputs, is_magnetic)
    assert state.shape == (2, 2, 1, 3)
    assert np.allclose(state[0], n)
    omega = magnetization_scipy.GAMMA_E / 1.04
    # precession along y and damping towards the field along z
    assert np.allclose(state[1, :, 0], [0, -omega, -0.2 * omega])
    state = illg.get_initial_state(n, inputs, is_magnetic, init_velocity=np.r_[0, 1e9, 0])
    assert np.allclose(state[1, :, 0], [0, 1e9, 0])
    with pytest.raises(ValueError, match="init_velocity"):
        illg.get_initial_state(n, inputs, is_magnetic, init_velocity=np.zeros((3, 3)))


def test_illg_free_nutation(tmp_path):
    """without a field an initial velocity v_0 decays on the inertial time tau
    while it rotates at 1/(alpha tau), which displaces the unit vector by
    v_0 tau alpha / (alpha + i) in the plane perpendicular to it"""
    alpha, tau, v0 = 0.5, 10e-15, 1e11
    S = make_structure((make_layer("m", gilbert_damping=alpha, inertial_time=tau * u.s), 1))
    illg = simulation(iLLG, S, tmp_path)
    delays = np.r_[0:1:0.01] * u.ps
    mag_map = illg.get_magnetization_map(delays, init_velocity=np.r_[0, v0, 0])
    m = convert_polar_to_cartesian(mag_map)[-1, 0]
    displacement = v0 * tau * alpha / (alpha + 1j)
    assert np.isclose(m[1] + 1j * m[2], displacement, rtol=1e-3)
