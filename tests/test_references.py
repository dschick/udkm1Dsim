"""Tests of the multilayer formalisms against reference results of external
packages, see ``tests/data/references/README.md``."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import udkm1Dsim as ud
from udkm1Dsim import u

REFERENCES = sorted(
    path
    for path in (Path(__file__).parent / "data" / "references").glob("*.npz")
    if not path.stem.startswith("xray_bragg")
)
BRAGG = Path(__file__).parent / "data" / "references" / "xray_bragg_si004.npz"
# np.trapz is renamed to np.trapezoid in numpy 2.0
trapezoid = getattr(np, "trapezoid", None) or np.trapz
EXACT = [ud.BerremanElzo, ud.BerremanStepanov]
FIRST_ORDER = [ud.Elzo, ud.Stepanov]


def load(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}, json.loads(str(data["meta"]))


def complex_value(value):
    return complex(*value) if isinstance(value, list) else complex(value)


def build_layer(spec, eps):
    """Layer with a constant dielectric tensor from the inputs of a case,
    where the substrate has no thickness, so that the transmission is given
    at its top interface as by the references."""
    if "g" in spec:
        g = complex_value(spec["g"])
        dielectric = ud.Dielectric(
            eps0=ud.Dispersion.from_ref_index(complex_value(spec["n"])),
            eps_mag=ud.Dispersion.from_array([1e-3, 1e5], [g, g], kind="eps"),
            strain_model="none",
        )
    elif "principal" in spec:
        dielectric = ud.Dielectric(
            eps0=ud.Dispersion.from_array([1e-3, 1e5], np.stack([eps, eps]), kind="eps"),
            strain_model="none",
        )
    else:
        dielectric = ud.Dielectric(
            eps0=ud.Dispersion.from_ref_index(complex_value(spec["n"])), strain_model="none"
        )
    return ud.Layer(
        spec["name"],
        spec["name"],
        structural=ud.StructuralParameters(
            thickness=spec.get("thickness", 0.0) * u.nm,
            density=1000 * u.kg / u.m**3,
            roughness=spec.get("roughness", 0.0) * u.nm,
        ),
        optical=ud.OpticalParameters(dielectric=dielectric),
        magnetic=ud.MagneticParameters(magnetization=tuple(spec.get("magnetization", (0, 0, 0)))),
    )


def build_structure(data, meta):
    layers = [build_layer(spec, eps) for spec, eps in zip(meta["layers"], data["eps"])]
    S = ud.Structure(meta["case"])
    for layer in layers[1:-1]:
        S.add_sub_structure(layer, 1)
    S.add_superstrate(layers[0])
    S.add_substrate(layers[-1])
    return S, layers


def jones(cls, S, data, meta):
    sim = cls(S, True, disp_messages=False, progress_bar=False, save_data=False)
    sim.energy = np.r_[meta["energy"]] * u.eV
    sim.theta = data["theta"] * u.deg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.homogeneous_jones_matrices()
        R, T = res.jones_reflection, res.jones_transmission
    return R[0], T[0]


def assert_close(actual, reference, rtol):
    """deviation relative to the largest value of the reference"""
    deviation = np.max(np.abs(actual - reference)) / np.max(np.abs(reference))
    assert deviation < rtol, f"deviation {deviation:.1e} > {rtol:.0e}"


def formalisms(meta):
    case = meta["case"]
    if case.startswith("birefringent"):
        return EXACT
    if case.startswith("magnetic"):
        return EXACT + FIRST_ORDER
    return [ud.Parratt] + EXACT + FIRST_ORDER


CASES = [
    pytest.param(path, cls, id=f"{path.stem}-{cls.__name__}")
    for path in REFERENCES
    for cls in formalisms(load(path)[1])
]


@pytest.mark.parametrize("path", REFERENCES, ids=[path.stem for path in REFERENCES])
def test_reference_dielectric_tensors(path):
    """the layers reproduce the dielectric tensors of the references in the
    frame of Elzo, also the gyrotropic part from their magnetization"""
    data, meta = load(path)
    _, layers = build_structure(data, meta)
    for layer, eps in zip(layers, data["eps"]):
        np.testing.assert_allclose(
            layer.get_dielectric_tensor(meta["energy"])[0], eps, rtol=1e-12, atol=1e-15
        )


@pytest.mark.parametrize("path, cls", CASES)
def test_reflection_against_dyna(path, cls):
    """complex Jones matrices of the reflection against the exact Berreman
    formalism of Dyna, where Elzo and Stepanov are exact only to first order
    in the gyrotropic part of the dielectric tensor"""
    data, meta = load(path)
    S, _ = build_structure(data, meta)
    R, _ = jones(cls, S, data, meta)
    rtol = 1e-12
    if cls in FIRST_ORDER:
        rtol = 1e-9
        if meta["case"].startswith("magnetic_optics"):
            rtol = 1e-3
        elif meta["case"].startswith("magnetic_xray"):
            rtol = 3e-5
    assert_close(R, data["dyna_R"], rtol)


@pytest.mark.parametrize("path, cls", [c for c in CASES if "tmm_r_s" in load(c.values[0])[0]])
def test_reflection_transmission_against_tmm(path, cls):
    """complex reflection and transmission coefficients of isotropic stacks
    against tmm, which uses the same sign convention of the pi polarization"""
    data, meta = load(path)
    S, _ = build_structure(data, meta)
    R, T = jones(cls, S, data, meta)
    assert_close(R[:, 0, 0], data["tmm_r_s"], 1e-9)
    assert_close(R[:, 1, 1], data["tmm_r_p"], 1e-9)
    assert_close(T[:, 0, 0], data["tmm_t_s"], 1e-9)
    assert_close(T[:, 1, 1], data["tmm_t_p"], 1e-9)


@pytest.mark.parametrize("path, cls", [c for c in CASES if "refnx_R" in load(c.values[0])[0]])
def test_xray_reflectivity_against_refnx(path, cls):
    """X-ray reflectivity of isotropic stacks with and without the
    roughness of Névot and Croce against refnx"""
    data, meta = load(path)
    S, _ = build_structure(data, meta)
    R, _ = jones(cls, S, data, meta)
    assert_close(np.abs(R[:, 0, 0]) ** 2, data["refnx_R"], 1e-9)


@pytest.mark.parametrize("path, cls", [c for c in CASES if "pygtm_r" in load(c.values[0])[0]])
def test_reflection_transmission_against_pygtm(path, cls):
    """intensities of the reflection and the transmitted fields against
    PyGTM, which deviates by up to 2e-5 from Dyna and udkm1Dsim for a tilted
    optic axis, while both agree to 1e-15"""
    data, meta = load(path)
    S, _ = build_structure(data, meta)
    R, T = jones(cls, S, data, meta)
    r, t = data["pygtm_r"], data["pygtm_t"]  # (rpp, rps, rss, rsp) and (tpp, tps, tsp, tss)
    rtol = 1e-4 if meta["case"] == "birefringent_tilted" else 1e-12
    assert_close(np.abs(R[:, 0, 0]) ** 2, np.abs(r[:, 2]) ** 2, rtol)
    assert_close(np.abs(R[:, 1, 1]) ** 2, np.abs(r[:, 0]) ** 2, rtol)
    assert_close(np.abs(T[:, 0, 0]), np.abs(t[:, 3]), rtol)
    assert_close(np.abs(T[:, 1, 1]), np.abs(t[:, 0]), rtol)


def first_order_rtol(cls, meta, rtol):
    """tolerance of the formalisms, which are exact only to first order in the
    gyrotropic part of the dielectric tensor"""
    if cls in FIRST_ORDER and meta["case"].startswith("magnetic_optics"):
        return max(rtol, 1e-3)
    if cls in FIRST_ORDER and meta["case"].startswith("magnetic_xray"):
        return max(rtol, 3e-5)
    return rtol


@pytest.mark.parametrize("path, cls", [c for c in CASES if "genx_r" in load(c.values[0])[0]])
def test_reflectivity_against_genx_parratt(path, cls):
    """sigma reflectivity of isotropic stacks with and without roughness
    against the Parratt recursion of GenX"""
    data, meta = load(path)
    S, _ = build_structure(data, meta)
    R, _ = jones(cls, S, data, meta)
    assert_close(np.abs(R[:, 0, 0]) ** 2, np.abs(data["genx_r"]) ** 2, 1e-9)


@pytest.mark.parametrize("path, cls", [c for c in CASES if "genx_W" in load(c.values[0])[0]])
def test_reflectivity_against_genx_xrmr(path, cls):
    """all four polarization channels of the resonant magnetic X-ray
    reflectivity against the recursion of Stepanov and Sinha of GenX, which
    treats a transverse magnetization separately and agrees with Dyna only to
    2e-6 for it"""
    data, meta = load(path)
    S, _ = build_structure(data, meta)
    R, _ = jones(cls, S, data, meta)
    rtol = 1e-5 if meta["case"].endswith("transverse") else 1e-9
    assert_close(np.abs(R) ** 2, np.abs(data["genx_W"]) ** 2, first_order_rtol(cls, meta, rtol))


@pytest.mark.parametrize(
    "path, cls",
    [
        c
        for c in CASES
        if "pyllama_R" in load(c.values[0])[0]
        and np.any(np.isfinite(load(c.values[0])[0]["pyllama_R"]))
    ],
)
def test_reflectivity_against_pyllama(path, cls):
    """all four polarization channels of the reflectivity against PyLlama in
    its basis (p, s) at the angles, where the sorting of its modes succeeds"""
    data, meta = load(path)
    S, _ = build_structure(data, meta)
    R, _ = jones(cls, S, data, meta)
    reference = np.abs(data["pyllama_R"][:, ::-1, ::-1]) ** 2
    valid = np.all(np.isfinite(reference), axis=(1, 2))
    rtol = first_order_rtol(cls, meta, 1e-8)
    assert_close(np.abs(R[valid]) ** 2, reference[valid], rtol)


@pytest.mark.parametrize(
    "path, cls", [c for c in CASES if "magsl_R_sigma" in load(c.values[0])[0]]
)
def test_reflectivity_against_mag_sl(path, cls):
    """resonant magnetic X-ray reflectivity for sigma, pi, and circular
    incidence without analyzer against MAG_sl of the X-ray Server, whose
    output has six significant digits"""
    data, meta = load(path)
    S, _ = build_structure(data, meta)
    R, _ = jones(cls, S, data, meta)
    circ_plus = np.array([-np.sqrt(0.5), -1j * np.sqrt(0.5)])
    circ_minus = np.array([np.sqrt(0.5), -1j * np.sqrt(0.5)])
    rtol = first_order_rtol(cls, meta, 5e-6)
    assert_close(np.sum(np.abs(R[:, :, 0]) ** 2, axis=1), data["magsl_R_sigma"], rtol)
    assert_close(np.sum(np.abs(R[:, :, 1]) ** 2, axis=1), data["magsl_R_pi"], rtol)
    assert_close(np.sum(np.abs(R @ circ_plus) ** 2, axis=1), data["magsl_R_circ_plus"], rtol)
    assert_close(np.sum(np.abs(R @ circ_minus) ** 2, axis=1), data["magsl_R_circ_minus"], rtol)


@pytest.mark.parametrize("name", ["birefringent_inplane", "birefringent_tilted"])
def test_euler_angles_of_pygtm(name):
    """the Euler angles of PyGTM correspond to (theta, 90 deg - psi, -phi),
    see :func:`~udkm1Dsim.structures.dielectric.euler_matrix`"""
    data, meta = load(Path(__file__).parent / "data" / "references" / f"{name}.npz")
    spec = meta["layers"][1]
    theta, phi, psi = spec["euler_pygtm"]
    principal = np.diag([complex_value(e) for e in spec["principal"]])
    dielectric = ud.Dielectric(
        eps0=ud.Dispersion.from_array([1e-3, 1e5], np.stack([principal] * 2), kind="eps"),
        strain_model="none",
        euler_angles=(theta, 90 - psi, -phi),
    )
    np.testing.assert_allclose(
        dielectric.get_tensor(meta["energy"])[0], data["eps"][1], atol=1e-14
    )


@pytest.mark.parametrize("pol, key", [("sigma", "gid_R_sigma"), ("pi", "gid_R_pi")])
def test_bragg_reflection_against_gid_sl(pol, key):
    """Bragg curve of Si (004) at Cu Ka1 of XrayDyn against GID_sl of the
    X-ray Server with the same susceptibilities chi_0 and chi_h from the unit
    cell of udkm1Dsim, including its Debye-Waller factor, where the Darwin
    layers of XrayDyn deviate from the exact dynamical theory of GID_sl by a
    few percent of the peak at the steep flanks only"""
    data, meta = load(BRAGG)
    a = meta["a"]
    Si = ud.Atom("Si")
    uc = ud.UnitCell(
        "Si",
        "Si",
        c_axis=a * u.angstrom,
        a_axis=a * u.angstrom,
        b_axis=a * u.angstrom,
        deb_wal_fac=meta["deb_wal_fac"] * u.m**2,
    )
    for z in [0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75]:
        uc.add_atom(Si, z)
    S = ud.Structure("Si")
    S.add_sub_structure(uc, int(round(10 * meta["thickness"] / a)))
    sim = ud.XrayDyn(S, True, disp_messages=False, progress_bar=False, save_data=False)
    sim.energy = np.r_[meta["energy"]] * u.eV
    sim.theta = data["theta"] * u.deg
    sim.set_polarization(pol, "none")
    R = sim.homogeneous_reflectivity().reflectivity[0]
    reference = data[key]
    deviation = np.abs(R - reference) / np.max(reference)
    assert np.sqrt(np.mean(deviation**2)) < 1e-2
    assert np.max(deviation) < 0.1
    assert np.argmax(R) == np.argmax(reference)
    np.testing.assert_allclose(np.max(R), np.max(reference), rtol=3e-3)
    integral = trapezoid(R, data["offset"]) / trapezoid(reference, data["offset"])
    assert abs(integral - 1) < 1e-2
