#!/usr/bin/env python

import copy

import numpy as np
import pint
import pytest
from pint.testing import assert_allclose

from udkm1Dsim import (
    AmorphousLayer,
    Atom,
    Dielectric,
    Dispersion,
    Layer,
    Structure,
    UnitCell,
    u,
)
from udkm1Dsim.structures.dielectric import euler_matrix, gyrotropic_tensor, to_energy

# ============================================================================
# Dispersion
# ============================================================================


def test_to_energy():
    np.testing.assert_allclose(to_energy(800 * u.nm), 1.5498, rtol=1e-4)
    np.testing.assert_allclose(to_energy(1 * u.keV), 1000)
    np.testing.assert_allclose(to_energy(2.0), 2.0)


def test_dispersion_from_ref_index():
    d = Dispersion.from_ref_index(2 + 0.1j)
    assert d.scalar
    np.testing.assert_allclose(d.chi([1, 1000]), (2 + 0.1j) ** 2 - 1)
    assert repr(d) == "Dispersion(n=(2+0.1j))"
    # uniaxial and diagonal
    d = Dispersion.from_ref_index([1.5, 2.0])
    assert not d.scalar
    np.testing.assert_allclose(np.diagonal(d.chi(1.0)[0]), [1.25, 1.25, 3])
    d = Dispersion.from_ref_index([1.5, 2.0, 3.0])
    np.testing.assert_allclose(np.diagonal(d.chi(1.0)[0]), [1.25, 3, 8])
    with pytest.raises(ValueError, match="principal axes"):
        Dispersion.from_ref_index(np.eye(3))


def test_dispersion_from_array_kinds_and_shapes():
    energy = [1, 2, 3] * u.eV
    eps = Dispersion.from_array(energy, [2, 3, 4])
    np.testing.assert_allclose(eps.chi([1.5, 2.5]), [1.5, 2.5])
    chi = Dispersion.from_array(energy, [2, 3, 4], kind="chi")
    np.testing.assert_allclose(chi.chi(2), [3])
    n = Dispersion.from_array(energy, [2, 3, 4], kind="n")
    np.testing.assert_allclose(n.chi(2), [8])
    tensor = Dispersion.from_array(energy, np.ones((3, 3, 3)))
    np.testing.assert_allclose(tensor.chi(2)[0], np.ones((3, 3)) - np.eye(3))
    with pytest.raises(ValueError, match="shape"):
        Dispersion.from_array(energy, np.ones((3, 4)))
    with pytest.raises(ValueError, match="first axis"):
        Dispersion.from_array(energy, [1, 2])
    with pytest.raises(ValueError, match="kind"):
        Dispersion.from_array(energy, [1, 2, 3], kind="foo")


def test_dispersion_from_array_spectral_units():
    # wavelengths are converted to photon energies and sorted
    d = Dispersion.from_array([400, 800] * u.nm, [4, 2])
    lo, hi = d.energy_range
    np.testing.assert_allclose([lo, hi], [1.5498, 3.0996], rtol=1e-4)
    np.testing.assert_allclose(d.chi(800 * u.nm), [1])
    np.testing.assert_allclose(d.chi(400 * u.nm), [3])


def test_dispersion_out_of_range():
    d = Dispersion.from_array([1, 2], [2, 3])
    with pytest.raises(
        ValueError, match="not defined for the photon energy 3 eV,\nbut only from 1 eV"
    ):
        d.chi([1.5, 3])
    # the range is given in the quantity and unit of the input
    d = Dispersion.from_array([400, 1000] * u.nm, [2, 3])
    with pytest.raises(
        ValueError,
        match="for the wavelengths from 1100 nm to 1200 nm,\nbut only from 400 nm to 1000 nm",
    ):
        d.eps([1100, 1200] * u.nm)
    with pytest.raises(ValueError, match="for the frequency 200 THz,\nbut only from 299.8 THz"):
        d.eps(200 * u.THz)
    # gaps between the ranges of a fallback
    gap = d | Dispersion.from_array([5, 10] * u.eV, [2, 3])
    assert len(gap.energy_ranges) == 2
    with pytest.raises(
        ValueError, match="but only from 1.24 eV to 3.1 eV\nand from 5 eV to 10 eV"
    ):
        gap.eps(4 * u.eV)
    # also through the dielectric of a layer
    layer = Layer("l", "l", dielectric=d)
    with pytest.raises(
        # pint prints the micro sign (U+00B5) or, since 0.26, the Greek mu (U+03BC)
        ValueError,
        match="for the wavelength 1.2 [µμ]m,\nbut only from 0.4 [µμ]m to 1 [µμ]m",
    ):
        layer.get_ref_index(1.2 * u.um)


def test_dispersion_fallback_and_sum():
    table = Dispersion.from_array([1, 2], [2, 3])
    constant = Dispersion.from_ref_index(1.5)
    fallback = table | constant
    np.testing.assert_allclose(fallback.chi([1.5, 10]), [1.5, 1.25])
    # the left one has priority
    np.testing.assert_allclose((constant | table).chi(1.5), [1.25])
    total = table + constant
    np.testing.assert_allclose(total.chi(1.5), [1.5 + 1.25])
    # mixing scalar and tensor dispersions gives tensors
    tensor = Dispersion.from_ref_index([1.5, 2.0])
    np.testing.assert_allclose(np.diagonal((table | tensor).chi([1.5, 10])[1]), [1.25, 1.25, 3])
    np.testing.assert_allclose(np.diagonal((table + tensor).chi(1.5)[0]), [2.75, 2.75, 4.5])
    with pytest.raises(TypeError):
        table | 1.5


def test_dispersion_from_file(tmp_path):
    whitespace = tmp_path / "test.nk"
    whitespace.write_text("# wl n k\n0.4 2.0 0.1\n0.8 1.5 0.0\n")
    comma = tmp_path / "test.csv"
    comma.write_text("wl,n,k\n0.4,2.0,0.1\n0.8,1.5,0.0\n")
    for filename in (whitespace, comma):
        d = Dispersion.from_file(filename)
        np.testing.assert_allclose(d.chi(0.4 * u.um), [(2 + 0.1j) ** 2 - 1])
        assert str(filename) in repr(d)
    d = Dispersion.from_file(whitespace, x_unit="nm")
    assert d.energy_range[0] > 1000


def test_dispersion_from_refractiveindex():
    refractiveindex = pytest.importorskip("refractiveindex")
    url = "https://refractiveindex.info/?shelf=main&book=Cr&page=Johnson"
    d = Dispersion.from_refractiveindex(url=url, num=500)
    material = refractiveindex.RefractiveIndexMaterial(shelf="main", book="Cr", page="Johnson")
    lo, hi = material.get_wl_range(unit="nm")
    np.testing.assert_allclose(sorted(to_energy([lo, hi] * u.nm)), d.energy_range)
    np.testing.assert_allclose(d.eps(800 * u.nm), material.get_epsilon(800), rtol=1e-3)
    assert "refractiveindex.info: main/Cr/Johnson" in d._label()
    assert repr(Dispersion.from_refractiveindex("main", "Cr", "Johnson", num=500)) == repr(d)
    # a page without extinction coefficient
    with pytest.warns(UserWarning, match="kappa = 0"):
        d = Dispersion.from_refractiveindex("main", "SiO2", "Malitson")
    np.testing.assert_allclose(d.eps(800 * u.nm).imag, 0)
    with pytest.raises(ValueError, match="either the url"):
        Dispersion.from_refractiveindex("main", url=url)
    with pytest.raises(ValueError, match="must be given"):
        Dispersion.from_refractiveindex("main", "Cr")


def test_dispersion_repr_depends_on_data():
    a = Dispersion.from_array([1, 2], [2, 3])
    b = Dispersion.from_array([1, 2], [2, 3.1])
    assert repr(a) == repr(Dispersion.from_array([1, 2], [2, 3]))
    assert repr(a) != repr(b)


def test_dispersion_voigt():
    q = Dispersion.from_array(None, 0.01j, kind="voigt")
    np.testing.assert_allclose(q.evaluate(1.0, vacuum=0, eps_xx=np.array([4.0])), [0.04j])
    with pytest.raises(ValueError, match="eps_xx"):
        q.evaluate(1.0, vacuum=0)
    with pytest.raises(ValueError, match="scalar"):
        Dispersion.from_array(None, [0.01, 0.02], kind="voigt")


def test_dispersion_vacuum_contribution():
    # the non-magnetic part includes the vacuum contribution, the gyrotropic not
    eps = Dispersion.from_array(None, 0.1j)
    chi = Dispersion.from_array(None, 0.1j, kind="chi")
    np.testing.assert_allclose(eps.eps(1.0), [0.1j])
    np.testing.assert_allclose(chi.eps(1.0), [1 + 0.1j])
    np.testing.assert_allclose(eps.evaluate(1.0, vacuum=0), [0.1j])
    np.testing.assert_allclose(chi.evaluate(1.0, vacuum=0), [0.1j])
    # sums add the susceptibilities for both parts
    total = Dispersion.from_ref_index(2) + Dispersion.from_ref_index(3)
    np.testing.assert_allclose(total.eps(1.0), [4 + 9 - 1])
    np.testing.assert_allclose((eps + chi).evaluate(1.0, vacuum=0), [0.2j])
    tensor = Dispersion.from_ref_index([2, 3]) + Dispersion.from_ref_index(2)
    np.testing.assert_allclose(np.diagonal(tensor.eps(1.0)[0]), [7, 7, 12])


# ============================================================================
# Dielectric
# ============================================================================


def test_dielectric_unset():
    d = Dielectric(name="dielectric")
    assert not d.is_set
    assert repr(d) == "Dielectric(dielectric: not set)"
    with pytest.raises(ValueError, match="not set"):
        d.get_tensor(1.0)


def test_dielectric_eps_mag_checks():
    with pytest.raises(ValueError, match="refractive index"):
        Dielectric(eps0=Dispersion.from_ref_index(2), eps_mag=Dispersion.from_ref_index(1))
    with pytest.raises(ValueError, match="scalar"):
        Dielectric(eps_mag=Dispersion.from_array(None, [0.1, 0.2], kind="chi"))
    with pytest.raises(TypeError):
        Dielectric(eps0=2.0)
    with pytest.raises(ValueError, match="strain_model"):
        Dielectric(strain_model="foo")


def test_dielectric_gyrotropic_tensor():
    chi0 = np.array([0.1 + 0.01j])
    chim = np.array([0.02 + 0.03j])
    m = 0.7 * np.array([np.sin(0.4) * np.cos(1.1), np.sin(0.4) * np.sin(1.1), np.cos(0.4)])
    eps = gyrotropic_tensor(1 + chi0, chim, m)[0]
    # the magnetic part is antisymmetric and reverses with the magnetization
    reversed_eps = gyrotropic_tensor(1 + chi0, chim, -m)[0]
    np.testing.assert_allclose(reversed_eps, eps.T)
    np.testing.assert_allclose(np.diagonal(eps), 1.1 + 0.01j)
    # the same magnetization in polar coordinates, also as Quantities
    np.testing.assert_allclose(
        gyrotropic_tensor(1 + chi0, chim, (0.7, 0.4, 1.1), polar=True)[0], eps
    )
    np.testing.assert_allclose(
        gyrotropic_tensor(1 + chi0, chim, (0.7, 0.4 * u.rad, 1.1 * u.rad), polar=True)[0], eps
    )
    # same as -i chim eps_ijk m_k
    levi_civita = np.zeros((3, 3, 3))
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        levi_civita[i, j, k], levi_civita[j, i, k] = 1, -1
    expected = (1.1 + 0.01j) * np.eye(3) - 1j * chim[0] * np.einsum("ijk,k->ij", levi_civita, m)
    np.testing.assert_allclose(eps, expected)


def test_dielectric_eps_mag_kinds():
    eps0 = Dispersion.from_ref_index(2 + 0.1j)
    magnetization = (0, 0, 1)  # along z
    tensors = [
        Dielectric(eps0=eps0, eps_mag=Dispersion.from_array(None, 0.02j, kind=kind)).get_tensor(
            1.0, magnetization=magnetization
        )[0]
        for kind in ("eps", "chi")
    ]
    np.testing.assert_allclose(tensors[0], tensors[1])
    np.testing.assert_allclose(tensors[0][0, 1], -1j * 0.02j)
    voigt = Dielectric(eps0=eps0, eps_mag=Dispersion.from_array(None, 0.01, kind="voigt"))
    np.testing.assert_allclose(
        voigt.get_tensor(1.0, magnetization=magnetization)[0, 0, 1], -1j * 0.01 * (2 + 0.1j) ** 2
    )


def test_dielectric_strain_models():
    eps0 = Dispersion.from_array(None, 1 + 0.2j, kind="chi")
    assert Dielectric(eps0=eps0).strain_model == "density"
    none = Dielectric(eps0=eps0, strain_model="none")
    density = Dielectric(eps0=eps0, strain_model="density")
    photoelastic = Dielectric(eps0=eps0, strain_model="photoelastic", deps_dstrain=0.5)
    np.testing.assert_allclose(none.get_tensor(1, 0.1)[0, 0, 0], 2 + 0.2j)
    np.testing.assert_allclose(density.get_tensor(1, 0.1)[0, 0, 0], 1 + (1 + 0.2j) / 1.1)
    np.testing.assert_allclose(photoelastic.get_tensor(1, 0.1)[0, 0, 0], 2.05 + 0.2j)
    with pytest.raises(ValueError, match="strain_model"):
        Dielectric(eps0=eps0, strain_model="linear")
    # the gyrotropic part scales with the density for both strain models
    eps_mag = Dispersion.from_array(None, 0.02j)
    magnetization = (0, 0, 1)  # along z
    for strain_model, scale in [("none", 1), ("density", 1 / 1.1), ("photoelastic", 1 / 1.1)]:
        dielectric = Dielectric(eps0=eps0, eps_mag=eps_mag, strain_model=strain_model)
        np.testing.assert_allclose(
            dielectric.get_tensor(1, 0.1, magnetization)[0, 0, 1], -1j * 0.02j * scale
        )


def test_euler_matrix():
    R = euler_matrix(0.3, 0.7, -1.1)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(R), 1)
    # rotation by theta about x moves the z-axis towards -y
    np.testing.assert_allclose(euler_matrix(np.pi / 2, 0, 0) @ [0, 0, 1], [0, -1, 0], atol=1e-15)
    # psi and phi rotate about z
    np.testing.assert_allclose(euler_matrix(0, 0.4, 0.2), euler_matrix(0, 0.6, 0), atol=1e-15)


def test_dielectric_euler_angles():
    eps0 = Dispersion.from_ref_index([1.5, 2.0])  # uniaxial along the c-axis of the crystal
    dielectric = Dielectric(eps0=eps0, euler_angles=(90, 0, 0))
    np.testing.assert_allclose(dielectric.euler_angles.magnitude, [90, 0, 0])
    # the c-axis is rotated into the surface plane
    np.testing.assert_allclose(
        np.diagonal(dielectric.get_tensor(1.0)[0]), [2.25, 4, 2.25], atol=1e-12
    )
    # the trace is invariant for any rotation
    dielectric.euler_angles = (30, 45, 10) * u.deg
    eps = dielectric.get_tensor(1.0)[0]
    np.testing.assert_allclose(np.trace(eps), 2 * 2.25 + 4)
    np.testing.assert_allclose(eps, eps.T)
    # isotropic dielectrics do not change
    iso = Dielectric(eps0=Dispersion.from_ref_index(2), euler_angles=(30, 45, 10))
    np.testing.assert_allclose(iso.get_tensor(1.0)[0], 4 * np.eye(3), atol=1e-12)
    # the magnetization and the photoelastic change are given in the frame of the sample
    rotated = Dielectric(
        eps0=Dispersion.from_ref_index(2),
        eps_mag=Dispersion.from_array(None, 0.02j),
        strain_model="photoelastic",
        deps_dstrain=[0, 0, 1],
        euler_angles=(90, 0, 0),
    )
    eps = rotated.get_tensor(1.0, 0.1, (0, 0, 1))[0]
    np.testing.assert_allclose(eps[2, 2], 4.1)
    np.testing.assert_allclose(eps[0, 1], -1j * 0.02j / 1.1)
    # the same magnetization along z in polar coordinates
    np.testing.assert_allclose(rotated.get_tensor(1.0, 0.1, (1, 0, 0), polar=True)[0], eps)
    # the rotation is part of the representation and the display
    assert "euler_angles" in repr(dielectric)
    assert repr(dielectric) != repr(Dielectric(eps0=eps0))
    assert "R = Euler (z-x-z): (30°, 45°, 10°)" in str(dielectric)
    text = str(rotated)
    assert "dε/dη = (0, 0, 1) for ε⁰" in text
    assert "s(η) = 1/(1+η) for εᵐ" in text
    # a Dielectric assigned to a layer keeps its rotation
    layer = Layer("l", "l", dielectric=dielectric)
    np.testing.assert_allclose(layer.optical.dielectric.euler_angles.magnitude, [30, 45, 10])
    with pytest.raises(ValueError, match="theta, phi, psi"):
        dielectric.euler_angles = (1, 2)


def test_dielectric_ref_index_and_pen_depth():
    d = Dielectric(eps0=Dispersion.from_ref_index(2 + 0.1j))
    np.testing.assert_allclose(d.get_ref_index(800 * u.nm), [2 + 0.1j])
    zeta = d.get_opt_pen_depth(800 * u.nm)
    np.testing.assert_allclose(zeta.to("nm").magnitude, [800 / (4 * np.pi * 0.1)], rtol=1e-6)
    anisotropic = Dielectric(eps0=Dispersion.from_ref_index([2, 3]))
    assert not anisotropic.is_isotropic
    with pytest.raises(ValueError, match="anisotropic"):
        anisotropic.get_ref_index(1.0)


def test_dielectric_in_parameter_group():
    layer = Layer("l", "l")
    assert not layer.optical.dielectric.is_set
    # a Dispersion sets eps0
    layer.optical.dielectric = Dispersion.from_ref_index(1.5)
    assert layer.optical.dielectric.is_set
    assert layer.optical.dielectric._caller is layer.optical
    # a Dielectric replaces the parameter, which keeps its name and connection
    layer.optical.dielectric = Dielectric(
        eps0=Dispersion.from_ref_index(2), strain_model="density"
    )
    assert layer.optical.dielectric.strain_model == "density"
    assert layer.optical.dielectric.name == "dielectric"
    assert layer.optical.dielectric._caller is layer.optical
    assert layer.get_ref_index(1.0) == 2
    # unset
    layer.optical.dielectric = None
    assert not layer.optical.dielectric.is_set
    with pytest.raises(TypeError, match="from_ref_index"):
        layer.optical.dielectric = 1.5


def test_dielectric_changes_structure_hash():
    layer = Layer("l", "l", thickness=1 * u.nm, dielectric=Dispersion.from_ref_index(1.5))
    S = Structure("s")
    S.add_sub_structure(layer, 2)
    old = S.get_hash(types=["heat"])
    S.get_layer_handle(0).optical.dielectric = Dispersion.from_array([1, 2], [2, 3])
    assert S.get_hash(types=["heat"]) != old


def test_dielectric_deepcopy():
    layer = Layer("l", "l", dielectric=Dispersion.from_ref_index(1.5))
    new = copy.deepcopy(layer)
    new.optical.dielectric = Dispersion.from_ref_index(2)
    assert layer.get_ref_index(1.0) == 1.5


# ============================================================================
# Dielectric from atoms
# ============================================================================


@pytest.fixture(scope="module")
def iron():
    atom = Atom("Fe")
    atom.magnetization = {"r": 1.0, "phi": 90 * u.deg, "gamma": 0 * u.deg}
    return atom


def test_atom_get_susceptibility(iron):
    energy = np.array([700.0, 710.0])
    chi0, chim = iron.get_susceptibility(energy, 7874)
    molar_density = 7874 / 1000 / iron.mass_number_a
    factor = 830.9471 / energy**2
    np.testing.assert_allclose(chi0, -factor * molar_density * iron.get_atomic_form_factor(energy))
    np.testing.assert_allclose(
        chim, factor * molar_density * iron.get_magnetic_form_factor(energy)
    )


def test_layer_dielectric_from_atoms(iron):
    layer = AmorphousLayer("fe", "fe", thickness=1 * u.nm, density=7874 * u.kg / u.m**3, atom=iron)
    assert not layer.optical.dielectric.is_set
    dielectric = layer.dielectric_from_atoms()
    assert dielectric.strain_model == "density"
    assert dielectric.eps_mag is not None
    layer.optical.dielectric = dielectric
    energy = np.array([700.0, 710.0])
    chi0, chim = iron.get_susceptibility(energy, 7874)
    eps = layer.get_dielectric_tensor(energy)
    np.testing.assert_allclose(eps, gyrotropic_tensor(1 + chi0, chim, (1.0, 0, 0)))
    # strained layers have a lower density
    chi0, chim = iron.get_susceptibility(energy, 7874 / 1.01)
    np.testing.assert_allclose(
        layer.get_dielectric_tensor(energy, 0.01),
        gyrotropic_tensor(1 + chi0, chim, (1.0, 0, 0)),
    )
    # the dispersion follows the density of the layer
    layer.density = 5000 * u.kg / u.m**3
    chi0, _ = iron.get_susceptibility(energy, 5000)
    np.testing.assert_allclose(
        layer.get_dielectric_tensor(energy, magnetization=(0, 0, 0))[:, 0, 0], 1 + chi0
    )
    # the atomic form factors are not tabulated in the optical range
    with pytest.raises(ValueError, match="not defined"):
        layer.get_ref_index(800 * u.nm)
    layer.optical.dielectric.eps0 = (
        Dispersion.from_ref_index(2.9 + 3.0j) | layer.dispersion_from_atoms()
    )
    np.testing.assert_allclose(layer.get_ref_index(800 * u.nm), 2.9 + 3.0j)


def test_unit_cell_dispersion_from_atoms(iron):
    oxygen = Atom("O")
    uc = UnitCell("uc", "uc", c_axis=3 * u.angstrom, a_axis=3 * u.angstrom, b_axis=3 * u.angstrom)
    uc.add_atom(iron, 0)
    uc.add_atom(oxygen, 0.5)
    energy = np.array([500.0, 800.0])
    volume = (3e-10) ** 3
    chi_fe, _ = iron.get_susceptibility(energy, iron.mass.to("kg").magnitude / volume)
    chi_o, _ = oxygen.get_susceptibility(energy, oxygen.mass.to("kg").magnitude / volume)
    chi = uc.dispersion_from_atoms().chi(energy, uc.structural.density.magnitude)
    np.testing.assert_allclose(chi, chi_fe + chi_o, rtol=1e-12)


def test_layer_without_atoms():
    with pytest.raises(TypeError, match="no atoms"):
        Layer("l", "l").dispersion_from_atoms()
    with pytest.raises(ValueError, match="Layer 'l'"):
        Layer("l", "l").get_dielectric_tensor(1.0)


# ============================================================================
# display
# ============================================================================


def test_dielectric_display(iron):
    layer = AmorphousLayer("fe", "fe", thickness=1 * u.nm, density=7874 * u.kg / u.m**3, atom=iron)
    assert str(layer.optical.dielectric) == "ε = not set"
    layer.optical.dielectric = layer.dielectric_from_atoms()
    dielectric = layer.optical.dielectric
    dielectric.eps0 = layer.dispersion_from_atoms() | Dispersion.from_ref_index(2.87 + 3.36j)
    text = str(dielectric)
    assert "ε⁰: atomic form factors of Fe" in text
    assert "| n = 2.87 + 3.36i" in text
    assert "magnetic form factors of Fe" in text
    assert "s(η) = 1/(1+η) for ε⁰ and εᵐ" in text
    old_repr = repr(dielectric)
    # values at the display energy, which does not change the hash
    dielectric.display_energy = 800 * u.nm
    assert repr(dielectric) == old_repr
    text = str(dielectric)
    assert "n(800 nm) = 2.87 + 3.36i" in text
    assert "not defined, only from" in text
    dielectric.display_energy = 707
    assert "n(707 eV) = 1 + (0.0002647 + 0.0002143i)" in str(dielectric)
    html = layer.optical._html_table()
    assert '<td rowspan="4">dielectric</td>' in html
    assert "<sup>0</sup>(707 eV)" in html
    assert dielectric._repr_latex_().startswith(r"$\begin{aligned}\varepsilon^0(\text{707 eV})")
    with pytest.raises(pint.DimensionalityError):
        dielectric.display_energy = 1 * u.kg


def test_dielectric_display_tensor():
    layer = Layer("l", "l", dielectric=Dispersion.from_ref_index([1.5, 2.0]))
    layer.optical.dielectric.display_energy = 1.0
    text = str(layer.optical.dielectric)
    assert "ε⁰ₓₓ(1 eV) = 2.25" in text
    assert "ε⁰_zz(1 eV) = 4" in text


# ============================================================================
# operators, further inputs and outputs
# ============================================================================


def test_dispersion_difference_and_scaling():
    a = Dispersion.from_ref_index(2)  # eps = 4
    b = Dispersion.from_array(None, 0.5 + 0.1j, kind="chi")
    np.testing.assert_allclose((a - b).eps(1.0), [4 - 0.5 - 0.1j])
    np.testing.assert_allclose((0.5 * a).eps(1.0), [1 + 0.5 * 3])
    np.testing.assert_allclose((a * 2).eps(1.0), [1 + 2 * 3])
    np.testing.assert_allclose((-b).chi(1.0), [-0.5 - 0.1j])
    np.testing.assert_allclose((a - a).eps(1.0), [1])
    # tensors and the gyrotropic part without vacuum contribution
    tensor = Dispersion.from_ref_index([2, 3])
    np.testing.assert_allclose(np.diagonal((0.5 * tensor).eps(1.0)[0]), [2.5, 2.5, 5])
    np.testing.assert_allclose((2 * b).evaluate(1.0, vacuum=0), [1 + 0.2j])
    assert "-" in repr(a - b) and "2" in repr(2 * b)
    with pytest.raises(TypeError):
        a * b


def test_dispersion_from_delta_beta():
    d = Dispersion.from_array(None, 1e-5 + 2e-6j, kind="delta_beta")
    np.testing.assert_allclose(d.eps(1000), [(1 - 1e-5 + 2e-6j) ** 2])
    d = Dispersion.from_array([100, 200], [[1e-4, 2e-4], [1e-5, 0]], kind="delta_beta")
    np.testing.assert_allclose(np.diagonal(d.eps(200)[0]), [(1 - 1e-5) ** 2] * 2 + [1])


def test_dispersion_from_opt_pen_depth():
    d = Dispersion.from_opt_pen_depth(10 * u.nm, n=2)
    dielectric = Dielectric(eps0=d)
    for wavelength in (400, 800, 1600) * u.nm:
        assert_allclose(dielectric.get_opt_pen_depth(wavelength)[0], 10 * u.nm)
        np.testing.assert_allclose(dielectric.get_ref_index(wavelength).real, 2)
    assert d._label() == "ζ = 10 nm (κ = λ/4πζ), n' = 2"
    with pytest.raises(ValueError, match="positive"):
        Dispersion.from_opt_pen_depth(0)
    with pytest.raises(ValueError, match="cannot be given"):
        Dielectric(eps0=d, eps_mag=d)


def test_dispersion_from_mass_attenuation():
    with pytest.warns(UserWarning, match="delta = 0"):
        d = Dispersion.from_mass_attenuation([1, 2] * u.keV, [100, 50] * u.cm**2 / u.g)
    layer = Layer("l", "l", density=2 * u.g / u.cm**3, dielectric=d)
    assert_allclose(layer.get_mass_attenuation_coefficient(1 * u.keV), 100 * u.cm**2 / u.g)
    assert_allclose(layer.get_absorption_coefficient(1 * u.keV), 200 / u.cm)
    assert layer.get_delta_beta(1500)[0] == pytest.approx(0)
    with pytest.raises(ValueError, match="density"):
        d.eps(1000)


def test_dielectric_outputs():
    layer = Layer(
        "l", "l", density=2 * u.g / u.cm**3, dielectric=Dispersion.from_ref_index(1 - 1e-5 + 1e-6j)
    )
    delta, beta = layer.get_delta_beta(8 * u.keV)
    assert delta == pytest.approx(1e-5)
    assert beta == pytest.approx(1e-6)
    wavelength = (8 * u.keV).to("m", "sp")
    assert_allclose(layer.get_absorption_coefficient(8 * u.keV), 4 * np.pi * 1e-6 / wavelength)
    assert_allclose(
        1 / layer.get_absorption_coefficient(8 * u.keV), layer.get_opt_pen_depth(8 * u.keV)
    )
    assert_allclose(
        layer.get_mass_attenuation_coefficient(8 * u.keV),
        layer.get_absorption_coefficient(8 * u.keV) / (2 * u.g / u.cm**3),
    )
    assert_allclose(layer.get_critical_angle(8 * u.keV), np.sqrt(2e-5) * u.rad, rtol=1e-4)
    # no total reflection for n' >= 1
    layer.optical.dielectric = Dispersion.from_ref_index(1.5)
    assert layer.get_critical_angle(1.0).magnitude == 0
    # arrays
    assert layer.get_delta_beta([1, 2])[0].shape == (2,)


def test_layer_set_dielectric_from_atoms_and_form_factor(iron):
    layer = AmorphousLayer("fe", "fe", thickness=1 * u.nm, density=7874 * u.kg / u.m**3, atom=iron)
    layer.set_dielectric_from_atoms(fallback=Dispersion.from_ref_index(2.9 + 3.4j))
    assert layer.optical.dielectric.strain_model == "density"
    assert layer.optical.dielectric.eps_mag is not None
    np.testing.assert_allclose(layer.get_ref_index(800 * u.nm), 2.9 + 3.4j)
    # the effective form factor equals the one of the atoms
    np.testing.assert_allclose(
        layer.get_form_factor([700, 710] * u.eV), iron.get_atomic_form_factor([700, 710])
    )
    np.testing.assert_allclose(layer.get_form_factor(705), iron.get_atomic_form_factor(705))
    with pytest.raises(TypeError, match="no atoms"):
        Layer("l", "l").get_form_factor(700)


def test_dispersion_from_form_factors(iron, tmp_path):
    from pathlib import Path

    # the same file gives the same dispersion as the atom
    filename = Path(__file__).parent / "data" / "Fe.cf"
    atom = Atom("Fe", atomic_form_factor_path=str(filename))
    layer = AmorphousLayer("fe", "fe", thickness=1 * u.nm, density=7874 * u.kg / u.m**3, atom=atom)
    from_atoms = layer.dispersion_from_atoms()
    from_file = Dispersion.from_form_factors(filename, molar_mass=atom.mass_number_a)
    energy = [700, 705, 710] * u.eV
    np.testing.assert_allclose(from_file.eps(energy, 7874), from_atoms.eps(energy, 7874))
    assert "Fe.cf" in from_file._label() and "g/mol" in repr(from_file)
    # as kind of from_file and from_array
    np.testing.assert_allclose(
        Dispersion.from_file(filename, kind="f", molar_mass=55.845 * u.g / u.mol).eps(
            energy, 7874
        ),
        from_file.eps(energy, 7874),
        rtol=1e-4,
    )
    f = atom.get_atomic_form_factor(energy)
    np.testing.assert_allclose(
        Dispersion.from_array(energy, f, kind="f", molar_mass=atom.mass_number_a).eps(
            energy, 7874
        ),
        from_atoms.eps(energy, 7874),
    )
    # magnetic form factors are the gyrotropic part
    m = iron.get_magnetic_form_factor(energy)
    eps_mag = Dispersion.from_array(energy, m, kind="m", molar_mass=iron.mass_number_a)
    layer.optical.dielectric = Dielectric(eps0=from_file, eps_mag=eps_mag)
    _, chim = iron.get_susceptibility(energy, 7874)
    np.testing.assert_allclose(
        layer.get_dielectric_tensor(energy, magnetization=(0, 0, 1))[:, 0, 1], -1j * chim
    )
    # invalid entries of Henke et al. and wrong parts
    henke = tmp_path / "henke.nff"
    henke.write_text("E(eV)\tf1\tf2\n10\t-9999.\t1\n20\t5\t1\n30\t6\t1\n")
    d = Dispersion.from_form_factors(henke, molar_mass=10)
    assert d.energy_range == (20, 30)
    with pytest.raises(ValueError, match="molar_mass"):
        Dispersion.from_form_factors(henke, molar_mass=None)
    with pytest.raises(ValueError, match="non-magnetic"):
        Dielectric(eps0=eps_mag)
    with pytest.raises(ValueError, match="atomic form factors"):
        Dielectric(eps0=from_file, eps_mag=from_file)


def test_delta_beta_mag(iron):
    layer = AmorphousLayer("fe", "fe", thickness=1 * u.nm, density=7874 * u.kg / u.m**3, atom=iron)
    layer.set_dielectric_from_atoms()
    energy = [700, 707, 710] * u.eV
    # proportional to the magnetic form factors m = m1 - i m2 as delta and beta to f1 and f2
    delta, beta = layer.get_delta_beta(energy)
    delta_mag, beta_mag = layer.get_delta_beta_mag(energy)
    chi0, chim = iron.get_susceptibility(energy, 7874)
    np.testing.assert_allclose(delta_mag, chim.real / 2)
    np.testing.assert_allclose(beta_mag, -chim.imag / 2)
    m = iron.get_magnetic_form_factor(energy)
    f = iron.get_atomic_form_factor(energy)
    # to first order as the definition of delta_mag and beta_mag
    np.testing.assert_allclose(delta_mag / m.real, -chi0.real / 2 / f.real)
    np.testing.assert_allclose(beta_mag / -m.imag, chi0.imag / 2 / -f.imag)
    np.testing.assert_allclose(delta, -chi0.real / 2, rtol=1e-2)
    # the density scaling of the strain
    delta_mag_strained, _ = layer.get_delta_beta_mag(energy, strain=0.1)
    np.testing.assert_allclose(delta_mag_strained, delta_mag / 1.1)
    # the same as input of the magnetic part
    layer.optical.dielectric.eps_mag = Dispersion.from_array(
        energy, delta_mag + 1j * beta_mag, kind="delta_beta"
    )
    np.testing.assert_allclose(layer.get_delta_beta_mag(energy), (delta_mag, beta_mag))
    np.testing.assert_allclose(
        layer.get_dielectric_tensor(energy, magnetization=(0, 0, 1))[:, 0, 1], -1j * chim
    )
    # the non-magnetic part with kind delta_beta is unchanged
    d = Dispersion.from_array(None, 1e-5 + 2e-6j, kind="delta_beta")
    np.testing.assert_allclose(d.eps(1000), [(1 - 1e-5 + 2e-6j) ** 2])
    # without magnetic part and without effect of a rotated anisotropic part
    rotated = Layer(
        "l",
        "l",
        dielectric=Dielectric(eps0=Dispersion.from_ref_index([1.5, 2]), euler_angles=(30, 20, 0)),
    )
    np.testing.assert_allclose(rotated.get_delta_beta_mag(1.0), (0, 0))
    rotated.optical.dielectric.eps_mag = Dispersion.from_array(
        None, 1e-3 + 2e-3j, kind="delta_beta"
    )
    np.testing.assert_allclose(rotated.get_delta_beta_mag(1.0), (1e-3, 2e-3))


@pytest.mark.parametrize(
    "strain_model, kwargs",
    [("none", {}), ("density", {}), ("photoelastic", {"deps_dstrain": 0.3 + 0.01j})],
)
def test_dielectric_tensor_for_array_of_strains(strain_model, kwargs):
    """an array of strains yields the tensors of the single strains, e.g. of
    a layer for all delays"""
    dielectric = Dielectric(
        eps0=Dispersion.from_ref_index(2.4 + 0.1j),
        eps_mag=Dispersion.from_array(None, 0.02j, kind="voigt"),
        strain_model=strain_model,
        **kwargs,
    )
    energy = np.r_[1.8, 2.0, 2.5]
    strains = np.array([[-1e-3, 0.0], [2e-3, 5e-3]])
    magnetization = (1.0, np.pi / 3, np.pi / 4)
    eps = dielectric.get_tensor(energy, strains, magnetization)
    assert eps.shape == strains.shape + (3, 3, 3)
    for index in np.ndindex(strains.shape):
        np.testing.assert_allclose(
            eps[index], dielectric.get_tensor(energy, strains[index], magnetization), rtol=1e-14
        )
