# Reference results of external packages

The files in this folder are the outputs of external packages for a few layer
stacks, which test the multilayer formalisms of udkm1Dsim in
`tests/test_references.py`. Only the outputs are part of udkm1Dsim, not the
calculations, which were run outside of this repository by a separate script
with these packages:

- [refnx](https://github.com/refnx/refnx) 0.1.67 (BSD-3-Clause): X-ray reflectivity by the Abeles matrices with the roughness of Névot and Croce
- [tmm](https://github.com/sbyrnes321/tmm) 0.2.0 (MIT): complex reflection and transmission coefficients of isotropic stacks
- [GenX](https://github.com/aglavic/genx) 3.8.14 (GPL-3.0): the Parratt recursion and the resonant magnetic X-ray reflectivity by the recursion of Stepanov and Sinha (`xrmr`)
- [PyGTM](https://github.com/pyMatJ/pyGTM) 7a228b7 (2023-11-16) (GPL-3.0): the generalized transfer matrices of Passler and Paarmann for anisotropic layers
- [PyLlama](https://github.com/VignoliniLab/PyLlama) e54fdb5 (2022-05-03) (GPL-3.0): the scattering matrices of the Berreman matrix of Bay et al. for arbitrary dielectric tensors
- [Dyna](https://gitlab.com/dynadevgroup/Dyna) 6.15.0, a8bc11a (2026-10-22) (GPL-2.0-or-later): the exact Berreman formalism with the scattering-matrix recursion of Stepanov, also for gyrotropic dielectric tensors
- [MAG_sl](https://x-server.gmca.aps.anl.gov/MAG_sl.html) of the X-ray Server of S. Stepanov, version 99_4.5: resonant magnetic X-ray reflectivity for σ, π, and circular incidence without analyzer, requested by its automation interface
- [GID_sl](https://x-server.gmca.aps.anl.gov/GID_sl.html) of the X-ray Server of S. Stepanov, version M7: exact dynamical X-ray diffraction, for the Bragg curve of Si (004) of `XrayDyn` in `xray_bragg_si004.npz`

Every `.npz` file holds the grazing angles `theta` (deg), the dielectric
tensors `eps` of all layers in the sample frame of Elzo, as udkm1Dsim uses
them, the outputs of the packages, and a JSON string `meta` with the inputs,
the package versions, and the agreement of the packages with each other.

Conventions, which follow from the comparison:

- Dyna and udkm1Dsim agree in their (σ, π) basis and phases, so that their
  complex Jones matrices of the reflection are compared directly, but the
  transmission of Dyna is normalized to the eigenmodes of the substrate
  instead of the electric field and is not compared.
- tmm uses the same sign of the π polarization as udkm1Dsim. Its
  transmission is given at the top of the substrate, while udkm1Dsim gives it
  after the thickness of the substrate layer, which is zero in the tests.
- PyGTM, PyLlama, and GenX have their plane of incidence in xz instead of yz.
  For GenX, (x, y, z) are (y, x, -z) of udkm1Dsim, as follows from the
  comparison with Dyna. GenX treats a magnetization along its y, i.e. a
  transverse one, separately due to a singularity, which agrees with Dyna to
  2e-6 only.
- PyGTM rotates the dielectric tensor of a crystal into the lab frame by
  `inv(R) eps R` of its Euler matrix `R`, i.e. passively. Hence, its tensors
  are stored after the rotation, and the tests use these tensors, while the
  conversion of its Euler angles, see `euler_matrix()`, is tested separately.
- MAG_sl has x along the beam, y along σ, and z along the external normal, so
  that its magnetization is (-m_y, m_x, m_z) of udkm1Dsim, as follows from
  the circular dichroism compared with Dyna. Its resonant amplitudes are set
  to F11 = a + d, F1T = a - d, and F10 = a, so that its dielectric tensor
  equals the one of udkm1Dsim, see its `.TBL` files. Its output has six
  significant digits.
- GID_sl calculates a 200 µm thick Si layer on its Si substrate from the
  database, whose susceptibilities χ₀ and χ_h are given by the Si unit cell
  of udkm1Dsim with a Debye-Waller factor ⟨u²⟩ = B / (8π²) for B = 0.46 Å², so
  that both codes use the same susceptibilities and the substrate is not
  reached. The Darwin layers of `XrayDyn` agree with the exact dynamical
  theory of GID_sl to 3e-3 in the integrated reflectivity and 1e-4 at the
  peak, but deviate by up to 5 % of the peak at the steep flanks.
- The sorting of the modes of PyLlama fails for some angles of a tilted optic
  axis and of a magnetization with a polar component, where its results are
  NaN. For the X-ray cases with a mixed direction of the magnetization it
  fails for all angles, so that they are compared with GenX only. These cases
  test the cross terms of the magnetization components, which go beyond the
  linear MOKE form.

Largest deviation between the packages relative to the largest value of each
curve, for intensities, which do not depend on their phase conventions:

| case | agreement of the packages |
| :--- | :--- |
| `birefringent_inplane` | R_ss PyGTM vs Dyna 1e-15; R_pp PyGTM vs Dyna 2e-15; R_cross PyGTM vs Dyna 7e-13; R PyLlama vs Dyna 1e-14 |
| `birefringent_tilted` | R_ss PyGTM vs Dyna 4e-06; R_pp PyGTM vs Dyna 2e-06; R_cross PyGTM vs Dyna 1e-03; R PyLlama vs Dyna 2e-15; PyLlama failed angles 3 |
| `magnetic_optics_inplane` | R PyLlama vs Dyna 9e-16 |
| `magnetic_optics_longitudinal` | R PyLlama vs Dyna 8e-16 |
| `magnetic_optics_oblique` | R PyLlama vs Dyna 1e-15; PyLlama failed angles 13 |
| `magnetic_optics_polar` | R PyLlama vs Dyna 2e-11; PyLlama failed angles 9 |
| `magnetic_optics_transverse` | R PyLlama vs Dyna 5e-16 |
| `magnetic_xray_inplane` | R GenX XRMR vs Dyna 5e-14; PyLlama failed all angles |
| `magnetic_xray_longitudinal` | R GenX XRMR vs Dyna 5e-14; R PyLlama vs Dyna 1e-15; R_sigma MAG_sl vs Dyna 5e-07; R_pi MAG_sl vs Dyna 7e-07; R_circ MAG_sl vs Dyna 7e-07 |
| `magnetic_xray_oblique` | R GenX XRMR vs Dyna 6e-12; PyLlama failed all angles |
| `magnetic_xray_polar` | R GenX XRMR vs Dyna 5e-14; R PyLlama vs Dyna 2e-09; PyLlama failed angles 17; R_sigma MAG_sl vs Dyna 8e-07; R_pi MAG_sl vs Dyna 6e-07; R_circ MAG_sl vs Dyna 7e-07 |
| `magnetic_xray_transverse` | R GenX XRMR vs Dyna 2e-06; R PyLlama vs Dyna 5e-16; R_sigma MAG_sl vs Dyna 7e-07; R_pi MAG_sl vs Dyna 6e-07; R_circ MAG_sl vs Dyna 7e-07 |
| `optics_isotropic` | R_s tmm vs Dyna 2e-15; R_p tmm vs Dyna 1e-15; R_ss PyGTM vs Dyna 2e-15; R_pp PyGTM vs Dyna 1e-15; R_s GenX Parratt vs Dyna 2e-15; R PyLlama vs Dyna 2e-15 |
| `xray_isotropic` | R_s tmm vs Dyna 7e-12; R_p tmm vs Dyna 7e-12; R_s GenX Parratt vs Dyna 2e-15; R GenX XRMR vs Dyna 9e-12; R PyLlama vs Dyna 7e-12; R refnx vs Dyna 9e-12; R refnx vs tmm 1e-11 |
| `xray_rough` | R_s GenX Parratt vs Dyna 4e-15; R refnx vs Dyna 9e-12 |

The deviation of PyGTM for the tilted optic axis is the same for Dyna,
PyLlama, and udkm1Dsim, which agree with each other to 1e-15.
