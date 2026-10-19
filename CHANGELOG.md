# Changelog

All notable changes to the **udkm1Dsim** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Vacuum added by [#204] closing [#202]
- example for XrayKin added by [#204]
- superstrate introduced by [#204] closing [#134]
- enable automatic numerical integration of layer properties if sympy fails ([#198]) replacing [#110] and closing issue [#109] and [#108]
- Parameter and ParamterGroup classes organizing paramters closing [#162] by [#215]
- temperature-dependent parameter functionals are calculated on demand and are cached by [#215]; the Numba solvers work on the sympy expressions instead of compiled material functions, closing [#206] and [#208]
- VectorParameter enabling input and output in cartesian or polar coordinates closing [#138] by [#215]
- `XrayDynMag.set_polarization` accepts elliptical incoming and outgoing polarizations to set both at the same time ([#217])
- `simulations.solvers` sub-module for the ODE/PDE solvers of `Heat` and `Phonon` with selectable `backend` by [#216] closing [#205]
- Numba backend for numerical phonons `PhononNum(backend='numba')` by [#220]
- `__repr__` and `_repr_html_` for all structures and simulations rendering their properties as tables in Jupyter and in the documentation
- `Structure.check_num_sub_systems()` to check the number of sub-systems of the relevant thermal parameters of all layers

[#185]: https://github.com/dschick/udkm1Dsim/pull/185
[#188]: https://github.com/dschick/udkm1Dsim/pull/188
[#197]: https://github.com/dschick/udkm1Dsim/pull/197
[#198]: https://github.com/dschick/udkm1Dsim/pull/198
[#110]: https://github.com/dschick/udkm1Dsim/pull/110
[#109]: https://github.com/dschick/udkm1Dsim/issue/109
[#108]: https://github.com/dschick/udkm1Dsim/issue/108
[#204]: https://github.com/dschick/udkm1Dsim/pull/204
[#202]: https://github.com/dschick/udkm1Dsim/issue/202
[#134]: https://github.com/dschick/udkm1Dsim/issue/134
[#162]: https://github.com/dschick/udkm1Dsim/discussions/162
[#138]: https://github.com/dschick/udkm1Dsim/issue/138
[#206]: https://github.com/dschick/udkm1Dsim/issue/206
[#208]: https://github.com/dschick/udkm1Dsim/issue/208
[#215]: https://github.com/dschick/udkm1Dsim/pull/215
[#217]: https://github.com/dschick/udkm1Dsim/pull/217
[#205]: https://github.com/dschick/udkm1Dsim/issue/205
[#216]: https://github.com/dschick/udkm1Dsim/pull/216
[#220]: https://github.com/dschick/udkm1Dsim/pull/220

### Changed

- substrates can be a single layer
- Vacuum is default superstrate and substrate
- `XrayDynMag` polarization setters check the consistency of the number of incoming and outgoing elliptical polarizations ([#217])
- layer properties are organized in ParameterGroups (`structural`, `thermal`, `elastic`, `optical`, `magnetic`, and `lattice` for `UnitCell`), the layer shortcuts such as `layer.heat_capacity` remain ([#215])
- the magnetization of a layer is a `VectorParameter`, e.g. `layer.magnetic.magnetization.polar` replaces `layer.magnetization['phi']` ([#215])
- all internal calls of `Structure.get_layer_property_vector()` use explicit ParameterGroup paths, e.g. `'thermal.heat_capacity.integral'`, closing [#214]
- `Layer.get_property_dict()` returns the `Parameter` objects keyed by their path, which changes all hashes of cached simulation results
- `Layer.mag_moment` is returned in A m² instead of Bohr magnetons
- `LLB.calc_Brillouin()` takes `mf_exch_coupling` as float in J as documented
- `__str__` methods of structures and simulations are replaced by `__repr__`, simulations show their properties before the structure
- examples display objects via their HTML representation instead of `print()`
- `Structure.num_sub_systems` and `Layer.num_sub_systems` are derived from the thermal parameters of the layers and cannot be set anymore; the consistency of the number of sub-systems is checked by `Heat` and `Phonon` for their relevant parameters when a simulation starts, instead of printing warnings on every assignment

### Deprecated

### Removed

### Fixed

- bug in energy spectrum calculation of analytical phonons ([#212]) closing issue [#84]
- bug for multiple elliptical polarizations in `XrayDynMag.inhomogeneous_reflectivity` ([#217]) closing issue [#196]
- `XrayDynMag.get_hash` includes the elliptical polarization vectors ([#217])
- changing a lattice axis of a `UnitCell` updates its area, volume, density, and elastic parameters, while its mass is given by its atoms
- mean-field magnetization of temperatures just below the Curie temperature was zero, causing divisions by zero in the LLB equations, and could be non-zero above it

[#212]: https://github.com/dschick/udkm1Dsim/pull/212
[#84]: https://github.com/dschick/udkm1Dsim/issue/84
[#196]: https://github.com/dschick/udkm1Dsim/issue/196
[#214]: https://github.com/dschick/udkm1Dsim/issue/214

### Security

## [2.4.1] 2026-09-10

### Added

- links to the API from the exmaples ([#185])
- benchmark test via codspeed.io ([#188])
- add example gallery view to docs via myst-sphinx-gallery ([#197])

[#185]: https://github.com/dschick/udkm1Dsim/pull/185
[#188]: https://github.com/dschick/udkm1Dsim/pull/188
[#197]: https://github.com/dschick/udkm1Dsim/pull/197

### Changed

- publication list with new bibliography style and reverse ordering ([#195])
- restructured documentation ([#199])
- use pyproject.toml instead of setup.py and use hatchling as build backend ([#210])
- replace flake8 with ruff for linting and formatting ([#211])

[#195]: https://github.com/dschick/udkm1Dsim/pull/195
[#199]: https://github.com/dschick/udkm1Dsim/pull/199
[#210]: https://github.com/dschick/udkm1Dsim/pull/210
[#211]: https://github.com/dschick/udkm1Dsim/pull/211

### Removed

- drop MATLAB support for heat diffusion calculations ([#190] and PR[#191])

[#190]: https://github.com/dschick/udkm1Dsim/issues/190
[#191]: https://github.com/dschick/udkm1Dsim/pull/191

### Fixed

- matplotlib.cm deprecation in Matplotlib 3.11 ([#794da43])

[794da43]: https://github.com/dschick/udkm1Dsim/commit/794da43

## [2.4.0] 2026-07-09
### Added

- calculation of energy maps and energy flux maps from heat diffusion ([#183])
- include variable/elliptical polarization in XrayDynMag ([#180])

[#183]: https://github.com/dschick/udkm1Dsim/issues/183
[#180]: https://github.com/dschick/udkm1Dsim/pull/180

## [2.3.0] 2026-03-16
### Added

- installation instruction of new conda package, see [udkm1dsim-feedstock](https://github.com/conda-forge/udkm1dsim-feedstock) for more details
- temperature-dependence of Debye-Waller factor with example ([#61] and PR[#177])
- fitting example in the docs ([#59] and PR[#179])
- s-polarization in multilayer absorption formalism ([#28] and PR[#181])

### Fixed

- errors in building the docs ([33d3a3c])
- formatting of pint quantities in __str__() methods of structure and simulation objects ([#175] and PR[#176])
- math of Debye-Waller factor (PR[#178])
- heat diffusion calculation is broken when progressbar is disabled ([#167] and [8a5e32d])

[33d3a3c]: https://github.com/dschick/udkm1Dsim/commit/33d3a3c
[#175]: https://github.com/dschick/udkm1Dsim/issues/175
[#176]: https://github.com/dschick/udkm1Dsim/pull/176
[#178]: https://github.com/dschick/udkm1Dsim/pull/178
[#61]: https://github.com/dschick/udkm1Dsim/issues/61
[#177]: https://github.com/dschick/udkm1Dsim/pull/177
[#59]: https://github.com/dschick/udkm1Dsim/issues/59
[#179]: https://github.com/dschick/udkm1Dsim/pull/179
[#167]: https://github.com/dschick/udkm1Dsim/issues/167
[8a5e32d]: https://github.com/dschick/udkm1Dsim/commit/8a5e32d
[#181]: https://github.com/dschick/udkm1Dsim/pull/181
[#28]: https://github.com/dschick/udkm1Dsim/issues/28

## [2.2.0] 2026-03-10

### Added

- compatibility for numpy>=2.0.0 and newer python version >= 3.10 and <= 3.14 ([#157] and PR[#174])

[#157]: https://github.com/dschick/udkm1Dsim/issues/157
[#174]: https://github.com/dschick/udkm1Dsim/pull/174

## [2.1.0] 2026-03-09

### Added

- introduce `nox` automatic testing tool for local testing of multiple python and/or package versions in virtual environments (PR[#171])
- add unit to mf_exch_coupling ([dd134ae])
- improve testing coverage to > 80% (PR [#115])

### Changed

- `Layer.check_input()` always returns `float` ([f71eb1c])
- change GitHub CI Matrix to python 3.9--3.12 ([97b17d8])
- change import of `tqdm` to `auto` module to work with scripts and notebooks ([0c8cecc])

### Fixed

- raise `TypeError` in `Magnetization` class for `UnitCell`s in sample ([7e92efb])

[#171]: https://github.com/dschick/udkm1Dsim/pull/171
[dd134ae]: https://github.com/dschick/udkm1Dsim/commit/dd134ae
[#115]: https://github.com/dschick/udkm1Dsim/pull/115
[f71eb1c]: https://github.com/dschick/udkm1Dsim/commit/f71eb1c
[97b17d8]: https://github.com/dschick/udkm1Dsim/commit/97b17d8
[0c8cecc]: https://github.com/dschick/udkm1Dsim/commit/0c8cecc
[7e92efb]: https://github.com/dschick/udkm1Dsim/commit/7e92efb

## [2.0.5] 2026-03-04

### Fixed

- fix wrong strain dependence of `Atom`s in `UnitCell` when calling `add_atom` method ([#172] and PR[#173])

[#172]: https://github.com/dschick/udkm1Dsim/issues/172
[#173]: https://github.com/dschick/udkm1Dsim/pull/173
