# Changelog

All notable changes to the **udkm1Dsim** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- installation instruction of new conda package, see [udkm1dsim-feedstock](https://github.com/conda-forge/udkm1dsim-feedstock) for more details
- temperature-dependence of Debye-Waller factor with example ([#61] and PR[#177])
- fitting example in the docs ([#59] and PR[#179])
- s-polarization in multilayer absorption formalism ([#28] and PR[#181])

### Changed

### Deprecated

### Removed

### Fixed

- errors in building the docs ([33d3a3c])
- formatting of pint quantities in __str__() methods of structure and simulation objects ([#175] and PR[#176])
- math of Debye-Waller factor (PR[#178])
- heat diffusion calculation is broken when progressbar is disabled (#167 and 8a5e32d)

### Security

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
