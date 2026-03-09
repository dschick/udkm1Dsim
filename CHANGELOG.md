# Changelog

All notable changes to the udkm1Dsim project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- introduce `nox` automatic testing tool for local testing of multiple python and/or package versions in virtual environments (PR #171)
- add unit to mf_exch_coupling (https://github.com/dschick/udkm1Dsim/pull/115/commits/dd134ae4e7286ce792a8d53044768879c9d00a72)

### Changed

- `Layer.check_input()` always returns `float` (https://github.com/dschick/udkm1Dsim/pull/115/commits/f71eb1cea3a47665eb63aa6e9219e61758dc2e16)
- change GitHub CI Matrix to python 3.9--3.12 (https://github.com/dschick/udkm1Dsim/pull/115/commits/97b17d8ff5295c68119aa2da9b691db3dcfb6b18)
- change import of `tqdm` to `auto` module to work with scripts and notebooks (https://github.com/dschick/udkm1Dsim/pull/115/commits/0c8cecc6b0b2a843334bc293a0e811640d6d8659)

### Deprecated

### Removed

### Fixed

- raise `TypeError` in `Magnetization` class for `UnitCell`s in sample (ttps://github.com/dschick/udkm1Dsim/pull/115/commits/7e92efbb345ea6a84dc11cb33ba9f5132aeaf255)

### Security

## [2.0.5] 2026-03-04

### Fixed

- fix wrong strain dependence of `Atom`s in `UnitCell` when calling `add_atom` method (#172 and PR #173)
