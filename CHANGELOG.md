# Changelog

All notable changes to the **udkm1Dsim** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **breaking:** the argument-free getters of the structures are read-only properties: `Structure.num_sub_structures`, `num_layers`, `num_unique_layers`, `unique_layers`, `layer_vectors`, `positions_per_unique_layer`, `layer_distances`, and `interface_distances` instead of `get_number_of_sub_structures()`, `get_number_of_layers()`, `get_number_of_unique_layers()`, `get_unique_layers()`, `get_layer_vectors()`, `get_all_positions_per_unique_layer()`, `get_distances_of_layers()`, and `get_distances_of_interfaces()`, as well as `UnitCell.atom_ids` instead of `get_atom_ids()`, where the distances are always returned as Quantities, while the `get_*()` methods with arguments are kept, e.g. `get_layer_property_vector()` and `get_atom_positions()`

## [3.0.0a3] 2026-11-13

### Added

- `Result` as dict-like container of the results of the simulations, which are accessed by name, e.g. `result["temp_map"]` or `result.temp_map`, or as dicts `result.data` and `result.coords`, as plain numpy arrays with their coordinates, e.g. `delays`, `distances`, `energy`, `qz`, and `theta`, the names of their dimensions, their units, e.g. by `result.quantity("temp_map")` or converted by `result.quantity("delays", "ps")`, and the metadata of the simulation, i.e. its class, the version, the hash, the sample structure, and its parameters, see [#166]
- short aliases of the entries of the results, e.g. `result.R` for `result.reflectivity`, `R_phi`, `T`, `jones_R`, `rot_R`, and `ell_R` of the scattering results, `A`, `B`, `Xi`, `omega`, and `E` of `PhononAna`, and `mf_mag_map` of the mean-field magnetization, which are listed by `Result.aliases` and in the table of a result, but not by its `keys()`, and saved to its HDF5 file
- `Result.save()` and `Result.load()` for HDF5 files with the data and coordinates as datasets, the 1D coordinates as dimension scales, and the units and metadata as attributes, which can be read by any HDF5 viewer, and `Result.to_xarray()`, see [#165] and [#166]
- `Simulation.new_result()`, `Simulation.new_map_result()`, and `Scattering.new_scattering_result()` for results with the metadata of the simulation and the coordinates of the maps or of the experiment, which are shared by all simulations
- the maps of the upstream simulations can be passed on as their `Result`, e.g. `get_strain_map(delays, temp)` for the result `temp` of `get_temp_map()`, where `delta_temp_map` is optional, see `Phonon.parse_temp_maps()` and `MultilayerScattering.parse_result_maps()`, and whose coordinates are passed on, e.g. the `delays` to the inhomogeneous scattering results, while the metadata `polar` of a magnetization map is used by the multilayer formalisms
- the `temp_map` of `get_mean_field_mag_map()` can be the result of `get_temp_map()`, whose `temp_sub_system` is used, and the result of `get_temp_map()` holds the checked `init_temp`
- `Heat.get_energy_map()` and `Heat.get_energy_flux_map()` for the result of `get_temp_map()` with the total, coupling, and diffusion energy flux as named entries
- `calc_equilibrium_magnetization()`, `calc_energy_per_eigenmode()`, and `calc_temperature_after_delta_excitation()` with plain arrays as twins of the `get_*()` methods, where the result of `get_equilibrium_magnetization()` can be passed on as `init_magnetization`
- `calc_homogeneous_jones_matrices()`, `calc_homogeneous_reflectivity_transmissivity()`, `calc_homogeneous_rotation_ellipticity()`, `calc_inhomogeneous_reflectivity_transmissivity()`, and `calc_inhomogeneous_rotation_ellipticity()` of the multilayer formalisms, as well as `calc_homogeneous_reflectivity()` of `XrayKin` and `XrayDyn` and `calc_inhomogeneous_reflectivity()` of `XrayDyn`, with plain arrays without the cache and the metadata of a `Result`, e.g. for fits, as used in the data fitting example
- the gallery section basics with the new example on the simulations and their results, i.e. the naming convention, the `Result`, its HDF5 files, passing results on to the following simulations, and a table of the entries and aliases of all results, which is generated from the code, which replaces the section structures
- `h5py` as dependency and `xarray` in the test dependencies

### Changed

- **breaking:** the methods of the simulations are named by what they return: the `get_*()` methods return a `Result`, which cannot be unpacked, e.g. `get_temp_map()`, `get_strain_map()`, `get_energy_per_eigenmode()`, `get_temperature_after_delta_excitation()`, `get_magnetization_map()`, `get_sublattice_magnetization_map()`, `get_equilibrium_magnetization()`, `get_mean_field_mag_map()`, and `get_sublattice_mean_field_mag_map()`, as well as the `homogeneous_*()` and `inhomogeneous_*()` methods of all scattering formalisms, while the `calc_*()` methods return plain arrays; the `get_*()` methods of the structures, layers, and atoms keep returning plain values
- **breaking:** the entries of the results have descriptive names, e.g. `reflectivity`, `reflectivity_phi`, `transmissivity`, and `transmissivity_phi`, `jones_reflection`, or `reflection_rotation` and `reflection_ellipticity` in (deg) instead of tuples of Quantities, `coefficients_A` and `coefficients_B` of `PhononAna`, and `sub_structure_fields` and `sub_structure_matrices` of `XrayKin` and `XrayDyn`
- **breaking:** `solve_eigenproblem()` of `PhononAna` is replaced by `get_eigenproblem()` with a `Result` and `calc_eigenproblem()` with the eigenvectors and eigenfrequencies
- **breaking:** the computing helpers of the simulations are named `calc_*()` instead of `get_*()`, as they return plain arrays: `calc_all_strains_per_unique_layer()` and `calc_reduced_strains_per_unique_layer()` of `Phonon`, `calc_uc_atomic_form_factors()`, `calc_uc_structure_factor()`, `calc_interference_function()`, and `calc_Ep()` of `XrayKin`, `calc_uc_ref_trans_matrix()`, `calc_atom_ref_trans_matrix()`, `calc_atom_reflection_factor()`, `calc_atom_transmission_factor()`, `calc_atom_phase_matrix()`, and `calc_atom_phase_factor()` of `XrayDyn`, `calc_polarization_factor()` of `Scattering`, `calc_layer_dielectric_tensor()`, `calc_homogeneous_layers()`, and `calc_inhomogeneous_layers()` of `MultilayerScattering`, `calc_layer_ref_index()` of `Parratt`, and `calc_directional_exchange_stiffnesses()`, `calc_interlayer_exchange_stiffnesses()`, `calc_magnetic_parameters()`, and `calc_initial_state()` of `Magnetization`, while the cache of `XrayDyn` is private as `_get_all_ref_trans_matrices()`
- **breaking:** the former `calc_inhomogeneous_reflectivity()` of `XrayDyn` for a single delay is renamed to `calc_delay_reflectivity()`, as `calc_inhomogeneous_reflectivity()` is now the plain-array twin of `inhomogeneous_reflectivity()` for all delays
- the default absolute tolerance `atol` of the `ode_options` of `PhononNum` is 1 pm instead of 1 µm for the shifts of the layers, which are of the order of picometers, so that the error of the solvers is no longer only controlled by the relative tolerance `rtol`, and the `scipy` and `numba` backends agree to about 1e-7 instead of 1e-2 relative to the largest strain at the same run time, which changes the strain maps and their hashes slightly
- **breaking:** the results are cached as HDF5 files `{name}_{hash}.h5` instead of `.npz` files, so that former caches are recalculated, and `Simulation.save()` saves a `Result`
- all examples use the results of the simulations, named with the prefix `res_`, pass them on to the following simulations, and plot them against their coordinates as pint quantities, e.g. `result.quantity("delays", "ps")`, as well as the aliases of the scattering results, e.g. `result.R`
- the strain pulse example compares the run times of the `numba` and `scipy` backends of `PhononNum`
- the energy spectrum of the eigenmodes in the strain pulse example is plotted against the angular frequency in (rad/s) instead of the mislabeled frequency in (Hz)

### Fixed

- the temperature change of `get_temperature_after_delta_excitation()` was always zero, as the final temperature was calculated in place of the initial one

## [3.0.0a2] 2026-11-11

### Added

- tests of the multilayer formalisms against the reference results of refnx, tmm, GenX, PyGTM, PyLlama, Dyna, and MAG_sl and GID_sl of the Stepanov X-ray Server for isotropic, rough, birefringent, magneto-optical, and resonant magnetic X-ray stacks, where only their outputs are stored in `tests/data/references`
- the tests and benchmarks of all scattering formalisms in `tests/test_scattering.py` and `tests/benchmarks/test_scattering.py` as for the `simulations.scattering` package, where the benchmarks include all multilayer formalisms and the formalisms are compared against the exact modes and stable assembly of `BerremanStepanov`
- polarizations can be given by their names, e.g. `set_polarization("sigma", "none")`, as well as by their ids
- `simulations.scattering` package with the X-ray and light scattering formalisms, which are named by their method, restructuring the former `simulations.xrays` module, see [#136]
- multilayer formalisms from the X-ray to the optical range, which take the dielectric tensor of the layers and share the polarizations, `homogeneous_reflectivity_transmissivity()`, `inhomogeneous_reflectivity_transmissivity()`, and the new `homogeneous_jones_matrices()` with the Jones matrices of the reflection and transmission in the linear (σ, π) basis in their base class `MultilayerScattering`, ported from [Dyna](https://gitlab.com/dynadevgroup/Dyna):
  - `Parratt` for isotropic layers by the scalar recursion of Parratt (1954) with the roughness of Névot and Croce (1980), neglecting the magnetization
  - `Stepanov` with the analytical eigenmodes of `Elzo` and the numerically stable scattering-matrix recursion of Stepanov and Sinha (2000), e.g. for thick, strongly absorbing, or highly repeated stacks
  - `BerremanElzo` and `BerremanStepanov` with the exact eigenmodes of an arbitrary dielectric tensor (Berreman 1972), e.g. of anisotropic, birefringent, or strongly magnetic layers, with the transfer matrices or the scattering-matrix recursion
- `homogeneous_jones_matrices()` and `inhomogeneous_jones_matrices()` of the multilayer formalisms as single expensive calculation, from which `calc_reflectivity_transmissivity()` and `calc_rotation_ellipticity()` follow, e.g. the exact Kerr or Faraday rotation and ellipticity, also by `homogeneous_rotation_ellipticity()` and `inhomogeneous_rotation_ellipticity()` for the static and time-resolved Kerr and Faraday effect
- examples restructured into the gallery sections scattering, optics, X-ray reflectivity, and X-ray diffraction, with the new examples of the scattering overview of all formalisms, polarizations, and high-level methods for the W/Fe/W trilayer of Elzo et al. (2012), of the data fitting of this trilayer, i.e. of its measured hard X-ray reflectivity by `Parratt` and of its transient resonant magnetic reflectivity from an M3TM by `Elzo`, replacing the former data fitting example, of the infrared reflectance of a MoO3/AlN/SiC heterostructure with hyperbolic phonon polaritons after Passler et al. (2020), of the transient Kerr effect of a magnetization gradient after Ashok et al. (2025), and of the Brillouin scattering of white light from nonlinearly propagating strain pulses in SrTiO3 after Bojahr et al. (2012), as well as a forward fit of the parameters of the M3TM to the resonant magnetic X-ray reflectivity in the data fitting example
- the elliptical polarizations and lists of polarizations of the former `XrayDynMag` for all multilayer formalisms, which calculate all polarizations at once from the same Jones matrices with an additional last axis of the results
- the homogeneous and inhomogeneous methods of the multilayer formalisms take optional `temps` per sub-structure and a `temp_map`, e.g. of `Heat`, which is reserved for the temperature dependence of the dielectric tensor, e.g. on the electron temperature, and raises a `NotImplementedError` until it is implemented
- `Parratt` calculates the `inhomogeneous_reflectivity_transmissivity()` for blocks of delays at once, where its recursion runs over the layers for all delays, photon energies, and angles in parallel, with blocks of about 2^13 elements, which keep the arrays in the CPU cache, and at least one block per worker thread for `calc_type="parallel"`, and from the scalar refractive index of the layers instead of their dielectric tensors, e.g. for the Brillouin scattering from strain pulses in thousands of layers; other formalisms can do the same by `calc_inhomogeneous_jones_matrices()`
- `Dielectric.get_ref_index()` and `Layer.get_ref_index()` accept an array of strains and are calculated from the scalar dielectric function without the dielectric tensor
- `Dielectric.get_tensor()` and `Layer.get_dielectric_tensor()` accept an array of strains, e.g. of a layer for all delays

- `ExperimentParameters` as `experiment` of all scattering simulations, a `ParameterGroup` of the photon energy, wavelength, wavenumber, incidence angle, and scattering vector in eV and SI units; `sim.energy`, `sim.wl`, `sim.k`, `sim.theta`, and `sim.qz` remain as shortcuts in their units

### Changed

- the parallel scattering simulations send the arrays of their tasks, e.g. the blocks of the strain map, to the dask workers before the tasks, which keeps the task graph small and avoids the warning of dask about large graphs
- **breaking:** the homogeneous methods of the multilayer formalisms take the explicit optional arguments `strains`, `magnetizations`, and `temps` instead of `*args`, and `polar` and all further arguments of the homogeneous and inhomogeneous methods are keyword-only, after the optional `strain_map`, `magnetization_map`, and `temp_map` of the latter
- the grazing angle θ of the multilayer formalisms is the angle in the superstrate, i.e. the in-plane wave vector is k_y = n k cos θ with the real part of the refractive index n of the superstrate, as in tmm, PyGTM, and PyLlama, e.g. for the coupling by a prism in the Otto or Kretschmann geometry, while it is unchanged for a superstrate of vacuum and the nominal q_z = 2 k sin θ of `ExperimentParameters` is given for vacuum
- **breaking:** `homogeneous_reflectivity()` and `inhomogeneous_reflectivity()` of the multilayer formalisms are renamed to `homogeneous_reflectivity_transmissivity()` and `inhomogeneous_reflectivity_transmissivity()`, as they return both, and `calc_intensities()` and `calc_intensities_from_matrix()` to `calc_reflectivity_transmissivity()` and `calc_reflectivity_transmissivity_from_matrix()`, while `XrayKin` and `XrayDyn` keep `homogeneous_reflectivity()` and `inhomogeneous_reflectivity()`
- `Elzo` calculates the homogeneous and inhomogeneous Jones matrices by the shared engine of the multilayer formalisms instead of its own methods, which is about 40 % faster for inhomogeneous structures, and shares the analytical eigenmodes and the isotropy check with `Stepanov`
- repeated sub-structures and substrates are assembled only once by all multilayer formalisms with matrix formalisms and raised to the power of their repetitions, for the transfer matrices by integer powers and for the scattering matrices by binary powers as in Dyna
- `compose_scattering_matrices()` is renamed to `compose_scattering_stacks()` as in Dyna
- **breaking:** unknown polarization states and those not supported by a formalism, e.g. circular polarization for `XrayKin` and `XrayDyn`, raise a `ValueError` instead of falling back to sigma, unpolarized, or no analyzer
- **breaking:** `pol_in_state` and `pol_out_state` are read-only, use `set_polarization()`, and `calc_incoming_polarization()` and `calc_outgoing_polarization()` only return the polarization factor
- the default polarizations of scattering simulations are set without a message on initialization
- the incidence angle or scattering vector, whichever was set last, is kept for changes of the photon energy, and two-dimensional inputs per photon energy are used for all of them
- `update_experiment()` and the private attributes `_energy`, `_wl`, `_k`, `_theta`, and `_qz` of the scattering simulations are replaced by `experiment`
- the documentation of the `euler_angles` (θ, φ, ψ) of a `Dielectric`: they are the proper z-x-z Euler angles R = R_z(φ) R_x(θ) R_z(ψ) as on Wikipedia, i.e. the crystal axes x′, y′, z′ coincide with x, y, z without rotation, φ is the angle of the line of nodes from x, θ the tilt of z′ about it, and ψ the angle of x′ from it, with an interactive sketch in the example of anisotropic layers. They rotate the crystal actively in the sample frame of Elzo and Dyna with the plane of incidence yz, which differs from the passive rotation of PyGTM in its frame with the plane of incidence xz, see `euler_matrix()` for the conversion of the angles of PyGTM
- `XrayDynMag` is renamed to `Elzo` after the authors of its formalism, and `Xray` to `Scattering` as base class of all scattering simulations, without aliases of the former names; the title of `Elzo` is "Elzo multilayer scattering simulation"
- the transmission of `Elzo` is given for the electric field at the bottom of the last layer of the substrate, i.e. after its thickness, per incident electric field in the superstrate, i.e. multiplied by the non-magnetic dielectric function of the superstrate and divided by the one of the substrate, as for the other multilayer formalisms; it is unchanged for a superstrate and substrate of vacuum
- the magnetization of an `AmorphousLayer` is taken from its atom only on initialization, and its dielectric tensor uses the magnetization of the layer, as `LLG`, `iLLG`, and `LLB` do for their initial magnetization; a later change of the magnetization of the atom does not affect the layer, so set `layer.magnetization` instead

### Deprecated

- the module `simulations.xrays`, which will be removed in a future version, see `simulations.scattering` instead

### Removed

- **breaking:** `calc_type="distributed"` and the stub `distributed_inhomogeneous_reflectivity()` of `XrayDyn`, as `calc_type="parallel"` runs on a cluster by its Dask client as well
- **breaking:** `sequential_inhomogeneous_reflectivity()`, `parallel_inhomogeneous_reflectivity()`, `distributed_inhomogeneous_reflectivity()`, `calc_inhomogeneous_matrix()`, `calc_homogeneous_transfer_matrix()`, `calc_homogeneous_matrix()`, `get_layer_boundary_phase_matrix()`, `calc_boundary_phase_matrix()`, `calc_reflectivity_transmissivity_from_matrix()`, and `last_layer_ref_trans_matrices` of `Elzo` and the function `phase_matrix()`, the static `calc_reflectivity_transmissivity_from_matrix(RT, pol_in, pol_out)` and `calc_kerr_effect_from_matrix()` of the multilayer formalisms, see `calc_rotation_ellipticity()` instead
- **breaking:** the special treatment of `UnitCell`s in `Elzo` by the slabs of its atoms, `calc_uc_boundary_phase_matrix()`, `get_atom_boundary_phase_matrix()`, and `calc_atom_boundary_phase_matrix()`, see [#201]; a `UnitCell` is treated as all other layers with its dielectric tensor, which must be set, e.g. by `set_dielectric_from_atoms()`, and its magnetization, so that it shows no Bragg peaks anymore, for which `XrayDyn` is available
- **breaking:** the former names `XrayDynMag` and `Xray`, see `Elzo` and `Scattering` instead
- **breaking:** the polar shortcuts `mag_amplitude`, `mag_phi`, and `mag_gamma` of `Atom` and `AtomMixed`, as keyword arguments, which now raise a `TypeError`, and as attributes, see `magnetization={"r": ..., "phi": ..., "gamma": ...}` and `atom.atomic.magnetization` instead

### Fixed

- unpolarized incoming light of the multilayer formalisms, formerly `XrayDynMag`, is the incoherent average of the incoming sigma and pi polarization instead of the coherent linear polarization at 45°, which changes the results for magnetic or anisotropic layers with polarization conversion, e.g. by up to 1 % without and 30 % with a sigma analyzer for the W/Fe/W trilayer at the Fe L3 edge; `calc_incoming_polarization(0)` returns the zero vector to mark the unpolarized state, as in Dyna the coherent vector was used for its magnetic formalisms
- **breaking:** the transmission factor `get_atom_transmission_factor(atom, area)` of the atomic layers of `XrayDyn` does not include the Debye-Waller factor anymore and has no `deb_wal_fac` argument, which only applies to the reflection with momentum transfer, so that the absorption and refraction do not depend on it; it reduced the absorption, e.g. by 6 % for Si (004) with B = 0.46 Å², and increased the peak reflectivity, as found by the comparison with GID_sl of the X-ray Server
- setting the photon energy does not warn anymore for the scattering vectors of the previous energies, which cannot be reached, but the simulations warn on a calculation for undefined incidence angles
- all scattering simulations raise a `ValueError` with a hint to `Structure.add_superstrate()` and `add_substrate()`, if the superstrate or substrate is not a `Layer`, and `XrayKin` and `XrayDyn` warn that they neglect a superstrate other than `Vacuum` and a substrate other than `UnitCell`s or `Vacuum`
- the parallel inhomogeneous reflectivity of the scattering simulations sends a single copy of the simulation to the dask workers instead of one per task, which strongly reduces the time and memory of the main process, and shows a progress bar
- the inhomogeneous reflectivity of `Elzo` for layers does not calculate hashes of its unused cache anymore, and atoms only build the tables of their form factors once, which recovers the speed before the dielectric tensor of the layers
- scattering simulations can be displayed before their photon energy is set
- the scattering vector and the incidence angle of the scattering simulations are consistent for several photon energies after changing the energy, wavelength, or wavenumber
- the `phonon_damping` of `PhononNum` acts as dashpot on the rate of deformation between neighbouring layers, as described in its documentation and by Bojahr et al. (2012), instead of the velocity of each layer, so that the damping rate of a phonon grows with the square of its wave vector; the damping in kg/s must be positive, negative values raise an error, and the time step of the explicit solvers accounts for the damping
- the analytical eigenmodes of `Elzo` divide the gyrotropic part of the dielectric tensor by its diagonal in the first-order inverse permittivity, which Eq. (20) of Elzo et al. (2012) omits, as corrected in Dyna after version 6.13.1; it changes the conversion between σ and π polarization by about 1-2 % at soft X-ray resonances and by orders of magnitude for visible light, which now agree with the exact Berreman eigenmodes

[#136]: https://github.com/dschick/udkm1Dsim/issues/136
[#165]: https://github.com/dschick/udkm1Dsim/pull/165
[#166]: https://github.com/dschick/udkm1Dsim/issues/166
[#201]: https://github.com/dschick/udkm1Dsim/issues/201

## [3.0.0a1] 2026-11-09

### Added

- read-only `norm` property of `VectorParameter`, the non-negative length of the vector, e.g. of `atom.atomic.magnetization.norm`, whereas its radius `r` is negative for the reversed direction of the angles
- `Dispersion` for energy-dependent dielectric quantities, evaluated as dielectric function `eps()` or susceptibility `chi()`, from constants or tables by `from_array()`, `from_ref_index()`, and `from_file()`, e.g. for `.nk` files, with photon energies, wavelengths, frequencies, or wavenumbers as spectral axis; dispersions are isotropic, uniaxial, diagonal, or full tensors, are combined by `a | b` for the fallback to `b` outside the energy range of `a` and by `a + b` for the sum of their contributions, and raise an error outside their energy range instead of an extrapolation, which gives the defined ranges in the quantity and unit of the input, e.g. as wavelengths in nm
- `Dielectric` parameter `dielectric` of `OpticalParameters` with the non-magnetic dispersion `eps0`, the gyrotropic dispersion `eps_mag`, also as Voigt constant, and the `strain_model` `none`, `density` (default) for the scaling of the susceptibilities with the density, or `photoelastic` for a measured, total change per strain `deps_dstrain` of the non-magnetic part with the density scaling of the gyrotropic part, from which `get_tensor()`, `get_ref_index()`, and `get_opt_pen_depth()` follow
- `dispersion_from_atoms()`, `dielectric_from_atoms()`, and `set_dielectric_from_atoms()` of `AmorphousLayer` and `UnitCell` from the atomic and magnetic form factors of their atoms with an optional `fallback` dispersion, e.g. a refractive index in the optical range, closing [#112]
- methods of `Layer` for the optical quantities of its dielectric tensor for its density: `get_dielectric_tensor()`, `get_ref_index()`, `get_delta_beta()`, `get_opt_pen_depth()`, `get_absorption_coefficient()`, `get_mass_attenuation_coefficient()`, `get_critical_angle()`, and the effective atomic form factor `get_form_factor()`, which are also available for a `Dielectric`
- `Dispersion.from_opt_pen_depth()` with a constant optical penetration depth at all wavelengths and `Layer.set_opt_pen_depth()`, which only changes the extinction coefficient of the dielectric function of a layer with a warning, e.g. for the Lambert-Beer absorption
- dispersions from $\delta + i\beta$ of X-ray tables by `kind="delta_beta"`, also for the magnetic optical constants $\Delta\delta + i\Delta\beta$ of the magnetic part, and from tabulated mass attenuation coefficients by `Dispersion.from_mass_attenuation()`
- `get_delta_beta_mag()` of `Dielectric` and `Layer` with the magnetic optical constants $\Delta\delta$ and $\Delta\beta$ of the magnetic part $\varepsilon^m = 2 (\Delta\delta - i\Delta\beta)$
- `a - b` and `c * a` for the difference and scaling of the contributions of dispersions
- dispersions from tabulated atomic and magnetic form factors per formula unit, e.g. of `.cf`, `.nff`, or `.mf` files, with their molar mass by `Dispersion.from_form_factors()` or `kind="f"` and `kind="m"`
- `helpers.read_columns()` and `helpers.SpectralTable`, which read and interpolate the tabulated spectral data of the dispersions and of the form factors of the atoms
- `euler_angles` of a `Dielectric` rotate its non-magnetic part from the frame of the crystal into the frame of the sample by z-x-z Euler angles as in PyGTM, see `euler_matrix()`, while its magnetization and photoelastic change are given in the frame of the sample
- `Dispersion.from_refractiveindex()` loads the optical constants of a page of the [refractiveindex.info](https://refractiveindex.info) database by its shelf, book, and page or its url with the optional package [refractiveindex](https://github.com/toftul/refractiveindex), installed by `pip install udkm1Dsim[refractiveindex]`, following the discussion [#192]
- `Atom.get_susceptibility()` with the non-magnetic and gyrotropic susceptibility from the atomic and magnetic form factors
- the magnetization of `gyrotropic_tensor()`, `Dielectric.get_tensor()`, and `Layer.get_dielectric_tensor()` is given in cartesian coordinates `(x, y, z)` by default or in polar coordinates `(amplitude, phi, gamma)` with the angles in rad for `polar=True`, as the magnetization maps
- `atomic_form_factor_range`, `magnetic_form_factor_range`, `has_magnetic_form_factor`, and `form_factor_source()` of `Atom` and `AtomMixed`, whose display shows the sources and energy ranges of their form factors, e.g. `Atomic form factors: Chantler from 10.69 eV to 30.01 keV`
- example of the atomic form factors and dielectric functions with a short introduction in the structure example
- `print()` and `display()` of a `Dielectric` describe its dispersions, e.g. `table of n: 49 values from 0.64 eV to 6.6 eV (Cr.nk) | n = 2.87 + 3.36i`, or show their values for its `display_energy`, e.g. `ε⁰(800 nm) = −3.053 + 19.29i`, with values close to 1 as deviation from 1 for X-rays

### Changed

- the tables of atoms, layers, and parameter groups show a `VectorParameter`, e.g. the magnetization, by its cartesian components instead of its polar components
- **breaking:** `get_atomic_form_factor()` and `get_magnetic_form_factor()` of `Atom` and `AtomMixed` accept wavelengths, frequencies, and wavenumbers as pint Quantity and raise a `ValueError` outside of the tabulated energy range in the quantity and unit of the input instead of extrapolating, also skipping the invalid entries of Henke et al.
- **breaking:** an `Atom` without tabulated magnetic form factors has `magnetic_form_factor_coeff = None` instead of zeros and does not print a message anymore, while a missing file given by `magnetic_form_factor_path` raises a `FileNotFoundError`
- **breaking:** `XrayDynMag` and the multilayer absorption of `Heat` use the `dielectric` of the layers, which must be set explicitly, e.g. by `layer.set_dielectric_from_atoms()` for the previous behavior of `XrayDynMag` for an `AmorphousLayer`, while `UnitCell`s still use their atoms; any homogeneous `Layer` with an isotropic dielectric tensor can be used by `XrayDynMag`
- **breaking:** the optical properties of a layer are only given by its `dielectric`, from which the refractive index and the optical penetration depth follow for the wavelength of the excitation

### Removed

- **breaking:** `opt_ref_index`, `opt_pen_depth`, `opt_ref_index_per_strain`, and `set_opt_pen_depth_from_ref_index()` of layers and `OpticalParameters`, which raise an error with their replacement, see `Dispersion.from_ref_index()`, `Dispersion.from_opt_pen_depth()`, `Layer.set_opt_pen_depth()`, the `strain_model` of `Dielectric`, and `Layer.get_opt_pen_depth()` instead

### Fixed

- the homogeneous reflectivity of `XrayDynMag` includes the thickness of `Vacuum` layers and scales the density of strained amorphous layers by `1/(1+strain)` instead of `(1+strain)` as the inhomogeneous reflectivity

### Security

[#112]: https://github.com/dschick/udkm1Dsim/issues/112
[#192]: https://github.com/dschick/udkm1Dsim/discussions/192

## [2.6.1] 2026-11-07

### Added

- `polar` switch of `get_magnetization_map`, `get_sublattice_magnetization_map`, and `get_equilibrium_magnetization` of `Magnetization` and of `inhomogeneous_reflectivity` and `homogeneous_reflectivity` of `XrayDynMag` for magnetization maps in polar coordinates `(amplitude, phi, gamma)` with the angles in rad
- `XrayDynMag.parse_magnetization_map` returns a magnetization map given in cartesian or polar coordinates as cartesian array

### Changed

- **breaking:** the magnetization maps of `Magnetization`, `LLG`, `iLLG`, and `LLB` and the `magnetization_map` input of `XrayDynMag` are given in cartesian coordinates `(x, y, z)` by default, as the `magnetization` of layers and atoms and the `init_magnetization`, instead of polar coordinates; use `polar=True` for the former polar coordinates
- the cartesian result of `get_equilibrium_magnetization` can be used directly as `init_magnetization`
- `calc_magnetization_map` of custom `Magnetization` simulations must return cartesian maps, e.g. converted by `helpers.convert_polar_to_cartesian`
- the magnetization maps are cached in cartesian coordinates, so that caches of former versions are recalculated
- `XrayDynMag` uses the cartesian magnetization of the atoms internally, which gives the same results as the former polar implementation
- the resonant magnetic X-ray example sets the magnetization of the atoms in cartesian coordinates, uses a two-temperature model of the electrons and phonons, and derives the magnetization amplitude from the mean-field magnetization of the electron temperature
- the LLB example gives the electronic heat capacities directly as mass-specific values

## [2.6.0] 2026-11-06

### Added

- `Layer`, `AmorphousLayer`, and `UnitCell` accept their properties as parameter groups or dicts, e.g. `thermal=ThermalParameters(...)` or `lattice={"c_axis": ...}`; groups are copied on assignment and can be reused, mistyped parameter names raise an error, and parameters given twice raise an error
- parameter groups of a layer can be replaced after its creation, e.g. `layer.elastic = {"sound_vel": ...}`
- read-only `acoustic_impedance` property of `Layer`
- pre-commit hooks running `ruff check` and keeping `uv.lock` up to date
- parameters have an editable `symbol` in LaTeX with defaults following the formulas of the simulations, e.g. `layer.thermal.therm_cond.symbol = r"\kappa"`, closing [#224]
- `print()` and `display()` of a parameter show its symbol, value or expression, and unit, e.g. `c_e(T_e) = 0.027·T_e J/K/kg`
- `sub_system_labels` of `ThermalParameters`, e.g. `["e", "p"]`, are displayed as subscripts of the temperature-dependent parameters and the temperatures, and default to the index of the sub-system
- helpers to format symbols given in LaTeX, numbers, and sympy expressions as plain text with unicode characters, HTML, or LaTeX, e.g. `latex_to_text`, `format_number`, and `expression_to_html`
- `AtomParameters` and `AtomMixedParameters` organize the atomic number, mass number, mass, ionicity, magnetization, and, for `Atom` only, the Cromer-Mann coefficients `cromer_mann_a`, `cromer_mann_b`, and `cromer_mann_c` in their `atomic` parameter group, e.g. `atom.atomic.mass`, while the attributes of the atoms, e.g. `atom.mass`, are kept
- `magnetization` property of `Atom` and `AtomMixed` with the cartesian magnetization as for layers
- `VectorParameter` keeps its angles, when they are not defined by its cartesian components, e.g. for a zero amplitude, so that setting the amplitude to zero and back restores its direction
- the magnetization of an `Atom`, `AtomMixed`, and `Layer` is given by `magnetization` on initialization and assignment, either as cartesian components, e.g. `(0, 0, 1)`, as dict of polar or cartesian components, e.g. `{"r": -1, "phi": 90 * u.deg, "gamma": 0 * u.deg}` or `{"x": 0, "y": 0, "z": 1}`, or as `VectorParameter`; components missing in a dict keep their values, e.g. `atom.magnetization = {"r": 0.5}` only changes the amplitude
- `VectorParameter.set()` for all these inputs
- `simulations.solvers.magnetization_scipy` with the right-hand side of the `LLB` assembled from one function per physical term, vectorized over all magnetic layers, and `simulations.solvers.mean_field` with the equilibrium magnetization, the damping parameters and the longitudinal susceptibility of the mean-field model; the solver takes the `MagneticParameters` of the magnetic layers, ported from [friweber/udkm1Dsim_LLG-LLB](https://github.com/friweber/udkm1Dsim_LLG-LLB) by [@friweber](https://github.com/friweber)
- classical `LLB` with the Langevin function for an infinite `eff_spin`
- `LLG` and inertial `iLLG` (Ciornei et al. 2011) simulations with a prescribed magnetization amplitude, given as `mag_amplitude_map`, taken from the `mag_sub_system` of the `temp_map`, derived from its `temp_sub_system` by the mean-field model, or kept at the initial amplitude; the `iLLG` accepts an `init_velocity`, which defaults to the one of the LLG, ported from [friweber/udkm1Dsim_LLG-LLB](https://github.com/friweber/udkm1Dsim_LLG-LLB) by [@friweber](https://github.com/friweber), superseding [#164]
- magnetic parameters `gilbert_damping`, `inertial_time`, and `magnetoelastic_coupling` of layers
- magnetoelastic field of the out-of-plane strain for `LLG`, `iLLG`, and `LLB`, which accept a `strain_map`
- magnetic sublattices of layers, e.g. of ferri- and antiferromagnets, for `LLG`, `iLLG`, and `LLB`, ported from [friweber/udkm1Dsim_LLG-LLB](https://github.com/friweber/udkm1Dsim_LLG-LLB) by [@friweber](https://github.com/friweber): the per-sublattice parameters of `MagneticParameters` are given as scalar for all sublattices or one value per sublattice, `magnetization` as a list of vectors, and `mf_exch_coupling` as `PxP` exchange matrix, from which the `curie_temp` is derived; `num_sublattices` and `check_sublattices()` of `MagneticParameters`, `num_sublattices` of `Layer`
- an `AmorphousLayer` of an `AtomMixed` with one constituent per sublattice inherits the magnetizations of the constituents for its sublattices
- `get_sublattice_magnetization_map` and `get_sublattice_mean_field_mag_map` of `Magnetization`, which are calculated and cached together with the net `get_magnetization_map` and `get_mean_field_mag_map`; `init_magnetization` of several sublattices is given as `Px3` or `LxPx3` array, `mag_amplitude_map` of the `LLG` and `init_velocity` of the `iLLG` also per sublattice
- `mf_exch_coupling` of `MagneticParameters` and `Layer` can be set, which makes the `curie_temp` the derived parameter, and vice versa for a single sublattice
- exchange relaxation between two sublattices of the `LLB` (Raposo et al. 2022), which transfers angular momentum directly between them and reverses e.g. GdFeCo by all-optical switching, with `exchange_relaxation=True` and the magnetic parameters `exch_relax_rate` and `concentration`, which are equal for all sublattices unless they are given or taken from the fractions of an `AtomMixed`, ported from [friweber/udkm1Dsim_LLG-LLB](https://github.com/friweber/udkm1Dsim_LLG-LLB) by [@friweber](https://github.com/friweber)
- `Magnetization.get_equilibrium_magnetization` and the option `equilibrate` of `get_magnetization_map` relax the directions of the initial magnetization of `LLG`, `iLLG`, and `LLB` into the equilibrium of the external field, the anisotropy, the exchange, and the strain by an overdamped flow without precession, see `solvers.magnetization_scipy.relax_directions`, instead of simulating the precession for a long delay range before the excitation, closing [#139]
- `Magnetization.get_magnetic_parameters` and `Magnetization.get_interlayer_exchange_stiffnesses` collect the inputs of the solver from the structure
- `Magnetization.temp_sub_system` to select the sub-system of the `temp_map` that drives the magnetization dynamics
- `LLB.get_magnetization_map` accepts an external field `B_ext` per delay as `Mx3` array
- `LLB` warns if a layer ends at a vanishing magnetization below its Curie temperature, since `m = 0` is a fixed point of the LLB
- magnetic parameter `gyromagnetic_ratio` of layers per sublattice for `LLG`, `iLLG`, and `LLB`, which defaults to the one of a free electron `GAMMA_E`, now taken from `scipy.constants` (-1.76086e11 instead of -1.761e11 rad/(s T))

### Changed

- the external magnetic field of the magnetization simulations is given as `B_ext` in tesla instead of `H_ext`, consistent with the fields $\mathbf{B}$ in the equations and their units, also as `pint` quantity in any unit of the field
- the initial magnetization of the magnetization simulations is given as `init_magnetization` in cartesian coordinates instead of `init_mag` in polar coordinates, consistent with `B_ext` and the `magnetization` of the layers; it can also be given in polar coordinates as dict `{"r": ..., "phi": ..., "gamma": ...}` or as `VectorParameter`, see `Magnetization.parse_initial_magnetization`, and the former `init_mag` raises a `TypeError`
- the magnetization examples are replaced by `01-LLG` (precession, equilibrium, amplitude from the temperature or a map, magnetoelastic coupling, reproducing the heterostructure-dependent precession of Jarecki et al. 2024), `02-iLLG` (susceptibility with the ferromagnetic resonance and the nutation, THz-driven nutation resonance after Neeraj et al. 2021, nutation after an optical excitation after De et al. 2025), `03-LLB` (longitudinal dynamics, comparison with the LLG, a simple antiferromagnet), and `04-eLLB` (all-optical switching of GdFeCo after Raposo et al. 2022, switching diagram of fluence and pulse duration compared with Davies et al. 2020 and Jakobs and Atxitia 2022), adapted from the examples of [friweber/udkm1Dsim_LLG-LLB](https://github.com/friweber/udkm1Dsim_LLG-LLB) by [@friweber](https://github.com/friweber)
- the default integrator of the magnetization simulations is `LSODA` instead of `RK45`, which is much faster for the stiff equations of strongly coupled sublattices, the inertial LLG, and the LLB at low temperatures; with the default tolerances it is also more accurate, e.g. the largest deviation of the magnetization amplitude in the LLB example from a tightly converged solution drops from 0.03 to 0.004
- `Magnetization.check_initial_magnetization` returns the initial magnetization of all sublattices as `NxPx3` array, and `calc_magnetization_map` of `LLG`, `iLLG`, and `LLB` returns the net and the sublattice magnetization maps
- caches of magnetization maps of former versions are recalculated, as they hold no sublattice maps
- parameter groups notify their layer on changes, which updates all parameters depending on several groups, e.g. the spring constant
- the `density` of a `UnitCell` with atoms cannot be set, as it is given by their mass and its volume, and its `thickness` is its `c_axis`
- `AmorphousLayer` does not require `thickness` and `density` as positional arguments anymore
- `UnitCell` does not require `c_axis` as positional argument if `lattice` is given
- `a_axis` and `b_axis` of `LatticeParameters` default to its `c_axis`
- `ElasticParameters` has no `spring_const` argument anymore, as it is always derived
- the table views of parameter groups and layers show the name, symbol, value, and unit of each parameter in separate columns and one row per sub-system
- the examples define their layers by parameter groups and label the sub-systems of the two- and three-temperature models, e.g. `sub_system_labels=["e", "p", "s"]`
- the `mass` of an atom is derived from its mass number and cannot be set, and the atomic number, mass number, and ionicity of an `AtomMixed` are derived from its atoms
- the table views of `Atom` and `AtomMixed` show their parameters as the parameter groups of layers
- the magnetization of `Atom` and `AtomMixed` is a `VectorParameter` and behaves as for layers, e.g. its gamma angle is given in -180 .. 180 deg
- `VectorParameter` stores its polar form (r, phi, gamma), and its cartesian components are kept as given or calculated on demand, where components, which are zero within the numerical precision, are exactly zero
- the radius of a `VectorParameter`, e.g. the magnetization amplitude, can be negative, which reverses the direction given by its angles, and is not limited in size, so that its magnitude can be changed while keeping its angles
- the examples set the magnetization of atoms by `magnetization={"r": ..., "phi": ..., "gamma": ...}`
- vector parameters, e.g. the magnetization, are shown by their amplitude and angles in deg in the table views
- the symbols of the physical quantities in the docstrings are consistent across all simulations and parameter groups, e.g. the thickness is `d`, the thermal conductivity `κ`, and the anisotropy exponent `β`
- the atomic number and mass number of an `Atom` are given by its symbol and cannot be set
- for an ionicity, which is not tabulated for the Cromer-Mann coefficients, the ones of the closest tabulated ionicity are used with a warning instead of raising an `IndexError`
- the physics of the `LLB` moved from its static methods into `simulations.solvers.magnetization_scipy` and `simulations.solvers.mean_field`; the temperature and the external field are interpolated linearly in time instead of being held constant between delays, and the mean-field magnetization, damping parameters, and susceptibility are interpolated from tables per layer instead of a root search per temperature
- `get_mean_field_mag_map` and `get_directional_exchange_stiffnesses` moved from `LLB` to `Magnetization`; the mean-field magnetization is zero above the Curie temperature instead of raising an `IndexError` if no temperature is below it
- without explicit `init_magnetization` the `LLB` starts in the mean-field equilibrium of the initial temperature along the magnetization of the layers
- the anisotropy field of the `LLB` is `2/M_s m_eq^(β-2) (K_x m_x, K_y m_y, K_z m_z)` according to Raposo et al. (2022); the former expression had the same torque but an additional field `-2/M_s m_eq^(β-2) (K_x+K_y+K_z) m` along the magnetization, which altered the longitudinal relaxation
- the hash of a magnetization simulation includes its type and the `temp_sub_system`, hence cached results are recalculated

### Deprecated

### Removed

- the static methods `LLB.odefunc`, `calc_uniaxial_anisotropy_field`, `calc_exchange_field`, `calc_thermal_field`, `calc_Brillouin`, `calc_dBrillouin_dx`, `calc_transverse_damping`, `calc_longitudinal_damping`, `calc_qs`, and `calc_long_susceptibility`, see `simulations.solvers.magnetization_scipy` and `simulations.solvers.mean_field` instead

### Fixed

- the header of the HTML table of a `Structure`, e.g. in the representation of the simulations, is styled as the headers of the other tables
- assigning a value to a parameter of a group, e.g. `layer.magnetic.curie_temp = 500 * u.K`, replaced the `Parameter` by the value, so that depending parameters were not updated anymore; it sets the value of the parameter now, a `Parameter` of the same type replaces it after its conversion to the unit of the group, and derived parameters raise an `AttributeError`, closing [#234]
- `Parameter` accepts lists and numpy scalars as magnitude
- changing parameters directly via their group, e.g. `layer.structural.thickness`, did not update depending parameters of other groups, e.g. the spring constant
- API documentation of the `parameters`, `parameter_groups`, and `solvers` modules was missing
- `AtomMixed` had no `_mass`, which is used for the density of its unit cell by `XrayDynMag`
- the gamma angle of zero vectors with signed zero components was -180 deg instead of 0 deg in `convert_cartesian_to_polar`
- the angles phi and gamma were interchanged in the docstrings of `convert_cartesian_to_polar` and `convert_polar_to_cartesian`
- the pdf documentation on Read the Docs failed, as pdflatex cannot handle the unicode characters of the notebook outputs; it is built with xelatex now
- the Cromer-Mann atomic form factor of `XrayKin` and `XrayDyn` used wrong coefficients and converted `q_z` wrongly, so that its `q_z`-dependence was wrong; cached results require `force_recalc=True`, closing [#226]
- Cromer-Mann coefficients of `Cs1+`, `Ba2+`, `Ru4+`, `Bi5+`, `Np3+`, `Np4+`, `Np6+`, and `Pu` corrected and 16 rounded coefficients completed according to the International Tables for Crystallography, Vol. C, Table 6.1.1.4
- the Cromer-Mann coefficients of an `Atom` were not updated on a change of its ionicity
- the backside of the structure was clamped instead of free for `PhononAna`, so that strain waves were reflected with a wrong sign, e.g. for freestanding layers; the analytical and numerical phonon models have the same boundary conditions now and are compared in the tests; cached results require `force_recalc=True`, closing [#186]
- the exchange field of the `LLB` mixed the vector components of neighbouring layers and had the wrong sign, which pulled neighbouring layers out of alignment instead of aligning them
- magnetic layers separated by a non-magnetic spacer are not coupled by the exchange field of the `LLB` anymore
- the thermal field of the `LLB` above the Curie temperature was multiplied by the squared magnetization; it is `-(1/chi)(1 + c_S T_C/(T - T_C) m²) m` now (Atxitia et al. 2012), with `c_S` from the Brillouin expansion and the classical `3/5` for an infinite effective spin
- the parameter `q_s` of the quantum `LLB` was calculated with swapped arguments, i.e. from the effective spin in place of the mean-field magnetization, which suppressed the longitudinal relaxation of layers with a large spin and Curie temperature and made their transverse damping negative, and used `2S+1` instead of `2(S+1)` (Nieves et al. 2014), and the longitudinal damping diverged at the Curie temperature; the longitudinal and transverse damping parameters are continuous at the Curie temperature now and reduce to the ones of Garanin for an infinite effective spin
- the `mf_exch_coupling` of a layer was not defined for an infinite `eff_spin`
- `LLB.get_magnetization_map` rejected a valid `Nx3` initial magnetization
- `LLB.get_magnetization_map` failed for `progress_bar=False`
- `LLB.get_magnetization_map` failed for a single delay and for structures without magnetic layers
- the adaptive step size of the magnetization solver could grow so large for a magnetization in equilibrium that the solver stepped over the excitation, e.g. for low initial temperatures; its `max_step` is limited to the shortest delay step now
- the `LLB` overflowed or stalled at low temperatures, where the tiny longitudinal susceptibility makes the longitudinal field huge, whose rounding errors entered the torques; it only enters the longitudinal relaxation now
- the anisotropy and exchange fields of the `LLB` were not finite for layers with a Curie temperature but without saturation magnetization, which stalled the solver; these fields vanish now
- `LLB` raises an error for layers with a Curie temperature but without an effective spin instead of silently keeping them at zero magnetization
- `LLB` raises an error for layers with an anisotropy or a magnetoelastic coupling and an `aniso_exponent` below 2, for which these fields diverge at the Curie temperature and stalled the solver

### Security

[#139]: https://github.com/dschick/udkm1Dsim/issues/139
[#164]: https://github.com/dschick/udkm1Dsim/pull/164
[#186]: https://github.com/dschick/udkm1Dsim/issues/186
[#224]: https://github.com/dschick/udkm1Dsim/issues/224
[#226]: https://github.com/dschick/udkm1Dsim/issues/226
[#234]: https://github.com/dschick/udkm1Dsim/issues/234


## [2.5.0] 2026-10-20

### Added

- Parameter and ParamterGroup classes organizing paramters closing [#162] by [#215]
- temperature-dependent parameter functionals are calculated on demand and are cached by [#215]; the Numba solvers work on the sympy expressions instead of compiled material functions, closing [#206] and [#208]
- VectorParameter enabling input and output in cartesian or polar coordinates closing [#138] by [#215]
- `simulations.solvers` sub-module for the ODE/PDE solvers of `Heat` and `Phonon` with selectable `backend` by [#216] closing [#205]
- Numba backend for numerical phonons `PhononNum(backend='numba')` by [#220]
- `__repr__` and `_repr_html_` for all structures and simulations rendering their properties as tables in Jupyter and in the documentation
- `Structure.check_num_sub_systems()` to check the number of sub-systems of the relevant thermal parameters of all layers
- optional dependency `numba` for the Numba solver backends (`pip install udkm1Dsim[numba]`)

[#162]: https://github.com/dschick/udkm1Dsim/discussions/162
[#138]: https://github.com/dschick/udkm1Dsim/issue/138
[#206]: https://github.com/dschick/udkm1Dsim/issue/206
[#208]: https://github.com/dschick/udkm1Dsim/issue/208
[#215]: https://github.com/dschick/udkm1Dsim/pull/215
[#205]: https://github.com/dschick/udkm1Dsim/issue/205
[#216]: https://github.com/dschick/udkm1Dsim/pull/216
[#220]: https://github.com/dschick/udkm1Dsim/pull/220

### Changed

- layer properties are organized in ParameterGroups (`structural`, `thermal`, `elastic`, `optical`, `magnetic`, and `lattice` for `UnitCell`), the layer shortcuts such as `layer.heat_capacity` remain ([#215])
- the magnetization of a layer is a `VectorParameter`, e.g. `layer.magnetic.magnetization.polar` replaces `layer.magnetization['phi']` ([#215])
- all internal calls of `Structure.get_layer_property_vector()` use explicit ParameterGroup paths, e.g. `'thermal.heat_capacity.integral'`, closing [#214]
- `Layer.get_property_dict()` returns the `Parameter` objects keyed by their path, which changes all hashes of cached simulation results
- `Layer.mag_moment` is returned in A m² instead of Bohr magnetons
- `LLB.calc_Brillouin()` takes `mf_exch_coupling` as float in J as documented
- `__str__` methods of structures and simulations are replaced by `__repr__`, simulations show their properties before the structure
- examples display objects via their HTML representation instead of `print()`
- `Structure.num_sub_systems` and `Layer.num_sub_systems` are derived from the thermal parameters of the layers and cannot be set anymore; the consistency of the number of sub-systems is checked by `Heat` and `Phonon` for their relevant parameters when a simulation starts, instead of printing warnings on every assignment

### Fixed

- changing a lattice axis of a `UnitCell` updates its area, volume, density, and elastic parameters, while its mass is given by its atoms
- mean-field magnetization of temperatures just below the Curie temperature was zero, causing divisions by zero in the LLB equations, and could be non-zero above it

[#214]: https://github.com/dschick/udkm1Dsim/issue/214

## [2.4.2] 2026-10-07

### Added

- Vacuum added by [#204] closing [#202]
- example for XrayKin added by [#204]
- superstrate introduced by [#204] closing [#134]
- enable automatic numerical integration of layer properties if sympy fails ([#198]) replacing [#110] and closing issue [#109] and [#108]
- `XrayDynMag.set_polarization` accepts elliptical incoming and outgoing polarizations to set both at the same time ([#217])

[#198]: https://github.com/dschick/udkm1Dsim/pull/198
[#110]: https://github.com/dschick/udkm1Dsim/pull/110
[#109]: https://github.com/dschick/udkm1Dsim/issue/109
[#108]: https://github.com/dschick/udkm1Dsim/issue/108
[#204]: https://github.com/dschick/udkm1Dsim/pull/204
[#202]: https://github.com/dschick/udkm1Dsim/issue/202
[#134]: https://github.com/dschick/udkm1Dsim/issue/134
[#217]: https://github.com/dschick/udkm1Dsim/pull/217

### Changed

- substrates can be a single layer
- Vacuum is default superstrate and substrate
- `XrayDynMag` polarization setters check the consistency of the number of incoming and outgoing elliptical polarizations ([#217])

### Fixed

- bug in energy spectrum calculation of analytical phonons ([#212]) closing issue [#84]
- bug for multiple elliptical polarizations in `XrayDynMag.inhomogeneous_reflectivity` ([#217]) closing issue [#196]
- `XrayDynMag.get_hash` includes the elliptical polarization vectors ([#217])

[#212]: https://github.com/dschick/udkm1Dsim/pull/212
[#84]: https://github.com/dschick/udkm1Dsim/issue/84
[#196]: https://github.com/dschick/udkm1Dsim/issue/196

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
