# Welcome to the **udkm1Dsim** toolbox

[![Documentation Status](https://readthedocs.org/projects/udkm1dsim/badge/?version=latest)](https://udkm1dsim.readthedocs.io/en/latest/?badge=latest)
![CI](https://github.com/dschick/udkm1Dsim/actions/workflows/main.yml/badge.svg)
![pypi](https://github.com/dschick/udkm1Dsim/actions/workflows/upload-to-pypi.yml/badge.svg)
[![codecov](https://codecov.io/gh/dschick/udkm1Dsim/branch/develop/graph/badge.svg?token=9J3BQYE6CE)](https://codecov.io/gh/dschick/udkm1Dsim)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://app.codspeed.io//badge.json)](https://app.codspeed.io//dschick/udkm1Dsim?utm_source=badge)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/udkm1Dsim.svg)](https://anaconda.org/conda-forge/udkm1Dsim)
[![PyPI](https://img.shields.io/pypi/v/udkm1Dsim)](https://pypi.org/project/udkm1Dsim/)
[![PyPI downloads](https://img.shields.io/pypi/dm/pypistats.svg)](https://pypistats.org/packages/udkm1Dsim)

The **udkm1Dsim** toolbox is a collection of Python classes and routines for simulating the thermal, structural, and magnetic dynamics of one-dimensional sample structures following ultrafast laser excitation. 
It also enables the calculation of the corresponding light and X-ray scattering response, providing a unified framework for modeling coupled ultrafast dynamics and their experimental observables.


The **udkm1Dsim** toolbox provides a comprehensive framework for defining arbitrary layered structures at the atomic level, supported by a rich database of element-specific physical properties. Ultrafast excitation is described using an *N*-temperature model, a widely used approach for simulating energy transfer following ultrafast optical excitation. Structural dynamics driven by thermally induced stresses are calculated using a linear-chain model of coupled masses and springs, while magnetization dynamics can be simulated with Landau–Lifshitz-type models. The resulting optical and X-ray scattering response is computed using kinematical or dynamical scattering theory, including resonant magnetic X-ray scattering, enabling direct comparison between time-resolved simulations and a wide range of ultrafast scattering experiments.

The simulation framework is highly modular. Each module returns standard `numpy` 
arrays that can be inspected, modified, or used directly as input for 
subsequent simulations, allowing users to build customized workflows with ease.

> [!NOTE]
>The **udkm1Dsim** toolbox was initially developed for MATLAB® in the
>[Ultrafast Dynamics in Condensed Matter](https://www.uni-potsdam.de/en/udkm) group of Prof. Matias Bargheer at the
>*University of Potsdam*, Germany.

> [!HINT]
> The MATLAB® source code is still available at [github.com/dschick/udkm1DsimML](https://github.com/dschick/udkm1DsimML).

The current toolbox, written in Python, is maintained by [Daniel Schick](mailto:schick@mbi-berlin.de) at the
[Max Born Institut](https://mbi-berlin.de), Berlin, Germany.

## Documentation
The documentation can be found at [udkm1Dsim.readthedocs.io](http://udkm1Dsim.readthedocs.io).

## Citation

Please cite the latest publication if you use the toolbox in your own work:

> [!NOTE]
> Schick, D., *udkm1Dsim - A Python toolbox for simulating 1D ultrafast dynamics in condensed matter*, [Comput. Phys. Commun. 266, 108031 (2021)](https://doi.org/10.1016/j.cpc.2021.108031) [(preprint)](https://arxiv.org/abs/2102.12144).


You can also cite the original publication if appropriate:

> [!NOTE]
> Schick, D., Bojahr, A., Herzog, M., Shayduk, R., von Korff Schmising, C. & Bargheer, M., *udkm1Dsim - A Simulation Toolkit for 1D Ultrafast Dynamics in Condensed Matter*,
[Comput. Phys. Commun. 185, 651 (2014)](http://doi.org/10.1016/j.cpc.2013.10.009) [(preprint)](http://www.udkm.physik.uni-potsdam.de/medien/udkm1Dsim/udkm1DsimManuscriptPrePrint.pdf)

## Installation

### Installing with pip

You can either install directly from pypi.org using the command

    > pip install udkm1Dsim

or if you want to work on the latest develop release you can clone 
udkm1Dsim from the main git repository:

    > git clone https://github.com/dschick/udkm1Dsim.git udkm1Dsim

To work in editable mode (source is only linked 
but not copied to the python site-packages), just do:

    > pip install -e ./udkm1Dsim

Or to do a normal install with

    > pip install ./udkm1Dsim

Optionally, you can also let pip install directly from the repository: 

    > pip install git+https://github.com/dschick/udkm1Dsim.git

You can have the following optional installation to enable parallel
computations, unit tests, as well as building the documentation:

    > pip install udkm1Dsim[parallel]
    > pip install udkm1Dsim[testing]

### Installing with conda

You can install directly from conda-forge using `conda`:

    > conda install -c conda-forge udkm1dsim

or using `mamba`:

    > mamba install udkm1dsim

See [udkm1dsim-feedstock](https://github.com/conda-forge/udkm1dsim-feedstock) for more details

## Contribute & Support

If you are having issues please let us know via the
[issue tracker](https://github.com/dschick/udkm1Dsim/issues).

You can also ask questions, share ideas, or engage with community members via the
[discussions](https://github.com/dschick/udkm1Dsim/discussions).

You can contribute to the project via pull-requests following the
[GitHub flow concept](https://docs.github.com/en/get-started/quickstart/github-flow).

## Contributers

Thanks to all who have contributed to the **udkm1Dsim** toolbox!

<a href="https://github.com/dschick/udkm1Dsim/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dschick/udkm1Dsim&max=40&columns=8" style="background-color: transparent" />
</a>

## License

The project is licensed under the MIT license.