# Getting Started

The **udkm1Dsim** toolbox provides a comprehensive framework for defining 
arbitrary layered structures at the atomic level, supported by a rich database 
of element-specific physical properties. Ultrafast excitation is described using 
an *N*-temperature model, a widely used approach for simulating energy transfer 
following ultrafast optical excitation. Structural dynamics driven by thermally 
induced stresses are calculated using a linear-chain model of coupled masses and 
springs, while magnetization dynamics can be simulated with Landau–Lifshitz-type 
models. The resulting optical and X-ray scattering response is computed using 
kinematical or dynamical scattering theory, including resonant magnetic X-ray 
scattering, enabling direct comparison between time-resolved simulations and a 
wide range of ultrafast scattering experiments.

## Installation

The **udkm1Dsim** toolbox is distributed as a Python package and can be installed 
together with all required dependencies using your preferred package manager.

::::{tab-set}

:::{tab-item} pip

    > pip install udkm1Dsim

:::

:::{tab-item} conda

    > conda install -c conda-forge udkm1dsim

:::

:::{tab-item} mamba

    > mamba install udkm1dsim

:::

::::

## Design Philosophy

::::{grid} 1 1 2 2
:gutter: 4
:reverse:

:::{grid-item}
```{image} _static/structure.png
:width: 400px
:class: sd-m-auto
```
:::

:::{grid-item}

The sketch illustrates the internal architecture of **udkm1Dsim** toolbox.
All physical material properties are stored in a 
{py:class}`Structure <udkm1Dsim.structures.structure.Structure>` 
object composed of {py:class}`Layer <udkm1Dsim.structures.layers.Layer>` 
objects, , which in turn consist of 
{py:class}`Atom <udkm1Dsim.structures.atoms.Atom>` objetcs.

All simulation modules operate on the same 
{py:class}`Structure <udkm1Dsim.structures.structure.Structure>` 
instance, ensuring consistency and reproducibility across different calculations.

The simulation framework is highly modular. Each module returns standard `numpy` 
arrays that can be inspected, modified, or used directly as input for 
subsequent simulations, allowing users to build customized workflows with ease.

:::
::::

## Definitions

:::{attention}
The sketched experimental geometry defines the coordinate system and angle 
conventions, which are used consistently throughout **udkm1Dsim**.
:::

::::{grid} 1 1 2 2
:gutter: 4
:reverse:

:::{grid-item}
```{image} _static/experimental_scheme.png
:width: 400px
:class: sd-m-auto
```
:::

:::{grid-item}

The incoming and outgoing wavevectors $\vec{k}_\text{in}$ and
$\vec{k}_\text{out}$ define the scattering plane which must be co-planar
to the sample surface, hence only accessing scattering vectors $\vec{q_z}$
along the surface normal.
The pointing of the magnetization vector $\vec{m}$ is defined by the two
angles $\phi$ and $\gamma$ and allows any direction in the
$xyz$ coordinate system.

The sample structure is defined from top to bottom. Accordingly the distances
with in the sample structure start at its surface and proceed along
$-z$ direction.

:::

::::

## Community

::::{grid} 1 2 2 4
:gutter: 1 1 1 2

:::{grid-item-card} {material-regular}`photo_library;2em` Examples
:link: examples/index
:link-type: doc

Explore a growing collection of user-inspired Jupyter notebook examples.

:::

:::{grid-item-card} {material-regular}`school;2em` Publications
:link: publications
:link-type: doc

Discover scientific publications that use the toolbox in real-world research.

:::

:::{grid-item-card} {material-regular}`forum;2em` Discuss
:link: https://github.com/dschick/udkm1Dsim/discussions

Ask questions, exchange ideas, and discuss new features with the community.

:::

:::{grid-item-card} {material-regular}`live_help;2em` Issues
:link: https://github.com/dschick/udkm1Dsim/issues

Found a bug or have a feature request? We'd love to hear from you.

:::

::::
