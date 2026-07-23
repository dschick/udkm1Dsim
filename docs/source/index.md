---
sd_hide_title: true
---

# Overview

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #222533 0%, #3C415A 89%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} ./_static/logo.png
:width: 200px
:class: sd-m-auto sd-rounded-circle sd-shadow-sm
```
:::


:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-text-white sd-fs-3

Welcome to the **udkm1Dsim** toolbox.



:::{button-ref} getting_started
:ref-type: doc
:outline:
:color: white
:class: sd-px-4 sd-fs-5

Get Started
:::

:::
::::

::::::

-----------

:::{card}
The **udkm1Dsim** toolbox is a collection of Python classes and routines to simulate the thermal, structural, and magnetic dynamics after laser excitation as well as the according X-ray scattering response in one-dimensional sample structures after ultrafast excitation.
:::

-----------

::::{grid} 1 2 2 4
:gutter: 1 1 1 2

:::{grid-item-card} {material-regular}`photo_library;2em` Examples
:link: auto_examples/index
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

-----------

If you use **udkm1Dsim** in your research, please cite the latest publication:

:::{admonition} Citation
:class: seealso

D. Schick, *udkm1Dsim - A Python toolbox for simulating 1D ultrafast dynamics in condensed matter*,
[Comput. Phys. Commun. 266, 108031 (2021)](https://doi.org/10.1016/j.cpc.2021.108031). ([preprint](https://arxiv.org/abs/2102.12144))

:::

If your work is based on the original implementation of **udkm1Dsim** in MATLAB©, you may also cite the original publication:

:::{admonition} Citation
:class: seealso

D. Schick, A. Bojahr, M. Herzog, R. Shayduk, C. von Korff Schmising & M. Bargheer,
*udkm1Dsim - A Simulation Toolkit for 1D Ultrafast Dynamics in Condensed Matter*,
[Comput. Phys. Commun. 185, 651 (2014)](http://doi.org/10.1016/j.cpc.2013.10.009). ([preprint](_static/udkm1DsimManuscriptPrePrint.pdf))

:::

```{toctree}
:caption: General
:hidden:
:maxdepth: 1

getting_started
Examples <auto_examples/index>
publications
```

```{toctree}
:caption: Reference
:hidden:
:titlesonly:
:maxdepth: 1

api
references
changelog
Project Page <https://github.com/dschick/udkm1Dsim>
```
