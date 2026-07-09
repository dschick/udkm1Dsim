User Guide
==========

The **udkm1Dsim** toolbox comes as a Python package which can be easily installed
with all of its dependencies.

:::{hint}
In case you encounter any issues during the installation or usage of the
package, please open an 
[issue at GitHub](https://github.com/dschick/udkm1Dsim/issues).
:::

The internal structure of the **udkm1Dsim** toolbox is sketched in the lower figure:

```{image} _static/structure.png
:alt: Structure of the udkm1Dsim toolbox
:width: 500px
:align: center
```
</br>

The common experimental scheme is sketched below and shows the general
definition of coordinates and angles that are used throughout the **udkm1Dsim**
toolbox.
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

```{image} _static/experimental_scheme.png
:alt: General experimental scheme
:width: 500px
:align: center
```

:::{hint}
To get started, please work yourself through the [Examples](./examples.rst).  
For more detailed information, also regarding the physics, you might refer to
the [API Documentation](./api.rst).
:::