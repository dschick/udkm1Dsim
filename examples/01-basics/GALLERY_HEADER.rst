Basics
======

Every simulation requires a one-dimensional sample structure.
The sample structure can be build from nested sub-structures but ultimatively is build from layer.
Such layers hold the actual physical material properties which are organized in parameter groups, 
e.g. for the thermal, elastic, optical, and magnetic properties.
A layer is a generic, homogeneous slab of any material, i.e. a solid, a liquid, or a gas, whose internal structure is often not even known.
Only for X-ray diffraction, unit cells with atoms or mixed atoms at their positions are required as most fundamental building blocks.
Their optical properties are given by the dielectric tensor, which follows from dispersions of the dielectric function.
All simulations operate on such a structure and return their results with coordinates, units, and metadata, which can be passed on to the following simulations and saved to HDF5 files.
