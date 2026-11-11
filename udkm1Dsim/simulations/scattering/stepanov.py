#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2020 Daniel Schick
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

"""Multilayer formalism with the analytical eigenmodes of
:cite:t:`elzo2012` and the numerically stable scattering-matrix recursion of
:cite:t:`stepanov2000`, as in
`Dyna <https://gitlab.com/dynadevgroup/Dyna>`_."""

__all__ = ["Stepanov"]

__docformat__ = "restructuredtext"


from .elzo import _ElzoModes
from .multilayer import MultilayerScattering


class Stepanov(_ElzoModes, MultilayerScattering):
    r"""Stepanov

    Multilayer scattering simulations with the analytical eigenmodes of
    :cite:t:`elzo2012`, see
    :func:`~udkm1Dsim.simulations.scattering.elzo.elzo_modes`, which are
    assembled by the scattering-matrix recursion of :cite:t:`stepanov2000`
    instead of transfer matrices. Hence, it gives the same results as
    :class:`~udkm1Dsim.simulations.scattering.elzo.Elzo` for homogeneous
    layers, but stays numerically stable for thick, strongly absorbing, or
    highly repeated stacks, e.g. for visible light. Ported from
    `Dyna <https://gitlab.com/dynadevgroup/Dyna>`_.

    The dielectric tensor of every layer must be isotropic except for its
    gyrotropic part, and a :class:`.UnitCell` is treated as homogeneous layer
    with its dielectric tensor.

    Args:
        S (Structure): sample to do simulations with.
        force_recalc (boolean): force recalculation of results.

    Keyword Args:
        save_data (boolean): true to save simulation results.
        cache_dir (str): path to cached data.
        disp_messages (boolean): true to display messages from within the
            simulations.
        progress_bar (boolean): enable tqdm progress bar.

    """

    _title = "Stepanov multilayer scattering simulation"
    _assembly = "scattering"
