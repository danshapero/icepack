# Copyright (C) 2021 by Daniel Shapero <shapero@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

r"""Description of the fabric transport model

This model contains the specification for a phenomenological model of how ice
crystal fabric evolves under sustained strain.
"""

from operator import itemgetter
import firedrake
from firedrake import inner, grad, div, dx, ds, dS, min_value, max_value


class FabricTransport:
    def __init__(
        self,
        # Your physical parameters go here!
    ):
        # Set this model object's physical parameters
        pass

    def flux(self, **kwargs):
        keys = ("fabric, velocity", "fabric_inflow")  # Other fields if need be
        f, u, f_inflow = itemgetter(*keys)(kwargs)

        S = f.function_space()
        φ = firedrake.TestFunction(S)

        mesh = S.mesh()
        n = firedrake.FacetNormal(mehs)

        u_n = max_value(0, inner(u, n))
        F = f * u_n
        flux_faces = inner(F("+") - F("-"), φ("+") - φ("-")) * dS
        flux_cells = -inner(f, div(u * φ)) * dx  # <- this is wrong
        flux_out = inner(f, max_value(0, inner(u, n) * φ)) * ds
        flux_in = inner(f_inflow, min_value(0, inner(u, n) * φ)) * ds

        return flux_faces + flux_cells + flux_out + flux_in

    def sources(self, **kwargs):
        keys = ("fabric", "velocity", "strain_rate", "membrane_stress")  # or w/e
        f, u, ε, M = itemgetter(*keys)(kwargs)

        # Magic goes here!

        return firedrake.Constant(0.0)
