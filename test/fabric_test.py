import firedrake
from firedrake import outer, as_vector, as_tensor, Constant
import icepack
import numpy as np

def test_transport():
    nx, ny = 32, 32
    Lx, Ly = 20e3, 20e3
    mesh = firedrake.RectangleMesh(nx, ny, Lx, Ly)

    V = firedrake.VectorFunctionSpace(mesh, "CG", 2)
    S = firedrake.TensorFunctionSpace(mesh, "DG", 0, symmetry=True)

    u = firedrake.interpolate(Constant((1.0, 0.0)), V)

    x = firedrake.SpatialCoordinate(mesh)
    f = firedrake.project((outer(x, x)), S)
    f_inflow = f.copy(deepcopy=True)

    model = icepack.models.FabricTransport()
    flux = model.flux(fabric=f, velocity=u, fabric_inflow=f_inflow)
