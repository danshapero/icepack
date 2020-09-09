import firedrake
from firedrake import sqrt, exp, inner, Constant
import icepack

def test_ring_monitor():
    mesh = firedrake.UnitDiskMesh(4)

    x = firedrake.SpatialCoordinate(mesh)
    r = sqrt(inner(x, x))
    R = Constant(1/2)
    a = Constant(1/8)
    c = Constant(3.)

    def sech(z):
        return 2 * exp(z) / (exp(2 * z) + 1)

    monitor = 1 + (c - 1) * sech((r - R) / a)

    X = icepack.adapt(monitor, timestep=1e-2, time=2.)
    new_mesh = firedrake.Mesh(X)
