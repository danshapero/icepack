import numpy as np
from numpy import pi as π
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import firedrake
from firedrake import (
    Constant, interpolate, inner, grad, dx, assemble, derivative, avg, jump
)
from firedrake.petsc import PETSc


a = 1.0
b = (-1/4, +1/4)
c = 1/8
k = (6, 2)
f = 1/5


def true_field(x):
    if isinstance(x, firedrake.SpatialCoordinate):
        A = Constant(a)
        B = Constant(b)
        C = Constant(c)
        K = Constant(k)
        F = Constant(f)
        return A + inner(B, x) + C * firedrake.cos(π * (inner(K, x) + F))

    B = np.array(b)
    K = np.array(k)
    return a + x @ b + c * np.cos(π * (x @ k + f))


# Make a mesh
nx = 32
mesh = firedrake.UnitSquareMesh(nx, nx, diagonal="crossed")

# Make a point cloud and some synthetic observations
xs = np.array([[1/3, 1/3], [2/3, 1/3], [2/3, 2/3], [1/3, 2/3]])
point_cloud = firedrake.VertexOnlyMesh(mesh, xs, reorder=False)
raw_data = true_field(xs)

# Make function spaces on the mesh and the point cloud
Q = firedrake.FunctionSpace(mesh, "CG", 1)
p = firedrake.Function(Q)

D = firedrake.FunctionSpace(point_cloud, "DG", 0)
p_obs = firedrake.Function(D)
p_obs.dat.data[:] = raw_data

# Make the operator that interpolates functions on the mesh to the point cloud
q = firedrake.TrialFunction(Q)
I = assemble(interpolate(q, D)).M.handle
assert I.getSize() == (len(xs), Q.dim())

# Make the right-hand side
F = PETSc.Vec().createSeq(Q.dim())
with p_obs.dat.vec_ro as P_obs:
    I.multTranspose(P_obs, F)

# Make the regularization matrix
R = 0.5 * inner(grad(p), grad(p)) * dx
A = assemble(derivative(derivative(R, p), p)).M.handle

# Form the matrix product `tranpose(I) * I`
It = PETSc.Mat().createTranspose(I)
IIt = It.matMult(I)

# Form the system matrix = sum of regularization + interpolation round-trip
H = A.copy()
H.axpy(1.0, IIt)

# Solve the linear system
ksp = PETSc.KSP().create()
ksp.setOperators(H)
ksp.setFromOptions()

with p.dat.vec as P:
    ksp.solve(F, P)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
firedrake.trisurf(p, axes=ax)
plt.show()

x = firedrake.SpatialCoordinate(mesh)
p_true = firedrake.Function(Q).interpolate(true_field(x))

print(firedrake.norm(p - p_true))
