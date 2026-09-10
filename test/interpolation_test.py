import numpy as np
from numpy import pi as π
from scipy.stats.qmc import PoissonDisk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import firedrake
from firedrake import (
    Constant, interpolate, inner, dot, grad, dx, ds, dS, assemble, derivative, avg, jump
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
rng = np.random.default_rng(seed=1729)
radius = 0.1
sampler = PoissonDisk(2, radius=radius, rng=rng)
xs = sampler.fill_space()
point_cloud = firedrake.VertexOnlyMesh(mesh, xs, reorder=False)
raw_data = true_field(xs)
print(f"Number of samples: {len(raw_data)}")

# Make function spaces on the mesh and the point cloud
Q = firedrake.FunctionSpace(mesh, "CG", 1)
S = firedrake.FunctionSpace(mesh, "HHJ", 0)
Z = Q * S

D = firedrake.FunctionSpace(point_cloud, "DG", 0)
p_obs = firedrake.Function(D)
p_obs.dat.data[:] = raw_data

# Make the operator that interpolates functions on the mesh to the point cloud
q, _ = firedrake.TrialFunctions(Z)
I = assemble(interpolate(q, D)).M.handle
assert I.getSize() == (len(xs), Z.dim())

# Make the right-hand side
F = PETSc.Vec().createSeq(Z.dim())
with p_obs.dat.vec_ro as P_obs:
    I.multTranspose(P_obs, F)

# Make the regularization matrix
w = firedrake.Function(Z)
p, s = firedrake.split(w)
α = Constant(1 / 64)
ν = firedrake.FacetNormal(mesh)
R_cells = (0.5 * inner(s, s) - inner(s, grad(grad(p)))) * dx
R_facets = avg(inner(ν, dot(s, ν))) * jump(grad(p), ν) * dS
R_boundary = inner(ν, dot(s, ν)) * inner(grad(p), ν) * ds
R = -α**2 * (R_cells + R_facets + R_boundary)
A = assemble(derivative(derivative(R, w), w)).M.handle

# Form the matrix product `tranpose(I) * I`
It = PETSc.Mat().createTranspose(I)
IIt = It.matMult(I)

# Form the system matrix = sum of regularization + interpolation round-trip
H = A.copy()
H.axpy(1.0, IIt)

# Solve the linear system
ksp = PETSc.KSP().create()
ksp.setOperators(H)
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
ksp.setFromOptions()

with w.dat.vec as W:
    ksp.solve(F, W)

p, s = w.subfunctions

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
firedrake.trisurf(p, axes=ax, alpha=0.5)
ax.scatter(*xs.T, raw_data)
plt.show()

x = firedrake.SpatialCoordinate(mesh)
p_true = firedrake.Function(Q).interpolate(true_field(x))

print(firedrake.norm(p - p_true))
