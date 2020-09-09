# Copyright 2020 Daniel Shapero

import firedrake
from firedrake import assemble, Constant, dot, inner, det, grad, div, dx, ds

def solve_monge_ampere(monitor, time=1., timestep=1.):
    r"""Approximately solve the Monge-Ampere equation for some positive monitor
    function"""
    mesh = monitor.ufl_domain()
    area = assemble(Constant(1.) * dx(mesh))

    Φ = firedrake.FunctionSpace(mesh, family='CG', degree=2)
    Σ = firedrake.TensorFunctionSpace(mesh, family='CG', degree=2)

    δt = Constant(timestep)

    φ, σ = firedrake.Function(Φ), firedrake.Function(Σ)
    ψ, τ = firedrake.TestFunction(Φ), firedrake.TestFunction(Σ)

    φ_0 = firedrake.Function(Φ)

    I = firedrake.Identity(mesh.geometric_dimension())
    n = firedrake.FacetNormal(mesh)

    f = monitor * det(I + σ)
    μ = Constant(assemble(f * dx) / area)

    F_φ = (inner(grad(φ - φ_0), grad(ψ)) - δt * (f - μ) * ψ) * dx
    F_σ = (
        (inner(σ, τ) + inner(div(τ), grad(φ))) * dx -
        inner(dot(τ, n), grad(φ)) * ds
    )

    parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    constant_fns = firedrake.VectorSpaceBasis(constant=True)
    problem_φ = firedrake.NonlinearVariationalProblem(F_φ, φ)
    solver_φ = firedrake.NonlinearVariationalSolver(
        problem_φ, nullspace=constant_fns, solver_parameters=parameters
    )

    problem_σ = firedrake.NonlinearVariationalProblem(F_σ, σ)
    solver_σ = firedrake.NonlinearVariationalSolver(
        problem_σ, solver_parameters=parameters
    )

    # TODO: Add a try/except loop if the timestep is too small and halve it
    # if something diverges
    num_steps = int(time / timestep)
    for step in range(num_steps):
        μ.assign(assemble(f * dx) / area)
        solver_φ.solve()
        solver_σ.solve()

        φ_0.assign(φ)

    return φ


def adapt(monitor, **kwargs):
    r"""Calculate a new set of coordinates that equidistribute a positive
    monitor function

    The monitor function should be bounded below by some positive constant. The
    ratio of the maximum to the minimum value of the monitor function is what
    sets overall contrast in the sizes of the final mesh cells.
    """
    φ = solve_monge_ampere(monitor, **kwargs)
    mesh = monitor.ufl_domain()
    V = mesh.coordinates.function_space()
    x = firedrake.SpatialCoordinate(mesh)
    return firedrake.interpolate(x + grad(φ), V)
