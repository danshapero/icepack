# Copyright 2020 Daniel Shapero

from abc import ABC, abstractmethod
import firedrake
from firedrake import assemble, Constant, dot, inner, det, grad, div, dx, ds
from .utilities import default_solver_parameters


class MongeAmpereSolver(ABC):
    @abstractmethod
    def step(self, timestep):
        pass

    def compute_coordinates(self):
        r"""Calculate the coordinate field resulting from the potential"""
        mesh = self.potential.ufl_domain()
        V = mesh.coordinates.function_space()
        x = firedrake.SpatialCoordinate(mesh)
        return firedrake.interpolate(x + grad(self.potential), V)


class PseudoTimeMarchingSolver(MongeAmpereSolver):
    def __init__(self, monitor, solver_parameters=default_solver_parameters):
        r"""Solve the Monge-Ampere equation using G Awanou's pseudo-time
        marching scheme"""
        mesh = monitor.ufl_domain()
        area = assemble(Constant(1.0) * dx(mesh))

        Φ = firedrake.FunctionSpace(mesh, 'CG', 2)
        Σ = firedrake.TensorFunctionSpace(mesh, 'CG', 2)

        δt = Constant(1.0)

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

        constant_fns = firedrake.VectorSpaceBasis(constant=True)
        problem_φ = firedrake.NonlinearVariationalProblem(F_φ, φ)
        solver_φ = firedrake.NonlinearVariationalSolver(
            problem_φ,
            nullspace=constant_fns,
            solver_parameters=solver_parameters
        )

        problem_σ = firedrake.NonlinearVariationalProblem(F_σ, σ)
        solver_σ = firedrake.NonlinearVariationalSolver(
            problem_σ, solver_parameters=solver_parameters
        )

        self.potential = φ
        self.hessian = σ
        self.timestep = δt
        self.potential_solver = solver_φ
        self.hessian_solver = solver_σ

        self._potential_0 = φ_0
        self._rhs = f
        self._area = area
        self._normalization = μ

    def step(self, timestep):
        self._potential_0.assign(self.potential)
        self.timestep.assign(timestep)
        self._normalization.assign(assemble(self._rhs * dx) / self._area)
        self.potential_solver.solve()
        self.hessian_solver.solve()


class ParabolicSolver(MongeAmpereSolver):
    def __init__(
        self,
        monitor,
        smoothing=None,
        solver_parameters=default_solver_parameters
    ):
        r"""Solve the Monge-Ampere equation using C Budd's parabolic
        relaxation method"""
        mesh = monitor.ufl_domain()

        Φ = firedrake.FunctionSpace(mesh, 'CG', 2)
        Σ = firedrake.TensorFunctionSpace(mesh, 'CG', 2)

        δt = Constant(1.0)

        d = mesh.geometric_dimension()
        if smoothing is None:
            volume = assemble(Constant(1.0) * dx(mesh))
            smoothing = (volume / 2) ** (1 / d)
        λ = Constant(smoothing)

        φ, σ = firedrake.Function(Φ), firedrake.Function(Σ)
        ψ, τ = firedrake.TestFunction(Φ), firedrake.TestFunction(Σ)

        φ_0, σ_0 = firedrake.Function(Φ), firedrake.Function(Σ)

        I = firedrake.Identity(d)
        F_ϕ = (
            ((ϕ - ϕ_0) * ψ + λ**2 * inner(grad(ϕ - ϕ_0), grad(ψ))) -
            δt * (monitor * abs(det(I + σ))) ** (1 / d) * ψ
        ) * dx

        F_σ = (
            (inner(σ - σ_0, τ) + λ**2 * inner(grad(σ - σ_0), grad(τ))) +
            δt * (inner(grad(ϕ_0), div(τ)) + inner(σ, τ))
        ) * dx

        problem_ϕ = firedrake.NonlinearVariationalProblem(F_ϕ, ϕ)
        solver_ϕ = firedrake.NonlinearVariationalSolver(
            problem_ϕ, solver_parameters=solver_parameters
        )

        problem_σ = firedrake.NonlinearVariationalProblem(F_σ, σ)
        solver_σ = firedrake.NonlinearVariationalSolver(
            problem_σ, solver_parameters=solver_parameters
        )

        self.potential = φ
        self.hessian = σ
        self.timestep = δt
        self.potential_solver = solver_φ
        self.hessian_solver = solver_σ

        self._potential_0 = φ_0
        self._hessian_0 = σ_0

    def step(self, timestep):
        self._potential_0.assign(self.potential)
        self._hessian_0.assign(self.hessian)

        self.timestep.assign(timestep)
        self.hessian_solver.solve()
        self.potential_solver.solve()
