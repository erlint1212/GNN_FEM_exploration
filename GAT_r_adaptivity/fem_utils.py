# fem_utils.py
import dolfin
import numpy as np
# Import necessary strings from config, or pass them as arguments
from config import (F_EXPRESSION_STR_SQUARE, U_EXACT_EXPRESSION_STR_SQUARE,
                    U_DIRICHLET_EXPRESSION_STR_SQUARE, EXACT_SOL_DEGREE_SQUARE,
                    F_EXPRESSION_STR_PIPE, U_EXACT_EXPRESSION_STR_PIPE,
                    U_DIRICHLET_EXPRESSION_STR_PIPE, EXACT_SOL_DEGREE_PIPE,
                    PIPE_LENGTH, PIPE_HEIGHT) # Add other needed config vars

def solve_fem_problem(mesh, mesh_type, mesh_dimensions=None):
    """
    Solves a Poisson problem on the given mesh.
    -Laplace(u) = f
    u = u_D on boundary
    """
    V = dolfin.FunctionSpace(mesh, 'P', 1) # Numerical solution is P1

    if mesh_type == 'square':
        f_expr_str = F_EXPRESSION_STR_SQUARE
        u_d_expr_str = U_DIRICHLET_EXPRESSION_STR_SQUARE
        f = dolfin.Expression(f_expr_str, degree=2, user_pi=dolfin.pi)
        u_D = dolfin.Expression(u_d_expr_str, degree=2, user_pi=dolfin.pi)
    elif mesh_type == 'pipe':
        L_pipe = mesh_dimensions.get("width", PIPE_LENGTH) if mesh_dimensions else PIPE_LENGTH
        H_pipe = mesh_dimensions.get("height", PIPE_HEIGHT) if mesh_dimensions else PIPE_HEIGHT
        f = dolfin.Expression(F_EXPRESSION_STR_PIPE, degree=2, L=L_pipe, H=H_pipe)
        u_D = dolfin.Expression(U_DIRICHLET_EXPRESSION_STR_PIPE, degree=2, L=L_pipe, H=H_pipe)
    else:
        raise ValueError(f"Unknown mesh type for FEM solve: {mesh_type}")

    def boundary(x, on_boundary):
        return on_boundary
    bc = dolfin.DirichletBC(V, u_D, boundary)

    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)
    a = dolfin.dot(dolfin.grad(u), dolfin.grad(v)) * dolfin.dx
    L_form = f * v * dolfin.dx

    u_sol = dolfin.Function(V)
    try:
        dolfin.solve(a == L_form, u_sol, bc)
    except Exception as e:
        print(f"FEniCS solver failed: {e}. Returning None for solution.")
        return None
    return u_sol

def calculate_l2_error(u_numerical, mesh_type, mesh_dimensions=None, mesh=None):
    """
    Calculates the L2 error of the numerical solution against an exact solution.
    Returns -1.0 if no exact solution is defined for the mesh_type, -2.0 if u_numerical is None.
    """
    if u_numerical is None:
        return -2.0 # Indicates solver failure for the numerical solution

    current_mesh = mesh or u_numerical.function_space().mesh()

    if mesh_type == 'square':
        u_exact_str = U_EXACT_EXPRESSION_STR_SQUARE
        exact_degree = EXACT_SOL_DEGREE_SQUARE
        u_exact = dolfin.Expression(u_exact_str, degree=exact_degree, user_pi=dolfin.pi)
    elif mesh_type == 'pipe':
        if U_EXACT_EXPRESSION_STR_PIPE is None:
            return -1.0 # Special value indicating no exact solution for error calculation
        u_exact_str = U_EXACT_EXPRESSION_STR_PIPE
        exact_degree = EXACT_SOL_DEGREE_PIPE
        L_pipe = mesh_dimensions.get("width", PIPE_LENGTH) if mesh_dimensions else PIPE_LENGTH
        H_pipe = mesh_dimensions.get("height", PIPE_HEIGHT) if mesh_dimensions else PIPE_HEIGHT
        u_exact = dolfin.Expression(u_exact_str, degree=exact_degree, L=L_pipe, H=H_pipe, user_pi=dolfin.pi)
    else:
        raise ValueError(f"Unknown mesh type for L2 error: {mesh_type}")

    L2_error = dolfin.errornorm(u_exact, u_numerical, 'L2', mesh=current_mesh)
    return L2_error

def get_solution_based_monitor_function(u_solution, mesh):
    """
    Generates a monitor function based on the L2 norm of the gradient of the FEM solution.
    Returns a NumPy array of nodal values.
    """
    if u_solution is None:
        print("Warning: FEM solution is None in get_solution_based_monitor_function. Returning uniform monitor.")
        # Ensure the returned array matches the number of vertices if mesh is valid
        return np.ones(mesh.num_vertices() if mesh else 0) * 0.5

    V_scalar = dolfin.FunctionSpace(mesh, "CG", 1) # P1 space for monitor function values
    grad_u_sq = dolfin.project(dolfin.inner(dolfin.grad(u_solution), dolfin.grad(u_solution)), V_scalar)
    monitor_values_nodal = grad_u_sq.compute_vertex_values(mesh) # Already nodal for CG1

    min_val = np.min(monitor_values_nodal)
    max_val = np.max(monitor_values_nodal)
    if max_val - min_val < 1e-9: # Avoid division by zero if gradient is constant (e.g. zero)
        return np.ones_like(monitor_values_nodal) * 0.5
    
    normalized_monitor_values = (monitor_values_nodal - min_val) / (max_val - min_val)
    return normalized_monitor_values
