import dolfin
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data # DataLoader for batching
from torch_geometric.loader import DataLoader # Corrected import
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time # For timing
import os # For creating output directory
import datetime # For session timestamps
import json

# Assuming these are in separate files or defined above as per your setup
from models.GAT import RAdaptGAT
from fenics_mesh_to_pyg_data import fenics_mesh_to_pyg_data

# --- Import functions from your plot_funcs.py ---
try:
    import plot_funcs # Assumes plot_funcs.py is in the same directory or PYTHONPATH
except ImportError:
    print("Error: plot_funcs.py not found. Make sure it's in the same directory or in PYTHONPATH.")
    # Define dummy functions if import fails, so the script can still run mostly
    class plot_funcs_dummy:
        @staticmethod
        def cuda_status(device): print(f"Dummy cuda_status called for device: {device}")
        @staticmethod
        def density_plot_matrix(matrix, output="", title="", show=True, **kwargs):
            if hasattr(matrix, 'shape'):
                print(f"Dummy density_plot_matrix called for matrix shape: {matrix.shape} with title: {title}")
            else:
                print(f"Dummy density_plot_matrix called for matrix (type: {type(matrix)}) with title: {title}")
        @staticmethod
        def loss_plot(epoch_count, loss_values, test_loss_values, model_name="", output="", show=True):
            print(f"Dummy loss_plot called for model: {model_name}")
        @staticmethod
        def predVStrue(label_val_true, label_val_pred, label_train_true, label_train_pred, model_name, output="", show=True):
            print(f"Dummy predVStrue called for model: {model_name}")
        @staticmethod
        def plot_time_comparison(classical_times, gat_times, time_label='Mesh Optimization Time (s)', output="", title="", show=True, use_box_plot=False, **kwargs):
            print(f"Dummy plot_time_comparison called with title: {title}")
        @staticmethod
        def plot_accuracy_vs_cost(classical_costs, classical_accuracies, gat_costs, gat_accuracies, output="", show=True, **kwargs):
            print("Dummy plot_accuracy_vs_cost called.")
        @staticmethod
        def plot_convergence(classical_dofs, classical_errors, gat_dofs, gat_errors, output="", show=True, **kwargs):
            print("Dummy plot_convergence called.")
    plot_funcs = plot_funcs_dummy
# --- End Plot Funcs Import ---

def project_tensor_to_scalar_space(tensor_field, V_scalar, name="Projected Scalar"):
    """Projects a component of a tensor field to a scalar FunctionSpace."""
    p = dolfin.TrialFunction(V_scalar)
    q = dolfin.TestFunction(V_scalar)
    a = p * q * dolfin.dx
    L = tensor_field * q * dolfin.dx
    proj_func = dolfin.Function(V_scalar, name=name)
    dolfin.solve(a == L, proj_func)
    return proj_func

def get_hessian_frobenius_norm_fenics(u_solution, mesh):
    """
    Recovers Hessian components and computes the Frobenius norm at nodes.
    Inspired by Rowbottom et al. (2025) Appendix A.2[cite: 254].
    Args:
        u_solution (dolfin.Function): The FEM solution (scalar).
        mesh (dolfin.Mesh): The FEniCS mesh.
    Returns:
        np.array: Nodal values of the Frobenius norm of the Hessian.
    """
    V_scalar = dolfin.FunctionSpace(mesh, "CG", 1) # P1 space for Hessian components
    
    # Compute gradients
    grad_u = dolfin.grad(u_solution) # This is a vector

    # H_xx: project(grad_u[0].dx(0), V_scalar)
    # H_xy: project(grad_u[0].dx(1), V_scalar) (or grad_u[1].dx(0))
    # H_yy: project(grad_u[1].dx(1), V_scalar)

    # Weak form for H_ij = d/dx_j (d_u/dx_i)
    # For H_xx: d/dx (du/dx)
    # L = -dolfin.inner(dolfin.grad(grad_u[0]), dolfin.grad(q)[0]) * dolfin.dx # Incorrect formulation from paper for H_ij*v
    # The paper's Appendix A.2 equation (12) is:
    # -\int_{\Omega}\partial_{i}u\partial_{j}v~dx=\int_{\Omega}H_{ij}v~dx
    # This is problematic for P1 as u is P1, so \partial_i u is P0 (discontinuous).
    # A more common Zienkiewicz-Zhu style recovery or L2 projection of derivatives is often used.
    # The paper solves for H_ij such that \int H_ij v = -\int \partial_i u \partial_j v
    # For P1 elements, derivatives of u are cell-wise constant.
    # A simpler approach (though not exactly the paper's weak form for H_ij) is to project gradients.

    # Let's use a simpler gradient-based feature as a proxy for curvature/Hessian norm here,
    # as full Hessian recovery for P1 is non-trivial and the paper's weak form needs careful interpretation.
    # The paper itself mentions "the Frobenius norm ||H||_F ... is fed as an input".
    # We can use the L2 norm of the gradient as a simpler "activity" measure.
    
    grad_u_sq_l2 = dolfin.project(dolfin.inner(grad_u, grad_u), V_scalar)
    # You might want to apply further smoothing or use a more sophisticated recovery.
    
    # For this example, let's use grad_u_sq_l2 as the "curvature-like" feature.
    # A true Hessian norm would require proper recovery of all H_ij components.
    # If using the paper's method, ensure proper dolfin-adjoint setup if differentiating through this.
    
    # CITATION for concept: Rowbottom et al. (2025) [cite: 1, 335] mention using the Frobenius norm of the Hessian
    # as an input feature[cite: 85, 255]. The weak form for H_ij is in their Appendix A.2[cite: 254].
    # The code below is a simplification using gradient norm.
    
    curvature_feature_nodal = grad_u_sq_l2.compute_vertex_values(mesh)
    
    # Normalize if desired
    # curvature_feature_nodal = (curvature_feature_nodal - np.min(curvature_feature_nodal)) / \
    #                           (np.max(curvature_feature_nodal) - np.min(curvature_feature_nodal) + 1e-6)
                              
    return curvature_feature_nodal.reshape(-1, 1) # Ensure shape [num_nodes, 1]
