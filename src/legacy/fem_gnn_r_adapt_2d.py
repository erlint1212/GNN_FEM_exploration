# fem_gnn_r_adapt_2d.py
import dolfin
import torch
import torch.optim as optim
# import torch.nn.functional as F # Not directly used in this script's main flow after GAT model
from torch_geometric.loader import DataLoader
import numpy as np
# import itertools # Moved to fenics_mesh_to_pyg_data
import matplotlib.pyplot as plt
import time
import os
import datetime
import json

# --- Import from our new mesh generation script ---
try:
    from mesh_generators_2 import create_square_mesh, create_pipe_with_obstacle_mesh_gmsh
except ImportError as e:
    print("CRITICAL ERROR during import of mesh_generators_2.py. Original error was:")
    print(e)
    print("\nMake sure mesh_generators_2.py is in the same directory and that all its internal imports (like 'gmsh' or 'dolfin') are working correctly within your FEniCS environment.")
    exit()

from models.GAT import RAdaptGAT
from fenics_mesh_to_pyg_data import fenics_mesh_to_pyg_data

# --- Import functions from your plot_funcs.py ---
try:
    import plot_funcs
except ImportError:
    print("Error: plot_funcs.py not found. Make sure it's in the same directory or in PYTHONPATH.")
    class plot_funcs_dummy:
        @staticmethod
        def cuda_status(device): print(f"Dummy cuda_status called for device: {device}")
        @staticmethod
        def density_plot_matrix(matrix, output="", title="", show=True, **kwargs):
            if hasattr(matrix, 'shape'): print(f"Dummy density_plot_matrix for matrix shape: {matrix.shape}, title: {title}")
            else: print(f"Dummy density_plot_matrix for matrix type: {type(matrix)}, title: {title}")
        @staticmethod
        def loss_plot(epoch_count, loss_values, test_loss_values, model_name="", output="", show=True): print(f"Dummy loss_plot for model: {model_name}")
        @staticmethod
        def predVStrue(label_val_true, label_val_pred, label_train_true, label_train_pred, model_name, output="", show=True): print(f"Dummy predVStrue for model: {model_name}")
        @staticmethod
        def plot_time_comparison(classical_times, gat_times, time_label='Mesh Optimization Time (s)', output="", title="", show=True, use_box_plot=False, **kwargs): print(f"Dummy plot_time_comparison with title: {title}")
        @staticmethod
        def plot_accuracy_vs_cost(classical_costs, classical_accuracies, gat_costs, gat_accuracies, cost_label='Nodes', error_label='L2 Error', output="", show=True, **kwargs): print("Dummy plot_accuracy_vs_cost called.")
        @staticmethod
        def plot_convergence(classical_dofs, classical_errors, gat_dofs, gat_errors, dof_label='DoFs', error_label='L2 Error', output="", show=True, **kwargs): print("Dummy plot_convergence called.")
    plot_funcs = plot_funcs_dummy
# --- End Plot Funcs Import ---

# --- Global Parameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-3
EPOCHS = 200
BATCH_SIZE = 8
NUM_SAMPLES = 500

# --- Mesh Type Selector ---
MESH_TYPE = 'square' # 'square' or 'pipe'

# --- FEM Problem Parameters (Poisson Equation) ---
F_EXPRESSION_STR_SQUARE = "2*pow(user_pi,2)*sin(user_pi*x[0])*sin(user_pi*x[1])"
U_EXACT_EXPRESSION_STR_SQUARE = "sin(user_pi*x[0])*sin(user_pi*x[1])"
U_DIRICHLET_EXPRESSION_STR_SQUARE = "0.0"

F_EXPRESSION_STR_PIPE = "10 * exp(-(pow(x[0] - 0.5*L, 2) + pow(x[1] - 0.5*H, 2)) / (2*pow(0.1*H,2)))"
U_EXACT_EXPRESSION_STR_PIPE = None
U_DIRICHLET_EXPRESSION_STR_PIPE = "0.0"

# --- Parameters specific to MESH_TYPE ---
if MESH_TYPE == 'square':
    MODEL_NAME_SUFFIX = "SquareMesh_Poisson"
    MESH_SIZE_MIN = 8
    MESH_SIZE_MAX = 16
    FEATURE_CENTER_X_FACTOR = 0.5
    FEATURE_CENTER_Y_FACTOR = 0.5
elif MESH_TYPE == 'pipe':
    MODEL_NAME_SUFFIX = "PipeObstacleMesh_Poisson"
    MESH_SIZE_FACTOR_MIN = 0.1
    MESH_SIZE_FACTOR_MAX = 0.25
    PIPE_LENGTH = 3.0
    PIPE_HEIGHT = 1.0
    OBSTACLE_CENTER_X_FACTOR = 0.3
    OBSTACLE_CENTER_Y_FACTOR = 0.5
    OBSTACLE_RADIUS_FACTOR = 0.15
else:
    raise ValueError(f"Unknown MESH_TYPE: {MESH_TYPE}. Choose 'square' or 'pipe'.")

MODEL_NAME = f"RAdaptGAT_{MODEL_NAME_SUFFIX}"
BASE_OUTPUT_DIR = f"gat_{MODEL_NAME_SUFFIX.lower()}_outputs"

# GNN Model Hyperparameters
HIDDEN_CHANNELS = 64*2
OUT_CHANNELS = 2
HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.5

# --- FEM Solver Functions ---
def solve_fem_problem(mesh, mesh_type, mesh_dimensions=None):
    V = dolfin.FunctionSpace(mesh, 'P', 1) # Numerical solution is P1

    if mesh_type == 'square':
        f_expr_str = F_EXPRESSION_STR_SQUARE
        u_d_expr_str = U_DIRICHLET_EXPRESSION_STR_SQUARE
        # For Expression, degree should be high enough for the functions involved
        # For f = 2*pi^2*sin(pi*x)*sin(pi*y), degree=2 for Expression is fine if it means it can represent it,
        # but higher is safer if JIT compiler is used (though elements are P1).
        # The crucial degree is for the exact solution in errornorm.
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
    if u_numerical is None:
        return -2.0

    current_mesh = mesh or u_numerical.function_space().mesh()
    
    # Increase degree of exact solution Expression for more accurate error calculation
    # If u_numerical is P1, u_exact degree should ideally be higher.
    # degree=5 is a safer choice than degree=3 for P1 numerical solutions.
    EXACT_SOL_DEGREE = 5 # CHANGED from 3

    if mesh_type == 'square':
        u_exact_str = U_EXACT_EXPRESSION_STR_SQUARE
        u_exact = dolfin.Expression(u_exact_str, degree=EXACT_SOL_DEGREE, user_pi=dolfin.pi)
    elif mesh_type == 'pipe':
        if U_EXACT_EXPRESSION_STR_PIPE is None:
            return -1.0
        L_pipe = mesh_dimensions.get("width", PIPE_LENGTH) if mesh_dimensions else PIPE_LENGTH
        H_pipe = mesh_dimensions.get("height", PIPE_HEIGHT) if mesh_dimensions else PIPE_HEIGHT
        u_exact = dolfin.Expression(U_EXACT_EXPRESSION_STR_PIPE, degree=EXACT_SOL_DEGREE, L=L_pipe, H=H_pipe, user_pi=dolfin.pi)
    else:
        raise ValueError(f"Unknown mesh type for L2 error: {mesh_type}")

    L2_error = dolfin.errornorm(u_exact, u_numerical, 'L2', mesh=current_mesh)
    return L2_error

# --- Monitor Function (Solution-Based) & R-Adaptivity ---
def get_solution_based_monitor_function(u_solution, mesh):
    if u_solution is None:
        print("Warning: FEM solution is None in get_solution_based_monitor_function. Returning uniform monitor.")
        return np.ones(mesh.num_vertices()) * 0.5

    V_scalar = dolfin.FunctionSpace(mesh, "CG", 1)
    grad_u_sq = dolfin.project(dolfin.inner(dolfin.grad(u_solution), dolfin.grad(u_solution)), V_scalar)
    monitor_values_nodal = grad_u_sq.compute_vertex_values(mesh)

    min_val = np.min(monitor_values_nodal)
    max_val = np.max(monitor_values_nodal)
    if max_val - min_val < 1e-9:
        return np.ones_like(monitor_values_nodal) * 0.5

    normalized_monitor_values = (monitor_values_nodal - min_val) / (max_val - min_val)
    return normalized_monitor_values

def dummy_classical_r_adaptivity(mesh, monitor_values, strength=0.05, mesh_dimensions=None):
    old_coords = np.copy(mesh.coordinates())
    new_coords = np.copy(old_coords)
    num_nodes = mesh.num_vertices()
    geo_dim = mesh.geometry().dim()

    boundary_nodes = set()
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    dolfin.DomainBoundary().mark(boundary_markers, 1)

    for f in dolfin.facets(mesh):
        if boundary_markers[f.index()] == 1:
            for v_idx in f.entities(0):
                boundary_nodes.add(v_idx)

    for i in range(num_nodes):
        if i in boundary_nodes:
            continue

        direction_vector = np.zeros(geo_dim)
        total_weight = 0.0
        for j in range(num_nodes):
            if i == j: continue
            diff = old_coords[j] - old_coords[i]
            dist_sq = np.sum(diff**2)
            if dist_sq < 1e-12: continue

            weight = (monitor_values[j] + monitor_values[i]) / 2.0 / (dist_sq + 1e-6)
            direction_vector += weight * diff
            total_weight += weight

        if total_weight > 1e-6:
            displacement = strength * (direction_vector / total_weight)
            new_coords[i] += displacement

    if mesh_dimensions:
        min_x, max_x = 0.0, mesh_dimensions.get("width", 1.0)
        min_y, max_y = 0.0, mesh_dimensions.get("height", 1.0)

        new_coords[:, 0] = np.clip(new_coords[:, 0], min_x, max_x)
        new_coords[:, 1] = np.clip(new_coords[:, 1], min_y, max_y)

        if mesh_dimensions.get("type") == "pipe_gmsh_meshio" and mesh_dimensions.get("obstacle_radius", 0) > 0: # or "pipe_gmsh_direct_msh"
            pass

    return new_coords

# --- Data Generation (Modified to use MESH_TYPE and FEM solves) ---
def generate_dataset(num_samples, session_output_dir, plot_first_sample_details=False):
    dataset, all_classical_times, all_initial_errors, all_classical_adapted_errors = [], [], [], []
    print(f"Generating {num_samples} data samples for MESH_TYPE: '{MESH_TYPE}'...")

    generated_count = 0
    attempts = 0
    max_attempts = num_samples * 3

    while generated_count < num_samples and attempts < max_attempts:
        attempts += 1
        print(f"\nAttempt {attempts} for sample {generated_count + 1}...")
        initial_mesh = None
        mesh_dims = None

        if MESH_TYPE == 'square':
            nx = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
            ny = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
            initial_mesh, mesh_dims = create_square_mesh(nx, ny)
            current_res_info = f"nx={nx}, ny={ny}"
        elif MESH_TYPE == 'pipe':
            current_mesh_size_factor = np.random.uniform(MESH_SIZE_FACTOR_MIN, MESH_SIZE_FACTOR_MAX)
            initial_mesh, mesh_dims = create_pipe_with_obstacle_mesh_gmsh(
                    mesh_size_factor=current_mesh_size_factor,
                    pipe_length=PIPE_LENGTH, pipe_height=PIPE_HEIGHT,
                    obstacle_cx_factor=OBSTACLE_CENTER_X_FACTOR,
                    obstacle_cy_factor=OBSTACLE_CENTER_Y_FACTOR,
                    obstacle_r_factor=OBSTACLE_RADIUS_FACTOR
                    )
            current_res_info = f"Factor={current_mesh_size_factor:.3f}"

        if initial_mesh is None or initial_mesh.num_cells() == 0 or initial_mesh.num_vertices() == 0:
            print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - initial mesh invalid. Skipping.")
            if initial_mesh is not None:
                 print(f"    Mesh details: {initial_mesh.num_cells()} cells, {initial_mesh.num_vertices()} vertices.")
            continue

        print(f"  Solving FEM on initial mesh ({current_res_info})...")
        u_initial = solve_fem_problem(initial_mesh, MESH_TYPE, mesh_dims)
        if u_initial is None:
            print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - FEM solve failed on initial mesh. Skipping.")
            continue
        l2_error_initial = calculate_l2_error(u_initial, MESH_TYPE, mesh_dims, initial_mesh)
        print(f"  Initial L2 Error: {l2_error_initial:.6e} (DoFs: {u_initial.function_space().dim()})")
        if l2_error_initial == -2.0:
             print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - L2 error indicates solver failure. Skipping.")
             continue

        monitor_vals = get_solution_based_monitor_function(u_initial, initial_mesh)

        classical_start_time = time.time()
        optimized_coords_classical = dummy_classical_r_adaptivity(initial_mesh, monitor_vals, mesh_dimensions=mesh_dims)
        classical_duration = time.time() - classical_start_time

        l2_error_optimized_classical = -1.0
        optimized_mesh_classical_viz = dolfin.Mesh(initial_mesh)
        if optimized_coords_classical.shape[0] == optimized_mesh_classical_viz.num_vertices():
            optimized_mesh_classical_viz.coordinates()[:] = optimized_coords_classical

            min_cell_vol_val = -10.0
            if optimized_mesh_classical_viz.num_cells() > 0 :
                try:
                    V_dg0 = dolfin.FunctionSpace(optimized_mesh_classical_viz, "DG", 0)
                    # CellVolume(mesh) is a UFL expression for the cell volume
                    cell_volumes_p0 = dolfin.project(dolfin.CellVolume(optimized_mesh_classical_viz), V_dg0)
                    if cell_volumes_p0.vector().size() > 0:
                        min_cell_vol_val = np.min(cell_volumes_p0.vector().get_local())
                    else:
                        min_cell_vol_val = -1.0
                except Exception as vol_exc:
                    print(f"  Could not compute min_cell_volume: {vol_exc}")
                    min_cell_vol_val = -2.0
            else:
                min_cell_vol_val = -3.0

            if min_cell_vol_val < 1e-12 and min_cell_vol_val > -5.0 :
                print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - Classical r-adaptivity likely resulted in a tangled mesh. Min cell volume: {min_cell_vol_val:.2e}. Skipping FEM solve on adapted mesh.")
                l2_error_optimized_classical = -3.0
            else:
                print(f"  Solving FEM on classically adapted mesh (Min cell vol: {min_cell_vol_val:.2e})...")
                u_optimized_classical = solve_fem_problem(optimized_mesh_classical_viz, MESH_TYPE, mesh_dims)
                if u_optimized_classical is None:
                    print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - FEM solve failed on classically adapted mesh.")
                    l2_error_optimized_classical = -2.0
                else:
                    l2_error_optimized_classical = calculate_l2_error(u_optimized_classical, MESH_TYPE, mesh_dims, optimized_mesh_classical_viz)
                    print(f"  Classical Adapted L2 Error: {l2_error_optimized_classical:.6e}")
        else:
            print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - Coord shape mismatch for classical opt mesh. Skipping FEM solve.")
            l2_error_optimized_classical = -4.0

        if l2_error_optimized_classical < -1.5:
            print(f"  Skipping sample {generated_count+1} due to issues with classical adaptation (error code: {l2_error_optimized_classical}).")
            # continue

        pyg_data_sample = fenics_mesh_to_pyg_data(initial_mesh, device=DEVICE)
        if pyg_data_sample.num_nodes == 0:
             print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - empty PyG graph. Skipping.")
             continue

        pyg_data_sample.y = torch.tensor(optimized_coords_classical, dtype=torch.float).to(DEVICE)
        pyg_data_sample.classical_time = classical_duration
        pyg_data_sample.l2_error_initial = l2_error_initial
        pyg_data_sample.l2_error_classical_adapted = l2_error_optimized_classical
        pyg_data_sample.num_dofs = u_initial.function_space().dim() if u_initial else -1
        pyg_data_sample.mesh_type_str = MESH_TYPE
        pyg_data_sample.mesh_dimensions_str = str(mesh_dims)

        dataset.append(pyg_data_sample)
        all_classical_times.append(classical_duration)
        all_initial_errors.append(l2_error_initial)
        all_classical_adapted_errors.append(l2_error_optimized_classical)
        generated_count += 1

        if plot_first_sample_details and generated_count == 1:
            print(f"  Plotting details for first successful sample...")
            fig_mesh, axs_mesh = plt.subplots(1, 2, figsize=(12, 5))
            plt.sca(axs_mesh[0]); dolfin.plot(initial_mesh); axs_mesh[0].set_title(f"Initial Mesh (Sample 1, {current_res_info})\nL2 Err: {l2_error_initial:.2e}")
            if MESH_TYPE == 'pipe': axs_mesh[0].set_aspect('equal')

            plt.sca(axs_mesh[1]); dolfin.plot(optimized_mesh_classical_viz); axs_mesh[1].set_title(f"Classical Adapted (Sample 1)\nL2 Err: {l2_error_optimized_classical:.2e}")
            if MESH_TYPE == 'pipe': axs_mesh[1].set_aspect('equal')

            plt.tight_layout()
            plot_filename = os.path.join(session_output_dir, f"{MODEL_NAME}_Sample1_Meshes.png")
            plt.savefig(plot_filename); plt.close(fig_mesh)
            print(f"    Saved mesh plot for sample 1 to {plot_filename}")

            fig_sol, axs_sol = plt.subplots(1, 2, figsize=(12, 5))
            plt.sca(axs_sol[0]); cax0 = dolfin.plot(u_initial, title="Initial Solution (Sample 1)"); plt.colorbar(cax0, ax=axs_sol[0])
            if MESH_TYPE == 'pipe': axs_sol[0].set_aspect('equal')

            if 'u_optimized_classical' in locals() and u_optimized_classical is not None:
                plt.sca(axs_sol[1]); cax1 = dolfin.plot(u_optimized_classical, title="Classical Adapted Solution (Sample 1)"); plt.colorbar(cax1, ax=axs_sol[1])
                if MESH_TYPE == 'pipe': axs_sol[1].set_aspect('equal')
            else:
                axs_sol[1].text(0.5, 0.5, "No classical solution plotted", ha='center', va='center')

            plt.tight_layout()
            plot_filename_sol = os.path.join(session_output_dir, f"{MODEL_NAME}_Sample1_Solutions.png")
            plt.savefig(plot_filename_sol); plt.close(fig_sol)
            print(f"    Saved solution plot for sample 1 to {plot_filename_sol}")

            plt.figure(figsize=(7,5))
            V_monitor_plot = dolfin.FunctionSpace(initial_mesh, "CG", 1)
            monitor_func_plot = dolfin.Function(V_monitor_plot)

            vertex_to_dof_map_mon = dolfin.vertex_to_dof_map(V_monitor_plot)
            if monitor_vals.shape[0] == V_monitor_plot.dim():
                 monitor_func_plot.vector()[:] = monitor_vals[vertex_to_dof_map_mon]
                 cax_m = dolfin.plot(monitor_func_plot, title="Monitor Function (grad(u_initial) norm, Sample 1)")
                 plt.colorbar(cax_m)
                 if MESH_TYPE == 'pipe': plt.gca().set_aspect('equal')
                 plt.tight_layout()
                 plot_filename_mon = os.path.join(session_output_dir, f"{MODEL_NAME}_Sample1_MonitorFunction.png")
                 plt.savefig(plot_filename_mon); plt.close()
                 print(f"    Saved monitor function plot for sample 1 to {plot_filename_mon}")
            else:
                print(f"    Skipping monitor plot due to shape/map mismatch: monitor_vals shape {monitor_vals.shape[0]} vs V_monitor_plot dofs {V_monitor_plot.dim()}")


        if generated_count > 0 and generated_count % (num_samples // 10 or 1) == 0:
            print(f"  Generated {generated_count}/{num_samples} valid samples (Total attempts: {attempts}).")

    if generated_count < num_samples:
        print(f"Warning: Only generated {generated_count}/{num_samples} valid samples after {max_attempts} attempts.")
    if not dataset:
        print("CRITICAL: Dataset generation resulted in NO valid samples. Check mesh creation, FEM solver, and r-adaptivity parameters and functions.")

    return dataset, all_classical_times, all_initial_errors, all_classical_adapted_errors

# --- Main Script ---
if __name__ == '__main__':
    dolfin.set_log_level(dolfin.LogLevel.WARNING)

    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"session_{session_timestamp}")
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)
    print(f"Using MESH_TYPE: '{MESH_TYPE}'")
    print(f"Output for this session will be in: {SESSION_OUTPUT_DIR}")

    plot_funcs.cuda_status(DEVICE)

    dataset, classical_r_adapt_times_all, l2_errors_initial_all, l2_errors_classical_adapted_all = [], [], [], []
    try:
        dataset, classical_r_adapt_times_all, l2_errors_initial_all, l2_errors_classical_adapted_all = generate_dataset(
            NUM_SAMPLES, SESSION_OUTPUT_DIR, plot_first_sample_details=True
        )
    except Exception as e:
        print(f"CRITICAL Error during data generation: {e}")
        import traceback
        traceback.print_exc()
        exit()

    if not dataset:
        print("Dataset is empty after generation. Exiting.")
        exit()

    train_size = int(0.8 * len(dataset))
    if len(dataset) > 0 and train_size == 0 : train_size = 1

    if train_size >= len(dataset):
        print("Warning: Dataset too small for a separate validation set. Using all data for training and validation benchmarks.")
        train_dataset = dataset
        val_dataset = list(dataset)
    else:
        train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]

    print(f"Dataset size: Total={len(dataset)}, Train={len(train_dataset)}, Val={len(val_dataset)}")

    if not train_dataset:
        print("Error: Training dataset is empty. Cannot proceed.")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None

    if dataset and hasattr(dataset[0], 'x') and dataset[0].x is not None:
        in_feat_dim = dataset[0].x.size(1)
    else:
        print("Error: Dataset is empty or first sample has no features 'x'. Cannot determine input feature dimension.")
        exit()

    gat_model = RAdaptGAT(in_channels=in_feat_dim, hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS, heads=HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(gat_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    print(f"\n--- Training {MODEL_NAME} on {DEVICE} ---")
    if classical_r_adapt_times_all: print(f"  Avg classical r-adapt time (dataset gen): {np.mean(classical_r_adapt_times_all):.4f}s")

    valid_initial_errors = [e for e in l2_errors_initial_all if e is not None and e >= 0]
    if valid_initial_errors: print(f"  Avg initial L2 error (dataset gen, valid only): {np.mean(valid_initial_errors):.4e}")

    valid_classical_adapted_errors = [e for e in l2_errors_classical_adapted_all if e is not None and e >= 0]
    if valid_classical_adapted_errors: print(f"  Avg classical adapted L2 error (dataset gen, valid only): {np.mean(valid_classical_adapted_errors):.4e}")


    epochs_list, train_losses_history, val_losses_history, gat_epoch_times_train = [], [], [], []
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        gat_model.train()
        current_epoch_train_loss, num_train_batches = 0, 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = gat_model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            current_epoch_train_loss += loss.item()
            num_train_batches += 1

        avg_epoch_train_loss = current_epoch_train_loss / num_train_batches if num_train_batches > 0 else float('nan')
        train_losses_history.append(avg_epoch_train_loss)
        gat_epoch_times_train.append(time.time() - epoch_start_time)

        current_epoch_val_loss, num_val_batches = 0, 0
        if val_loader:
            gat_model.eval()
            with torch.inference_mode():
                for batch in val_loader:
                    out = gat_model(batch.x, batch.edge_index)
                    loss = loss_fn(out, batch.y)
                    current_epoch_val_loss += loss.item()
                    num_val_batches += 1
            avg_epoch_val_loss = current_epoch_val_loss / num_val_batches if num_val_batches > 0 else float('nan')
            val_losses_history.append(avg_epoch_val_loss)
        else:
            val_losses_history.append(float('nan'))

        epochs_list.append(epoch + 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses_history[-1]:.6f} | Val Loss: {val_losses_history[-1]:.6f} | Time: {gat_epoch_times_train[-1]:.2f}s")

    print("--- Training complete ---")
    if gat_epoch_times_train: print(f"Avg GAT training epoch time: {np.mean(gat_epoch_times_train):.4f}s")

    model_save_path = os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_epoch{EPOCHS}.pt")
    torch.save(gat_model.state_dict(), model_save_path)
    print(f"Saved trained model state_dict to: {model_save_path}")

    plot_funcs.loss_plot(epochs_list, train_losses_history, val_losses_history,
                         MODEL_NAME, SESSION_OUTPUT_DIR, show=False)
    print(f"Saved training loss plot to {SESSION_OUTPUT_DIR}/")

    val_classical_l2_errors, val_gat_l2_errors, val_initial_l2_errors = [], [], []
    val_dofs_list = []
    val_classical_times_list_bench = []
    val_gat_inference_times_list_bench = []

    if val_dataset:
        print(f"\n--- Evaluation on Validation Set ({len(val_dataset)} samples) ---")

        if val_loader:
            try:
                val_batch_sample_iter = iter(val_loader)
                val_batch_sample = next(val_batch_sample_iter)
                val_batch_sample = val_batch_sample.to(DEVICE)
                gat_model.eval()
                with torch.inference_mode():
                    val_pred_coords_np = gat_model(val_batch_sample.x, val_batch_sample.edge_index).cpu().numpy()
                val_true_coords_np = val_batch_sample.y.cpu().numpy()
                pred_vs_true_filename_base = f"{MODEL_NAME}_ValBatchCoords" # Removed _predVStrue
                plot_funcs.predVStrue([val_true_coords_np.flatten()], [val_pred_coords_np.flatten()],
                                      [], [], pred_vs_true_filename_base, SESSION_OUTPUT_DIR, show=False)
                # print(f"Saved Pred vs True plot (node coordinates) for a val batch to {SESSION_OUTPUT_DIR}/") # Printed by plot_funcs
            except StopIteration:
                print("Warning: Validation loader is empty, cannot plot PredVsTrue for validation batch.")
            except Exception as e_predtrue:
                 print(f"Error during PredVsTrue plot for validation batch: {e_predtrue}")


        num_inference_runs_per_sample = 5

        for i, data_sample in enumerate(val_dataset):
            print(f"  Processing validation sample {i+1}/{len(val_dataset)}...")

            try:
                current_mesh_dims = eval(data_sample.mesh_dimensions_str)
            except Exception as e_eval:
                print(f"    Error evaluating mesh_dimensions_str '{data_sample.mesh_dimensions_str}': {e_eval}. Skipping sample.")
                val_initial_l2_errors.append(-10.0)
                val_classical_l2_errors.append(-10.0)
                val_gat_l2_errors.append(-10.0)
                val_dofs_list.append(-1)
                continue
            current_mesh_type = data_sample.mesh_type_str

            initial_mesh_val = None
            if current_mesh_type == 'square':
                nx_val = current_mesh_dims.get('nx')
                ny_val = current_mesh_dims.get('ny')
                if nx_val is None or ny_val is None:
                    print(f"    nx/ny not found in mesh_dims for square validation sample {i+1}. Skipping.")
                    val_initial_l2_errors.append(-10.0)
                    val_classical_l2_errors.append(-10.0)
                    val_gat_l2_errors.append(-10.0)
                    val_dofs_list.append(-1)
                    continue
                initial_mesh_val, _ = create_square_mesh(nx_val, ny_val)
            elif current_mesh_type == 'pipe':
                ms_factor_val = current_mesh_dims.get('mesh_size_factor')
                if ms_factor_val is None:
                     ms_factor_val = (MESH_SIZE_FACTOR_MIN + MESH_SIZE_FACTOR_MAX) / 2.0
                initial_mesh_val, _ = create_pipe_with_obstacle_mesh_gmsh(
                    mesh_size_factor=ms_factor_val, pipe_length=PIPE_LENGTH, pipe_height=PIPE_HEIGHT,
                    obstacle_cx_factor=OBSTACLE_CENTER_X_FACTOR, obstacle_cy_factor=OBSTACLE_CENTER_Y_FACTOR,
                    obstacle_r_factor=OBSTACLE_RADIUS_FACTOR
                )

            if initial_mesh_val is None or initial_mesh_val.num_cells() == 0:
                print(f"    Could not regenerate validation mesh for sample {i+1}. Skipping.")
                val_initial_l2_errors.append(-11.0)
                val_classical_l2_errors.append(-11.0)
                val_gat_l2_errors.append(-11.0)
                val_dofs_list.append(-1)
                continue

            u_initial_val = solve_fem_problem(initial_mesh_val, current_mesh_type, current_mesh_dims)
            if u_initial_val is None:
                print(f"    FEM solve failed on regenerated initial val mesh {i+1}. Skipping.");
                val_initial_l2_errors.append(-2.0)
                val_classical_l2_errors.append(-2.0)
                val_gat_l2_errors.append(-2.0)
                val_dofs_list.append(-1 if not val_dofs_list else val_dofs_list[-1]) # try to keep length same
                continue
            err_init_val = calculate_l2_error(u_initial_val, current_mesh_type, current_mesh_dims, initial_mesh_val)
            val_initial_l2_errors.append(err_init_val)
            val_dofs_list.append(u_initial_val.function_space().dim())

            monitor_val = get_solution_based_monitor_function(u_initial_val, initial_mesh_val)

            classical_start_val = time.time()
            opt_coords_classical_val = dummy_classical_r_adaptivity(initial_mesh_val, monitor_val, mesh_dimensions=current_mesh_dims)
            val_classical_times_list_bench.append(time.time() - classical_start_val)

            classical_adapted_mesh_val = dolfin.Mesh(initial_mesh_val)
            if opt_coords_classical_val.shape[0] == classical_adapted_mesh_val.num_vertices():
                classical_adapted_mesh_val.coordinates()[:] = opt_coords_classical_val
                u_class_val = solve_fem_problem(classical_adapted_mesh_val, current_mesh_type, current_mesh_dims)
                if u_class_val: val_classical_l2_errors.append(calculate_l2_error(u_class_val, current_mesh_type, current_mesh_dims, classical_adapted_mesh_val))
                else: val_classical_l2_errors.append(-2.0)
            else: val_classical_l2_errors.append(-4.0)

            pyg_val_sample = fenics_mesh_to_pyg_data(initial_mesh_val, device=DEVICE)

            current_gat_inference_times = []
            gat_adapted_coords_val_np = None
            if pyg_val_sample.num_nodes > 0 and pyg_val_sample.x is not None :
                gat_model.eval()
                with torch.inference_mode():
                    gat_adapted_coords_val = gat_model(pyg_val_sample.x, pyg_val_sample.edge_index)
                    gat_adapted_coords_val_np = gat_adapted_coords_val.cpu().numpy()
                    for _ in range(num_inference_runs_per_sample):
                        start_time_gat = time.time()
                        _ = gat_model(pyg_val_sample.x, pyg_val_sample.edge_index)
                        current_gat_inference_times.append(time.time() - start_time_gat)
                if current_gat_inference_times: val_gat_inference_times_list_bench.append(np.mean(current_gat_inference_times))
            else:
                val_gat_inference_times_list_bench.append(0.0)
                print(f"    Skipping GAT inference for val sample {i+1} due to empty graph or no features.")


            if gat_adapted_coords_val_np is not None:
                gat_adapted_mesh_val = dolfin.Mesh(initial_mesh_val)
                if gat_adapted_coords_val_np.shape[0] == gat_adapted_mesh_val.num_vertices():
                    gat_adapted_mesh_val.coordinates()[:] = gat_adapted_coords_val_np
                    u_gat_val = solve_fem_problem(gat_adapted_mesh_val, current_mesh_type, current_mesh_dims)
                    if u_gat_val: val_gat_l2_errors.append(calculate_l2_error(u_gat_val, current_mesh_type, current_mesh_dims, gat_adapted_mesh_val))
                    else: val_gat_l2_errors.append(-2.0)
                else:
                    val_gat_l2_errors.append(-4.0)
            else:
                val_gat_l2_errors.append(-5.0)

        valid_val_initial_errors = [e for e in val_initial_l2_errors if e is not None and e >= 0]
        if valid_val_initial_errors: print(f"  Avg L2 Error (Initial - Val Set, valid only): {np.mean(valid_val_initial_errors):.4e}")

        valid_val_classical_errors = [e for e in val_classical_l2_errors if e is not None and e >= 0]
        if valid_val_classical_errors: print(f"  Avg L2 Error (Classical Adapted - Val Set, valid only): {np.mean(valid_val_classical_errors):.4e}")

        valid_val_gat_errors = [e for e in val_gat_l2_errors if e is not None and e >= 0]
        if valid_val_gat_errors: print(f"  Avg L2 Error (GAT Adapted - Val Set, valid only): {np.mean(valid_val_gat_errors):.4e}")


        if val_classical_times_list_bench: print(f"  Avg Classical R-Adapt Time (Val Set Bench): {np.mean(val_classical_times_list_bench):.6f}s")
        if val_gat_inference_times_list_bench: print(f"  Avg GAT Inference Time (Val Set Bench): {np.mean(val_gat_inference_times_list_bench):.6f}s")

        benchmark_summary = {
            "model_name": MODEL_NAME, "mesh_type": MESH_TYPE, "session_timestamp": session_timestamp,
            "parameters": {
                "num_total_samples": NUM_SAMPLES, "num_validation_samples_benchmarked": len(val_dataset),
                "epochs_trained": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                "gat_params": {"in_feat": in_feat_dim, "hidden": HIDDEN_CHANNELS, "out_feat": OUT_CHANNELS, "heads": HEADS, "layers": NUM_LAYERS, "dropout": DROPOUT},
            },
            "avg_l2_error_initial_validation": float(np.mean(valid_val_initial_errors)) if valid_val_initial_errors else -1.0,
            "avg_l2_error_classical_adapted_validation": float(np.mean(valid_val_classical_errors)) if valid_val_classical_errors else -1.0,
            "avg_l2_error_gat_adapted_validation": float(np.mean(valid_val_gat_errors)) if valid_val_gat_errors else -1.0,
            "classical_r_adaptivity_times_seconds_validation": {"all_values": val_classical_times_list_bench, "mean": float(np.mean(val_classical_times_list_bench)) if val_classical_times_list_bench else -1.0},
            "gat_inference_times_seconds_validation": {"all_values": val_gat_inference_times_list_bench, "mean": float(np.mean(val_gat_inference_times_list_bench)) if val_gat_inference_times_list_bench else -1.0},
            "dofs_validation": val_dofs_list,
            "l2_errors_initial_validation_all": val_initial_l2_errors,
            "l2_errors_classical_adapted_validation_all": val_classical_l2_errors,
            "l2_errors_gat_adapted_validation_all": val_gat_l2_errors,
        }
        benchmark_file_path = os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_benchmark_summary.json")
        try:
            with open(benchmark_file_path, 'w') as f: json.dump(benchmark_summary, f, indent=4)
            print(f"Saved benchmark summary to: {benchmark_file_path}")
        except Exception as e: print(f"Error saving benchmark summary: {e}")

        if val_classical_times_list_bench and val_gat_inference_times_list_bench:
            plot_funcs.plot_time_comparison(val_classical_times_list_bench, val_gat_inference_times_list_bench,
                                            title=f"{MODEL_NAME} R-Adapt Time Distribution (Val Set)",
                                            output=SESSION_OUTPUT_DIR, use_box_plot=True, show=False)

        classical_plot_data = [(d, e) for d, e in zip(val_dofs_list, val_classical_l2_errors) if e is not None and e >= 0 and d > 0]
        gat_plot_data = [(d, e) for d, e in zip(val_dofs_list, val_gat_l2_errors) if e is not None and e >= 0 and d > 0]

        if classical_plot_data and gat_plot_data:
            classical_dofs_plot, classical_errors_plot = map(list, zip(*classical_plot_data)) if classical_plot_data else ([],[])
            gat_dofs_plot, gat_errors_plot = map(list, zip(*gat_plot_data)) if gat_plot_data else ([],[])

            if classical_dofs_plot and gat_dofs_plot :
                 plot_funcs.plot_convergence(classical_dofs_plot, classical_errors_plot,
                                             gat_dofs_plot, gat_errors_plot,
                                             title=f"{MODEL_NAME} Error vs. DoFs (Val Set)",
                                             output=SESSION_OUTPUT_DIR, show=False)

            classical_time_cost_plot = []
            classical_err_for_time_plot = []
            for i in range(min(len(val_classical_l2_errors), len(val_classical_times_list_bench))):
                err_c = val_classical_l2_errors[i]
                if err_c is not None and err_c >= 0:
                    classical_time_cost_plot.append(val_classical_times_list_bench[i])
                    classical_err_for_time_plot.append(err_c)

            gat_time_cost_plot = []
            gat_err_for_time_plot = []
            for i in range(min(len(val_gat_l2_errors), len(val_gat_inference_times_list_bench))):
                err_g = val_gat_l2_errors[i]
                if err_g is not None and err_g >= 0:
                    gat_time_cost_plot.append(val_gat_inference_times_list_bench[i])
                    gat_err_for_time_plot.append(err_g)

            if classical_time_cost_plot and gat_time_cost_plot and classical_err_for_time_plot and gat_err_for_time_plot:
                plot_funcs.plot_accuracy_vs_cost(
                    classical_time_cost_plot, classical_err_for_time_plot,
                    gat_time_cost_plot, gat_err_for_time_plot,
                    cost_label='Adaptation Time (s)', accuracy_label='L2 Error (Lower is Better)',
                    title=f'{MODEL_NAME} L2 Error vs. Adaptation Time (Val Set)',
                    output=SESSION_OUTPUT_DIR, show=False
                )
    else:
        print("Validation dataset is empty. Skipping validation benchmarks and plots.")

    # --- Example Inference on a New Mesh ---
    print(f"\n--- Example Inference & Plot ({MESH_TYPE} Geometry) ---")
    example_initial_mesh, example_mesh_dims = None, None
    example_res_info = ""
    nx_ex, ny_ex, factor_ex = 0,0,0

    if MESH_TYPE == 'square':
        nx_ex = (MESH_SIZE_MIN + MESH_SIZE_MAX) // 2 + 1
        ny_ex = (MESH_SIZE_MIN + MESH_SIZE_MAX) // 2 + 1
        example_initial_mesh, example_mesh_dims = create_square_mesh(nx_ex, ny_ex)
        example_res_info = f"nx={nx_ex}, ny={ny_ex}"
    elif MESH_TYPE == 'pipe':
        factor_ex = (MESH_SIZE_FACTOR_MIN + MESH_SIZE_FACTOR_MAX) / 2.0 * 0.9
        example_initial_mesh, example_mesh_dims = create_pipe_with_obstacle_mesh_gmsh(
            mesh_size_factor=factor_ex, pipe_length=PIPE_LENGTH, pipe_height=PIPE_HEIGHT,
            obstacle_cx_factor=OBSTACLE_CENTER_X_FACTOR, obstacle_cy_factor=OBSTACLE_CENTER_Y_FACTOR,
            obstacle_r_factor=OBSTACLE_RADIUS_FACTOR
        )
        example_res_info = f"Factor={factor_ex:.3f}"

    l2_err_initial_ex, l2_err_classical_ex, l2_err_gat_ex = -1.0,-1.0,-1.0
    u_initial_ex, u_classical_ex, u_gat_ex = None, None, None
    example_classical_adapted_mesh, example_gat_adapted_mesh = None, None
    predicted_optimized_coords_gat_np = None


    if example_initial_mesh is None or example_initial_mesh.num_vertices() == 0:
        print("Error: Failed to create example initial mesh for inference. Exiting example.")
    else:
        print("  Solving FEM on example initial mesh...")
        u_initial_ex = solve_fem_problem(example_initial_mesh, MESH_TYPE, example_mesh_dims)
        if u_initial_ex:
            l2_err_initial_ex = calculate_l2_error(u_initial_ex, MESH_TYPE, example_mesh_dims, example_initial_mesh)
            print(f"  Example Initial L2 Error: {l2_err_initial_ex:.4e} (DoFs: {u_initial_ex.function_space().dim()})")

            monitor_vals_ex = get_solution_based_monitor_function(u_initial_ex, example_initial_mesh)

            classical_start_time_ex = time.time()
            example_optimized_classical_coords = dummy_classical_r_adaptivity(example_initial_mesh, monitor_vals_ex, mesh_dimensions=example_mesh_dims)
            classical_duration_ex = time.time() - classical_start_time_ex
            print(f"  Classical r-adapt time for example mesh: {classical_duration_ex:.6f}s")

            example_classical_adapted_mesh = dolfin.Mesh(example_initial_mesh)
            if example_optimized_classical_coords.shape[0] == example_classical_adapted_mesh.num_vertices():
                example_classical_adapted_mesh.coordinates()[:] = example_optimized_classical_coords
                u_classical_ex = solve_fem_problem(example_classical_adapted_mesh, MESH_TYPE, example_mesh_dims)
                if u_classical_ex: l2_err_classical_ex = calculate_l2_error(u_classical_ex, MESH_TYPE, example_mesh_dims, example_classical_adapted_mesh)
                print(f"  Example Classical Adapted L2 Error: {l2_err_classical_ex:.4e}")

            pyg_example_data = fenics_mesh_to_pyg_data(example_initial_mesh, device=DEVICE)

            if pyg_example_data.num_nodes > 0 and pyg_example_data.x is not None:
                gat_model.eval()
                gat_inference_times_ex = []
                with torch.inference_mode():
                    predicted_optimized_coords_gat = gat_model(pyg_example_data.x, pyg_example_data.edge_index)
                    predicted_optimized_coords_gat_np = predicted_optimized_coords_gat.cpu().numpy()
                    for _ in range(num_inference_runs_per_sample):
                        gat_start_time = time.time()
                        _ = gat_model(pyg_example_data.x, pyg_example_data.edge_index)
                        gat_inference_times_ex.append(time.time() - gat_start_time)
                avg_gat_duration_ex = np.mean(gat_inference_times_ex) if gat_inference_times_ex else 0
                print(f"  GAT inference time for example mesh (avg over {len(gat_inference_times_ex)} runs): {avg_gat_duration_ex:.6f}s")

                if predicted_optimized_coords_gat_np is not None:
                    example_gat_adapted_mesh = dolfin.Mesh(example_initial_mesh)
                    if predicted_optimized_coords_gat_np.shape[0] == example_gat_adapted_mesh.num_vertices():
                        example_gat_adapted_mesh.coordinates()[:] = predicted_optimized_coords_gat_np
                        u_gat_ex = solve_fem_problem(example_gat_adapted_mesh, MESH_TYPE, example_mesh_dims)
                        if u_gat_ex: l2_err_gat_ex = calculate_l2_error(u_gat_ex, MESH_TYPE, example_mesh_dims, example_gat_adapted_mesh)
                        print(f"  Example GAT Adapted L2 Error: {l2_err_gat_ex:.4e}")
                    else: print("  GAT coord shape mismatch for example mesh.")
                else: print("  GAT inference did not produce coordinates.")

                if classical_duration_ex > 0 and avg_gat_duration_ex > 0 :
                    plot_funcs.plot_time_comparison([classical_duration_ex], [avg_gat_duration_ex],
                                                    title=f"{MODEL_NAME} Inference Time ({MESH_TYPE}, {example_res_info})",
                                                    output=SESSION_OUTPUT_DIR, show=False)
            else:
                print("  Failed to convert example mesh to PyG data for GAT inference (no nodes or features).")

            if example_initial_mesh and example_classical_adapted_mesh and example_gat_adapted_mesh:
                fig_final_mesh, axs_final_mesh = plt.subplots(1, 3, figsize=(18, 6))
                title_suffix = f"{MESH_TYPE.capitalize()} ({example_res_info})"

                plt.sca(axs_final_mesh[0]); dolfin.plot(example_initial_mesh); axs_final_mesh[0].set_title(f"Initial\nL2Err: {l2_err_initial_ex:.2e}")
                if MESH_TYPE == 'pipe': axs_final_mesh[0].set_aspect('equal')

                plt.sca(axs_final_mesh[1]); dolfin.plot(example_classical_adapted_mesh); axs_final_mesh[1].set_title(f"Classical R-Adapted\nL2Err: {l2_err_classical_ex:.2e}")
                if MESH_TYPE == 'pipe': axs_final_mesh[1].set_aspect('equal')

                plt.sca(axs_final_mesh[2]); dolfin.plot(example_gat_adapted_mesh); axs_final_mesh[2].set_title(f"GAT R-Adapted\nL2Err: {l2_err_gat_ex:.2e}")
                if MESH_TYPE == 'pipe': axs_final_mesh[2].set_aspect('equal')

                fig_final_mesh.suptitle(f"{title_suffix} Mesh Adaptation Comparison", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                plt.savefig(os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_final_mesh_comparison.png"))
                plt.close(fig_final_mesh)
                # print(f"  Saved final mesh comparison plot to {SESSION_OUTPUT_DIR}/") # Printed by plot_funcs

                fig_final_sol, axs_final_sol = plt.subplots(1, 3, figsize=(18, 6))
                common_elements = {"cmap": "viridis"}

                if u_initial_ex:
                    plt.sca(axs_final_sol[0]); cax0 = dolfin.plot(u_initial_ex, **common_elements); axs_final_sol[0].set_title(f"Initial Solution\nL2Err: {l2_err_initial_ex:.2e}"); plt.colorbar(cax0, ax=axs_final_sol[0], fraction=0.046, pad=0.04)
                    if MESH_TYPE == 'pipe': axs_final_sol[0].set_aspect('equal')

                if u_classical_ex:
                    plt.sca(axs_final_sol[1]); cax1 = dolfin.plot(u_classical_ex, **common_elements); axs_final_sol[1].set_title(f"Classical Adapted Solution\nL2Err: {l2_err_classical_ex:.2e}"); plt.colorbar(cax1, ax=axs_final_sol[1], fraction=0.046, pad=0.04)
                    if MESH_TYPE == 'pipe': axs_final_sol[1].set_aspect('equal')

                if u_gat_ex:
                    plt.sca(axs_final_sol[2]); cax2 = dolfin.plot(u_gat_ex, **common_elements); axs_final_sol[2].set_title(f"GAT Adapted Solution\nL2Err: {l2_err_gat_ex:.2e}"); plt.colorbar(cax2, ax=axs_final_sol[2], fraction=0.046, pad=0.04)
                    if MESH_TYPE == 'pipe': axs_final_sol[2].set_aspect('equal')

                fig_final_sol.suptitle(f"{title_suffix} Solution Comparison", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                plt.savefig(os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_final_solution_comparison.png"))
                plt.close(fig_final_sol)
                # print(f"  Saved final solution comparison plot to {SESSION_OUTPUT_DIR}/") # Printed by plot_funcs
            else:
                print("  Skipping final example plot: some meshes or solutions were not generated/valid.")
        else:
            print("  FEM solve failed for example initial mesh. Skipping detailed example plots.")

    print(f"\nAll generated plots, model, and summaries saved to directory: {SESSION_OUTPUT_DIR}")
