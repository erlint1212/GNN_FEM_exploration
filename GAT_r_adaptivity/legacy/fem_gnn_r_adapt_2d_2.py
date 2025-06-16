# fem_gnn_r_adapt_2d.py
import dolfin
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import datetime
import json

try:
    from mesh_generators_2 import create_square_mesh, create_pipe_with_obstacle_mesh_gmsh
except ImportError as e:
    print("CRITICAL ERROR during import of mesh_generators_2.py. Original error was:")
    print(e); exit()

from models.GAT import RAdaptGAT
from fenics_mesh_to_pyg_data import fenics_mesh_to_pyg_data

try:
    import plot_funcs
except ImportError:
    print("Error: plot_funcs.py not found.")
    class plot_funcs_dummy: # Ensure this dummy class is defined if import fails
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 8
NUM_SAMPLES = 100

MESH_TYPE = 'square'
F_EXPRESSION_STR_SQUARE = "2*pow(user_pi,2)*sin(user_pi*x[0])*sin(user_pi*x[1])"
U_EXACT_EXPRESSION_STR_SQUARE = "sin(user_pi*x[0])*sin(user_pi*x[1])"
U_DIRICHLET_EXPRESSION_STR_SQUARE = "0.0"
F_EXPRESSION_STR_PIPE = "10 * exp(-(pow(x[0] - 0.5*L, 2) + pow(x[1] - 0.5*H, 2)) / (2*pow(0.1*H,2)))"
U_EXACT_EXPRESSION_STR_PIPE = None
U_DIRICHLET_EXPRESSION_STR_PIPE = "0.0"

if MESH_TYPE == 'square':
    MODEL_NAME_SUFFIX = "SquareMesh_Poisson_FeatEngV2" # Updated suffix for new changes
    MESH_SIZE_MIN = 8
    MESH_SIZE_MAX = 20
elif MESH_TYPE == 'pipe':
    MODEL_NAME_SUFFIX = "PipeObstacleMesh_Poisson_FeatEngV2"
    MESH_SIZE_FACTOR_MIN = 0.08; MESH_SIZE_FACTOR_MAX = 0.20
    PIPE_LENGTH = 3.0; PIPE_HEIGHT = 1.0
    OBSTACLE_CENTER_X_FACTOR = 0.3; OBSTACLE_CENTER_Y_FACTOR = 0.5; OBSTACLE_RADIUS_FACTOR = 0.15
else: raise ValueError(f"Unknown MESH_TYPE: {MESH_TYPE}")

MODEL_NAME = f"RAdaptGAT_{MODEL_NAME_SUFFIX}"
BASE_OUTPUT_DIR = f"gat_{MODEL_NAME_SUFFIX.lower()}_outputs"

HIDDEN_CHANNELS = 128
OUT_CHANNELS = 2
HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.5
USE_MONITOR_AS_FEATURE = True

# --- Helper function to get boundary nodes ---
def get_boundary_nodes(mesh):
    boundary_nodes_set = set()
    # Create a MeshFunction to mark exterior facets
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    dolfin.DomainBoundary().mark(boundary_markers, 1) # Mark all exterior facets
    for f in dolfin.facets(mesh):
        if boundary_markers[f.index()] == 1: # If it's an exterior facet
            for v_idx in f.entities(0): # Get vertex indices of the facet
                boundary_nodes_set.add(v_idx)
    return list(boundary_nodes_set)


def solve_fem_problem(mesh, mesh_type, mesh_dimensions=None):
    V = dolfin.FunctionSpace(mesh, 'P', 1)
    if mesh_type == 'square':
        f_expr_str = F_EXPRESSION_STR_SQUARE; u_d_expr_str = U_DIRICHLET_EXPRESSION_STR_SQUARE
        f = dolfin.Expression(f_expr_str, degree=2, user_pi=dolfin.pi)
        u_D = dolfin.Expression(u_d_expr_str, degree=2, user_pi=dolfin.pi)
    elif mesh_type == 'pipe':
        L_pipe = mesh_dimensions.get("width", PIPE_LENGTH) if mesh_dimensions else PIPE_LENGTH
        H_pipe = mesh_dimensions.get("height", PIPE_HEIGHT) if mesh_dimensions else PIPE_HEIGHT
        f = dolfin.Expression(F_EXPRESSION_STR_PIPE, degree=2, L=L_pipe, H=H_pipe)
        u_D = dolfin.Expression(U_DIRICHLET_EXPRESSION_STR_PIPE, degree=2, L=L_pipe, H=H_pipe)
    else: raise ValueError(f"Unknown mesh type for FEM solve: {mesh_type}")
    def boundary(x, on_boundary): return on_boundary
    bc = dolfin.DirichletBC(V, u_D, boundary)
    u = dolfin.TrialFunction(V); v = dolfin.TestFunction(V)
    a = dolfin.dot(dolfin.grad(u), dolfin.grad(v)) * dolfin.dx
    L_form = f * v * dolfin.dx
    u_sol = dolfin.Function(V)
    try: dolfin.solve(a == L_form, u_sol, bc)
    except Exception as e: print(f"FEniCS solver failed: {e}."); return None
    return u_sol

def calculate_l2_error(u_numerical, mesh_type, mesh_dimensions=None, mesh=None):
    if u_numerical is None: return -2.0
    current_mesh = mesh or u_numerical.function_space().mesh()
    EXACT_SOL_DEGREE = 5
    if mesh_type == 'square':
        u_exact_str = U_EXACT_EXPRESSION_STR_SQUARE
        u_exact = dolfin.Expression(u_exact_str, degree=EXACT_SOL_DEGREE, user_pi=dolfin.pi)
    elif mesh_type == 'pipe':
        if U_EXACT_EXPRESSION_STR_PIPE is None: return -1.0
        L_pipe = mesh_dimensions.get("width", PIPE_LENGTH) if mesh_dimensions else PIPE_LENGTH
        H_pipe = mesh_dimensions.get("height", PIPE_HEIGHT) if mesh_dimensions else PIPE_HEIGHT
        u_exact = dolfin.Expression(U_EXACT_EXPRESSION_STR_PIPE, degree=EXACT_SOL_DEGREE, L=L_pipe, H=H_pipe, user_pi=dolfin.pi)
    else: raise ValueError(f"Unknown mesh type for L2 error: {mesh_type}")
    L2_error = dolfin.errornorm(u_exact, u_numerical, 'L2', mesh=current_mesh)
    return L2_error

def get_solution_based_monitor_function(u_solution, mesh):
    if u_solution is None: return np.ones(mesh.num_vertices()) * 0.5
    V_scalar = dolfin.FunctionSpace(mesh, "CG", 1)
    grad_u_sq = dolfin.project(dolfin.inner(dolfin.grad(u_solution), dolfin.grad(u_solution)), V_scalar)
    monitor_values_nodal = grad_u_sq.compute_vertex_values(mesh)
    min_val = np.min(monitor_values_nodal); max_val = np.max(monitor_values_nodal)
    if max_val - min_val < 1e-9: return np.ones_like(monitor_values_nodal) * 0.5
    return (monitor_values_nodal - min_val) / (max_val - min_val)

def dummy_classical_r_adaptivity(mesh, monitor_values, strength=0.05, mesh_dimensions=None):
    original_coords = mesh.coordinates() # Keep original coordinates for boundary nodes
    old_coords = np.copy(original_coords)
    new_coords = np.copy(old_coords)
    num_nodes = mesh.num_vertices(); geo_dim = mesh.geometry().dim()
    
    boundary_node_indices = get_boundary_nodes(mesh) # Use helper

    for i in range(num_nodes):
        if i in boundary_node_indices:
            new_coords[i] = original_coords[i] # Keep boundary nodes fixed to original positions
            continue
        direction_vector = np.zeros(geo_dim); total_weight = 0.0
        for j in range(num_nodes):
            if i == j: continue
            diff = old_coords[j] - old_coords[i]; dist_sq = np.sum(diff**2)
            if dist_sq < 1e-12: continue
            weight = (monitor_values[j] + monitor_values[i]) / 2.0 / (dist_sq + 1e-6)
            direction_vector += weight * diff; total_weight += weight
        if total_weight > 1e-6: new_coords[i] += strength * (direction_vector / total_weight)
    
    # Clipping for interior nodes (boundary nodes are already fixed)
    if mesh_dimensions:
        min_x, max_x = 0.0, mesh_dimensions.get("width", 1.0)
        min_y, max_y = 0.0, mesh_dimensions.get("height", 1.0)
        for i in range(num_nodes):
            if i not in boundary_node_indices: # Only clip interior nodes
                new_coords[i, 0] = np.clip(new_coords[i, 0], min_x, max_x)
                new_coords[i, 1] = np.clip(new_coords[i, 1], min_y, max_y)
    return new_coords

# --- Helper to check mesh quality ---
def check_mesh_quality(mesh, operation_name=""):
    min_cell_vol_val = -10.0
    if mesh.num_cells() > 0:
        try:
            V_dg0 = dolfin.FunctionSpace(mesh, "DG", 0)
            cell_volumes_p0 = dolfin.project(dolfin.CellVolume(mesh), V_dg0)
            if cell_volumes_p0.vector().size() > 0:
                min_cell_vol_val = np.min(cell_volumes_p0.vector().get_local())
            else: min_cell_vol_val = -1.0 # No cell values
        except Exception as vol_exc:
            print(f"  Could not compute min_cell_volume for {operation_name}: {vol_exc}")
            min_cell_vol_val = -2.0 # Error during volume check
    else: min_cell_vol_val = -3.0 # No cells
    
    if min_cell_vol_val < 1e-12 and min_cell_vol_val > -5.0: # Check if computed and positive
        print(f"  Warning: Mesh from {operation_name} likely tangled. Min cell volume: {min_cell_vol_val:.2e}.")
        return False, min_cell_vol_val # Bad quality
    print(f"  Mesh quality OK for {operation_name} (Min cell vol: {min_cell_vol_val:.2e}).")
    return True, min_cell_vol_val # Good quality

# --- Data Generation ---
def generate_dataset(num_samples, session_output_dir, plot_first_sample_details=False):
    # ... (initial part of generate_dataset remains the same) ...
    dataset, all_classical_times, all_initial_errors, all_classical_adapted_errors = [], [], [], []
    print(f"Generating {num_samples} data samples for MESH_TYPE: '{MESH_TYPE}'...")
    generated_count = 0; attempts = 0; max_attempts = num_samples * 3
    while generated_count < num_samples and attempts < max_attempts:
        attempts += 1; print(f"\nAttempt {attempts} for sample {generated_count + 1}...")
        initial_mesh = None; mesh_dims = None
        if MESH_TYPE == 'square':
            nx = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
            ny = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
            initial_mesh, mesh_dims = create_square_mesh(nx, ny)
            current_res_info = f"nx={nx}, ny={ny}"
        elif MESH_TYPE == 'pipe':
            current_mesh_size_factor = np.random.uniform(MESH_SIZE_FACTOR_MIN, MESH_SIZE_FACTOR_MAX)
            initial_mesh, mesh_dims = create_pipe_with_obstacle_mesh_gmsh(
                    mesh_size_factor=current_mesh_size_factor, pipe_length=PIPE_LENGTH, pipe_height=PIPE_HEIGHT,
                    obstacle_cx_factor=OBSTACLE_CENTER_X_FACTOR, obstacle_cy_factor=OBSTACLE_CENTER_Y_FACTOR,
                    obstacle_r_factor=OBSTACLE_RADIUS_FACTOR)
            current_res_info = f"Factor={current_mesh_size_factor:.3f}"

        if initial_mesh is None or initial_mesh.num_cells() == 0 or initial_mesh.num_vertices() == 0:
            print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - initial mesh invalid. Skipping."); continue
        
        u_initial = solve_fem_problem(initial_mesh, MESH_TYPE, mesh_dims)
        if u_initial is None: print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - FEM solve failed on initial. Skipping."); continue
        l2_error_initial = calculate_l2_error(u_initial, MESH_TYPE, mesh_dims, initial_mesh)
        print(f"  Initial L2 Error: {l2_error_initial:.6e} (DoFs: {u_initial.function_space().dim()})")
        if l2_error_initial <= -1.5: print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - Initial L2 error indicates problem ({l2_error_initial}). Skipping."); continue

        monitor_vals_np = get_solution_based_monitor_function(u_initial, initial_mesh)

        classical_start_time = time.time()
        optimized_coords_classical = dummy_classical_r_adaptivity(initial_mesh, monitor_vals_np, mesh_dimensions=mesh_dims)
        classical_duration = time.time() - classical_start_time
        
        l2_error_optimized_classical = -1.0
        optimized_mesh_classical_viz = dolfin.Mesh(initial_mesh)
        if optimized_coords_classical.shape[0] == optimized_mesh_classical_viz.num_vertices():
            optimized_mesh_classical_viz.coordinates()[:] = optimized_coords_classical
            
            quality_ok, _ = check_mesh_quality(optimized_mesh_classical_viz, "Classical R-Adapt")
            if not quality_ok:
                l2_error_optimized_classical = -3.0 # Error code for tangled mesh
            else:
                u_optimized_classical = solve_fem_problem(optimized_mesh_classical_viz, MESH_TYPE, mesh_dims)
                if u_optimized_classical is None: l2_error_optimized_classical = -2.0
                else: l2_error_optimized_classical = calculate_l2_error(u_optimized_classical, MESH_TYPE, mesh_dims, optimized_mesh_classical_viz)
            print(f"  Classical Adapted L2 Error: {l2_error_optimized_classical:.6e}")
        else: l2_error_optimized_classical = -4.0

        add_feat = None
        if USE_MONITOR_AS_FEATURE: add_feat = monitor_vals_np.reshape(-1, 1)

        pyg_data_sample = fenics_mesh_to_pyg_data(initial_mesh, device=DEVICE, additional_features=add_feat)
        if pyg_data_sample.num_nodes == 0: print(f"  Warning: ... empty PyG graph. Skipping."); continue

        pyg_data_sample.y = torch.tensor(optimized_coords_classical, dtype=torch.float).to(DEVICE)
        pyg_data_sample.classical_time = classical_duration
        pyg_data_sample.l2_error_initial = l2_error_initial
        pyg_data_sample.l2_error_classical_adapted = l2_error_optimized_classical
        pyg_data_sample.num_dofs = u_initial.function_space().dim() if u_initial else -1
        pyg_data_sample.mesh_type_str = MESH_TYPE
        pyg_data_sample.mesh_dimensions_str = str(mesh_dims)
        pyg_data_sample.original_coords_str = str(initial_mesh.coordinates()) # Store original coords for boundary fixing

        dataset.append(pyg_data_sample); all_classical_times.append(classical_duration)
        all_initial_errors.append(l2_error_initial); all_classical_adapted_errors.append(l2_error_optimized_classical)
        generated_count += 1
        
        if plot_first_sample_details and generated_count == 1: # ... (plotting code as before) ...
            print(f"  Plotting details for first successful sample...")
            fig_mesh, axs_mesh = plt.subplots(1, 2, figsize=(12, 5))
            plt.sca(axs_mesh[0]); dolfin.plot(initial_mesh); axs_mesh[0].set_title(f"Initial Mesh (Sample 1, {current_res_info})\nL2 Err: {l2_error_initial:.2e}")
            if MESH_TYPE == 'pipe': axs_mesh[0].set_aspect('equal')
            plt.sca(axs_mesh[1]); dolfin.plot(optimized_mesh_classical_viz); axs_mesh[1].set_title(f"Classical Adapted (Sample 1)\nL2 Err: {l2_error_optimized_classical:.2e}")
            if MESH_TYPE == 'pipe': axs_mesh[1].set_aspect('equal')
            plt.tight_layout(); plot_filename = os.path.join(session_output_dir, f"{MODEL_NAME}_Sample1_Meshes.png")
            plt.savefig(plot_filename); plt.close(fig_mesh); print(f"    Saved mesh plot to {plot_filename}")

            fig_sol, axs_sol = plt.subplots(1, 2, figsize=(12, 5))
            plt.sca(axs_sol[0]); cax0 = dolfin.plot(u_initial, title="Initial Solution (Sample 1)"); plt.colorbar(cax0, ax=axs_sol[0])
            if MESH_TYPE == 'pipe': axs_sol[0].set_aspect('equal')
            if 'u_optimized_classical' in locals() and u_optimized_classical is not None:
                plt.sca(axs_sol[1]); cax1 = dolfin.plot(u_optimized_classical, title="Classical Adapted Solution (Sample 1)"); plt.colorbar(cax1, ax=axs_sol[1])
                if MESH_TYPE == 'pipe': axs_sol[1].set_aspect('equal')
            else: axs_sol[1].text(0.5, 0.5, "No classical solution plotted", ha='center', va='center')
            plt.tight_layout(); plot_filename_sol = os.path.join(session_output_dir, f"{MODEL_NAME}_Sample1_Solutions.png")
            plt.savefig(plot_filename_sol); plt.close(fig_sol); print(f"    Saved solution plot to {plot_filename_sol}")

            plt.figure(figsize=(7,5))
            V_monitor_plot = dolfin.FunctionSpace(initial_mesh, "CG", 1)
            monitor_func_plot = dolfin.Function(V_monitor_plot)
            vertex_to_dof_map_mon = dolfin.vertex_to_dof_map(V_monitor_plot)
            if monitor_vals_np.shape[0] == V_monitor_plot.dim():
                 monitor_func_plot.vector()[:] = monitor_vals_np[vertex_to_dof_map_mon]
                 cax_m = dolfin.plot(monitor_func_plot, title="Monitor Function (grad(u_initial) norm, Sample 1)")
                 plt.colorbar(cax_m);
                 if MESH_TYPE == 'pipe': plt.gca().set_aspect('equal')
                 plt.tight_layout(); plot_filename_mon = os.path.join(session_output_dir, f"{MODEL_NAME}_Sample1_MonitorFunction.png")
                 plt.savefig(plot_filename_mon); plt.close(); print(f"    Saved monitor function plot to {plot_filename_mon}")
            else: print(f"    Skipping monitor plot due to shape/map mismatch: {monitor_vals_np.shape[0]} vs {V_monitor_plot.dim()}")

        if generated_count > 0 and generated_count % (num_samples // 10 or 1) == 0:
            print(f"  Generated {generated_count}/{num_samples} valid samples (Total attempts: {attempts}).")
    if generated_count < num_samples: print(f"Warning: Only generated {generated_count}/{num_samples} valid samples after {max_attempts} attempts.")
    if not dataset: print("CRITICAL: Dataset generation resulted in NO valid samples.")
    return dataset, all_classical_times, all_initial_errors, all_classical_adapted_errors

# --- Main Script ---
if __name__ == '__main__':
    # ... (Setup: dolfin log, session dir, cuda status - same as before) ...
    dolfin.set_log_level(dolfin.LogLevel.WARNING)
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"session_{session_timestamp}")
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)
    print(f"Using MESH_TYPE: '{MESH_TYPE}'"); print(f"Output for this session: {SESSION_OUTPUT_DIR}")
    plot_funcs.cuda_status(DEVICE)
    
    dataset, classical_r_adapt_times_all, l2_errors_initial_all, l2_errors_classical_adapted_all = generate_dataset(
        NUM_SAMPLES, SESSION_OUTPUT_DIR, plot_first_sample_details=True
    )
    if not dataset: print("Dataset empty. Exiting."); exit()

    # ... (Dataset Split, DataLoader, Model, Optimizer, Loss - same as before) ...
    train_size = int(0.8 * len(dataset));
    if len(dataset) > 0 and train_size == 0 : train_size = 1
    if train_size >= len(dataset): train_dataset = dataset; val_dataset = list(dataset)
    else: train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]
    print(f"Dataset size: Total={len(dataset)}, Train={len(train_dataset)}, Val={len(val_dataset)}")
    if not train_dataset: print("Error: Training dataset empty."); exit()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None
    if dataset and hasattr(dataset[0], 'x') and dataset[0].x is not None:
        in_feat_dim = dataset[0].x.size(1); print(f"Determined input feature dimension: {in_feat_dim}")
    else: print("Error: Cannot determine input feature dimension."); exit()
    gat_model = RAdaptGAT(in_channels=in_feat_dim, hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS, heads=HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(gat_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    print(f"\n--- Training {MODEL_NAME} on {DEVICE} ---")
    
    # ... (Training Loop - same as before) ...
    epochs_list, train_losses_history, val_losses_history, gat_epoch_times_train = [], [], [], []
    for epoch in range(EPOCHS):
        epoch_start_time = time.time(); gat_model.train()
        current_epoch_train_loss, num_train_batches = 0,0
        for batch in train_loader:
            optimizer.zero_grad()
            out = gat_model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y); loss.backward(); optimizer.step()
            current_epoch_train_loss += loss.item(); num_train_batches +=1
        avg_epoch_train_loss = current_epoch_train_loss / num_train_batches if num_train_batches >0 else float('nan')
        train_losses_history.append(avg_epoch_train_loss)
        gat_epoch_times_train.append(time.time() - epoch_start_time)
        current_epoch_val_loss, num_val_batches = 0,0
        if val_loader:
            gat_model.eval()
            with torch.inference_mode():
                for batch in val_loader:
                    out = gat_model(batch.x, batch.edge_index)
                    loss = loss_fn(out, batch.y)
                    current_epoch_val_loss += loss.item(); num_val_batches +=1
            avg_epoch_val_loss = current_epoch_val_loss / num_val_batches if num_val_batches > 0 else float('nan')
            val_losses_history.append(avg_epoch_val_loss)
        else: val_losses_history.append(float('nan'))
        epochs_list.append(epoch+1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses_history[-1]:.6f} | Val Loss: {val_losses_history[-1]:.6f} | Time: {gat_epoch_times_train[-1]:.2f}s")
    print("--- Training complete ---")
    if gat_epoch_times_train: print(f"Avg GAT training epoch time: {np.mean(gat_epoch_times_train):.4f}s")
    model_save_path = os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_epoch{EPOCHS}.pt")
    torch.save(gat_model.state_dict(), model_save_path)
    print(f"Saved trained model state_dict to: {model_save_path}")
    plot_funcs.loss_plot(epochs_list, train_losses_history, val_losses_history, MODEL_NAME, SESSION_OUTPUT_DIR, show=False)

    # --- Validation Set Evaluation & Benchmarking ---
    val_classical_l2_errors, val_gat_l2_errors, val_initial_l2_errors = [], [], []
    val_dofs_list = []; val_classical_times_list_bench = []; val_gat_inference_times_list_bench = []

    if val_dataset:
        print(f"\n--- Evaluation on Validation Set ({len(val_dataset)} samples) ---")
        # ... (PredVsTrue plotting as before) ...

        num_inference_runs_per_sample = 5
        for i, data_sample in enumerate(val_dataset):
            print(f"  Processing validation sample {i+1}/{len(val_dataset)}...")
            try: current_mesh_dims = eval(data_sample.mesh_dimensions_str)
            except Exception as e_eval: print(f"    Error eval mesh_dims: '{data_sample.mesh_dimensions_str}' - {e_eval}. Skip."); continue
            current_mesh_type = data_sample.mesh_type_str
            
            # Regenerate initial mesh from stored parameters
            initial_mesh_val = None; original_coords_val = None
            if hasattr(data_sample, 'original_coords_str'): # Use stored coords if available
                try: original_coords_val = np.array(eval(data_sample.original_coords_str))
                except: print("Could not eval original_coords_str"); original_coords_val = None
            
            if current_mesh_type == 'square':
                nx_val = current_mesh_dims.get('nx'); ny_val = current_mesh_dims.get('ny')
                if nx_val is None or ny_val is None: print(f"    nx/ny not in mesh_dims for val sample {i+1}. Skip."); continue
                initial_mesh_val, _ = create_square_mesh(nx_val, ny_val)
            elif current_mesh_type == 'pipe': # ... (pipe mesh regen) ...
                ms_factor_val = current_mesh_dims.get('mesh_size_factor', (MESH_SIZE_FACTOR_MIN + MESH_SIZE_FACTOR_MAX) / 2.0)
                initial_mesh_val, _ = create_pipe_with_obstacle_mesh_gmsh(mesh_size_factor=ms_factor_val, pipe_length=PIPE_LENGTH, pipe_height=PIPE_HEIGHT, obstacle_cx_factor=OBSTACLE_CENTER_X_FACTOR, obstacle_cy_factor=OBSTACLE_CENTER_Y_FACTOR, obstacle_r_factor=OBSTACLE_RADIUS_FACTOR)

            if initial_mesh_val is None or initial_mesh_val.num_cells() == 0: print(f"    Could not regen val mesh {i+1}. Skip."); continue
            if original_coords_val is None or original_coords_val.shape[0] != initial_mesh_val.num_vertices():
                original_coords_val = initial_mesh_val.coordinates() # Fallback if not stored or mismatch

            boundary_node_indices_val = get_boundary_nodes(initial_mesh_val)


            u_initial_val = solve_fem_problem(initial_mesh_val, current_mesh_type, current_mesh_dims)
            if u_initial_val is None: print(f"    FEM solve failed initial val mesh {i+1}. Skip."); continue
            val_initial_l2_errors.append(calculate_l2_error(u_initial_val, current_mesh_type, current_mesh_dims, initial_mesh_val))
            val_dofs_list.append(u_initial_val.function_space().dim())
            
            monitor_val_np = get_solution_based_monitor_function(u_initial_val, initial_mesh_val)
            
            classical_start_val = time.time()
            opt_coords_classical_val = dummy_classical_r_adaptivity(initial_mesh_val, monitor_val_np, mesh_dimensions=current_mesh_dims)
            val_classical_times_list_bench.append(time.time() - classical_start_val)
            classical_adapted_mesh_val = dolfin.Mesh(initial_mesh_val)
            if opt_coords_classical_val.shape[0] == classical_adapted_mesh_val.num_vertices():
                classical_adapted_mesh_val.coordinates()[:] = opt_coords_classical_val
                quality_ok_classical, _ = check_mesh_quality(classical_adapted_mesh_val, "Classical Val")
                if quality_ok_classical:
                    u_class_val = solve_fem_problem(classical_adapted_mesh_val, current_mesh_type, current_mesh_dims)
                    if u_class_val: val_classical_l2_errors.append(calculate_l2_error(u_class_val, current_mesh_type, current_mesh_dims, classical_adapted_mesh_val))
                    else: val_classical_l2_errors.append(-2.0)
                else: val_classical_l2_errors.append(-3.0) # Tangled
            else: val_classical_l2_errors.append(-4.0)

            # GAT Evaluation
            add_feat_val = monitor_val_np.reshape(-1,1) if USE_MONITOR_AS_FEATURE else None
            pyg_val_sample = fenics_mesh_to_pyg_data(initial_mesh_val, device=DEVICE, additional_features=add_feat_val)
            current_gat_inference_times = []; gat_adapted_coords_val_np = None
            if pyg_val_sample.num_nodes > 0 and pyg_val_sample.x is not None:
                gat_model.eval();
                with torch.inference_mode():
                    gat_adapted_coords_raw = gat_model(pyg_val_sample.x, pyg_val_sample.edge_index)
                    gat_adapted_coords_val_np = gat_adapted_coords_raw.cpu().numpy()
                    
                    # Robust Boundary Handling & Clipping for GAT output
                    gat_adapted_coords_val_np[:, 0] = np.clip(gat_adapted_coords_val_np[:, 0], 0.0, current_mesh_dims.get("width",1.0))
                    gat_adapted_coords_val_np[:, 1] = np.clip(gat_adapted_coords_val_np[:, 1], 0.0, current_mesh_dims.get("height",1.0))
                    if original_coords_val is not None and boundary_node_indices_val: # Fix boundary nodes
                        for bn_idx in boundary_node_indices_val:
                            if bn_idx < gat_adapted_coords_val_np.shape[0] and bn_idx < original_coords_val.shape[0]:
                                gat_adapted_coords_val_np[bn_idx] = original_coords_val[bn_idx]
                    
                    for _ in range(num_inference_runs_per_sample): # Timing
                        start_time_gat = time.time(); _ = gat_model(pyg_val_sample.x, pyg_val_sample.edge_index); current_gat_inference_times.append(time.time() - start_time_gat)
                if current_gat_inference_times: val_gat_inference_times_list_bench.append(np.mean(current_gat_inference_times))
            else: val_gat_inference_times_list_bench.append(0.0)

            if gat_adapted_coords_val_np is not None:
                gat_adapted_mesh_val = dolfin.Mesh(initial_mesh_val)
                if gat_adapted_coords_val_np.shape[0] == gat_adapted_mesh_val.num_vertices():
                    gat_adapted_mesh_val.coordinates()[:] = gat_adapted_coords_val_np
                    quality_ok_gat, _ = check_mesh_quality(gat_adapted_mesh_val, "GAT Val")
                    if quality_ok_gat:
                        u_gat_val = solve_fem_problem(gat_adapted_mesh_val, current_mesh_type, current_mesh_dims)
                        if u_gat_val: val_gat_l2_errors.append(calculate_l2_error(u_gat_val, current_mesh_type, current_mesh_dims, gat_adapted_mesh_val))
                        else: val_gat_l2_errors.append(-2.0)
                    else: val_gat_l2_errors.append(-3.0) # Tangled
                else: val_gat_l2_errors.append(-4.0)
            else: val_gat_l2_errors.append(-5.0)
        # ... (Reporting and Plotting validation results) ...

    # --- Example Inference on a New Mesh ---
    print(f"\n--- Example Inference & Plot ({MESH_TYPE} Geometry) ---")
    # ... (Example mesh generation) ...
    l2_err_initial_ex, l2_err_classical_ex, l2_err_gat_ex = -1.0,-1.0,-1.0
    u_initial_ex, u_classical_ex, u_gat_ex = None, None, None
    example_classical_adapted_mesh, example_gat_adapted_mesh = None, None
    predicted_optimized_coords_gat_np = None # Initialize

    if MESH_TYPE == 'square':
        nx_ex = (MESH_SIZE_MIN + MESH_SIZE_MAX) // 2 + 2 # Slightly different example
        ny_ex = (MESH_SIZE_MIN + MESH_SIZE_MAX) // 2 + 2
        example_initial_mesh, example_mesh_dims = create_square_mesh(nx_ex, ny_ex)
        example_res_info = f"nx={nx_ex}, ny={ny_ex}"
    # ... (Pipe case)

    if example_initial_mesh is None or example_initial_mesh.num_vertices() == 0: print("Error: Failed to create example mesh.")
    else:
        original_coords_ex = example_initial_mesh.coordinates() # Get original coords for boundary fixing
        boundary_node_indices_ex = get_boundary_nodes(example_initial_mesh)

        u_initial_ex = solve_fem_problem(example_initial_mesh, MESH_TYPE, example_mesh_dims)
        if u_initial_ex:
            l2_err_initial_ex = calculate_l2_error(u_initial_ex, MESH_TYPE, example_mesh_dims, example_initial_mesh)
            print(f"  Example Initial L2 Error: {l2_err_initial_ex:.4e}")
            monitor_vals_ex_np = get_solution_based_monitor_function(u_initial_ex, example_initial_mesh)
            
            example_optimized_classical_coords = dummy_classical_r_adaptivity(example_initial_mesh, monitor_vals_ex_np, mesh_dimensions=example_mesh_dims)
            example_classical_adapted_mesh = dolfin.Mesh(example_initial_mesh)
            if example_optimized_classical_coords.shape[0] == example_classical_adapted_mesh.num_vertices():
                example_classical_adapted_mesh.coordinates()[:] = example_optimized_classical_coords
                quality_ok_ex_classical, _ = check_mesh_quality(example_classical_adapted_mesh, "Classical Example")
                if quality_ok_ex_classical:
                    u_classical_ex = solve_fem_problem(example_classical_adapted_mesh, MESH_TYPE, example_mesh_dims)
                    if u_classical_ex: l2_err_classical_ex = calculate_l2_error(u_classical_ex, MESH_TYPE, example_mesh_dims, example_classical_adapted_mesh)
                else: l2_err_classical_ex = -3.0
                print(f"  Example Classical Adapted L2 Error: {l2_err_classical_ex:.4e}")

            add_feat_ex = monitor_vals_ex_np.reshape(-1,1) if USE_MONITOR_AS_FEATURE else None
            pyg_example_data = fenics_mesh_to_pyg_data(example_initial_mesh, device=DEVICE, additional_features=add_feat_ex)
            
            if pyg_example_data.num_nodes > 0 and pyg_example_data.x is not None:
                gat_model.eval()
                with torch.inference_mode():
                    predicted_optimized_coords_gat_raw = gat_model(pyg_example_data.x, pyg_example_data.edge_index)
                    predicted_optimized_coords_gat_np = predicted_optimized_coords_gat_raw.cpu().numpy()
                    
                    # Robust Boundary Handling & Clipping for GAT example output
                    predicted_optimized_coords_gat_np[:, 0] = np.clip(predicted_optimized_coords_gat_np[:, 0], 0.0, example_mesh_dims.get("width",1.0))
                    predicted_optimized_coords_gat_np[:, 1] = np.clip(predicted_optimized_coords_gat_np[:, 1], 0.0, example_mesh_dims.get("height",1.0))
                    if boundary_node_indices_ex: # Fix boundary nodes
                        for bn_idx in boundary_node_indices_ex:
                             if bn_idx < predicted_optimized_coords_gat_np.shape[0] and bn_idx < original_coords_ex.shape[0]:
                                predicted_optimized_coords_gat_np[bn_idx] = original_coords_ex[bn_idx]
                
                if predicted_optimized_coords_gat_np is not None:
                    example_gat_adapted_mesh = dolfin.Mesh(example_initial_mesh)
                    if predicted_optimized_coords_gat_np.shape[0] == example_gat_adapted_mesh.num_vertices():
                        example_gat_adapted_mesh.coordinates()[:] = predicted_optimized_coords_gat_np
                        quality_ok_ex_gat, _ = check_mesh_quality(example_gat_adapted_mesh, "GAT Example")
                        if quality_ok_ex_gat:
                            u_gat_ex = solve_fem_problem(example_gat_adapted_mesh, MESH_TYPE, example_mesh_dims)
                            if u_gat_ex: l2_err_gat_ex = calculate_l2_error(u_gat_ex, MESH_TYPE, example_mesh_dims, example_gat_adapted_mesh)
                        else: l2_err_gat_ex = -3.0 # Tangled
                    print(f"  Example GAT Adapted L2 Error: {float(l2_err_gat_ex):.4e}") # Explicit float cast for print
            # ... (Plotting code for example - same as your working version) ...
    print(f"\nAll outputs saved to: {SESSION_OUTPUT_DIR}")
