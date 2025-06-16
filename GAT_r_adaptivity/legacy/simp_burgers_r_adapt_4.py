# simp_burgers_r_adapt_4.py
import dolfin
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import os
import datetime
import json
import h5py   # Do not touch, have preload for some reason or else the bin fail

# --- Import from our new mesh generation script ---
try:
    from mesh_generators_2 import create_square_mesh, create_pipe_with_obstacle_mesh_gmsh
except ImportError as e: # Capture the exception instance as 'e'
    print("CRITICAL ERROR during import of mesh_generators.py. Original error was:")
    print(e) # Print the original error message
    print("\nMake sure mesh_generators.py is in the same directory and that all its internal imports (like 'mshr' or 'dolfin') are working correctly within your FEniCS environment.")
    exit()
# Assuming these are in separate files or defined above as per your setup
from models.GAT import RAdaptGAT # Make sure this path is correct
from fenics_mesh_to_pyg_data import fenics_mesh_to_pyg_data

# --- Import functions from your plot_funcs.py ---
try:
    import plot_funcs
except ImportError:
    print("Error: plot_funcs.py not found. Make sure it's in the same directory or in PYTHONPATH.")
    class plot_funcs_dummy: # Dummy class if import fails
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
        def plot_accuracy_vs_cost(classical_costs, classical_accuracies, gat_costs, gat_accuracies, output="", show=True, **kwargs): print("Dummy plot_accuracy_vs_cost called.")
        @staticmethod
        def plot_convergence(classical_dofs, classical_errors, gat_dofs, gat_errors, output="", show=True, **kwargs): print("Dummy plot_convergence called.")
    plot_funcs = plot_funcs_dummy
# --- End Plot Funcs Import ---

# --- Global Parameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 10
NUM_SAMPLES = 100 # Number of mesh samples to generate for the dataset

# --- Mesh Type Selector ---
# Choose 'square' or 'pipe'
MESH_TYPE = 'square' # <<<< ------ SWITCH MESH TYPE HERE ------ >>>>

# --- Parameters specific to MESH_TYPE ---
if MESH_TYPE == 'square':
    MODEL_NAME_SUFFIX = "SquareMesh"
    MESH_SIZE_MIN = 5       # For UnitSquareMesh: min nx or ny
    MESH_SIZE_MAX = 10      # For UnitSquareMesh: max nx or ny
    MESH_SIZE_FACTOR_MIN = 0.08  # Example: for a somewhat finer mesh
    MESH_SIZE_FACTOR_MAX = 0.15  # Example: for a somewhat coarser mesh
    # For square, dimensions are fixed (unit square), feature center can be relative
    FEATURE_CENTER_X_FACTOR = 0.75 # Relative to width
    FEATURE_CENTER_Y_FACTOR = 0.75 # Relative to height
elif MESH_TYPE == 'pipe':
    MODEL_NAME_SUFFIX = "PipeObstacleMesh"
    MESH_RESOLUTION_MIN = 15 # For generate_mesh: min resolution
    MESH_RESOLUTION_MAX = 25 # For generate_mesh: max resolution
    MESH_SIZE_FACTOR_MIN = 0.08  # Example: for a somewhat finer mesh
    MESH_SIZE_FACTOR_MAX = 0.15  # Example: for a somewhat coarser mesh
    # Pipe Geometry Parameters (can also be inside mesh_generators.py as defaults)
    PIPE_LENGTH = 3.0
    PIPE_HEIGHT = 1.0
    OBSTACLE_CENTER_X_FACTOR = 0.3
    OBSTACLE_CENTER_Y_FACTOR = 0.5
    OBSTACLE_RADIUS_FACTOR = 0.15
    # For pipe, feature can be downstream of obstacle
    FEATURE_CENTER_X_FACTOR = 0.6 # Relative to pipe_length
    FEATURE_CENTER_Y_FACTOR = 0.5 # Relative to pipe_height

else:
    raise ValueError(f"Unknown MESH_TYPE: {MESH_TYPE}. Choose 'square' or 'pipe'.")

MODEL_NAME = f"RAdaptGAT_{MODEL_NAME_SUFFIX}"
BASE_OUTPUT_DIR = f"gat_{MODEL_NAME_SUFFIX.lower()}_plots"

# GNN Model Hyperparameters
HIDDEN_CHANNELS=128 # Adjusted
OUT_CHANNELS=2    # Outputting new x, y coordinates
HEADS=8           # Adjusted
NUM_LAYERS=4      # Adjusted
DROPOUT=0.5

# --- Monitor Function & R-Adaptivity (Now more generic) ---
def get_monitor_function(mesh_coords, mesh_dimensions):
    """
    Generates a monitor function based on mesh coordinates and dimensions.
    The feature is centered based on factors relative to the mesh dimensions.
    """
    width = mesh_dimensions.get("width", 1.0)
    height = mesh_dimensions.get("height", 1.0)

    # Use the globally set FEATURE_CENTER_X/Y_FACTOR based on MESH_TYPE
    feature_center_x = width * FEATURE_CENTER_X_FACTOR
    feature_center_y = height * FEATURE_CENTER_Y_FACTOR
    feature_center = np.array([feature_center_x, feature_center_y])

    # Example: focus on an area (e.g., downstream of obstacle for pipe, or a corner for square)
    distances_sq = np.sum((mesh_coords - feature_center)**2, axis=1)
    monitor_values = 1.0 / (distances_sq + 1e-2)

    min_val = np.min(monitor_values)
    max_val = np.max(monitor_values)
    if max_val - min_val < 1e-6:
        return np.ones_like(monitor_values) * 0.5
    return (monitor_values - min_val) / (max_val - min_val)

def dummy_classical_r_adaptivity(mesh, monitor_values, strength=0.05, mesh_dimensions=None):
    old_coords = np.copy(mesh.coordinates())
    new_coords = np.copy(old_coords)
    num_nodes = mesh.num_vertices()
    geo_dim = mesh.geometry().dim()

    boundary_nodes = set()
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    
    class AllBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    all_boundary_domain = AllBoundary()
    all_boundary_domain.mark(boundary_markers, 1)

    for f in dolfin.facets(mesh):
        if boundary_markers[f.index()] == 1:
            for v in dolfin.vertices(f):
                boundary_nodes.add(v.index())
    
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
            
            weight = (monitor_values[j] + monitor_values[i])/2.0 / (dist_sq + 1e-6)
            direction_vector += weight * diff
            total_weight += weight
        
        if total_weight > 1e-6:
            displacement = strength * (direction_vector / total_weight)
            new_coords[i] += displacement

    # Clipping coordinates to the domain boundaries
    if mesh_dimensions and mesh_dimensions.get("type") == "pipe":
        pipe_len = mesh_dimensions.get("width", 1.0) # Assuming width is length for pipe
        pipe_h = mesh_dimensions.get("height", 1.0)
        new_coords[:, 0] = np.clip(new_coords[:, 0], 0.0, pipe_len)
        new_coords[:, 1] = np.clip(new_coords[:, 1], 0.0, pipe_h)
        # More complex clipping for internal obstacle boundary would be needed for true accuracy
    elif mesh_dimensions and mesh_dimensions.get("type") == "square": # UnitSquare is 0 to 1
        new_coords[:, 0] = np.clip(new_coords[:, 0], 0.0, 1.0)
        new_coords[:, 1] = np.clip(new_coords[:, 1], 0.0, 1.0)
    # Else, no specific clipping if dimensions unknown, could be risky

    return new_coords

# --- Data Generation (Modified to use MESH_TYPE) ---
def generate_dataset(num_samples, session_output_dir, plot_first_sample_density=False):
    dataset, all_classical_times = [], []
    print(f"Generating {num_samples} data samples for MESH_TYPE: '{MESH_TYPE}'...")
    
    generated_count = 0
    attempts = 0
    max_attempts = num_samples * 2 # Try a bit harder to get the desired number of samples

    while generated_count < num_samples and attempts < max_attempts:
        attempts += 1
        initial_mesh = None
        mesh_dims = None # To store dimensions like width, height, center_x, center_y

        if MESH_TYPE == 'square':
            nx = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
            ny = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
            initial_mesh, mesh_dims = create_square_mesh(nx, ny)
            current_res_info = f"nx={nx}, ny={ny}"
        elif MESH_TYPE == 'pipe':
            # Use np.random.uniform for floating point factors
            current_mesh_size_factor = np.random.uniform(MESH_SIZE_FACTOR_MIN, MESH_SIZE_FACTOR_MAX)
            initial_mesh, mesh_dims = create_pipe_with_obstacle_mesh_gmsh(
                    mesh_size_factor=current_mesh_size_factor, # Pass the correct argument
                    pipe_length=PIPE_LENGTH, pipe_height=PIPE_HEIGHT,
                    obstacle_cx_factor=OBSTACLE_CENTER_X_FACTOR,
                    obstacle_cy_factor=OBSTACLE_CENTER_Y_FACTOR,
                    obstacle_r_factor=OBSTACLE_RADIUS_FACTOR
                    )
            current_res_info = f"Factor={current_mesh_size_factor:.3f}" # Update info string

        if initial_mesh is None or initial_mesh.num_cells() == 0:
            print(f"Warning: Attempt {attempts}, Sample {generated_count+1} resulted in an empty/invalid mesh. Skipping.")
            continue

        initial_coords = initial_mesh.coordinates()
        if initial_coords.shape[0] == 0:
            print(f"Warning: Attempt {attempts}, Sample {generated_count+1} has no coordinates. Skipping.")
            continue

        monitor_vals = get_monitor_function(initial_coords, mesh_dims)

        classical_start_time = time.time()
        optimized_coords_classical = dummy_classical_r_adaptivity(initial_mesh, monitor_vals, mesh_dimensions=mesh_dims)
        classical_duration = time.time() - classical_start_time
        all_classical_times.append(classical_duration)

        pyg_data_sample = fenics_mesh_to_pyg_data(initial_mesh, device=DEVICE)
        
        if pyg_data_sample.num_nodes == 0:
             print(f"Warning: Attempt {attempts}, Sample {generated_count+1} resulted in an empty PyG graph. Skipping.")
             continue

        pyg_data_sample.y = torch.tensor(optimized_coords_classical, dtype=torch.float).to(DEVICE)
        pyg_data_sample.classical_time = classical_duration
        pyg_data_sample.mesh_type = MESH_TYPE # Store mesh type if needed later
        pyg_data_sample.mesh_dimensions = str(mesh_dims) # Store dimensions as string
            
        dataset.append(pyg_data_sample)
        generated_count += 1

        if plot_first_sample_density and generated_count == 1 and initial_coords.shape[0] > 0:
            plot_funcs.density_plot_matrix(initial_coords, output=session_output_dir, title=f"{MODEL_NAME} Initial ({current_res_info}, Sample 1)", show=False)
            plot_funcs.density_plot_matrix(optimized_coords_classical, output=session_output_dir, title=f"{MODEL_NAME} Classical ({current_res_info}, Sample 1)", show=False)
            
            fig_mesh, axs_mesh = plt.subplots(1, 2, figsize=(12, 5))
            plt.sca(axs_mesh[0])
            dolfin.plot(initial_mesh)
            axs_mesh[0].set_title(f"Initial Mesh (Sample 1, {current_res_info})")
            if MESH_TYPE == 'pipe': axs_mesh[0].set_aspect('equal')


            optimized_mesh_viz = dolfin.Mesh(initial_mesh)
            if optimized_coords_classical.shape[0] == optimized_mesh_viz.num_vertices():
                 optimized_mesh_viz.coordinates()[:] = optimized_coords_classical
                 plt.sca(axs_mesh[1])
                 dolfin.plot(optimized_mesh_viz)
                 axs_mesh[1].set_title(f"Classical Adapted (Sample 1)")
                 if MESH_TYPE == 'pipe': axs_mesh[1].set_aspect('equal')

            else:
                print(f"Warning: Coord shape mismatch for plotting optimized mesh. Expected {optimized_mesh_viz.num_vertices()}, got {optimized_coords_classical.shape[0]}")
            
            plt.tight_layout()
            plot_filename = os.path.join(session_output_dir, f"{MODEL_NAME}_Sample1_Mesh_Initial_vs_Classical.png")
            plt.savefig(plot_filename)
            plt.close(fig_mesh)
            print(f"  Saved mesh plot for sample 1 to {plot_filename}")
            print(f"  Saved density plots for sample 1 to {session_output_dir}/")

        if generated_count % (num_samples // 10 or 1) == 0:
            print(f"  Generated {generated_count}/{num_samples} samples (attempts: {attempts}).")
    
    if generated_count < num_samples:
        print(f"Warning: Only generated {generated_count}/{num_samples} valid samples after {attempts} attempts.")
    if not dataset:
        raise ValueError("Dataset generation resulted in NO valid samples. Check mesh creation parameters and functions.")
    return dataset, all_classical_times

# --- Main Script ---
if __name__ == '__main__':
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"session_{session_timestamp}")
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)
    print(f"Using MESH_TYPE: '{MESH_TYPE}'")
    print(f"Saving plots and model for this session to: {SESSION_OUTPUT_DIR}")

    plot_funcs.cuda_status(DEVICE)
    
    try:
        dataset, classical_r_adapt_times_all = generate_dataset(NUM_SAMPLES, SESSION_OUTPUT_DIR, plot_first_sample_density=True)
    except ValueError as e:
        print(f"Critical Error during data generation: {e}")
        exit()

    train_size = int(0.8 * len(dataset))
    if train_size == 0 and len(dataset) > 0: train_size = 1
    
    if train_size >= len(dataset):
        train_dataset, val_dataset = dataset, []
    else:
        train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]

    if not train_dataset:
        print("Error: Training dataset is empty. Cannot proceed.")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None

    if dataset and dataset[0].x is not None:
        in_feat_dim = dataset[0].x.size(1)
    else:
        print("Error: Dataset is empty or first sample has no features 'x'. Cannot determine input feature dimension.")
        exit()

    gat_model = RAdaptGAT(in_channels=in_feat_dim, hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS, heads=HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(gat_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    print(f"\nTraining {MODEL_NAME} on {DEVICE}...")
    if classical_r_adapt_times_all: print(f"Average classical r-adaptivity time for generated dataset: {np.mean(classical_r_adapt_times_all):.4f}s")

    epochs_list, train_losses_history, val_losses_history, gat_epoch_times_train = [], [], [], []
    # (Training loop remains largely the same, ensure it handles val_loader being None)
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        gat_model.train()
        current_epoch_train_loss, num_train_batches = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = gat_model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            current_epoch_train_loss += loss.item() * batch.num_graphs
            num_train_batches += batch.num_graphs
        train_losses_history.append(current_epoch_train_loss / num_train_batches if num_train_batches > 0 else float('nan'))
        gat_epoch_times_train.append(time.time() - epoch_start_time)

        current_epoch_val_loss, num_val_batches = 0, 0
        if val_loader:
            gat_model.eval()
            with torch.inference_mode():
                for batch in val_loader:
                    out = gat_model(batch.x, batch.edge_index)
                    loss = loss_fn(out, batch.y)
                    current_epoch_val_loss += loss.item() * batch.num_graphs
                    num_val_batches += batch.num_graphs
            val_losses_history.append(current_epoch_val_loss / num_val_batches if num_val_batches > 0 else float('nan'))
        else:
            val_losses_history.append(float('nan'))
        epochs_list.append(epoch + 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses_history[-1]:.6f} | Val Loss: {val_losses_history[-1]:.6f} | Time: {gat_epoch_times_train[-1]:.2f}s")


    print("Training complete.")
    if gat_epoch_times_train: print(f"Avg GAT training epoch time: {np.mean(gat_epoch_times_train):.4f}s")

    model_save_path = os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_epoch{EPOCHS}.pt")
    torch.save(gat_model.state_dict(), model_save_path)
    print(f"Saved trained model state_dict to: {model_save_path}")

    plot_funcs.loss_plot(epochs_list, train_losses_history, val_losses_history, MODEL_NAME, SESSION_OUTPUT_DIR, False)
    print(f"Saved training loss plot to {SESSION_OUTPUT_DIR}/")

    # (Validation plot and benchmark sections remain largely the same, ensure they handle val_loader/val_dataset being None/empty)
    if val_loader and val_dataset:
        val_batch_sample = next(iter(val_loader)).to(DEVICE)
        gat_model.eval()
        with torch.inference_mode():
            val_pred_np = gat_model(val_batch_sample.x, val_batch_sample.edge_index).cpu().numpy()
        val_true_np = val_batch_sample.y.cpu().numpy()
        plot_funcs.predVStrue([val_true_np.flatten()], [val_pred_np.flatten()], [], [], f"{MODEL_NAME}_ValBatch", SESSION_OUTPUT_DIR, False)
        print(f"Saved Pred vs True plot for val batch to {SESSION_OUTPUT_DIR}/")

        # Time Benchmark over Validation Set
        print("\n--- Time Benchmark over Validation Set ---")
        val_classical_times_list = [s.classical_time for s in val_dataset if hasattr(s, 'classical_time') and s.classical_time is not None]
        val_gat_inference_times_list = []
        num_inference_runs_per_sample = 5
        gat_model.eval()
        for data_sample_idx, data_sample in enumerate(val_dataset):
            current_x, current_edge_index = data_sample.x.to(DEVICE), data_sample.edge_index.to(DEVICE)
            sample_gat_times_for_avg = []
            for _ in range(num_inference_runs_per_sample):
                with torch.inference_mode():
                    start_time = time.time(); _ = gat_model(current_x, current_edge_index); end_time = time.time()
                    sample_gat_times_for_avg.append(end_time - start_time)
            if sample_gat_times_for_avg: val_gat_inference_times_list.append(np.mean(sample_gat_times_for_avg))
            if (data_sample_idx + 1) % (len(val_dataset) // 5 or 1) == 0: print(f"  Benchmarked {data_sample_idx + 1}/{len(val_dataset)} val samples for time.")
        
        if val_classical_times_list and val_gat_inference_times_list:
            # (Reporting and JSON saving logic - condensed for brevity, similar to previous version)
            classical_mean, gat_mean = np.mean(val_classical_times_list), np.mean(val_gat_inference_times_list)
            print(f"Classical R-Adaptivity (Val Set): Mean={classical_mean:.6f}s ({len(val_classical_times_list)} samples)")
            print(f"GAT Inference (Val Set): Mean={gat_mean:.6f}s ({len(val_gat_inference_times_list)} samples, avg over {num_inference_runs_per_sample} runs each)")
            # Save JSON and plots as before
            # ... (Ensure benchmark_summary includes MESH_TYPE specific params if any)
            benchmark_summary = {
                "model_name": MODEL_NAME, "mesh_type": MESH_TYPE, "session_timestamp": session_timestamp, "device": str(DEVICE),
                 "parameters": {
                    "num_total_samples": NUM_SAMPLES, "num_validation_samples_benchmarked": len(val_dataset),
                    "epochs_trained": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                    "gat_params": {"in_feat": in_feat_dim, "hidden": HIDDEN_CHANNELS, "out_feat": OUT_CHANNELS, "heads": HEADS, "layers": NUM_LAYERS, "dropout": DROPOUT},
                    # Add MESH_TYPE specific params to dict if they vary per run beyond global definition
                },
                "classical_r_adaptivity_times_seconds": {"all_values_on_val_set": val_classical_times_list, "mean": classical_mean},
                "gat_inference_times_seconds": {"all_values_on_val_set": val_gat_inference_times_list, "mean": gat_mean}
            }
            benchmark_file_path = os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_time_benchmark_summary.json")
            try:
                with open(benchmark_file_path, 'w') as f: json.dump(benchmark_summary, f, indent=4)
                print(f"Saved time benchmark summary to: {benchmark_file_path}")
            except Exception as e: print(f"Error saving benchmark summary to JSON: {e}")

            plot_funcs.plot_time_comparison([classical_mean], [gat_mean], title=f"{MODEL_NAME} Avg. Time (Val Set)", output=SESSION_OUTPUT_DIR, use_box_plot=False, show=False)
            plot_funcs.plot_time_comparison(val_classical_times_list, val_gat_inference_times_list, title=f"{MODEL_NAME} Time Distribution (Val Set)", output=SESSION_OUTPUT_DIR, use_box_plot=True, show=False)
        else: print("Not enough data for validation set time benchmark.")
    else: print("Validation dataset is empty. Skipping validation benchmarks.")


    print(f"\n--- Example Inference & Time Plot ({MESH_TYPE} Geometry) ---")
    example_initial_mesh, example_mesh_dims = None, None
    example_res_info = ""

    if MESH_TYPE == 'square':
        ex_nx, ex_ny = (MESH_SIZE_MIN + MESH_SIZE_MAX) // 2, (MESH_SIZE_MIN + MESH_SIZE_MAX) // 2
        example_initial_mesh, example_mesh_dims = create_square_mesh(ex_nx, ex_ny)
        example_res_info = f"nx={ex_nx}, ny={ex_ny}"
    elif MESH_TYPE == 'pipe':
        # Use an average or a specific factor for the example
        ex_mesh_size_factor = (MESH_SIZE_FACTOR_MIN + MESH_SIZE_FACTOR_MAX) / 2.0
        example_initial_mesh, example_mesh_dims = create_pipe_with_obstacle_mesh_gmsh(
            mesh_size_factor=ex_mesh_size_factor, # Pass the correct argument
            pipe_length=PIPE_LENGTH, pipe_height=PIPE_HEIGHT,
            obstacle_cx_factor=OBSTACLE_CENTER_X_FACTOR, obstacle_cy_factor=OBSTACLE_CENTER_Y_FACTOR,
            obstacle_r_factor=OBSTACLE_RADIUS_FACTOR
        )
        example_res_info = f"Factor={ex_mesh_size_factor:.3f}" # Update info string

    if example_initial_mesh is None or example_initial_mesh.num_vertices() == 0:
        print("Error: Failed to create example initial mesh for inference. Exiting example.")
        predicted_optimized_coords_gat_np = None # Ensure this is defined for the final plot check
    else:
        monitor_vals_test = get_monitor_function(example_initial_mesh.coordinates(), example_mesh_dims)
        classical_start_time = time.time()
        example_optimized_classical_coords = dummy_classical_r_adaptivity(example_initial_mesh, monitor_vals_test, mesh_dimensions=example_mesh_dims)
        classical_duration_test = time.time() - classical_start_time
        print(f"Classical r-adaptivity time for test {MESH_TYPE} mesh ({example_res_info}): {classical_duration_test:.6f}s")

        pyg_test_data = fenics_mesh_to_pyg_data(example_initial_mesh, device=DEVICE)
        
        if pyg_test_data.num_nodes == 0:
            print("Error: Failed to convert example test mesh to PyG data. Skipping GAT inference example.")
            predicted_optimized_coords_gat_np = None
        else:
            gat_model.eval()
            gat_inference_times, predicted_optimized_coords_gat_np = [], None
            num_test_inference_runs = 10
            for i in range(num_test_inference_runs):
                with torch.inference_mode():
                    gat_start_time = time.time()
                    current_preds = gat_model(pyg_test_data.x, pyg_test_data.edge_index)
                    gat_inference_times.append(time.time() - gat_start_time)
                    if i == num_test_inference_runs - 1: predicted_optimized_coords_gat_np = current_preds.cpu().numpy()
            
            avg_gat_duration_test = np.mean(gat_inference_times) if gat_inference_times else 0
            print(f"GAT inference time for test {MESH_TYPE} mesh ({example_res_info}, avg over {len(gat_inference_times)} runs): {avg_gat_duration_test:.6f}s")
            plot_funcs.plot_time_comparison([classical_duration_test], [avg_gat_duration_test], title=f"{MODEL_NAME} Inference Time ({MESH_TYPE}, {example_res_info})", output=SESSION_OUTPUT_DIR, show=False)

        # Final Visualization
        if predicted_optimized_coords_gat_np is not None and example_initial_mesh.num_vertices() > 0 and \
           example_optimized_classical_coords.shape[0] == example_initial_mesh.num_vertices() and \
           predicted_optimized_coords_gat_np.shape[0] == example_initial_mesh.num_vertices():
            
            fig_final, axs_final = plt.subplots(1, 3, figsize=(18, 6))
            common_title_prefix = f"{MESH_TYPE.capitalize()} Mesh ({example_res_info})"

            plt.sca(axs_final[0]); dolfin.plot(example_initial_mesh); axs_final[0].set_title(f"Initial {common_title_prefix}")
            if MESH_TYPE == 'pipe': axs_final[0].set_aspect('equal')
            
            adapted_mesh_classical_viz = dolfin.Mesh(example_initial_mesh)
            adapted_mesh_classical_viz.coordinates()[:] = example_optimized_classical_coords
            plt.sca(axs_final[1]); dolfin.plot(adapted_mesh_classical_viz); axs_final[1].set_title(f"Classical R-Adapted")
            if MESH_TYPE == 'pipe': axs_final[1].set_aspect('equal')

            adapted_mesh_gat_viz = dolfin.Mesh(example_initial_mesh)
            adapted_mesh_gat_viz.coordinates()[:] = predicted_optimized_coords_gat_np
            plt.sca(axs_final[2]); dolfin.plot(adapted_mesh_gat_viz); axs_final[2].set_title(f"GAT R-Adapted")
            if MESH_TYPE == 'pipe': axs_final[2].set_aspect('equal')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
            fig_final.suptitle(common_title_prefix + " Adaptation Comparison", fontsize=16)
            plt.savefig(os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_final_mesh_adaptation_comparison.png"))
            print(f"\nSaved final {MESH_TYPE} mesh comparison plot to {SESSION_OUTPUT_DIR}/")
            plt.close(fig_final)
        else:
            print("\nSkipping final mesh comparison: GAT/classical predictions or initial example mesh not valid, or shape mismatch.")
    
    print(f"\nAll generated plots and model saved to directory: {SESSION_OUTPUT_DIR}")
