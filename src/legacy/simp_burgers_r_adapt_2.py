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
import datetime # <--- ADD THIS IMPORT

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
        def plot_time_comparison(classical_times, gat_times, time_label='Mesh Optimization Time (s)', output="", title="", show=True, use_box_plot=False, **kwargs): # Added time_label
            print(f"Dummy plot_time_comparison called with title: {title}")
        @staticmethod
        def plot_accuracy_vs_cost(classical_costs, classical_accuracies, gat_costs, gat_accuracies, output="", show=True, **kwargs):
            print("Dummy plot_accuracy_vs_cost called.")
        @staticmethod
        def plot_convergence(classical_dofs, classical_errors, gat_dofs, gat_errors, output="", show=True, **kwargs):
            print("Dummy plot_convergence called.")
    plot_funcs = plot_funcs_dummy # Use dummy if import fails
# --- End Plot Funcs Import ---

# --- Parameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_SAMPLES = 100 # Number of mesh samples to generate for training
MESH_SIZE_MIN = 5
MESH_SIZE_MAX = 10 # Max N for NxN UnitSquareMesh
LEARNING_RATE = 1e-3
EPOCHS = 50 # Keep or increase for better loss plots
BATCH_SIZE = 10
MODEL_NAME = "RAdaptGAT_BurgersSim"
# BASE_OUTPUT_DIR will be the parent directory for all sessions
BASE_OUTPUT_DIR = "gat_burgers_plots"
# SESSION_OUTPUT_DIR will be defined dynamically in __main__
# os.makedirs(OUTPUT_DIR, exist_ok=True) # This will be done for SESSION_OUTPUT_DIR


# --- Simplified "Burgers' Solution" and Monitor Function ---
# (Your get_monitor_function and dummy_classical_r_adaptivity functions remain the same)
def get_monitor_function(mesh_coords):
    feature_center = np.array([0.75, 0.75])
    distances_sq = np.sum((mesh_coords - feature_center)**2, axis=1)
    monitor_values = 1.0 / (distances_sq + 1e-3)
    monitor_values = (monitor_values - np.min(monitor_values)) / (np.max(monitor_values) - np.min(monitor_values) + 1e-6)
    return monitor_values

def dummy_classical_r_adaptivity(mesh, monitor_values, strength=0.1):
    old_coords = np.copy(mesh.coordinates())
    new_coords = np.copy(old_coords)
    num_nodes = mesh.num_vertices()
    geo_dim = mesh.geometry().dim()
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    dolfin.DomainBoundary().mark(boundary_markers, 1)
    boundary_node_indices = set()
    marker_array = boundary_markers.array()
    for facet_idx in range(mesh.num_facets()):
        if marker_array[facet_idx] == 1:
            facet = dolfin.Facet(mesh, facet_idx)
            for vertex in dolfin.vertices(facet):
                boundary_node_indices.add(vertex.index())
    for i in range(num_nodes):
        if i in boundary_node_indices:
            continue
        direction_vector = np.zeros(geo_dim)
        total_weight = 0
        for j in range(num_nodes):
            if i == j: continue
            diff = old_coords[j] - old_coords[i]
            dist_sq = np.sum(diff**2) + 1e-6
            weight = monitor_values[j] / dist_sq
            direction_vector += weight * diff
            total_weight += weight
        if total_weight > 1e-6:
            new_coords[i] += strength * (direction_vector / total_weight)
    return np.clip(new_coords, 0.0, 1.0)

# --- Data Generation ---
# Modified to accept session_output_dir
def generate_dataset(num_samples, session_output_dir, plot_first_sample_density=False):
    dataset = []
    all_classical_times = []
    print(f"Generating {num_samples} data samples...")
    for i in range(num_samples):
        nx = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
        ny = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
        initial_mesh = dolfin.UnitSquareMesh(nx, ny)
        initial_coords = initial_mesh.coordinates()

        monitor_vals = get_monitor_function(initial_coords)
        classical_start_time = time.time()
        optimized_coords_classical = dummy_classical_r_adaptivity(initial_mesh, monitor_vals)
        classical_duration = time.time() - classical_start_time
        all_classical_times.append(classical_duration)

        pyg_data_sample = fenics_mesh_to_pyg_data(initial_mesh, device=DEVICE)
        pyg_data_sample.y = torch.tensor(optimized_coords_classical, dtype=torch.float).to(DEVICE)
        pyg_data_sample.classical_time = classical_duration
        dataset.append(pyg_data_sample)

        if plot_first_sample_density and i == 0:
            plot_funcs.density_plot_matrix(initial_coords, output=session_output_dir, title=f"{MODEL_NAME} Initial Coords (Sample 0)", show=False)
            plot_funcs.density_plot_matrix(optimized_coords_classical, output=session_output_dir, title=f"{MODEL_NAME} Classical Optimized Coords (Sample 0)", show=False)
            print(f"  Saved density plots for sample 0 to {session_output_dir}/")

        if (i + 1) % (num_samples // 10 or 1) == 0:
            print(f"  Generated {i+1}/{num_samples} samples.")
    return dataset, all_classical_times

# --- Main Script ---
if __name__ == '__main__':
    # --- Create a unique output directory for this session ---
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"session_{session_timestamp}")
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)
    print(f"Saving plots for this session to: {SESSION_OUTPUT_DIR}")
    # --- End output directory setup ---

    plot_funcs.cuda_status(DEVICE)

    # Pass SESSION_OUTPUT_DIR to generate_dataset
    dataset, classical_r_adapt_times_all = generate_dataset(NUM_SAMPLES, SESSION_OUTPUT_DIR, plot_first_sample_density=True)

    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    in_feat_dim = dataset[0].x.size(1)
    gat_model = RAdaptGAT(in_channels=in_feat_dim,
                          hidden_channels=64,
                          out_channels=2,
                          heads=4,
                          num_layers=3,
                          dropout=0.5).to(DEVICE)

    optimizer = optim.Adam(gat_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    print(f"\nTraining GAT model on {DEVICE}...")
    if classical_r_adapt_times_all:
        print(f"Average classical r-adaptivity time over {NUM_SAMPLES} samples: {np.mean(classical_r_adapt_times_all):.4f}s")

    epochs_list = []
    train_losses_history = []
    val_losses_history = []
    gat_epoch_times_train = []

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        gat_model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            predicted_positions = gat_model(batch.x, batch.edge_index)
            loss = loss_fn(predicted_positions, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs # Assuming PyG DataLoader behavior
        avg_train_loss = total_train_loss / len(train_loader.dataset) # Per-sample average loss
        train_losses_history.append(avg_train_loss)
        epoch_duration = time.time() - epoch_start_time
        gat_epoch_times_train.append(epoch_duration)

        gat_model.eval()
        total_val_loss = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                predicted_positions = gat_model(batch.x, batch.edge_index)
                total_val_loss += loss_fn(predicted_positions, batch.y).item() * batch.num_graphs
        avg_val_loss = total_val_loss / len(val_loader.dataset) # Per-sample average loss
        val_losses_history.append(avg_val_loss)
        epochs_list.append(epoch + 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_duration:.2f}s")

    print("Training complete.")
    if gat_epoch_times_train:
        print(f"Average GAT training epoch time: {np.mean(gat_epoch_times_train):.4f}s")

    # Update calls to use SESSION_OUTPUT_DIR
    plot_funcs.loss_plot(epoch_count=epochs_list,
                         loss_values=train_losses_history,
                         test_loss_values=val_losses_history,
                         model_name=MODEL_NAME,
                         output=SESSION_OUTPUT_DIR, # <--- MODIFIED
                         show=False)
    print(f"Saved training loss plot to {SESSION_OUTPUT_DIR}/")

    if val_loader:
        val_batch_sample = next(iter(val_loader))
        val_batch_sample = val_batch_sample.to(DEVICE)
        gat_model.eval()
        with torch.inference_mode():
            val_pred_positions_tensor = gat_model(val_batch_sample.x, val_batch_sample.edge_index)
            val_pred_positions_np = val_pred_positions_tensor.cpu().numpy()
        val_true_positions_np = val_batch_sample.y.cpu().numpy()
        plot_funcs.predVStrue(label_val_true=[val_true_positions_np.flatten()],
                              label_val_pred=[val_pred_positions_np.flatten()],
                              label_train_true=[],
                              label_train_pred=[],
                              model_name=MODEL_NAME + "_ValBatch",
                              output=SESSION_OUTPUT_DIR, # <--- MODIFIED
                              show=False)
        print(f"Saved Prediction vs True plot for a validation batch to {SESSION_OUTPUT_DIR}/")

    print("\n--- Example Inference & Time Plot ---")
    test_mesh_nx, test_mesh_ny = 8, 8
    example_initial_mesh = dolfin.UnitSquareMesh(test_mesh_nx, test_mesh_ny)
    monitor_vals_test = get_monitor_function(example_initial_mesh.coordinates())

    classical_start_time = time.time()
    example_optimized_classical_coords = dummy_classical_r_adaptivity(example_initial_mesh, monitor_vals_test)
    classical_duration_test = time.time() - classical_start_time
    print(f"Classical r-adaptivity time for test mesh: {classical_duration_test:.6f}s")

    pyg_test_data = fenics_mesh_to_pyg_data(example_initial_mesh, device=DEVICE)
    gat_model.eval()
    gat_inference_times = []
    for _ in range(10):
        with torch.inference_mode():
            gat_start_time = time.time()
            predicted_optimized_coords_gat_np = gat_model(pyg_test_data.x, pyg_test_data.edge_index).cpu().numpy()
            gat_inference_times.append(time.time() - gat_start_time)
    avg_gat_duration_test = np.mean(gat_inference_times)
    print(f"GAT inference time for test mesh (avg over 10 runs): {avg_gat_duration_test:.6f}s")

    plot_funcs.plot_time_comparison(classical_times=[classical_duration_test],
                                    gat_times=[avg_gat_duration_test],
                                    title=f"{MODEL_NAME} Inference Time: Classical vs GAT",
                                    time_label='Mesh Optimization Time (s)', # Make sure this kwarg is in plot_funcs
                                    output=SESSION_OUTPUT_DIR, # <--- MODIFIED
                                    use_box_plot=False,
                                    show=False)
    print(f"Saved inference time comparison plot to {SESSION_OUTPUT_DIR}/")

    print("\n--- Placeholder for Accuracy/Convergence Plots ---")
    # ... (comments remain the same, but ensure output=SESSION_OUTPUT_DIR in any calls)

    # --- Final Visualization of mesh adaptation (Corrected Version) ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    dolfin.plot(example_initial_mesh, ax=axs[0])
    axs[0].set_title("Initial Mesh")
    adapted_mesh_classical = dolfin.Mesh(example_initial_mesh)
    adapted_mesh_classical.coordinates()[:] = example_optimized_classical_coords
    dolfin.plot(adapted_mesh_classical, ax=axs[1])
    axs[1].set_title("Classical R-Adapted (Dummy)")
    adapted_mesh_gat = dolfin.Mesh(example_initial_mesh)
    adapted_mesh_gat.coordinates()[:] = predicted_optimized_coords_gat_np
    dolfin.plot(adapted_mesh_gat, ax=axs[2])
    axs[2].set_title("GAT R-Adapted")
    plt.tight_layout()
    plt.savefig(os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_mesh_adaptation_comparison.png")) # <--- MODIFIED
    print(f"\nSaved final mesh comparison plot to {SESSION_OUTPUT_DIR}/")
    # plt.show()

    print(f"\nAll generated plots saved to directory: {SESSION_OUTPUT_DIR}")
