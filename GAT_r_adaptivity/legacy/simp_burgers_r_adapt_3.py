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
from hessian_recovery import *

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

# --- Parameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_SAMPLES = 100
MESH_SIZE_MIN = 5
MESH_SIZE_MAX = 10
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 10
MODEL_NAME = "RAdaptGAT_BurgersSim"
BASE_OUTPUT_DIR = "gat_burgers_plots"

HIDDEN_CHANNELS=64
OUT_CHANNELS=2 
HEADS=4
NUM_LAYERS=4 # 3 to 5, more will lead to oversmoothing
DROPOUT=0.5

# --- Simplified Monitor Function & R-Adaptivity ---
def get_monitor_function(mesh_coords):
    feature_center = np.array([0.75, 0.75])
    distances_sq = np.sum((mesh_coords - feature_center)**2, axis=1)
    monitor_values = 1.0 / (distances_sq + 1e-3)
    return (monitor_values - np.min(monitor_values)) / (np.max(monitor_values) - np.min(monitor_values) + 1e-6)

def dummy_classical_r_adaptivity(mesh, monitor_values, strength=0.1):
    old_coords = np.copy(mesh.coordinates())
    new_coords = np.copy(old_coords)
    num_nodes = mesh.num_vertices()
    geo_dim = mesh.geometry().dim()
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    dolfin.DomainBoundary().mark(boundary_markers, 1)
    boundary_node_indices = {v.index() for f_idx in range(mesh.num_facets()) if boundary_markers.array()[f_idx] == 1 for v in dolfin.vertices(dolfin.Facet(mesh, f_idx))}
    for i in range(num_nodes):
        if i in boundary_node_indices: continue
        direction_vector, total_weight = np.zeros(geo_dim), 0
        for j in range(num_nodes):
            if i == j: continue
            diff = old_coords[j] - old_coords[i]
            weight = monitor_values[j] / (np.sum(diff**2) + 1e-6)
            direction_vector += weight * diff
            total_weight += weight
        if total_weight > 1e-6: new_coords[i] += strength * (direction_vector / total_weight)
    return np.clip(new_coords, 0.0, 1.0)

# --- Data Generation ---
def generate_dataset(num_samples, session_output_dir, plot_first_sample_density=False):
    dataset, all_classical_times = [], []
    print(f"Generating {num_samples} data samples...")
    for i in range(num_samples):
        nx, ny = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1, size=2)
        initial_mesh = dolfin.UnitSquareMesh(nx, ny)
        initial_coords = initial_mesh.coordinates()
        monitor_vals = get_monitor_function(initial_coords)

        classical_start_time = time.time()
        optimized_coords_classical = dummy_classical_r_adaptivity(initial_mesh, monitor_vals)
        classical_duration = time.time() - classical_start_time # Calculate and store duration
        all_classical_times.append(classical_duration) # Keep this for overall average if needed

        pyg_data_sample = fenics_mesh_to_pyg_data(initial_mesh, device=DEVICE)
        pyg_data_sample.y = torch.tensor(optimized_coords_classical, dtype=torch.float).to(DEVICE)
        
        # --- ADD THIS LINE ---
        pyg_data_sample.classical_time = classical_duration 
        # --- END OF ADDED LINE ---
            
        dataset.append(pyg_data_sample)

        if plot_first_sample_density and i == 0:
            plot_funcs.density_plot_matrix(initial_coords, output=session_output_dir, title=f"{MODEL_NAME} Initial (Sample 0)", show=False)
            plot_funcs.density_plot_matrix(optimized_coords_classical, output=session_output_dir, title=f"{MODEL_NAME} Classical (Sample 0)", show=False)
            print(f"  Saved density plots for sample 0 to {session_output_dir}/")
        if (i + 1) % (num_samples // 10 or 1) == 0: print(f"  Generated {i+1}/{num_samples} samples.")
    return dataset, all_classical_times

# --- Main Script ---
if __name__ == '__main__':
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"session_{session_timestamp}")
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)
    print(f"Saving plots and model for this session to: {SESSION_OUTPUT_DIR}")

    plot_funcs.cuda_status(DEVICE)
    dataset, classical_r_adapt_times_all = generate_dataset(NUM_SAMPLES, SESSION_OUTPUT_DIR, plot_first_sample_density=True)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    in_feat_dim = dataset[0].x.size(1)
    gat_model = RAdaptGAT(in_channels=in_feat_dim, hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS, heads=HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(gat_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    print(f"\nTraining GAT model on {DEVICE}...")
    if classical_r_adapt_times_all: print(f"Average classical r-adaptivity time: {np.mean(classical_r_adapt_times_all):.4f}s")

    epochs_list, train_losses_history, val_losses_history, gat_epoch_times_train = [], [], [], []
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        gat_model.train()
        total_train_loss = sum(loss_fn(gat_model(batch.x, batch.edge_index), batch.y).item() * batch.num_graphs for batch in train_loader for _ in [optimizer.zero_grad(), loss_fn(gat_model(batch.x, batch.edge_index), batch.y).backward(), optimizer.step()])
        train_losses_history.append(total_train_loss / len(train_loader.dataset))
        gat_epoch_times_train.append(time.time() - epoch_start_time)

        gat_model.eval()
        with torch.inference_mode():
            total_val_loss = sum(loss_fn(gat_model(batch.x, batch.edge_index), batch.y).item() * batch.num_graphs for batch in val_loader)
        val_losses_history.append(total_val_loss / len(val_loader.dataset))
        epochs_list.append(epoch + 1)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses_history[-1]:.6f} | Val Loss: {val_losses_history[-1]:.6f} | Time: {gat_epoch_times_train[-1]:.2f}s")

    print("Training complete.")
    if gat_epoch_times_train: print(f"Avg GAT training epoch time: {np.mean(gat_epoch_times_train):.4f}s")

    model_save_path = os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_epoch{EPOCHS}.pt")
    torch.save(gat_model.state_dict(), model_save_path)
    print(f"Saved trained model state_dict to: {model_save_path}")

    plot_funcs.loss_plot(epochs_list, train_losses_history, val_losses_history, MODEL_NAME, SESSION_OUTPUT_DIR, False)
    print(f"Saved training loss plot to {SESSION_OUTPUT_DIR}/")

    if val_loader:
        val_batch_sample = next(iter(val_loader)).to(DEVICE)
        gat_model.eval()
        with torch.inference_mode():
            val_pred_np = gat_model(val_batch_sample.x, val_batch_sample.edge_index).cpu().numpy()
        val_true_np = val_batch_sample.y.cpu().numpy()
        plot_funcs.predVStrue([val_true_np.flatten()], [val_pred_np.flatten()], [], [], f"{MODEL_NAME}_ValBatch", SESSION_OUTPUT_DIR, False)
        print(f"Saved Pred vs True plot for val batch to {SESSION_OUTPUT_DIR}/")

    # --- Time Benchmark over Validation Set ---
    print("\n--- Time Benchmark over Validation Set ---")
    val_classical_times_list = []
    val_gat_inference_times_list = []
    num_inference_runs_per_sample = 5  # How many times to run GAT inference per sample for averaging

    if val_dataset: # Proceed only if there is a validation dataset
        gat_model.eval() # Ensure model is in evaluation mode
        
        for data_sample_idx, data_sample in enumerate(val_dataset):
            val_classical_times_list.append(data_sample.classical_time)

            current_x = data_sample.x.to(DEVICE)
            current_edge_index = data_sample.edge_index.to(DEVICE)
            
            sample_gat_times_for_avg = []
            for _ in range(num_inference_runs_per_sample):
                with torch.inference_mode():
                    start_time = time.time()
                    _ = gat_model(current_x, current_edge_index)
                    end_time = time.time()
                    sample_gat_times_for_avg.append(end_time - start_time)
            
            if sample_gat_times_for_avg:
                val_gat_inference_times_list.append(np.mean(sample_gat_times_for_avg))
            
            if (data_sample_idx + 1) % (len(val_dataset) // 5 or 1) == 0 :
                 print(f"  Benchmarked {data_sample_idx + 1}/{len(val_dataset)} validation samples for time.")

        if val_classical_times_list and val_gat_inference_times_list:
            print("\n--- Time Benchmark Results (Validation Set) ---")
            
            # Calculate statistics
            classical_mean = np.mean(val_classical_times_list)
            classical_median = np.median(val_classical_times_list)
            classical_std = np.std(val_classical_times_list)
            classical_min = np.min(val_classical_times_list)
            classical_max = np.max(val_classical_times_list)

            gat_mean = np.mean(val_gat_inference_times_list)
            gat_median = np.median(val_gat_inference_times_list)
            gat_std = np.std(val_gat_inference_times_list)
            gat_min = np.min(val_gat_inference_times_list)
            gat_max = np.max(val_gat_inference_times_list)

            print(f"Classical R-Adaptivity ({len(val_classical_times_list)} samples):")
            print(f"  Mean: {classical_mean:.6f}s, Median: {classical_median:.6f}s, Std: {classical_std:.6f}s")
            print(f"  Min:  {classical_min:.6f}s, Max: {classical_max:.6f}s")
            
            print(f"GAT Inference ({len(val_gat_inference_times_list)} samples, avg over {num_inference_runs_per_sample} runs each):")
            print(f"  Mean: {gat_mean:.6f}s, Median: {gat_median:.6f}s, Std: {gat_std:.6f}s")
            print(f"  Min:  {gat_min:.6f}s, Max: {gat_max:.6f}s")

            # --- Prepare data for JSON ---
            benchmark_summary = {
                "model_name": MODEL_NAME,
                "session_timestamp": session_timestamp, # Assumes session_timestamp is defined earlier
                "device": str(DEVICE),
                "parameters": {
                    "num_total_samples": NUM_SAMPLES,
                    "num_validation_samples_benchmarked": len(val_dataset),
                    "epochs_trained": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "gat_input_features": in_feat_dim, # Assumes in_feat_dim is defined
                    # You can add more GAT parameters here if needed
                },
                "classical_r_adaptivity_times_seconds": {
                    "all_values_on_val_set": val_classical_times_list,
                    "mean": classical_mean,
                    "median": classical_median,
                    "std_dev": classical_std,
                    "min": classical_min,
                    "max": classical_max
                },
                "gat_inference_times_seconds": {
                    "all_values_on_val_set": val_gat_inference_times_list, # These are already averages per sample
                    "runs_per_sample_for_avg": num_inference_runs_per_sample,
                    "mean": gat_mean,
                    "median": gat_median,
                    "std_dev": gat_std,
                    "min": gat_min,
                    "max": gat_max
                }
            }

            # --- Save benchmark summary to JSON ---
            benchmark_file_path = os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_time_benchmark_summary.json")
            try:
                with open(benchmark_file_path, 'w') as f:
                    json.dump(benchmark_summary, f, indent=4)
                print(f"Saved time benchmark summary to: {benchmark_file_path}")
            except Exception as e:
                print(f"Error saving benchmark summary to JSON: {e}")

            # --- Plotting (as before) ---
            plot_funcs.plot_time_comparison(
                classical_times=[classical_mean], # Bar chart of means
                gat_times=[gat_mean],
                title=f"{MODEL_NAME} Avg. R-Adaptivity Time (Val Set)",
                time_label='Average Time per Mesh (s)',
                output=SESSION_OUTPUT_DIR,
                use_box_plot=False,
                show=False
            )
            print(f"Saved average time comparison bar plot to {SESSION_OUTPUT_DIR}/")

            plot_funcs.plot_time_comparison(
                classical_times=val_classical_times_list, # Box plot of all validation times
                gat_times=val_gat_inference_times_list,
                title=f"{MODEL_NAME} R-Adaptivity Time Distribution (Val Set)",
                time_label='Time per Mesh (s)',
                output=SESSION_OUTPUT_DIR,
                use_box_plot=True,
                show=False
            )
            print(f"Saved time comparison box plot to {SESSION_OUTPUT_DIR}/")
        else:
            print("Not enough data to generate or save time benchmark for validation set.")
    else:
        print("Validation dataset is empty. Skipping time benchmark.")


    print("\n--- Example Inference & Time Plot ---")
    example_initial_mesh = dolfin.UnitSquareMesh(8, 8)
    monitor_vals_test = get_monitor_function(example_initial_mesh.coordinates())
    classical_start_time = time.time()
    example_optimized_classical_coords = dummy_classical_r_adaptivity(example_initial_mesh, monitor_vals_test)
    classical_duration_test = time.time() - classical_start_time
    print(f"Classical r-adaptivity time for test mesh: {classical_duration_test:.6f}s")

    pyg_test_data = fenics_mesh_to_pyg_data(example_initial_mesh, device=DEVICE)
    gat_model.eval()
    gat_inference_times, predicted_optimized_coords_gat_np = [], None
    for i in range(10):
        with torch.inference_mode():
            gat_start_time = time.time()
            current_preds = gat_model(pyg_test_data.x, pyg_test_data.edge_index)
            gat_inference_times.append(time.time() - gat_start_time)
            if i == 9: predicted_optimized_coords_gat_np = current_preds.cpu().numpy()
    if predicted_optimized_coords_gat_np is None and gat_inference_times:
         with torch.inference_mode(): predicted_optimized_coords_gat_np = gat_model(pyg_test_data.x, pyg_test_data.edge_index).cpu().numpy()
    avg_gat_duration_test = np.mean(gat_inference_times) if gat_inference_times else 0
    print(f"GAT inference time (avg over {len(gat_inference_times)} runs): {avg_gat_duration_test:.6f}s")

    plot_funcs.plot_time_comparison([classical_duration_test], [avg_gat_duration_test], title=f"{MODEL_NAME} Inference Time", output=SESSION_OUTPUT_DIR, show=False)
    print(f"Saved inference time comparison plot to {SESSION_OUTPUT_DIR}/")

    print("\n--- Placeholder for Accuracy/Convergence Plots ---")
    # (Placeholder comments remain)

    # --- Final Visualization of mesh adaptation (Using plt.sca() method) ---
    if predicted_optimized_coords_gat_np is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        plt.sca(axs[0]) # Set current axis
        dolfin.plot(example_initial_mesh)
        axs[0].set_title("Initial Mesh")

        adapted_mesh_classical = dolfin.Mesh(example_initial_mesh)
        adapted_mesh_classical.coordinates()[:] = example_optimized_classical_coords
        plt.sca(axs[1]) # Set current axis
        dolfin.plot(adapted_mesh_classical)
        axs[1].set_title("Classical R-Adapted (Dummy)")

        adapted_mesh_gat = dolfin.Mesh(example_initial_mesh)
        adapted_mesh_gat.coordinates()[:] = predicted_optimized_coords_gat_np
        plt.sca(axs[2]) # Set current axis
        dolfin.plot(adapted_mesh_gat)
        axs[2].set_title("GAT R-Adapted")

        plt.tight_layout()
        plt.savefig(os.path.join(SESSION_OUTPUT_DIR, f"{MODEL_NAME}_mesh_adaptation_comparison.png"))
        print(f"\nSaved final mesh comparison plot to {SESSION_OUTPUT_DIR}/")
    else:
        print("\nSkipping final mesh comparison: GAT predictions not available.")
    
    # plt.show() # Uncomment to show plots interactively at the end.

    print(f"\nAll generated plots and model saved to directory: {SESSION_OUTPUT_DIR}")
