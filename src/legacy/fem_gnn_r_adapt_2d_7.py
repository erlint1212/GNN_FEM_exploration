# fem_gnn_r_adapt_2d.py (Main Script)
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
import sys

# Project-specific imports
import config
from utils import Tee
from data_generation import generate_dataset
from fem_utils import solve_fem_problem, calculate_l2_error, get_solution_based_monitor_function
from mesh_utils import dummy_classical_r_adaptivity, check_mesh_quality, get_boundary_nodes
from mesh_generators_2 import create_square_mesh, create_pipe_with_obstacle_mesh_gmsh
from fenics_mesh_to_pyg_data import fenics_mesh_to_pyg_data
from models.GAT import RAdaptGAT
import plot_funcs


# --- Main Script Execution ---
if __name__ == '__main__':
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_OUTPUT_DIR = os.path.join(config.BASE_OUTPUT_DIR, f"session_{session_timestamp}")
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)
    
    log_filepath = os.path.join(SESSION_OUTPUT_DIR, f"{config.MODEL_NAME}_session_{session_timestamp}.log")
    
    with Tee(log_filepath, mode='w'):
        dolfin.set_log_level(dolfin.LogLevel.WARNING)
        print(f"--- Starting FEM GNN R-Adaptivity Script ---")
        print(f"Timestamp: {session_timestamp}")
        print(f"Using MESH_TYPE: '{config.MESH_TYPE}'")
        print(f"Model Name: '{config.MODEL_NAME}'")
        print(f"Output for this session will be in: {SESSION_OUTPUT_DIR}")
        
        plot_funcs.cuda_status(config.DEVICE)
        
        dataset, _, _, _ = generate_dataset(config.NUM_SAMPLES, SESSION_OUTPUT_DIR, plot_first_sample_details=True)
        
        if not dataset:
            print("Dataset is empty. Exiting.")
            sys.exit(1)

        train_size = int(0.8 * len(dataset))
        if len(dataset) > 0 and train_size == 0: train_size = 1
        
        if train_size >= len(dataset):
            train_dataset, val_dataset = (dataset, list(dataset))
        else:
            train_dataset, val_dataset = (dataset[:train_size], dataset[train_size:])
        print(f"Dataset size: Total={len(dataset)}, Train={len(train_dataset)}, Val={len(val_dataset)}")

        if not train_dataset: print("Error: Training dataset empty. Exiting."); sys.exit(1)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False) if val_dataset else None

        in_feat_dim = dataset[0].x.size(1) if dataset and hasattr(dataset[0], 'x') and dataset[0].x is not None else 0
        if in_feat_dim == 0:
            print("Error: Cannot determine input feature dimension. Exiting.")
            sys.exit(1)
        print(f"Determined input feature dimension: {in_feat_dim}")

        gat_model = RAdaptGAT(in_channels=in_feat_dim, 
                              hidden_channels=config.HIDDEN_CHANNELS, 
                              out_channels=config.OUT_CHANNELS, 
                              heads=config.HEADS, 
                              num_layers=config.NUM_LAYERS, 
                              dropout=config.DROPOUT).to(config.DEVICE)
        optimizer = optim.Adam(gat_model.parameters(), lr=config.LEARNING_RATE)
        loss_fn = torch.nn.MSELoss()

        print(f"\n--- Training {config.MODEL_NAME} on {config.DEVICE} ---")
        
        epochs_list, train_losses_history, val_losses_history, gat_epoch_times_train = [], [], [], []
        train_true_coords_history, train_pred_coords_history = [], []

        for epoch in range(config.EPOCHS):
            epoch_start_time = time.time(); gat_model.train()
            current_epoch_train_loss, num_train_batches = 0,0
            for batch_idx, batch_data in enumerate(train_loader):
                batch_data = batch_data.to(config.DEVICE)
                optimizer.zero_grad()
                out = gat_model(batch_data.x, batch_data.edge_index)
                loss = loss_fn(out, batch_data.y); loss.backward(); optimizer.step()
                current_epoch_train_loss += loss.item(); num_train_batches +=1
                if epoch == config.EPOCHS - 1:
                    train_true_coords_history.append(batch_data.y.cpu().numpy())
                    train_pred_coords_history.append(out.detach().cpu().numpy())

            avg_epoch_train_loss = current_epoch_train_loss / num_train_batches if num_train_batches > 0 else float('nan')
            train_losses_history.append(avg_epoch_train_loss)
            gat_epoch_times_train.append(time.time() - epoch_start_time)
            
            current_epoch_val_loss, num_val_batches = 0,0
            if val_loader:
                gat_model.eval()
                with torch.inference_mode():
                    for batch_data_val in val_loader:
                        batch_data_val = batch_data_val.to(config.DEVICE)
                        out = gat_model(batch_data_val.x, batch_data_val.edge_index)
                        loss = loss_fn(out, batch_data_val.y)
                        current_epoch_val_loss += loss.item(); num_val_batches +=1
                avg_epoch_val_loss = current_epoch_val_loss / num_val_batches if num_val_batches > 0 else float('nan')
                val_losses_history.append(avg_epoch_val_loss)
            else: val_losses_history.append(float('nan'))
            epochs_list.append(epoch+1)
            print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_losses_history[-1]:.6f} | Val Loss: {val_losses_history[-1]:.6f} | Time: {gat_epoch_times_train[-1]:.2f}s")
        
        print("--- Training complete ---")
        model_save_path = os.path.join(SESSION_OUTPUT_DIR, f"{config.MODEL_NAME}_epoch{config.EPOCHS}.pt")
        torch.save(gat_model.state_dict(), model_save_path)
        print(f"Saved trained model state_dict to: {model_save_path}")
        plot_funcs.loss_plot(epochs_list, train_losses_history, val_losses_history, config.MODEL_NAME, SESSION_OUTPUT_DIR, show=False)

        val_classical_l2_errors, val_gat_l2_errors, val_initial_l2_errors = [], [], []
        val_dofs_list = []; val_classical_times_list_bench = []; val_gat_inference_times_list_bench = []
        val_true_coords_list, val_pred_coords_list = [], []
        last_sample_data_for_plot = {}

        if val_dataset:
            print(f"\n--- Evaluation on Validation Set ({len(val_dataset)} samples) ---")
            
            for i, data_sample in enumerate(val_dataset):
                print(f"  Processing validation sample {i+1}/{len(val_dataset)}...")
                try: current_mesh_dims = eval(data_sample.mesh_dimensions_str)
                except Exception as e: print(f"    Error eval mesh_dims: {e}. Skip."); continue
                current_mesh_type = data_sample.mesh_type_str
                
                initial_mesh_val = None
                if current_mesh_type == 'square':
                    nx_val, ny_val = current_mesh_dims.get('nx'), current_mesh_dims.get('ny')
                    initial_mesh_val, _ = create_square_mesh(nx_val, ny_val)
                elif current_mesh_type == 'pipe':
                    ms_factor_val = current_mesh_dims.get('mesh_size_factor', 0.15)
                    initial_mesh_val, _ = create_pipe_with_obstacle_mesh_gmsh(mesh_size_factor=ms_factor_val)
                
                if initial_mesh_val is None or initial_mesh_val.num_cells() == 0: continue

                original_coords_val_np = np.array(eval(data_sample.original_coords_str))
                boundary_node_indices_val = get_boundary_nodes(initial_mesh_val)

                u_initial_val = solve_fem_problem(initial_mesh_val, current_mesh_type, current_mesh_dims)
                if u_initial_val is None: continue
                
                l2_initial = calculate_l2_error(u_initial_val, current_mesh_type, current_mesh_dims, initial_mesh_val)
                val_initial_l2_errors.append(l2_initial)
                val_dofs_list.append(u_initial_val.function_space().dim())
                
                monitor_val_np = get_solution_based_monitor_function(u_initial_val, initial_mesh_val)
                
                classical_start_val = time.time()
                opt_coords_classical_val = dummy_classical_r_adaptivity(initial_mesh_val, monitor_val_np, mesh_dimensions=current_mesh_dims)
                val_classical_times_list_bench.append(time.time() - classical_start_val)
                classical_adapted_mesh_val = dolfin.Mesh(initial_mesh_val)
                l2_classical = -4.0
                if opt_coords_classical_val.shape[0] == classical_adapted_mesh_val.num_vertices():
                    classical_adapted_mesh_val.coordinates()[:] = opt_coords_classical_val
                    if check_mesh_quality(classical_adapted_mesh_val, "Classical Val")[0]:
                        u_class_val = solve_fem_problem(classical_adapted_mesh_val, current_mesh_type, current_mesh_dims)
                        l2_classical = calculate_l2_error(u_class_val, current_mesh_type, current_mesh_dims, classical_adapted_mesh_val) if u_class_val else -2.0
                    else: l2_classical = -3.0
                val_classical_l2_errors.append(l2_classical)

                pyg_val_sample = fenics_mesh_to_pyg_data(initial_mesh_val, device=config.DEVICE, additional_features=(monitor_val_np.reshape(-1,1) if config.USE_MONITOR_AS_FEATURE else None))
                gat_adapted_mesh_val, l2_gat = None, -5.0
                if pyg_val_sample.num_nodes > 0 and pyg_val_sample.x is not None:
                    gat_model.eval()
                    with torch.inference_mode():
                        predicted_displacement = gat_model(pyg_val_sample.x, pyg_val_sample.edge_index)
                        initial_coords_tensor = torch.tensor(original_coords_val_np, dtype=torch.float, device=config.DEVICE)
                        gat_adapted_coords_tensor = initial_coords_tensor + predicted_displacement
                        gat_adapted_coords_val_np = gat_adapted_coords_tensor.cpu().numpy()

                        val_true_coords_list.append(data_sample.y.cpu().numpy())
                        val_pred_coords_list.append(predicted_displacement.cpu().numpy())
                        
                        gat_adapted_coords_val_np[:, 0] = np.clip(gat_adapted_coords_val_np[:, 0], 0.0, current_mesh_dims.get("width",1.0))
                        gat_adapted_coords_val_np[:, 1] = np.clip(gat_adapted_coords_val_np[:, 1], 0.0, current_mesh_dims.get("height",1.0))
                        for bn_idx in boundary_node_indices_val:
                            if bn_idx < len(gat_adapted_coords_val_np): gat_adapted_coords_val_np[bn_idx] = original_coords_val_np[bn_idx]
                        
                        gat_adapted_mesh_val = dolfin.Mesh(initial_mesh_val)
                        if gat_adapted_coords_val_np.shape[0] == gat_adapted_mesh_val.num_vertices():
                            gat_adapted_mesh_val.coordinates()[:] = gat_adapted_coords_val_np
                            if check_mesh_quality(gat_adapted_mesh_val, "GAT Val")[0]:
                                u_gat_val = solve_fem_problem(gat_adapted_mesh_val, current_mesh_type, current_mesh_dims)
                                l2_gat = calculate_l2_error(u_gat_val, current_mesh_type, current_mesh_dims, gat_adapted_mesh_val) if u_gat_val else -2.0
                            else: l2_gat = -3.0
                        else: l2_gat = -4.0
                val_gat_l2_errors.append(l2_gat)
                
                last_sample_data_for_plot = {
                    "initial_mesh": initial_mesh_val, "l2_initial": l2_initial,
                    "classical_mesh": classical_adapted_mesh_val, "l2_classical": l2_classical,
                    "gat_mesh": gat_adapted_mesh_val, "l2_gat": l2_gat,
                }
            
            # --- Generate and Save Performance Plots ---
            print("\n--- Generating Performance Plots ---")
            plot_funcs.predVStrue(label_val_true=val_true_coords_list, label_val_pred=val_pred_coords_list,
                                  label_train_true=train_true_coords_history, label_train_pred=train_pred_coords_history,
                                  model_name=config.MODEL_NAME, output=SESSION_OUTPUT_DIR, show=False)
            
            plot_funcs.plot_time_comparison(classical_times=val_classical_times_list_bench, gat_times=val_gat_inference_times_list_bench,
                                            title=f"Mesh Adapt Time ({config.MODEL_NAME} vs Classical)",
                                            use_box_plot=True, output=SESSION_OUTPUT_DIR, show=False)
            
            valid_indices = [i for i, err in enumerate(val_gat_l2_errors) if err >= 0 and i < len(val_classical_l2_errors) and val_classical_l2_errors[i] >= 0]
            
            if valid_indices:
                plot_dofs = [val_dofs_list[i] for i in valid_indices]
                plot_classical_errors = [val_classical_l2_errors[i] for i in valid_indices]
                plot_gat_errors = [val_gat_l2_errors[i] for i in valid_indices]
                plot_funcs.plot_accuracy_vs_cost(classical_costs=plot_dofs, classical_accuracies=plot_classical_errors,
                                                gat_costs=plot_dofs, gat_accuracies=plot_gat_errors,
                                                title=f"Accuracy vs. Cost ({config.MODEL_NAME})",
                                                output=SESSION_OUTPUT_DIR, show=False)
                plot_funcs.plot_convergence(classical_dofs=plot_dofs, classical_errors=plot_classical_errors,
                                            gat_dofs=plot_dofs, gat_errors=plot_gat_errors,
                                            title=f"Convergence Plot ({config.MODEL_NAME})",
                                            output=SESSION_OUTPUT_DIR, show=False)
            else:
                print("Warning: Not enough valid data points to generate accuracy/convergence plots.")

            # --- Generate and Save Benchmark Summary JSON ---
            print("\n--- Generating Benchmark Summary JSON ---")
            summary_data = {
                "model_name": config.MODEL_NAME,
                "session_timestamp": session_timestamp,
                "device": str(config.DEVICE),
                "parameters": {
                    "epochs": config.EPOCHS, "learning_rate": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE,
                    "num_layers": config.NUM_LAYERS, "hidden_channels": config.HIDDEN_CHANNELS, "heads": config.HEADS,
                    "dropout": config.DROPOUT, "mesh_type": config.MESH_TYPE, "num_total_samples": len(dataset),
                    "num_validation_samples_benchmarked": len(val_dataset) if val_dataset else 0,
                    "inference_runs_per_sample": config.NUM_INFERENCE_RUNS_PER_SAMPLE_FOR_TIMING,
                },
                "classical_r_adaptivity_times_seconds": {
                    "mean": np.mean(val_classical_times_list_bench).item() if val_classical_times_list_bench else None,
                    "median": np.median(val_classical_times_list_bench).item() if val_classical_times_list_bench else None,
                    "std_dev": np.std(val_classical_times_list_bench).item() if val_classical_times_list_bench else None,
                    "min": np.min(val_classical_times_list_bench).item() if val_classical_times_list_bench else None,
                    "max": np.max(val_classical_times_list_bench).item() if val_classical_times_list_bench else None,
                    "count": len(val_classical_times_list_bench),
                },
                "gat_inference_times_seconds": {
                    "mean": np.mean(val_gat_inference_times_list_bench).item() if val_gat_inference_times_list_bench else None,
                    "median": np.median(val_gat_inference_times_list_bench).item() if val_gat_inference_times_list_bench else None,
                    "std_dev": np.std(val_gat_inference_times_list_bench).item() if val_gat_inference_times_list_bench else None,
                    "min": np.min(val_gat_inference_times_list_bench).item() if val_gat_inference_times_list_bench else None,
                    "max": np.max(val_gat_inference_times_list_bench).item() if val_gat_inference_times_list_bench else None,
                    "count": len(val_gat_inference_times_list_bench),
                },
                "l2_error_stats": {
                    "initial_mean": np.mean([e for e in val_initial_l2_errors if e >= 0]).item() if any(e >= 0 for e in val_initial_l2_errors) else None,
                    "classical_adapted_mean": np.mean([e for e in val_classical_l2_errors if e >= 0]).item() if any(e >= 0 for e in val_classical_l2_errors) else None,
                    "gat_adapted_mean": np.mean([e for e in val_gat_l2_errors if e >= 0]).item() if any(e >= 0 for e in val_gat_l2_errors) else None,
                }
            }
            json_save_path = os.path.join(SESSION_OUTPUT_DIR, f"{config.MODEL_NAME}_benchmark_summary.json")
            try:
                with open(json_save_path, 'w') as f: json.dump(summary_data, f, indent=4)
                print(f"Saved benchmark summary to: {json_save_path}")
                print(f"\nTo generate a LaTeX table, run:\npython json_to_latex_2.py {json_save_path}")
            except Exception as e: print(f"Error saving benchmark summary JSON: {e}")

        # --- Final Visualizations for an Example Sample ---
        print(f"\n--- Generating Final Visualizations for an Example Sample ---")
        if last_sample_data_for_plot and last_sample_data_for_plot.get("gat_mesh"):
            initial_mesh = last_sample_data_for_plot["initial_mesh"]
            classical_mesh = last_sample_data_for_plot["classical_mesh"]
            gat_mesh = last_sample_data_for_plot["gat_mesh"]
            l2_i, l2_c, l2_g = last_sample_data_for_plot["l2_initial"], last_sample_data_for_plot["l2_classical"], last_sample_data_for_plot["l2_gat"]
            
            print("  Generating final mesh adaptation comparison plot...")
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            plt.sca(axes[0]); dolfin.plot(initial_mesh); axes[0].set_title(f"Initial Mesh\nL2 Err: {l2_i:.2e}" if l2_i >= 0 else "Initial Mesh")
            plt.sca(axes[1]); dolfin.plot(classical_mesh); axes[1].set_title(f"Classical Adapted Mesh\nL2 Err: {l2_c:.2e}" if l2_c >= 0 else "Classical Adapted (Solve Failed)")
            plt.sca(axes[2]); dolfin.plot(gat_mesh); axes[2].set_title(f"GAT Adapted Mesh\nL2 Err: {l2_g:.2e}" if l2_g >= 0 else "GAT Adapted (Solve Failed)")
            for ax in axes:
                if config.MESH_TYPE == 'pipe': ax.set_aspect('equal')
            plt.tight_layout()
            plot_filename = os.path.join(SESSION_OUTPUT_DIR, f"{config.MODEL_NAME}_final_mesh_comparison.png")
            plt.savefig(plot_filename); plt.close(fig)
            print(f"    Saved final comparison plot to {plot_filename}")

            print("  Generating final coordinate density plots...")
            plot_funcs.density_plot_matrix(initial_mesh.coordinates(), output=SESSION_OUTPUT_DIR, title=f"Example Initial Node Coords ({config.MODEL_NAME})", show=False)
            plot_funcs.density_plot_matrix(classical_mesh.coordinates(), output=SESSION_OUTPUT_DIR, title=f"Example Classical-Adapted Node Coords ({config.MODEL_NAME})", show=False)
            plot_funcs.density_plot_matrix(gat_mesh.coordinates(), output=SESSION_OUTPUT_DIR, title=f"Example GAT-Adapted Node Coords ({config.MODEL_NAME})", show=False)
        else:
            print("  Skipping final example visualizations as last validation sample was not fully processed.")

        print(f"\nAll outputs saved to: {SESSION_OUTPUT_DIR}")
        print(f"--- FEM GNN R-Adaptivity Script Finished ---")
