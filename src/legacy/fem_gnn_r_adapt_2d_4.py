# fem_gnn_r_adapt_2d.py (Main Script)
import dolfin
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt # Keep for direct plotting if any, though most moved to plot_funcs
import time
import os
import datetime
import json
import sys # For Tee

# Project-specific imports
import config # Import all config variables
from utils import Tee # Import the Tee logger
from data_generation import generate_dataset
from fem_utils import solve_fem_problem, calculate_l2_error, get_solution_based_monitor_function
from mesh_utils import dummy_classical_r_adaptivity, check_mesh_quality, get_boundary_nodes
from mesh_generators_2 import create_square_mesh, create_pipe_with_obstacle_mesh_gmsh # Needed for regen in val/test
from fenics_mesh_to_pyg_data import fenics_mesh_to_pyg_data
from models.GAT import RAdaptGAT
import plot_funcs # plot_funcs will now handle its own imports like matplotlib


# --- Main Script Execution ---
if __name__ == '__main__':
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Construct SESSION_OUTPUT_DIR using config.BASE_OUTPUT_DIR
    SESSION_OUTPUT_DIR = os.path.join(config.BASE_OUTPUT_DIR, f"session_{session_timestamp}")
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)
    
    log_filepath = os.path.join(SESSION_OUTPUT_DIR, f"{config.MODEL_NAME}_session_{session_timestamp}.log")
    
    with Tee(log_filepath, mode='w'): # Start logging
        dolfin.set_log_level(dolfin.LogLevel.WARNING)
        print(f"--- Starting FEM GNN R-Adaptivity Script ---")
        print(f"Timestamp: {session_timestamp}")
        print(f"Using MESH_TYPE: '{config.MESH_TYPE}'")
        print(f"Model Name: '{config.MODEL_NAME}'")
        print(f"Output for this session (plots, model, log) will be in: {SESSION_OUTPUT_DIR}")
        
        plot_funcs.cuda_status(config.DEVICE)
        
        dataset, classical_r_adapt_times_all, l2_errors_initial_all, l2_errors_classical_adapted_all = \
            generate_dataset(config.NUM_SAMPLES, SESSION_OUTPUT_DIR, plot_first_sample_details=True)
        
        if not dataset:
            print("Dataset is empty after generation. Exiting.")
            sys.exit(1)

        # Dataset Split
        train_size = int(0.8 * len(dataset))
        if len(dataset) > 0 and train_size == 0: train_size = 1
        
        if train_size >= len(dataset):
            print("Warning: Dataset too small for a separate validation set. Using all data for training and validation.")
            train_dataset = dataset
            val_dataset = list(dataset)
        else:
            train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]
        print(f"Dataset size: Total={len(dataset)}, Train={len(train_dataset)}, Val={len(val_dataset)}")

        if not train_dataset: print("Error: Training dataset empty. Exiting."); sys.exit(1)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False) if val_dataset else None

        if dataset and hasattr(dataset[0], 'x') and dataset[0].x is not None:
            in_feat_dim = dataset[0].x.size(1)
            print(f"Determined input feature dimension: {in_feat_dim}")
        else:
            print("Error: Dataset is empty or first sample has no features 'x'. Exiting.")
            sys.exit(1)

        # Model, Optimizer, Loss
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
        if gat_epoch_times_train: print(f"Avg GAT training epoch time: {np.mean(gat_epoch_times_train):.4f}s")
        model_save_path = os.path.join(SESSION_OUTPUT_DIR, f"{config.MODEL_NAME}_epoch{config.EPOCHS}.pt")
        torch.save(gat_model.state_dict(), model_save_path)
        print(f"Saved trained model state_dict to: {model_save_path}")
        plot_funcs.loss_plot(epochs_list, train_losses_history, val_losses_history, config.MODEL_NAME, SESSION_OUTPUT_DIR, show=False)

        # Validation Set Evaluation & Benchmarking
        val_classical_l2_errors, val_gat_l2_errors, val_initial_l2_errors = [], [], []
        val_dofs_list = []; val_classical_times_list_bench = []; val_gat_inference_times_list_bench = []
        val_true_coords_list, val_pred_coords_list = [], []

        if val_dataset:
            print(f"\n--- Evaluation on Validation Set ({len(val_dataset)} samples) ---")
            
            for i, data_sample in enumerate(val_dataset):
                print(f"  Processing validation sample {i+1}/{len(val_dataset)}...")
                try: current_mesh_dims = eval(data_sample.mesh_dimensions_str)
                except Exception as e_eval: print(f"    Error eval mesh_dims: '{data_sample.mesh_dimensions_str}' - {e_eval}. Skip."); continue
                current_mesh_type = data_sample.mesh_type_str
                initial_mesh_val = None; original_coords_val_np = None
                try: original_coords_val_np = np.array(eval(data_sample.original_coords_str))
                except: print("Could not eval original_coords_str"); original_coords_val_np = None
                
                if current_mesh_type == 'square':
                    nx_val, ny_val = current_mesh_dims.get('nx'), current_mesh_dims.get('ny')
                    if nx_val is None or ny_val is None: print(f"    nx/ny not in val sample {i+1}. Skip."); continue
                    initial_mesh_val, _ = create_square_mesh(nx_val, ny_val)
                elif current_mesh_type == 'pipe':
                    ms_factor_val = current_mesh_dims.get('mesh_size_factor', (config.MESH_SIZE_FACTOR_MIN + config.MESH_SIZE_FACTOR_MAX) / 2.0)
                    initial_mesh_val, _ = create_pipe_with_obstacle_mesh_gmsh(mesh_size_factor=ms_factor_val, pipe_length=config.PIPE_LENGTH, pipe_height=config.PIPE_HEIGHT, obstacle_cx_factor=config.OBSTACLE_CENTER_X_FACTOR, obstacle_cy_factor=config.OBSTACLE_CENTER_Y_FACTOR, obstacle_r_factor=config.OBSTACLE_RADIUS_FACTOR)

                if initial_mesh_val is None or initial_mesh_val.num_cells() == 0: print(f"    Could not regen val mesh {i+1}. Skip."); continue
                if original_coords_val_np is None or original_coords_val_np.shape[0] != initial_mesh_val.num_vertices():
                    original_coords_val_np = initial_mesh_val.coordinates()
                boundary_node_indices_val = get_boundary_nodes(initial_mesh_val)

                u_initial_val = solve_fem_problem(initial_mesh_val, current_mesh_type, current_mesh_dims)
                if u_initial_val is None: print(f"    FEM solve failed on initial val mesh {i+1}. Skip."); continue
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
                        val_classical_l2_errors.append(calculate_l2_error(u_class_val, current_mesh_type, current_mesh_dims, classical_adapted_mesh_val) if u_class_val else -2.0)
                    else: val_classical_l2_errors.append(-3.0)
                else: val_classical_l2_errors.append(-4.0)

                add_feat_val = monitor_val_np.reshape(-1,1) if config.USE_MONITOR_AS_FEATURE else None
                pyg_val_sample = fenics_mesh_to_pyg_data(initial_mesh_val, device=config.DEVICE, additional_features=add_feat_val)
                current_gat_inference_times = []; gat_adapted_coords_val_np = None
                if pyg_val_sample.num_nodes > 0 and pyg_val_sample.x is not None:
                    gat_model.eval();
                    with torch.inference_mode():
                        gat_adapted_coords_raw = gat_model(pyg_val_sample.x, pyg_val_sample.edge_index)
                        gat_adapted_coords_val_np = gat_adapted_coords_raw.cpu().numpy()
                        val_true_coords_list.append(data_sample.y.cpu().numpy())
                        val_pred_coords_list.append(gat_adapted_coords_val_np)
                        gat_adapted_coords_val_np[:, 0] = np.clip(gat_adapted_coords_val_np[:, 0], 0.0, current_mesh_dims.get("width",1.0))
                        gat_adapted_coords_val_np[:, 1] = np.clip(gat_adapted_coords_val_np[:, 1], 0.0, current_mesh_dims.get("height",1.0))
                        if original_coords_val_np is not None and boundary_node_indices_val:
                            for bn_idx in boundary_node_indices_val:
                                if bn_idx < gat_adapted_coords_val_np.shape[0] and bn_idx < original_coords_val_np.shape[0]: gat_adapted_coords_val_np[bn_idx] = original_coords_val_np[bn_idx]
                        for _ in range(config.NUM_INFERENCE_RUNS_PER_SAMPLE_FOR_TIMING):
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
                            val_gat_l2_errors.append(calculate_l2_error(u_gat_val, current_mesh_type, current_mesh_dims, gat_adapted_mesh_val) if u_gat_val else -2.0)
                        else: val_gat_l2_errors.append(-3.0)
                    else: val_gat_l2_errors.append(-4.0)
                else: val_gat_l2_errors.append(-5.0)
            
            # --- Generate and Save Additional Validation Plots ---
            print("\n--- Generating Additional Validation Plots ---")
            plot_funcs.predVStrue(label_val_true=val_true_coords_list, label_val_pred=val_pred_coords_list,
                                  label_train_true=train_true_coords_history, label_train_pred=train_pred_coords_history,
                                  model_name=config.MODEL_NAME, output=SESSION_OUTPUT_DIR, show=False)
            plot_funcs.plot_time_comparison(classical_times=val_classical_times_list_bench, gat_times=val_gat_inference_times_list_bench,
                                            title=f"Mesh Adapt Time ({config.MODEL_NAME} vs Classical)",
                                            use_box_plot=True, output=SESSION_OUTPUT_DIR, show=False)
            valid_indices = [i for i, err in enumerate(val_gat_l2_errors) if err >= 0 and val_classical_l2_errors[i] >= 0]
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
            else: print("Warning: Not enough valid data points for accuracy/convergence plots.")

            # --- Generate and Save Benchmark Summary JSON for LaTeX script ---
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
                },
                "classical_r_adaptivity_times_seconds": {
                    "mean": np.mean(val_classical_times_list_bench).item() if val_classical_times_list_bench else None,
                    "std_dev": np.std(val_classical_times_list_bench).item() if val_classical_times_list_bench else None,
                    "count": len(val_classical_times_list_bench),
                },
                "gat_inference_times_seconds": {
                    "mean": np.mean(val_gat_inference_times_list_bench).item() if val_gat_inference_times_list_bench else None,
                    "std_dev": np.std(val_gat_inference_times_list_bench).item() if val_gat_inference_times_list_bench else None,
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

        # Example Inference with density plots
        print(f"\n--- Example Inference & Plot ({config.MESH_TYPE} Geometry) ---")
        if 'initial_mesh_val' in locals() and 'opt_coords_classical_val' in locals() and 'gat_adapted_coords_val_np' in locals():
            print("Generating density plots for the last validation sample's coordinates...")
            plot_funcs.density_plot_matrix(initial_mesh_val.coordinates(), output=SESSION_OUTPUT_DIR,
                                           title=f"Example Initial Node Coords ({config.MODEL_NAME})", show=False)
            plot_funcs.density_plot_matrix(opt_coords_classical_val, output=SESSION_OUTPUT_DIR,
                                           title=f"Example Classical-Adapted Node Coords ({config.MODEL_NAME})", show=False)
            plot_funcs.density_plot_matrix(gat_adapted_coords_val_np, output=SESSION_OUTPUT_DIR,
                                           title=f"Example GAT-Adapted Node Coords ({config.MODEL_NAME})", show=False)
        else: print("Skipping example density plots as last validation sample was not fully processed.")

        print(f"\nAll outputs saved to: {SESSION_OUTPUT_DIR}")
        print(f"--- FEM GNN R-Adaptivity Script Finished ---")
