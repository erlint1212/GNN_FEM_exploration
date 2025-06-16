# data_generation.py
import dolfin
import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt # For plotting first sample

# Project-specific imports
from config import (MESH_TYPE, MESH_SIZE_MIN, MESH_SIZE_MAX,
                    MESH_SIZE_FACTOR_MIN, MESH_SIZE_FACTOR_MAX,
                    PIPE_LENGTH, PIPE_HEIGHT, OBSTACLE_CENTER_X_FACTOR,
                    OBSTACLE_CENTER_Y_FACTOR, OBSTACLE_RADIUS_FACTOR,
                    USE_MONITOR_AS_FEATURE, MODEL_NAME, DEVICE)

from mesh_generators_2 import create_square_mesh, create_pipe_with_obstacle_mesh_gmsh
from fenics_mesh_to_pyg_data import fenics_mesh_to_pyg_data
from fem_utils import solve_fem_problem, calculate_l2_error, get_solution_based_monitor_function
from mesh_utils import dummy_classical_r_adaptivity, check_mesh_quality

def generate_dataset(num_samples, session_output_dir, plot_first_sample_details=False):
    dataset = []
    all_classical_times = []
    all_initial_errors = []
    all_classical_adapted_errors = []

    print(f"Generating {num_samples} data samples for MESH_TYPE: '{MESH_TYPE}'...")
    
    generated_count = 0
    attempts = 0
    max_attempts = num_samples * 3 # Allow more attempts

    while generated_count < num_samples and attempts < max_attempts:
        attempts += 1
        print(f"\nAttempt {attempts} for sample {generated_count + 1}...")
        initial_mesh = None
        mesh_dims = None
        current_res_info = "" # For logging

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
            continue

        u_initial = solve_fem_problem(initial_mesh, MESH_TYPE, mesh_dims)
        if u_initial is None:
            print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - FEM solve failed on initial mesh. Skipping.")
            continue
        l2_error_initial = calculate_l2_error(u_initial, MESH_TYPE, mesh_dims, initial_mesh)
        if l2_error_initial <= -1.5:
             print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - Initial L2 error indicates problem ({l2_error_initial}). Skipping.")
             continue

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
                l2_error_optimized_classical = -3.0
            else:
                u_optimized_classical = solve_fem_problem(optimized_mesh_classical_viz, MESH_TYPE, mesh_dims)
                if u_optimized_classical is None:
                    l2_error_optimized_classical = -2.0
                else:
                    l2_error_optimized_classical = calculate_l2_error(u_optimized_classical, MESH_TYPE, mesh_dims, optimized_mesh_classical_viz)
        else:
            print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - Coord shape mismatch. Skipping FEM solve.")
            l2_error_optimized_classical = -4.0

        add_feat = None
        if USE_MONITOR_AS_FEATURE:
            add_feat = monitor_vals_np.reshape(-1, 1)

        pyg_data_sample = fenics_mesh_to_pyg_data(initial_mesh, device=DEVICE, additional_features=add_feat)
        if pyg_data_sample.num_nodes == 0:
             print(f"  Warning: Attempt {attempts}, Sample {generated_count+1} - empty PyG graph. Skipping.")
             continue

        # --- MODIFICATION: The target 'y' is now the displacement vector ---
        initial_coords_tensor = torch.tensor(initial_mesh.coordinates(), dtype=torch.float)
        optimized_coords_tensor = torch.tensor(optimized_coords_classical, dtype=torch.float)
        displacement = optimized_coords_tensor - initial_coords_tensor
        pyg_data_sample.y = displacement.to(DEVICE)
        # --- END MODIFICATION ---
            
        pyg_data_sample.classical_time = classical_duration
        pyg_data_sample.l2_error_initial = l2_error_initial
        pyg_data_sample.l2_error_classical_adapted = l2_error_optimized_classical
        pyg_data_sample.num_dofs = u_initial.function_space().dim() if u_initial else -1
        pyg_data_sample.mesh_type_str = MESH_TYPE 
        pyg_data_sample.mesh_dimensions_str = str(mesh_dims) 
        pyg_data_sample.original_coords_str = str(initial_mesh.coordinates().tolist())
            
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
            print(f"    Saved mesh plot to {plot_filename}")

    if generated_count < num_samples:
        print(f"Warning: Only generated {generated_count}/{num_samples} valid samples after {max_attempts} attempts.")
    
    return dataset, all_classical_times, all_initial_errors, all_classical_adapted_errors
