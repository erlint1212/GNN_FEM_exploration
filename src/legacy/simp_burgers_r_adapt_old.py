import dolfin
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader # DataLoader for batching
from torch_geometric.nn import GATv2Conv
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time # For timing

from models.GAT import RAdaptGAT
from fenics_mesh_to_pyg_data import *

# --- Parameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_SAMPLES = 100 # Number of mesh samples to generate for training
MESH_SIZE_MIN = 5
MESH_SIZE_MAX = 10 # Max N for NxN UnitSquareMesh
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 10

# --- Simplified "Burgers' Solution" and Monitor Function ---
def get_monitor_function(mesh_coords):
    """
    Generates a simple monitor function based on mesh coordinates.
    Nodes closer to a 'feature_center' will have higher monitor values.
    This simulates a solution feature that needs higher resolution.
    Args:
        mesh_coords (np.array): Array of node coordinates (num_nodes, 2)
    Returns:
        np.array: Monitor function values at each node (num_nodes,)
    """
    feature_center = np.array([0.75, 0.75]) # Arbitrary feature location
    distances_sq = np.sum((mesh_coords - feature_center)**2, axis=1)
    monitor_values = 1.0 / (distances_sq + 1e-3) # Higher value for smaller distance
    monitor_values = (monitor_values - np.min(monitor_values)) / (np.max(monitor_values) - np.min(monitor_values) + 1e-6) # Normalize
    return monitor_values

# --- Simplified "Classical R-Adaptivity" ---
def dummy_classical_r_adaptivity(mesh, monitor_values, strength=0.1):
    """
    Simulates r-adaptivity by moving nodes towards regions of high monitor values.
    This is a placeholder and not a physically accurate r-adaptivity method.
    Args:
        mesh (dolfin.Mesh): The initial FEniCS mesh.
        monitor_values (np.array): Monitor function values at each node.
        strength (float): How strongly nodes are pulled.
    Returns:
        np.array: New "optimized" node coordinates.
    """
    old_coords = np.copy(mesh.coordinates())
    new_coords = np.copy(old_coords)
    num_nodes = mesh.num_vertices()

    # For each node, calculate a weighted average of all other node positions,
    # weighted by their monitor function values (pull towards high monitor values)
    # This is a highly simplified "attraction"
    for i in range(num_nodes):
        if mesh.isałocal(i): # Only move local nodes if in parallel, but here all are local
            # Skip boundary nodes for simplicity in this dummy version
            # A real r-adaptivity needs proper boundary handling.
            on_boundary = False
            for facet in dolfin.facets(mesh):
                for vertex in dolfin.vertices(facet):
                    if vertex.index() == i and facet.exterior():
                        on_boundary = True
                        break
                if on_boundary:
                    break
            
            if on_boundary: # Keep boundary nodes fixed in this simple example
                continue

            direction_vector = np.zeros(2)
            total_weight = 0
            for j in range(num_nodes):
                if i == j:
                    continue
                # Vector from node i to node j
                diff = old_coords[j] - old_coords[i]
                # Weight by monitor value of node j (attract more to important nodes)
                # and inversely by distance (stronger pull from closer important nodes)
                dist_sq = np.sum(diff**2) + 1e-6
                weight = monitor_values[j] / dist_sq
                
                direction_vector += weight * diff
                total_weight += weight
            
            if total_weight > 1e-6:
                new_coords[i] += strength * (direction_vector / total_weight)

    # Ensure nodes don't move too far or out of bounds (simple clipping)
    new_coords = np.clip(new_coords, 0.0, 1.0)
    return new_coords


# --- Data Generation ---
def generate_dataset(num_samples):
    dataset = []
    print(f"Generating {num_samples} data samples...")
    for i in range(num_samples):
        nx = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
        ny = np.random.randint(MESH_SIZE_MIN, MESH_SIZE_MAX + 1)
        initial_mesh = dolfin.UnitSquareMesh(nx, ny)

        # 1. Get "solution" (represented by monitor function)
        monitor_vals = get_monitor_function(initial_mesh.coordinates())

        # 2. Apply "classical" r-adaptivity
        # Start timer for classical method
        classical_start_time = time.time()
        optimized_coords_classical = dummy_classical_r_adaptivity(initial_mesh, monitor_vals)
        classical_duration = time.time() - classical_start_time
        # print(f"Sample {i+1}, Classical R-Adapt time: {classical_duration:.4f}s")


        # 3. Convert initial mesh to PyG data
        pyg_data_sample = fenics_mesh_to_pyg_data(initial_mesh, device=DEVICE)
        pyg_data_sample.y = torch.tensor(optimized_coords_classical, dtype=torch.float).to(DEVICE)
        pyg_data_sample.classical_time = classical_duration # Store for later comparison

        # Add monitor function as a node feature (optional, could help GAT)
        # pyg_data_sample.x = torch.cat(
        #     [pyg_data_sample.x, torch.tensor(monitor_vals, dtype=torch.float).unsqueeze(1).to(DEVICE)],
        #     dim=1
        # )

        dataset.append(pyg_data_sample)
        if (i + 1) % (num_samples // 10) == 0:
            print(f"  Generated {i+1}/{num_samples} samples.")
    return dataset

# --- Main Script ---
if __name__ == '__main__':
    # 1. Generate Data
    dataset = generate_dataset(NUM_SAMPLES)
    
    # Split data (simple split for demonstration)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize GAT Model
    # Input features: 2 (x,y coordinates) + 0 or 1 (if monitor function is added)
    # Output features: 2 (new x,y coordinates)
    in_feat_dim = dataset[0].x.size(1)
    gat_model = RAdaptGAT(in_channels=in_feat_dim,
                          hidden_channels=64,
                          out_channels=2, # Predicting 2D coordinates
                          heads=4,
                          num_layers=3,
                          dropout=0.5).to(DEVICE)

    optimizer = optim.Adam(gat_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    print(f"\nTraining GAT model on {DEVICE}...")
    classical_times_total = [data.classical_time for data in train_dataset] + [data.classical_time for data in val_dataset]
    print(f"Average classical r-adaptivity time: {np.mean(classical_times_total):.4f}s")

    # 3. Training Loop
    gat_epoch_times = []
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # Set the model to training mode
        gat_model.train()

        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE) # Ensure batch is on the correct device
            # Ensure x is used as input features, not pos if they differ by features
            # 3.1 Forward pass
            predicted_positions = gat_model(batch.x, batch.edge_index)
            # 3.2 Calculate the loss
            loss = loss_fn(predicted_positions, batch.y)
            # 3.3 Optimizer zero grad
            optimizer.zero_grad()
            # 3.4 Perform backpropegation
            loss.backward()
            # 3.5 Step the optimizer (perform gradient decent)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        epoch_duration = time.time() - epoch_start_time
        gat_epoch_times.append(epoch_duration)

        # Validation
        gat_model.eval()
        val_loss = 0
        with torch.inference_mode(): 
            for batch in val_loader:
                batch = batch.to(DEVICE)
                predicted_positions = gat_model(batch.x, batch.edge_index)
                val_loss += loss_fn(predicted_positions, batch.y).item() * batch.num_graphs
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_duration:.2f}s")

    print("Training complete.")
    if gat_epoch_times:
        print(f"Average GAT training epoch time: {np.mean(gat_epoch_times):.4f}s")


    # 4. Example: Use trained GAT for r-adaptivity and compare time
    print("\n--- Example Inference ---")
    test_mesh_nx, test_mesh_ny = 8, 8
    example_initial_mesh = dolfin.UnitSquareMesh(test_mesh_nx, test_mesh_ny)
    
    # Classical R-Adaptivity for reference
    monitor_vals_test = get_monitor_function(example_initial_mesh.coordinates())
    classical_start_time = time.time()
    example_optimized_classical = dummy_classical_r_adaptivity(example_initial_mesh, monitor_vals_test)
    classical_duration_test = time.time() - classical_start_time
    print(f"Classical r-adaptivity time for test mesh: {classical_duration_test:.6f}s")

    # GAT-based R-Adaptivity
    pyg_test_data = fenics_mesh_to_pyg_data(example_initial_mesh, device=DEVICE)
    # If monitor function was used as feature during training, add it here too:
    # monitor_tensor_test = torch.tensor(monitor_vals_test, dtype=torch.float).unsqueeze(1).to(DEVICE)
    # pyg_test_data.x = torch.cat([pyg_test_data.x, monitor_tensor_test], dim=1)

    gat_model.eval()
    with torch.inference_mode(): 
        gat_start_time = time.time()
        predicted_optimized_coords_gat = gat_model(pyg_test_data.x, pyg_test_data.edge_index).cpu().numpy()
        gat_duration_test = time.time() - gat_start_time
    print(f"GAT inference time for test mesh: {gat_duration_test:.6f}s")

    # --- Visualization (simple plot of initial and adapted meshes) ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    dolfin.plot(example_initial_mesh, ax=axs[0])
    axs[0].set_title("Initial Mesh")

    # Create a FEniCS mesh from classical adapted coords (for plotting)
    adapted_mesh_classical = dolfin.Mesh(example_initial_mesh) # Copy topology
    adapted_mesh_classical.coordinates()[:] = example_optimized_classical
    dolfin.plot(adapted_mesh_classical, ax=axs[1])
    axs[1].set_title("Classical R-Adapted (Dummy)")

    # Create a FEniCS mesh from GAT adapted coords
    adapted_mesh_gat = dolfin.Mesh(example_initial_mesh) # Copy topology
    adapted_mesh_gat.coordinates()[:] = predicted_optimized_coords_gat
    dolfin.plot(adapted_mesh_gat, ax=axs[2])
    axs[2].set_title("GAT R-Adapted")

    plt.tight_layout()
    plt.savefig("mesh_adaptivity_comparison.png")
    print("\nSaved comparison plot to mesh_adaptivity_comparison.png")
    # plt.show() # Uncomment to display plot interactively
