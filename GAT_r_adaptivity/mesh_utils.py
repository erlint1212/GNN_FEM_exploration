# mesh_utils.py
import dolfin
import numpy as np
from config import CLASSICAL_ADAPT_STRENGTH # Import constants if needed

def get_boundary_nodes(mesh):
    """Identifies boundary nodes of a FEniCS mesh."""
    boundary_nodes_set = set()
    # Create a MeshFunction to mark exterior facets
    # Ensure we use the mesh's full dimension for facets (dim-1)
    # and 0 for vertices.
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    dolfin.DomainBoundary().mark(boundary_markers, 1) # Mark all exterior facets
    
    for f in dolfin.facets(mesh):
        if boundary_markers[f.index()] == 1: # If it's an exterior facet
            for v_idx in f.entities(0): # Get vertex indices of the facet
                boundary_nodes_set.add(v_idx)
    return list(boundary_nodes_set)

def dummy_classical_r_adaptivity(mesh, monitor_values, strength=CLASSICAL_ADAPT_STRENGTH, mesh_dimensions=None):
    """
    Performs a simple r-adaptivity step.
    Boundary nodes are kept fixed to their original positions.
    Interior nodes are moved based on the monitor function.
    """
    original_coords = mesh.coordinates() # Keep original coordinates for boundary nodes
    old_coords = np.copy(original_coords)
    new_coords = np.copy(old_coords) # Start with original, modify interior nodes
    num_nodes = mesh.num_vertices()
    geo_dim = mesh.geometry().dim()
    
    boundary_node_indices = get_boundary_nodes(mesh)

    for i in range(num_nodes):
        if i in boundary_node_indices:
            # new_coords[i] = original_coords[i] # This is already implicitly true as new_coords starts as copy
            continue # Don't move boundary nodes based on monitor averaging

        direction_vector = np.zeros(geo_dim)
        total_weight = 0.0
        for j in range(num_nodes):
            if i == j: continue
            diff = old_coords[j] - old_coords[i]
            dist_sq = np.sum(diff**2)
            if dist_sq < 1e-12: continue # Avoid division by zero for coincident nodes
            
            weight = (monitor_values[j] + monitor_values[i]) / 2.0 / (dist_sq + 1e-6) # Small epsilon
            direction_vector += weight * diff
            total_weight += weight
        
        if total_weight > 1e-6:
            displacement = strength * (direction_vector / total_weight)
            new_coords[i] += displacement # new_coords[i] was old_coords[i] before this

    # Clipping for interior nodes (boundary nodes are already fixed to original, valid positions)
    if mesh_dimensions:
        min_x, max_x = 0.0, mesh_dimensions.get("width", 1.0)
        min_y, max_y = 0.0, mesh_dimensions.get("height", 1.0)
        for i in range(num_nodes):
            if i not in boundary_node_indices: # Only clip interior nodes
                new_coords[i, 0] = np.clip(new_coords[i, 0], min_x, max_x)
                new_coords[i, 1] = np.clip(new_coords[i, 1], min_y, max_y)
        # Special clipping for pipe with obstacle might be needed here if interior nodes can enter the obstacle.
    return new_coords


def check_mesh_quality(mesh, operation_name=""):
    """
    Checks mesh quality by looking at minimum cell volume.
    Returns (bool: quality_ok, float: min_cell_volume_value)
    """
    min_cell_vol_val = -10.0 # Default to a value indicating it wasn't computed or failed
    if mesh.num_cells() > 0 :
        try:
            # Project CellVolume (UFL expression) to DG0 space to get per-cell values
            V_dg0 = dolfin.FunctionSpace(mesh, "DG", 0)
            cell_volumes_p0 = dolfin.project(dolfin.CellVolume(mesh), V_dg0)
            if cell_volumes_p0.vector().size() > 0:
                min_cell_vol_val = np.min(cell_volumes_p0.vector().get_local())
            else: # Should not happen if num_cells > 0
                min_cell_vol_val = -1.0 # Indicate no cell values found
        except Exception as vol_exc:
            print(f"  Could not compute min_cell_volume for {operation_name}: {vol_exc}")
            min_cell_vol_val = -2.0 # Indicate error during volume check
    else: # no cells
        min_cell_vol_val = -3.0 # Indicate no cells to check volume for
    
    # Check if computed and positive (min_cell_vol_val > -5 means it was likely computed)
    if min_cell_vol_val < 1e-12 and min_cell_vol_val > -5.0:
        print(f"  Warning: Mesh from {operation_name} likely tangled. Min cell volume: {min_cell_vol_val:.2e}.")
        return False, min_cell_vol_val # Bad quality
    
    # print(f"  Mesh quality OK for {operation_name} (Min cell vol: {min_cell_vol_val:.2e}).") # Can be verbose
    return True, min_cell_vol_val # Good quality
