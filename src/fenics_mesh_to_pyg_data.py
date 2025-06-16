import dolfin
from torch_geometric.data import Data
import torch
import numpy as np
import itertools

def fenics_mesh_to_pyg_data(mesh, device='cpu', additional_features=None): # Changed default device, added additional_features
    """
    Converts a FEniCS (dolfin) mesh to a PyTorch Geometric Data object.
    Node features ('x') include coordinates and optional additional features.
    'pos' stores only the coordinates for PyG visualization/geometric ops.
    Edges ('edge_index') are defined between nodes sharing a cell.

    Args:
        mesh (dolfin.Mesh): The FEniCS mesh.
        device (str): The torch device ('cuda' or 'cpu').
        additional_features (np.ndarray, optional): A NumPy array of shape [num_nodes, num_add_features]
                                                     to be concatenated with coordinates.
                                                     Typically, these could be monitor function values.
    """
    node_coordinates = mesh.coordinates()
    num_nodes = mesh.num_vertices()
    geo_dim = mesh.geometry().dim()

    if num_nodes == 0:
        node_coordinates_reshaped = np.empty((0, geo_dim if geo_dim > 0 else 1), dtype=float)
        # If additional_features are expected, handle their empty shape too
        num_add_feat = additional_features.shape[1] if additional_features is not None and additional_features.ndim == 2 else 0
        combined_features_np = np.empty((0, geo_dim + num_add_feat), dtype=float)

    else:
        node_coordinates_reshaped = node_coordinates.reshape(num_nodes, geo_dim)
        if additional_features is not None:
            if additional_features.shape[0] != num_nodes:
                raise ValueError(f"Shape mismatch: Coordinates have {num_nodes} nodes, "
                                 f"but additional_features have {additional_features.shape[0]} nodes.")
            # Ensure additional_features is 2D
            if additional_features.ndim == 1:
                additional_features = additional_features.reshape(-1, 1)
            combined_features_np = np.concatenate((node_coordinates_reshaped, additional_features), axis=1)
        else:
            combined_features_np = node_coordinates_reshaped

    # 'pos' should always be just the geometric coordinates
    pos_tensor = torch.tensor(node_coordinates_reshaped, dtype=torch.float).to(device)
    # 'x' contains coordinates + additional features
    x_tensor = torch.tensor(combined_features_np, dtype=torch.float).to(device)


    cells = mesh.cells()
    edge_list = set()
    # Ensure cells is not empty before iterating
    if cells.shape[0] > 0 :
        for cell in cells:
            # Ensure cell has at least 2 vertices for combinations
            if len(cell) >= 2:
                for u, v in itertools.combinations(cell, 2):
                    edge = tuple(sorted((u, v))) # Ensure consistent edge representation
                    edge_list.add(edge)

    source_nodes = []
    target_nodes = []
    for u, v in edge_list:
        source_nodes.extend([u, v]) # Add edge in both directions for undirected graph
        target_nodes.extend([v, u])

    edge_index_tensor = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to(device)

    data = Data(x=x_tensor, edge_index=edge_index_tensor, pos=pos_tensor)

    if data.num_nodes > 0 or data.num_edges > 0:
        try:
            data.validate(raise_on_error=True)
        except Exception as e:
            print(f"Warning: PyG Data validation failed: {e}. Data details: Nodes={data.num_nodes}, Edges={data.num_edges/2}, Features={data.num_node_features}")
            # You might want to inspect data.x, data.edge_index, data.pos here if validation fails
    return data
