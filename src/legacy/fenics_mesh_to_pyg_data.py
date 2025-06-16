import dolfin
from torch_geometric.data import Data
import torch
import numpy as np
import itertools

def fenics_mesh_to_pyg_data(mesh, device='coda'): # Added device option
    """
    Converts a FEniCS (dolfin) mesh to a PyTorch Geometric Data object.
    Node features ('x' and 'pos') are set to the node coordinates.
    Edges ('edge_index') are defined between nodes sharing a cell.
    """
    node_coordinates = mesh.coordinates()
    num_nodes = mesh.num_vertices()
    geo_dim = mesh.geometry().dim()

    if num_nodes == 0: # Handle empty mesh case
        node_coordinates_reshaped = np.empty((0, geo_dim if geo_dim > 0 else 1), dtype=float)
    else:
        node_coordinates_reshaped = node_coordinates.reshape(num_nodes, geo_dim)

    pos_tensor = torch.tensor(node_coordinates_reshaped, dtype=torch.float).to(device)
    x_tensor = torch.tensor(node_coordinates_reshaped, dtype=torch.float).to(device)

    cells = mesh.cells()
    edge_list = set()
    for cell in cells:
        for u, v in itertools.combinations(cell, 2):
            edge = tuple(sorted((u, v)))
            edge_list.add(edge)

    source_nodes = []
    target_nodes = []
    for u, v in edge_list:
        source_nodes.extend([u, v])
        target_nodes.extend([v, u])

    edge_index_tensor = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to(device)

    data = Data(x=x_tensor, edge_index=edge_index_tensor, pos=pos_tensor)

    # PyG's validate can be strict on fully empty graphs.
    if data.num_nodes > 0 or data.num_edges > 0:
        try:
            data.validate(raise_on_error=True)
        except Exception as e:
            print(f"Warning: PyG Data validation failed: {e}")
    return data
