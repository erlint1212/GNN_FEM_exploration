# models/GraphSAGE.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class RAdaptGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3,  # Default to 3 layers, configurable
                 dropout=0.5,
                 aggr='mean'): # Aggregation method for GraphSAGE ('mean', 'add', 'max')
        super(RAdaptGraphSAGE, self).__init__()

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() # BatchNorm layers

        current_dim = in_channels

        if self.num_layers == 1:
            # Single layer: in_channels -> out_channels
            self.convs.append(SAGEConv(current_dim, out_channels, aggr=aggr))
            # No BatchNorm or activation for a single layer directly outputting regression values
        else:
            # Input Layer (Layer 0)
            self.convs.append(SAGEConv(current_dim, hidden_channels, aggr=aggr))
            self.bns.append(BatchNorm(hidden_channels))
            current_dim = hidden_channels

            # Hidden Layers (Layers 1 to num_layers - 2)
            for _ in range(self.num_layers - 2): # This loop runs num_layers - 2 times
                self.convs.append(SAGEConv(current_dim, hidden_channels, aggr=aggr))
                self.bns.append(BatchNorm(hidden_channels))
                # current_dim remains hidden_channels

            # Output Layer (Layer num_layers - 1)
            # Input is current_dim (which is hidden_channels)
            # Output is out_channels.
            self.convs.append(SAGEConv(current_dim, out_channels, aggr=aggr))
            # No BatchNorm or activation after the final output layer for regression

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)

            # Apply BatchNorm and activation (e.g., ReLU or ELU) for all layers EXCEPT the output layer
            if i < self.num_layers - 1:
                x = self.bns[i](x) 
                x = F.relu(x) # Or F.elu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            # For the last layer (i == self.num_layers - 1), no BatchNorm, activation, or dropout is applied.
            
        return x
