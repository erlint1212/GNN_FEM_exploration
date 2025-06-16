import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm # Ensure BatchNorm is imported

class RAdaptGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=4,  # Default to 4 layers, configurable
                 heads=8,
                 dropout=0.6):
        super(RAdaptGAT, self).__init__()

        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() # BatchNorm layers

        current_dim = in_channels

        if self.num_layers == 1:
            # Single layer: in_channels -> out_channels, no concat needed for output head
            # Typically, a single head is used if concat=False for the output layer.
            # If you intend multiple heads to be averaged, heads for GATv2Conv should be set,
            # and concat=False will average them.
            output_heads = 1 # For regression output, usually 1 head or averaged heads
            self.convs.append(GATv2Conv(current_dim, out_channels, heads=output_heads, concat=False, dropout=dropout))
            # No BatchNorm or activation for a single layer directly outputting regression values
        else:
            # Input Layer (Layer 0)
            # Output of this layer will be hidden_channels * heads
            self.convs.append(GATv2Conv(current_dim, hidden_channels, heads=heads, concat=True, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels * heads))
            current_dim = hidden_channels * heads

            # Hidden Layers (Layers 1 to num_layers - 2)
            for _ in range(self.num_layers - 2): # This loop runs num_layers - 2 times
                self.convs.append(GATv2Conv(current_dim, hidden_channels, heads=heads, concat=True, dropout=dropout))
                self.bns.append(BatchNorm(hidden_channels * heads))
                # current_dim remains hidden_channels * heads due to concat=True

            # Output Layer (Layer num_layers - 1)
            # Input is current_dim (which is hidden_channels * heads)
            # Output is out_channels. Use heads=1 and concat=False for the final regression.
            self.convs.append(GATv2Conv(current_dim, out_channels, heads=1, concat=False, dropout=dropout))
            # No BatchNorm or activation after the final output layer

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            # Nodal dropout before convolution
            x_dropped = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            x = self.convs[i](x_dropped, edge_index)

            # Apply BatchNorm and ELU activation for all layers EXCEPT the output layer
            if i < self.num_layers - 1:
                x = self.bns[i](x) # self.bns will have num_layers - 1 elements
                x = F.elu(x)
            # For the last layer (i == self.num_layers - 1), no BatchNorm or ELU is applied.
            
        return x
