import unittest
import torch

# Assuming GAT.py is in GAT_r_adaptivity/models/
from GAT_r_adaptivity.models.GAT import RAdaptGAT

class TestRAdaptGAT(unittest.TestCase):

    def _create_dummy_data(self, num_nodes, in_channels, device='cpu'):
        x = torch.randn((num_nodes, in_channels), device=device)
        # Create a simple edge_index (e.g., fully connected for small num_nodes, or a chain)
        if num_nodes <= 1:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            # Simple chain: 0-1, 1-2, ...
            source_nodes = torch.arange(0, num_nodes - 1, device=device)
            target_nodes = torch.arange(1, num_nodes, device=device)
            edge_index = torch.stack([
                torch.cat([source_nodes, target_nodes]),
                torch.cat([target_nodes, source_nodes])
            ], dim=0).to(dtype=torch.long)
        return x, edge_index

    def test_model_instantiation_and_forward(self):
        """Test model instantiation with various parameters and a forward pass."""
        configs = [
            {"in_channels": 16, "hidden_channels": 32, "out_channels": 2, "heads": 4, "num_layers": 1},
            {"in_channels": 8, "hidden_channels": 16, "out_channels": 3, "heads": 2, "num_layers": 2},
            {"in_channels": 32, "hidden_channels": 64, "out_channels": 2, "heads": 8, "num_layers": 3},
        ]
        num_nodes = 10
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for config in configs:
            model = RAdaptGAT(
                in_channels=config["in_channels"],
                hidden_channels=config["hidden_channels"],
                out_channels=config["out_channels"],
                heads=config["heads"],
                num_layers=config["num_layers"],
                dropout=0.1
            ).to(device)
            
            x, edge_index = self._create_dummy_data(num_nodes, config["in_channels"], device=device)
            
            # Test forward pass
            model.train() # To activate dropout if it behaves differently
            output_train = model(x, edge_index)
            self.assertEqual(output_train.shape, (num_nodes, config["out_channels"]))
            self.assertEqual(output_train.device.type, device)

            model.eval() # To deactivate dropout
            output_eval = model(x, edge_index)
            self.assertEqual(output_eval.shape, (num_nodes, config["out_channels"]))
            self.assertEqual(output_eval.device.type, device)


    def test_single_node_graph(self):
        """Test with a graph having a single node (no edges)."""
        in_channels = 16
        out_channels = 2
        num_nodes = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = RAdaptGAT(in_channels, 32, out_channels, heads=4, num_layers=2).to(device)
        x, edge_index = self._create_dummy_data(num_nodes, in_channels, device=device)
        
        output = model(x, edge_index)
        self.assertEqual(output.shape, (num_nodes, out_channels))

    def test_no_edge_graph(self):
        """Test with a graph having multiple nodes but no edges."""
        in_channels = 16
        out_channels = 2
        num_nodes = 5
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = RAdaptGAT(in_channels, 32, out_channels, heads=4, num_layers=2).to(device)
        x = torch.randn((num_nodes, in_channels), device=device)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device) # No edges
        
        output = model(x, edge_index)
        self.assertEqual(output.shape, (num_nodes, out_channels))


if __name__ == '__main__':
    unittest.main()
