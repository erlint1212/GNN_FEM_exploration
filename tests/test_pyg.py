import torch
import os
import sys
import textwrap

print("--- PyTorch Geometric Installation Test ---")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("-" * 40)

test_passed = True
messages = []

# 1. Check PyTorch and CUDA Availability
try:
    import torch
    messages.append(f"[SUCCESS] PyTorch imported (Version: {torch.__version__})")
    cuda_available = torch.cuda.is_available()
    messages.append(f"[INFO] torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        messages.append(f"  [INFO] CUDA device count: {torch.cuda.device_count()}")
        try:
            cudnn_version = torch.backends.cudnn.version()
            messages.append(f"  [INFO] CuDNN version: {cudnn_version}")
        except Exception as e:
             messages.append(f"  [WARNING] Could not get CuDNN version: {e}")
        for i in range(torch.cuda.device_count()):
             messages.append(f"  [INFO] Device {i} name: {torch.cuda.get_device_name(i)}")
    else:
        messages.append("[WARNING] CUDA not available according to PyTorch. GPU tests skipped.")

except ImportError as e:
    messages.append(f"[FAILURE] Failed to import PyTorch. Installation is broken. Error: {e}")
    test_passed = False
    # Exit early if torch isn't found
    print("\n".join(messages))
    print("\n--- Test FAILED (PyTorch import error) ---")
    sys.exit(1)
except Exception as e:
    messages.append(f"[FAILURE] Error during PyTorch/CUDA check. Error: {e}")
    test_passed = False


# 2. Check Core PyG and Dependencies Imports
dependencies = ['torch_geometric', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'pyg_lib']
for dep in dependencies:
    try:
        module = __import__(dep)
        version = getattr(module, '__version__', 'N/A')
        messages.append(f"[SUCCESS] {dep} imported (Version: {version})")
    except ImportError as e:
        # pyg_lib might be optional or integrated in newer PyG versions, treat as warning first
        level = "[FAILURE]" if dep != 'pyg_lib' else "[WARNING]"
        messages.append(f"{level} Failed to import {dep}. Check installation. Error: {e}")
        if level == "[FAILURE]":
            test_passed = False
    except Exception as e:
        messages.append(f"[FAILURE] Error importing {dep}. Error: {e}")
        test_passed = False


# Exit if core dependencies failed
if not test_passed:
    print("\n".join(messages))
    print("\n--- Test FAILED (Core PyG/dependency import errors) ---")
    sys.exit(1)


# 3. Test Creating PyG Data Object and a Layer
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv

    # Simple graph: 3 nodes (0, 1, 2), 2 edges (0->1, 1->2)
    edge_index = torch.tensor([[0, 1],
                               [1, 2]], dtype=torch.long)
    # Node features: 3 nodes, 4 features each
    x = torch.randn(3, 4, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    messages.append(f"[SUCCESS] Created torch_geometric.data.Data object: {data}")

    # Instantiate a GCN layer
    num_node_features = data.num_node_features
    num_classes = 2 # Example output classes
    layer = GCNConv(num_node_features, num_classes)
    messages.append(f"[SUCCESS] Instantiated torch_geometric.nn.GCNConv layer.")

    # Basic forward pass on CPU
    output_cpu = layer(data.x, data.edge_index)
    messages.append(f"[SUCCESS] GCNConv forward pass on CPU completed. Output shape: {output_cpu.shape}")

except Exception as e:
    messages.append(f"[FAILURE] Error during basic PyG object/layer test. Error: {e}")
    test_passed = False


# 4. Test GPU Usage (if available and previous steps passed)
if cuda_available and test_passed:
    messages.append("[INFO] Attempting GPU tests...")
    try:
        device = torch.device('cuda')
        gpu_data = data.to(device)
        messages.append(f"  [SUCCESS] Moved Data object to {device}.")

        gpu_layer = layer.to(device)
        messages.append(f"  [SUCCESS] Moved GCNConv layer to {device}.")

        # Perform a simple forward pass on GPU
        output_gpu = gpu_layer(gpu_data.x, gpu_data.edge_index)
        messages.append(f"  [SUCCESS] GCNConv forward pass on GPU completed. Output shape: {output_gpu.shape}")

        # Check output is on GPU
        if output_gpu.device.type != 'cuda':
             messages.append(f"  [FAILURE] Output tensor is not on CUDA device ({output_gpu.device})")
             test_passed = False
        else:
             messages.append(f"  [SUCCESS] Output tensor is on CUDA device ({output_gpu.device})")

    except Exception as e:
        messages.append(f"  [FAILURE] GPU test failed. Error: {e}")
        test_passed = False

# --- Final Result ---
print("-" * 40)
print("\n".join(messages))
print("-" * 40)

if test_passed:
    print("\n--- PyTorch Geometric Installation Test PASSED ---")
else:
    print("\n--- PyTorch Geometric Installation Test FAILED ---")
