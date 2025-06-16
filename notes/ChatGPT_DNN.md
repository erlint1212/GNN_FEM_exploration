For mesh‐based finite element (FEM) problems it’s very effective to use networks that “know” about the mesh’s connectivity. In practice, you might want to combine approaches that:

1. **Incorporate mesh structure:**  
   • **Graph Neural Networks (GNNs)** naturally encode mesh connectivity. They operate on nodes (mesh vertices) and edges (connectivity between elements) so that local geometric and topological features are preserved.  
   • Specialized architectures such as **MeshCNN** and **MeshGraphNets** have been designed specifically for learning from mesh‐structured data.

2. **Enforce physical consistency:**  
   • **Physics-Informed Neural Networks (PINNs)** integrate the governing PDEs (and their boundary conditions) into the loss function. This can help ensure that the DNN’s predictions respect the underlying mechanics, which is especially valuable in FEM applications.

3. **Hybrid or hierarchical approaches:**  
   • Recent methods combine ideas from both worlds—for example, MeshGraphNets (or its variants like MAgNET) use GNN layers in an encoder–processor–decoder structure that captures both local (through message passing) and global information (using pooling or attention mechanisms).

In PyTorch, you can implement these using libraries like [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) to handle graph data.

Here are some free, open-access papers that discuss these approaches in detail:

- **MeshCNN: A Network with an Edge**  
  This work presents a convolutional network designed for meshes by defining convolutions on mesh edges.  
  [https://arxiv.org/abs/1809.05910](https://arxiv.org/abs/1809.05910)  
  citeturn0academia20

- **Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets)**  
  This paper introduces MeshGraphNets—a hybrid model that learns the dynamics of mesh-based simulations via GNNs.  
  [https://arxiv.org/abs/2010.03409](https://arxiv.org/abs/2010.03409)  
  citeturn0search2

- **MAgNET: A Graph U-Net Architecture for Mesh-Based Simulations**  
  This recent work develops a hierarchical graph U-Net architecture tailored for FEM simulations.  
  [https://arxiv.org/abs/2211.00713](https://arxiv.org/abs/2211.00713)  
  citeturn0academia21

- **Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations**  
  Although not mesh-specific, this paper explains how to incorporate PDE constraints into a DNN’s loss, which can be very useful when you want to enforce physical laws alongside learning from FEM data.  
  [https://arxiv.org/abs/1711.10561](https://arxiv.org/abs/1711.10561)  
  citeturn0academia22

By combining a mesh-aware model (like a GNN-based architecture) with physics-informed elements, you can build a DNN that is both efficient at handling irregular mesh data and faithful to the underlying FEM physics. This hybrid approach has shown excellent results in recent literature and is well supported in Python with PyTorch.
