DNN-enhanced Mesh Refinement for FEM using GNNs
=============================================

This documentation details a novel approach to Finite Element Method (FEM) mesh refinement using Graph Neural Networks (GNNs).  The project explores the application of various GNN architectures to optimize mesh quality through r-adaptivity, leading to improved accuracy and efficiency in FEM simulations.

Introduction
------------

Finite Element Method (FEM) simulations are crucial in various engineering and scientific fields. However, the accuracy and computational cost of FEM strongly depend on the quality of the mesh.  R-adaptivity, a mesh refinement technique, aims to optimize the mesh by relocating mesh nodes. This project leverages the power of GNNs to predict optimal node positions, surpassing traditional refinement methods.

Key Features
------------

* **GNN-based R-adaptivity:** Utilizes GNNs to predict optimal node locations for r-adaptive mesh refinement.
* **Support for Multiple GNN Architectures:** Evaluates the performance of different GNN models (e.g., GCN, GraphSAGE, GAT) for mesh optimization.
* **Mesh Quality Metrics:** Implements various metrics to assess and improve mesh quality (e.g., element distortion, aspect ratio).
* **Integration with FEM Solvers:** Provides tools and examples for integrating the GNN-driven refinement with existing FEM solvers (e.g., FEniCS).
* **Performance Analysis:** Benchmarks the performance of GNN-based refinement against traditional r-adaptivity methods.

Getting Started
-------------

1.  **Installation:** Follow the instructions in the :ref:`installation` guide to set up the project.
2.  **Quickstart:** The :ref:`quickstart` section provides a minimal working example of GNN-based mesh refinement.
3.  **GNN Model Configuration:** Learn how to configure and train different GNN models in the :ref:`gnn_configuration` section.

.. toctree::
   :maxdepth: 2

   introduction
   installation
   quickstart
   modules
   contact

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
