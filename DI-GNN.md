# DI-GNN: Dynamic Interaction Graph Neural Network for Protein–Ligand Docking

This repository contains an educational implementation of the DI-GNN architecture for predicting protein–ligand binding affinity and pose correction. The design is inspired by our originally planned architecture for DI-GNN, which integrates dynamic graph construction, multiple graph convolution layers with normalization and dropout, and custom attention pooling for graph-level embedding. The model is intended to refine docking poses by predicting binding affinity and per-graph pose correction.

---

## Overview of the Proposed Architecture

**Original Architecture (Conceptual Design):**
- **Dynamic Graph Construction:**  
  Construct graphs dynamically using k-nearest neighbors (KNN) based on 3D atomic positions.
- **Multi-layer GCN Backbone:**  
  Several (originally planned two or more) graph convolution layers process node features and coordinates.
- **Feature Aggregation:**  
  A global pooling mechanism (originally proposed as GlobalAttention) aggregates node embeddings into a fixed-size graph-level representation.
- **Dual Prediction Heads:**  
  - A regression head predicts the binding affinity of the complex.
  - An RL-inspired module predicts a ligand pose correction (adjustment vector) to refine docking poses.
- **Loss and Optimization:**  
  Combined Mean Squared Error (MSE) losses are applied to both binding affinity and pose correction outputs, with loss weighting to balance the tasks.

**Our Implementation:**
- **Dataset:**  
  A dummy dataset (`DummyProteinLigandDataset`) simulates protein–ligand complexes.  
  - Node features (`x`) and 3D positions (`pos`) are randomly generated.
  - Binding affinity (`y`) is defined as a function of the means of node features and positions (with added noise) to embed a non-trivial signal.
  - A dummy pose correction vector (`pose`) is provided.
- **Graph Construction:**  
  The model dynamically computes the graph using `knn_graph` based on node positions.
- **Model Backbone:**  
  Our DI-GNN uses three GCN layers with Batch Normalization and Dropout to learn complex representations.  
- **Global Attention Pooling:**  
  A custom `GlobalAttentionPooling` module (implemented with `scatter_softmax` and `scatter_add`) aggregates node embeddings into a graph-level feature vector.
- **Prediction Heads:**  
  - The binding affinity head outputs a scalar prediction per graph.
  - The RL-inspired head outputs a pose correction vector.
- **Training & Evaluation:**  
  A training loop with combined MSE losses (for affinity and pose) is used, and evaluation is performed by calculating RMSE and R² as well as visualizing predictions against ground truth.

---

## What We Have Implemented

- **Dynamic Graph Construction:**  
  Utilizes `knn_graph` to create graph connectivity from atomic positions on-the-fly.

- **Multi-layer GCN Backbone:**  
  Three layers of GCN with BatchNorm and Dropout are implemented, enabling deeper feature extraction.

- **Custom Global Attention Pooling:**  
  A self-implemented pooling layer aggregates node features based on learned attention scores.

- **Dual Prediction Heads:**  
  - A fully connected layer predicts binding affinity.
  - An RL-inspired module predicts pose corrections.

- **Training and Evaluation Pipelines:**  
  The training loop uses combined loss weighting; evaluation includes computation of RMSE, R², and scatter plot visualization.

- **Dummy Dataset with Embedded Signal:**  
  The dummy dataset has been enhanced to include a non-linear relationship (using trigonometric functions) in the binding affinity, making the learning task less trivial.

---

## What Remains to be Implemented

- **Integration with Real Protein–Ligand Data:**  
  The current dataset is synthetic. A full implementation would use real docking datasets (e.g., PDBbind) with proper pre-processing.

- **Protein–Ligand Joint Representation:**  
  Our current model focuses on ligand graphs only. For a complete system, incorporating protein (or binding pocket) features is essential.

- **Rotation Prediction and Full SE(3) Equivariance:**  
  The full DI-GNN should predict both rotation and translation corrections for a complete rigid-body transformation. Our implementation currently only predicts a scalar binding affinity and pose correction vector.

- **Advanced Attention Mechanisms:**  
  The original design may include multi-head attention or more complex pooling mechanisms to better capture interactions. This remains a potential area for extension.

- **Enhanced Loss Functions and Scoring:**  
  Additional components (e.g., custom energy-based loss functions or reinforcement learning components) can be integrated to further improve docking pose ranking and refinement.

---

## How to Run

1. **Prerequisites:**  
   - Python 3.x  
   - PyTorch and PyTorch Geometric (and related dependencies)  
   - matplotlib, numpy, and scikit-learn

2. **Run the Model:**  
   Execute the main Python script:
   ```bash
   python di_gnn_model.py
   ```
