# MedusaGraph-Inspired Model

This repository provides a simplified, educational re-implementation of a MedusaGraph-inspired model for protein–ligand docking refinement. The ideas here are inspired by recent MedusaGraph-related publications.

## Overview

MedusaGraph refines docking poses by using a graph neural network to predict per-atom coordinate corrections for a ligand. In the full MedusaGraph approach, both protein and ligand features are integrated, and the network outputs adjustments that improve the docking pose quality.

## What We Have Implemented

- **GCN-Based Refinement Network:**  
  We implemented a model that uses a stack of Graph Convolutional Network (GCN) layers (with ReLU activations, BatchNorm, and Dropout) to process node features extracted from the ligand.

- **Dynamic Graph Construction:**  
  The model builds a k-NN graph (using k=6) from ligand atomic positions to determine connectivity and capture local interactions.

- **Per-Atom Correction Prediction:**  
  A final linear layer predicts coordinate corrections for each atom. These corrections are added to the original coordinates to produce refined ligand positions.

## What Remains to be Implemented

- **Protein Feature Integration:**  
  The original MedusaGraph architecture uses both protein and ligand graphs. Our implementation currently focuses solely on the ligand. Incorporating protein features is needed for a full system.

- **Pose Scoring and Ranking:**  
  MedusaGraph also includes mechanisms to score and rank the refined poses. Our model outputs only refined coordinates; additional modules for scoring are left for future work.

- **Advanced Architectures and Attention:**  
  More sophisticated architectures (e.g., using attention mechanisms across protein–ligand interfaces) could improve performance. Our implementation uses basic GCN layers.

- **Training on Real Datasets:**  
  We demonstrate the approach with dummy data. Applying the model to real docking data with proper pre-processing would be necessary for practical applications.

## Usage

Run the model using Python 3:
```bash
python medusagraph_model.py
