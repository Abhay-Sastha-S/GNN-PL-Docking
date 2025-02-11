# EquiBind-Inspired Model

This repository contains a simplified, educational re-implementation of an EquiBind-inspired model for protein–ligand docking. The original EquiBind paper:

> **Reference:**  
> Stärk, H., et al. “EquiBind: Geometric deep learning for drug binding prediction.” *Nature Communications*, 2022.

## Overview

EquiBind presents a novel approach to predict a ligand’s binding pose by directly estimating a rigid-body transformation (rotation and translation) using an SE(3)-equivariant network. Its key idea is to learn representations that are invariant (or equivariant) to rotations and translations so that the model can predict the optimal binding configuration without exhaustive sampling.

## What We Have Implemented

- **Simplified E(3)-Equivariant Layer (EGNNLayer):**  
  We implemented an `EGNNLayer` that updates node features and coordinates based on the relative differences between neighboring nodes. This mimics the equivariant properties described in the EquiBind paper.

- **Stacked EGNN Architecture:**  
  The `EquiBindModel` stacks several EGNN layers and applies global pooling (via mean pooling) to produce a graph-level embedding. A fully connected head then predicts a translation vector.  
  *Note:* For simplicity, our implementation predicts only a translation (not rotation).

- **Graph Construction:**  
  We use k-nearest neighbors (k-NN) based on ligand atomic positions to dynamically create the graph connectivity.

## What Remains to be Implemented

- **Rotation Prediction:**  
  The full EquiBind model predicts both rotation and translation. Extending our model to predict a rotation matrix or quaternion is a remaining task.

- **Protein Pocket Integration:**  
  EquiBind works on protein–ligand complexes. Our implementation currently uses only ligand data. Incorporating protein pocket features would more closely replicate the paper.

- **Advanced Equivariance Mechanisms:**  
  The published model uses specialized equivariant layers and potentially multi-head attention to guarantee SE(3) equivariance. Our simplified EGNN layer approximates this behavior but could be enhanced.

- **Training on Real Data:**  
  Our code runs on synthetic (dummy) ligand graphs. Integrating real protein–ligand datasets and a large-scale pre-training scheme is needed for a full implementation.

## Usage

Run the model using Python 3:
```bash
python equibind_model.py
