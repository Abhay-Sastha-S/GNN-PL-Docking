# GNN-PL-Docking

**GNN-PL-Docking** is a repository for protein–ligand docking using Graph Neural Networks (GNNs). The repository contains re-implementations inspired by recent state-of-the-art approaches—including DI-GNN, MedusaGraph, and EquiBind—and serves as a platform for exploring how different GNN architectures and feature selection techniques can be applied to docking tasks.

## Overview

Protein–ligand docking is a critical step in computational drug discovery. Classical docking methods can be computationally intensive and may not capture the complex, non-linear interactions that govern binding affinity. This repository focuses on:
- **Developing and comparing GNN-based docking models** that learn to predict binding affinity and pose corrections from graph representations of molecular structures.
- **Implementing multiple architectures** (e.g., DI-GNN, MedusaGraph-inspired, and EquiBind-inspired models) to study their performance on docking tasks.
- **Conducting feature selection experiments** using various techniques (e.g., FastICA, PCA, and Variance Threshold) with composition-based and graphlet features to enhance model performance.

## Repository Structure

- **Model Implementations:**
  - `DI-GNN.py` and `DI-GNN.md`:  
    Contains the proposed mock implementation and documentation for the Dynamic Interaction Graph Neural Network (DI-GNN) model. This model uses dynamic graph construction via k-NN, multiple GCN layers with BatchNorm and Dropout, and a custom global attention pooling mechanism to predict both binding affinity and ligand pose corrections.
    
  - `equibind.py` and `equibind.md`:  
    Provides an educational, simplified re-implementation inspired by EquiBind. This version focuses on predicting a translation (as a proxy for the full rigid-body transformation) using an equivariant graph neural network.
    
  - `medusagraph.py` and `MedusaGraph.md`:  
    Contains a MedusaGraph-inspired model that refines docking poses by predicting per-atom coordinate corrections. This implementation uses stacked GCN layers to process ligand graphs and output coordinate adjustments.

- **Proposal and Documentation:**
  - `proposal.md`:  
    A project proposal document that explains the conceptual design of DI-GNN, detailing the proposed architecture, what has been implemented so far, and what remains to be developed for a full implementation.

- **Proposal and Documentation:**
  - `ZINC-in-tranches.smiles.zip` and `pdbbind_index.tar.gz`:
    Proposed databases that could be clubbed to create the final database to train the model on protein-ligand docking.
    - [PDBBind](http://www.pdbbind.org.cn/download/pdbbind_v2020_plain_text_index.tar.gz)
    - [ZINC Smiles](https://files.docking.org/2D/)
      
## Main Use and Goal

The primary goal of this repository is to:
- **Facilitate the comparison of different GNN architectures** (such as DI-GNN, EquiBind-inspired, and MedusaGraph-inspired models) on docking tasks.
- **Enable experiments with advanced feature selection techniques** (including PCA, ICA, and variance thresholding) for optimizing input features for docking.
- **Support drug repurposing analysis:**  
  The repository also includes modules for analyzing docking results to compute drug consensus scores, which can be used to suggest potential therapeutic candidates for emerging diseases.

This repository is aimed at the academic and research community to serve as a stepping stone for further exploration and development of deep learning methods in structural bioinformatics and drug discovery.

## ----NOTE----
the image added to this repository is from the trial run of the proposed DI-GNN model for PL Docking, it seems to have achieved 
```bash
Epoch 800/800, Loss: 0.0470
Evaluation Metrics:
RMSE: 0.0358
R²: 0.9747
```
but these high scores might be a result of the data leakage through the simple feature engineering and non-realistic mapping of functions, as it was a POC. But can be fixed via more realistic dataset as input.
- Non-Linear Relationship Between Features and Target
- Heterogeneous Node Features and Edge Attributes
- More Noise & Randomness in Graph Structure 

