import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # Updated import
from torch_geometric.nn import GCNConv, knn_graph
from torch_scatter import scatter_softmax, scatter_add  # for global pooling
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Custom Global Attention Pooling
# -------------------------------
class GlobalAttentionPooling(nn.Module):
    def __init__(self, gate_nn):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn  # A neural network that computes attention coefficients

    def forward(self, x, batch):
        # x: [N, F] node features, batch: [N] indicating graph assignment.
        gate = self.gate_nn(x)  # shape [N, 1]
        gate = gate.squeeze(1)  # shape [N]
        # Compute softmax over each graph in the batch.
        attn = scatter_softmax(gate, batch)
        # Multiply node features by attention coefficients.
        x_weighted = x * attn.unsqueeze(1)
        # Sum over nodes for each graph.
        out = scatter_add(x_weighted, batch, dim=0)
        return out

# -------------------------------
# 1. Dummy Dataset Creation with Underlying Signal
# -------------------------------
class DummyProteinLigandDataset(torch.utils.data.Dataset):
    """
    Simulates a dataset of protein-ligand complexes.
    
    Each sample is a graph with:
      - pos: 3D positions of atoms (Nx3 tensor)
      - x: Additional node features (e.g., atomic properties, dimension D)
      - edge_index: Will be computed dynamically using KNN on pos.
      - edge_attr: Dummy edge attributes (unused)
      - y: Ground truth binding affinity (regression target)
      - pose: Ground truth ligand pose correction vector, stored as a 2D tensor of shape [1, 3]
    
    Here, binding affinity is defined as a function of the average of x and pos plus noise.
    """
    def __init__(self, num_samples=200, num_nodes=20, feat_dim=3):
        self.num_samples = num_samples
        self.data_list = []
        for _ in range(num_samples):
            pos = torch.randn(num_nodes, 3)  # Simulate node positions (3D)
            x = torch.randn(num_nodes, feat_dim)  # Additional node features
            edge_attr = torch.randn(num_nodes * 4, 3)  # Dummy edge attributes
            # Define binding affinity as the mean of x and pos plus small noise.
            base_affinity = (x.mean() + pos.mean()).unsqueeze(0)
            noise = torch.randn(1) * 0.1
            binding_affinity = base_affinity + noise
            pose = torch.randn(1, 3)  # Pose correction vector.
            data = Data(x=x, pos=pos, edge_attr=edge_attr, y=binding_affinity, pose=pose)
            self.data_list.append(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_list[idx]


# -------------------------------
# 2. Improved DI-GNN Model Definition with Increased Complexity and Dropout
# -------------------------------
class DI_GNN(nn.Module):
    """
    Dynamic Interaction Graph Neural Network (DI-GNN)

    Improvements:
      - Three GCN layers with BatchNorm and Dropout.
      - Uses custom GlobalAttentionPooling for graph-level pooling.
      - Separate prediction heads for binding affinity and pose refinement.

    Input: Concatenation of node features and 3D positions.
    """
    def __init__(self, in_feat, pos_dim, hidden_channels, out_channels, pose_dim, k=6, dropout_prob=0.3):
        """
        in_feat: Dimension of additional node features.
        pos_dim: Dimension of positions (typically 3).
        hidden_channels: Hidden dimension for GCN layers.
        out_channels: Output dimension for binding affinity (typically 1).
        pose_dim: Dimension of the pose correction vector (typically 3).
        k: Number of nearest neighbors for dynamic graph construction.
        dropout_prob: Dropout probability.
        """
        super(DI_GNN, self).__init__()
        self.k = k
        self.input_dim = in_feat + pos_dim
        
        # Three GCN layers with BatchNorm and Dropout.
        self.conv1 = GCNConv(self.input_dim, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Custom global attention pooling.
        self.attention = GlobalAttentionPooling(nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.ReLU()
        ))
        # Head for binding affinity prediction.
        self.fc_affinity = nn.Linear(hidden_channels, out_channels)
        # RL-inspired module for pose refinement.
        self.rl_module = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, pose_dim)
        )

    def forward(self, data):
        # Dynamically compute graph connectivity using KNN on positions.
        edge_index = knn_graph(data.pos, k=self.k, batch=data.batch if hasattr(data, 'batch') else None)
        # Concatenate node features with positions.
        x = torch.cat([data.x, data.pos], dim=1)  # Shape: (N, in_feat + pos_dim)
        
        # First GCN layer.
        x = torch.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        # Second GCN layer.
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        # Third GCN layer.
        x = torch.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        
        # Global pooling with attention. Requires batch vector.
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_embedding = self.attention(x, batch)
        
        # Predict binding affinity.
        affinity = self.fc_affinity(graph_embedding)
        # Predict pose correction.
        pose_correction = self.rl_module(graph_embedding)
        return affinity, pose_correction


# -------------------------------
# 3. Training Loop with Combined Loss and Loss Weighting
# -------------------------------
def train(model, loader, optimizer, device, loss_weight_affinity=1.0, loss_weight_pose=0.5):
    model.train()
    total_loss = 0
    mse_loss = nn.MSELoss()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        affinity_pred, pose_pred = model(data)
        # Squeeze affinity_pred from shape [batch, 1] to [batch]
        affinity_pred = affinity_pred.squeeze(dim=1)
        loss_affinity = mse_loss(affinity_pred, data.y)
        loss_pose = mse_loss(pose_pred, data.pose)
        loss = loss_weight_affinity * loss_affinity + loss_weight_pose * loss_pose
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# -------------------------------
# 4. Evaluation and Visualization
# -------------------------------
def evaluate_model(model, loader, device):
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            affinity_pred, _ = model(data)
            affinity_pred = affinity_pred.squeeze(dim=1)
            predictions.extend(affinity_pred.cpu().numpy())
            ground_truths.extend(data.y.cpu().numpy())
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    mse = mean_squared_error(ground_truths, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(ground_truths, predictions)
    print(f"Evaluation Metrics:\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")
    
    # Scatter plot of predictions vs. ground truth.
    plt.figure(figsize=(8, 8))
    plt.scatter(ground_truths, predictions, color='blue', alpha=0.6, label='Predictions')
    min_val = min(ground_truths.min(), predictions.min())
    max_val = max(ground_truths.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('Ground Truth Binding Affinity')
    plt.ylabel('Predicted Binding Affinity')
    plt.title('Predictions vs. Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------------
# 5. Main Execution: Training and Evaluation Combined
# -------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the dummy dataset and DataLoader.
    dataset = DummyProteinLigandDataset(num_samples=200, num_nodes=20, feat_dim=3)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize the improved DI-GNN model.
    model = DI_GNN(in_feat=3, pos_dim=3, hidden_channels=128, out_channels=1, pose_dim=3, k=6, dropout_prob=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 800  # Increased training epochs.
    for epoch in range(num_epochs):
        loss = train(model, loader, optimizer, device, loss_weight_affinity=1.0, loss_weight_pose=0.5)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Evaluate and visualize model predictions.
    evaluate_model(model, loader, device)

if __name__ == '__main__':
    main()
