import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_add, scatter_mean

# --- Simplified EGNN Layer ---
class EGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(EGNNLayer, self).__init__()
        # Linear transformation for node features
        self.fc_h = nn.Linear(in_features, out_features)
        # Linear layer to compute scalar weight for coordinate update
        self.fc_x = nn.Linear(in_features, 1)
    
    def forward(self, h, x, edge_index):
        # h: [N, F] node features, x: [N, 3] coordinates, edge_index: [2, E]
        row, col = edge_index
        h_j = h[col]
        # Compute coordinate differences for each edge
        delta_x = x[row] - x[col]  # [E, 3]
        
        # Message passing for features
        m = F.relu(self.fc_h(h_j))  # [E, out_features]
        agg_h = scatter_add(m, row, dim=0, dim_size=h.size(0))
        h_new = h + agg_h
        
        # Compute a scalar weight for each edge and update coordinates
        weight = F.relu(self.fc_x(h_j))  # [E, 1]
        weighted_delta = delta_x * weight  # [E, 3]
        agg_x = scatter_add(weighted_delta, row, dim=0, dim_size=x.size(0))  # [N, 3]
        x_new = x + agg_x
        
        return h_new, x_new

# --- EGNN Model Inspired by EquiBind ---
class EquiBindModel(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers=3, k=6):
        super(EquiBindModel, self).__init__()
        self.k = k
        self.layers = nn.ModuleList()
        # First layer maps input features to hidden_dim
        self.layers.append(EGNNLayer(in_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(EGNNLayer(hidden_dim, hidden_dim))
        # Global pooling: average pooling over nodes per graph
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        # Final head predicts a translation vector (3D) for the ligand transformation\n
        self.trans_head = nn.Linear(hidden_dim, 3)
    
    def forward(self, data):
        # data.x: ligand node features, data.pos: ligand coordinates
        # Build kNN graph from ligand coordinates
        edge_index = knn_graph(data.pos, k=self.k, batch=data.batch if hasattr(data, 'batch') else None)
        h, x = data.x, data.pos
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        # Global pooling: mean over nodes
        if hasattr(data, 'batch'):
            batch = data.batch
            h_pool = scatter_mean(h, batch, dim=0)
        else:
            h_pool = h.mean(dim=0, keepdim=True)
        emb = F.relu(self.fc(h_pool))
        translation = self.trans_head(emb)  # Predicted translation vector\n
        # Apply the predicted translation to all ligand atoms\n
        x_transformed = x + translation  # broadcasting translation\n
        return x_transformed

# Example usage:
if __name__ == '__main__':
    from torch_geometric.data import Data
    num_atoms = 30
    # Simulate 30 atoms with 4-dimensional features
    x = torch.randn(num_atoms, 4)
    # Simulate 3D coordinates for these atoms
    pos = torch.randn(num_atoms, 3)
    data = Data(x=x, pos=pos)
    model = EquiBindModel(in_features=4, hidden_dim=64, num_layers=3, k=6)
    new_coords = model(data)
    print("EquiBind-inspired transformed coordinates:\n", new_coords)
