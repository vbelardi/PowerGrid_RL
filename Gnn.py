import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class GNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64, n_layers=2):
        super().__init__()
        self.first_conv = GCNConv(node_feat_dim, hidden_dim)
        self.first_norm = nn.LayerNorm(hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.edge_enc = nn.Linear(edge_feat_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        # Node convs
        x = F.relu(self.first_norm(self.first_conv(x, edge_index)))
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
        # Edge features
        e = F.relu(self.edge_norm(self.edge_enc(edge_attr)))
        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst], e], dim=1)
        logits = self.actor(edge_repr).squeeze(-1)
        # Critic
        graph_emb = global_mean_pool(x, batch)
        value = self.critic(graph_emb).squeeze(-1)
        return logits, value
