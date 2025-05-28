import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        deg = torch.bincount(row, minlength=x.size(0)).unsqueeze(1).clamp(min=1)
        agg = agg / deg
        return F.relu(self.linear(agg))

class GNNModel(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        self.gnn = SimpleGNNLayer(embed_dim, embed_dim)

    def forward(self, edge_index):
        x = self.embedding(torch.arange(self.embedding.num_embeddings, device=edge_index.device))
        return self.gnn(x, edge_index)