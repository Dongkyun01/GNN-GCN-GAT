import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index
        self_loop = torch.arange(0, num_nodes, dtype=torch.long, device=x.device)
        self_loop = torch.stack([self_loop, self_loop])
        edge_index = torch.cat([edge_index, self_loop], dim=1)

        deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

        messages = norm.unsqueeze(1) * x[edge_index[1]]
        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_index[0], messages)

        return F.relu(self.linear(agg))

class GCNModel(nn.Module):
    def __init__(self, num_nodes, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        self.gcn1 = SimpleGCNLayer(embed_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, embed_dim)

    def forward(self, edge_index):
        x = self.embedding(torch.arange(self.embedding.num_embeddings, device=edge_index.device))
        x = self.gcn1(x, edge_index, x.size(0))
        x = self.gcn2(x, edge_index, x.size(0))
        return x
