import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.W(x)
        row, col = edge_index
        h_i = h[row]
        h_j = h[col]
        a_input = torch.cat([h_i, h_j], dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.attn)).squeeze()

        e = torch.exp(e)
        e_sum = torch.zeros(x.size(0), device=x.device)
        e_sum = e_sum.index_add(0, row, e)
        alpha = e / (e_sum[row] + 1e-16)
        alpha = self.dropout(alpha)

        out = torch.zeros_like(h)
        out = out.index_add(0, row, alpha.unsqueeze(1) * h_j)

        return F.elu(out)

class GATModel(nn.Module):
    def __init__(self, num_nodes, in_dim=64, hidden_dim=64, out_dim=64):
        super(GATModel, self).__init__()
        self.embedding = nn.Embedding(num_nodes, in_dim)
        self.gat1 = GraphAttentionLayer(in_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, out_dim)

    def forward(self, edge_index):
        x = self.embedding(torch.arange(self.embedding.num_embeddings, device=edge_index.device))
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        return x
