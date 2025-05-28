import torch
from torch.utils.data import DataLoader
from dataset import load_dataset
from gnn_model import GNNModel
from train_gnn import train_gnn
from eval_gnn import eval_gnn
from utils import set_seed

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, edge_index, num_nodes = load_dataset()
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024)

    model = GNNModel(num_nodes, embed_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 11):
        print(f"\n[Epoch {epoch}]")
        train_gnn(model, train_loader, edge_index, optimizer, device)
        eval_gnn(model, test_loader, edge_index, device)