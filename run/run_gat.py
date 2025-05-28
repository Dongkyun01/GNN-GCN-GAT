import torch
from torch.utils.data import DataLoader
from dataset import load_dataset
from models.gat_model import GATModel
from train.train_gat import train_gat
from eval.eval_gat import eval_gat
from utils import set_seed

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, edge_index, num_nodes = load_dataset()
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024)

    model = GATModel(num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 11):
        print(f"\n[Epoch {epoch}]")
        train_gat(model, train_loader, edge_index, optimizer, device)
        eval_gat(model, test_loader, edge_index, device)