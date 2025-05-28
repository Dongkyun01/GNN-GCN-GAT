import torch
import torch.nn.functional as F

def train_gat(model, train_loader, edge_index, optimizer, device):
    model.train()
    total_loss = 0
    for batch_user, batch_item, batch_label in train_loader:
        batch_user = batch_user.to(device)
        batch_item = batch_item.to(device)
        batch_label = batch_label.float().to(device)

        optimizer.zero_grad()
        node_emb = model(edge_index.to(device))

        user_emb = node_emb[batch_user]
        item_emb = node_emb[batch_item]
        pred = (user_emb * item_emb).sum(dim=1)

        loss = F.binary_cross_entropy_with_logits(pred, batch_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[GAT] Train Loss: {avg_loss:.4f}")
    return avg_loss