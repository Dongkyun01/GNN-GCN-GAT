# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class InteractionDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, labels):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_tensor[idx], self.item_tensor[idx], self.labels[idx]

def load_dataset():
    # Load data
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    ratings['user'] = user_enc.fit_transform(ratings['user'])
    ratings['item'] = item_enc.fit_transform(ratings['item'])

    num_users = ratings['user'].nunique()
    num_items = ratings['item'].nunique()
    num_nodes = num_users + num_items

    user_tensor = torch.tensor(ratings['user'].values, dtype=torch.long)
    item_tensor = torch.tensor(ratings['item'].values + num_users, dtype=torch.long)
    edge_index = torch.stack([user_tensor, item_tensor], dim=0)
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    # Train/Test split
    train_u, test_u, train_v, test_v = train_test_split(user_tensor, item_tensor, test_size=0.2, random_state=42)

    # Negative sampling
    existing_edges = set((u.item(), v.item()) for u, v in zip(user_tensor, item_tensor))
    def sample_neg(u_list, v_list, num_neg):
        neg_u, neg_v = [], []
        while len(neg_u) < num_neg:
            u = torch.randint(0, num_users, (1,)).item()
            v = torch.randint(0, num_items, (1,)).item() + num_users
            if (u, v) not in existing_edges:
                neg_u.append(u)
                neg_v.append(v)
        return torch.tensor(neg_u), torch.tensor(neg_v)

    train_neg_u, train_neg_v = sample_neg(train_u, train_v, len(train_u))
    test_neg_u, test_neg_v = sample_neg(test_u, test_v, len(test_u))

    # Positive + Negative 결합
    train_user = torch.cat([train_u, train_neg_u])
    train_item = torch.cat([train_v, train_neg_v])
    train_label = torch.cat([torch.ones(len(train_u)), torch.zeros(len(train_neg_u))])

    test_user = torch.cat([test_u, test_neg_u])
    test_item = torch.cat([test_v, test_neg_v])
    test_label = torch.cat([torch.ones(len(test_u)), torch.zeros(len(test_neg_u))])

    return InteractionDataset(train_user, train_item, train_label), InteractionDataset(test_user, test_item, test_label), edge_index, num_nodes

