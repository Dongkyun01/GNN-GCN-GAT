import torch
from sklearn.metrics import roc_auc_score , precision_score, recall_score,f1_score,precision_recall_curve, roc_curve
import json
import os



# 최적 threshold 탐색 + F1 곡선 생성
def find_best_threshold(labels, preds):
    best_f1 = 0.0
    best_threshold = 0.5
    f1_curve = {}
    
    for t in torch.linspace(0.01, 0.99, steps=99):
        binary = (preds >= t).int()
        f1 = f1_score(labels.int(), binary, zero_division=0)
        f1_curve[round(t.item(), 2)] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t.item()
    
    return best_threshold, f1_curve


def eval_gnn(model, test_loader, edge_index, device,save_path="results/gnn.json"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        node_emb = model(edge_index.to(device))
        for batch_user, batch_item, batch_label in test_loader:
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_label = batch_label.float().to(device)

            user_emb = node_emb[batch_user]
            item_emb = node_emb[batch_item]
            pred = (user_emb * item_emb).sum(dim=1).sigmoid()

            all_preds.append(pred.cpu())
            all_labels.append(batch_label.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # ✅ 최적 threshold + F1 곡선
    best_threshold, f1_curve = find_best_threshold(all_labels, all_preds)
    pred_binary = (all_preds >= best_threshold).int()

    # ✅ 지표 계산
    auc = roc_auc_score(all_labels, all_preds)
    hit = (pred_binary == all_labels.int()).float().mean().item()
    precision = precision_score(all_labels.int(), pred_binary, zero_division=0)
    recall = recall_score(all_labels.int(), pred_binary, zero_division=0)

    print(f"[GNN] Eval AUC: {auc:.4f} | Hit@1: {hit:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Best Thresh: {best_threshold:.2f}")

    # 곡선용 데이터
    precisions, recalls, pr_thresholds = precision_recall_curve(all_labels, all_preds)
    fprs, tprs, roc_thresholds = roc_curve(all_labels, all_preds)

# 저장
    os.makedirs("results", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "AUC": float(auc),
            "Hit@1": float(hit),
            "Precision": float(precision),
            "Recall": float(recall),
            "Best_Threshold": float(best_threshold),
            "F1_Curve": f1_curve,
            "PR_Curve": {
                "precision": precisions.tolist(),
                "recall": recalls.tolist()
            },
            "ROC_Curve": {
                "fpr": fprs.tolist(),
                "tpr": tprs.tolist()
            }
        }, f)

    return auc, hit, precision, recall, best_threshold