import matplotlib.pyplot as plt
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')  # 그래프를 띄우지 않고 저장만

model_names = ["gnn", "gcn", "gat"]
results = {}

# 결과 불러오기
for name in model_names:
    path = f"results/{name}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            results[name.upper()] = json.load(f)
    else:
        print(f"[⚠️ Warning] {path} not found")

models = list(results.keys())

# ──────────────────────────────────────────────
# [1] 막대그래프 성능 비교
# ──────────────────────────────────────────────
auc_scores = [results[m]["AUC"] for m in models]
hit_scores = [results[m]["Hit@1"] for m in models]
precision_scores = [results[m]["Precision"] for m in models]
recall_scores = [results[m]["Recall"] for m in models]

x = np.arange(len(models))
width = 0.3

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# AUC
axes[0, 0].bar(x, auc_scores, width, color='skyblue')
axes[0, 0].set_title("AUC Comparison")
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(models)
axes[0, 0].set_ylim(0.0, 1.1)
axes[0, 0].set_ylabel("AUC")

# Hit@1
axes[0, 1].bar(x, hit_scores, width, color='lightgreen')
axes[0, 1].set_title("Hit@1 Comparison")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(models)
axes[0, 1].set_ylim(0.0, 1.1)
axes[0, 1].set_ylabel("Hit@1")

# Precision
axes[1, 0].bar(x, precision_scores, width, color='salmon')
axes[1, 0].set_title("Precision Comparison")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(models)
axes[1, 0].set_ylim(0.0, 1.1)
axes[1, 0].set_ylabel("Precision")

# Recall
axes[1, 1].bar(x, recall_scores, width, color='orange')
axes[1, 1].set_title("Recall Comparison")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models)
axes[1, 1].set_ylim(0.0, 1.1)
axes[1, 1].set_ylabel("Recall")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/full_comparison.png")

print("✅ full_comparison.png 저장 완료")

# ──────────────────────────────────────────────
# [2] F1-Score 변화 곡선 그래프 (모델별)
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
for m in models:
    f1_dict = results[m].get("F1_Curve", None)
    if f1_dict:
        thresholds = list(map(float, f1_dict.keys()))
        f1_scores = list(f1_dict.values())
        plt.plot(thresholds, f1_scores, label=m)

plt.title("Threshold vs F1-Score Curve")
plt.xlabel("Threshold")
plt.ylabel("F1-Score")
plt.ylim(0.0, 1.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/f1_curve_comparison.png")

print("✅ f1_curve_comparison.png 저장 완료")


# Precision-Recall Curve
# ────────────────
plt.figure(figsize=(8, 6))
for m in results:
    pr = results[m].get("PR_Curve", None)
    if pr:
        plt.plot(pr["recall"], pr["precision"], label=m)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/precision_recall_curve.png")
print("✅ Precision-Recall Curve 저장 완료")

# ────────────────
# ROC Curve
# ────────────────
plt.figure(figsize=(8, 6))
for m in results:
    roc = results[m].get("ROC_Curve", None)
    if roc:
        plt.plot(roc["fpr"], roc["tpr"], label=m)
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/roc_curve.png")
print("✅ ROC Curve 저장 완료")