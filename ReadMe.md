
#  Graph Neural Network Models for Link Prediction (GNN vs GCN vs GAT)

이 프로젝트는 **Graph Neural Network (GNN)** 기반의 다양한 모델(GNN, GCN, GAT)을 활용하여 **사용자-아이템 간 링크 예측 문제**를 해결하고, 각 모델의 성능을 비교하는 것을 목표로 합니다.

---

##  프로젝트 목표

- MovieLens 100K 데이터셋을 기반으로, **사용자-아이템 상호작용**을 그래프로 모델링
- GNN, GCN, GAT 세 가지 그래프 신경망 모델을 구축하고 **링크 예측 문제** 해결
- 모델 성능을 AUC, Precision, Recall, Hit@1, F1-Score 등의 다양한 지표로 평가
- **임계값(threshold)에 따른 F1 곡선, PR Curve, ROC Curve** 등을 시각화

---

##  데이터셋

- **사용한 데이터**: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- **노드 구성**: 
  - 사용자 노드: `user_id`
  - 아이템 노드: `item_id`
- **엣지 구성**: 
  - 사용자-아이템 간의 평점 기록 (양방향 처리)
- **전처리 내용**:
  - 사용자/아이템 ID → 정수 인코딩(Label Encoding)
  - 아이템 노드는 사용자 노드 수만큼 index offset 적용
  - 전체 데이터를 **train/test**로 분할
  - 각 데이터셋에 대해 **negative sampling** 적용 (존재하지 않는 링크)

---

##  모델 구성

###  1. Simple GNN
- Mean Aggregation 기반의 가장 기본적인 GNN 구조
- 1-layer 구조

###  2. GCN (Graph Convolutional Network)
- Kipf & Welling의 GCN 논문 기반 2-layer 구조
- Degree normalization, self-loop 포함

###  3. GAT (Graph Attention Network)
- Attention coefficient를 통해 이웃 가중치 학습
- 2-layer 구조 (단일 헤드)

---

##  학습

- 임베딩 차원: `64`
- Optimizer: `Adam`
- Loss: Binary Cross Entropy Loss (`BCEWithLogits`)
- Epochs: 10
- 학습 루프에서는 배치 기반 학습 및 PyTorch `DataLoader` 사용

---

##  평가 지표

| 지표 | 설명 |
|------|------|
| AUC (ROC-AUC) | 모델의 분류 구분 능력 |
| Hit@1 | 상위 1개의 예측 결과에 정답이 포함될 확률 |
| Precision | 정답으로 예측한 것 중 실제 정답인 비율 |
| Recall | 전체 정답 중에서 모델이 맞춘 비율 |
| Best Threshold | F1-score 기준으로 가장 성능이 좋았던 threshold 값 |
| F1 Curve | threshold에 따른 F1-score 변화 시각화 |
| PR Curve | Precision vs Recall 곡선 |
| ROC Curve | TPR vs FPR 곡선 |

---

##  시각화 결과

- `results/` 폴더 내에 저장된 `.json` 파일을 기반으로 다음 그래프들을 자동 생성:

### 🔹 모델별 지표 비교 (Bar Chart)
- AUC, Hit@1, Precision, Recall

### 🔹 Threshold vs F1-Score Curve
- 최적의 임계값을 시각적으로 확인 가능

### 🔹 Precision-Recall Curve

### 🔹 ROC Curve

---

##  폴더 구조

```

.
├── models/               # 모델 모음 폴더
│   ├── gnn_model.py      # GNN Model
│   ├── gcn_model.py      # GCN Model
│   └── gat_model.py      # GAT Model
├── train/                # 학습 루프 모음 폴더
│   ├── train_gnn.py      # GNN 학습 루프
│   ├── train_gcn.py      # GCN 학습 루프
│   └── train_gat.py      # GAT 학습 루프
├── eval/                 # 평가 함수 모음 폴더
│   ├── eval_gnn.py       # GNN 평가 함수
│   ├── eval_gcn.py       # GCN 평가 함수
│   └── eval_gat.py       # GAT 평가 함수
├── run/                  # 학습 및 평가 실행 모음 폴더
│   ├── run_gnn.py        # GNN 학습 및 평가 실행
│   ├── run_gcn.py        # GCN 학습 및 평가 실행
│   └── run_gat.py        # GAT 학습 및 평가 실행
├── dataset.py            # 데이터 로딩 및 전처리
├── utils.py              # seed 설정
├── compare_results.py    # 모든 모델 결과 및 비교 시각화
└── results/
    ├── gnn.json
    ├── gcn.json
    ├── gat.json
    └── \*.png (그래프 이미지들)

```
##  결론 및 인사이트

* **GAT 모델이 전반적으로 가장 우수한 성능**을 보여줌 (AUC, Recall, PR/ROC Curve)
* GNN과 GCN도 경량 구조로서 실용적인 성능을 제공
* Attention 메커니즘이 사용자-아이템 간 중요한 연결을 더 잘 학습함을 확인

---

