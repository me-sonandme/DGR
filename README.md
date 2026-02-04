# DGR: A General Graph Desmoothing Framework for Recommendation via Global and Local Perspectives

This repository contains the implementation of our paper **"DGR: A General Graph Desmoothing Framework for Recommendation via Global and Local Perspectives"**.

## Overview

DGR is a general graph de-smoothing framework that addresses the **over-smoothing problem** in Graph Neural Network (GNN)-based recommender systems from two complementary perspectives:

- **Global Perspective**: Layer-wise de-smoothing during graph convolution to preserve node distinctiveness
- **Local Perspective**: Item-item constraint learning to maintain fine-grained collaborative signals

## Requirements

```
Python >= 3.7
PyTorch >= 1.7.0
NumPy >= 1.20.0
SciPy >= 1.6.0
```

## Dataset: MovieLens-1M

We use MovieLens-1M as the example dataset. The data format is:

```
user_id item_id rating
```

Example (`dataset/ml-1M/train.txt`):
```
1 3408 4
1 2355 5
1 1287 5
1 2804 5
...
```

## Quick Start

```bash
cd lgmov
python main.py LightGCN
```

## Configuration

Configuration file: `conf/LightGCN.conf`

```ini
# Dataset (MovieLens-1M)
training.set=./dataset/ml-1M/train.txt
test.set=./dataset/ml-1M/test.txt

# Model
model.name=LightGCN
model.type=graph

# Training
embedding.size=64
num.max.epoch=600
batch_size=2048
learnRate=0.001
reg.lambda=0.0001

# Evaluation
item.ranking=-topN 10,20

# DGR Parameters
LightGCN=-n_layer 3 -layer_a 0.1 -layer_b 1.1 -layer_c 0.0 -lambda_1 1e-2 -lambda_2 5e-6

# Output
output.setup=-dir ./results/
```

## DGR Parameters

### Global De-smoothing (Layer-wise)

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `-n_layer` | Number of GCN layers | 3 | 2-4 |
| `-layer_a` | Layer 1 de-smoothing coefficient | 0.1 | 0.0-0.5 |
| `-layer_b` | Layer 2 de-smoothing coefficient | 1.1 | 0.1-2.0 |
| `-layer_c` | Layer 3 de-smoothing coefficient | 0.0 | 0.0-0.5 |

**Formula:**

$$\mathbf{E}^{(l)} = (1 + \alpha_l) \cdot \tilde{\mathbf{A}} \mathbf{E}^{(l-1)} - \alpha_l \cdot \mathbf{A}_i \mathbf{A}_j \tilde{\mathbf{A}} \mathbf{E}^{(l-1)}$$

- $\alpha = 0$: Standard GCN (no de-smoothing)
- $\alpha > 0$: Suppresses over-smoothing, larger = stronger

### Local De-smoothing (Item-Item Constraints)

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `-lambda_1` | Positive neighbor constraint weight | 1e-2 | 1e-3 ~ 1e-1 |
| `-lambda_2` | Negative neighbor constraint weight | 5e-6 | 1e-7 ~ 1e-5 |

**Loss Function:**

$$\mathcal{L} = \mathcal{L}_{BPR} + \lambda_{reg} \cdot \mathcal{L}_{L2} + \lambda_1 \cdot \mathcal{L}_{pos} - \lambda_2 \cdot \mathcal{L}_{neg}$$

## Code Implementation

### Global De-smoothing (`model/graph/LightGCN.py:221-240`)

```python
# Layer 1
all_emb1 = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
m = torch.sparse.mm(self.aj, all_emb1)
m = torch.sparse.mm(self.ai, m)
all_emb1 = (1 + self.layer_coef_a) * all_emb1 - self.layer_coef_a * m

# Layer 2
all_emb2 = torch.sparse.mm(self.sparse_norm_adj, all_emb1)
m = torch.sparse.mm(self.aj, all_emb2)
m = torch.sparse.mm(self.ai, m)
all_emb2 = (1 + self.layer_coef_b) * all_emb2 - self.layer_coef_b * m

# Layer 3
all_emb3 = torch.sparse.mm(self.sparse_norm_adj, all_emb2)
m = torch.sparse.mm(self.aj, all_emb3)
m = torch.sparse.mm(self.ai, m)
all_emb3 = (1 + self.layer_coef_c) * all_emb3 - self.layer_coef_c * m
```

### Local De-smoothing (`model/graph/LightGCN.py:138-143`)

```python
def create_desmoothing_loss(self, user_idx, pos_idx, rec_user_emb, rec_item_emb):
    loss1 = self.lambda_1 * self.cal_loss_I(user_idx, pos_idx, rec_user_emb, rec_item_emb)
    loss2 = self.lambda_2 * self.cal_loss_item_neg(user_idx, pos_idx, rec_user_emb, rec_item_emb)
    return loss1 - loss2
```

### Ai/Aj Matrix Computation (`data/graph.py:30-38`)

```python
n = adj_mat.shape[0]
m = adj_mat.sum()  # total edges
row = np.array(adj_mat.sum(1)).reshape(-1)  # node degrees

Ai = np.array([np.power(row[i], 0.5) for i in range(n)]).reshape(n, 1)
Aj = np.array([np.power(row[j], 0.5) for j in range(n)]).reshape(1, n) * (1.0 / (2 * m))
```

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DGR Framework                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: User-Item Graph (ml-1M)                             │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Global De-smoothing (Each GCN Layer)                │   │
│  │ E^(l) = (1+α)·Ã·E^(l-1) - α·Ai·Aj·Ã·E^(l-1)        │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  Final Embeddings: E = Mean(E^(0), E^(1), E^(2), E^(3))    │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Local De-smoothing Loss                             │   │
│  │ L_local = λ1·L_pos - λ2·L_neg                       │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  Total Loss: L = L_BPR + L_L2 + L_local                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Output

Results saved to `./results/`:
- `LightGCN@YYYY-MM-DD HH-MM-SS-performance.txt`: Recall@K, NDCG@K metrics

## Contributors

- [me-sonandme](https://github.com/me-sonandme)

## Acknowledgements

This implementation is built upon the following excellent works:

### SELFRec

> **[SELFRec](https://github.com/Coder-Yu/SELFRec)** is a Python framework for self-supervised recommendation (SSR) which integrates commonly used datasets and metrics, and implements many state-of-the-art SSR models. SELFRec has a lightweight architecture and provides user-friendly interfaces. It can facilitate model implementation and evaluation.

### UltraGCN

> **[UltraGCN](https://github.com/xue-pai/UltraGCN)** is an ultra-simplified formulation of graph convolutional networks for collaborative filtering. UltraGCN skips explicit message passing and instead approximates infinite-layer graph convolutions using a constraint loss.




