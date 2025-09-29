from __future__ import annotations

import numpy as np


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_score = y_score.astype(np.float64)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.argsort(np.argsort(y_score)) + 1
    pos_ranks = ranks[y_true == 1]
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


__all__ = ["roc_auc_score"]
