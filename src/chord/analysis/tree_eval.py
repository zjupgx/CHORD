from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .tree import _prepare_normalized_embeddings


def consensus_linkage_wo_na(
    embeddings_list: Sequence[np.ndarray],
    labels_master: Sequence[str],
    row_labels_list: Sequence[Sequence[str]] | None = None,
    method: str = "ward",
    nas: Sequence[int] | None = None,
) -> np.ndarray:
    n_labels = len(labels_master)

    if nas is None:
        idx_keep = list(range(n_labels))
    else:
        nas_set = set(nas)
        idx_keep = [i for i in range(n_labels) if i not in nas_set]

    if len(idx_keep) < 2:
        raise ValueError("At least two non-NA labels are required to build a tree.")
    filtered_embeddings = _prepare_normalized_embeddings(
        embeddings_list,
        labels_master,
        row_labels_list=row_labels_list,
        row_indices=idx_keep,
    )
    joint_embedding = (
        np.concatenate(filtered_embeddings, axis=1)
        / np.sqrt(float(len(filtered_embeddings)))
    )
    Z_main = linkage(joint_embedding, method=method, metric="euclidean")
    return Z_main

def evaluate_tree_clustering(
    Z: np.ndarray,
    truth_ids: Sequence[int],
    n_clusters: int,
) -> dict[str, float | np.ndarray]:
    pred_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    nmi = normalized_mutual_info_score(list(pred_labels), list(truth_ids))
    ari = adjusted_rand_score(list(pred_labels), list(truth_ids))

    return {
        "pred_labels": pred_labels,
        "nmi": float(nmi),
        "ari": float(ari),
    }

