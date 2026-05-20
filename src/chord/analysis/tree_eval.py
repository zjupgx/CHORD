from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, to_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
from datetime import datetime
from .evaluation import transfer_labels_across_species


def core_type_label(name: str) -> str:
    """
    Convert a label to its core cell-type name by:
    1. removing the species prefix before the first underscore
    2. removing a trailing numeric suffix such as '_1' or '1'
    3. normalizing internal whitespace
    """
    s = name.strip()
    s = re.sub(r"^[^_]+_", "", s)
    s = re.sub(r"_?\d+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tag_labels(names: Sequence[str]) -> tuple[list[str], list[int], dict[str, int], list[int]]:
    """
    Map raw labels to normalized labels and integer ids.

    Parameters
    ----------
    names
        Raw label names.

    Returns
    -------
    labels
        Normalized labels.
    ids
        Integer ids for non-NA labels, in original order with NA entries skipped.
    label2id
        Mapping from normalized label to integer id.
    nas
        Indices of entries whose normalized label is 'NA'.
    """
    labels = [core_type_label(n) for n in names]
    label2id: dict[str, int] = {}
    ids: list[int] = []
    nas: list[int] = []

    for i, lab in enumerate(labels):
        if lab == "NA":
            nas.append(i)
            continue
        if lab not in label2id:
            label2id[lab] = len(label2id)
        ids.append(label2id[lab])

    return labels, ids, label2id, nas


def reorder_embeddings(
    emb: np.ndarray,
    row_labels: Sequence[str],
    label2idx: dict[str, int],
) -> np.ndarray:
    """
    Reorder embedding rows to match a master label order.
    """
    order = [label2idx[l] for l in row_labels]
    return emb[order]

def consensus_linkage_wo_na(
    embeddings_list: Sequence[np.ndarray],
    labels_master: Sequence[str],
    row_labels_list: Sequence[Sequence[str]] | None = None,
    method: str = "ward",
    nas: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Build a consensus linkage matrix from multiple embedding snapshots.

    Each embedding is:
    1. optionally reordered to the master label order
    2. filtered to remove NA rows
    3. converted to cosine-based Euclidean distance
    4. averaged across snapshots

    Parameters
    ----------
    embeddings_list
        List of embedding matrices with shape (n_labels, d).
    labels_master
        Master row labels.
    row_labels_list
        Optional row labels for each embedding matrix.
    method
        Linkage method for scipy.cluster.hierarchy.linkage.
    nas
        Indices in labels_master to exclude, e.g. NA entries.

    Returns
    -------
    Z_main
        Linkage matrix computed from the averaged distance matrix.
    """
    n_labels = len(labels_master)
    label2idx_master = {lab: i for i, lab in enumerate(labels_master)}

    if nas is None:
        idx_keep = list(range(n_labels))
    else:
        nas_set = set(nas)
        idx_keep = [i for i in range(n_labels) if i not in nas_set]

    D_sum = None

    for r, emb in enumerate(embeddings_list):
        emb_r = emb
        if row_labels_list is not None:
            emb_r = reorder_embeddings(emb_r, row_labels_list[r], label2idx_master)

        emb_r = emb_r[idx_keep, :]
        euc = np.sqrt(2.0 * cosine_distances(emb_r))

        if D_sum is None:
            D_sum = euc
        else:
            D_sum = D_sum + euc

    D_avg = D_sum / len(embeddings_list)
    Z_main = linkage(squareform(D_avg, checks=False), method=method)
    return Z_main

def evaluate_tree_clustering(
    Z: np.ndarray,
    truth_ids: Sequence[int],
    n_clusters: int,
) -> dict[str, float | np.ndarray]:
    """
    Cut a hierarchical tree into n_clusters and compare to truth ids.

    Parameters
    ----------
    Z
        Linkage matrix.
    truth_ids
        Ground-truth integer labels for non-NA entries.
    n_clusters
        Number of clusters used in fcluster(..., criterion='maxclust').

    Returns
    -------
    result
        Dictionary containing predicted cluster labels, NMI and ARI.
    """
    pred_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    nmi = normalized_mutual_info_score(list(pred_labels), list(truth_ids))
    ari = adjusted_rand_score(list(pred_labels), list(truth_ids))

    return {
        "pred_labels": pred_labels,
        "nmi": float(nmi),
        "ari": float(ari),
    }

def evaluate_consensus_embedding_tree(
    embeddings_list: Sequence[np.ndarray],
    all_type_labels: Sequence[str],
    n_clusters: int,
    *,
    row_labels_list: Sequence[Sequence[str]] | None = None,
    method: str = "ward",
) -> dict[str, object]:
    """
    Full pipeline for multiple embedding snapshots:
    normalize labels -> remove NA -> consensus tree -> evaluate clustering.
    """
    _, truth_ids, label2id, nas = tag_labels(all_type_labels)

    Z = consensus_linkage_wo_na(
        embeddings_list=embeddings_list,
        labels_master=all_type_labels,
        row_labels_list=row_labels_list,
        method=method,
        nas=nas,
    )
    metrics = evaluate_tree_clustering(Z, truth_ids=truth_ids, n_clusters=n_clusters)

    return {
        "label2id": label2id,
        "nas": nas,
        **metrics,
    }

def eval_transfer_row(adata, species_src, species_tgt, model_name="", run_tag=None, include_confusion=False):
    adata_sub = adata[adata.obs["cell_type"] != "NA"].copy()
    res = transfer_labels_across_species(
    adata_sub, species_src, species_tgt, return_confusion_matrix=include_confusion)
    row = {
    'timestamp': datetime.now().isoformat(timespec='seconds'),
    'model': model_name,
    'direction': f'{species_tgt} to {species_src}',
    'run_tag': run_tag,
    'src': species_src,
    'tgt': species_tgt,
    'n_cells': adata_sub.n_obs,
    'accuracy': res.get('accuracy'),
    'f1_micro': res['f1'].get('micro'),
    'f1_macro': res['f1'].get('macro'),
    'f1_weighted': res['f1'].get('weighted'),
    }
    cm = res.get('confusion_matrix') if include_confusion else None
    return row, cm