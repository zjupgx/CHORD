import warnings
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


def summarize_cell_types(
    species_adatas: Dict[str, Tuple[AnnData, str]]
) -> Tuple[Set[str], pd.DataFrame]:
    """
    Summarize shared and species-specific cell types.

    Parameters
    ----------
    species_adatas
        Mapping: {"species": (AnnData, label_column)}

    Returns
    -------
    shared_types
        Cell types shared across all species.
    summary_df
        DataFrame of species-specific cell types with columns:
        ['species', 'cell_type', 'n_cells']
    """
    species_to_types = {}
    count_tables = []

    for species, (adata, label_col) in species_adatas.items():
        labels = adata.obs[label_col]
        species_to_types[species] = set(labels.unique())

        counts = labels.value_counts().rename("n_cells")
        counts = counts.rename_axis("cell_type").reset_index()
        counts["species"] = species
        count_tables.append(counts)

    shared_types = set.intersection(*species_to_types.values())

    counts_df = pd.concat(count_tables, ignore_index=True)
    summary_df = counts_df.loc[
        ~counts_df["cell_type"].isin(shared_types),
        ["species", "cell_type", "n_cells"],
    ]

    return shared_types, summary_df


def osr_split_simple(
    species_adatas: Dict[str, Tuple[AnnData, str]],
    shared_types: Set[str],
    keep_specific: Dict[str, Sequence[str]],
    ratio_known_train: float = 0.6,
) -> Dict[str, Dict[str, Tuple[AnnData, str]]]:
    """
    Split each species into train / test for open-set evaluation.

    Known cell types are split into train and test.
    Unknown cell types are assigned to test only.

    Returns
    -------
    splits
        {
            "train": {"fish": (adata_train, label_col), ...},
            "test":  {"fish": (adata_test, label_col), ...},
        }
    """
    rng = np.random.default_rng(0)
    splits = {split: {} for split in ("train", "test")}

    for species, (adata, label_col) in species_adatas.items():
        ad = adata.copy()
        ad.obs["split"] = "test"
        ad.obs["label_unk"] = ad.obs[label_col].astype(str)

        known_types = set(shared_types) | set(keep_specific.get(species, []))
        unknown_types = set(ad.obs[label_col].unique()) - known_types

        print(f"[{species}] known={len(known_types)}, unknown={len(unknown_types)}")

        split_col = ad.obs.columns.get_loc("split")

        for cell_type in known_types:
            idx = np.where(ad.obs[label_col] == cell_type)[0]
            rng.shuffle(idx)

            n = len(idx)
            n_train = int(n * ratio_known_train)

            ad.obs.iloc[idx[:n_train], split_col] = "train"

        is_train = ad.obs["split"] == "train"
        is_unknown = ad.obs[label_col].isin(unknown_types)

        ad.obs.loc[~is_train & is_unknown, "label_unk"] = "unknown"
        ad.obs.loc[is_train & is_unknown, "label_unk"] = "IGNORE"

        for split in splits:
            splits[split][species] = (ad[ad.obs["split"] == split].copy(), label_col)

        print(
            f"[{species}] train/test = "
            f"{[(ad.obs['split'] == s).sum() for s in ('train', 'test')]}"
        )

    return splits


def collect_label_ids(
    global_le,
    shared_types: Set[str],
    keep_specific: Dict[str, Sequence[str]],
    *,
    unique: bool = True,
    sort_ids: bool = True,
) -> List[str]:
    """
    Collect shared and kept species-specific labels.

    Parameters
    ----------
    global_le
        Kept for interface compatibility.
    shared_types
        Shared cell-type labels.
    keep_specific
        Species-specific labels to keep.
    unique
        Whether to remove duplicates.
    sort_ids
        Kept for interface compatibility.

    Returns
    -------
    labels
        List of label names.
    """
    labels = list(shared_types)
    for label_list in keep_specific.values():
        labels.extend(label_list)

    if unique:
        labels = list(dict.fromkeys(labels))

    return labels


def open_curve_logistic(
    adata_train,
    adata_test,
    known_id_list,
    obsm_key: str = "X_pca",
    n_tau: int = 200,
    label_key: str = "celltype",
    unknown_id: str = "Unknown",
    solver: str = "lbfgs",
    C: float = 1.0,
    max_iter: int = 200,
    class_weight=None,
):
    """
    Open-set evaluation with multinomial logistic regression.
    """
    X_train = adata_train.obsm[obsm_key]
    y_train_raw = adata_train.obs[label_key].to_numpy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver=solver,
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        n_jobs=None,
    )
    clf.fit(X_train, y_train)

    X_test = adata_test.obsm[obsm_key]
    y_true_raw = adata_test.obs[label_key].to_numpy()

    prob = clf.predict_proba(X_test)
    margin = prob.max(axis=1)
    top1_idx = prob.argmax(axis=1)
    pred_labels = le.inverse_transform(top1_idx)

    known_arr = np.asarray(known_id_list)
    is_known = np.isin(y_true_raw, known_arr)

    y_true_ood = (~is_known).astype(int)
    y_score_ood = 1.0 - margin

    if y_true_ood.min() != y_true_ood.max():
        ood_auroc = roc_auc_score(y_true_ood, y_score_ood)
        ood_aupr = average_precision_score(y_true_ood, y_score_ood)
    else:
        ood_auroc = np.nan
        ood_aupr = np.nan

    tau_list = np.linspace(0.0, 1.0, n_tau)

    acc_known_list = []
    f1_macro_known_list = []
    oscr_points = []

    for tau in tau_list:
        y_pred = np.where(margin >= tau, pred_labels, unknown_id)

        mask_known = is_known
        acc_known = (y_pred[mask_known] == y_true_raw[mask_known]).mean()
        acc_known_list.append(acc_known)

        f1_known = f1_score(
            y_true_raw[mask_known],
            y_pred[mask_known],
            labels=known_id_list,
            average="macro",
            zero_division=0,
        )
        f1_macro_known_list.append(f1_known)

        ccr_known = np.mean(
            (margin[mask_known] >= tau)
            & (pred_labels[mask_known] == y_true_raw[mask_known])
        )

        mask_unknown = ~is_known
        fpr_unknown = np.mean(margin[mask_unknown] >= tau) if mask_unknown.any() else np.nan
        oscr_points.append((fpr_unknown, ccr_known))

    return {
        "tau_list": np.asarray(tau_list),
        "acc_known_list": np.asarray(acc_known_list),
        "f1_macro_known_list": np.asarray(f1_macro_known_list),
        "ood_auroc": float(ood_auroc),
        "ood_aupr": float(ood_aupr),
        "oscr_points": np.asarray(oscr_points),
    }


def open_curve_acc(
    adata,
    species_Q: str,
    species_R: str,
    known_id_list,
    n_tau: int = 200,
    label_key: str = "cell_type",
    item_key: str = "item_type",
    unknown_id: str = "Unknown",
    return_style: str = "dict",
):
    """
    Open-set evaluation with cell-type-embedding-based classification.
    """
    q_mask = (adata.obs["species"] == species_Q) & (adata.obs[item_key] == "cell")
    r_mask = (adata.obs["species"] == species_R) & (adata.obs[item_key] == "celltype")

    if q_mask.sum() == 0:
        warnings.warn(f"[{species_Q}] no cell data found")
        if return_style == "legacy":
            return (np.array([]),) * 4
        return {
            "tau_list": np.array([]),
            "acc_known_list": np.array([]),
            "f1_macro_known_list": np.array([]),
            "ood_auroc": np.nan,
            "ood_aupr": np.nan,
            "oscr_points": np.empty((0, 2)),
        }

    if r_mask.sum() == 0:
        warnings.warn(f"[{species_R}] no cell-type embedding data found")
        if return_style == "legacy":
            return (np.array([]),) * 4
        return {
            "tau_list": np.array([]),
            "acc_known_list": np.array([]),
            "f1_macro_known_list": np.array([]),
            "ood_auroc": np.nan,
            "ood_aupr": np.nan,
            "oscr_points": np.empty((0, 2)),
        }

    X_q = np.asarray(adata[q_mask].X)
    P_r = np.asarray(adata[r_mask].X)
    y_q = adata.obs.loc[q_mask, label_key].to_numpy()
    P_name = adata[r_mask].obs["cell_type"].to_numpy()

    sim = cosine_similarity(X_q, P_r)
    logits = sim / 0.1
    prob = softmax(logits, axis=1)

    top1_idx = prob.argmax(axis=1)
    margin = prob[np.arange(prob.shape[0]), top1_idx]
    pred_best_label = P_name[top1_idx]

    known_arr = np.asarray(known_id_list)
    is_known = np.isin(y_q, known_arr)

    y_true_ood = (~is_known).astype(int)
    y_score_ood = 1.0 - margin

    if y_true_ood.min() != y_true_ood.max():
        ood_auroc = roc_auc_score(y_true_ood, y_score_ood)
        ood_aupr = average_precision_score(y_true_ood, y_score_ood)
    else:
        ood_auroc = np.nan
        ood_aupr = np.nan

    tau_list = np.linspace(0.0, 1.0, n_tau)

    acc_known_list = []
    f1_macro_known_list = []
    oscr_points = []

    for tau in tau_list:
        y_pred = np.where(margin >= tau, pred_best_label, unknown_id)

        mask_known = is_known
        acc_known = (y_pred[mask_known] == y_q[mask_known]).mean() if mask_known.any() else np.nan
        acc_known_list.append(acc_known)

        f1_known = (
            f1_score(
                y_q[mask_known],
                y_pred[mask_known],
                labels=known_id_list,
                average="macro",
                zero_division=0,
            )
            if mask_known.any()
            else np.nan
        )
        f1_macro_known_list.append(f1_known)

        ccr_known = (
            np.mean((margin[mask_known] >= tau) & (pred_best_label[mask_known] == y_q[mask_known]))
            if mask_known.any()
            else np.nan
        )

        mask_unknown = ~is_known
        fpr_unknown = np.mean(margin[mask_unknown] >= tau) if mask_unknown.any() else np.nan
        oscr_points.append((fpr_unknown, ccr_known))

    if return_style == "legacy":
        ood_auc_list = np.full_like(
            np.asarray(acc_known_list, dtype=float),
            fill_value=ood_auroc,
            dtype=float,
        )
        return (
            np.asarray(tau_list),
            np.asarray(acc_known_list),
            ood_auc_list,
            np.asarray(f1_macro_known_list),
        )

    return {
        "tau_list": np.asarray(tau_list),
        "acc_known_list": np.asarray(acc_known_list),
        "f1_macro_known_list": np.asarray(f1_macro_known_list),
        "ood_auroc": float(ood_auroc),
        "ood_aupr": float(ood_aupr),
        "oscr_points": np.asarray(oscr_points),
    }


def open_compact(
    adata,
    label_col,
    X=None,
    resolution_range=None,
    leiden_key: str = "leiden_temp",
):
    """
    Search for the best Leiden resolution.

    If the ground-truth labels contain at least two classes, ARI is used
    as the main selection criterion. Otherwise, clustering quality is used.
    """
    Xmat = adata.obsm[X] if X is not None else (adata.X.A if hasattr(adata.X, "A") else adata.X)
    y_true_cat = adata.obs[label_col].astype("category")
    y_true_codes = y_true_cat.cat.codes.values
    n_true_classes = y_true_cat.cat.categories.size

    if resolution_range is None:
        resolution_range = np.arange(0.1, 2.05, 0.1)

    best = {
        "Resolution": None,
        "ARI": None,
        "NMI": None,
        "Silhouette": None,
        "DB": None,
    }

    best_ari = -np.inf if n_true_classes >= 2 else None

    for res in resolution_range:
        sc.tl.leiden(adata, resolution=res, key_added=leiden_key)
        y_pred = adata.obs[leiden_key].to_numpy()

        unique_pred = np.unique(y_pred)
        n_clusters = unique_pred.size

        sil = None
        db = None
        if n_clusters >= 2 and all((y_pred == u).sum() >= 2 for u in unique_pred):
            sil = float(silhouette_score(Xmat, y_pred))
            db = float(davies_bouldin_score(Xmat, y_pred))

        if n_true_classes >= 2:
            y_pred_codes = pd.Categorical(y_pred).codes
            ari = float(adjusted_rand_score(y_true_codes, y_pred_codes))
            nmi = float(normalized_mutual_info_score(y_true_codes, y_pred_codes))

            if ari > best_ari:
                best_ari = ari
                best.update(
                    {
                        "Resolution": float(res),
                        "ARI": ari,
                        "NMI": nmi,
                        "Silhouette": sil,
                        "DB": db,
                    }
                )
        else:
            score = sil if sil is not None else (-db if db is not None else None)
            prev_score = (
                best["Silhouette"]
                if best["Silhouette"] is not None
                else (-best["DB"] if best["DB"] is not None else None)
            )

            if score is not None and (prev_score is None or score > prev_score):
                best.update(
                    {
                        "Resolution": float(res),
                        "ARI": None,
                        "NMI": None,
                        "Silhouette": sil,
                        "DB": db,
                    }
                )

    if leiden_key in adata.obs.columns:
        del adata.obs[leiden_key]

    return best
