import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


def _macro_known_ccr(
    y_true: np.ndarray,
    pred_labels: np.ndarray,
    margin: np.ndarray,
    mask_known: np.ndarray,
    threshold: float,
) -> float:
    known_type_ccrs = []
    for label in pd.unique(y_true[mask_known]):
        type_mask = mask_known & (y_true == label)
        known_type_ccrs.append(
            np.mean(
                (margin[type_mask] >= threshold)
                & (pred_labels[type_mask] == y_true[type_mask])
            )
        )
    return float(np.mean(known_type_ccrs)) if known_type_ccrs else float("nan")


def _empty_open_curve_result(return_style: str):
    if return_style == "legacy":
        return (np.array([]),) * 4
    return {
        "tau_list": np.array([]),
        "acc_known_list": np.array([]),
        "f1_macro_known_list": np.array([]),
        "known_macro_ccr_list": np.array([]),
        "unknown_recall_list": np.array([]),
        "unknown_macro_recall_list": np.array([]),
        "ood_auroc": np.nan,
        "ood_aupr": np.nan,
        "oscr_points": np.empty((0, 2)),
    }


def _selected_embedding_matrix(adata, selection_mask, representation: str):
    if representation == "X":
        values = adata.X[selection_mask]
    else:
        if representation not in adata.obsm:
            raise KeyError(
                f"Embedding representation {representation!r} is absent from "
                "adata.obsm."
            )
        values = adata.obsm[representation][selection_mask]

    if hasattr(values, "toarray"):
        values = values.toarray()
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(
            f"Embedding representation {representation!r} must be two-dimensional."
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(
            f"Embedding representation {representation!r} must be numeric."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"Embedding representation {representation!r} contains non-finite values."
        )
    return matrix


def _cell_selection_mask(
    adata,
    species: str,
    *,
    species_key: str,
    item_key: str | None,
    cell_item_value: str,
):
    if species_key not in adata.obs:
        raise KeyError(f"OSR embedding is missing obs column {species_key!r}.")
    mask = adata.obs[species_key].astype(str).to_numpy() == species
    if item_key is not None and item_key in adata.obs:
        mask &= (
            adata.obs[item_key].astype(str).to_numpy() == cell_item_value
        )
    return mask


def _base_open_curve_cache(
    adata,
    query_mask,
    *,
    species_Q: str,
    species_R: str,
    label_key: str,
    known_status_key: str,
    split_key: str,
):
    if label_key not in adata.obs:
        raise KeyError(f"OSR embedding is missing obs column {label_key!r}.")
    return {
        "query_species": species_Q,
        "reference_species": species_R,
        "known_status_key": known_status_key,
        "split_key": split_key,
        "query_obs_names": np.asarray(adata.obs_names[query_mask]).copy(),
        "y_true": adata.obs.loc[query_mask, label_key].astype(str).to_numpy(),
        "known_status": (
            adata.obs.loc[query_mask, known_status_key].astype(str).to_numpy()
            if known_status_key in adata.obs
            else None
        ),
        "split": (
            adata.obs.loc[query_mask, split_key].astype(str).to_numpy()
            if split_key in adata.obs
            else None
        ),
    }


def _validate_prototype_temperature(value, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a finite number greater than 0.")
    return float(value)


def _validated_species_prototypes(
    adata,
    species: str,
    *,
    representation: str,
    species_key: str,
    label_key: str,
    train_label_key: str | None,
    item_key: str | None,
    cell_item_value: str,
    prototype_item_value: str,
    known_status_key: str,
    split_key: str,
    require_known_train_source: bool,
):
    if item_key is None or item_key not in adata.obs:
        raise KeyError(
            "Prototype OSR evaluation requires an item-type obs column; "
            f"configured item_type_key={item_key!r}."
        )
    required = {species_key, label_key}
    missing = required - set(adata.obs.columns)
    if missing:
        raise KeyError(
            "OSR embedding is missing required prototype obs columns: "
            f"{sorted(missing)}"
        )

    prototype_mask = (
        adata.obs[species_key].astype(str).to_numpy() == species
    ) & (
        adata.obs[item_key].astype(str).to_numpy() == prototype_item_value
    )
    if not prototype_mask.any():
        raise ValueError(
            f"[{species}] no {prototype_item_value!r} prototypes are available."
        )

    prototype_label_values = adata.obs.loc[prototype_mask, label_key]
    if prototype_label_values.isna().any():
        raise ValueError(f"[{species}] prototype labels contain missing values.")
    prototype_labels = prototype_label_values.astype(str).to_numpy()
    duplicate_labels = sorted(
        pd.Series(prototype_labels)[pd.Series(prototype_labels).duplicated()]
        .unique()
        .tolist()
    )
    if duplicate_labels:
        raise ValueError(
            f"[{species}] prototype labels must be unique; duplicates="
            f"{duplicate_labels}."
        )

    if require_known_train_source:
        metadata_required = {known_status_key, split_key}
        metadata_missing = metadata_required - set(adata.obs.columns)
        if metadata_missing:
            raise KeyError(
                "Prototype mapping requires prototype-training metadata "
                f"columns: {sorted(metadata_missing)}"
            )
        prototype_status = set(
            adata.obs.loc[prototype_mask, known_status_key].astype(str)
        )
        if prototype_status != {"known"}:
            raise ValueError(
                f"[{species}] prototypes must all have "
                f"{known_status_key!r}='known'; found={sorted(prototype_status)}."
            )
        prototype_splits = set(
            adata.obs.loc[prototype_mask, split_key].astype(str)
        )
        if prototype_splits != {"prototype"}:
            raise ValueError(
                f"[{species}] query prototype rows must all have "
                f"{split_key!r}='prototype'; found={sorted(prototype_splits)}."
            )

        training_mask = _cell_selection_mask(
            adata,
            species,
            species_key=species_key,
            item_key=item_key,
            cell_item_value=cell_item_value,
        )
        training_mask &= (
            adata.obs[known_status_key].astype(str).to_numpy() == "known"
        )
        training_mask &= (
            adata.obs[split_key].astype(str).to_numpy() == "train"
        )
        if not training_mask.any():
            raise ValueError(
                f"[{species}] prototype mapping requires known/train cells "
                "to verify the query prototype set."
            )
        resolved_train_label_key = train_label_key
        if resolved_train_label_key is None:
            resolved_train_label_key = (
                "train_label" if "train_label" in adata.obs else label_key
            )
        if resolved_train_label_key not in adata.obs:
            raise KeyError(
                f"[{species}] prototype training label column is absent: "
                f"{resolved_train_label_key!r}."
            )
        training_label_values = adata.obs.loc[
            training_mask, resolved_train_label_key
        ]
        if training_label_values.isna().any():
            raise ValueError(
                f"[{species}] known/train labels in "
                f"{resolved_train_label_key!r} contain missing values."
            )
        training_types = set(training_label_values.astype(str))
        unsupported_prototypes = sorted(set(prototype_labels) - training_types)
        if unsupported_prototypes:
            raise ValueError(
                f"[{species}] prototypes are not backed by known/train "
                f"cell types: {unsupported_prototypes}."
            )

    prototype_embeddings = _selected_embedding_matrix(
        adata, prototype_mask, representation
    )
    return prototype_embeddings, prototype_labels


def prepare_query_prototype_probability_cache(
    adata,
    query_species: str,
    *,
    temperature: float = 0.1,
    representation: str = "X",
    species_key: str = "species",
    label_key: str = "cell_type",
    train_label_key: str | None = None,
    item_key: str | None = "item_type",
    cell_item_value: str = "cell",
    prototype_item_value: str = "celltype",
    known_status_key: str = "known_status",
    split_key: str = "split",
):
    temperature = _validate_prototype_temperature(
        temperature, "Prototype cell temperature"
    )
    query_mask = _cell_selection_mask(
        adata,
        query_species,
        species_key=species_key,
        item_key=item_key,
        cell_item_value=cell_item_value,
    )
    base_cache = _base_open_curve_cache(
        adata,
        query_mask,
        species_Q=query_species,
        species_R=query_species,
        label_key=label_key,
        known_status_key=known_status_key,
        split_key=split_key,
    )
    if not query_mask.any():
        raise ValueError(f"[{query_species}] no query cells are available.")

    prototype_embeddings, prototype_labels = _validated_species_prototypes(
        adata,
        query_species,
        representation=representation,
        species_key=species_key,
        label_key=label_key,
        train_label_key=train_label_key,
        item_key=item_key,
        cell_item_value=cell_item_value,
        prototype_item_value=prototype_item_value,
        known_status_key=known_status_key,
        split_key=split_key,
        require_known_train_source=True,
    )
    query_embeddings = _selected_embedding_matrix(
        adata, query_mask, representation
    )
    probability = softmax(
        cosine_similarity(query_embeddings, prototype_embeddings) / temperature,
        axis=1,
    )
    return {
        **base_cache,
        "prototype_probability": probability,
        "prototype_embeddings": prototype_embeddings,
        "prototype_labels": prototype_labels,
    }


def prepare_mapped_prototype_cache(query_cache, reference_cache):
    query_species = query_cache["query_species"]
    reference_species = reference_cache["query_species"]
    query_probability = np.asarray(query_cache["prototype_probability"])
    query_prototypes = np.asarray(query_cache["prototype_embeddings"])
    query_labels = np.asarray(query_cache["prototype_labels"])
    reference_prototypes = np.asarray(reference_cache["prototype_embeddings"])
    reference_labels = np.asarray(reference_cache["prototype_labels"])

    if query_species == reference_species:
        if not np.array_equal(
            query_labels.astype(str), reference_labels.astype(str)
        ):
            raise ValueError(
                f"[{query_species}] self-mapping prototype labels do not match."
            )
        reference_probability = query_probability
    else:
        if query_prototypes.shape[1] != reference_prototypes.shape[1]:
            raise ValueError(
                f"[{query_species}->{reference_species}] prototype embedding "
                "dimensions do not match: "
                f"{query_prototypes.shape[1]} != {reference_prototypes.shape[1]}."
            )
        nearest_reference = cosine_similarity(
            query_prototypes, reference_prototypes
        ).argmax(axis=1)
        hard_mapping = np.zeros(
            (query_prototypes.shape[0], reference_prototypes.shape[0]),
            dtype=query_probability.dtype,
        )
        hard_mapping[np.arange(query_prototypes.shape[0]), nearest_reference] = 1
        reference_probability = query_probability @ hard_mapping

    if not np.allclose(reference_probability.sum(axis=1), 1.0):
        raise ValueError(
            f"[{query_species}->{reference_species}] mapped reference "
            "probabilities do not sum to 1."
        )
    top1_idx = reference_probability.argmax(axis=1)
    row_indices = np.arange(reference_probability.shape[0])
    return {
        "query_species": query_species,
        "reference_species": reference_species,
        "known_status_key": query_cache["known_status_key"],
        "split_key": query_cache["split_key"],
        "query_obs_names": query_cache["query_obs_names"],
        "y_true": query_cache["y_true"],
        "known_status": query_cache["known_status"],
        "split": query_cache["split"],
        "pred_best_label": reference_labels[top1_idx].astype(str),
        "margin": reference_probability[row_indices, top1_idx],
        "query_margin": query_probability.max(axis=1),
    }


def prepare_open_curve_acc_cache(
    adata,
    species_Q: str,
    species_R: str,
    label_key: str = "cell_type",
    item_key: str | None = "item_type",
    known_status_key: str = "known_status",
    split_key: str = "split",
    representation: str = "X",
    species_key: str = "species",
    cell_item_value: str = "cell",
    prototype_item_value: str = "celltype",
    temperature: float = 0.1,
):
    temperature = _validate_prototype_temperature(
        temperature, "Prototype cell temperature"
    )
    if item_key is None or item_key not in adata.obs:
        raise KeyError(
            "Prototype OSR evaluation requires an item-type obs column; "
            f"configured item_type_key={item_key!r}."
        )
    q_mask = _cell_selection_mask(
        adata,
        species_Q,
        species_key=species_key,
        item_key=item_key,
        cell_item_value=cell_item_value,
    )
    r_mask = (
        adata.obs[species_key].astype(str).to_numpy() == species_R
    ) & (
        adata.obs[item_key].astype(str).to_numpy() == prototype_item_value
    )

    base_cache = _base_open_curve_cache(
        adata,
        q_mask,
        species_Q=species_Q,
        species_R=species_R,
        label_key=label_key,
        known_status_key=known_status_key,
        split_key=split_key,
    )

    if q_mask.sum() == 0:
        warnings.warn(f"[{species_Q}] no cell data found")
        return base_cache
    if r_mask.sum() == 0:
        warnings.warn(f"[{species_R}] no cell-type embedding data found")
        return base_cache

    X_q = _selected_embedding_matrix(adata, q_mask, representation)
    P_r = _selected_embedding_matrix(adata, r_mask, representation)
    P_name = adata.obs.loc[r_mask, label_key].astype(str).to_numpy()

    sim = cosine_similarity(X_q, P_r)
    prob = softmax(sim / temperature, axis=1)
    top1_idx = prob.argmax(axis=1)

    return {
        **base_cache,
        "pred_best_label": P_name[top1_idx],
        "margin": prob[np.arange(prob.shape[0]), top1_idx],
    }


def fit_open_set_logistic_classifier(
    adata,
    reference_species: str,
    *,
    representation: str = "X",
    species_key: str = "species",
    label_key: str = "cell_type",
    train_label_key: str | None = None,
    item_key: str | None = "item_type",
    cell_item_value: str = "cell",
    known_status_key: str = "known_status",
    split_key: str = "split",
    preprocessing: str | None = None,
    standardize: bool | None = None,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight=None,
    random_state: int | None = None,
):
    required = {species_key, label_key, known_status_key, split_key}
    missing = required - set(adata.obs.columns)
    if missing:
        raise KeyError(
            "OSR embedding is missing required obs columns: "
            f"{sorted(missing)}"
        )

    resolved_train_label_key = train_label_key
    if resolved_train_label_key is None:
        resolved_train_label_key = (
            "train_label" if "train_label" in adata.obs else label_key
        )
    if resolved_train_label_key not in adata.obs:
        raise KeyError(
            "OSR classifier training label column is absent: "
            f"{resolved_train_label_key!r}."
        )

    train_mask = _cell_selection_mask(
        adata,
        reference_species,
        species_key=species_key,
        item_key=item_key,
        cell_item_value=cell_item_value,
    )
    train_mask &= (
        adata.obs[known_status_key].astype(str).to_numpy() == "known"
    )
    train_mask &= adata.obs[split_key].astype(str).to_numpy() == "train"
    if not train_mask.any():
        raise ValueError(
            f"[{reference_species}] no known training cells are available "
            "for classifier fitting."
        )

    labels = adata.obs.loc[train_mask, resolved_train_label_key]
    if labels.isna().any():
        raise ValueError(
            f"[{reference_species}] classifier training labels in "
            f"{resolved_train_label_key!r} contain missing values."
        )
    y_train = labels.astype(str).to_numpy()
    classes = pd.unique(y_train)
    if len(classes) < 2:
        raise ValueError(
            f"[{reference_species}] logistic regression requires at least "
            f"two known training classes; found {len(classes)}."
        )

    X_train = _selected_embedding_matrix(adata, train_mask, representation)
    if preprocessing is None:
        preprocessing = "standard" if standardize is not False else "none"
    elif standardize is not None:
        raise ValueError(
            "Specify classifier preprocessing or legacy standardize, not both."
        )
    if not isinstance(preprocessing, str):
        raise ValueError("Classifier preprocessing must be a string.")
    preprocessing = preprocessing.strip().lower()
    if preprocessing not in {"none", "standard", "l2"}:
        raise ValueError(
            "Classifier preprocessing must be 'none', 'standard', or 'l2'."
        )

    logistic = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    if preprocessing == "standard":
        classifier = make_pipeline(StandardScaler(), logistic)
    elif preprocessing == "l2":
        classifier = make_pipeline(Normalizer(norm="l2"), logistic)
    else:
        classifier = logistic
    classifier.fit(X_train, y_train)
    return classifier


def prepare_open_curve_logistic_cache(
    adata,
    species_Q: str,
    species_R: str,
    classifier,
    *,
    representation: str = "X",
    species_key: str = "species",
    label_key: str = "cell_type",
    item_key: str | None = "item_type",
    cell_item_value: str = "cell",
    known_status_key: str = "known_status",
    split_key: str = "split",
):
    q_mask = _cell_selection_mask(
        adata,
        species_Q,
        species_key=species_key,
        item_key=item_key,
        cell_item_value=cell_item_value,
    )
    base_cache = _base_open_curve_cache(
        adata,
        q_mask,
        species_Q=species_Q,
        species_R=species_R,
        label_key=label_key,
        known_status_key=known_status_key,
        split_key=split_key,
    )
    if q_mask.sum() == 0:
        warnings.warn(f"[{species_Q}] no cell data found")
        return base_cache

    X_q = _selected_embedding_matrix(adata, q_mask, representation)
    prob = classifier.predict_proba(X_q)
    top1_idx = prob.argmax(axis=1)
    classes = np.asarray(classifier.classes_)
    return {
        **base_cache,
        "pred_best_label": classes[top1_idx].astype(str),
        "margin": prob[np.arange(prob.shape[0]), top1_idx],
    }


def open_curve_acc_from_cache(
    cache,
    *,
    selection_mask=None,
    known_id_list=None,
    n_tau: int = 200,
    thresholds=None,
    unknown_id: str = "Unknown",
    return_style: str = "dict",
):
    if "margin" not in cache:
        return _empty_open_curve_result(return_style)

    n_cells = len(cache["margin"])
    if selection_mask is None:
        selection_mask = np.ones(n_cells, dtype=bool)
    else:
        selection_mask = np.asarray(selection_mask, dtype=bool)
        if selection_mask.shape != (n_cells,):
            raise ValueError(
                "selection_mask must contain one boolean per cached query cell."
            )
    if not selection_mask.any():
        warnings.warn(f"[{cache['query_species']}] no cell data found")
        return _empty_open_curve_result(return_style)

    y_q = cache["y_true"][selection_mask]
    margin = cache["margin"][selection_mask]
    pred_best_label = cache["pred_best_label"][selection_mask]

    if known_id_list is None:
        cached_status = cache["known_status"]
        if cached_status is None:
            known_status_key = cache.get("known_status_key", "known_status")
            raise ValueError(
                "known_id_list is required when the input has no "
                f"{known_status_key!r} column."
            )
        status = cached_status[selection_mask]
        invalid_status = set(status) - {"known", "unknown"}
        if invalid_status:
            known_status_key = cache.get("known_status_key", "known_status")
            raise ValueError(
                f"{known_status_key!r} contains unsupported values: "
                f"{sorted(invalid_status)}"
            )
        is_known = status == "known"
        known_labels = list(dict.fromkeys(y_q[is_known].tolist()))
    else:
        known_labels = list(known_id_list)
        is_known = np.isin(y_q, np.asarray(known_labels))

    y_true_ood = (~is_known).astype(int)
    y_score_ood = 1.0 - margin
    if y_true_ood.min() != y_true_ood.max():
        ood_auroc = roc_auc_score(y_true_ood, y_score_ood)
        ood_aupr = average_precision_score(y_true_ood, y_score_ood)
    else:
        ood_auroc = np.nan
        ood_aupr = np.nan

    if thresholds is None:
        tau_list = np.linspace(0.0, 1.0, n_tau)
    else:
        tau_list = np.asarray(thresholds, dtype=float)
        if tau_list.ndim != 1 or not len(tau_list):
            raise ValueError("thresholds must be a non-empty one-dimensional array.")
        if not np.isfinite(tau_list).all() or np.any(
            (tau_list < 0.0) | (tau_list > 1.0)
        ):
            raise ValueError("thresholds must contain finite values from 0 to 1.")
    acc_known_list = []
    f1_macro_known_list = []
    known_macro_ccr_list = []
    unknown_recall_list = []
    unknown_macro_recall_list = []
    oscr_points = []
    mask_known = is_known
    mask_unknown = ~is_known

    for tau in tau_list:
        y_pred = np.where(margin >= tau, pred_best_label, unknown_id)
        acc_known_list.append(
            (y_pred[mask_known] == y_q[mask_known]).mean()
            if mask_known.any()
            else np.nan
        )
        f1_macro_known_list.append(
            f1_score(
                y_q[mask_known],
                y_pred[mask_known],
                labels=known_labels,
                average="macro",
                zero_division=0,
            )
            if mask_known.any()
            else np.nan
        )
        ccr_known = (
            np.mean(
                (margin[mask_known] >= tau)
                & (pred_best_label[mask_known] == y_q[mask_known])
            )
            if mask_known.any()
            else np.nan
        )
        known_macro_ccr_list.append(
            _macro_known_ccr(
                y_q,
                pred_best_label,
                margin,
                mask_known,
                tau,
            )
        )
        fpr_unknown = (
            np.mean(margin[mask_unknown] >= tau)
            if mask_unknown.any()
            else np.nan
        )
        unknown_recall_list.append(
            np.mean(margin[mask_unknown] < tau)
            if mask_unknown.any()
            else np.nan
        )
        unknown_type_recalls = [
            np.mean(margin[mask_unknown & (y_q == label)] < tau)
            for label in pd.unique(y_q[mask_unknown])
        ]
        unknown_macro_recall_list.append(
            np.mean(unknown_type_recalls) if unknown_type_recalls else np.nan
        )
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
        "known_macro_ccr_list": np.asarray(known_macro_ccr_list),
        "unknown_recall_list": np.asarray(unknown_recall_list),
        "unknown_macro_recall_list": np.asarray(unknown_macro_recall_list),
        "ood_auroc": float(ood_auroc),
        "ood_aupr": float(ood_aupr),
        "oscr_points": np.asarray(oscr_points),
    }


