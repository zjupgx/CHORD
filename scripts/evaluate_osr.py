from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import yaml

from chord.analysis.osr import (
    fit_open_set_logistic_classifier,
    open_curve_acc_from_cache,
    prepare_open_curve_acc_cache,
    prepare_open_curve_logistic_cache,
    prepare_query_prototype_probability_cache,
    prepare_mapped_prototype_cache,
)
from chord.data import (
    generate_unknown_type_rotations,
    validate_osr_evaluation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rotating OSR splits from saved CHORD or external "
            "latent embeddings."
        )
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must define a top-level dictionary.")
    return cfg


def _harmonic_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second)
    result = np.full(first.shape, np.nan, dtype=float)
    result[valid & (denominator == 0)] = 0.0
    nonzero = valid & (denominator != 0)
    result[nonzero] = (
        2.0 * first[nonzero] * second[nonzero] / denominator[nonzero]
    )
    return result


def _arithmetic_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    result = np.full(first.shape, np.nan, dtype=float)
    result[valid] = (first[valid] + second[valid]) / 2.0
    return result


def _oscr_auc(result: dict[str, Any]) -> float:
    points = np.asarray(result["oscr_points"], dtype=float)
    if points.size == 0:
        return float("nan")
    finite = np.isfinite(points).all(axis=1)
    if finite.sum() < 2:
        return float("nan")
    x = points[finite, 0]
    y = points[finite, 1]
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def _curve_value(result: dict[str, Any], key: str, index: int) -> float:
    values = np.asarray(result[key], dtype=float)
    return float(values[index]) if len(values) else float("nan")


def _select_validation_threshold(
    result: dict[str, Any],
    selection_config: dict[str, Any],
) -> tuple[int, float]:
    known_accuracy = np.asarray(result["acc_known_list"], dtype=float)
    unknown_recall = np.asarray(result["unknown_recall_list"], dtype=float)
    method = selection_config["method"]

    if method == "harmonic_mean":
        scores = _harmonic_mean(known_accuracy, unknown_recall)
        if not np.isfinite(scores).any():
            raise ValueError(
                "Validation split cannot compute the harmonic mean of known "
                "accuracy and unknown recall."
            )
        best_score = float(np.nanmax(scores))
        candidates = np.flatnonzero(np.isclose(scores, best_score, equal_nan=False))
        return int(candidates[-1]), best_score

    if method == "arithmetic_mean":
        scores = _arithmetic_mean(known_accuracy, unknown_recall)
        if not np.isfinite(scores).any():
            raise ValueError(
                "Validation split cannot compute the arithmetic mean of known "
                "accuracy and unknown recall."
            )
        best_score = float(np.nanmax(scores))
        candidates = np.flatnonzero(np.isclose(scores, best_score, equal_nan=False))
        return int(candidates[-1]), best_score

    min_unknown_recall = float(selection_config["min_unknown_recall"])
    eligible = (
        np.isfinite(known_accuracy)
        & np.isfinite(unknown_recall)
        & (unknown_recall >= min_unknown_recall)
    )
    if not eligible.any():
        maximum = (
            float(np.nanmax(unknown_recall))
            if np.isfinite(unknown_recall).any()
            else float("nan")
        )
        raise ValueError(
            "No validation threshold satisfies the fixed unknown-recall "
            f"constraint: required >= {min_unknown_recall}, maximum={maximum}."
        )
    best_known_accuracy = float(np.nanmax(known_accuracy[eligible]))
    candidates = np.flatnonzero(
        eligible
        & np.isclose(known_accuracy, best_known_accuracy, equal_nan=False)
    )
    return int(candidates[-1]), best_known_accuracy


def _reported_min_unknown_recall(
    selection_config: dict[str, Any],
) -> float | str:
    if selection_config["method"] == "fixed_unknown_recall":
        return float(selection_config["min_unknown_recall"])
    return "NA"


def _exact_unknown_recall_threshold(
    cache: dict[str, Any],
    selection_mask: np.ndarray,
    target_recall: float,
) -> float:
    if not np.isfinite(target_recall) or not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be greater than 0 and at most 1.")
    statuses = cache.get("known_status")
    if statuses is None:
        raise ValueError("Exact recall threshold requires known_status values.")
    margins = np.asarray(cache["margin"], dtype=float)
    selected = np.asarray(selection_mask, dtype=bool)
    if selected.shape != margins.shape:
        raise ValueError("selection_mask must align with cached margins.")
    unknown_margins = margins[selected & (np.asarray(statuses) == "unknown")]
    if not len(unknown_margins):
        raise ValueError(
            "Exact recall threshold requires validation unknown cells."
        )
    if not np.isfinite(unknown_margins).all():
        raise ValueError("Validation unknown margins contain non-finite values.")

    required = int(np.ceil(float(target_recall) * len(unknown_margins)))
    boundary = float(np.partition(unknown_margins, required - 1)[required - 1])
    threshold = float(np.nextafter(boundary, np.inf))
    if threshold > 1.0:
        threshold = 1.0
    achieved = float(np.mean(unknown_margins < threshold))
    if achieved + 1e-12 < target_recall:
        raise ValueError(
            "No threshold from 0 to 1 can satisfy the requested validation "
            f"unknown recall: target={target_recall}, maximum={achieved}."
        )
    return threshold


def _recall_target_prefix(target_recall: float) -> str:
    text = f"{float(target_recall):.6g}".replace(".", "_")
    return f"recall_target_{text}"


def _single_threshold_result(
    cache: dict[str, Any],
    selection_mask: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    return open_curve_acc_from_cache(
        cache,
        selection_mask=selection_mask,
        thresholds=np.asarray([threshold], dtype=float),
    )


def _add_recall_target_metrics(
    row: dict[str, Any],
    cache: dict[str, Any],
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    target_recall: float,
) -> None:
    prefix = _recall_target_prefix(target_recall)
    try:
        threshold = _exact_unknown_recall_threshold(
            cache, validation_mask, target_recall
        )
    except ValueError:
        row.update(
            {
                f"{prefix}_feasible": False,
                f"{prefix}_validation_threshold": np.nan,
                f"{prefix}_validation_known_accuracy": np.nan,
                f"{prefix}_validation_known_macro_ccr": np.nan,
                f"{prefix}_validation_unknown_recall": np.nan,
                f"{prefix}_test_known_accuracy": np.nan,
                f"{prefix}_test_known_macro_ccr": np.nan,
                f"{prefix}_test_unknown_recall": np.nan,
                f"{prefix}_test_unknown_macro_recall": np.nan,
            }
        )
        return

    validation = _single_threshold_result(
        cache, validation_mask, threshold
    )
    test = _single_threshold_result(cache, test_mask, threshold)
    row.update(
        {
            f"{prefix}_feasible": True,
            f"{prefix}_validation_threshold": threshold,
            f"{prefix}_validation_known_accuracy": _curve_value(
                validation, "acc_known_list", 0
            ),
            f"{prefix}_validation_known_macro_ccr": _curve_value(
                validation, "known_macro_ccr_list", 0
            ),
            f"{prefix}_validation_unknown_recall": _curve_value(
                validation, "unknown_recall_list", 0
            ),
            f"{prefix}_test_known_accuracy": _curve_value(
                test, "acc_known_list", 0
            ),
            f"{prefix}_test_known_macro_ccr": _curve_value(
                test, "known_macro_ccr_list", 0
            ),
            f"{prefix}_test_unknown_recall": _curve_value(
                test, "unknown_recall_list", 0
            ),
            f"{prefix}_test_unknown_macro_recall": _curve_value(
                test, "unknown_macro_recall_list", 0
            ),
        }
    )


def _partition_counts(cache, selection_mask: np.ndarray) -> tuple[int, int]:
    statuses = cache["known_status"]
    if statuses is None:
        raise ValueError("OSR prediction cache has no known_status values.")
    selected = statuses[selection_mask]
    return int(np.sum(selected == "known")), int(np.sum(selected == "unknown"))


def _rotation_partition_masks(
    cache,
    rotation_assignments: pd.DataFrame,
    query_species: str,
) -> dict[str, np.ndarray]:
    statuses = cache["known_status"]
    base_splits = cache["split"]
    if statuses is None or base_splits is None:
        raise ValueError(
            "OSR prediction cache requires known_status and split values."
        )
    labels = cache["y_true"].astype(str)
    species_assignments = rotation_assignments.loc[
        rotation_assignments["species"].astype(str) == query_species
    ]
    assigned_types = {
        partition: set(
            species_assignments.loc[
                species_assignments["split"].astype(str) == partition,
                "cell_type",
            ].astype(str)
        )
        for partition in ("validation", "test")
    }
    observed_unknown = set(labels[statuses == "unknown"])
    rotation_unknown = assigned_types["validation"] | assigned_types["test"]
    if observed_unknown != rotation_unknown:
        raise ValueError(
            f"Rotation assignment for {query_species!r} does not match cached "
            f"unknown types: cached={sorted(observed_unknown)}, "
            f"assigned={sorted(rotation_unknown)}."
        )

    return {
        partition: (
            ((statuses == "known") & (base_splits == partition))
            | (
                (statuses == "unknown")
                & np.isin(labels, list(assigned_types[partition]))
            )
        )
        for partition in ("validation", "test")
    }


def _curve_rows(
    result: dict[str, Any],
    *,
    rotation: int,
    query_species: str,
    reference_species: str,
    partition: str,
) -> list[dict[str, Any]]:
    rows = []
    points = np.asarray(result["oscr_points"], dtype=float)
    known_accuracy = np.asarray(result["acc_known_list"], dtype=float)
    unknown_recall = np.asarray(result["unknown_recall_list"], dtype=float)
    macro_known_ccr = np.asarray(result["known_macro_ccr_list"], dtype=float)
    harmonic = _harmonic_mean(known_accuracy, unknown_recall)
    arithmetic = _arithmetic_mean(known_accuracy, unknown_recall)
    for index, threshold in enumerate(result["tau_list"]):
        rows.append(
            {
                "rotation": rotation,
                "query_species": query_species,
                "reference_species": reference_species,
                "partition": partition,
                "threshold": float(threshold),
                "known_accuracy": _curve_value(
                    result, "acc_known_list", index
                ),
                "known_macro_f1": _curve_value(
                    result, "f1_macro_known_list", index
                ),
                "unknown_recall": _curve_value(
                    result, "unknown_recall_list", index
                ),
                "unknown_macro_recall": _curve_value(
                    result, "unknown_macro_recall_list", index
                ),
                "unknown_false_positive_rate": float(points[index, 0]),
                "known_correct_classification_rate": float(points[index, 1]),
                "known_macro_correct_classification_rate": float(
                    macro_known_ccr[index]
                ),
                "known_unknown_harmonic_mean": float(harmonic[index]),
                "known_unknown_arithmetic_mean": float(arithmetic[index]),
            }
        )
    return rows


def evaluate_rotations(
    adata_cell,
    rotations: pd.DataFrame,
    *,
    species_names: list[str],
    n_thresholds: int,
    selection_config: dict[str, Any],
    prediction_config: dict[str, Any] | None = None,
    input_config: dict[str, Any] | None = None,
    seed: int = 0,
    progress_every: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    rotation_ids = sorted(rotations["rotation"].unique())
    total_rotations = len(rotation_ids)
    species_pairs = list(product(species_names, species_names))
    if prediction_config is None:
        prediction_config = {
            "method": "prototype",
            "representation": "X",
            "prototype": {
                "mode": "mapping",
                "cell_temperature": 0.1,
            },
            "classifier": {
                "preprocessing": "standard",
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": None,
            },
        }
    if input_config is None:
        input_config = {
            "species_key": "species",
            "label_key": "cell_type",
            "train_label_key": None,
            "item_type_key": "item_type",
            "cell_item_value": "cell",
            "prototype_item_value": "celltype",
            "known_status_key": "known_status",
            "split_key": "split",
        }
    common_cache_kwargs = {
        "representation": prediction_config["representation"],
        "species_key": input_config["species_key"],
        "label_key": input_config["label_key"],
        "item_key": input_config["item_type_key"],
        "cell_item_value": input_config["cell_item_value"],
        "known_status_key": input_config["known_status_key"],
        "split_key": input_config["split_key"],
    }
    prediction_method = prediction_config["method"]
    if prediction_method == "prototype":
        prototype_config = prediction_config.get(
            "prototype",
            {
                "mode": "mapping",
                "cell_temperature": 0.1,
            },
        )
        prediction_config["prototype"] = prototype_config
        if prototype_config["mode"] == "direct":
            prediction_cache = {
                (query_species, reference_species): prepare_open_curve_acc_cache(
                    adata_cell,
                    query_species,
                    reference_species,
                    prototype_item_value=input_config["prototype_item_value"],
                    temperature=prototype_config["cell_temperature"],
                    **common_cache_kwargs,
                )
                for query_species, reference_species in species_pairs
            }
        elif prototype_config["mode"] == "mapping":
            query_caches = {
                query_species: prepare_query_prototype_probability_cache(
                    adata_cell,
                    query_species,
                    temperature=prototype_config["cell_temperature"],
                    train_label_key=input_config["train_label_key"],
                    prototype_item_value=input_config["prototype_item_value"],
                    **common_cache_kwargs,
                )
                for query_species in species_names
            }
            prediction_cache = {
                (query_species, reference_species): (
                    prepare_mapped_prototype_cache(
                        query_caches[query_species],
                        query_caches[reference_species],
                    )
                )
                for query_species, reference_species in species_pairs
            }
            del query_caches
        else:
            raise ValueError(
                "Unsupported prototype prediction mode: "
                f"{prototype_config['mode']!r}."
            )
    elif prediction_method == "logistic_regression":
        classifier_config = prediction_config["classifier"]
        classifiers = {
            reference_species: fit_open_set_logistic_classifier(
                adata_cell,
                reference_species,
                train_label_key=input_config["train_label_key"],
                preprocessing=classifier_config["preprocessing"],
                C=classifier_config["C"],
                max_iter=classifier_config["max_iter"],
                class_weight=classifier_config["class_weight"],
                random_state=seed,
                **common_cache_kwargs,
            )
            for reference_species in species_names
        }
        prediction_cache = {
            (query_species, reference_species): prepare_open_curve_logistic_cache(
                adata_cell,
                query_species,
                reference_species,
                classifiers[reference_species],
                **common_cache_kwargs,
            )
            for query_species, reference_species in species_pairs
        }
    else:
        raise ValueError(f"Unsupported OSR prediction method: {prediction_method!r}.")

    for completed, rotation in enumerate(rotation_ids, start=1):
        current_assignments = rotations.loc[rotations["rotation"] == rotation]
        partition_masks = {
            query_species: _rotation_partition_masks(
                prediction_cache[(query_species, species_names[0])],
                current_assignments,
                query_species,
            )
            for query_species in species_names
        }
        for query_species, reference_species in species_pairs:
            cache = prediction_cache[(query_species, reference_species)]
            validation_mask = partition_masks[query_species]["validation"]
            test_mask = partition_masks[query_species]["test"]
            validation_curve = open_curve_acc_from_cache(
                cache,
                selection_mask=validation_mask,
                n_tau=n_thresholds,
            )
            test_curve = open_curve_acc_from_cache(
                cache,
                selection_mask=test_mask,
                n_tau=n_thresholds,
            )
            if not len(validation_curve["tau_list"]) or not len(
                test_curve["tau_list"]
            ):
                raise ValueError(
                    f"Rotation {rotation}, {query_species}->{reference_species} "
                    "has an empty validation or test partition."
                )
            if selection_config["method"] == "fixed_unknown_recall":
                threshold = _exact_unknown_recall_threshold(
                    cache,
                    validation_mask,
                    float(selection_config["min_unknown_recall"]),
                )
                validation_point = _single_threshold_result(
                    cache, validation_mask, threshold
                )
                selection_score = _curve_value(
                    validation_point, "acc_known_list", 0
                )
            else:
                best_index, selection_score = _select_validation_threshold(
                    validation_curve, selection_config
                )
                threshold = float(
                    validation_curve["tau_list"][best_index]
                )
                validation_point = _single_threshold_result(
                    cache, validation_mask, threshold
                )
            test_point = _single_threshold_result(
                cache, test_mask, threshold
            )
            val_known, val_unknown = _partition_counts(cache, validation_mask)
            test_known, test_unknown = _partition_counts(cache, test_mask)

            row: dict[str, Any] = {
                "rotation": int(rotation),
                "query_species": query_species,
                "reference_species": reference_species,
                "prediction_method": prediction_method,
                "representation": prediction_config["representation"],
                "prototype_mode": (
                    prediction_config["prototype"]["mode"]
                    if prediction_method == "prototype"
                    else "NA"
                ),
                "prototype_cell_temperature": (
                    prediction_config["prototype"]["cell_temperature"]
                    if prediction_method == "prototype"
                    else "NA"
                ),
                "prototype_confidence_stage": (
                    "reference" if prediction_method == "prototype" else "NA"
                ),
                "classifier_preprocessing": (
                    prediction_config["classifier"]["preprocessing"]
                    if prediction_method == "logistic_regression"
                    else "NA"
                ),
                "validation_threshold": threshold,
                "threshold_selection_method": selection_config["method"],
                "min_unknown_recall": _reported_min_unknown_recall(
                    selection_config
                ),
                "validation_selection_score": selection_score,
                "validation_n_known_cells": val_known,
                "validation_n_unknown_cells": val_unknown,
                "test_n_known_cells": test_known,
                "test_n_unknown_cells": test_unknown,
            }
            for target_recall in selection_config.get(
                "report_unknown_recall_targets", (0.85, 0.95)
            ):
                _add_recall_target_metrics(
                    row,
                    cache,
                    validation_mask,
                    test_mask,
                    float(target_recall),
                )

            for prefix, result, curve_result in (
                ("validation", validation_point, validation_curve),
                ("test", test_point, test_curve),
            ):
                index = 0
                points = np.asarray(result["oscr_points"], dtype=float)
                known_ccr = float(points[index, 1])
                known_accuracy = _curve_value(
                    result, "acc_known_list", index
                )
                unknown_recall = _curve_value(
                    result, "unknown_recall_list", index
                )
                known_macro_ccr = _curve_value(
                    result, "known_macro_ccr_list", index
                )
                unknown_macro_recall = _curve_value(
                    result, "unknown_macro_recall_list", index
                )
                row.update(
                    {
                        f"{prefix}_known_accuracy": known_accuracy,
                        f"{prefix}_known_macro_f1": _curve_value(
                            result, "f1_macro_known_list", index
                        ),
                        f"{prefix}_known_ccr": known_ccr,
                        f"{prefix}_known_macro_ccr": known_macro_ccr,
                        f"{prefix}_unknown_recall": unknown_recall,
                        f"{prefix}_unknown_macro_recall": unknown_macro_recall,
                        f"{prefix}_harmonic_mean": float(
                            _harmonic_mean(
                                np.asarray([known_accuracy]),
                                np.asarray([unknown_recall]),
                            )[0]
                        ),
                        f"{prefix}_arithmetic_mean": float(
                            _arithmetic_mean(
                                np.asarray([known_accuracy]),
                                np.asarray([unknown_recall]),
                            )[0]
                        ),
                        f"{prefix}_ood_auroc": float(
                            curve_result["ood_auroc"]
                        ),
                        f"{prefix}_ood_aupr": float(
                            curve_result["ood_aupr"]
                        ),
                        f"{prefix}_oscr_auc": _oscr_auc(curve_result),
                    }
                )

            for prefix, curve_result in (
                ("validation", validation_curve),
                ("test", test_curve),
            ):
                curve_rows.extend(
                    _curve_rows(
                        curve_result,
                        rotation=int(rotation),
                        query_species=query_species,
                        reference_species=reference_species,
                        partition=prefix,
                    )
                )

            metric_rows.append(row)

        if completed % progress_every == 0 or completed == total_rotations:
            percent = 100.0 * completed / total_rotations
            print(
                "OSR evaluation progress: "
                f"{completed}/{total_rotations} rotations ({percent:.1f}%)",
                flush=True,
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(curve_rows)


def _discover_unknown_types(
    adata_cell,
    input_config: dict[str, Any],
) -> dict[str, list[str]]:
    species_key = input_config["species_key"]
    label_key = input_config["label_key"]
    item_type_key = input_config["item_type_key"]
    known_status_key = input_config["known_status_key"]
    split_key = input_config["split_key"]
    required = {species_key, label_key, known_status_key, split_key}
    missing = required - set(adata_cell.obs.columns)
    if missing:
        raise KeyError(
            "OSR embedding is missing required obs columns: "
            f"{sorted(missing)}"
        )
    obs = adata_cell.obs
    if item_type_key is None or item_type_key not in obs:
        cell_mask = np.ones(adata_cell.n_obs, dtype=bool)
    else:
        cell_mask = (
            obs[item_type_key].astype(str).to_numpy()
            == input_config["cell_item_value"]
        )
    unknown_mask = (
        obs[known_status_key].astype(str).to_numpy() == "unknown"
    )
    species_values = obs[species_key].astype(str).to_numpy()
    labels = obs[label_key].astype(str).to_numpy()
    result: dict[str, list[str]] = {}
    for species in sorted(set(species_values[cell_mask])):
        mask = cell_mask & unknown_mask & (species_values == species)
        result[species] = sorted(set(labels[mask]))
    return result


def _retained_mask(adata, retained_key: str, source_path: Path) -> np.ndarray:
    if retained_key not in adata.obs:
        raise KeyError(
            f"Split OSR embedding {source_path} is missing required obs "
            f"column {retained_key!r}."
        )
    values = adata.obs[retained_key]
    if values.isna().any():
        raise ValueError(
            f"Split OSR embedding {source_path} has missing values in "
            f"obs[{retained_key!r}]."
        )
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)

    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(
            f"Split OSR embedding {source_path} has unsupported values in "
            f"obs[{retained_key!r}]: {invalid}."
        )
    return normalized.map(mapping).to_numpy(dtype=bool)


def _prepare_split_embedding(
    adata,
    *,
    source_path: Path,
    partition: str,
    input_config: dict[str, Any],
):
    required = {
        input_config["species_key"],
        input_config["label_key"],
        input_config["split_key"],
        input_config["known_status_key"],
        input_config["retained_key"],
    }
    missing = required - set(adata.obs.columns)
    if missing:
        raise KeyError(
            f"Split OSR embedding {source_path} is missing required obs "
            f"columns: {sorted(missing)}."
        )

    retained = _retained_mask(
        adata, input_config["retained_key"], source_path
    )
    if not retained.any():
        raise ValueError(
            f"Split OSR embedding {source_path} contains no retained cells."
        )
    if not retained.all():
        print(
            f"Filtered {int((~retained).sum())} non-retained observations "
            f"from: {source_path}"
        )
    selected = adata[retained].copy()
    split_values = selected.obs[input_config["split_key"]].astype(str)
    status_values = selected.obs[input_config["known_status_key"]].astype(str)
    if partition == "train":
        invalid_split = sorted(set(split_values) - {"train"})
        invalid_status = sorted(set(status_values) - {"known"})
        if invalid_split or invalid_status:
            raise ValueError(
                "The train embedding must contain only retained known/train "
                f"cells; invalid splits={invalid_split}, "
                f"invalid known_status values={invalid_status}."
            )
    else:
        invalid_status = sorted(set(status_values) - {"known", "unknown"})
        invalid_split = sorted(
            set(split_values) - {"validation", "test", "unassigned"}
        )
        known_with_invalid_split = (
            (status_values == "known")
            & ~split_values.isin(["validation", "test"])
        ).any()
        if invalid_status or invalid_split or known_with_invalid_split:
            raise ValueError(
                "The nontrain embedding must use known_status values "
                "known/unknown, must not contain train cells, and known cells "
                "must use validation/test splits; "
                f"invalid splits={invalid_split}, "
                f"invalid known_status values={invalid_status}."
            )
    return selected


def _validate_split_representation(
    train_adata,
    nontrain_adata,
    representation: str,
) -> None:
    if representation == "X":
        if train_adata.n_vars != nontrain_adata.n_vars:
            raise ValueError(
                "Train and nontrain X embeddings have different dimensions: "
                f"{train_adata.n_vars} != {nontrain_adata.n_vars}."
            )
        if not train_adata.var_names.equals(nontrain_adata.var_names):
            raise ValueError(
                "Train and nontrain X embeddings must have identical var_names "
                "in the same order."
            )
        return

    missing = [
        partition
        for partition, adata in (
            ("train", train_adata),
            ("nontrain", nontrain_adata),
        )
        if representation not in adata.obsm
    ]
    if missing:
        raise KeyError(
            f"Representation {representation!r} is missing from these split "
            f"embeddings: {missing}."
        )
    train_dims = train_adata.obsm[representation].shape[1]
    nontrain_dims = nontrain_adata.obsm[representation].shape[1]
    if train_dims != nontrain_dims:
        raise ValueError(
            f"Train and nontrain {representation!r} embeddings have different "
            f"dimensions: {train_dims} != {nontrain_dims}."
        )


def _load_evaluation_embeddings(osr_cfg: dict[str, Any]):
    if osr_cfg["input_mode"] == "single":
        embedding_path = Path(osr_cfg["cell_embeddings_path"])
        if not embedding_path.is_file():
            raise FileNotFoundError(
                f"OSR cell embedding not found: {embedding_path}"
            )
        return sc.read_h5ad(embedding_path), [embedding_path]

    train_path = Path(osr_cfg["train_cell_embeddings_path"])
    nontrain_path = Path(osr_cfg["nontrain_cell_embeddings_path"])
    missing = [path for path in (train_path, nontrain_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "OSR split embedding file(s) not found: "
            + ", ".join(str(path) for path in missing)
        )

    train_adata = _prepare_split_embedding(
        sc.read_h5ad(train_path),
        source_path=train_path,
        partition="train",
        input_config=osr_cfg["input"],
    )
    nontrain_adata = _prepare_split_embedding(
        sc.read_h5ad(nontrain_path),
        source_path=nontrain_path,
        partition="nontrain",
        input_config=osr_cfg["input"],
    )
    _validate_split_representation(
        train_adata,
        nontrain_adata,
        osr_cfg["prediction"]["representation"],
    )
    species_order = train_adata.uns.get(
        "species_order", nontrain_adata.uns.get("species_order")
    )
    combined = sc.concat(
        {"train": train_adata, "nontrain": nontrain_adata},
        join="inner",
        merge="same",
        uns_merge="first",
        label="_osr_input_partition",
        index_unique="-",
    )
    if species_order is not None:
        combined.uns["species_order"] = list(species_order)
    return combined, [train_path, nontrain_path]


def run_evaluation(cfg: dict[str, Any]) -> None:
    output_cfg = cfg.get("output")
    if not isinstance(output_cfg, dict) or not output_cfg.get("outdir"):
        raise ValueError("Evaluation config requires output.outdir.")
    training_cfg = cfg.get("training", {})
    default_seed = (
        training_cfg.get("seed", 42)
        if isinstance(training_cfg, dict)
        else 42
    )
    osr_cfg = validate_osr_evaluation_config(cfg, default_seed=default_seed)
    outdir = Path(output_cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    output_names = [
        osr_cfg["metrics_filename"],
        osr_cfg["curves_filename"],
        osr_cfg["rotation_assignments_filename"],
    ]
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError(
            "OSR evaluation outputs must use distinct filenames: "
            f"{output_names}"
        )

    adata_cell, embedding_paths = _load_evaluation_embeddings(osr_cfg)
    prediction_method = osr_cfg["prediction"]["method"]
    if prediction_method == "logistic_regression":
        preprocessing = osr_cfg["prediction"]["classifier"]["preprocessing"]
        print(
            "OSR prediction backend: logistic_regression "
            f"(preprocessing={preprocessing})"
        )
    else:
        prototype_cfg = osr_cfg["prediction"]["prototype"]
        print(
            "OSR prediction backend: prototype "
            f"(mode={prototype_cfg['mode']}, "
            f"cell_temperature={prototype_cfg['cell_temperature']}, "
            "confidence_stage=reference)"
        )

    unknown_by_species = _discover_unknown_types(adata_cell, osr_cfg["input"])
    rotations = generate_unknown_type_rotations(
        unknown_by_species,
        validation_types_per_species=osr_cfg[
            "validation_unknown_types_per_species"
        ],
        seed=osr_cfg["seed"],
    )
    all_rotation_ids = sorted(rotations["rotation"].unique())
    total_available_rotations = len(all_rotation_ids)
    max_rotations = osr_cfg["max_rotations"]
    selected_rotation_ids = (
        all_rotation_ids
        if max_rotations is None
        else all_rotation_ids[:max_rotations]
    )
    rotations = rotations.loc[
        rotations["rotation"].isin(selected_rotation_ids)
    ].copy()
    species_names = [
        str(species)
        for species in adata_cell.uns.get(
            "species_order", sorted(unknown_by_species)
        )
    ]
    if set(species_names) != set(unknown_by_species):
        raise ValueError(
            "Embedding species_order does not match species with unknown cells: "
            f"species_order={species_names}, unknown={sorted(unknown_by_species)}"
        )

    n_rotations = int(rotations["rotation"].nunique())
    metrics, curves = evaluate_rotations(
        adata_cell,
        rotations,
        species_names=species_names,
        n_thresholds=osr_cfg["n_thresholds"],
        selection_config=osr_cfg["threshold_selection"],
        prediction_config=osr_cfg["prediction"],
        input_config=osr_cfg["input"],
        seed=osr_cfg["seed"],
        progress_every=osr_cfg["progress_every"],
    )
    metrics_path = outdir / osr_cfg["metrics_filename"]
    curves_path = outdir / osr_cfg["curves_filename"]
    rotations_path = outdir / osr_cfg["rotation_assignments_filename"]
    metrics.to_csv(metrics_path, sep="\t", index=False)
    curves.to_csv(curves_path, sep="\t", index=False)
    rotations.to_csv(rotations_path, sep="\t", index=False)
    print(
        "Loaded OSR embeddings from: "
        + ", ".join(str(path) for path in embedding_paths)
    )
    if n_rotations == total_available_rotations:
        print(f"Evaluated all {n_rotations} rotations")
    else:
        print(
            f"Evaluated the first {n_rotations}/{total_available_rotations} "
            "rotations"
        )
    print(f"Saved OSR metrics to: {metrics_path}")
    print(f"Saved OSR threshold curves to: {curves_path}")
    print(f"Saved OSR rotation assignments to: {rotations_path}")


def main() -> None:
    args = parse_args()
    run_evaluation(load_config(args.config))


if __name__ == "__main__":
    main()
