from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .holdout import get_cell_type_holdout_config

from .interventions import (
    KNOWN_STATUS_COLUMN,
    SPLIT_COLUMN,
    TRUE_LABEL_COLUMN,
    build_training_species_adatas,
    parse_fraction,
    select_hvg_from_training_cells,
)


_SPLIT_NAMES = ("train", "validation", "test")


def _validate_filename(value: Any, field_name: str, suffix: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty filename.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace.")
    path = Path(value)
    if "/" in value or "\\" in value or path.is_absolute() or path.drive:
        raise ValueError(
            f"{field_name} must be a filename only; output.outdir controls "
            "the directory."
        )
    if path.suffix.lower() != suffix or not path.stem:
        raise ValueError(f"{field_name} must end with {suffix}.")
    return value


def _validate_absolute_h5ad_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty absolute path.")
    normalized = value.strip()
    path = Path(normalized)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path.")
    if path.suffix.lower() != ".h5ad":
        raise ValueError(f"{field_name} must point to an .h5ad file.")
    return normalized


def parse_known_cell_split_ratios(
    config: dict[str, Any],
    *,
    field_name: str = "osr.known_cell_split_ratios",
) -> dict[str, float]:
    raw = config.get(
        "known_cell_split_ratios",
        {"train": 0.8, "validation": 0.1, "test": 0.1},
    )
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a dictionary.")
    missing = set(_SPLIT_NAMES) - set(raw)
    extra = set(raw) - set(_SPLIT_NAMES)
    if missing or extra:
        raise ValueError(
            f"{field_name} must contain exactly {list(_SPLIT_NAMES)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    ratios = {
        split: parse_fraction(raw[split], f"{field_name}.{split}")
        for split in _SPLIT_NAMES
    }
    total = sum(ratios.values())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{field_name} values must sum to 1.")
    if any(ratios[split] <= 0.0 for split in _SPLIT_NAMES):
        raise ValueError(
            f"All {field_name} values must be greater than 0 for OSR evaluation."
        )
    return {split: value / total for split, value in ratios.items()}


def validate_osr_training_config(
    cfg: dict[str, Any],
    *,
    default_seed: int,
) -> dict[str, Any]:
    if get_cell_type_holdout_config(cfg) is not None:
        raise ValueError("Cell-type hold-out cannot use OSR training.")
    config = cfg.get("osr")
    if not isinstance(config, dict):
        raise ValueError("run_osr requires a top-level osr dictionary.")

    training_keys = {
        "seed",
        "unknown_min_cells",
        "known_cell_split_ratios",
        "save_split_tsv",
        "save_split_h5ad",
        "split_tsv_filename",
        "cell_embeddings_filename",
        "gene_embeddings_filename",
        "model_filename",
    }
    evaluation_keys = {
        "validation_unknown_types_per_species",
        "n_thresholds",
        "metrics_filename",
        "curves_filename",
        "cell_embeddings_path",
        "train_cell_embeddings_path",
        "nontrain_cell_embeddings_path",
        "threshold_selection",
        "rotation_assignments_filename",
        "progress_every",
        "max_rotations",
        "prediction",
        "input",
    }
    supported = training_keys | evaluation_keys
    unknown_keys = set(config) - supported
    if unknown_keys:
        raise ValueError(
            f"Unsupported osr training settings: {sorted(unknown_keys)}"
        )

    seed = config.get("seed", default_seed)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("osr.seed must be a non-negative integer.")

    min_cells = config.get("unknown_min_cells", 1)
    if isinstance(min_cells, bool) or not isinstance(min_cells, int) or min_cells < 1:
        raise ValueError("osr.unknown_min_cells must be a positive integer.")

    save_split_tsv = config.get("save_split_tsv", False)
    if not isinstance(save_split_tsv, bool):
        raise ValueError("osr.save_split_tsv must be a boolean.")
    save_split_h5ad = config.get("save_split_h5ad", False)
    if not isinstance(save_split_h5ad, bool):
        raise ValueError("osr.save_split_h5ad must be a boolean.")

    normalized = dict(config)
    normalized.update(
        {
            "seed": seed,
            "unknown_min_cells": min_cells,
            "known_cell_split_ratios": parse_known_cell_split_ratios(
                config,
                field_name="osr.known_cell_split_ratios",
            ),
            "save_split_tsv": save_split_tsv,
            "save_split_h5ad": save_split_h5ad,
            "split_tsv_filename": _validate_filename(
                config.get("split_tsv_filename", "split_manifest.tsv"),
                "osr.split_tsv_filename",
                ".tsv",
            ),
            "cell_embeddings_filename": _validate_filename(
                config.get(
                    "cell_embeddings_filename", "osr_cell_embeddings.h5ad"
                ),
                "osr.cell_embeddings_filename",
                ".h5ad",
            ),
            "gene_embeddings_filename": _validate_filename(
                config.get(
                    "gene_embeddings_filename", "osr_gene_embeddings.h5ad"
                ),
                "osr.gene_embeddings_filename",
                ".h5ad",
            ),
            "model_filename": _validate_filename(
                config.get("model_filename", "osr_best_model.pth"),
                "osr.model_filename",
                ".pth",
            ),
        }
    )
    if (
        normalized["cell_embeddings_filename"].casefold()
        == normalized["gene_embeddings_filename"].casefold()
    ):
        raise ValueError(
            "osr.cell_embeddings_filename and osr.gene_embeddings_filename "
            "must be different."
        )
    return normalized


def validate_osr_evaluation_config(
    cfg: dict[str, Any],
    *,
    default_seed: int,
) -> dict[str, Any]:
    if get_cell_type_holdout_config(cfg) is not None:
        raise ValueError("Cell-type hold-out cannot use OSR evaluation.")
    config = cfg.get("osr")
    if not isinstance(config, dict):
        raise ValueError("evaluate_osr requires a top-level osr dictionary.")
    evaluation_keys = {
        "seed",
        "validation_unknown_types_per_species",
        "n_thresholds",
        "cell_embeddings_path",
        "train_cell_embeddings_path",
        "nontrain_cell_embeddings_path",
        "threshold_selection",
        "metrics_filename",
        "curves_filename",
        "rotation_assignments_filename",
        "progress_every",
        "max_rotations",
        "prediction",
        "input",
    }
    legacy_training_keys = {
        "unknown_min_cells",
        "known_cell_split_ratios",
        "save_split_tsv",
        "save_split_h5ad",
        "split_tsv_filename",
        "gene_embeddings_filename",
        "model_filename",
        "cell_embeddings_filename",
    }
    unknown_keys = set(config) - evaluation_keys - legacy_training_keys
    if unknown_keys:
        raise ValueError(
            f"Unsupported osr evaluation settings: {sorted(unknown_keys)}"
        )

    seed = config.get("seed", default_seed)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("osr.seed must be a non-negative integer.")

    m = config.get("validation_unknown_types_per_species", 1)
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError(
            "osr.validation_unknown_types_per_species must be a positive integer."
        )

    n_thresholds = config.get("n_thresholds", 200)
    if (
        isinstance(n_thresholds, bool)
        or not isinstance(n_thresholds, int)
        or n_thresholds < 2
    ):
        raise ValueError("osr.n_thresholds must be an integer of at least 2.")

    progress_every = config.get("progress_every", 10)
    if (
        isinstance(progress_every, bool)
        or not isinstance(progress_every, int)
        or progress_every < 1
    ):
        raise ValueError("osr.progress_every must be a positive integer.")

    max_rotations = config.get("max_rotations", "all")
    if max_rotations is None or (
        isinstance(max_rotations, str)
        and max_rotations.strip().lower() == "all"
    ):
        max_rotations = None
    elif (
        isinstance(max_rotations, bool)
        or not isinstance(max_rotations, int)
        or max_rotations < 1
    ):
        raise ValueError(
            "osr.max_rotations must be 'all', null, or a positive integer."
        )

    embedding_path = config.get("cell_embeddings_path")
    train_embedding_path = config.get("train_cell_embeddings_path")
    nontrain_embedding_path = config.get("nontrain_cell_embeddings_path")
    has_single_input = embedding_path is not None
    has_any_split_input = (
        train_embedding_path is not None or nontrain_embedding_path is not None
    )
    if has_single_input and has_any_split_input:
        raise ValueError(
            "Configure either osr.cell_embeddings_path or the train/nontrain "
            "embedding paths, not both."
        )
    if has_single_input:
        input_mode = "single"
        embedding_path = _validate_absolute_h5ad_path(
            embedding_path, "osr.cell_embeddings_path"
        )
        train_embedding_path = None
        nontrain_embedding_path = None
    elif has_any_split_input:
        if train_embedding_path is None or nontrain_embedding_path is None:
            raise ValueError(
                "Both osr.train_cell_embeddings_path and "
                "osr.nontrain_cell_embeddings_path are required for split input."
            )
        input_mode = "train_nontrain"
        train_embedding_path = _validate_absolute_h5ad_path(
            train_embedding_path, "osr.train_cell_embeddings_path"
        )
        nontrain_embedding_path = _validate_absolute_h5ad_path(
            nontrain_embedding_path, "osr.nontrain_cell_embeddings_path"
        )
        if Path(train_embedding_path) == Path(nontrain_embedding_path):
            raise ValueError(
                "OSR train and nontrain embedding paths must be different."
            )
        embedding_path = None
    else:
        raise ValueError(
            "OSR evaluation requires either osr.cell_embeddings_path or both "
            "osr.train_cell_embeddings_path and "
            "osr.nontrain_cell_embeddings_path."
        )

    raw_selection = config.get("threshold_selection", {})
    if not isinstance(raw_selection, dict):
        raise ValueError("osr.threshold_selection must be a dictionary.")
    extra_selection = set(raw_selection) - {
        "method",
        "min_unknown_recall",
        "report_unknown_recall_targets",
    }
    if extra_selection:
        raise ValueError(
            "Unsupported osr.threshold_selection settings: "
            f"{sorted(extra_selection)}"
        )
    selection_method = raw_selection.get("method", "harmonic_mean")
    if selection_method not in {
        "harmonic_mean",
        "arithmetic_mean",
        "fixed_unknown_recall",
    }:
        raise ValueError(
            "osr.threshold_selection.method must be 'harmonic_mean', "
            "'arithmetic_mean', or 'fixed_unknown_recall'."
        )
    min_unknown_recall = parse_fraction(
        raw_selection.get("min_unknown_recall", 0.9),
        "osr.threshold_selection.min_unknown_recall",
    )
    raw_report_targets = raw_selection.get(
        "report_unknown_recall_targets", [0.85, 0.95]
    )
    if not isinstance(raw_report_targets, (list, tuple)):
        raise ValueError(
            "osr.threshold_selection.report_unknown_recall_targets must be "
            "a list."
        )
    report_unknown_recall_targets = [
        parse_fraction(
            value,
            "osr.threshold_selection.report_unknown_recall_targets",
        )
        for value in raw_report_targets
    ]
    if len(set(report_unknown_recall_targets)) != len(
        report_unknown_recall_targets
    ):
        raise ValueError(
            "osr.threshold_selection.report_unknown_recall_targets must not "
            "contain duplicates."
        )

    raw_prediction = config.get("prediction", {})
    if not isinstance(raw_prediction, dict):
        raise ValueError("osr.prediction must be a dictionary.")
    extra_prediction = set(raw_prediction) - {
        "method",
        "representation",
        "prototype",
        "classifier",
    }
    if extra_prediction:
        raise ValueError(
            "Unsupported osr.prediction settings: "
            f"{sorted(extra_prediction)}"
        )
    default_prediction_method = (
        "logistic_regression"
        if input_mode == "train_nontrain"
        else "prototype"
    )
    prediction_method = raw_prediction.get("method", default_prediction_method)
    if prediction_method not in {"prototype", "logistic_regression"}:
        raise ValueError(
            "osr.prediction.method must be 'prototype' or "
            "'logistic_regression'."
        )
    if input_mode == "train_nontrain" and prediction_method != "logistic_regression":
        raise ValueError(
            "Train/nontrain embedding input requires "
            "osr.prediction.method='logistic_regression'."
        )
    representation = raw_prediction.get("representation", "X")
    if not isinstance(representation, str) or not representation.strip():
        raise ValueError(
            "osr.prediction.representation must be 'X' or a non-empty obsm key."
        )
    representation = representation.strip()

    raw_prototype = raw_prediction.get("prototype", {})
    if not isinstance(raw_prototype, dict):
        raise ValueError("osr.prediction.prototype must be a dictionary.")
    extra_prototype = set(raw_prototype) - {
        "mode",
        "cell_temperature",
    }
    if extra_prototype:
        raise ValueError(
            "Unsupported osr.prediction.prototype settings: "
            f"{sorted(extra_prototype)}"
        )
    prototype_mode = raw_prototype.get("mode", "mapping")
    if prototype_mode not in {"mapping", "direct"}:
        raise ValueError(
            "osr.prediction.prototype.mode must be 'mapping' or 'direct'."
        )
    cell_temperature = raw_prototype.get("cell_temperature", 0.1)
    if (
        isinstance(cell_temperature, bool)
        or not isinstance(cell_temperature, (int, float))
        or not np.isfinite(cell_temperature)
        or cell_temperature <= 0
    ):
        raise ValueError(
            "osr.prediction.prototype.cell_temperature must be a finite "
            "number greater than 0."
        )

    raw_classifier = raw_prediction.get("classifier", {})
    if not isinstance(raw_classifier, dict):
        raise ValueError("osr.prediction.classifier must be a dictionary.")
    extra_classifier = set(raw_classifier) - {
        "preprocessing",
        "standardize",
        "C",
        "max_iter",
        "class_weight",
    }
    if extra_classifier:
        raise ValueError(
            "Unsupported osr.prediction.classifier settings: "
            f"{sorted(extra_classifier)}"
        )
    if "preprocessing" in raw_classifier and "standardize" in raw_classifier:
        raise ValueError(
            "osr.prediction.classifier cannot specify both preprocessing and "
            "legacy standardize."
        )
    if "preprocessing" in raw_classifier:
        preprocessing = raw_classifier["preprocessing"]
        if not isinstance(preprocessing, str):
            raise ValueError(
                "osr.prediction.classifier.preprocessing must be a string."
            )
        preprocessing = preprocessing.strip().lower()
        if preprocessing not in {"none", "standard", "l2"}:
            raise ValueError(
                "osr.prediction.classifier.preprocessing must be one of "
                "'none', 'standard', or 'l2'."
            )
    else:
        standardize = raw_classifier.get("standardize", True)
        if not isinstance(standardize, bool):
            raise ValueError(
                "osr.prediction.classifier.standardize must be a boolean."
            )
        preprocessing = "standard" if standardize else "none"
    classifier_c = raw_classifier.get("C", 1.0)
    if (
        isinstance(classifier_c, bool)
        or not isinstance(classifier_c, (int, float))
        or not np.isfinite(classifier_c)
        or classifier_c <= 0
    ):
        raise ValueError(
            "osr.prediction.classifier.C must be a finite number greater than 0."
        )
    classifier_max_iter = raw_classifier.get("max_iter", 1000)
    if (
        isinstance(classifier_max_iter, bool)
        or not isinstance(classifier_max_iter, int)
        or classifier_max_iter < 1
    ):
        raise ValueError(
            "osr.prediction.classifier.max_iter must be a positive integer."
        )
    class_weight = raw_classifier.get("class_weight")
    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError(
                "osr.prediction.classifier.class_weight must be null, "
                "'balanced', or a dictionary."
            )
    elif class_weight is not None:
        if not isinstance(class_weight, dict):
            raise ValueError(
                "osr.prediction.classifier.class_weight must be null, "
                "'balanced', or a dictionary."
            )
        invalid_weights = [
            label
            for label, weight in class_weight.items()
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not np.isfinite(weight)
                or weight <= 0
            )
        ]
        if invalid_weights:
            raise ValueError(
                "osr.prediction.classifier.class_weight values must be finite "
                f"and greater than 0; invalid labels={invalid_weights}."
            )
        class_weight = {
            str(label): float(weight) for label, weight in class_weight.items()
        }

    raw_input = config.get("input", {})
    if not isinstance(raw_input, dict):
        raise ValueError("osr.input must be a dictionary.")
    input_defaults = {
        "species_key": "species",
        "label_key": (
            "true_label" if input_mode == "train_nontrain" else "cell_type"
        ),
        "train_label_key": None,
        "item_type_key": (
            None if input_mode == "train_nontrain" else "item_type"
        ),
        "cell_item_value": "cell",
        "prototype_item_value": "celltype",
        "known_status_key": "known_status",
        "split_key": "split",
        "retained_key": (
            "retained" if input_mode == "train_nontrain" else None
        ),
    }
    extra_input = set(raw_input) - set(input_defaults)
    if extra_input:
        raise ValueError(
            f"Unsupported osr.input settings: {sorted(extra_input)}"
        )
    normalized_input = {**input_defaults, **raw_input}
    for key, value in normalized_input.items():
        if key in {
            "train_label_key",
            "item_type_key",
            "retained_key",
        } and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"osr.input.{key} must be a non-empty string.")
        normalized_input[key] = value.strip()
    if (
        input_mode == "train_nontrain"
        and normalized_input["retained_key"] is None
    ):
        raise ValueError(
            "osr.input.retained_key cannot be null for train/nontrain input."
        )

    normalized = dict(config)
    normalized.update(
        {
            "seed": seed,
            "validation_unknown_types_per_species": m,
            "n_thresholds": n_thresholds,
            "progress_every": progress_every,
            "max_rotations": max_rotations,
            "input_mode": input_mode,
            "cell_embeddings_path": embedding_path,
            "train_cell_embeddings_path": train_embedding_path,
            "nontrain_cell_embeddings_path": nontrain_embedding_path,
            "threshold_selection": {
                "method": selection_method,
                "min_unknown_recall": min_unknown_recall,
                "report_unknown_recall_targets": (
                    report_unknown_recall_targets
                ),
            },
            "prediction": {
                "method": prediction_method,
                "representation": representation,
                "prototype": {
                    "mode": prototype_mode,
                    "cell_temperature": float(cell_temperature),
                },
                "classifier": {
                    "preprocessing": preprocessing,
                    "C": float(classifier_c),
                    "max_iter": classifier_max_iter,
                    "class_weight": class_weight,
                },
            },
            "input": normalized_input,
            "metrics_filename": _validate_filename(
                config.get("metrics_filename", "osr_metrics.tsv"),
                "osr.metrics_filename",
                ".tsv",
            ),
            "curves_filename": _validate_filename(
                config.get("curves_filename", "osr_curves.tsv"),
                "osr.curves_filename",
                ".tsv",
            ),
            "rotation_assignments_filename": _validate_filename(
                config.get(
                    "rotation_assignments_filename",
                    "osr_rotation_assignments.tsv",
                ),
                "osr.rotation_assignments_filename",
                ".tsv",
            ),
        }
    )
    return normalized


def _stable_digest(seed: int, *parts: str) -> int:
    payload = "\0".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _allocate_counts(n_items: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = np.asarray([n_items * ratios[name] for name in _SPLIT_NAMES])
    counts = np.floor(raw).astype(int)
    remainder = n_items - int(counts.sum())
    fractional = raw - counts
    order = sorted(
        range(len(_SPLIT_NAMES)),
        key=lambda index: (-fractional[index], index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    required = [0, 0, 0]
    if n_items:
        required[0] = 1
    if n_items >= len(_SPLIT_NAMES):
        required = [1, 1, 1]
    for target, minimum in enumerate(required):
        if counts[target] >= minimum:
            continue
        donors = [
            index
            for index in range(len(_SPLIT_NAMES))
            if counts[index] > required[index]
        ]
        if not donors:
            continue
        donor = max(donors, key=lambda index: (counts[index], -index))
        counts[donor] -= 1
        counts[target] += 1
    return {name: int(counts[index]) for index, name in enumerate(_SPLIT_NAMES)}


def discover_osr_cell_types(
    species_adatas: dict[str, tuple[Any, str]],
    *,
    unknown_min_cells: int,
) -> tuple[set[str], dict[str, list[str]], pd.DataFrame]:
    label_species: dict[str, set[str]] = {}
    counts: dict[tuple[str, str], int] = {}
    for species, (adata, label_col) in species_adatas.items():
        if not adata.obs_names.is_unique:
            raise ValueError(
                f"[{species}] OSR requires unique cell IDs in adata.obs_names."
            )
        if label_col not in adata.obs:
            raise KeyError(f"[{species}] missing label column {label_col!r}.")
        if adata.obs[label_col].isna().any():
            raise ValueError(
                f"[{species}] label column {label_col!r} contains missing values."
            )
        labels = adata.obs[label_col].astype(str)
        for label, count in labels.value_counts(sort=False).items():
            label = str(label)
            counts[(species, label)] = int(count)
            label_species.setdefault(label, set()).add(species)

    known_types = {
        label for label, species_set in label_species.items() if len(species_set) >= 2
    }
    if not known_types:
        raise ValueError(
            "OSR found no known cell types: no label occurs in at least two species."
        )

    unknown_by_species = {species: [] for species in species_adatas}
    rows: list[dict[str, Any]] = []
    for (species, label), count in sorted(counts.items()):
        occurrence = len(label_species[label])
        if occurrence >= 2:
            status = "known"
            retained = True
        elif count >= unknown_min_cells:
            status = "unknown"
            retained = True
            unknown_by_species[species].append(label)
        else:
            status = "filtered"
            retained = False
        rows.append(
            {
                "species": species,
                "cell_type": label,
                "n_cells": count,
                "n_species": occurrence,
                KNOWN_STATUS_COLUMN: status,
                "retained": retained,
            }
        )

    for species in unknown_by_species:
        unknown_by_species[species].sort()
    return known_types, unknown_by_species, pd.DataFrame(rows)


def prepare_osr_species_views(
    species_adatas: dict[str, tuple[Any, str]],
    osr_config: dict[str, Any],
    *,
    n_top_genes: int,
) -> tuple[
    dict[str, tuple[Any, str]],
    dict[str, tuple[Any, str]],
    dict[str, Any],
]:
    known_types, unknown_by_species, type_table = discover_osr_cell_types(
        species_adatas,
        unknown_min_cells=osr_config["unknown_min_cells"],
    )

    status_lookup = {
        (str(row.species), str(row.cell_type)): str(row.known_status)
        for row in type_table.itertuples(index=False)
    }
    ratios = osr_config["known_cell_split_ratios"]
    seed = osr_config["seed"]
    annotated: dict[str, tuple[Any, str]] = {}

    for species, (adata, label_col) in species_adatas.items():
        labels_all = adata.obs[label_col].astype(str).to_numpy()
        retain_mask = np.asarray(
            [status_lookup[(species, label)] != "filtered" for label in labels_all]
        )
        retained = adata[retain_mask].copy()
        labels = retained.obs[label_col].astype(str).to_numpy()
        statuses = np.asarray(
            [status_lookup[(species, label)] for label in labels], dtype=object
        )
        splits = np.full(retained.n_obs, "unassigned", dtype=object)

        cell_ids = np.asarray(retained.obs_names.astype(str))
        for label in sorted(set(labels[statuses == "known"])):
            positions = np.flatnonzero((labels == label) & (statuses == "known"))
            positions = np.asarray(
                sorted(
                    positions,
                    key=lambda position: _stable_digest(
                        seed, species, label, cell_ids[position]
                    ),
                )
            )
            counts = _allocate_counts(len(positions), ratios)
            start = 0
            for split in _SPLIT_NAMES:
                stop = start + counts[split]
                splits[positions[start:stop]] = split
                start = stop

        missing_known_splits = [
            split
            for split in _SPLIT_NAMES
            if not np.any((statuses == "known") & (splits == split))
        ]
        if missing_known_splits:
            raise ValueError(
                f"[{species}] known cells cannot populate these requested "
                f"splits: {missing_known_splits}. Increase known-cell counts "
                "or adjust osr.known_cell_split_ratios."
            )

        retained.obs[TRUE_LABEL_COLUMN] = labels
        retained.obs[KNOWN_STATUS_COLUMN] = statuses
        retained.obs[SPLIT_COLUMN] = splits
        annotated[species] = (retained, label_col)

    annotated = select_hvg_from_training_cells(annotated, n_top_genes)
    training = build_training_species_adatas(
        annotated,
        split_enabled=True,
        label_perturbation_enabled=False,
    )
    metadata = {
        "known_cell_types": sorted(known_types),
        "unknown_cell_types_by_species": unknown_by_species,
        "cell_type_table": type_table,
        "known_cell_split_ratios": dict(ratios),
    }
    return annotated, training, metadata


def generate_unknown_type_rotations(
    unknown_by_species: dict[str, list[str]],
    *,
    validation_types_per_species: int,
    seed: int,
) -> pd.DataFrame:
    ordered: dict[str, list[str]] = {}
    for species, labels in sorted(unknown_by_species.items()):
        ordered[species] = sorted(
            labels,
            key=lambda label: _stable_digest(seed, "unknown-rotation", species, label),
        )

    combination_lists: dict[str, list[tuple[str, ...]]] = {}
    for species, labels in ordered.items():
        if validation_types_per_species >= len(labels):
            raise ValueError(
                f"[{species}] validation_unknown_types_per_species must be "
                "smaller than the number of unknown types so test retains "
                f"at least one type; m={validation_types_per_species}, "
                f"n_unknown={len(labels)}."
            )
        combination_lists[species] = list(
            combinations(labels, validation_types_per_species)
        )

    species_order = list(ordered)
    n_rotations = max(len(values) for values in combination_lists.values())
    rows: list[dict[str, Any]] = []
    for rotation in range(n_rotations):
        for species in species_order:
            labels = ordered[species]
            combinations_for_species = combination_lists[species]
            validation_tuple = combinations_for_species[
                rotation % len(combinations_for_species)
            ]
            validation = set(validation_tuple)
            for label in labels:
                rows.append(
                    {
                        "rotation": rotation,
                        "species": species,
                        "cell_type": label,
                        SPLIT_COLUMN: (
                            "validation" if label in validation else "test"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_osr_training_split_manifest(
    original_species_adatas: dict[str, tuple[Any, str]],
    prepared_species_adatas: dict[str, tuple[Any, str]],
) -> pd.DataFrame:
    retained_lookup: dict[tuple[str, str], tuple[str, str]] = {}
    for species, (adata, _label_col) in prepared_species_adatas.items():
        for cell_id, split, status in zip(
            adata.obs_names.astype(str),
            adata.obs[SPLIT_COLUMN].astype(str),
            adata.obs[KNOWN_STATUS_COLUMN].astype(str),
        ):
            retained_lookup[(species, str(cell_id))] = (str(split), str(status))

    rows: list[dict[str, Any]] = []
    for species, (adata, label_col) in original_species_adatas.items():
        for cell_id, label in zip(
            adata.obs_names.astype(str), adata.obs[label_col].astype(str)
        ):
            retained = retained_lookup.get((species, str(cell_id)))
            if retained is None:
                split, status, is_retained = "filtered", "filtered", False
            else:
                split, status = retained
                is_retained = True
            rows.append(
                {
                    "species": species,
                    "cell_id": str(cell_id),
                    TRUE_LABEL_COLUMN: str(label),
                    SPLIT_COLUMN: split,
                    KNOWN_STATUS_COLUMN: status,
                    "retained": is_retained,
                }
            )
    return pd.DataFrame(rows)


def split_osr_raw_species_adatas(
    raw_species_adatas: dict[str, Any],
    manifest: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    required = {
        "species",
        "cell_id",
        TRUE_LABEL_COLUMN,
        SPLIT_COLUMN,
        KNOWN_STATUS_COLUMN,
        "retained",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"OSR split manifest is missing columns: {sorted(missing)}")

    manifest_species = set(manifest["species"].astype(str))
    input_species = set(raw_species_adatas)
    if manifest_species != input_species:
        raise ValueError(
            "Raw h5ad species do not match the split manifest: "
            f"raw={sorted(input_species)}, manifest={sorted(manifest_species)}."
        )

    result: dict[str, dict[str, Any]] = {}
    nontrain_splits = {"validation", "test", "unassigned"}
    for species, raw_adata in raw_species_adatas.items():
        if not raw_adata.obs_names.is_unique:
            raise ValueError(
                f"[{species}] source h5ad requires unique cell IDs in obs_names."
            )
        species_manifest = manifest.loc[
            manifest["species"].astype(str) == species
        ].copy()
        if species_manifest["cell_id"].astype(str).duplicated().any():
            raise ValueError(
                f"[{species}] split manifest contains duplicate cell IDs."
            )

        source_ids = set(raw_adata.obs_names.astype(str))
        manifest_ids = set(species_manifest["cell_id"].astype(str))
        missing_ids = manifest_ids - source_ids
        if missing_ids:
            preview = sorted(missing_ids)[:10]
            raise KeyError(
                f"[{species}] {len(missing_ids)} manifest cells are absent "
                f"from the source h5ad; examples={preview}."
            )

        retained = species_manifest["retained"].astype(bool)
        split_values = species_manifest[SPLIT_COLUMN].astype(str)
        status_values = species_manifest[KNOWN_STATUS_COLUMN].astype(str)
        partition_masks = {
            "train": retained
            & (split_values == "train")
            & (status_values != "filtered"),
            "nontrain": retained
            & split_values.isin(nontrain_splits)
            & (status_values != "filtered"),
        }
        covered = partition_masks["train"] | partition_masks["nontrain"]
        if not np.all(covered == retained):
            unexpected = sorted(set(split_values[retained & ~covered]))
            raise ValueError(
                f"[{species}] retained cells have unsupported split values: "
                f"{unexpected}."
            )

        result[species] = {}
        for partition, mask in partition_masks.items():
            selected_manifest = species_manifest.loc[mask].copy()
            selected_ids = selected_manifest["cell_id"].astype(str).tolist()
            subset = raw_adata[selected_ids].copy()
            selected_manifest.index = selected_ids
            subset.obs[TRUE_LABEL_COLUMN] = selected_manifest[
                TRUE_LABEL_COLUMN
            ].astype(str)
            subset.obs[SPLIT_COLUMN] = selected_manifest[SPLIT_COLUMN].astype(
                str
            )
            subset.obs[KNOWN_STATUS_COLUMN] = selected_manifest[
                KNOWN_STATUS_COLUMN
            ].astype(str)
            subset.obs["retained"] = selected_manifest["retained"].astype(bool)
            subset.uns["osr_species"] = species
            subset.uns["osr_partition"] = partition
            result[species][partition] = subset

    return result
