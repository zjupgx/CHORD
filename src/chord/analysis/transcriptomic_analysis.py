import json
import re
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def _numeric_time(label):
    match = re.search(r"\d+(?:\.\d+)?", str(label))
    if match is None:
        raise ValueError(f"Cannot order time label {label!r}; supply time_orders.")
    return float(match.group())


def _ordered_times(values, order=None):
    values = list(dict.fromkeys(str(value) for value in values))
    if order is not None:
        order = [str(value) for value in order]
        if len(order) != len(set(order)):
            raise ValueError("An explicit time order must not contain duplicates.")
        missing = set(values) - set(order)
        if missing:
            raise ValueError(f"Time order is missing observed labels: {sorted(missing)}")
        return [value for value in order if value in set(values)]
    numbers = [_numeric_time(value) for value in values]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Ambiguous numeric time labels; supply an explicit time order.")
    return sorted(values, key=_numeric_time)


def _cell_view(adata):
    if "item_type" in adata.obs:
        return adata[adata.obs["item_type"].astype(str) == "cell"].copy()
    return adata.copy()


def _embedding_array(adata, embedding_key):
    values = adata.X if embedding_key == "X" else adata.obsm[embedding_key]
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Cell embeddings must be a nonempty two-dimensional matrix.")
    if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("Cosine distances require finite, nonzero cell embeddings.")
    return values


def _require_obs(adata, columns):
    for column in set(columns):
        if column not in adata.obs or adata.obs[column].isna().any():
            raise ValueError(f"Cell metadata requires a complete obs[{column!r}].")


def _similarity_frame(matrix, row_times=None, column_times=None):
    if isinstance(matrix, pd.DataFrame):
        frame = matrix.copy()
    else:
        if row_times is None or column_times is None:
            raise ValueError("An array similarity matrix requires row_times and column_times.")
        frame = pd.DataFrame(matrix, index=row_times, columns=column_times)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    values = frame.to_numpy(dtype=float)
    if frame.empty or not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Similarity matrix labels must be unique and nonempty.")
    if np.isinf(values).any() or np.any(np.abs(values[np.isfinite(values)]) > 1 + 1e-6):
        raise ValueError("Expected cosine similarities in [-1, 1], or NaN for unavailable pairs.")
    return pd.DataFrame(np.clip(values, -1, 1), index=frame.index, columns=frame.columns)


def build_transcriptomic_cache(
    adata, *, species_col="species", time_col="Time",
    cell_type_col="cell_type", cell_id_col="cell_id", embedding_key="X",
):
    cells = _cell_view(adata)
    grouping_columns = [species_col, time_col]
    if cell_type_col is not None:
        grouping_columns.append(cell_type_col)
    _require_obs(cells, grouping_columns)
    if not cells.obs_names.is_unique:
        raise ValueError("obs_names must be unique for cached score assignment.")
    values = _embedding_array(cells, embedding_key)
    obs = cells.obs.copy()
    labels = obs[grouping_columns].astype(str)
    ids = obs[cell_id_col].astype(str).to_numpy() if cell_id_col in obs else obs.index.astype(str).to_numpy()
    grouped = labels.groupby(list(labels.columns), sort=True, observed=True).indices
    ordered_positions = [
        np.asarray(positions)[np.argsort(ids[np.asarray(positions)], kind="stable")]
        for positions in grouped.values()
    ]
    permutation = np.concatenate(ordered_positions)
    obs = obs.iloc[permutation].copy()
    raw = np.asarray(values[permutation])
    norms = np.linalg.norm(np.asarray(raw, dtype=float), axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Cached cell embeddings must have finite, nonzero norms.")
    normalized = raw / norms[:, None]
    groups, start = {}, 0
    for key, positions in zip(grouped, ordered_positions):
        stop = start + len(positions)
        mean = raw[start:stop].mean(axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(mean))
        valid = np.isfinite(norm) and norm > 0
        groups[tuple(map(str, key))] = {
            "positions": np.arange(start, stop),
            "cell_keys": obs.index[start:stop].astype(str).to_numpy(),
            "normalized_embeddings": normalized[start:stop],
            "mean_embedding": mean,
            "normalized_centroid": mean / norm if valid else np.full_like(mean, np.nan),
            "mean_normalized_embedding": normalized[start:stop].mean(axis=0, dtype=np.float64),
            "n_cells": stop - start, "centroid_norm": norm,
            "status": "ok" if valid else "unavailable_zero_centroid",
        }
        start = stop
    stage_groups = {}
    for key, group in groups.items():
        stage_key = key[:2]
        if stage_key not in stage_groups:
            stage_groups[stage_key] = {
                "n_cells": 0,
                "embedding_sum": np.zeros_like(group["mean_embedding"]),
                "normalized_embedding_sum": np.zeros_like(group["mean_normalized_embedding"]),
            }
        stage = stage_groups[stage_key]
        stage["n_cells"] += group["n_cells"]
        stage["embedding_sum"] += group["mean_embedding"] * group["n_cells"]
        stage["normalized_embedding_sum"] += group["mean_normalized_embedding"] * group["n_cells"]
    for stage in stage_groups.values():
        mean = stage.pop("embedding_sum") / stage["n_cells"]
        norm = float(np.linalg.norm(mean))
        valid = np.isfinite(norm) and norm > 0
        stage.update({
            "mean_embedding": mean,
            "mean_normalized_embedding": stage.pop("normalized_embedding_sum") / stage["n_cells"],
            "normalized_centroid": mean / norm if valid else np.full_like(mean, np.nan),
            "centroid_norm": norm,
            "status": "ok" if valid else "unavailable_zero_stage_centroid",
        })
    return {
        "groups": groups, "stage_groups": stage_groups, "obs": obs, "embeddings": raw,
        "normalized_embeddings": normalized,
        "species_col": species_col, "time_col": time_col,
        "cell_type_col": cell_type_col, "cell_id_col": cell_id_col,
        "embedding_key": embedding_key,
    }


def _resolve_transcriptomic_cache(
    adata, cache, *, species_col, time_col, cell_type_col, embedding_key,
):
    if cache is None:
        return build_transcriptomic_cache(
            adata, species_col=species_col, time_col=time_col,
            cell_type_col=cell_type_col, embedding_key=embedding_key,
        )
    expected = {"species_col": species_col, "time_col": time_col,
                "cell_type_col": cell_type_col, "embedding_key": embedding_key}
    if any(cache.get(key) != value for key, value in expected.items()):
        raise ValueError("Cache columns/representation do not match this analysis.")
    if adata is not None:
        mask = (adata.obs["item_type"].astype(str) == "cell" if "item_type" in adata.obs
                else np.ones(adata.n_obs, dtype=bool))
        actual = adata.obs.loc[mask]
        cached = cache["obs"]
        columns = [species_col, time_col]
        if cell_type_col is not None:
            columns.append(cell_type_col)
        if not actual.index.is_unique or set(actual.index) != set(cached.index):
            raise ValueError("Cache cell IDs do not match the supplied AnnData.")
        if not actual.loc[cached.index, columns].astype(str).equals(cached[columns].astype(str)):
            raise ValueError("Cache grouping metadata is stale; rebuild it.")
    return cache


def _cell_type_averaged_stage_groups(cache):
    if cache.get("cell_type_col") is None:
        raise ValueError("Equal cell-type weighting requires a cell_type_col in the cache.")
    by_stage = {}
    for (species, time, label), group in cache["groups"].items():
        by_stage.setdefault((species, time), []).append(group)
    stages = {}
    for key, groups in by_stage.items():
        mean = np.mean([group["mean_embedding"] for group in groups], axis=0)
        norm = float(np.linalg.norm(mean))
        valid = np.isfinite(norm) and norm > 0
        stages[key] = {
            "n_cells": sum(group["n_cells"] for group in groups), "n_types": len(groups),
            "mean_embedding": mean,
            "mean_normalized_embedding": np.mean([group["mean_normalized_embedding"] for group in groups], axis=0),
            "normalized_centroid": mean / norm if valid else np.full_like(mean, np.nan),
            "centroid_norm": norm, "status": "ok" if valid else "unavailable_zero_stage_centroid",
        }
    return stages


def compute_developmental_similarity(
    adata=None, species1="frog", species2="fish", *, time_col="Time",
    species_col="species", embedding_key="X", time_orders=None,
    cache=None, return_components=False, cell_type_weighting="cell_count", cell_type_col="cell_type",
):
    if cell_type_weighting not in {"cell_count", "equal"}:
        raise ValueError("cell_type_weighting must be 'cell_count' or 'equal'.")
    cache = _resolve_transcriptomic_cache(
        adata, cache, species_col=species_col, time_col=time_col,
        cell_type_col=(cache.get("cell_type_col") if cache is not None
                       else cell_type_col if cell_type_weighting == "equal" else None),
        embedding_key=embedding_key,
    )
    groups = _cell_type_averaged_stage_groups(cache) if cell_type_weighting == "equal" else cache["stage_groups"]
    row_times = _ordered_times([key[1] for key in groups if key[0] == species1], (time_orders or {}).get(species1))
    column_times = _ordered_times([key[1] for key in groups if key[0] == species2], (time_orders or {}).get(species2))
    if not row_times or not column_times:
        raise ValueError("Both alignment species must contain cells.")
    shape = (len(row_times), len(column_times))
    forward, reverse, similarity = (np.full(shape, np.nan) for _ in range(3))
    coverage = []
    for i, row in enumerate(row_times):
        source = groups[species1, row]
        for j, column in enumerate(column_times):
            target = groups[species2, column]
            available = source["status"] == "ok" and target["status"] == "ok"
            if available:
                forward[i, j] = 1 - np.clip(np.dot(source["mean_normalized_embedding"], target["normalized_centroid"]), -1, 1)
                reverse[i, j] = 1 - np.clip(np.dot(target["mean_normalized_embedding"], source["normalized_centroid"]), -1, 1)
                similarity[i, j] = 1 - (forward[i, j] + reverse[i, j]) / 2
            coverage.append({
                "stage": row, "time": column, "status": "ok" if available else "unavailable",
                "reason": "" if available else "zero_stage_centroid",
                "species1": species1, "species2": species2,
                "n_cells_species1": source["n_cells"], "n_cells_species2": target["n_cells"],
                "cell_type_weighting": cell_type_weighting,
                "n_types_species1": source.get("n_types"), "n_types_species2": target.get("n_types"),
                "source_centroid_norm": source["centroid_norm"],
                "target_centroid_norm": target["centroid_norm"],
                "source_cell_coverage": 1.0 if available else 0.0,
                "target_cell_coverage": 1.0 if available else 0.0,
            })
    frame = pd.DataFrame(similarity, index=row_times, columns=column_times)
    frame.attrs["method"] = ("symmetric_cell_to_stage_centroid_cosine" if cell_type_weighting == "cell_count"
                             else "equal_cell_type_weighted_symmetric_cell_to_stage_centroid_cosine")
    if not return_components:
        return frame
    return {
        "similarity": frame,
        "distance": pd.DataFrame(1-similarity, index=row_times, columns=column_times),
        "forward_distance": pd.DataFrame(forward, index=row_times, columns=column_times),
        "reverse_distance": pd.DataFrame(reverse, index=row_times, columns=column_times),
        "coverage": pd.DataFrame(coverage),
    }


def dtw_developmental_alignment(
    similarity_matrix, row_times=None, column_times=None, *, open_end=True,
    query_axis="rows",
):
    if query_axis not in {"rows", "columns"}:
        raise ValueError("query_axis must be 'rows' or 'columns'.")
    similarity = _similarity_frame(similarity_matrix, row_times, column_times)
    working = similarity if query_axis == "rows" else similarity.T
    distance = 1 - working.to_numpy()
    cost = np.where(np.isfinite(distance), distance, np.inf)
    n, m = cost.shape
    accumulated = np.full((n + 1, m + 1), np.inf)
    predecessor = np.full((n, m), -1, dtype=np.int8)
    accumulated[1, 1] = cost[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            candidates = (accumulated[i, j] + 2 * cost[i, j],
                          accumulated[i, j + 1] + cost[i, j],
                          accumulated[i + 1, j] + cost[i, j])
            step = int(np.argmin(candidates))
            predecessor[i, j] = step
            accumulated[i + 1, j + 1] = candidates[step]
    normalized_end_costs = accumulated[n, 1:] / (n + np.arange(1, m + 1))
    endpoint = int(np.argmin(normalized_end_costs)) if open_end else m - 1
    available = bool(np.isfinite(accumulated[n, endpoint + 1]))
    path = []
    if available:
        i, j = n - 1, endpoint
        while True:
            path.append((i, j))
            if i == 0 and j == 0:
                break
            step = predecessor[i, j]
            i -= int(step in (0, 1))
            j -= int(step in (0, 2))
        path.reverse()
    path_rows = []
    for index, (i, j) in enumerate(path):
        diagonal = index > 0 and i > path[index - 1][0] and j > path[index - 1][1]
        weight = 2 if diagonal else 1
        row, column = (i, j) if query_axis == "rows" else (j, i)
        path_rows.append({
            'row_index': row, 'column_index': column, 'stage': similarity.index[row],
            'time': similarity.columns[column], 'cosine_distance': cost[i, j],
            'step_weight': weight, 'weighted_cosine_distance': weight * cost[i, j],
        })
    path_frame = pd.DataFrame(path_rows, columns=[
        'row_index', 'column_index', 'stage', 'time', 'cosine_distance',
        'step_weight', 'weighted_cosine_distance',
    ])
    def directional_matches(frame, pairs):
        distances = np.where(np.isfinite(frame.to_numpy()), 1 - frame.to_numpy(), np.inf)
        best = np.argmin(distances, axis=1)
        finite_rows = np.isfinite(distances).any(axis=1)
        records = []
        for i, stage in enumerate(frame.index):
            columns = [j for row, j in pairs if row == i]
            labels = [frame.columns[j] for j in columns]
            records.append({
                'stage': stage,
                'best_match': frame.columns[best[i]] if finite_rows[i] else None,
                'best_match_distance': distances[i, best[i]] if finite_rows[i] else np.nan,
                'dtw_times': json.dumps(labels),
                'dtw_first_time': labels[0] if labels else None,
                'dtw_last_time': labels[-1] if labels else None,
                'dtw_median_column_index': float(np.median(columns)) if columns else np.nan,
                'best_match_on_dtw_path': bool(finite_rows[i] and int(best[i]) in columns),
                'is_stage_endpoint': i in (0, len(frame) - 1),
            })
        return pd.DataFrame(records)

    canonical_pairs = list(path_frame[['row_index', 'column_index']].itertuples(index=False, name=None))
    stage_matches = directional_matches(similarity, canonical_pairs)
    time_matches = directional_matches(similarity.T, [(j, i) for i, j in canonical_pairs]).rename(columns={
        'stage': 'time', 'dtw_times': 'dtw_stages', 'dtw_first_time': 'dtw_first_stage',
        'dtw_last_time': 'dtw_last_stage', 'dtw_median_column_index': 'dtw_median_row_index',
        'is_stage_endpoint': 'is_time_endpoint',
    })
    best = np.argmin(cost, axis=1)
    finite_rows = np.isfinite(cost).any(axis=1)
    matched_reference_count = endpoint + 1 if available else 0
    return {
        'similarity': similarity,
        'distance': 1 - similarity,
        'path': path_frame, 'matches': stage_matches if query_axis == "rows" else time_matches,
        'stage_matches': stage_matches, 'time_matches': time_matches,
        'status': 'ok' if available else 'unavailable',
        'reason': '' if available else 'no_finite_monotonic_path_through_available_stage_pairs',
        'total_cost': float(accumulated[n, endpoint + 1]) if available else np.nan,
        'normalized_cost': float(normalized_end_costs[endpoint]) if available else np.nan,
        'mean_path_distance': float(path_frame['cosine_distance'].mean()) if available else np.nan,
        'path_length': len(path), 'open_end': bool(open_end), 'step_pattern': 'symmetric2',
        'query_axis': query_axis,
        'query_coverage': 1.0 if available else 0.0,
        'n_matched_reference_timepoints': matched_reference_count,
        'reference_coverage': matched_reference_count / m,
        'reference_start_time': working.columns[0] if available else None,
        'reference_end_time': working.columns[endpoint] if available else None,
        'unmatched_reference_times': json.dumps(working.columns[matched_reference_count:].tolist()),
        'best_match_backward_steps': int(np.sum(np.diff(best[finite_rows]) < 0)),
        'normalization': 'N + matched_reference_length; no time-interval normalization',
        'dtw_rule': ('fixed-start open-end' if open_end else 'closed-end')
                    + ' DTW; symmetric2; diagonal/vertical/horizontal; unavailable pairs forbidden',
    }


def descriptive_effect_size(group_a, group_b):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    median_a = float(np.median(a)) if len(a) else np.nan
    median_b = float(np.median(b)) if len(b) else np.nan
    delta = np.nan
    if len(a) and len(b):
        sorted_b = np.sort(b)
        less = np.searchsorted(sorted_b, a, side="left")
        greater = len(b) - np.searchsorted(sorted_b, a, side="right")
        delta = float(np.sum(less - greater) / (len(a) * len(b)))
    return {"n_a": len(a), "n_b": len(b), "median_a": median_a,
            "median_b": median_b, "median_difference": median_a - median_b,
            "cliffs_delta": delta}


def load_developmental_embeddings(
    embedding_h5ad=None, *, model_path=None, training_config_path=None,
    device="auto", batch_size=256, species_metadata=None,
    species_col="species", time_col="Time", cell_id_col="cell_id",
    cell_type_col="cell_type",
):
    if embedding_h5ad is not None:
        adata = _cell_view(anndata.read_h5ad(embedding_h5ad))
    else:
        if model_path is None or training_config_path is None:
            raise ValueError("Provide embedding_h5ad, or both model_path and training_config_path.")
        from ..data import create_val_loaders
        from ..training import organize_result, validate
        from ..utils.model_loading import load_config, load_chord_model_and_homology, resolve_device

        cfg = load_config(training_config_path)
        model, _, species_config, species_data, species_expr_avg, global_types = (
            load_chord_model_and_homology(cfg, model_path, device=device)
        )
        result = validate(model, create_val_loaders(species_data, batch_size), resolve_device(device))
        adata, _ = organize_result(result, species_config, species_expr_avg, global_types)
        adata = _cell_view(adata)
    _require_obs(adata, [species_col])
    if cell_id_col not in adata.obs:
        adata.obs[cell_id_col] = adata.obs_names.astype(str)
    _require_obs(adata, [cell_id_col])
    join_keys = pd.MultiIndex.from_arrays([
        adata.obs[species_col].astype(str), adata.obs[cell_id_col].astype(str),
    ])
    if join_keys.has_duplicates:
        raise ValueError("species + cell_id must be unique in the embedding file.")
    for species, source in (species_metadata or {}).items():
        mask = adata.obs[species_col].astype(str) == str(species)
        if not mask.any():
            raise ValueError(f"Metadata species {species!r} is absent from embeddings.")
        raw = anndata.read_h5ad(source["h5ad"], backed="r")
        try:
            metadata = raw.obs.copy()
        finally:
            raw.file.close()
        id_source = source.get("cell_id_col")
        ids = metadata[id_source] if id_source else metadata.index
        if pd.isna(ids).any():
            raise ValueError(f"Missing source cell IDs for {species}.")
        metadata.index = pd.Index(ids.astype(str))
        if not metadata.index.is_unique:
            raise ValueError(f"Source cell IDs are not unique for {species}.")
        target_ids = adata.obs.loc[mask, cell_id_col].astype(str)
        missing = set(target_ids) - set(metadata.index)
        if missing:
            raise ValueError(f"{species}: {len(missing)} embedding IDs absent from source metadata.")
        columns = [("time_col", time_col), ("cell_type_col", cell_type_col)]
        for source_key, destination in columns:
            column = source.get(source_key)
            if column is not None:
                if column not in metadata:
                    raise KeyError(f"{species}: source obs has no {column!r}.")
                if destination not in adata.obs:
                    adata.obs[destination] = None
                adata.obs[destination] = adata.obs[destination].astype(object)
                adata.obs.loc[mask, destination] = metadata.loc[target_ids, column].to_numpy()
    _require_obs(adata, [species_col, time_col, cell_type_col])
    return adata


def _write_similarity_matrix(frame, path, row_species):
    frame.to_csv(path, sep="\t", index_label=f"{row_species}_stage")


def _best_match_table(frame, query_axis="rows"):
    if query_axis not in {"rows", "columns"}:
        raise ValueError("query_axis must be 'rows' or 'columns'.")
    working = frame if query_axis == "rows" else frame.T
    values = working.to_numpy(dtype=float)
    available = np.isfinite(values).any(axis=1)
    best = np.argmax(np.where(np.isfinite(values), values, -np.inf), axis=1)
    return pd.DataFrame({
        "stage" if query_axis == "rows" else "time": working.index,
        "best_match": [working.columns[index] if valid else None for index, valid in zip(best, available)],
        "best_match_similarity": [values[i, index] if available[i] else np.nan for i, index in enumerate(best)],
        "status": np.where(available, "available", "unavailable"),
    })


def _full_cell_species_effects(cell_scores, species_pair, time_pairs):
    records = []
    for label, group in cell_scores.groupby("cell_type", sort=True, observed=True):
        a = group[group["species"] == species_pair[0]]
        b = group[group["species"] == species_pair[1]]
        comparisons = [("pooled_observed_transitions", None, None, a, b)]
        for time_a, time_b in dict.fromkeys(time_pairs):
            comparisons.append(("aligned_stage_pair", time_a, time_b,
                                a[a["time"] == time_a], b[b["time"] == time_b]))
        for scope, time_a, time_b, values_a, values_b in comparisons:
            records.append({
                "cell_type": label, "score": "inter_timepoint_dissimilarity", "scope": scope,
                "species_a": species_pair[0], "species_b": species_pair[1],
                "time_a": time_a, "time_b": time_b,
                **descriptive_effect_size(values_a["inter_timepoint_dissimilarity"],
                                          values_b["inter_timepoint_dissimilarity"]),
            })
    return pd.DataFrame(records)


def run_developmental_dissimilarity_analysis(
    *, output_dir, embedding_h5ad=None, model_path=None, training_config_path=None,
    species_metadata=None, device="auto", batch_size=256,
    species_pair=("frog", "fish"), species_col="species", time_col="Time",
    cell_id_col="cell_id", cell_type_col="cell_type",
    embedding_key="X", similarity_matrix_path=None,
    alignment_ranges=None, similarity_matrix_method=None, alignment_transpose_axes=True,
    alignment_query_axis="rows", dtw_open_end=True,
    time_orders=None, make_plots=True, gene_analysis=None,
):
    if len(species_pair) != 2 or species_pair[0] == species_pair[1]:
        raise ValueError("species_pair must contain two different species.")
    if alignment_query_axis not in {"rows", "columns"}:
        raise ValueError("alignment_query_axis must be 'rows' or 'columns'.")
    ranges = {"stage_8_22": (8, 22)} if alignment_ranges is None else alignment_ranges
    if len(ranges) != 1:
        raise ValueError("Configure only the Stage 8-22 alignment range; stage-removal comparisons were removed.")
    adata = load_developmental_embeddings(
        embedding_h5ad, model_path=model_path, training_config_path=training_config_path,
        device=device, batch_size=batch_size, species_metadata=species_metadata,
        species_col=species_col, time_col=time_col, cell_id_col=cell_id_col,
        cell_type_col=cell_type_col,
    )
    adata = adata[adata.obs[species_col].astype(str).isin(species_pair)].copy()
    if set(adata.obs[species_col].astype(str)) != set(species_pair):
        raise ValueError("Both configured species must exist in the cell embeddings.")
    cache = build_transcriptomic_cache(
        adata, species_col=species_col, time_col=time_col,
        cell_type_col=cell_type_col, cell_id_col=cell_id_col, embedding_key=embedding_key,
    )
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"species": sp, "time": time, "cell_type": label,
         "n_cells": group["n_cells"], "centroid_norm": group["centroid_norm"],
         "status": group["status"], "mean_embedding": json.dumps(group["mean_embedding"].tolist())}
        for (sp, time, label), group in cache["groups"].items()
    ]).to_csv(outdir / "centroid_cache_groups.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"species": sp, "time": time, "n_cells": group["n_cells"],
         "centroid_norm": group["centroid_norm"], "status": group["status"],
         "mean_embedding": json.dumps(group["mean_embedding"].tolist())}
        for (sp, time), group in cache["stage_groups"].items()
    ]).to_csv(outdir / "stage_centroid_cache.tsv", sep="\t", index=False)
    if similarity_matrix_path is None:
        components = compute_developmental_similarity(
            adata, *species_pair, time_col=time_col, species_col=species_col,
            embedding_key=embedding_key, time_orders=time_orders, cache=cache, return_components=True,
        )
        similarity = components["similarity"]
        components["coverage"].to_csv(outdir / "developmental_alignment_coverage.tsv", sep="\t", index=False)
    else:
        if similarity_matrix_method != "symmetric_cell_to_stage_centroid_cosine":
            raise ValueError("A supplied matrix must use the all-cell symmetric cell-to-stage-centroid cosine formula.")
        similarity = _similarity_frame(pd.read_csv(similarity_matrix_path, sep=None, engine="python", index_col=0))
        row_order = _ordered_times(similarity.index, (time_orders or {}).get(species_pair[0]))
        column_order = _ordered_times(similarity.columns, (time_orders or {}).get(species_pair[1]))
        similarity = similarity.loc[row_order, column_order]
        for sp, labels in ((species_pair[0], similarity.index), (species_pair[1], similarity.columns)):
            observed = set(adata.obs.loc[adata.obs[species_col].astype(str) == sp, time_col].astype(str))
            if set(labels) - observed:
                raise ValueError(f"Supplied similarity labels do not match {sp} embedding time labels.")
    _write_similarity_matrix(similarity, outdir / "developmental_similarity.tsv", species_pair[0])
    equal_components = compute_developmental_similarity(
        adata, *species_pair, time_col=time_col, species_col=species_col,
        embedding_key=embedding_key, time_orders=time_orders, cache=cache,
        return_components=True, cell_type_weighting="equal", cell_type_col=cell_type_col,
    )
    equal_similarity = equal_components["similarity"]
    equal_components["coverage"].to_csv(outdir / "cell_type_averaged_alignment_coverage.tsv", sep="\t", index=False)
    alignments, alignment_summary, time_pairs = {}, [], []
    for name, limits in ranges.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Alignment range names must be simple filenames.")
        keep = list(similarity.index) if limits is None else [
            label for label in similarity.index if limits[0] <= _numeric_time(label) <= limits[1]
        ]
        subset = similarity.loc[keep]
        equal_subset = equal_similarity.loc[keep, similarity.columns]
        result = dtw_developmental_alignment(subset, open_end=dtw_open_end, query_axis=alignment_query_axis)
        equal_matches = _best_match_table(equal_subset, alignment_query_axis)
        result["cell_type_averaged_similarity"] = equal_subset
        result["cell_type_averaged_matches"] = equal_matches
        alignments[name] = result
        _write_similarity_matrix(subset, outdir / f"alignment_{name}_similarity.tsv", species_pair[0])
        _write_similarity_matrix(equal_subset, outdir / f"alignment_{name}_cell_type_averaged_similarity.tsv", species_pair[0])
        equal_matches.to_csv(outdir / f"alignment_{name}_cell_type_averaged_matches.tsv", sep="\t", index=False)
        result["path"].to_csv(outdir / f"alignment_{name}_dtw_path.tsv", sep="\t", index=False)
        result["matches"].to_csv(outdir / f"alignment_{name}_matches.tsv", sep="\t", index=False)
        result["stage_matches"].to_csv(outdir / f"alignment_{name}_stage_matches.tsv", sep="\t", index=False)
        time_pairs = list(result["path"][["stage", "time"]].itertuples(index=False, name=None))
        alignment_summary.append({
            "range": name, "n_stages": len(subset), "n_times": subset.shape[1],
            "query_species": species_pair[0] if alignment_query_axis == "rows" else species_pair[1],
            "reference_species": species_pair[1] if alignment_query_axis == "rows" else species_pair[0],
            **{key: result[key] for key in (
                "status", "reason", "total_cost", "mean_path_distance", "normalized_cost", "normalization",
                "open_end", "step_pattern", "path_length", "query_coverage", "reference_coverage",
                "reference_start_time", "reference_end_time", "n_matched_reference_timepoints",
                "unmatched_reference_times", "best_match_backward_steps", "dtw_rule",
            )},
        })
        if make_plots:
            plot_developmental_alignment(
                None, *species_pair, similarity_matrix=subset, transpose_axes=alignment_transpose_axes,
                query_axis=alignment_query_axis, dtw_open_end=dtw_open_end, alignment_result=result,
            )
            fig = plt.gcf()
            fig.savefig(outdir / f"alignment_{name}.pdf", bbox_inches="tight")
            fig.savefig(outdir / f"alignment_{name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            plot_developmental_alignment(
                None, *species_pair, similarity_matrix=equal_subset, transpose_axes=alignment_transpose_axes,
                query_axis=alignment_query_axis, show_dtw=False,
                title="Cell-type-averaged stage similarity: best match",
            )
            fig = plt.gcf()
            fig.savefig(outdir / f"alignment_{name}_cell_type_averaged.pdf", bbox_inches="tight")
            fig.savefig(outdir / f"alignment_{name}_cell_type_averaged.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    pd.DataFrame(alignment_summary).to_csv(outdir / "alignment_summary.tsv", sep="\t", index=False)
    baseline = calculate_inter_timepoint_dissimilarity(
        adata, species_col=species_col, time_col=time_col, embedding_key=embedding_key,
        cell_type_col=cell_type_col, matching_mode="cell_type_constrained", time_orders=time_orders, cache=cache,
    )
    baseline.write_h5ad(outdir / "adata_cell_dissimilarity.h5ad")
    summary = _summarize_cell_type_transitions(baseline, cache=cache, time_orders=time_orders, species_order=species_pair)
    summary.to_csv(outdir / "cell_type_stage_summary.tsv", sep="\t", index=False)
    for sp in species_pair:
        filename_species = re.sub(r"[^\w.-]+", "_", str(sp))
        summary.loc[summary["species"] == sp].to_csv(
            outdir / f"cell_type_stage_summary_{filename_species}.tsv", sep="\t", index=False,
        )
    cell_scores = baseline.obs[[species_col, time_col, cell_type_col, cell_id_col,
                                "inter_timepoint_dissimilarity", "dissimilarity_status", "dissimilarity_availability"]].copy()
    cell_scores = cell_scores.rename(columns={species_col: "species", time_col: "time", cell_type_col: "cell_type", cell_id_col: "cell_id"})
    cell_scores.insert(0, "cell_key", baseline.obs_names.astype(str))
    cell_scores.to_csv(outdir / "cell_scores.tsv.gz", sep="\t", index=False)
    effects = _full_cell_species_effects(cell_scores, species_pair, time_pairs)
    effects.to_csv(outdir / "species_effects.tsv", sep="\t", index=False)
    gene_results = None
    if gene_analysis is not None and gene_analysis.get("enabled", False):
        gene_results = run_gene_dissimilarity_analysis(
            baseline, gene_analysis, outdir / "gene_analysis", species_pair=species_pair, make_plots=make_plots,
            species_col=species_col, cell_type_col=cell_type_col, cell_id_col=cell_id_col, time_col=time_col,
            time_orders=time_orders,
        )
    settings = {
        "metric": "inter-time-point transcriptomic dissimilarity", "time_normalization": False,
        "representation": embedding_key, "embedding_dimensions": int(_embedding_array(adata, embedding_key).shape[1]),
        "species_pair": species_pair, "alignment_ranges": ranges,
        "alignment_plot_x_species": species_pair[0] if alignment_transpose_axes else species_pair[1],
        "alignment_plot_y_species": species_pair[1] if alignment_transpose_axes else species_pair[0],
        "alignment_query_axis": alignment_query_axis,
        "dtw_open_end": dtw_open_end, "dtw_start": "fixed first query and reference timepoints",
        "dtw_step_pattern": "symmetric2", "dtw_endpoint_objective": "cost / (N + matched reference length)",
        "similarity_matrix_path": similarity_matrix_path,
        "alignment_matrix_method": "symmetric_cell_to_stage_centroid_cosine",
        "cell_type_averaged_method": "equal_cell_type_weighted_symmetric_cell_to_stage_centroid_cosine",
        "cell_type_averaged_alignment": "best match only; no DTW; no cross-species cell-type matching",
        "matrix_exports": "similarities; explicit source-stage corner header",
        "cell_score_method": "cell_to_next_timepoint_same_cell_type_full_centroid_cosine",
        "sampling": "disabled; all available cells; no seeds, balancing, pool-size modes or density correction",
        "time_orders": time_orders, "cell_type_col": cell_type_col, "cell_type_violin_plots": False,
        "gene_analysis": gene_analysis, "gene_analysis_sampling": "none; once per species and requested cell type",
        "effect_size_direction": f"{species_pair[0]} minus {species_pair[1]}",
        "pooled_effects": "descriptive full-cell comparisons; no developmental-speed interpretation",
    }
    (outdir / "analysis_settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    print(f"Saved developmental alignment and dissimilarity analyses to {outdir}")
    return {"adata": baseline, "cache": cache, "alignments": alignments, "gene_results": gene_results,
            "cell_type_stage_summary": summary, "cell_scores": cell_scores, "species_effects": effects}




def plot_developmental_alignment(
    adata: anndata.AnnData,
    species1: str,
    species2: str,
    time_col: str = "Time",
    species_col: str = "species",
    embedding_key: str = "X",
    figsize: tuple = (5, 5),
    modified_align=None,
    *,
    similarity_matrix=None,
    row_times=None,
    column_times=None,
    show_dtw: bool = True,
    time_orders=None, cache=None, transpose_axes=False,
    dtw_open_end=True, alignment_result=None, query_axis="rows", title=None,
):

    def parse_time(time_str):
        match = re.search(r"\d+", str(time_str))
        return int(match.group()) if match else -1

    if similarity_matrix is None:
        frame = compute_developmental_similarity(
            adata, species1, species2, time_col=time_col,
            species_col=species_col, embedding_key=embedding_key,
            time_orders=time_orders, cache=cache,
        )
    else:
        frame = _similarity_frame(similarity_matrix, row_times, column_times)
    s1_times_sorted = frame.index.tolist()
    s2_times_sorted = frame.columns.tolist()
    similarity_matrix = frame.to_numpy()

    def _map_time_to_frac_index(times_sorted_str, t_num):
        vals = np.array([parse_time(t) for t in times_sorted_str], dtype=float)
        n = len(vals)
        if n == 0:
            return 0.0, True
        if t_num <= vals[0]:
            return 0.0, True
        if t_num >= vals[-1]:
            return float(n - 1), True
        idx = np.searchsorted(vals, t_num)
        i0, i1 = idx - 1, idx
        v0, v1 = vals[i0], vals[i1]
        frac = i0 + (t_num - v0) / (v1 - v0)
        return float(frac), False

    if query_axis not in {"rows", "columns"}:
        raise ValueError("query_axis must be 'rows' or 'columns'.")
    reference_axis = 1 if query_axis == "rows" else 0
    best_match_indices = np.argmax(
        np.where(np.isfinite(similarity_matrix), similarity_matrix, -np.inf), axis=reference_axis,
    ).astype(float)
    best_match_indices[~np.isfinite(similarity_matrix).any(axis=reference_axis)] = np.nan

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    display_matrix = similarity_matrix.T if transpose_axes else similarity_matrix
    hm = sns.heatmap(
        display_matrix,
        xticklabels=s1_times_sorted if transpose_axes else s2_times_sorted,
        yticklabels=s2_times_sorted if transpose_axes else s1_times_sorted,
        cmap="Blues",
        vmin=-1, vmax=1,
        cbar=True,
        cbar_kws={"shrink": 0.8, "label": "Cosine similarity"},
        square=True,
        ax=ax,
    )

    if display_matrix.shape[1] > 0:
        ax.set_box_aspect(display_matrix.shape[0] / display_matrix.shape[1])

    ax.invert_yaxis()
    query_species, reference_species = (species1, species2) if query_axis == "rows" else (species2, species1)
    ax.set_title(title if title is not None else f"Stage-level similarity with alignment: {query_species} → {reference_species}")
    ax.set_xlabel(f"{species1 if transpose_axes else species2} time")
    ax.set_ylabel(f"{species2 if transpose_axes else species1} time")

    ax.tick_params(axis="x", which="both", width=0.6, length=3, labelsize=8)
    ax.tick_params(axis="y", which="both", width=0.6, length=3, labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    cbar = hm.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(width=0.6, length=3, labelsize=8)
        try:
            cbar.outline.set_linewidth(0.6)
        except Exception:
            pass

    if query_axis == "rows":
        x_on_grid, y_on_grid = best_match_indices, np.arange(len(s1_times_sorted))
    else:
        x_on_grid, y_on_grid = np.arange(len(s2_times_sorted)), best_match_indices
    if transpose_axes:
        x_on_grid, y_on_grid = y_on_grid, x_on_grid
    ax.plot(
        x_on_grid + 0.5,
        y_on_grid + 0.5,
        linestyle="--",
        linewidth=2,
        marker="o",
        markersize=4,
        color="#F3D266",
        markerfacecolor="#F3D266",
        markeredgecolor="#F3D266",
        markeredgewidth=0.6,
        zorder=10,
        label="Best match",
        path_effects=[pe.Stroke(linewidth=3.0, foreground="#F3D266"), pe.Normal()],
    )

    if show_dtw:
        alignment = alignment_result
        if alignment is None:
            alignment = dtw_developmental_alignment(frame, open_end=dtw_open_end, query_axis=query_axis)
        elif (not alignment["similarity"].equals(frame)
              or alignment["open_end"] != bool(dtw_open_end)
              or alignment["query_axis"] != query_axis):
            raise ValueError("Supplied DTW result does not match the plotted matrix/boundaries/query axis.")
        path = alignment["path"]
        path_x, path_y = path["column_index"], path["row_index"]
        if transpose_axes:
            path_x, path_y = path_y, path_x
        ax.plot(
            path_x + 0.5, path_y + 0.5,
            color="#D55E00", linewidth=1.5, marker="o", markersize=3,
            label="DTW (open end)" if alignment["open_end"] else "DTW (closed endpoints)", zorder=11,
        )
        if alignment["status"] != "ok":
            ax.text(0.5, 1.08, "DTW unavailable: no finite monotonic path", transform=ax.transAxes,
                    ha="center", fontsize=8)

    if modified_align is not None and len(modified_align) > 0:
        mod_y_real = [parse_time(k) for k in modified_align.keys()]
        mod_x_real = list(modified_align.values())

        mod_y_idx, mod_y_clip = [], []
        mod_x_idx, mod_x_clip = [], []
        for ty in mod_y_real:
            fi, clipped = _map_time_to_frac_index(s1_times_sorted, ty)
            mod_y_idx.append(fi)
            mod_y_clip.append(clipped)
        for tx in mod_x_real:
            fi, clipped = _map_time_to_frac_index(s2_times_sorted, tx)
            mod_x_idx.append(fi)
            mod_x_clip.append(clipped)

        order = np.argsort(mod_y_real)
        X = np.array(mod_x_idx)[order] + 0.5
        Y = np.array(mod_y_idx)[order] + 0.5
        if transpose_axes:
            X, Y = Y, X

        Y_plot = Y.astype(float).copy()

        ax.plot(
            X,
            Y_plot,
            linestyle="-",
            linewidth=1.0,
            marker="o",
            markersize=0,
            color="#FC5252",
            alpha=0.2,
            markerfacecolor="#FC5252",
            markeredgecolor="#FC5252",
            markeredgewidth=0.6,
            zorder=11,
            label="Interpolated alignment",
            path_effects=[pe.Stroke(linewidth=3.0, foreground="#FC5252"), pe.Normal()],
        )

    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return similarity_matrix


def _species_centroid_scores(
    adata_species, *, time_col, embedding_key, cell_type_col="cell_type",
    matching_mode="cell_type_constrained",
    time_order=None, cache=None, species_col="species",
):
    if matching_mode not in {"cell_type_constrained", "unrestricted"}:
        raise ValueError(f"Unknown centroid matching mode: {matching_mode!r}")
    _require_obs(adata_species, [species_col, time_col, cell_type_col])
    cache = _resolve_transcriptomic_cache(
        adata_species, cache, species_col=species_col, time_col=time_col,
        cell_type_col=cell_type_col, embedding_key=embedding_key,
    )
    obs = cache["obs"]
    species_values = obs[species_col].astype(str).to_numpy()
    times = obs[time_col].astype(str).to_numpy()
    normalized = cache["normalized_embeddings"]
    scores = pd.Series(np.nan, index=obs.index, dtype=float)
    statuses = pd.Series("unavailable_no_next_timepoint", index=obs.index, dtype=object)
    groups = cache["groups"]
    for sp in sorted(set(species_values)):
        sorted_times = _ordered_times(times[species_values == sp], (time_order or {}).get(sp) if isinstance(time_order, dict) else time_order)
        for index, time in enumerate(sorted_times[:-1]):
            next_time = sorted_times[index + 1]
            for (group_sp, group_time, label), source in groups.items():
                if group_sp != sp or group_time != time:
                    continue
                positions = source["positions"]
                if matching_mode == "cell_type_constrained":
                    target = groups.get((sp, next_time, label))
                    if target is None:
                        statuses.iloc[positions] = "unavailable_missing_cell_type"
                        continue
                    if target["status"] != "ok":
                        statuses.iloc[positions] = "unavailable_zero_next_centroid"
                        continue
                    values = 1 - np.clip(source["normalized_embeddings"] @ target["normalized_centroid"], -1, 1)
                    scores.iloc[positions] = values
                    statuses.iloc[positions] = "ok"
                    continue
                mask = (species_values == sp) & (times == next_time)
                center = cache["embeddings"][mask].mean(axis=0, dtype=np.float64)
                norm = np.linalg.norm(center)
                if not np.isfinite(norm) or norm == 0:
                    statuses.iloc[positions] = "unavailable_zero_next_centroid"
                    continue
                scores.iloc[positions] = 1 - np.clip(normalized[positions] @ (center / norm), -1, 1)
                statuses.iloc[positions] = "ok"
    return (scores.reindex(adata_species.obs_names).to_numpy(),
            statuses.reindex(adata_species.obs_names).to_numpy())


def calculate_inter_timepoint_dissimilarity(
    adata, *, species_col="species", time_col="Time", embedding_key="X",
    cell_type_col="cell_type",
    matching_mode="cell_type_constrained", time_orders=None, cache=None, copy=True,
):
    cells = _cell_view(adata)
    _require_obs(cells, [species_col, time_col, cell_type_col])
    cache = _resolve_transcriptomic_cache(
        cells, cache, species_col=species_col, time_col=time_col,
        cell_type_col=cell_type_col, embedding_key=embedding_key,
    )
    output = adata.copy() if copy else adata
    if not output.obs_names.is_unique:
        raise ValueError("obs_names must be unique for assigning scores.")
    output.obs["inter_timepoint_dissimilarity"] = np.nan
    output.obs["dissimilarity_status"] = "unavailable_not_a_cell"
    values, statuses = _species_centroid_scores(
        cells, time_col=time_col, embedding_key=embedding_key,
        cell_type_col=cell_type_col,
        matching_mode=matching_mode, time_order=time_orders, cache=cache,
        species_col=species_col,
    )
    output.obs.loc[cells.obs_names, "inter_timepoint_dissimilarity"] = values
    output.obs.loc[cells.obs_names, "dissimilarity_status"] = statuses
    output.obs["dissimilarity_availability"] = np.where(
        np.isfinite(output.obs["inter_timepoint_dissimilarity"].to_numpy(dtype=float)),
        "available", "unavailable",
    )
    return output


def _summarize_cell_type_transitions(adata, *, cache, time_orders=None, species_order=None):
    groups = cache["groups"]
    scores = adata.obs["inter_timepoint_dissimilarity"]
    statuses = adata.obs["dissimilarity_status"]
    species = list(species_order) if species_order is not None else sorted({key[0] for key in groups})
    records = []
    for sp in species:
        times = _ordered_times(
            [key[1] for key in cache["stage_groups"] if key[0] == sp],
            (time_orders or {}).get(sp),
        )
        labels = sorted({key[2] for key in groups if key[0] == sp})
        for index, time in enumerate(times):
            next_time = times[index + 1] if index + 1 < len(times) else None
            for label in labels:
                source = groups.get((sp, time, label))
                target = groups.get((sp, next_time, label)) if next_time is not None else None
                count = source["n_cells"] if source is not None else 0
                values = scores.loc[source["cell_keys"]].to_numpy(dtype=float) if source is not None else np.array([])
                values = values[np.isfinite(values)]
                if not count:
                    reason = "missing_source_cell_type"
                elif next_time is None:
                    reason = "no_next_timepoint"
                elif target is None:
                    reason = "missing_next_cell_type"
                elif target["status"] != "ok":
                    reason = "zero_next_centroid"
                elif not len(values):
                    reason = ";".join(sorted(set(statuses.loc[source["cell_keys"]].astype(str))))
                else:
                    reason = ""
                records.append({
                    "species": sp, "cell_type": label, "time": time, "next_time": next_time,
                    "n_cells": count, "n_next_cells": target["n_cells"] if target is not None else 0,
                    "n_estimated": len(values),
                    "mean_inter_timepoint_dissimilarity": float(np.mean(values)) if len(values) else np.nan,
                    "estimation_coverage": len(values) / count if count else np.nan,
                    "status": "available" if len(values) else "unavailable", "reason": reason,
                })
    return pd.DataFrame(records, columns=[
        "species", "cell_type", "time", "next_time", "n_cells", "n_next_cells",
        "n_estimated", "mean_inter_timepoint_dissimilarity", "estimation_coverage", "status", "reason",
    ])












def gene_dissimilarity_correlation(adata, score_col="inter_timepoint_dissimilarity", *, compute_qvalues=True):
    y = adata.obs[score_col].to_numpy(dtype=float)
    valid = np.isfinite(y)
    y = y[valid]
    X = adata.X[valid]
    if len(y) < 3 or np.var(y) == 0:
        raise ValueError('Gene correlations require >= 3 finite cells and variable scores.')
    if hasattr(X, 'tocsc'):
        X = X.tocsc()
    rows = []
    for index, gene in enumerate(adata.var_names):
        column = X[:, index]
        expression = column.toarray().ravel() if hasattr(column, 'toarray') else np.asarray(column).ravel()
        finite = np.isfinite(expression)
        values, scores = expression[finite], y[finite]
        cor = pval = np.nan
        if len(values) >= 3 and np.var(values) > 0 and np.var(scores) > 0:
            association = spearmanr(values, scores)
            cor, pval = float(association.statistic), float(association.pvalue)
        rows.append({'gene': str(gene), 'cor': cor, 'pval': pval, 'n_cells': len(values)})
    result = pd.DataFrame(rows, columns=['gene', 'cor', 'pval', 'n_cells'])
    if not compute_qvalues:
        return result[['gene', 'cor', 'n_cells']]
    result['q'] = np.nan
    testable = np.isfinite(result['pval'].to_numpy(dtype=float))
    if testable.any():
        result.loc[testable, 'q'] = multipletests(result.loc[testable, 'pval'], method='fdr_bh')[1]
    return result


def _gene_dissimilarity_by_time(cells, cell_types, species_pair, *, species_col, time_col,
                               cell_type_col, score_col, time_orders):
    records = []
    obs = cells.obs
    for species in species_pair:
        species_obs = obs[obs[species_col].astype(str) == species]
        times = _ordered_times(species_obs[time_col].astype(str), (time_orders or {}).get(species))
        for label in cell_types:
            typed = species_obs[species_obs[cell_type_col].astype(str) == label]
            for index, time in enumerate(times):
                next_time = times[index + 1] if index + 1 < len(times) else None
                source = typed[typed[time_col].astype(str) == time]
                n_next = int((typed[time_col].astype(str) == next_time).sum()) if next_time is not None else 0
                values = source[score_col].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values):
                    reason = ""
                elif source.empty:
                    reason = "missing_source_cell_type"
                elif next_time is None:
                    reason = "no_next_timepoint"
                elif not n_next:
                    reason = "missing_next_cell_type"
                elif "dissimilarity_status" in source:
                    reason = ";".join(sorted(set(source["dissimilarity_status"].astype(str))))
                else:
                    reason = "no_finite_dissimilarity"
                records.append({
                    "cell_type": label, "species": species, "time": time, "next_time": next_time,
                    "n_cells": len(source), "n_next_cells": n_next, "n_estimated": len(values),
                    "mean_inter_timepoint_dissimilarity": float(np.mean(values)) if len(values) else np.nan,
                    "median_inter_timepoint_dissimilarity": float(np.median(values)) if len(values) else np.nan,
                    "estimation_coverage": len(values) / len(source) if len(source) else np.nan,
                    "status": "available" if len(values) else "unavailable", "reason": reason,
                })
    return pd.DataFrame(records)


def run_gene_dissimilarity_analysis(
    adata, config, output_dir, *, species_pair=("frog", "fish"), make_plots=True,
    species_col="species", cell_type_col="cell_type", cell_id_col="cell_id", time_col="Time",
    time_orders=None,
):
    if not isinstance(adata, anndata.AnnData):
        raise TypeError("Gene analysis requires full-reference AnnData.")
    labels = config.get("cell_types", config.get("cell_type", "Neuroectoderm"))
    labels = [labels] if isinstance(labels, str) else list(labels)
    if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("gene_analysis.cell_types must contain nonempty cell-type labels.")
    if len(labels) != len(set(labels)):
        raise ValueError("gene_analysis.cell_types must not contain duplicates.")
    cells = _cell_view(adata)
    score = config.get("score_col", "inter_timepoint_dissimilarity")
    _require_obs(cells, [species_col, time_col, cell_type_col, cell_id_col])
    if score not in cells.obs:
        raise ValueError(f"Full-reference AnnData has no score column {score!r}.")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_labels = sorted(set(cells.obs[cell_type_col].astype(str)) | set(labels))
    time_table = _gene_dissimilarity_by_time(
        cells, all_labels, species_pair, species_col=species_col, time_col=time_col,
        cell_type_col=cell_type_col, score_col=score, time_orders=time_orders,
    )
    time_table.to_csv(outdir / "dissimilarity_by_cell_type_species_time.tsv", sep="\t", index=False)
    for species in species_pair:
        species_filename = re.sub(r"[^\w.-]+", "_", species)
        time_table[time_table["species"] == species].to_csv(
            outdir / f"dissimilarity_by_time_{species_filename}.tsv", sep="\t", index=False,
        )
    directories, used_names = [], set()
    for label in labels:
        base = re.sub(r"[^\w.-]+", "_", label).strip(".") or "cell_type"
        safe = base
        counter = 1
        while safe.casefold() in used_names:
            counter += 1
            safe = f"{base}__{counter}"
        used_names.add(safe.casefold())
        directories.append({"cell_type": label, "directory": safe})
    pd.DataFrame(directories).to_csv(outdir / "cell_type_directories.tsv", sep="\t", index=False)
    results = {}
    for item in directories:
        label = item["cell_type"]
        folder = outdir / item["directory"]
        folder.mkdir(parents=True, exist_ok=True)
        typed_time = time_table[time_table["cell_type"] == label].copy()
        typed_time.to_csv(folder / "dissimilarity_by_time.tsv", sep="\t", index=False)
        for species in species_pair:
            species_filename = re.sub(r"[^\w.-]+", "_", species)
            typed_time[typed_time["species"] == species].to_csv(
                folder / f"dissimilarity_by_time_{species_filename}.tsv", sep="\t", index=False,
            )
        single_config = {**config, "cell_type": label,
                         "go_name_cache_dir": outdir / "go_term_names"}
        single_config.pop("cell_types", None)
        typed_cells = cells[cells.obs[cell_type_col].astype(str) == label]
        results[label] = _run_gene_dissimilarity_for_cell_type(
            typed_cells, single_config, folder, species_pair=species_pair, make_plots=make_plots,
            species_col=species_col, cell_type_col=cell_type_col, cell_id_col=cell_id_col, time_col=time_col,
        )
    return {"cell_type_results": results, "time_summary": time_table, "directories": pd.DataFrame(directories)}


def _run_gene_dissimilarity_for_cell_type(
    adata, config, output_dir, *, species_pair=("frog", "fish"), make_plots=True,
    species_col="species", cell_type_col="cell_type", cell_id_col="cell_id", time_col="Time",
):
    if not isinstance(adata, anndata.AnnData):
        raise TypeError('Gene analysis requires the full-reference AnnData, not sampled cell_scores.')
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cell_type = config.get('cell_type', 'Neuroectoderm')
    score = config.get('score_col', 'inter_timepoint_dissimilarity')
    q_threshold = float(config.get('gene_q_threshold', .05))
    label_top_n = config.get('label_top_n', 10)
    if not 0 < q_threshold <= 1:
        raise ValueError('gene_q_threshold must lie in (0, 1].')
    cells = _cell_view(adata)
    _require_obs(cells, [species_col, cell_type_col, cell_id_col, time_col])
    if score not in cells.obs:
        raise ValueError(f'Full-reference AnnData has no score column {score!r}.')
    selected = cells.obs.loc[
        (cells.obs[cell_type_col].astype(str) == cell_type)
        & cells.obs[species_col].astype(str).isin(species_pair)
    ].copy()
    statuses = []

    def record_status(step, status, reason='', **details):
        statuses.append({'step': step, 'status': status, 'reason': reason, **details})
        pd.DataFrame(statuses).to_csv(outdir / 'analysis_status.tsv', sep='\t', index=False)
        print(f'Gene analysis [{step}]: {status}' + (f' ({reason})' if reason else ''), flush=True)

    settings = {
        'sampling': 'none; all valid cells, one correlation per gene and species',
        'score_reference': 'full next-timepoint same-cell-type centroid',
        'cell_type': cell_type, 'score_col': score, 'correlation_method': 'spearman',
        'gene_multiple_testing': 'BH across all testable genes separately within each species, before ortholog filtering',
        'gene_q_threshold': q_threshold, 'gene_set_selection': 'q below threshold in both species; strictly same sign; no absolute-rho cutoff',
        'label_top_n': label_top_n, 'label_ranking': 'abs(rho_species1) + abs(rho_species2); no sign or q filtering',
        'go_method': 'gene-list over-representation with local GMT; separate positive and negative lists',
        'go_multiple_testing': 'BH within each direction', 'config': config,
    }
    (outdir / 'gene_analysis_settings.json').write_text(json.dumps(settings, indent=2, default=str), encoding='utf-8')
    saved_columns = [species_col, time_col, cell_type_col, cell_id_col, score]
    for column in ('dissimilarity_status', 'dissimilarity_availability'):
        if column in selected:
            saved_columns.append(column)
    selected[saved_columns].to_csv(outdir / 'gene_cell_scores.tsv.gz', sep='\t', index=False)
    if selected.empty:
        record_status('gene_correlations', 'unavailable', 'no_cells_for_requested_type', cell_type=cell_type)
        return {}
    correlations, coverage = [], []
    for species in species_pair:
        all_scores = selected[selected[species_col].astype(str) == species]
        scores = all_scores[np.isfinite(all_scores[score].to_numpy(dtype=float))]
        status = 'ok' if len(scores) >= 3 and scores[score].nunique() > 1 else 'insufficient_or_constant_scores'
        coverage.append({'species': species, 'n_total_cells': len(all_scores), 'n_finite_cells': len(scores),
                         'n_timepoints': scores[time_col].nunique(), 'status': status, 'sampling': 'none'})
        if status != 'ok':
            record_status(f'gene_correlations_{species}', 'unavailable', status, n_cells=len(scores))
            continue
        if scores[cell_id_col].astype(str).duplicated().any():
            raise ValueError(f'{species}: full-reference cell IDs must be unique.')
        source_config = config['species'][species]
        if not Path(source_config['h5ad']).is_file():
            record_status(f'gene_correlations_{species}', 'error', 'missing_expression_file', path=str(source_config['h5ad']))
            raise FileNotFoundError(source_config['h5ad'])
        raw = anndata.read_h5ad(source_config['h5ad'])
        if not raw.var_names.is_unique:
            raise ValueError(f'{species}: gene IDs must be unique.')
        id_source = source_config.get('cell_id_col')
        original_ids = raw.obs[id_source] if id_source else raw.obs_names
        if pd.isna(original_ids).any():
            raise ValueError(f'{species}: expression cell IDs contain missing values.')
        ids = pd.Index(original_ids.astype(str))
        if not ids.is_unique:
            raise ValueError(f'{species}: expression cell IDs must be unique.')
        positions = ids.get_indexer(scores[cell_id_col].astype(str))
        if np.any(positions < 0):
            raise ValueError(f'{species}: full-reference scored cells are missing from expression data.')
        view = raw[positions].copy()
        del raw
        if not source_config.get('is_preprocessed', True):
            import scanpy as sc
            sc.pp.normalize_total(view, target_sum=source_config.get('cpm', 1e4))
            sc.pp.log1p(view)
        view.obs[score] = scores[score].to_numpy(dtype=float)
        table = gene_dissimilarity_correlation(view, score, compute_qvalues=True)
        table['species'] = species
        correlations.append(table)
        record_status(f'gene_correlations_{species}', 'ok', n_cells=len(scores),
                      n_testable_genes=int(table['q'].notna().sum()), sampling='none')
        del view
    pd.DataFrame(coverage).to_csv(outdir / 'gene_score_coverage.tsv', sep='\t', index=False)
    if not correlations:
        record_status('gene_correlations', 'unavailable', 'no_estimable_species; see gene_score_coverage.tsv')
        return {}
    correlations = pd.concat(correlations, ignore_index=True)
    correlations.to_csv(outdir / 'gene_correlations.tsv.gz', sep='\t', index=False)
    correlations.to_csv(outdir / 'gene_correlations_summary.tsv', sep='\t', index=False)
    results = {'gene_correlations': correlations, 'coverage': pd.DataFrame(coverage)}
    if not Path(config['homology_path']).is_file():
        record_status('homolog_comparison', 'unavailable', 'missing_homology_file', path=str(config['homology_path']))
        return results
    gene_columns = config.get('homology_columns', {'fish': 'fish_gene', 'frog': 'frog_gene'})
    merged = pd.read_csv(config['homology_path'], sep=None, engine='python', dtype=str)
    for species in species_pair:
        gene_column = gene_columns[species]
        if gene_column not in merged:
            raise KeyError(f'Homology table is missing {gene_column}.')
        table = correlations[correlations['species'] == species].drop(columns='species')
        table = table.rename(columns={'gene': gene_column, **{name: f'{name}_{species}' for name in table if name != 'gene'}})
        merged = merged.merge(table, on=gene_column, how='inner', validate='many_to_one')
    merged = merged.dropna(subset=[f'cor_{sp}' for sp in species_pair])
    merged = merged.drop_duplicates(subset=[gene_columns[sp] for sp in species_pair]).copy()
    x_col, y_col = (f'cor_{sp}' for sp in species_pair)
    merged['association_strength'] = merged[x_col].abs() + merged[y_col].abs()
    merged.to_csv(outdir / 'homolog_correlations.tsv', sep='\t', index=False)
    record_status('homolog_comparison', 'ok' if len(merged) else 'unavailable',
                  '' if len(merged) else 'no_finite_ortholog_pairs', n_ortholog_pairs=len(merged))
    label_species = config.get('label_species', 'fish')
    labels = _representative_ortholog_pairs(merged, x_col, y_col, gene_columns[label_species], label_top_n)
    labels.to_csv(outdir / 'fig5d_labeled_genes.tsv', sep='\t', index=False)
    if make_plots and not merged.empty:
        fig, _ = plot_correlation_comparison(
            merged, x_col, y_col,
            f'{species_pair[0]}: expression–dissimilarity Spearman rho',
            f'{species_pair[1]}: expression–dissimilarity Spearman rho',
            output_filename=outdir / 'homolog_correlations.pdf',
            title=f'{cell_type}: all-cell gene associations', gene_name_col=gene_columns[label_species],
            top_n_labels=label_top_n, show=False,
        )
        fig.savefig(outdir / 'homolog_correlations.png', dpi=300, bbox_inches='tight')
        record_status('homolog_plot', 'ok', n_labels=len(labels))
    else:
        record_status('homolog_plot', 'skipped', 'make_plots_false' if not make_plots else 'no_usable_ortholog_pairs')
    gene_sets = _select_concordant_gene_sets(merged, species_pair, q_threshold)
    for direction, table in gene_sets.items():
        table.to_csv(outdir / f'conserved_{direction}_associations.tsv', sep='\t', index=False)
    results.update({'homolog_correlations': merged, 'gene_sets': gene_sets, 'labeled_genes': labels})
    results['go_results'] = _run_directional_go_enrichment(
        merged, gene_sets, config, outdir, gene_columns, record_status, make_plots=make_plots,
    )
    return results


def _go_term_details(term, description=""):
    term, description = str(term).strip(), str(description).strip()
    match = re.search(r"GO:\d{7}", term) or re.search(r"GO:\d{7}", description)
    go_id = match.group() if match else ""
    for candidate in (description, term):
        if candidate.lower() in {"", "na", "n/a", "nan", "none", "null", "-"}:
            continue
        if candidate.lower().startswith(("http:", "https:", "www.")):
            continue
        name = re.sub(r"GO:\d{7}", "", candidate).strip(" \t|:;()[]-_")
        if name:
            return go_id, name
    return go_id, None


def _read_go_obo_names(path):
    names, current, active = {}, {}, False

    def save_term():
        if current.get("id") and current.get("name"):
            for go_id in [current["id"], *current.get("alt_ids", [])]:
                names[go_id] = current["name"]

    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("["):
                save_term()
                current = {}
                active = line == "[Term]"
            elif active and line.startswith("id: GO:"):
                current["id"] = line[4:]
            elif active and line.startswith("name: "):
                current["name"] = line[6:]
            elif active and line.startswith("alt_id: GO:"):
                current.setdefault("alt_ids", []).append(line[8:])
    save_term()
    return names


def _load_go_gene_sets_and_names(gmt, config, outdir, record_status):
    records, annotated = [], set()
    with open(gmt, encoding="utf-8-sig") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 3 or not fields[0].strip():
                continue
            term, description = fields[:2]
            genes = [gene.strip() for gene in fields[2:] if gene.strip()]
            go_id, name = _go_term_details(term, description)
            records.append({"term": term, "go_id": go_id, "name": name, "genes": genes})
            annotated.update(genes)
    cache_dir = Path(config.get("go_name_cache_dir", Path(outdir) / "go_term_names"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    ontology_names = {}
    candidates = []
    explicit = config.get("obo_path")
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend([Path(gmt).parent / "go-basic.obo", Path(gmt).parent / "go.obo",
                       cache_dir / "go-basic.obo"])
    name_sources = []
    for path in dict.fromkeys(candidates):
        if path.is_file():
            try:
                found = _read_go_obo_names(path)
            except (OSError, UnicodeError) as error:
                record_status("go_term_name_source", "unavailable", "unreadable_obo", path=str(path), detail=str(error))
                continue
            for go_id, name in found.items():
                ontology_names.setdefault(go_id, name)
            name_sources.append(str(path))
    missing = {record["go_id"] for record in records
               if record["go_id"] and record["name"] is None and record["go_id"] not in ontology_names}
    if (missing and config.get("download_go_names", True)
            and str(cache_dir / "go-basic.obo") not in name_sources):
        from urllib.request import Request, urlopen
        import shutil
        url = "https://purl.obolibrary.org/obo/go/go-basic.obo"
        target = cache_dir / "go-basic.obo"
        download_path = cache_dir / "go-basic.obo.download"
        try:
            request = Request(url, headers={"User-Agent": "CHORD-GO-term-names"})
            with urlopen(request, timeout=30) as response, open(download_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
            downloaded = _read_go_obo_names(download_path)
            if not downloaded:
                raise ValueError("Downloaded file contains no GO term names.")
            download_path.replace(target)
            ontology_names.update(downloaded)
            name_sources.append(url)
        except (OSError, UnicodeError, ValueError) as error:
            download_path.unlink(missing_ok=True)
            record_status("go_term_name_download", "unavailable", "download_failed", url=url, detail=str(error))
    term_names, go_ids, mapping_rows = {}, {}, []
    for record in records:
        go_id = record["go_id"]
        name = ontology_names.get(go_id) or record["name"]
        term_names[record["term"]] = name or "Name unavailable"
        go_ids[record["term"]] = go_id
        mapping_rows.append({"GMT_Term": record["term"], "GO_ID": go_id,
                             "Term_Name": name, "status": "available" if name else "unavailable"})
    mapping = pd.DataFrame(mapping_rows, columns=["GMT_Term", "GO_ID", "Term_Name", "status"])
    mapping.to_csv(Path(outdir) / "go_term_names.tsv", sep="\t", index=False)
    with open(Path(outdir) / "go_gene_sets_with_names.gmt", "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            name = term_names[record["term"]]
            known = name != "Name unavailable"
            term = f'{record["go_id"]}|{name}' if record["go_id"] and known else record["term"]
            handle.write("\t".join([term, name if known else "NA", *record["genes"]]) + "\n")
    unresolved = int((mapping["status"] == "unavailable").sum())
    record_status("go_term_names", "ok" if not unresolved else "partial", "" if not unresolved else "unresolved_term_names",
                  n_terms=len(mapping), n_unresolved=unresolved, ontology_sources=json.dumps(name_sources))
    return annotated, term_names, go_ids


def _run_directional_go_enrichment(merged, gene_sets, config, outdir, gene_columns, record_status, *, make_plots):
    gmt = config.get('gmt_path')
    if gmt is None or not Path(gmt).is_file():
        reason = 'gmt_path_is_none' if gmt is None else 'missing_gmt_file'
        for direction in gene_sets:
            record_status(f'go_{direction}', 'unavailable', reason, path=str(gmt))
        return {}
    annotated, term_names, go_ids = _load_go_gene_sets_and_names(gmt, config, outdir, record_status)
    go_species = config.get('go_species', 'fish')
    gene_column = gene_columns[go_species]
    q_columns = [column for column in merged if column.startswith('q_')]
    testable = merged.dropna(subset=q_columns)
    background = sorted(set(testable[gene_column].astype(str)) & annotated)
    pd.Series(background, name='gene', dtype=str).to_csv(outdir / 'go_background_genes.tsv', sep='\t', index=False)
    if not background:
        for direction in gene_sets:
            record_status(f'go_{direction}', 'unavailable', 'no_tested_gene_ids_overlap_gmt')
        return {}
    try:
        import gseapy as gp
    except ModuleNotFoundError as error:
        for direction in gene_sets:
            record_status(f'go_{direction}', 'unavailable', 'missing_dependency', dependency=error.name)
        return {}
    go_fdr = float(config.get('go_fdr', .05))
    go_results = {}
    for direction, pairs in gene_sets.items():
        genes = sorted(set(pairs[gene_column].astype(str)) & set(background))
        pd.Series(genes, name='gene', dtype=str).to_csv(outdir / f'go_selected_genes_{direction}.tsv', sep='\t', index=False)
        if genes:
            enrichment = gp.enrichr(gene_list=genes, gene_sets=str(gmt), background=background,
                                    organism=config.get('organism', 'zebrafish'), outdir=None, cutoff=go_fdr)
            result = enrichment.results.copy()
        else:
            result = pd.DataFrame(columns=['Term', 'Overlap', 'P-value', 'Adjusted P-value', 'Genes'])
        if not result.empty:
            result = result.sort_values('Adjusted P-value').copy()
        result['Term_Name'] = result['Term'].map(term_names).fillna('Name unavailable')
        result['GO_ID'] = result['Term'].map(go_ids).fillna('')
        result['direction'] = direction
        result.to_csv(outdir / f'go_enrichment_{direction}.tsv', sep='\t', index=False)
        significant = result[result['Adjusted P-value'] < go_fdr] if not result.empty else result
        record_status(f'go_{direction}', 'ok' if genes else 'skipped',
                      '' if genes else 'no_q_significant_same_sign_genes_overlap_gmt',
                      n_selected_pairs=len(pairs), n_selected_genes=len(genes),
                      n_background=len(background), n_tested_terms=len(result), n_significant_terms=len(significant))
        go_results[direction] = result
        if make_plots:
            if not significant.empty:
                from ..utils import plot_enrichment_barchart
                fig, _ = plot_enrichment_barchart(
                    significant, top_n=config.get('go_top_n', 12),
                    title=f'GO: {direction} expression–dissimilarity associations', show=False,
                    go_id_col='GO_ID',
                )
            else:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.axis('off')
                text = ('No genes satisfy the q/sign criteria and GMT overlap' if not genes
                        else f'No GO terms meet FDR < {go_fdr:g}')
                ax.text(.5, .5, f'{direction.capitalize()} associations\n{text}\n'
                        f'Selected genes: {len(genes)}; tested terms: {len(result)}',
                        ha='center', va='center', transform=ax.transAxes)
            fig.savefig(outdir / f'go_enrichment_{direction}.pdf', bbox_inches='tight')
            fig.savefig(outdir / f'go_enrichment_{direction}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    pd.concat(list(go_results.values()), ignore_index=True).to_csv(outdir / 'go_enrichment.tsv', sep='\t', index=False)
    return go_results


def _select_concordant_gene_sets(df, species_pair=("frog", "fish"), q_threshold=.05):
    if not 0 < q_threshold <= 1:
        raise ValueError('gene_q_threshold must lie in (0, 1].')
    required = [f'{prefix}_{species}' for species in species_pair for prefix in ('cor', 'q')]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Gene selection requires columns: {sorted(missing)}')
    valid = np.isfinite(df[required].to_numpy(dtype=float)).all(axis=1)
    significant = valid & np.logical_and.reduce([(df[f'q_{sp}'] < q_threshold).to_numpy() for sp in species_pair])
    return {
        'positive': df[significant & np.logical_and.reduce([(df[f'cor_{sp}'] > 0).to_numpy() for sp in species_pair])].copy(),
        'negative': df[significant & np.logical_and.reduce([(df[f'cor_{sp}'] < 0).to_numpy() for sp in species_pair])].copy(),
    }


def _representative_ortholog_pairs(df, x_col, y_col, gene_name_col, top_n):
    if not isinstance(top_n, (int, np.integer)) or top_n < 0:
        raise ValueError('label_top_n must be a nonnegative integer.')
    finite = np.isfinite(df[[x_col, y_col]].to_numpy(dtype=float)).all(axis=1)
    ranked = df.loc[finite].copy()
    ranked['association_strength'] = ranked[x_col].abs() + ranked[y_col].abs()
    columns = ['association_strength']
    ascending = [False]
    if gene_name_col in ranked:
        columns.append(gene_name_col)
        ascending.append(True)
    return ranked.sort_values(columns, ascending=ascending, kind='stable').head(top_n)


def plot_correlation_comparison(
    df, x_col, y_col, x_label, y_label, output_filename=None,
    title="Expression–dissimilarity correlations", gene_name_col="fish_gene",
    top_n_labels=10, show=True,
):
    finite = np.isfinite(df[[x_col, y_col]].to_numpy(dtype=float)).all(axis=1)
    data = df.loc[finite].copy()
    labels = _representative_ortholog_pairs(data, x_col, y_col, gene_name_col, top_n_labels)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)
    ax.scatter(data[x_col], data[y_col], s=13, color='#A6A6A6', alpha=.65, linewidths=0)
    ax.axhline(0, color='0.7', linestyle='--', linewidth=.6, zorder=0)
    ax.axvline(0, color='0.7', linestyle='--', linewidth=.6, zorder=0)
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xlabel=x_label, ylabel=y_label)
    ax.set_title(f'{title}\nN = {len(data)} ortholog pairs')
    for _, row in labels.iterrows():
        name = str(row.get(gene_name_col, ''))
        if not name or name == 'nan':
            continue
        right = row[x_col] >= 0
        ax.annotate(name, xy=(row[x_col], row[y_col]),
                    xytext=(-4 if right else 4, 4), textcoords='offset points',
                    ha='right' if right else 'left', va='bottom', fontsize=8)
    if output_filename is not None:
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax
