from sklearn.metrics import silhouette_score, silhouette_samples
import hashlib
import inspect
import json
import os
from pathlib import Path

import scanpy as sc
import scib.metrics as sm
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from scipy import sparse
from collections.abc import Mapping


_NEIGHBOR_CACHE_KEY = "_chord_evaluation_neighbors"
_TRAJECTORY_REFERENCE_FORMAT_VERSION = 1
_TRAJECTORY_REFERENCE_ALGORITHM = "chord_multispecies_scib_style_deterministic_v1"


def _require_finite_metric(name: str, value) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite: {value}.")
    return value


def _trajectory_neighbors(
        adata,
        *,
        n_neighbors: int,
        seed: int,
        use_rep: str | None = None,
        n_pcs: int | None = None,
) -> None:
    kwargs = {"n_neighbors": int(n_neighbors)}
    if use_rep is not None:
        kwargs["use_rep"] = use_rep
    if n_pcs is not None:
        kwargs["n_pcs"] = int(n_pcs)

    neighbor_parameters = inspect.signature(sc.pp.neighbors).parameters
    if "random_state" in neighbor_parameters:
        kwargs["random_state"] = int(seed)
    elif "rng" in neighbor_parameters:
        kwargs["rng"] = int(seed)

    if "transformer" in neighbor_parameters:
        try:
            from pynndescent import PyNNDescentTransformer
        except ImportError as exc:
            raise ImportError(
                "Deterministic trajectory evaluation requires pynndescent, "
                "which is normally installed with Scanpy."
            ) from exc

        kwargs["transformer"] = PyNNDescentTransformer(
            n_neighbors=int(n_neighbors),
            metric="euclidean",
            random_state=int(seed),
            n_jobs=1,
            low_memory=True,
        )

    sc.pp.neighbors(adata, **kwargs)


def build_trajectory_reference_metadata(
        *,
        config_path,
        raw_data_base,
        species_items,
        start_stage_by_species,
        stage_key_by_species,
        raw_cell_id_key_by_species=None,
        n_top_genes: int = 2000,
        cpm=1e4,
        n_neighbors: int = 30,
        raw_n_comps: int = 50,
        raw_n_pcs: int = 30,
        root_selection_dpt_dims: int = 3,
        seed: int = 0,
):
    config_path = Path(config_path).resolve()
    raw_data_base = Path(raw_data_base).resolve()
    raw_cell_id_key_by_species = raw_cell_id_key_by_species or {}

    raw_inputs = []
    for item in species_items:
        species = str(item["name"])
        raw_path = Path(item["h5ad"])
        if not raw_path.is_absolute():
            raw_path = raw_data_base / raw_path
        raw_path = raw_path.resolve()
        stat = raw_path.stat()
        raw_inputs.append(
            {
                "species": species,
                "path": str(raw_path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "is_preprocessed": bool(item.get("is_preprocessed", True)),
                "start_stage": str(start_stage_by_species[species]),
                "stage_key": str(stage_key_by_species[species]),
                "raw_cell_id_key": raw_cell_id_key_by_species.get(species),
            }
        )

    return {
        "format_version": _TRAJECTORY_REFERENCE_FORMAT_VERSION,
        "algorithm": _TRAJECTORY_REFERENCE_ALGORITHM,
        "config_path": str(config_path),
        "raw_inputs": raw_inputs,
        "preprocessing": {
            "n_top_genes": int(n_top_genes),
            "cpm": None if cpm is None else float(cpm),
        },
        "trajectory": {
            "n_neighbors": int(n_neighbors),
            "raw_n_comps": int(raw_n_comps),
            "raw_n_pcs": int(raw_n_pcs),
            "root_selection_dpt_dims": int(root_selection_dpt_dims),
            "seed": int(seed),
        },
    }


def default_trajectory_reference_cache_path(cache_dir, config_stem, metadata):
    canonical_metadata = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_metadata.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{config_stem}_{digest}.npz"


def save_multispecies_trajectory_reference(reference, path, metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    species_names = list(reference)
    payload = {
        "format_version": np.asarray([_TRAJECTORY_REFERENCE_FORMAT_VERSION]),
        "metadata_json": np.asarray(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "species": np.asarray(species_names, dtype=str),
    }

    for index, species in enumerate(species_names):
        species_reference = reference[species]
        prefix = f"species_{index}"
        payload[f"{prefix}_cell_ids"] = np.asarray(
            species_reference["cell_ids"], dtype=str
        )
        payload[f"{prefix}_pseudotime"] = np.asarray(
            species_reference["pseudotime"], dtype=float
        )
        payload[f"{prefix}_start_cell_ids"] = np.asarray(
            species_reference["start_cell_ids"], dtype=str
        )
        payload[f"{prefix}_start_stage"] = np.asarray(
            str(species_reference["start_stage"])
        )
        payload[f"{prefix}_stage_key"] = np.asarray(
            str(species_reference["stage_key"])
        )
        payload[f"{prefix}_root_cell_id"] = np.asarray(
            str(species_reference["root_cell_id"])
        )

    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_multispecies_trajectory_reference(path, expected_metadata=None):
    path = Path(path)
    with np.load(path, allow_pickle=False) as cache:
        version = int(np.asarray(cache["format_version"]).reshape(-1)[0])
        if version != _TRAJECTORY_REFERENCE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported trajectory-reference cache version {version}; "
                f"expected {_TRAJECTORY_REFERENCE_FORMAT_VERSION}."
            )

        stored_metadata = json.loads(str(np.asarray(cache["metadata_json"]).item()))
        if expected_metadata is not None and stored_metadata != expected_metadata:
            raise ValueError(
                "Trajectory-reference cache metadata does not match the current "
                "raw inputs or evaluation parameters."
            )

        species_names = np.asarray(cache["species"]).astype(str).tolist()
        reference = {}
        for index, species in enumerate(species_names):
            prefix = f"species_{index}"
            cell_ids = np.asarray(cache[f"{prefix}_cell_ids"]).astype(str)
            pseudotime = np.asarray(
                cache[f"{prefix}_pseudotime"], dtype=float
            )
            start_cell_ids = np.asarray(
                cache[f"{prefix}_start_cell_ids"]
            ).astype(str)
            if pseudotime.shape != cell_ids.shape:
                raise ValueError(
                    f"Cached pseudotime and cell IDs have different shapes for "
                    f"species '{species}'."
                )
            reference[species] = {
                "cell_ids": cell_ids,
                "pseudotime": pseudotime,
                "start_cell_ids": start_cell_ids,
                "start_stage": str(
                    np.asarray(cache[f"{prefix}_start_stage"]).item()
                ),
                "stage_key": str(
                    np.asarray(cache[f"{prefix}_stage_key"]).item()
                ),
                "root_cell_id": str(
                    np.asarray(cache[f"{prefix}_root_cell_id"]).item()
                ),
            }

    return reference


def _get_cell_adata(adata):
    if "item_type" not in adata.obs.columns:
        return adata

    cell_mask = (adata.obs["item_type"] == "cell").to_numpy()
    if not np.any(cell_mask):
        raise ValueError("No cells found: adata.obs['item_type'] contains no 'cell' entries.")
    if np.all(cell_mask):
        return adata
    return adata[cell_mask].copy()


def _ensure_neighbors(adata, rep: str, n_neighbors: int, seed: int) -> None:
    cache = adata.uns.get(_NEIGHBOR_CACHE_KEY, {})
    has_graph = "connectivities" in adata.obsp and "distances" in adata.obsp
    cache_matches = (
        cache.get("rep") == rep
        and cache.get("n_neighbors") == n_neighbors
        and cache.get("seed") == seed
    )

    if has_graph and cache_matches:
        return

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=rep,
        random_state=seed,
    )
    adata.uns[_NEIGHBOR_CACHE_KEY] = {
        "rep": rep,
        "n_neighbors": n_neighbors,
        "seed": seed,
    }


def _compute_batch_asw(X, batch_labels, cell_labels) -> float:
    cell_type_scores = []

    for cell_type in pd.unique(cell_labels):
        cell_type_mask = cell_labels == cell_type
        group_batches = batch_labels[cell_type_mask]
        n_samples = int(np.sum(cell_type_mask))
        n_batches = len(np.unique(group_batches))

        if n_batches < 2 or n_batches >= n_samples:
            continue

        group_silhouette = silhouette_samples(X[cell_type_mask], group_batches)
        cell_type_scores.append(float(np.mean(1.0 - np.abs(group_silhouette))))

    if not cell_type_scores:
        return float("nan")
    return float(np.mean(cell_type_scores))


def _mapping_value(value, species: str, argument_name: str):
    if isinstance(value, Mapping):
        if species not in value:
            raise KeyError(
                f"{argument_name} is missing a value for species '{species}'."
            )
        return value[species]
    return value


def _select_extreme_diffmap_root(
        adata,
        candidate_positions,
        dpt_dim: int = 3,
) -> int:
    candidate_positions = np.asarray(candidate_positions, dtype=int)
    if candidate_positions.size == 0:
        raise ValueError("No candidate cells are available for root selection.")
    if "X_diffmap" not in adata.obsm:
        raise KeyError("Root selection requires adata.obsm['X_diffmap'].")

    diffusion_map = np.asarray(adata.obsm["X_diffmap"])
    n_dims = min(int(dpt_dim), diffusion_map.shape[1])
    if n_dims < 1:
        raise ValueError("At least one diffusion-map dimension is required.")

    votes = np.zeros(candidate_positions.size, dtype=int)
    for dim in range(n_dims):
        dimension = diffusion_map[:, dim]
        candidate_values = dimension[candidate_positions]
        if candidate_values.mean() < dimension.mean():
            winner = int(np.argmin(candidate_values))
        else:
            winner = int(np.argmax(candidate_values))
        votes[winner] += 1

    return int(candidate_positions[int(np.argmax(votes))])


def prepare_multispecies_trajectory_reference(
        adata_raw_by_species,
        start_stage_by_species,
        *,
        stage_key_by_species="stage",
        raw_cell_id_key_by_species=None,
        n_neighbors: int = 30,
        raw_n_comps: int = 50,
        raw_n_pcs: int = 30,
        root_selection_dpt_dims: int = 3,
        seed: int = 0,
):
    if not isinstance(adata_raw_by_species, Mapping) or not adata_raw_by_species:
        raise ValueError("adata_raw_by_species must be a non-empty mapping.")
    if not isinstance(start_stage_by_species, Mapping):
        raise TypeError("start_stage_by_species must map species to stage labels.")

    reference = {}
    for species, adata_raw in adata_raw_by_species.items():
        start_stage = _mapping_value(
            start_stage_by_species,
            species,
            "start_stage_by_species",
        )
        stage_key = _mapping_value(
            stage_key_by_species,
            species,
            "stage_key_by_species",
        )

        if stage_key not in adata_raw.obs.columns:
            raise KeyError(
                f"Raw AnnData for species '{species}' has no obs['{stage_key}']."
            )

        raw_cell_id_key = None
        if raw_cell_id_key_by_species is not None:
            raw_cell_id_key = _mapping_value(
                raw_cell_id_key_by_species,
                species,
                "raw_cell_id_key_by_species",
            )

        if raw_cell_id_key is None:
            raw_ids = pd.Index(
                [str(cell_id) for cell_id in adata_raw.obs_names],
                name="cell_id",
            )
        else:
            if raw_cell_id_key not in adata_raw.obs.columns:
                raise KeyError(
                    f"Raw AnnData for species '{species}' has no "
                    f"obs['{raw_cell_id_key}']."
                )
            raw_ids = pd.Index(
                adata_raw.obs[raw_cell_id_key].astype(str),
                name="cell_id",
            )

        if raw_ids.has_duplicates:
            raise ValueError(f"Raw cell IDs for species '{species}' must be unique.")
        if len(raw_ids) < 3:
            raise ValueError(
                f"At least three raw cells are required for species '{species}'."
            )

        start_mask = (
            adata_raw.obs[stage_key]
            .astype(str)
            .eq(str(start_stage))
            .to_numpy()
        )
        start_positions = np.flatnonzero(start_mask)
        if start_positions.size == 0:
            raise ValueError(
                f"No raw cells for species '{species}' have "
                f"obs['{stage_key}'] == {start_stage!r}."
            )

        adata_raw_work = adata_raw.copy()
        sc.pp.scale(adata_raw_work, max_value=10)

        max_raw_n_comps = min(adata_raw_work.n_obs, adata_raw_work.n_vars) - 1
        if max_raw_n_comps < 1:
            raise ValueError(
                f"Raw AnnData for species '{species}' is too small for PCA."
            )
        species_n_comps = min(raw_n_comps, max_raw_n_comps)
        species_n_neighbors = min(n_neighbors, adata_raw_work.n_obs - 1)

        sc.tl.pca(
            adata_raw_work,
            n_comps=species_n_comps,
            svd_solver="arpack",
            random_state=seed,
        )
        _trajectory_neighbors(
            adata_raw_work,
            n_neighbors=species_n_neighbors,
            n_pcs=min(raw_n_pcs, species_n_comps),
            seed=seed,
        )
        sc.tl.diffmap(adata_raw_work, random_state=seed)

        raw_root_position = _select_extreme_diffmap_root(
            adata_raw_work,
            start_positions,
            dpt_dim=root_selection_dpt_dims,
        )
        adata_raw_work.uns["iroot"] = raw_root_position
        sc.tl.dpt(adata_raw_work)

        reference[species] = {
            "cell_ids": raw_ids.to_numpy(dtype=str),
            "pseudotime": adata_raw_work.obs["dpt_pseudotime"].to_numpy(),
            "start_cell_ids": raw_ids[start_mask].to_numpy(dtype=str),
            "start_stage": str(start_stage),
            "stage_key": str(stage_key),
            "root_cell_id": str(raw_ids[raw_root_position]),
        }

    return reference


def evaluate_multispecies_trajectory_conservation(
        adata,
        trajectory_reference,
        *,
        rep: str = "X",
        species_key: str = "species",
        cell_id_key: str = "cell_id",
        n_neighbors: int = 30,
        joint_root_species: str | None = None,
        joint_root_cell_id=None,
        root_selection_dpt_dims: int = 3,
        seed: int = 0,
        zero_failed_species: bool = False,
):
    if not isinstance(trajectory_reference, Mapping) or not trajectory_reference:
        raise ValueError("trajectory_reference must be a non-empty mapping.")
    if (joint_root_species is None) != (joint_root_cell_id is None):
        raise ValueError(
            "joint_root_species and joint_root_cell_id must be provided together."
        )

    adata_method = _get_cell_adata(adata)
    if species_key not in adata_method.obs.columns:
        raise KeyError(f"Integrated AnnData has no obs['{species_key}'].")
    if cell_id_key not in adata_method.obs.columns:
        raise KeyError(
            f"Integrated AnnData is missing adata.obs['{cell_id_key}']; "
            "regenerate CHORD output with cell ID preservation enabled."
        )

    joint_positions = []
    joint_start_positions = []
    aligned_raw_pseudotime = {}
    species_slices = {}
    offset = 0

    for species, raw_reference in trajectory_reference.items():
        species_mask = (
            adata_method.obs[species_key].astype(str).eq(str(species)).to_numpy()
        )
        species_positions = np.flatnonzero(species_mask)
        if species_positions.size == 0:
            raise ValueError(
                f"No integrated cells were found for species '{species}'."
            )

        method_ids = pd.Index(
            adata_method.obs.iloc[species_positions][cell_id_key].astype(str),
            name="cell_id",
        )
        raw_ids = pd.Index(raw_reference["cell_ids"].astype(str), name="cell_id")
        start_ids = pd.Index(
            raw_reference["start_cell_ids"].astype(str),
            name="cell_id",
        )

        if method_ids.has_duplicates:
            raise ValueError(
                f"Integrated cell IDs for species '{species}' must be unique."
            )
        if raw_ids.has_duplicates:
            raise ValueError(
                f"Raw reference cell IDs for species '{species}' must be unique."
            )

        common_ids = method_ids[method_ids.isin(raw_ids)]
        if len(common_ids) < 3:
            raise ValueError(
                f"At least three shared cells are required for species '{species}'."
            )

        method_local_positions = method_ids.get_indexer(common_ids)
        raw_positions = raw_ids.get_indexer(common_ids)
        aligned_method_positions = species_positions[method_local_positions]

        joint_positions.extend(aligned_method_positions.tolist())
        aligned_raw_pseudotime[species] = np.asarray(
            raw_reference["pseudotime"]
        )[raw_positions]
        species_slices[species] = slice(offset, offset + len(common_ids))

        start_mask = common_ids.isin(start_ids)
        if not np.any(start_mask):
            raise ValueError(
                f"No starting-stage cells for species '{species}' are shared "
                "between the raw reference and integrated AnnData."
            )
        joint_start_positions.extend(
            (offset + np.flatnonzero(start_mask)).tolist()
        )
        offset += len(common_ids)

    adata_joint = adata_method[np.asarray(joint_positions, dtype=int)].copy()
    joint_n_neighbors = min(n_neighbors, adata_joint.n_obs - 1)
    _trajectory_neighbors(
        adata_joint,
        n_neighbors=joint_n_neighbors,
        use_rep=rep,
        seed=seed,
    )
    sc.tl.diffmap(adata_joint, random_state=seed)

    if joint_root_cell_id is None:
        joint_root_position = _select_extreme_diffmap_root(
            adata_joint,
            joint_start_positions,
            dpt_dim=root_selection_dpt_dims,
        )
    else:
        root_mask = (
            adata_joint.obs[species_key]
            .astype(str)
            .eq(str(joint_root_species))
            .to_numpy()
            & adata_joint.obs[cell_id_key]
            .astype(str)
            .eq(str(joint_root_cell_id))
            .to_numpy()
        )
        root_positions = np.flatnonzero(root_mask)
        if root_positions.size != 1:
            raise ValueError(
                "The explicit joint root must identify exactly one aligned "
                f"cell; found {root_positions.size}."
            )
        joint_root_position = int(root_positions[0])
        if joint_root_position not in set(joint_start_positions):
            raise ValueError(
                "The explicit joint root is not in the configured starting "
                "stage for its species."
            )

    adata_joint.uns["iroot"] = joint_root_position
    sc.tl.dpt(adata_joint)
    joint_pseudotime = adata_joint.obs["dpt_pseudotime"].to_numpy()

    species_correlations = {}
    species_scores = {}
    aligned_cells = {}
    zero_score_species = []
    for species, species_slice in species_slices.items():
        pt_raw = aligned_raw_pseudotime[species]
        pt_integrated = joint_pseudotime[species_slice]
        try:
            rho, _ = spearmanr(pt_raw, pt_integrated, nan_policy="omit")
            rho = _require_finite_metric(
                f"trajectory Spearman rho for species '{species}'",
                rho,
            )
        except ValueError:
            if not zero_failed_species:
                raise

            raw_values = np.asarray(pt_raw, dtype=float)
            integrated_values = np.asarray(pt_integrated, dtype=float)
            finite_integrated = integrated_values[np.isfinite(integrated_values)]
            raw_reference_valid = (
                raw_values.size >= 3
                and np.isfinite(raw_values).all()
                and np.unique(raw_values).size >= 2
            )
            integrated_pseudotime_unusable = (
                finite_integrated.size < 3
                or np.unique(finite_integrated).size < 2
            )
            if not raw_reference_valid or not integrated_pseudotime_unusable:
                raise

            rho = float("nan")
            species_scores[species] = 0.0
            zero_score_species.append(species)
            print(
                f"{species}: integrated trajectory is not estimable "
                f"(finite pseudotime cells={finite_integrated.size}/"
                f"{integrated_values.size}, "
                f"distinct finite values={np.unique(finite_integrated).size}); "
                "assigning trajectory score 0 with a valid raw reference."
            )
        else:
            species_scores[species] = float((rho + 1) / 2)
        species_correlations[species] = rho
        aligned_cells[species] = int(species_slice.stop - species_slice.start)

    mean_rho = float(np.mean(list(species_correlations.values())))
    if zero_score_species:
        trajectory_conservation = float(np.mean(list(species_scores.values())))
    else:
        trajectory_conservation = float((mean_rho + 1) / 2)
    root_obs = adata_joint.obs.iloc[joint_root_position]
    joint_root_species = str(root_obs[species_key])
    joint_root_cell_id = str(root_obs[cell_id_key])

    print(f"Joint trajectory cells: {adata_joint.n_obs}")
    print(
        f"Joint root: species={joint_root_species}, "
        f"cell_id={joint_root_cell_id}"
    )
    for species in trajectory_reference:
        print(
            f"{species}: aligned={aligned_cells[species]}, "
            f"Spearman rho={species_correlations[species]:.4f}, "
            f"trajectory={species_scores[species]:.4f}"
        )
    print(f"Macro trajectory conservation: {trajectory_conservation:.4f}")

    return {
        "trajectory_conservation": trajectory_conservation,
        "mean_spearman": mean_rho,
        "species_scores": species_scores,
        "species_correlations": species_correlations,
        "aligned_cells": aligned_cells,
        "raw_root_cell_ids": {
            species: str(reference["root_cell_id"])
            for species, reference in trajectory_reference.items()
        },
        "joint_root_species": joint_root_species,
        "joint_root_cell_id": joint_root_cell_id,
    }


def evaluate_trajectory_conservation(
        adata,
        adata_raw,
        root_cell_id,
        *,
        species: str | None = None,
        rep: str = "X",
        cell_id_key: str = "cell_id",
        raw_cell_id_key: str | None = None,
        n_neighbors: int = 30,
        raw_n_comps: int = 50,
        raw_n_pcs: int = 30,
        seed: int = 0,
):
    adata_method = _get_cell_adata(adata)

    if species is not None:
        if "species" not in adata_method.obs.columns:
            raise KeyError("trajectory species was provided but adata.obs has no 'species' column.")
        species_mask = (adata_method.obs["species"] == species).to_numpy()
        if not np.any(species_mask):
            raise ValueError(f"No cells found for trajectory species '{species}'.")
        adata_method = adata_method[species_mask].copy()
    else:
        if "species" in adata_method.obs.columns and adata_method.obs["species"].nunique() > 1:
            raise ValueError(
                "trajectory_species is required when the integrated AnnData "
                "contains more than one species."
            )
        adata_method = adata_method.copy()

    if cell_id_key not in adata_method.obs.columns:
        raise KeyError(
            f"Integrated AnnData is missing adata.obs['{cell_id_key}']; "
            "regenerate CHORD output with cell ID preservation enabled."
        )

    method_ids = pd.Index(
        adata_method.obs[cell_id_key].astype(str),
        name="cell_id",
    )
    if raw_cell_id_key is None:
        raw_ids = pd.Index(
            [str(cell_id) for cell_id in adata_raw.obs_names],
            name="cell_id",
        )
    else:
        if raw_cell_id_key not in adata_raw.obs.columns:
            raise KeyError(f"Raw AnnData is missing adata.obs['{raw_cell_id_key}'].")
        raw_ids = pd.Index(
            adata_raw.obs[raw_cell_id_key].astype(str),
            name="cell_id",
        )

    if method_ids.has_duplicates:
        raise ValueError("Integrated cell IDs must be unique within the trajectory species.")
    if raw_ids.has_duplicates:
        raise ValueError("Raw cell IDs must be unique.")

    common_ids = method_ids[method_ids.isin(raw_ids)]
    if len(common_ids) < 3:
        raise ValueError(
            "At least three shared cells are required after aligning integrated "
            "and raw AnnData objects by cell ID."
        )

    method_positions = method_ids.get_indexer(common_ids)
    raw_positions = raw_ids.get_indexer(common_ids)
    adata_method = adata_method[method_positions].copy()
    adata_raw_work = adata_raw[raw_positions].copy()

    root_cell_id = str(root_cell_id)
    if root_cell_id not in common_ids:
        raise ValueError(
            f"Root cell ID '{root_cell_id}' is not shared by the integrated "
            "and raw AnnData objects."
        )
    root_position = int(common_ids.get_loc(root_cell_id))
    adata_method.uns["iroot"] = root_position
    adata_raw_work.uns["iroot"] = root_position

    trajectory_n_neighbors = min(n_neighbors, len(common_ids) - 1)

    _trajectory_neighbors(
        adata_method,
        n_neighbors=trajectory_n_neighbors,
        use_rep=rep,
        seed=seed,
    )
    sc.tl.diffmap(adata_method, random_state=seed)
    sc.tl.dpt(adata_method)

    sc.pp.scale(adata_raw_work, max_value=10)
    max_raw_n_comps = min(adata_raw_work.n_obs, adata_raw_work.n_vars) - 1
    if max_raw_n_comps < 1:
        raise ValueError("Raw AnnData is too small to compute PCA for trajectory evaluation.")
    raw_n_comps = min(raw_n_comps, max_raw_n_comps)
    sc.tl.pca(
        adata_raw_work,
        n_comps=raw_n_comps,
        svd_solver="arpack",
        random_state=seed,
    )
    _trajectory_neighbors(
        adata_raw_work,
        n_neighbors=trajectory_n_neighbors,
        n_pcs=min(raw_n_pcs, raw_n_comps),
        seed=seed,
    )
    sc.tl.diffmap(adata_raw_work, random_state=seed)
    sc.tl.dpt(adata_raw_work)

    pt_method = adata_method.obs["dpt_pseudotime"].to_numpy()
    pt_raw = adata_raw_work.obs["dpt_pseudotime"].to_numpy()
    rho, _ = spearmanr(pt_raw, pt_method, nan_policy="omit")
    rho = _require_finite_metric("trajectory Spearman rho", rho)
    trajectory_conservation = float((rho + 1) / 2)

    print(f"Aligned trajectory cells: {len(common_ids)}")
    print(f"Spearman rho: {rho:.4f}")
    print(f"Trajectory conservation: {trajectory_conservation:.4f}")
    return trajectory_conservation


def evaluate_celltype(
        adata,
        rep: str = "X",
        n_neighbors: int = 15,
        seed: int = 0,
        gammas=np.arange(0.1, 2.05, 0.1),
        calculate_trajectory: bool = False,
        adata_raw=None,
        trajectory_species: str | None = None,
        trajectory_root_cell_id=None,
        trajectory_cell_id_key: str = "cell_id",
        raw_cell_id_key: str | None = None,
        trajectory_reference=None,
        trajectory_joint_root_species: str | None = None,
        trajectory_joint_root_cell_id=None,
):
    adata_work = _get_cell_adata(adata)

    _ensure_neighbors(adata_work, rep=rep, n_neighbors=n_neighbors, seed=seed)

    best_nmi = -np.inf
    best_ari = np.nan
    best_resolution = None
    for gamma in gammas:
        key = f"leiden_{gamma:.2f}"
        sc.tl.leiden(adata_work, resolution=gamma, key_added=key, random_state=seed)
        current_nmi = _require_finite_metric(
            f"NMI at Leiden resolution {gamma:g}",
            sm.nmi(adata_work, key, "cell_type"),
        )
        current_ari = _require_finite_metric(
            f"ARI at Leiden resolution {gamma:g}",
            sm.ari(adata_work, key, "cell_type"),
        )

        if current_nmi > best_nmi:
            best_nmi = current_nmi
            best_ari = current_ari
            best_resolution = float(gamma)

    if best_resolution is None:
        raise ValueError("gammas must contain at least one Leiden resolution.")

    X_use = adata_work.obsm[rep] if rep != "X" else adata_work.X
    uniq, cnt = np.unique(adata_work.obs["cell_type"], return_counts=True)
    mask = np.isin(adata_work.obs["cell_type"], uniq[cnt > 1])
    asw = _require_finite_metric(
        "cell-type ASW",
        (silhouette_score(X_use[mask], adata_work.obs["cell_type"][mask]) + 1) / 2,
    )

    print(
        f"NMI={best_nmi:.4f} | ASW={asw:.4f} | ARI={best_ari:.4f} "
        f"| Leiden resolution={best_resolution:g}"
    )

    if calculate_trajectory:
        if trajectory_reference is not None:
            trajectory_result = evaluate_multispecies_trajectory_conservation(
                adata_work,
                trajectory_reference,
                rep=rep,
                cell_id_key=trajectory_cell_id_key,
                joint_root_species=trajectory_joint_root_species,
                joint_root_cell_id=trajectory_joint_root_cell_id,
                seed=seed,
            )
            trajectory_conservation = trajectory_result["trajectory_conservation"]
        else:
            if adata_raw is None:
                raise ValueError(
                    "adata_raw or trajectory_reference is required when "
                    "calculate_trajectory=True."
                )
            if trajectory_root_cell_id is None:
                raise ValueError(
                    "trajectory_root_cell_id is required for single-species "
                    "trajectory evaluation."
                )
            trajectory_conservation = evaluate_trajectory_conservation(
                adata_work,
                adata_raw,
                trajectory_root_cell_id,
                species=trajectory_species,
                rep=rep,
                cell_id_key=trajectory_cell_id_key,
                raw_cell_id_key=raw_cell_id_key,
                seed=seed,
            )
        return best_nmi, asw, best_ari, trajectory_conservation

    return best_nmi, asw, best_ari


def evaluate_batch(
        adata,
        batch_key: str = "species",
        rep: str = "X",
        n_neighbors: int = 15,
        seed: int = 0,
):
    adata_cells = _get_cell_adata(adata)

    if not isinstance(adata_cells.obs["cell_type"].dtype, pd.CategoricalDtype):
        adata_cells.obs["cell_type"] = pd.Categorical(
            adata_cells.obs["cell_type"]
        )

    X = adata_cells.X if rep == "X" else adata_cells.obsm[rep]
    batch_labels = adata_cells.obs[batch_key].to_numpy()
    cell_labels = adata_cells.obs["cell_type"].to_numpy()

    batch_asw = _require_finite_metric(
        "batch ASW",
        _compute_batch_asw(X, batch_labels, cell_labels),
    )

    _ensure_neighbors(
        adata_cells,
        rep=rep,
        n_neighbors=n_neighbors,
        seed=seed,
    )
    gc = _require_finite_metric(
        "graph connectivity",
        sm.graph_connectivity(adata_cells, label_key="cell_type"),
    )

    print(f"Batch ASW: {batch_asw:.4f}, Graph Connectivity: {gc:.4f}")
    return batch_asw, gc


