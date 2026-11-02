
from __future__ import annotations

from typing import Any

import numpy as np


HOLDOUT_KEY = "cell_type_holdout"


def get_cell_type_holdout_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    interventions = cfg.get("interventions", {})
    if not isinstance(interventions, dict):
        raise ValueError("interventions must be a dictionary.")
    block = interventions.get(HOLDOUT_KEY)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("interventions.cell_type_holdout must be a dictionary.")
    enabled = block.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("interventions.cell_type_holdout.enabled must be a boolean.")
    if not enabled:
        return None

    conflicts = [
        f"interventions.{name}"
        for name, other in interventions.items()
        if name != HOLDOUT_KEY and isinstance(other, dict)
        and other.get("enabled") is True
    ]
    if "osr" in cfg:
        conflicts.append("osr")
    hierarchy = cfg.get("training", {}).get("hierarchy", {})
    if isinstance(hierarchy, dict) and hierarchy.get("enabled") is True:
        conflicts.append("training.hierarchy")
    if conflicts:
        raise ValueError(
            "interventions.cell_type_holdout conflicts with: "
            + ", ".join(sorted(conflicts))
        )

    unsupported = set(block) - {"enabled", "seed", "n_cell_types"}
    if unsupported:
        raise ValueError(f"Unsupported cell_type_holdout settings: {sorted(unsupported)}")
    seed = block.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("interventions.cell_type_holdout.seed must be a non-negative integer.")
    count = block.get("n_cell_types")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("interventions.cell_type_holdout.n_cell_types must be an integer >= 2.")
    return {"enabled": True, "seed": seed, "n_cell_types": count}


def _preprocess_holdout_features(
    species_adatas, *, raw_species, cpm, select_hvg, n_top_genes,
):
    if not raw_species and not select_hvg:
        return species_adatas
    import scanpy as sc

    processed = {}
    for species, (adata, label_col) in species_adatas.items():
        train_mask = adata.obs["split"].astype(str).to_numpy() == "train"
        if species in raw_species:
            train_adata = adata[train_mask].copy()
            sc.pp.filter_genes(train_adata, min_cells=10)
            if train_adata.n_vars == 0:
                raise ValueError(f"[{species}] training-only gene filtering retained no genes.")
            adata = adata[:, train_adata.var_names].copy()
            sc.pp.normalize_total(adata, target_sum=cpm)
            sc.pp.log1p(adata)
        if select_hvg:
            train_adata = adata[train_mask].copy()
            sc.pp.highly_variable_genes(train_adata, flavor="seurat", n_top_genes=n_top_genes)
            genes = train_adata.var_names[train_adata.var["highly_variable"].to_numpy()]
            if len(genes) == 0:
                raise ValueError(f"[{species}] training-only HVG selection retained no genes.")
            adata = adata[:, genes].copy()
        processed[species] = (adata, label_col)
    return processed


def prepare_cell_type_holdout_views(
    species_adatas,
    holdout_config,
    *,
    n_top_genes=2000,
    select_hvg=True,
    raw_species=(),
    cpm=10000,
):
    options = get_cell_type_holdout_config({
        "interventions": {HOLDOUT_KEY: {**holdout_config, "enabled": True}},
    })
    if len(species_adatas) < 2:
        raise ValueError("Cell-type hold-out requires at least two species.")
    labels_by_species = {}
    for species, (adata, label_col) in species_adatas.items():
        if not adata.obs_names.is_unique:
            raise ValueError(f"[{species}] hold-out requires unique cell IDs.")
        if label_col not in adata.obs:
            raise ValueError(f"[{species}] missing label column {label_col!r}.")
        if adata.n_obs == 0 or adata.obs[label_col].isna().any():
            raise ValueError(f"[{species}] cell-type labels are empty or contain missing values.")
        labels = adata.obs[label_col].astype(str)
        if labels.str.strip().eq("").any():
            raise ValueError(f"[{species}] cell-type labels contain empty names.")
        labels_by_species[species] = labels

    candidates = sorted(set.intersection(*(set(labels) for labels in labels_by_species.values())))
    count = options["n_cell_types"]
    if count > len(candidates):
        raise ValueError(
            f"Requested {count} hold-out cell types, but only {len(candidates)} "
            "occur in every species after cell QC."
        )
    selected = sorted(np.random.default_rng(options["seed"]).choice(
        np.asarray(candidates, dtype=object), size=count, replace=False,
    ).tolist())
    full = {}
    for species, (adata, label_col) in species_adatas.items():
        labels = labels_by_species[species]
        unknown = labels.isin(selected).to_numpy()
        if unknown.all():
            raise ValueError(f"[{species}] hold-out leaves no training cells; reduce n_cell_types.")
        annotated = adata.copy()
        annotated.obs[label_col] = labels.to_numpy()
        annotated.obs["true_label"] = labels.to_numpy()
        annotated.obs["split"] = np.where(unknown, "test", "train")
        annotated.obs["known_status"] = np.where(unknown, "unknown", "known")
        full[species] = (annotated, label_col)

    full = _preprocess_holdout_features(
        full, raw_species=set(raw_species), cpm=cpm,
        select_hvg=select_hvg, n_top_genes=n_top_genes,
    )
    training = {
        species: (adata[adata.obs["split"].astype(str).eq("train").to_numpy()].copy(), label_col)
        for species, (adata, label_col) in full.items()
    }
    metadata = {
        "seed": options["seed"], "n_cell_types": count,
        "candidate_cell_types": candidates, "cell_types": selected,
        "species": sorted(species_adatas),
    }
    return full, training, metadata


def build_cell_type_holdout_views(cfg):
    options = get_cell_type_holdout_config(cfg)
    if options is None:
        raise ValueError("Cell-type hold-out is not enabled.")
    import scanpy as sc

    species_adatas, raw_species = {}, set()
    for item in cfg["data"]["species"]:
        species = item["name"]
        adata = sc.read_h5ad(item["h5ad"])
        if not item.get("is_preprocessed", True):
            sc.pp.filter_cells(adata, min_counts=200)
            raw_species.add(species)
        species_adatas[species] = (adata, item["label_col"])
    return prepare_cell_type_holdout_views(
        species_adatas, options, n_top_genes=cfg["data"].get("n_HVG", 2000),
        select_hvg=cfg["data"].get("select_hvg", True), raw_species=raw_species,
        cpm=cfg["data"].get("cpm", 10000),
    )


def select_cell_type_holdout_cells(adata):
    required = {"species", "cell_type", "split", "known_status"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Hold-out embeddings are missing obs columns: {sorted(missing)}")
    cells = (adata.obs["item_type"].astype(str).eq("cell").to_numpy()
             if "item_type" in adata.obs else np.ones(adata.n_obs, dtype=bool))
    obs = adata.obs.loc[cells]
    if obs[list(required)].isna().any().any():
        raise ValueError("Hold-out embedding annotations contain missing values.")
    split = obs["split"].astype(str)
    status = obs["known_status"].astype(str)
    if not ((split.eq("train") & status.eq("known"))
            | (split.eq("test") & status.eq("unknown"))).all():
        raise ValueError("Hold-out cells must be train/known or test/unknown.")
    selected = adata[cells & adata.obs["known_status"].astype(str).eq("unknown").to_numpy()].copy()
    labels = selected.obs["cell_type"].astype(str)
    species = selected.obs["species"].astype(str)
    all_species = set(obs["species"].astype(str))
    if len(all_species) < 2 or set(species) != all_species or labels.nunique() < 2:
        raise ValueError("Hold-out evaluation requires >= 2 held-out types in every species.")
    expected_types = set(labels)
    if any(set(labels[species.eq(name)]) != expected_types for name in all_species):
        raise ValueError("Every held-out cell type must occur in every evaluation species.")
    if obs.loc[status.eq("known"), "cell_type"].astype(str).isin(expected_types).any():
        raise ValueError("Held-out cell types must not occur among training cells.")
    metadata = adata.uns.get(HOLDOUT_KEY, {})
    if "species" in metadata and set(metadata["species"]) != all_species:
        raise ValueError("Hold-out evaluation must contain all species recorded in the metadata.")
    if "cell_types" in metadata and set(metadata["cell_types"]) != expected_types:
        raise ValueError("Hold-out metadata does not match the embedding cell types.")
    return selected
