from __future__ import annotations

import argparse
import copy
from itertools import combinations
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "fish_frog.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pseudo_species_benchmark"
DEFAULT_CELLS_PER_SPECIES = (30_000,)
DEFAULT_K_VALUES = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixed-cell-count pseudo-species inputs for CHORD "
            "scalability benchmarks."
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help=(
            "A normal CHORD training config used as the model/training template. "
            "Its data, output, OSR, and intervention settings are replaced. "
            f"Default: {DEFAULT_BASE_CONFIG}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "New directory in which all benchmark scenarios are written. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--fish-h5ad",
        type=Path,
        default=None,
        help="Fish input h5ad. Defaults to the fish path in --base-config.",
    )
    parser.add_argument(
        "--frog-h5ad",
        type=Path,
        default=None,
        help="Frog input h5ad. Defaults to the frog path in --base-config.",
    )
    parser.add_argument(
        "--homology-csv",
        type=Path,
        default=None,
        help=(
            "Fish-frog homology CSV. Defaults to the matching pair in "
            "--base-config."
        ),
    )
    parser.add_argument(
        "--fish-name",
        default="fish",
        help="Fish species name used by the source config and homology columns.",
    )
    parser.add_argument(
        "--frog-name",
        default="frog",
        help="Frog species name used by the source config and homology columns.",
    )
    parser.add_argument(
        "--fish-label-col",
        default=None,
        help="Fish cell-type column. Defaults to the source config value.",
    )
    parser.add_argument(
        "--frog-label-col",
        default=None,
        help="Frog cell-type column. Defaults to the source config value.",
    )
    cell_count_group = parser.add_mutually_exclusive_group()
    cell_count_group.add_argument(
        "--cells-per-species",
        nargs="+",
        type=int,
        default=None,
        help=(
            "One or more cell counts to retain independently for both real "
            "species, for example: 10000 20000 63360. If neither cell-count "
            "option is given, the default is 30000 per species."
        ),
    )
    cell_count_group.add_argument(
        "--total-cells",
        nargs="+",
        type=int,
        default=None,
        help=(
            "One or more even total cell-row counts across fish and frog. "
            "Each total is divided equally between the two real species."
        ),
    )
    parser.add_argument(
        "--allow-cell-repetition",
        action="store_true",
        help=(
            "Allow requested counts to exceed the available unique cells by "
            "concatenating independently shuffled full-source cycles. Disabled "
            "by default. Intended only for computational scaling benchmarks."
        ),
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="Numbers of pseudo-species per real species. Default: 1 2 3 4 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for source downsampling and pseudo-species assignment.",
    )
    parser.add_argument(
        "--n-hvg",
        "--expected-genes",
        dest="n_hvg",
        type=int,
        default=None,
        help=(
            "Select this many HVGs independently from each complete real-species "
            "dataset before cell sampling and pseudo-species splitting. Defaults "
            "to data.n_HVG in --base-config (or 2000 if that key is absent). "
            "--expected-genes is retained as a backward-compatible alias."
        ),
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Base config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Base config must contain a top-level mapping.")
    for key in ("model", "training", "data", "output"):
        if key not in cfg:
            raise KeyError(f"Base config is missing required top-level key {key!r}.")
    return cfg


def _resolve_config_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _source_species_item(cfg: dict[str, Any], species: str) -> dict[str, Any]:
    matches = [
        item
        for item in cfg["data"].get("species", [])
        if str(item.get("name")) == species
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {species!r} entry in data.species; "
            f"found {len(matches)}."
        )
    return matches[0]


def _source_homology_item(
    cfg: dict[str, Any], species1: str, species2: str
) -> dict[str, Any]:
    expected = {species1, species2}
    matches = [
        item
        for item in cfg["data"].get("homology", [])
        if {str(item.get("species1")), str(item.get("species2"))} == expected
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one homology entry for {sorted(expected)}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _validate_source_adata(
    adata: ad.AnnData,
    *,
    species: str,
    label_col: str,
) -> None:
    if adata.n_obs == 0:
        raise ValueError(f"{species!r} contains no cells.")
    if adata.n_vars == 0:
        raise ValueError(f"{species!r} contains no genes.")
    if label_col not in adata.obs:
        raise KeyError(f"{species!r} is missing obs column {label_col!r}.")
    if adata.obs[label_col].isna().any():
        raise ValueError(f"{species!r} contains missing values in {label_col!r}.")
    if not adata.obs_names.is_unique:
        raise ValueError(f"{species!r} obs_names must be unique.")
    if not adata.var_names.is_unique:
        raise ValueError(f"{species!r} var_names must be unique.")


def _select_source_hvgs(
    adata: ad.AnnData,
    *,
    species: str,
    n_hvg: int,
    is_preprocessed: bool,
    cpm: float | None,
) -> ad.AnnData:
    if not is_preprocessed:
        sc.pp.filter_cells(adata, min_counts=200)
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.normalize_total(adata, target_sum=cpm)
        sc.pp.log1p(adata)

    if adata.n_vars < n_hvg:
        raise ValueError(
            f"{species!r} has only {adata.n_vars} genes after preprocessing; "
            f"cannot select {n_hvg} HVGs."
        )

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=n_hvg,
        inplace=True,
    )
    selected = adata.var["highly_variable"].to_numpy(dtype=bool)
    selected_count = int(selected.sum())
    if selected_count != n_hvg:
        raise ValueError(
            f"HVG selection for {species!r} returned {selected_count} genes; "
            f"expected exactly {n_hvg}."
        )

    result = adata[:, selected].copy()
    result.uns["pseudo_species_hvg_selection"] = {
        "method": "scanpy.pp.highly_variable_genes",
        "flavor": "seurat",
        "n_top_genes": n_hvg,
        "selected_before_cell_sampling": True,
        "source_was_preprocessed": is_preprocessed,
    }
    print(
        f"[{species}] selected {result.n_vars} HVGs from the complete "
        f"real-species dataset ({result.n_obs} cells)."
    )
    return result


def _build_nested_sampling_schedule(
    *,
    n_available: int,
    max_requested: int,
    rng: np.random.Generator,
    allow_repetition: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if n_available <= 0:
        raise ValueError("Cannot sample from an empty source dataset.")
    if max_requested > n_available and not allow_repetition:
        raise ValueError(
            f"Requested {max_requested} rows from {n_available} unique cells "
            "without --allow-cell-repetition."
        )

    index_parts: list[np.ndarray] = []
    repeat_parts: list[np.ndarray] = []
    remaining = max_requested
    repeat_index = 0
    while remaining > 0:
        permutation = rng.permutation(n_available)
        take = min(remaining, n_available)
        index_parts.append(permutation[:take])
        repeat_parts.append(np.full(take, repeat_index, dtype=np.int32))
        remaining -= take
        repeat_index += 1

    return np.concatenate(index_parts), np.concatenate(repeat_parts)


def _balanced_stratified_partition(
    labels: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n_cells = len(labels)
    if k > n_cells:
        raise ValueError(f"Cannot divide {n_cells} cells into {k} non-empty groups.")

    target_sizes = np.full(k, n_cells // k, dtype=int)
    target_sizes[: n_cells % k] += 1
    buckets: list[list[int]] = [[] for _ in range(k)]

    for label in sorted(pd.unique(labels).tolist()):
        positions = np.flatnonzero(labels == label)
        rng.shuffle(positions)
        start = int(rng.integers(0, k))
        for offset, position in enumerate(positions):
            buckets[(start + offset) % k].append(int(position))

    while True:
        donors = [i for i in range(k) if len(buckets[i]) > target_sizes[i]]
        receivers = [i for i in range(k) if len(buckets[i]) < target_sizes[i]]
        if not donors and not receivers:
            break
        if not donors or not receivers:
            raise RuntimeError("Internal error while balancing pseudo-species sizes.")
        donor = max(donors, key=lambda i: len(buckets[i]) - target_sizes[i])
        receiver = max(
            receivers, key=lambda i: target_sizes[i] - len(buckets[i])
        )
        move_index = int(rng.integers(0, len(buckets[donor])))
        buckets[receiver].append(buckets[donor].pop(move_index))

    result = []
    for bucket in buckets:
        positions = np.asarray(bucket, dtype=int)
        rng.shuffle(positions)
        result.append(positions)

    assigned = np.concatenate(result)
    if not np.array_equal(np.sort(assigned), np.arange(n_cells)):
        raise RuntimeError("Pseudo-species partition is not exhaustive and disjoint.")
    return result


def _prepare_source_homology(
    path: Path,
    fish_name: str,
    frog_name: str,
    fish_genes: list[str],
    frog_genes: list[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Homology CSV not found: {path}")
    table = pd.read_csv(path)
    fish_col = f"{fish_name}_gene"
    frog_col = f"{frog_name}_gene"
    missing = [column for column in (fish_col, frog_col) if column not in table]
    if missing:
        raise KeyError(f"Homology CSV is missing columns: {missing}")

    table = table[[fish_col, frog_col]].dropna().copy()
    table[fish_col] = table[fish_col].astype(str)
    table[frog_col] = table[frog_col].astype(str)
    table = table.loc[
        table[fish_col].isin(set(fish_genes))
        & table[frog_col].isin(set(frog_genes))
    ].drop_duplicates(ignore_index=True)
    if table.empty:
        raise ValueError(
            "No valid fish-frog homology pairs remain after matching source var_names."
        )
    return table


def _write_pair_homology(
    *,
    pseudo1: str,
    pseudo2: str,
    origin1: str,
    origin2: str,
    source_names: tuple[str, str],
    source_genes: dict[str, list[str]],
    source_homology: pd.DataFrame,
    path: Path,
) -> None:
    output_col1 = f"{pseudo1}_gene"
    output_col2 = f"{pseudo2}_gene"

    if origin1 == origin2:
        genes = source_genes[origin1]
        output = pd.DataFrame({output_col1: genes, output_col2: genes})
    else:
        fish_name, frog_name = source_names
        source_columns = {
            fish_name: f"{fish_name}_gene",
            frog_name: f"{frog_name}_gene",
        }
        output = source_homology[
            [source_columns[origin1], source_columns[origin2]]
        ].copy()
        output.columns = [output_col1, output_col2]

    output.to_csv(path, index=False)


def _scenario_config(
    base_cfg: dict[str, Any],
    *,
    species_entries: list[dict[str, Any]],
    homology_entries: list[dict[str, str]],
    result_dir: Path,
    seed: int,
    n_hvg: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.pop("osr", None)
    cfg.pop("interventions", None)
    cfg["data"]["n_HVG"] = n_hvg
    cfg["data"]["select_hvg"] = False
    cfg["data"]["species"] = species_entries
    cfg["data"]["homology"] = homology_entries
    cfg["training"]["seed"] = seed
    cfg["output"] = {"outdir": str(result_dir.resolve())}
    return cfg


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")

    using_total_cells = args.total_cells is not None
    if using_total_cells:
        total_cells = sorted(set(args.total_cells))
        if not total_cells or any(value <= 0 for value in total_cells):
            raise ValueError("--total-cells values must be positive integers.")
        odd_totals = [value for value in total_cells if value % 2 != 0]
        if odd_totals:
            raise ValueError(
                "--total-cells values must be even so fish and frog receive "
                f"equal row counts; invalid values: {odd_totals}."
            )
        cells_per_species = [value // 2 for value in total_cells]
    else:
        requested_per_species = (
            list(DEFAULT_CELLS_PER_SPECIES)
            if args.cells_per_species is None
            else args.cells_per_species
        )
        cells_per_species = sorted(set(requested_per_species))
    k_values = sorted(set(args.k_values))
    if not cells_per_species or any(value <= 0 for value in cells_per_species):
        raise ValueError("--cells-per-species values must be positive integers.")
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("--k-values values must be positive integers.")

    fish_name = str(args.fish_name)
    frog_name = str(args.frog_name)
    if fish_name == frog_name:
        raise ValueError("--fish-name and --frog-name must be different.")
    if any("." in name or not name for name in (fish_name, frog_name)):
        raise ValueError("Source species names must be non-empty and cannot contain '.'.")
    source_names = (fish_name, frog_name)

    base_config_path = args.base_config.expanduser().resolve()
    base_cfg = _read_yaml(base_config_path)
    n_hvg = args.n_hvg
    if n_hvg is None:
        n_hvg = base_cfg["data"].get("n_HVG", 2000)
    if isinstance(n_hvg, bool) or not isinstance(n_hvg, int) or n_hvg <= 0:
        raise ValueError("--n-hvg/data.n_HVG must be a positive integer.")
    cpm = base_cfg["data"].get("cpm", 1e4)
    config_dir = base_config_path.parent
    fish_item = _source_species_item(base_cfg, fish_name)
    frog_item = _source_species_item(base_cfg, frog_name)
    homology_item = _source_homology_item(base_cfg, fish_name, frog_name)

    fish_path = (
        args.fish_h5ad.expanduser().resolve()
        if args.fish_h5ad is not None
        else _resolve_config_path(fish_item["h5ad"], config_dir)
    )
    frog_path = (
        args.frog_h5ad.expanduser().resolve()
        if args.frog_h5ad is not None
        else _resolve_config_path(frog_item["h5ad"], config_dir)
    )
    homology_path = (
        args.homology_csv.expanduser().resolve()
        if args.homology_csv is not None
        else _resolve_config_path(homology_item["path"], config_dir)
    )
    fish_label_col = args.fish_label_col or str(fish_item["label_col"])
    frog_label_col = args.frog_label_col or str(frog_item["label_col"])
    label_columns = {fish_name: fish_label_col, frog_name: frog_label_col}
    source_was_preprocessed = {
        fish_name: fish_item.get("is_preprocessed", True),
        frog_name: frog_item.get("is_preprocessed", True),
    }
    for species, is_preprocessed in source_was_preprocessed.items():
        if not isinstance(is_preprocessed, bool):
            raise ValueError(
                f"data.species[{species!r}].is_preprocessed must be a boolean."
            )

    if not fish_path.is_file():
        raise FileNotFoundError(f"Fish h5ad not found: {fish_path}")
    if not frog_path.is_file():
        raise FileNotFoundError(f"Frog h5ad not found: {frog_path}")
    fish_adata = ad.read_h5ad(fish_path)
    frog_adata = ad.read_h5ad(frog_path)
    source_adatas = {fish_name: fish_adata, frog_name: frog_adata}
    source_matrix_was_sparse = {
        species: bool(sp.issparse(adata.X))
        for species, adata in source_adatas.items()
    }

    for species, adata in source_adatas.items():
        _validate_source_adata(
            adata,
            species=species,
            label_col=label_columns[species],
        )

    source_adatas = {
        species: _select_source_hvgs(
            adata,
            species=species,
            n_hvg=n_hvg,
            is_preprocessed=source_was_preprocessed[species],
            cpm=cpm,
        )
        for species, adata in source_adatas.items()
    }

    minimum_available = min(adata.n_obs for adata in source_adatas.values())
    too_large = [value for value in cells_per_species if value > minimum_available]
    if too_large and not args.allow_cell_repetition:
        raise ValueError(
            "Requested cell counts cannot exceed the smaller source dataset "
            f"({minimum_available}); invalid values: {too_large}. Pass "
            "--allow-cell-repetition only for a replicated-cell computational "
            "scaling benchmark."
        )
    if max(k_values) > min(cells_per_species):
        raise ValueError(
            "Every --k-values entry must be no larger than every requested "
            "cell count."
        )

    source_genes = {
        species: [str(gene) for gene in adata.var_names]
        for species, adata in source_adatas.items()
    }
    source_homology = _prepare_source_homology(
        homology_path,
        fish_name,
        frog_name,
        source_genes[fish_name],
        source_genes[frog_name],
    )

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    source_orders: dict[str, np.ndarray] = {}
    source_repeat_indices: dict[str, np.ndarray] = {}
    max_requested = max(cells_per_species)
    for source_index, (species, adata) in enumerate(source_adatas.items()):
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, source_index, 0])
        )
        order, repeat_indices = _build_nested_sampling_schedule(
            n_available=adata.n_obs,
            max_requested=max_requested,
            rng=rng,
            allow_repetition=args.allow_cell_repetition,
        )
        source_orders[species] = order
        source_repeat_indices[species] = repeat_indices

    inventory_rows: list[dict[str, Any]] = []
    for cell_count in cells_per_species:
        sampled_adatas: dict[str, ad.AnnData] = {}
        source_unique_cells_used: dict[str, int] = {}
        source_repeated_rows: dict[str, int] = {}
        for species, adata in source_adatas.items():
            selected_indices = source_orders[species][:cell_count]
            selected_repeat_indices = source_repeat_indices[species][:cell_count]
            sampled = adata[selected_indices, :].copy()

            source_unique_cells_used[species] = min(cell_count, int(adata.n_obs))
            source_repeated_rows[species] = (
                cell_count - source_unique_cells_used[species]
            )

            if args.allow_cell_repetition:
                original_ids = adata.obs_names.to_numpy(dtype=str)[selected_indices]
                sampled.obs["benchmark_original_cell_id"] = original_ids
                sampled.obs["benchmark_draw_index"] = np.arange(
                    cell_count, dtype=np.int64
                )
                sampled.obs["benchmark_repeat_index"] = selected_repeat_indices
                sampled.obs["benchmark_is_repeated"] = selected_repeat_indices > 0
                name_width = max(8, len(str(max_requested)))
                sampled.obs_names = pd.Index(
                    f"{species}__benchmark_draw_{index:0{name_width}d}"
                    for index in range(cell_count)
                )

            if source_matrix_was_sparse[species] and not sp.issparse(sampled.X):
                raise RuntimeError(
                    f"Sparse matrix storage was not preserved while sampling "
                    f"{species!r}."
                )
            sampled_adatas[species] = sampled

        for k in k_values:
            scale_dir_name = (
                f"total_cells_{2 * cell_count}"
                if using_total_cells
                else f"cells_per_source_{cell_count}"
            )
            scenario_dir = output_root / scale_dir_name / f"k_{k}"
            if scenario_dir.exists():
                raise FileExistsError(f"Scenario directory already exists: {scenario_dir}")
            data_dir = scenario_dir / "data"
            homology_dir = scenario_dir / "homology"
            result_dir = scenario_dir / "results"
            data_dir.mkdir(parents=True)
            homology_dir.mkdir(parents=True)

            pseudo_origins: dict[str, str] = {}
            species_entries: list[dict[str, Any]] = []
            manifest_frames: list[pd.DataFrame] = []
            split_sizes: dict[str, int] = {}

            for source_index, species in enumerate(source_names):
                sampled = sampled_adatas[species]
                labels = sampled.obs[label_columns[species]].astype(str).to_numpy()
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [args.seed, cell_count, k, source_index, 1]
                    )
                )
                partitions = _balanced_stratified_partition(labels, k, rng)
                width = max(2, len(str(k)))

                for split_index, positions in enumerate(partitions, start=1):
                    pseudo_species = f"{species}_p{split_index:0{width}d}"
                    pseudo_origins[pseudo_species] = species
                    chunk = sampled[positions, :].copy()
                    chunk.obs["source_species"] = species
                    chunk.obs["pseudo_species"] = pseudo_species
                    chunk.obs["benchmark_cells_per_source"] = cell_count
                    chunk.obs["benchmark_k"] = k
                    chunk.obs["benchmark_seed"] = args.seed
                    chunk.uns["pseudo_species_benchmark"] = {
                        "source_species": species,
                        "pseudo_species": pseudo_species,
                        "cells_per_source": cell_count,
                        "k": k,
                        "seed": args.seed,
                    }

                    h5ad_path = data_dir / f"{pseudo_species}.h5ad"
                    chunk.write_h5ad(h5ad_path, compression="gzip")
                    split_sizes[pseudo_species] = int(chunk.n_obs)
                    species_entries.append(
                        {
                            "name": pseudo_species,
                            "h5ad": str(h5ad_path.resolve()),
                            "label_col": label_columns[species],
                            "is_preprocessed": True,
                        }
                    )
                    manifest_data: dict[str, Any] = {
                        "source_species": species,
                        "pseudo_species": pseudo_species,
                        "cell_id": chunk.obs_names.astype(str),
                        "cell_type": chunk.obs[
                            label_columns[species]
                        ].astype(str).to_numpy(),
                        "cells_per_source": cell_count,
                        "total_input_cells": 2 * cell_count,
                        "k": k,
                        "split_index": split_index,
                        "split_size": int(chunk.n_obs),
                        "seed": args.seed,
                    }
                    if args.allow_cell_repetition:
                        manifest_data.update(
                            {
                                "original_cell_id": chunk.obs[
                                    "benchmark_original_cell_id"
                                ].astype(str).to_numpy(),
                                "source_repeat_index": chunk.obs[
                                    "benchmark_repeat_index"
                                ].to_numpy(),
                                "source_row_is_repeated": chunk.obs[
                                    "benchmark_is_repeated"
                                ].to_numpy(),
                            }
                        )
                    manifest_frames.append(pd.DataFrame(manifest_data))

            homology_entries: list[dict[str, str]] = []
            for pseudo1, pseudo2 in combinations(pseudo_origins, 2):
                pair_path = homology_dir / f"{pseudo1}__{pseudo2}.csv"
                _write_pair_homology(
                    pseudo1=pseudo1,
                    pseudo2=pseudo2,
                    origin1=pseudo_origins[pseudo1],
                    origin2=pseudo_origins[pseudo2],
                    source_names=source_names,
                    source_genes=source_genes,
                    source_homology=source_homology,
                    path=pair_path,
                )
                homology_entries.append(
                    {
                        "species1": pseudo1,
                        "species2": pseudo2,
                        "path": str(pair_path.resolve()),
                    }
                )

            scenario_cfg = _scenario_config(
                base_cfg,
                species_entries=species_entries,
                homology_entries=homology_entries,
                result_dir=result_dir,
                seed=args.seed,
                n_hvg=n_hvg,
            )
            config_path = scenario_dir / "config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    scenario_cfg,
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                )

            cell_manifest = pd.concat(manifest_frames, ignore_index=True)
            cell_manifest.to_csv(
                scenario_dir / "cell_manifest.tsv", sep="\t", index=False
            )

            n_pseudo_species = len(pseudo_origins)
            n_pairs = len(homology_entries)
            max_split_size = max(split_sizes.values())
            effective_cell_draws = n_pseudo_species * max_split_size
            repeated_cell_draws = effective_cell_draws - (2 * cell_count)
            metadata = {
                "seed": args.seed,
                "source_species": list(source_names),
                "source_h5ad": {
                    fish_name: str(fish_path),
                    frog_name: str(frog_path),
                },
                "source_homology_csv": str(homology_path),
                "source_available_cells": {
                    species: int(adata.n_obs)
                    for species, adata in source_adatas.items()
                },
                "cells_per_source": cell_count,
                "total_input_cells": 2 * cell_count,
                "total_unique_cells": sum(source_unique_cells_used.values()),
                "source_unique_cells_used": source_unique_cells_used,
                "source_repeated_rows": source_repeated_rows,
                "total_source_repeated_rows": sum(source_repeated_rows.values()),
                "allow_cell_repetition": args.allow_cell_repetition,
                "source_matrix_was_sparse": source_matrix_was_sparse,
                "k": k,
                "pseudo_species_count": n_pseudo_species,
                "homology_pair_count": n_pairs,
                "split_sizes": split_sizes,
                "dataloader_epoch_length": max_split_size,
                "effective_cell_draws_per_epoch": effective_cell_draws,
                "repeated_cell_draws_per_epoch": repeated_cell_draws,
                "select_hvg": False,
                "hvg_selection": {
                    "method": "scanpy.pp.highly_variable_genes",
                    "flavor": "seurat",
                    "n_top_genes": n_hvg,
                    "selected_before_cell_sampling": True,
                },
                "gene_counts": {
                    species: len(genes) for species, genes in source_genes.items()
                },
                "source_was_preprocessed": source_was_preprocessed,
                "generated_is_preprocessed": True,
            }
            with (scenario_dir / "scenario.yaml").open(
                "w", encoding="utf-8"
            ) as handle:
                yaml.safe_dump(metadata, handle, allow_unicode=True, sort_keys=False)

            inventory_rows.append(
                {
                    "cells_per_source": cell_count,
                    "total_input_cells": 2 * cell_count,
                    "total_unique_cells": sum(source_unique_cells_used.values()),
                    "total_source_repeated_rows": sum(source_repeated_rows.values()),
                    "allow_cell_repetition": args.allow_cell_repetition,
                    "k": k,
                    "pseudo_species_count": n_pseudo_species,
                    "homology_pair_count": n_pairs,
                    "min_split_size": min(split_sizes.values()),
                    "max_split_size": max_split_size,
                    "effective_cell_draws_per_epoch": effective_cell_draws,
                    "repeated_cell_draws_per_epoch": repeated_cell_draws,
                    "config_path": str(config_path.resolve()),
                    "result_dir": str(result_dir.resolve()),
                }
            )

    pd.DataFrame(inventory_rows).to_csv(
        output_root / "benchmark_inventory.tsv", sep="\t", index=False
    )
    print(f"Prepared {len(inventory_rows)} benchmark scenarios in {output_root}")


if __name__ == "__main__":
    main()
