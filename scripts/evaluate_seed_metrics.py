
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _thread_env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_name, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import scanpy as sc

from chord.analysis import (
    build_trajectory_reference_metadata,
    default_trajectory_reference_cache_path,
    evaluate_batch,
    evaluate_celltype,
    evaluate_multispecies_trajectory_conservation,
    load_multispecies_trajectory_reference,
    prepare_multispecies_trajectory_reference,
    save_multispecies_trajectory_reference,
)
from chord.preprocessing import data_preprocess
from chord.utils.model_loading import load_config


CORE_RESULT_COLUMNS = [
    "seed",
    "status",
    "n_cells",
    "nmi",
    "celltype_asw",
    "ari",
]
BATCH_RESULT_COLUMNS = [
    "batch_asw",
    "graph_connectivity",
    "h5ad",
    "error",
]

_EVAL_BATCH_KEY = "_chord_eval_batch"
_EVAL_SPECIES_KEY = "_chord_eval_species"
_EVAL_CELL_ID_KEY = "_chord_eval_cell_id"

_SETTINGS: dict[str, Any] = {}
_TRAJECTORY_REFERENCE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate NMI, cell-type ASW, ARI, batch ASW, graph connectivity, "
            "and optionally multispecies trajectory conservation."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="CHORD repository root (default: parent of the scripts directory).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help=(
            "Root containing evaluation inputs; relative paths are resolved "
            "against project-root."
        ),
    )
    parser.add_argument(
        "--input-pattern",
        default=None,
        help=(
            "Optional input path pattern relative to results-root. Use {seed} "
            "for the seed number, for example 'seed_{seed}/output.h5ad'. If "
            "omitted, seed_{seed}/<h5ad-filename> is used."
        ),
    )
    parser.add_argument(
        "--h5ad-filename",
        default="adata_cell_embeddings.h5ad",
        help="Filename inside each seed directory when input-pattern is omitted.",
    )
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--end-seed", type=int, default=30)
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Number of seed-level worker processes (default: 15).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_seed_1_30.csv"),
        help="Output CSV; relative paths are resolved against results-root.",
    )

    parser.add_argument(
        "--cell-type-key",
        default="cell_type",
        help="obs column containing cell-type labels (default: cell_type).",
    )
    parser.add_argument(
        "--batch-key",
        default="species",
        help="obs column containing batch labels (default: species).",
    )
    parser.add_argument(
        "--item-type-key",
        default="item_type",
        help=(
            "obs column distinguishing cells from other rows. When this column "
            "is absent, every observation is treated as a cell."
        ),
    )
    parser.add_argument(
        "--cell-item-value",
        default="cell",
        help="Value in item-type-key identifying cells (default: cell).",
    )
    parser.add_argument(
        "--rep",
        default="X",
        help="Embedding representation: X or a key in adata.obsm (default: X).",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument(
        "--metric-seed",
        type=int,
        default=0,
        help="Fixed random seed used by every evaluation run (default: 0).",
    )

    trajectory_group = parser.add_argument_group("optional trajectory evaluation")
    trajectory_group.add_argument(
        "--calculate-trajectory",
        action="store_true",
        help=(
            "Enable multispecies trajectory conservation. When omitted, no "
            "raw data are loaded and no trajectory code is run."
        ),
    )
    trajectory_group.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fish_frog.yaml"),
        help=(
            "Dataset config supplying raw species data for trajectory scoring; "
            "used only with --calculate-trajectory."
        ),
    )
    trajectory_group.add_argument(
        "--trajectory-start-stage",
        action="append",
        default=[],
        metavar="SPECIES=LABEL",
        help=(
            "Starting-stage label for one raw species. Repeat for every species "
            "in the config. Required only with --calculate-trajectory."
        ),
    )
    trajectory_group.add_argument(
        "--trajectory-stage-key",
        action="append",
        default=[],
        metavar="SPECIES=COLUMN",
        help="Per-species raw obs stage column; unspecified species use 'stage'.",
    )
    trajectory_group.add_argument(
        "--trajectory-raw-cell-id-key",
        action="append",
        default=[],
        metavar="SPECIES=COLUMN",
        help="Optional per-species raw obs cell-ID column; default is obs_names.",
    )
    trajectory_group.add_argument(
        "--trajectory-species-key",
        default=None,
        help=(
            "Integrated obs column containing species labels. Defaults to "
            "--batch-key."
        ),
    )
    trajectory_group.add_argument(
        "--cell-id-key",
        default="cell_id",
        help=(
            "Integrated obs cell-ID column for alignment. If absent, obs_names "
            "are used automatically. Pass 'obs_names' to request that explicitly."
        ),
    )
    trajectory_group.add_argument(
        "--species-map",
        action="append",
        default=[],
        metavar="INTEGRATED=REFERENCE",
        help=(
            "Map an integrated species label to the species name in the raw "
            "config, for example zebrafish=fish. Repeat as needed."
        ),
    )
    trajectory_group.add_argument(
        "--trajectory-n-neighbors",
        type=int,
        default=30,
        help="Neighbors for raw and joint trajectory graphs (default: 30).",
    )
    trajectory_group.add_argument(
        "--trajectory-raw-n-comps",
        type=int,
        default=50,
        help="Maximum PCA components for raw trajectory data (default: 50).",
    )
    trajectory_group.add_argument(
        "--trajectory-raw-n-pcs",
        type=int,
        default=30,
        help="Maximum raw PCs used for trajectory neighbors (default: 30).",
    )
    trajectory_group.add_argument(
        "--trajectory-root-selection-dims",
        type=int,
        default=3,
        help="Diffusion-map dimensions used by scIB-style root voting (default: 3).",
    )
    trajectory_group.add_argument(
        "--trajectory-reference-cache",
        type=Path,
        default=None,
        help=(
            "Optional raw-pseudotime reference cache path. By default, a "
            "parameter-keyed cache is stored under project-root/"
            "trajectory_reference_cache and shared across methods."
        ),
    )
    trajectory_group.add_argument(
        "--rebuild-trajectory-reference",
        action="store_true",
        help="Recompute and replace the matching raw trajectory reference cache.",
    )
    trajectory_group.add_argument(
        "--joint-root-species",
        default=None,
        help="Raw-reference species name of an optional explicit joint root.",
    )
    trajectory_group.add_argument(
        "--joint-root-cell-id",
        default=None,
        help="Cell ID of an optional explicit joint root.",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def parse_assignments(values, argument_name: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"{argument_name} values must use NAME=VALUE format; "
                f"received {value!r}."
            )
        name, assigned_value = value.split("=", 1)
        name = name.strip()
        assigned_value = assigned_value.strip()
        if not name or not assigned_value:
            raise ValueError(
                f"{argument_name} contains an empty name or value: {value!r}."
            )
        if name in assignments:
            raise ValueError(
                f"{argument_name} contains {name!r} more than once."
            )
        assignments[name] = assigned_value
    return assignments


def input_h5ad_path(settings: dict[str, Any], seed: int) -> Path:
    pattern = settings["input_pattern"]
    if pattern is None:
        return (
            settings["results_root"]
            / f"seed_{seed}"
            / settings["h5ad_filename"]
        )

    try:
        formatted_pattern = pattern.format(seed=seed)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"Invalid --input-pattern {pattern!r}; only {{seed}} is supported."
        ) from exc

    path = Path(formatted_pattern)
    if path.is_absolute():
        return path
    return settings["results_root"] / path


def result_columns(
    calculate_trajectory: bool,
    trajectory_species: list[str],
) -> list[str]:
    columns = CORE_RESULT_COLUMNS.copy()
    if calculate_trajectory:
        columns.append("trajectory_conservation")
        for species in trajectory_species:
            columns.extend(
                [
                    f"trajectory_{species}",
                    f"trajectory_rho_{species}",
                    f"trajectory_n_cells_{species}",
                    f"raw_root_cell_id_{species}",
                ]
            )
        columns.extend(["joint_root_species", "joint_root_cell_id"])
    columns.extend(BATCH_RESULT_COLUMNS)
    return columns


def error_result(
    seed: int,
    h5ad_path: Path,
    exc: BaseException,
    settings: dict[str, Any] | None = None,
    trajectory_reference=None,
) -> dict[str, Any]:
    if settings is None:
        settings = _SETTINGS
    if trajectory_reference is None:
        trajectory_reference = _TRAJECTORY_REFERENCE

    result: dict[str, Any] = {
        "seed": seed,
        "status": "error",
        "n_cells": np.nan,
        "nmi": np.nan,
        "celltype_asw": np.nan,
        "ari": np.nan,
        "batch_asw": np.nan,
        "graph_connectivity": np.nan,
        "h5ad": str(h5ad_path),
        "error": f"{type(exc).__name__}: {exc}",
    }

    if settings["calculate_trajectory"]:
        result["trajectory_conservation"] = np.nan
        result["joint_root_species"] = ""
        result["joint_root_cell_id"] = ""
        for species in settings["trajectory_species"]:
            result[f"trajectory_{species}"] = np.nan
            result[f"trajectory_rho_{species}"] = np.nan
            result[f"trajectory_n_cells_{species}"] = np.nan
            result[f"raw_root_cell_id_{species}"] = (
                trajectory_reference[species]["root_cell_id"]
                if trajectory_reference is not None
                else ""
            )
    return result


def initialize_worker(settings: dict[str, Any], trajectory_reference) -> None:
    global _SETTINGS, _TRAJECTORY_REFERENCE
    _SETTINGS = settings
    _TRAJECTORY_REFERENCE = trajectory_reference


def evaluate_one_seed(seed: int) -> dict[str, Any]:
    settings = _SETTINGS
    h5ad_path = input_h5ad_path(settings, seed)
    partial_result: dict[str, Any] = {}

    try:
        print(f"[seed_{seed}] reading {h5ad_path}", flush=True)
        if not h5ad_path.is_file():
            raise FileNotFoundError(f"Input H5AD not found: {h5ad_path}")

        adata = sc.read_h5ad(h5ad_path)
        cell_type_key = settings["cell_type_key"]
        batch_key = settings["batch_key"]
        item_type_key = settings["item_type_key"]

        required_source_columns = {cell_type_key, batch_key}
        if settings["calculate_trajectory"]:
            required_source_columns.add(settings["trajectory_species_key"])
        missing_columns = required_source_columns - set(adata.obs.columns)
        if missing_columns:
            raise KeyError(
                "H5AD is missing required obs column(s): "
                + ", ".join(sorted(missing_columns))
            )

        if item_type_key in adata.obs.columns:
            cell_mask = (
                adata.obs[item_type_key]
                .astype(str)
                .eq(settings["cell_item_value"])
                .to_numpy()
            )
            adata_cells = adata[cell_mask].copy()
            selection_message = (
                f"selected {adata_cells.n_obs}/{adata.n_obs} observations with "
                f"obs[{item_type_key!r}] == {settings['cell_item_value']!r}"
            )
        else:
            adata_cells = adata.copy()
            selection_message = (
                f"obs[{item_type_key!r}] is absent; treating all "
                f"{adata.n_obs} observations as cells"
            )

        print(f"[seed_{seed}] {selection_message}", flush=True)
        if adata_cells.n_obs == 0:
            raise ValueError("No cells remain after item-type filtering.")
        partial_result["n_cells"] = int(adata_cells.n_obs)

        for key in required_source_columns:
            if adata_cells.obs[key].isna().any():
                raise ValueError(f"obs[{key!r}] contains missing labels.")

        rep = settings["rep"]
        if rep != "X" and rep not in adata_cells.obsm:
            raise KeyError(f"Representation {rep!r} is absent from adata.obsm.")

        cell_type_labels = adata_cells.obs[cell_type_key].astype(str).copy()
        batch_labels = adata_cells.obs[batch_key].astype(str).copy()

        if settings["calculate_trajectory"]:
            source_species = adata_cells.obs[
                settings["trajectory_species_key"]
            ].astype(str).copy()
            requested_cell_id_key = settings["cell_id_key"]
            if requested_cell_id_key == "obs_names":
                integrated_cell_ids = pd.Index(adata_cells.obs_names).astype(str)
                cell_id_source = "obs_names (explicit)"
            elif requested_cell_id_key in adata_cells.obs.columns:
                integrated_cell_ids = adata_cells.obs[
                    requested_cell_id_key
                ].astype(str).to_numpy(copy=True)
                cell_id_source = f"obs[{requested_cell_id_key!r}]"
            else:
                integrated_cell_ids = pd.Index(adata_cells.obs_names).astype(str)
                cell_id_source = (
                    f"obs[{requested_cell_id_key!r}] absent; using obs_names"
                )

        adata_cells.obs["cell_type"] = pd.Categorical(cell_type_labels)
        adata_cells.obs[_EVAL_BATCH_KEY] = pd.Categorical(batch_labels)

        if settings["calculate_trajectory"]:
            species_map = settings["species_map"]
            mapped_species = source_species.map(
                lambda value: species_map.get(value, value)
            )
            adata_cells.obs[_EVAL_SPECIES_KEY] = pd.Categorical(mapped_species)
            adata_cells.obs[_EVAL_CELL_ID_KEY] = integrated_cell_ids
            print(f"[seed_{seed}] trajectory cell IDs: {cell_id_source}", flush=True)

        if "item_type" in adata_cells.obs.columns:
            del adata_cells.obs["item_type"]

        nmi, celltype_asw, ari = evaluate_celltype(
            adata_cells,
            rep=rep,
            n_neighbors=settings["n_neighbors"],
            seed=settings["metric_seed"],
            calculate_trajectory=False,
        )
        partial_result.update(
            nmi=float(nmi),
            celltype_asw=float(celltype_asw),
            ari=float(ari),
        )

        trajectory_result = None
        if settings["calculate_trajectory"]:
            trajectory_result = evaluate_multispecies_trajectory_conservation(
                adata_cells,
                _TRAJECTORY_REFERENCE,
                rep=rep,
                species_key=_EVAL_SPECIES_KEY,
                cell_id_key=_EVAL_CELL_ID_KEY,
                n_neighbors=settings["trajectory_n_neighbors"],
                joint_root_species=settings["joint_root_species"],
                joint_root_cell_id=settings["joint_root_cell_id"],
                root_selection_dpt_dims=settings[
                    "trajectory_root_selection_dims"
                ],
                seed=settings["metric_seed"],
                zero_failed_species=True,
            )

        trajectory_log = ""
        if trajectory_result is not None:
            partial_result["trajectory_conservation"] = float(
                trajectory_result["trajectory_conservation"]
            )
            partial_result["joint_root_species"] = trajectory_result[
                "joint_root_species"
            ]
            partial_result["joint_root_cell_id"] = trajectory_result[
                "joint_root_cell_id"
            ]
            for species in settings["trajectory_species"]:
                partial_result[f"trajectory_{species}"] = float(
                    trajectory_result["species_scores"][species]
                )
                partial_result[f"trajectory_rho_{species}"] = float(
                    trajectory_result["species_correlations"][species]
                )
                partial_result[f"trajectory_n_cells_{species}"] = int(
                    trajectory_result["aligned_cells"][species]
                )
                partial_result[f"raw_root_cell_id_{species}"] = trajectory_result[
                    "raw_root_cell_ids"
                ][species]
            trajectory_log = (
                f" | Trajectory="
                f"{trajectory_result['trajectory_conservation']:.4f}"
            )

        batch_asw, graph_connectivity = evaluate_batch(
            adata_cells,
            batch_key=_EVAL_BATCH_KEY,
            rep=rep,
            n_neighbors=settings["n_neighbors"],
            seed=settings["metric_seed"],
        )
        partial_result.update(
            batch_asw=float(batch_asw),
            graph_connectivity=float(graph_connectivity),
        )

        result: dict[str, Any] = {
            "seed": seed,
            "status": "ok",
            "n_cells": int(adata_cells.n_obs),
            "nmi": float(nmi),
            "celltype_asw": float(celltype_asw),
            "ari": float(ari),
            "batch_asw": float(batch_asw),
            "graph_connectivity": float(graph_connectivity),
            "h5ad": str(h5ad_path),
            "error": "",
        }
        result.update(partial_result)

        print(
            f"[seed_{seed}] finished | NMI={nmi:.4f} | "
            f"cASW={celltype_asw:.4f} | ARI={ari:.4f}"
            f"{trajectory_log} | bASW={batch_asw:.4f} | "
            f"GC={graph_connectivity:.4f}",
            flush=True,
        )
        return result

    except Exception as exc:
        print(f"[seed_{seed}] FAILED", flush=True)
        traceback.print_exc()
        result = error_result(seed, h5ad_path, exc)
        result.update(partial_result)
        return result

    finally:
        gc.collect()


def save_results(
    rows: list[dict[str, Any]],
    output_csv: Path,
    columns: list[str],
) -> None:
    dataframe = (
        pd.DataFrame(rows, columns=columns)
        .sort_values("seed")
        .reset_index(drop=True)
    )
    temporary_path = Path(str(output_csv) + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_csv)


def prepare_trajectory(args: argparse.Namespace, project_root: Path):
    config_path = resolve_path(args.config, project_root)
    cfg = load_config(config_path)
    species_items = cfg["data"]["species"]
    species_names = [item["name"] for item in species_items]
    if len(species_names) < 2:
        raise ValueError(
            "Multispecies trajectory evaluation requires at least two species."
        )
    if len(species_names) != len(set(species_names)):
        raise ValueError("Trajectory config contains duplicate species names.")

    start_stages = parse_assignments(
        args.trajectory_start_stage,
        "--trajectory-start-stage",
    )
    stage_key_overrides = parse_assignments(
        args.trajectory_stage_key,
        "--trajectory-stage-key",
    )
    raw_cell_id_key_overrides = parse_assignments(
        args.trajectory_raw_cell_id_key,
        "--trajectory-raw-cell-id-key",
    )

    configured_species = set(species_names)
    for argument_name, assignments in (
        ("--trajectory-start-stage", start_stages),
        ("--trajectory-stage-key", stage_key_overrides),
        ("--trajectory-raw-cell-id-key", raw_cell_id_key_overrides),
    ):
        unknown_species = set(assignments) - configured_species
        if unknown_species:
            raise ValueError(
                f"{argument_name} contains species absent from the config: "
                + ", ".join(sorted(unknown_species))
            )

    missing_start_species = configured_species - set(start_stages)
    if missing_start_species:
        raise ValueError(
            "--calculate-trajectory requires --trajectory-start-stage for "
            "every configured species; missing: "
            + ", ".join(sorted(missing_start_species))
        )

    if (args.joint_root_species is None) != (args.joint_root_cell_id is None):
        raise ValueError(
            "--joint-root-species and --joint-root-cell-id must be supplied together."
        )
    if (
        args.joint_root_species is not None
        and args.joint_root_species not in configured_species
    ):
        raise ValueError(
            f"Joint-root species {args.joint_root_species!r} is absent from "
            "the trajectory config."
        )

    stage_keys = {
        species: stage_key_overrides.get(species, "stage")
        for species in species_names
    }

    raw_cell_id_keys = (
        {
            species: raw_cell_id_key_overrides.get(species)
            for species in species_names
        }
        if raw_cell_id_key_overrides
        else None
    )
    n_top_genes = cfg["data"].get("n_HVG", 2000)
    cpm = cfg["data"].get("cpm", 1e4)
    reference_metadata = build_trajectory_reference_metadata(
        config_path=config_path,
        raw_data_base=project_root,
        species_items=species_items,
        start_stage_by_species=start_stages,
        stage_key_by_species=stage_keys,
        raw_cell_id_key_by_species=raw_cell_id_keys,
        n_top_genes=n_top_genes,
        cpm=cpm,
        n_neighbors=args.trajectory_n_neighbors,
        raw_n_comps=args.trajectory_raw_n_comps,
        raw_n_pcs=args.trajectory_raw_n_pcs,
        root_selection_dpt_dims=args.trajectory_root_selection_dims,
        seed=args.metric_seed,
    )
    if args.trajectory_reference_cache is None:
        reference_cache = default_trajectory_reference_cache_path(
            project_root / "trajectory_reference_cache",
            config_path.stem,
            reference_metadata,
        )
    else:
        reference_cache = resolve_path(
            args.trajectory_reference_cache,
            project_root,
        )

    if reference_cache.is_file() and not args.rebuild_trajectory_reference:
        trajectory_reference = load_multispecies_trajectory_reference(
            reference_cache,
            expected_metadata=reference_metadata,
        )
        print(
            f"Loaded fixed raw trajectory reference: {reference_cache}",
            flush=True,
        )
        return config_path, species_names, trajectory_reference, reference_cache

    raw_adatas_by_species = {}
    for species_item in species_items:
        species = species_item["name"]
        print(f"Preparing raw trajectory data for {species}", flush=True)
        raw_adatas_by_species[species] = data_preprocess(
            species_item["h5ad"],
            n_top_genes=n_top_genes,
            cpm=cpm,
            is_preprocessed=species_item.get("is_preprocessed", True),
        )

    trajectory_reference = prepare_multispecies_trajectory_reference(
        raw_adatas_by_species,
        start_stages,
        stage_key_by_species=stage_keys,
        raw_cell_id_key_by_species=raw_cell_id_keys,
        n_neighbors=args.trajectory_n_neighbors,
        raw_n_comps=args.trajectory_raw_n_comps,
        raw_n_pcs=args.trajectory_raw_n_pcs,
        root_selection_dpt_dims=args.trajectory_root_selection_dims,
        seed=args.metric_seed,
    )
    save_multispecies_trajectory_reference(
        trajectory_reference,
        reference_cache,
        reference_metadata,
    )
    print(f"Saved fixed raw trajectory reference: {reference_cache}", flush=True)
    del raw_adatas_by_species
    gc.collect()
    return config_path, species_names, trajectory_reference, reference_cache


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.start_seed > args.end_seed:
        raise ValueError("--start-seed cannot be greater than --end-seed.")
    if args.n_neighbors < 1:
        raise ValueError("--n-neighbors must be at least 1.")
    if args.trajectory_n_neighbors < 1:
        raise ValueError("--trajectory-n-neighbors must be at least 1.")
    if args.trajectory_raw_n_comps < 2:
        raise ValueError("--trajectory-raw-n-comps must be at least 2.")
    if args.trajectory_raw_n_pcs < 1:
        raise ValueError("--trajectory-raw-n-pcs must be at least 1.")
    if args.trajectory_root_selection_dims < 1:
        raise ValueError("--trajectory-root-selection-dims must be at least 1.")

    project_root = args.project_root.resolve()
    results_root = resolve_path(args.results_root, project_root)
    output_csv = resolve_path(args.output, results_root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(project_root)

    trajectory_reference = None
    trajectory_species: list[str] = []
    config_path = None
    trajectory_reference_cache = None
    species_map = parse_assignments(args.species_map, "--species-map")
    if args.calculate_trajectory:
        (
            config_path,
            trajectory_species,
            trajectory_reference,
            trajectory_reference_cache,
        ) = prepare_trajectory(args, project_root)
        unknown_targets = set(species_map.values()) - set(trajectory_species)
        if unknown_targets:
            raise ValueError(
                "--species-map targets species absent from the trajectory config: "
                + ", ".join(sorted(unknown_targets))
            )

    seeds = list(range(args.start_seed, args.end_seed + 1))
    trajectory_species_key = args.trajectory_species_key or args.batch_key
    settings: dict[str, Any] = {
        "results_root": results_root,
        "input_pattern": args.input_pattern,
        "h5ad_filename": args.h5ad_filename,
        "cell_type_key": args.cell_type_key,
        "batch_key": args.batch_key,
        "item_type_key": args.item_type_key,
        "cell_item_value": args.cell_item_value,
        "rep": args.rep,
        "n_neighbors": args.n_neighbors,
        "metric_seed": args.metric_seed,
        "calculate_trajectory": args.calculate_trajectory,
        "trajectory_species": trajectory_species,
        "trajectory_species_key": trajectory_species_key,
        "cell_id_key": args.cell_id_key,
        "species_map": species_map,
        "trajectory_n_neighbors": args.trajectory_n_neighbors,
        "trajectory_root_selection_dims": args.trajectory_root_selection_dims,
        "joint_root_species": args.joint_root_species,
        "joint_root_cell_id": args.joint_root_cell_id,
    }
    columns = result_columns(args.calculate_trajectory, trajectory_species)

    print(f"Project root: {project_root}", flush=True)
    print(f"Results root: {results_root}", flush=True)
    if args.input_pattern is None:
        print(f"Input layout: seed_{{seed}}/{args.h5ad_filename}", flush=True)
    else:
        print(f"Input pattern: {args.input_pattern}", flush=True)
    print(f"Seeds: {args.start_seed}-{args.end_seed}", flush=True)
    print(f"Workers: {min(args.workers, len(seeds))}", flush=True)
    print(f"Cell-type label: obs[{args.cell_type_key!r}]", flush=True)
    print(f"Batch label: obs[{args.batch_key!r}]", flush=True)
    print(f"Representation: {args.rep}", flush=True)

    if args.calculate_trajectory:
        print("Trajectory evaluation: enabled", flush=True)
        print(f"Trajectory config: {config_path}", flush=True)
        print(
            f"Fixed raw trajectory reference: {trajectory_reference_cache}",
            flush=True,
        )
        print(
            f"Trajectory integrated species label: obs[{trajectory_species_key!r}]",
            flush=True,
        )
        if species_map:
            print(f"Trajectory species mapping: {species_map}", flush=True)
        for species in trajectory_species:
            reference = trajectory_reference[species]
            print(
                f"{species}: {reference['stage_key']}="
                f"{reference['start_stage']!r}, raw root="
                f"{reference['root_cell_id']}",
                flush=True,
            )
        if args.joint_root_cell_id is None:
            print(
                "Joint root: selected independently for each seed from all "
                "starting-stage cells using the scIB diffusion-map rule",
                flush=True,
            )
        else:
            print(
                f"Joint root: species={args.joint_root_species}, "
                f"cell_id={args.joint_root_cell_id}",
                flush=True,
            )
    else:
        print(
            "Trajectory evaluation: disabled (raw data will not be loaded and "
            "trajectory calculations will not run)",
            flush=True,
        )

    results_by_seed: dict[int, dict[str, Any]] = {}
    worker_count = min(args.workers, len(seeds))
    executor_kwargs: dict[str, Any] = {
        "max_workers": worker_count,
        "initializer": initialize_worker,
        "initargs": (settings, trajectory_reference),
    }
    if "fork" in mp.get_all_start_methods():
        executor_kwargs["mp_context"] = mp.get_context("fork")

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        future_to_seed = {
            executor.submit(evaluate_one_seed, seed): seed for seed in seeds
        }

        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[seed_{seed}] worker crashed: {exc}", flush=True)
                h5ad_path = input_h5ad_path(settings, seed)
                result = error_result(
                    seed,
                    h5ad_path,
                    exc,
                    settings=settings,
                    trajectory_reference=trajectory_reference,
                )

            results_by_seed[seed] = result
            save_results(list(results_by_seed.values()), output_csv, columns)
            print(
                f"Progress: {len(results_by_seed)}/{len(seeds)} | "
                f"CSV: {output_csv}",
                flush=True,
            )

    final_results = pd.DataFrame(
        [results_by_seed[seed] for seed in sorted(results_by_seed)],
        columns=columns,
    )
    successful = int(final_results["status"].eq("ok").sum())
    failed = len(final_results) - successful

    print("=" * 80, flush=True)
    print(f"Evaluation completed: ok={successful}, failed={failed}", flush=True)
    print(f"Results: {output_csv}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
