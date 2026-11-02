from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc

from .holdout import get_cell_type_holdout_config

from ..preprocessing import get_homology_indices
from ..preprocessing.homology import (
    build_homology_index_views,
    collect_homologous_tuples,
    homology_tuple_id,
)


SPLIT_COLUMN = "split"
KNOWN_STATUS_COLUMN = "known_status"
TRUE_LABEL_COLUMN = "true_label"
TRAIN_LABEL_COLUMN = "train_label"
LABEL_PERTURBED_COLUMN = "label_perturbed"

_DEFAULT_TSV_FILENAMES = {
    "label_perturbation": "label_perturbation_manifest.tsv",
    "homology_anchor_withholding": "withheld_homology_pairs.tsv",
}


def intervention_enabled(cfg: dict[str, Any], name: str) -> bool:
    interventions = cfg.get("interventions", {})
    if not isinstance(interventions, dict):
        raise ValueError("interventions must be a dictionary when provided.")
    block = interventions.get(name)
    if block is None:
        return False
    if not isinstance(block, dict):
        raise ValueError(f"interventions.{name} must be a dictionary.")
    enabled = block.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"interventions.{name}.enabled must be a boolean.")
    return enabled


def get_intervention_block(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    interventions = cfg.get("interventions", {})
    if not isinstance(interventions, dict):
        raise ValueError("interventions must be a dictionary when provided.")
    block = interventions.get(name)
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"interventions.{name} must be a dictionary.")
    return dict(block)


def _validate_tsv_options(block: dict[str, Any], name: str) -> str | None:
    save_tsv = block.get("save_tsv", False)
    if not isinstance(save_tsv, bool):
        raise ValueError(f"interventions.{name}.save_tsv must be a boolean.")

    default_filename = _DEFAULT_TSV_FILENAMES[name]
    filename = block.get("tsv_filename", default_filename)
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(
            f"interventions.{name}.tsv_filename must be a non-empty string."
        )
    if filename != filename.strip():
        raise ValueError(
            f"interventions.{name}.tsv_filename cannot have surrounding whitespace."
        )
    filename_path = Path(filename)
    if (
        "/" in filename
        or "\\" in filename
        or filename_path.is_absolute()
        or bool(filename_path.drive)
    ):
        raise ValueError(
            f"interventions.{name}.tsv_filename must be a filename only; "
            "the output directory is controlled by output.outdir."
        )
    if filename_path.suffix.lower() != ".tsv" or not filename_path.stem:
        raise ValueError(
            f"interventions.{name}.tsv_filename must end with .tsv."
        )
    if save_tsv and block.get("enabled") is not True:
        raise ValueError(
            f"interventions.{name}.save_tsv=true requires enabled=true."
        )
    return filename if save_tsv else None


def get_intervention_tsv_filename(
    cfg: dict[str, Any],
    name: str,
) -> str | None:
    if name not in _DEFAULT_TSV_FILENAMES:
        raise ValueError(f"Unsupported intervention name: {name!r}")
    return _validate_tsv_options(get_intervention_block(cfg, name), name)


def parse_fraction(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a fraction, not a boolean.")

    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            try:
                fraction = float(text[:-1].strip()) / 100.0
            except ValueError as exc:
                raise ValueError(f"Invalid percentage for {field_name}: {value!r}") from exc
        else:
            try:
                fraction = float(text)
            except ValueError as exc:
                raise ValueError(f"Invalid fraction for {field_name}: {value!r}") from exc
    elif isinstance(value, (int, float, np.integer, np.floating)):
        fraction = float(value)
        if fraction > 1.0 and fraction <= 100.0:
            fraction /= 100.0
    else:
        raise ValueError(f"{field_name} must be numeric or a percentage string.")

    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1 (or 0% and 100%).")
    return fraction


def _validate_seed(value: Any, field_name: str, default_seed: int) -> int:
    seed = default_seed if value is None else value
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return int(seed)


def _rng_for_species(seed: int, species: str) -> np.random.Generator:
    digest = hashlib.sha256(species.encode("utf-8")).digest()
    species_key = int.from_bytes(digest[:8], byteorder="little", signed=False)
    seed_sequence = np.random.SeedSequence(
        [seed, species_key & 0xFFFFFFFF, species_key >> 32]
    )
    return np.random.default_rng(seed_sequence)


def _fraction_count(n_items: int, fraction: float, *, keep_remainder: bool) -> int:
    if n_items <= 0 or fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return n_items

    count = int(np.floor(n_items * fraction))
    if count == 0:
        count = 1
    if keep_remainder and n_items > 1:
        count = min(count, n_items - 1)
    return min(count, n_items)


def validate_interventions_config(cfg: dict[str, Any]) -> None:
    interventions = cfg.get("interventions")
    if interventions is None:
        return
    if not isinstance(interventions, dict):
        raise ValueError("interventions must be a dictionary.")

    supported = {"label_perturbation", "homology_anchor_withholding", "cell_type_holdout"}
    unknown = set(interventions) - supported
    if unknown:
        raise ValueError(f"Unsupported intervention blocks: {sorted(unknown)}")

    requested_tsv_filenames: list[tuple[str, str]] = []
    for name in supported:
        block = interventions.get(name)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"interventions.{name} must be a dictionary.")
        if "enabled" in block and not isinstance(block["enabled"], bool):
            raise ValueError(f"interventions.{name}.enabled must be a boolean.")
        if "seed" in block:
            _validate_seed(block["seed"], f"interventions.{name}.seed", 0)
        tsv_filename = _validate_tsv_options(block, name) if name in _DEFAULT_TSV_FILENAMES else None
        if tsv_filename is not None:
            requested_tsv_filenames.append((name, tsv_filename))

    normalized_filenames = [
        filename.casefold() for _name, filename in requested_tsv_filenames
    ]
    if len(normalized_filenames) != len(set(normalized_filenames)):
        raise ValueError(
            "Enabled intervention TSV outputs must use distinct filenames: "
            f"{requested_tsv_filenames}"
        )

    label_cfg = interventions.get("label_perturbation") or {}
    if "fraction" in label_cfg:
        parse_fraction(
            label_cfg["fraction"],
            "interventions.label_perturbation.fraction",
        )

    homology_cfg = interventions.get("homology_anchor_withholding") or {}
    mode = homology_cfg.get("mode", "pairwise")
    if not isinstance(mode, str) or mode not in {"pairwise", "shared_tuples"}:
        raise ValueError(
            "interventions.homology_anchor_withholding.mode must be "
            "'pairwise' or 'shared_tuples'."
        )
    if "fraction" in homology_cfg:
        parse_fraction(
            homology_cfg["fraction"],
            "interventions.homology_anchor_withholding.fraction",
        )

    get_cell_type_holdout_config(cfg)


def build_label_perturbation_manifest(
    species_adatas: dict[str, tuple[Any, str]],
) -> pd.DataFrame:
    columns = [
        "species",
        "cell_id",
        TRUE_LABEL_COLUMN,
        TRAIN_LABEL_COLUMN,
        LABEL_PERTURBED_COLUMN,
        SPLIT_COLUMN,
        KNOWN_STATUS_COLUMN,
    ]
    frames: list[pd.DataFrame] = []
    for species, (adata, _label_col) in species_adatas.items():
        required = {
            TRUE_LABEL_COLUMN,
            TRAIN_LABEL_COLUMN,
            LABEL_PERTURBED_COLUMN,
        }
        missing = required - set(adata.obs.columns)
        if missing:
            raise KeyError(
                f"[{species}] cannot build label perturbation manifest; "
                f"missing obs columns: {sorted(missing)}"
            )
        changed = adata.obs[LABEL_PERTURBED_COLUMN].astype(bool).to_numpy()
        split_values = (
            adata.obs[SPLIT_COLUMN].astype(str).to_numpy()
            if SPLIT_COLUMN in adata.obs
            else np.full(adata.n_obs, "train", dtype=object)
        )
        known_values = (
            adata.obs[KNOWN_STATUS_COLUMN].astype(str).to_numpy()
            if KNOWN_STATUS_COLUMN in adata.obs
            else np.full(adata.n_obs, "known", dtype=object)
        )
        frames.append(
            pd.DataFrame(
                {
                    "species": np.full(
                        int(changed.sum()), str(species), dtype=object
                    ),
                    "cell_id": np.asarray(adata.obs_names.astype(str))[changed],
                    TRUE_LABEL_COLUMN: adata.obs[TRUE_LABEL_COLUMN]
                    .astype(str)
                    .to_numpy()[changed],
                    TRAIN_LABEL_COLUMN: adata.obs[TRAIN_LABEL_COLUMN]
                    .astype(str)
                    .to_numpy()[changed],
                    LABEL_PERTURBED_COLUMN: np.ones(
                        int(changed.sum()), dtype=bool
                    ),
                    SPLIT_COLUMN: split_values[changed],
                    KNOWN_STATUS_COLUMN: known_values[changed],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def select_hvg_from_training_cells(
    species_adatas: dict[str, tuple[Any, str]],
    n_top_genes: int,
) -> dict[str, tuple[Any, str]]:
    processed: dict[str, tuple[Any, str]] = {}
    for species, (adata, label_col) in species_adatas.items():
        if SPLIT_COLUMN not in adata.obs:
            raise KeyError(f"[{species}] missing {SPLIT_COLUMN!r} column.")
        train_mask = adata.obs[SPLIT_COLUMN].astype(str).to_numpy() == "train"
        if not np.any(train_mask):
            raise ValueError(f"[{species}] cannot select HVGs without training cells.")

        train_adata = adata[train_mask].copy()
        sc.pp.highly_variable_genes(
            train_adata,
            flavor="seurat",
            n_top_genes=n_top_genes,
        )
        hvg_names = train_adata.var_names[
            train_adata.var["highly_variable"].to_numpy()
        ].tolist()
        if not hvg_names:
            raise ValueError(f"[{species}] training-based HVG selection returned no genes.")
        processed[species] = (adata[:, hvg_names].copy(), label_col)
        print(
            f"[{species}] selected {len(hvg_names)} HVGs from training cells "
            "and applied them to all splits"
        )
    return processed


def perturb_training_labels(
    species_adatas: dict[str, tuple[Any, str]],
    config: dict[str, Any],
    *,
    default_seed: int,
) -> tuple[dict[str, tuple[Any, str]], dict[str, int]]:
    seed = _validate_seed(
        config.get("seed"),
        "interventions.label_perturbation.seed",
        default_seed,
    )
    fraction = parse_fraction(
        config.get("fraction", 0.0),
        "interventions.label_perturbation.fraction",
    )

    annotated: dict[str, tuple[Any, str]] = {}
    changed_counts: dict[str, int] = {}
    for species, (adata, label_col) in species_adatas.items():
        adata = adata.copy()
        true_labels = adata.obs[label_col].astype(str).to_numpy(copy=True)
        train_labels = true_labels.copy()

        if SPLIT_COLUMN in adata.obs:
            candidate_mask = (
                adata.obs[SPLIT_COLUMN].astype(str).to_numpy() == "train"
            )
        else:
            candidate_mask = np.ones(adata.n_obs, dtype=bool)
        candidate_positions = np.flatnonzero(candidate_mask)
        n_change = _fraction_count(
            len(candidate_positions), fraction, keep_remainder=False
        )

        changed = np.zeros(adata.n_obs, dtype=bool)
        if n_change:
            available_labels = np.unique(true_labels[candidate_positions])
            if len(available_labels) < 2:
                raise ValueError(
                    f"[{species}] label perturbation needs at least two training "
                    "cell types."
                )
            rng = _rng_for_species(seed, species)
            selected = rng.choice(candidate_positions, size=n_change, replace=False)
            for position in selected:
                alternatives = available_labels[available_labels != train_labels[position]]
                train_labels[position] = rng.choice(alternatives)
            changed[selected] = True

        adata.obs[TRUE_LABEL_COLUMN] = true_labels
        adata.obs[TRAIN_LABEL_COLUMN] = train_labels
        adata.obs[LABEL_PERTURBED_COLUMN] = changed
        annotated[species] = (adata, label_col)
        changed_counts[species] = int(changed.sum())
    return annotated, changed_counts


def build_training_species_adatas(
    species_adatas: dict[str, tuple[Any, str]],
    *,
    split_enabled: bool,
    label_perturbation_enabled: bool,
) -> dict[str, tuple[Any, str]]:
    training: dict[str, tuple[Any, str]] = {}
    for species, (adata, original_label_col) in species_adatas.items():
        if split_enabled:
            mask = adata.obs[SPLIT_COLUMN].astype(str).to_numpy() == "train"
            if KNOWN_STATUS_COLUMN in adata.obs:
                train_status = (
                    adata.obs.loc[mask, KNOWN_STATUS_COLUMN].astype(str).to_numpy()
                )
                if np.any(train_status != "known"):
                    raise ValueError(
                        f"[{species}] training split contains unknown cells."
                    )
            train_adata = adata[mask].copy()
        else:
            train_adata = adata.copy()
        if train_adata.n_obs == 0:
            raise ValueError(f"[{species}] training view contains no cells.")
        label_col = TRAIN_LABEL_COLUMN if label_perturbation_enabled else original_label_col
        training[species] = (train_adata, label_col)
    return training


def prepare_species_views(
    species_adatas: dict[str, tuple[Any, str]],
    *,
    label_perturbation_config: dict[str, Any] | None,
    default_seed: int,
) -> tuple[
    dict[str, tuple[Any, str]],
    dict[str, tuple[Any, str]],
    dict[str, Any],
]:
    label_enabled = bool(
        label_perturbation_config
        and label_perturbation_config.get("enabled") is True
    )
    full_view = species_adatas
    metadata: dict[str, Any] = {}

    if label_enabled:
        full_view, changed_counts = perturb_training_labels(
            full_view,
            label_perturbation_config or {},
            default_seed=default_seed,
        )
        metadata["label_perturbation_counts"] = changed_counts

    training_view = build_training_species_adatas(
        full_view,
        split_enabled=False,
        label_perturbation_enabled=label_enabled,
    )
    return full_view, training_view, metadata


def build_homology_anchor_views(
    species_config: dict[str, dict[str, Any]],
    homology_files: dict[tuple[str, str], str | Path],
    config: dict[str, Any],
    *,
    default_seed: int,
) -> tuple[
    dict[tuple[str, str], tuple[list[int], list[int]]],
    dict[tuple[str, str], tuple[list[int], list[int]]],
    pd.DataFrame,
]:
    mode = config.get("mode", "pairwise")
    if mode == "shared_tuples":
        return _build_shared_tuple_anchor_views(
            species_config, homology_files, config, default_seed=default_seed
        )
    if mode != "pairwise":
        raise ValueError("homology_anchor_withholding.mode must be 'pairwise' or 'shared_tuples'.")
    full_indices = get_homology_indices(species_config, homology_files)
    anchor_indices: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
    fraction = parse_fraction(
        config.get("fraction", 0.0),
        "interventions.homology_anchor_withholding.fraction",
    )
    seed = _validate_seed(
        config.get("seed"),
        "interventions.homology_anchor_withholding.seed",
        default_seed,
    )
    withheld_rows: list[dict[str, Any]] = []

    for (species1, species2), (sp1_indices, sp2_indices) in full_indices.items():
        homo_df = pd.read_csv(homology_files[(species1, species2)])
        col1 = f"{species1}_gene"
        col2 = f"{species2}_gene"
        count1 = homo_df[col1].value_counts(dropna=False)
        count2 = homo_df[col2].value_counts(dropna=False)

        sp1_genes = list(species_config[species1]["gene_names"])
        sp2_genes = list(species_config[species2]["gene_names"])
        candidate_positions = []
        for position, (index1, index2) in enumerate(zip(sp1_indices, sp2_indices)):
            gene1 = sp1_genes[index1]
            gene2 = sp2_genes[index2]
            if int(count1.get(gene1, 0)) == 1 and int(count2.get(gene2, 0)) == 1:
                candidate_positions.append(position)

        pair_name = f"{species1}\0{species2}"
        rng = _rng_for_species(seed, pair_name)
        n_hide = _fraction_count(
            len(candidate_positions), fraction, keep_remainder=False
        )
        if n_hide:
            hidden_positions = set(
                int(value)
                for value in rng.choice(
                    candidate_positions, size=n_hide, replace=False
                )
            )
        else:
            hidden_positions = set()

        keep_positions = [
            position
            for position in range(len(sp1_indices))
            if position not in hidden_positions
        ]
        anchor_indices[(species1, species2)] = (
            [sp1_indices[position] for position in keep_positions],
            [sp2_indices[position] for position in keep_positions],
        )

        for position in sorted(hidden_positions):
            index1 = sp1_indices[position]
            index2 = sp2_indices[position]
            withheld_rows.append(
                {
                    "species1": species1,
                    "gene1": str(sp1_genes[index1]),
                    "gene1_index": int(index1),
                    "species2": species2,
                    "gene2": str(sp2_genes[index2]),
                    "gene2_index": int(index2),
                }
            )
        print(
            f"{species1}, {species2}: withheld {len(hidden_positions)} of "
            f"{len(candidate_positions)} valid one-to-one homology anchors"
        )

    columns = [
        "species1",
        "gene1",
        "gene1_index",
        "species2",
        "gene2",
        "gene2_index",
    ]
    withheld = pd.DataFrame(withheld_rows, columns=columns)
    withheld.index = withheld.index.astype(str)
    return full_indices, anchor_indices, withheld


def _build_shared_tuple_anchor_views(species_config, homology_files, config, *, default_seed):
    species = tuple(sorted(species_config))
    genes = {sp: list(map(str, species_config[sp]["gene_names"])) for sp in species}
    full, strict = build_homology_index_views(genes, homology_files)
    tuples = collect_homologous_tuples(strict, species)
    tuples.sort(key=lambda row: tuple(genes[sp][i] for sp, i in zip(species, row)))
    fraction = parse_fraction(config.get("fraction", 0.0),
                              "interventions.homology_anchor_withholding.fraction")
    seed = _validate_seed(config.get("seed"),
                          "interventions.homology_anchor_withholding.seed", default_seed)
    n_hide = _fraction_count(len(tuples), fraction, keep_remainder=False)
    rng = _rng_for_species(seed, "shared_tuples\0" + "\0".join(species))
    selected = sorted(rng.choice(len(tuples), size=n_hide, replace=False).tolist())
    hidden = [tuples[i] for i in selected]
    tuple_rows = []
    for row in hidden:
        names = [genes[sp][i] for sp, i in zip(species, row)]
        tuple_rows.append({"tuple_id": homology_tuple_id(species, names),
                           **{f"{sp}_gene": gene for sp, gene in zip(species, names)}})
    pair_rows, anchors = [], {}
    positions = {sp: i for i, sp in enumerate(species)}
    for (sp1, sp2), (left, right) in full.items():
        i, j = positions[sp1], positions[sp2]
        hidden_edges = {(row[i], row[j]) for row in hidden}
        keep = [(a, b) for a, b in zip(left, right) if (a, b) not in hidden_edges]
        if n_hide and not keep:
            raise ValueError(
                f"shared_tuples would remove every anchor for {(sp1, sp2)}; "
                "reduce fraction or retain more genes/homology anchors. At least "
                "one anchor is needed to construct the complete gene-loss target."
            )
        anchors[(sp1, sp2)] = ([a for a, _ in keep], [b for _, b in keep])
        for row, record in zip(hidden, tuple_rows):
            pair_rows.append({"species1": sp1, "gene1": genes[sp1][row[i]],
                              "gene1_index": int(row[i]), "species2": sp2,
                              "gene2": genes[sp2][row[j]], "gene2_index": int(row[j]),
                              "tuple_id": record["tuple_id"]})
    columns = ["species1", "gene1", "gene1_index", "species2", "gene2", "gene2_index", "tuple_id"]
    withheld = pd.DataFrame(pair_rows, columns=columns)
    withheld.index = withheld.index.astype(str)
    withheld.attrs.update({"mode": "shared_tuples", "tuple_species": list(species),
                           "n_candidate_tuples": len(tuples), "tuple_records": tuple_rows})
    print(f"shared_tuples: withheld {n_hide} of {len(tuples)} complete {len(species)}-species "
          f"one-to-one tuples ({len(withheld)} pairwise anchors)")
    return full, anchors, withheld


def annotate_homology_withholding(
    adata_gene,
    withheld_pairs: pd.DataFrame,
    *,
    fraction: float,
) -> None:
    withheld_genes: set[tuple[str, str]] = set()
    for row in withheld_pairs.itertuples(index=False):
        withheld_genes.add((str(row.species1), str(row.gene1)))
        withheld_genes.add((str(row.species2), str(row.gene2)))

    adata_gene.obs["homology_anchor_withheld"] = [
        (str(species), str(gene_id)) in withheld_genes
        for species, gene_id in zip(
            adata_gene.obs["species"], adata_gene.obs["gene_id"]
        )
    ]
    adata_gene.uns["homology_anchor_withholding_enabled"] = True
    adata_gene.uns["homology_anchor_withholding_fraction"] = float(fraction)
    adata_gene.uns["withheld_homology_pairs"] = withheld_pairs.copy()
    mode = withheld_pairs.attrs.get("mode", "pairwise")
    adata_gene.uns["homology_anchor_withholding_mode"] = mode
    if mode == "shared_tuples":
        species = withheld_pairs.attrs["tuple_species"]
        records = withheld_pairs.attrs["tuple_records"]
        table = pd.DataFrame(records, columns=["tuple_id"] + [f"{sp}_gene" for sp in species])
        table.index = table.index.astype(str)
        adata_gene.uns["withheld_homology_tuples"] = table
        adata_gene.uns["homology_anchor_withholding_species"] = np.asarray(species, dtype=str)
        adata_gene.uns["homology_anchor_withholding_n_candidate_tuples"] = int(
            withheld_pairs.attrs["n_candidate_tuples"]
        )
        adata_gene.uns["homology_anchor_withholding_n_withheld_tuples"] = len(table)
