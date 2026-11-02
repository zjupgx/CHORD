
from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import anndata
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from chord.analysis.tree_eval import consensus_linkage_wo_na, evaluate_tree_clustering


HIGH_TO_LOW = {
    "Astro 1": "Astro",
    "Astro 2": "Astro",
    "Chandelier": "Chandelier",
    "Endo": "Endo",
    "L2/3 IT": "L2/3 IT",
    "L5 ET 1": "L5 ET",
    "L5 ET 2": "L5 ET",
    "L5 IT 1": "L5 IT",
    "L5 IT 2": "L5 IT",
    "L5 IT 3": "L5 IT",
    "L5/6 NP": "L5/6 NP",
    "L6 CT 1": "L6 CT",
    "L6 CT 2": "L6 CT",
    "L6 IT 1": "L6 IT",
    "L6 IT 2": "L6 IT",
    "L6 IT 3": "L6 IT",
    "L6b": "L6b",
    "Lamp5 1": "Lamp5",
    "Lamp5 2": "Lamp5",
    "Lamp5 3": "Lamp5",
    "Lamp5 4": "Lamp5",
    "Lamp5 5": "Lamp5",
    "Microglia/PVM": "Microglia/PVM",
    "OPC": "OPC",
    "Oligo 1": "Oligo",
    "Oligo 2": "Oligo",
    "Pvalb 1": "Pvalb",
    "Pvalb 2": "Pvalb",
    "Sncg 1": "Sncg",
    "Sncg 2": "Sncg",
    "Sncg 3": "Sncg",
    "Sncg 4": "Sncg",
    "Sst Chodl": "Sst Chodl",
    "Sst 1": "Sst",
    "Sst 2": "Sst",
    "Sst 3": "Sst",
    "Sst 4": "Sst",
    "Sst 5": "Sst",
    "Sst 6": "Sst",
    "Sst 7": "Sst",
    "VLMC": "VLMC",
    "Vip 1": "Vip",
    "Vip 2": "Vip",
    "Vip 3": "Vip",
    "Vip 4": "Vip",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare low-resolution, hierarchy-finetuned, and high-resolution CHORD outputs.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--low-template", required=True, help="Adapted low-model cell H5AD; may contain {seed}.")
    parser.add_argument("--low2high-template", required=True, help="Hierarchy-finetuned cell H5AD; may contain {seed}.")
    parser.add_argument("--high-template", required=True, help="High-resolution-only cell H5AD; may contain {seed}.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--output-dir", type=Path, default=Path("hierarchy_model_evaluation"))
    parser.add_argument("--transfer-scope", choices=("intersection", "all"), default="intersection",
                        help="Default: score query cells whose high labels also occur in reference prototypes. 'all' scores every query cell.")
    return parser.parse_args()


def normalized_label(value):
    if pd.isna(value):
        return "NA"
    label = " ".join(str(value).split())
    if label in HIGH_TO_LOW:
        return label
    space_alias = " ".join(label.replace("_", " ").split())
    return space_alias if space_alias in HIGH_TO_LOW else label


def read_model(path):
    adata = anndata.read_h5ad(path)
    required = {"species", "item_type", "cell_type", "cell_id"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if adata.obs["species"].isna().any() or adata.obs["item_type"].isna().any():
        raise ValueError(f"{path}: missing species/item_type annotations.")
    adata.obs["species"] = adata.obs["species"].astype(str)
    adata.obs["item_type"] = adata.obs["item_type"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].map(normalized_label).astype(str)
    eligible = adata.obs["item_type"].isin(["cell", "celltype"])
    eligible &= ~adata.obs["cell_type"].isin(["NA", ""])
    adata = adata[eligible.to_numpy()].copy()
    labels = set(adata.obs["cell_type"])
    unmapped = labels - set(HIGH_TO_LOW)
    if unmapped:
        raise ValueError(
            f"{path}: unmapped high-resolution labels {sorted(unmapped)}. "
            "Check these names against HIGH_TO_LOW; genuinely new labels need "
            "an explicit low-resolution group in that mapping."
        )
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    x = np.asarray(adata.X)
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: non-finite embeddings.")
    if np.any(np.linalg.norm(x, axis=1) <= 0):
        raise ValueError(f"{path}: zero-norm embeddings.")
    cells = adata.obs["item_type"].eq("cell")
    prototypes = adata.obs["item_type"].eq("celltype")
    if not cells.any() or not prototypes.any():
        raise ValueError(f"{path}: requires both cell and celltype rows.")
    ids = adata.obs.loc[cells, "cell_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{path}: cells must have original cell_id values.")
    adata.obs["cell_id"] = adata.obs["cell_id"].astype(str)
    if adata.obs.loc[cells].duplicated(["species", "cell_id"]).any():
        raise ValueError(f"{path}: duplicate (species, cell_id).")
    if adata.obs.loc[prototypes].duplicated(["species", "cell_type"]).any():
        raise ValueError(f"{path}: duplicate (species, high-resolution prototype).")
    if set(adata.obs.loc[cells, "species"]) != set(adata.obs.loc[prototypes, "species"]):
        raise ValueError(f"{path}: cells and prototypes must cover the same species.")
    adata.obs["cell_type_high"] = adata.obs["cell_type"]
    adata.obs["cell_type_low"] = adata.obs["cell_type"].map(HIGH_TO_LOW)
    return adata


def comparison_signature(adata):
    cells = adata.obs.loc[adata.obs["item_type"].eq("cell")]
    prototypes = adata.obs.loc[adata.obs["item_type"].eq("celltype")]
    cell_labels = {
        (str(row.species), str(row.cell_id)): str(row.cell_type)
        for row in cells.itertuples(index=False)
    }
    prototype_keys = set(zip(prototypes["species"], prototypes["cell_type"]))
    return cell_labels, prototype_keys


def ensure_comparable(models, seed):
    reference_cells, reference_prototypes = comparison_signature(models["low"])
    for name, adata in models.items():
        cells, prototypes = comparison_signature(adata)
        if cells != reference_cells:
            missing = len(set(reference_cells) - set(cells))
            extra = len(set(cells) - set(reference_cells))
            raise ValueError(f"seed {seed}, {name}: cell IDs/labels differ from low (missing={missing}, extra={extra}).")
        if prototypes != reference_prototypes:
            raise ValueError(f"seed {seed}, {name}: prototype set differs from low.")
    species = sorted({sp for sp, _ in reference_cells})
    if len(species) < 2:
        raise ValueError("At least two species are required for label transfer.")
    for sp in species:
        cell_labels = {label for (cell_sp, _), label in reference_cells.items() if cell_sp == sp}
        prototype_labels = {label for prototype_sp, label in reference_prototypes if prototype_sp == sp}
        if cell_labels != prototype_labels:
            raise ValueError(f"[{sp}] observed high labels and available prototypes differ.")
    return species


def evaluate_hierarchy(adata, model_name, seed):
    positions = np.flatnonzero(adata.obs["item_type"].eq("celltype").to_numpy())
    obs = adata.obs.iloc[positions].copy()
    order = sorted(range(len(obs)), key=lambda i: (obs.iloc[i]["species"], obs.iloc[i]["cell_type"]))
    positions = positions[order]
    obs = adata.obs.iloc[positions].copy()
    labels = [json.dumps([sp, label], ensure_ascii=False)
              for sp, label in zip(obs["species"], obs["cell_type"])]
    low_groups = sorted(set(obs["cell_type_low"]))
    if len(low_groups) < 2:
        raise ValueError("Hierarchy evaluation requires at least two represented low-resolution groups.")
    group_ids = {group: i for i, group in enumerate(low_groups)}
    truth = [group_ids[group] for group in obs["cell_type_low"]]
    linkage = consensus_linkage_wo_na([np.asarray(adata.X[positions])], labels, method="ward")
    result = evaluate_tree_clustering(linkage, truth, n_clusters=len(low_groups))
    return {
        "model": model_name, "seed": seed, "ARI": result["ari"], "NMI": result["nmi"],
        "n_prototypes": len(positions), "n_low_groups": len(low_groups),
        "n_predicted_clusters": len(set(result["pred_labels"])),
    }


def evaluate_transfer(adata, model_name, seed, species, scope):
    caches = {}
    for name in species:
        cell_mask = (adata.obs["item_type"].eq("cell") & adata.obs["species"].eq(name)).to_numpy()
        prototype_mask = (adata.obs["item_type"].eq("celltype") & adata.obs["species"].eq(name)).to_numpy()
        prototypes = np.asarray(adata.X[prototype_mask])
        caches[name] = {
            "cell_labels": adata.obs.loc[cell_mask, "cell_type"].astype(str).to_numpy(),
            "prototype_labels": adata.obs.loc[prototype_mask, "cell_type"].astype(str).to_numpy(),
            "prototype_embeddings": prototypes,
            "prototype_probability": softmax(
                cosine_similarity(np.asarray(adata.X[cell_mask]), prototypes) / 0.1,
                axis=1,
            ),
        }
    rows = []
    for reference, query in permutations(species, 2):
        query_cache, reference_cache = caches[query], caches[reference]
        query_probability = query_cache["prototype_probability"]
        query_prototypes = query_cache["prototype_embeddings"]
        reference_prototypes = reference_cache["prototype_embeddings"]
        reference_labels = reference_cache["prototype_labels"]
        nearest_reference = cosine_similarity(query_prototypes, reference_prototypes).argmax(axis=1)
        hard_mapping = np.zeros(
            (query_prototypes.shape[0], reference_prototypes.shape[0]),
            dtype=query_probability.dtype,
        )
        hard_mapping[np.arange(query_prototypes.shape[0]), nearest_reference] = 1
        reference_probability = query_probability @ hard_mapping
        predictions = reference_labels[reference_probability.argmax(axis=1)]

        query_labels = query_cache["cell_labels"]
        shared_labels = sorted(set(query_labels) & set(reference_labels))
        shared_mask = np.isin(query_labels, shared_labels)
        shared_cells = int(shared_mask.sum())
        n_query = len(query_labels)
        evaluation_mask = shared_mask if scope == "intersection" else np.ones(n_query, dtype=bool)
        y_true, y_pred = query_labels[evaluation_mask], predictions[evaluation_mask]
        n_evaluated = len(y_true)
        labels_used = shared_labels if scope == "intersection" else sorted(set(y_true) | set(y_pred))
        accuracy = float(np.mean(y_true == y_pred)) if n_evaluated else np.nan
        macro_f1 = float(f1_score(
            y_true, y_pred, labels=labels_used, average="macro", zero_division=0,
        )) if n_evaluated else np.nan
        rows.append({
            "model": model_name, "seed": seed, "reference_species": reference,
            "query_species": query, "direction": f"{reference} -> {query}",
            "label_scope": scope, "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "n_query_cells": n_query, "n_evaluated_cells": n_evaluated,
            "shared_label_cell_coverage": shared_cells / n_query,
            "status": "ok" if n_evaluated else "no_shared_labels",
        })
    return rows


def summarize(frame, groups, metrics):
    rows = []
    for key, block in frame.groupby(groups, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        for metric in metrics:
            values = pd.to_numeric(block[metric], errors="raise").dropna()
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be unique non-negative integers.")
    templates = {"low": args.low_template, "low2high": args.low2high_template, "high": args.high_template}
    paths = {seed: {name: Path(template.format(seed=seed)).resolve()
                    for name, template in templates.items()} for seed in args.seeds}
    all_paths = [path for model_paths in paths.values() for path in model_paths.values()]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("Each model/seed must point to a distinct input H5AD; use {seed} for multiple seeds.")
    outdir = args.output_dir.resolve()
    tree_rows, transfer_rows = [], []
    expected_species = None
    expected_signature = None
    for seed in sorted(args.seeds):
        models = {name: read_model(path) for name, path in paths[seed].items()}
        species = ensure_comparable(models, seed)
        if expected_species is not None and species != expected_species:
            raise ValueError(f"seed {seed}: species differ from other seeds.")
        signature = comparison_signature(models["low"])
        if expected_signature is not None and signature != expected_signature:
            raise ValueError(f"seed {seed}: cell annotations or prototype sets differ from other seeds.")
        expected_species = species
        expected_signature = signature
        for name, adata in models.items():
            row = evaluate_hierarchy(adata, name, seed)
            row["h5ad"] = str(paths[seed][name])
            tree_rows.append(row)
            transfer_rows.extend(evaluate_transfer(adata, name, seed, species, args.transfer_scope))
            print(f"{name}, seed {seed}: tree ARI={row['ARI']:.4f}, NMI={row['NMI']:.4f}; transfer finished.", flush=True)
        del models

    outdir.mkdir(parents=True, exist_ok=True)
    tree = pd.DataFrame(tree_rows).sort_values(["model", "seed"])
    transfer = pd.DataFrame(transfer_rows).sort_values(["model", "seed", "reference_species", "query_species"])
    tables = {
        "hierarchy_metrics.tsv": tree,
        "transfer_metrics.tsv": transfer,
        "hierarchy_summary.tsv": summarize(tree, ["model"], ["ARI", "NMI"]),
        "transfer_summary.tsv": summarize(transfer, ["model", "reference_species", "query_species", "label_scope"], ["Accuracy", "Macro_F1"]),
    }
    direction_means = transfer.groupby(["model", "seed", "label_scope"], as_index=False).agg(
        Accuracy=("Accuracy", "mean"), Macro_F1=("Macro_F1", "mean"),
        n_valid_directions=("Accuracy", "count"),
    )
    tables["transfer_direction_mean_by_seed.tsv"] = direction_means
    tables["transfer_overall_summary.tsv"] = summarize(direction_means, ["model", "label_scope"], ["Accuracy", "Macro_F1"])
    for filename, frame in tables.items():
        frame.to_csv(outdir / filename, sep="\t", index=False)
    print(f"Evaluation saved to: {outdir}")


if __name__ == "__main__":
    main()
