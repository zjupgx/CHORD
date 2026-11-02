
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train CHORD and evaluate homologous gene embeddings.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Training YAML; for existing outputs prefer the saved config.yaml.")
    parser.add_argument("--skip-training", action="store_true", help="Evaluate existing gene embeddings only.")
    parser.add_argument("--gene-embeddings", type=Path, help="Override input H5AD; requires --skip-training.")
    parser.add_argument("--evaluation-outdir", type=Path, help="Default: output.outdir/homolog_evaluation.")
    parser.add_argument("--pair-selection", choices=["auto", "withheld", "all"], default="auto")
    parser.add_argument("--random-seed", type=int, default=42, help="Evaluation seed; independent of the training/withholding seed.")
    parser.add_argument("--n-randomizations", type=_positive_int, default=100, help="Valid negative-only control rounds.")
    parser.add_argument("--n-permutations", type=_positive_int, default=9999, help="Unrestricted null draws; enumerate exactly when n! fits.")
    parser.add_argument("--max-shuffle-attempts", type=_positive_int, default=10000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.random_seed < 0:
        parser.error("--random-seed must be non-negative")
    if args.gene_embeddings is not None and not args.skip_training:
        parser.error("--gene-embeddings requires --skip-training")
    return args


def build_homology_indices_from_embeddings(adata_gene, homology_items):
    import pandas as pd

    required = {"species", "gene_id"}
    if not required.issubset(adata_gene.obs.columns):
        raise ValueError("Gene embeddings require obs['species'] and obs['gene_id'].")
    species = adata_gene.obs["species"].astype(str)
    lookups = {}
    for name in species.unique():
        genes = adata_gene.obs.loc[species == name, "gene_id"].astype(str)
        if genes.duplicated().any():
            raise ValueError(f"Duplicate gene IDs in species {name!r}.")
        lookups[name] = {gene: i for i, gene in enumerate(genes)}
    indices = {}
    for item in homology_items:
        sp1, sp2 = item["species1"], item["species2"]
        if sp1 == sp2 or sp1 not in lookups or sp2 not in lookups:
            raise ValueError(f"Homology species {(sp1, sp2)} are invalid for the output embeddings.")
        if (sp1, sp2) in indices or (sp2, sp1) in indices:
            raise ValueError(f"Duplicate homology species pair: {(sp1, sp2)}")
        col1, col2 = f"{sp1}_gene", f"{sp2}_gene"
        table = pd.read_csv(item["path"], dtype=str, keep_default_na=False)
        if not {col1, col2}.issubset(table.columns):
            raise ValueError(f"Homology CSV must contain {col1!r} and {col2!r}: {item['path']}")
        pairs = {(lookups[sp1][g1], lookups[sp2][g2])
                 for g1, g2 in zip(table[col1], table[col2])
                 if g1 in lookups[sp1] and g2 in lookups[sp2]}
        pairs = sorted(pairs)
        indices[(sp1, sp2)] = ([a for a, _ in pairs], [b for _, b in pairs])
    if not indices:
        raise ValueError("data.homology must contain at least one species pair.")
    return indices


def run_benchmark(args):
    import yaml

    config_path = args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML dictionary.")
    outdir = Path(cfg["output"]["outdir"]).expanduser().resolve()
    withholding = (cfg.get("interventions") or {}).get("homology_anchor_withholding", {}) or {}
    if args.pair_selection == "withheld" and not args.skip_training and not withholding.get("enabled", False):
        raise ValueError("--pair-selection withheld requires enabled homology_anchor_withholding before training.")

    if not args.skip_training:
        from chord.run_training import run_pipeline

        run_pipeline(cfg)

    if args.gene_embeddings is not None:
        embedding_path = args.gene_embeddings.expanduser().resolve()
    else:
        embedding_path = outdir / "adata_gene_embeddings.h5ad"
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Gene embedding file not found: {embedding_path}")

    import anndata
    from chord.analysis.gene_homolog import evaluate_homolog_embeddings

    adata_gene = anndata.read_h5ad(embedding_path)
    if args.pair_selection in {"auto", "withheld"} and withholding.get("enabled", False):
        if "withheld_homology_pairs" not in adata_gene.uns:
            raise ValueError("Config enables withholding, but the gene H5AD has no saved withheld pair table. Use matching outputs or retrain.")
    indices = build_homology_indices_from_embeddings(adata_gene, cfg["data"]["homology"])
    evaluation_outdir = (args.evaluation_outdir.expanduser().resolve()
                         if args.evaluation_outdir else outdir / "homolog_evaluation")
    plot_path = evaluation_outdir / "homolog_vs_shuffled.pdf"
    summary, details = evaluate_homolog_embeddings(
        adata_gene, indices, pair_selection=args.pair_selection,
        random_seed=args.random_seed, n_randomizations=args.n_randomizations,
        n_permutations=args.n_permutations, max_shuffle_attempts=args.max_shuffle_attempts,
        make_plot=not args.no_plot, plot_path=plot_path, return_details=True,
    )
    evaluation_outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(evaluation_outdir / "summary.tsv", sep="\t", index=False)
    for name, table in details.items():
        table.to_csv(evaluation_outdir / f"{name}.tsv", sep="\t", index=False)
    settings = {
        "config_path": str(config_path), "gene_embeddings_path": str(embedding_path),
        "training_skipped": args.skip_training,
        "requested_pair_selection": args.pair_selection,
        "resolved_pair_selection": sorted(summary["pair_selection"].unique().tolist()),
        "random_seed": args.random_seed, "n_randomizations": args.n_randomizations,
        "n_permutations": args.n_permutations, "max_shuffle_attempts": args.max_shuffle_attempts,
        "homology_paths": [str(Path(item["path"]).expanduser().resolve()) for item in cfg["data"]["homology"]],
        "control": "same endpoint genes; reject every known homolog pair",
        "test": "one-sided mean-cosine unrestricted pairing permutation; BH correction across species pairs",
        "boxplot": "true pairs vs first valid shuffled round; equal group sizes",
        "scope": "anchor withholding; genes and pairs remain in the full training gene loss",
    }
    with (evaluation_outdir / "evaluation_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)
    print(summary.to_string(index=False))
    print(f"Saved homolog evaluation to: {evaluation_outdir}")
    return summary


def main():
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
