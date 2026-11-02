
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare species-pair similarities on withheld homologous gene tuples.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip-training", action="store_true", help="Evaluate previously exported synchronized withheld tuples.")
    parser.add_argument("--gene-embeddings", type=Path, help="Override H5AD; requires --skip-training.")
    parser.add_argument("--evaluation-outdir", type=Path, help="Default: output.outdir/homolog_tuple_evaluation.")
    parser.add_argument("--reference-pair", nargs=2, metavar=("SPECIES1", "SPECIES2"), help="Compare this species pair against each other species pair.")
    parser.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="two-sided")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=_positive_int, default=9999, help="Paired sign permutations; enumerate exactly when 2**K fits (K held-out tuples).")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.random_seed < 0:
        parser.error("--random-seed must be non-negative")
    if args.gene_embeddings is not None and not args.skip_training:
        parser.error("--gene-embeddings requires --skip-training")
    return args


def run_benchmark(args):
    import pandas as pd
    import yaml
    from chord.data.interventions import parse_fraction, validate_interventions_config

    config_path = args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML dictionary.")
    validate_interventions_config(cfg)
    outdir = Path(cfg["output"]["outdir"]).expanduser().resolve()
    species = [item["name"] for item in cfg["data"]["species"]]
    if len(species) < 2 or len(species) != len(set(species)):
        raise ValueError("data.species must contain at least two distinct species.")
    pair_names = list(combinations(species, 2))
    tests = None
    if args.reference_pair is not None:
        key = frozenset(args.reference_pair)
        reference = next((pair for pair in pair_names if frozenset(pair) == key), None)
        if reference is None or len(key) != 2:
            raise ValueError("--reference-pair must name two distinct configured species.")
        tests = [(reference, other) for other in pair_names if other != reference]
        if not tests:
            raise ValueError("--reference-pair requires at least three species to compare species pairs.")
    if not args.skip_training:
        block = (cfg.get("interventions") or {}).get("homology_anchor_withholding", {}) or {}
        if block.get("enabled") is not True or block.get("mode", "pairwise") != "shared_tuples":
            raise ValueError("Training this benchmark requires enabled: true and mode: shared_tuples in homology_anchor_withholding.")
        if parse_fraction(block.get("fraction", 0), "homology_anchor_withholding.fraction") <= 0:
            raise ValueError("This benchmark requires a positive tuple withholding fraction.")
        from chord.run_training import run_pipeline

        run_pipeline(cfg)

    embedding_path = (args.gene_embeddings.expanduser().resolve() if args.gene_embeddings
                      else outdir / "adata_gene_embeddings.h5ad")
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Gene embeddings not found: {embedding_path}")
    import anndata
    from chord.analysis.gene_homolog import (
        compute_homolog_tuple_similarity_table,
        evaluate_homolog_tuple_similarities,
        plot_homolog_tuple_similarities,
    )
    from chord.preprocessing.homology import build_homology_index_views

    adata = anndata.read_h5ad(embedding_path)
    if not {"species", "gene_id"}.issubset(adata.obs.columns):
        raise ValueError("Gene embeddings require obs['species'] and obs['gene_id'].")
    observed_species = adata.obs["species"].astype(str)
    if set(observed_species) != set(species):
        raise ValueError("Config and embedding species do not match.")
    genes = {sp: adata.obs.loc[observed_species == sp, "gene_id"].astype(str).tolist() for sp in species}
    files = {}
    for item in cfg["data"]["homology"]:
        pair = (item["species1"], item["species2"])
        if pair in files or pair[::-1] in files:
            raise ValueError(f"Duplicate homology species pair: {pair}")
        files[pair] = item["path"]
    _, strict = build_homology_index_views(genes, files)
    scores = compute_homolog_tuple_similarity_table(
        adata, strict, species_order=species, tuple_selection="withheld",
    )
    summary, details = evaluate_homolog_tuple_similarities(
        scores, species_order=species, comparisons=tests, alternative=args.alternative,
        random_seed=args.random_seed, n_permutations=args.n_permutations, return_details=True,
    )
    output = args.evaluation_outdir.expanduser().resolve() if args.evaluation_outdir else outdir / "homolog_tuple_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "tuple_similarities.tsv", sep="\t", index=False)
    scores[["tuple_id"] + [f"{sp}_gene" for sp in species]].to_csv(output / "withheld_tuples.tsv", sep="\t", index=False)
    adata.uns["withheld_homology_pairs"].to_csv(output / "withheld_pairs.tsv", sep="\t", index=False)
    pair_frames, pair_summary = [], []
    for sp1, sp2 in pair_names:
        values = scores[f"{sp1}_{sp2}_sim"]
        pair_frames.append(pd.DataFrame({"tuple_id": scores["tuple_id"], "species1": sp1,
                                         "species2": sp2, "gene1": scores[f"{sp1}_gene"],
                                         "gene2": scores[f"{sp2}_gene"], "similarity": values}))
        pair_summary.append({"species1": sp1, "species2": sp2, "n_tuples": len(scores),
                             "mean_similarity": values.mean(), "median_similarity": values.median(),
                             "q25": values.quantile(.25), "q75": values.quantile(.75)})
    pd.concat(pair_frames, ignore_index=True).to_csv(output / "pair_scores.tsv", sep="\t", index=False)
    pair_summary = pd.DataFrame(pair_summary)
    pair_summary.to_csv(output / "pair_summary.tsv", sep="\t", index=False)
    summary.to_csv(output / "comparisons.tsv", sep="\t", index=False)
    for name, table in details.items():
        table.to_csv(output / f"{name}.tsv", sep="\t", index=False)
    if not args.no_plot:
        plot_homolog_tuple_similarities(
            scores, species_order=species, plot_path=output / "homolog_tuple_boxplot.pdf",
            png_path=output / "homolog_tuple_boxplot.png",
        )
    settings = {
        "config_path": str(config_path), "gene_embeddings_path": str(embedding_path),
        "training_skipped": args.skip_training, "withholding_mode": "shared_tuples",
        "species_order": species, "n_species": len(species), "n_tuples": len(scores),
        "n_species_pairs": len(pair_names), "reference_pair": args.reference_pair,
        "alternative": args.alternative, "random_seed": args.random_seed,
        "n_permutations_budget": args.n_permutations,
        "n_candidate_tuples": int(adata.uns.get("homology_anchor_withholding_n_candidate_tuples", -1)),
        "withholding_seed": int(adata.uns.get("homology_anchor_withholding_seed", -1)),
        "withholding_fraction": float(adata.uns.get("homology_anchor_withholding_fraction", 0)),
        "homology_paths": [str(Path(path).expanduser().resolve()) for path in files.values()],
        "test": "paired within-tuple score swaps (sign flips); statistic=mean(score1-score2); BH across requested contrasts",
        "scope": "all true pairwise anchors of selected tuples withheld; complete gene loss retained",
        "boxplot": "all species-pair true ortholog similarities on the same withheld tuples",
    }
    with (output / "evaluation_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)
    print(pair_summary.to_string(index=False))
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"Saved withheld tuple evaluation to: {output}")
    return scores, summary


def main():
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
