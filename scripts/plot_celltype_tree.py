
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from string import Formatter

import matplotlib

matplotlib.use("Agg")

FIGURE_WIDTH_CM = 12.7
FIGURE_HEIGHT_CM = 22.86
LEAF_LABEL_SIZE_PT = 8.0
WARD_DISTANCE_LABEL_SIZE_PT = 5.0
OUTPUT_DPI = 300
CM_PER_INCH = 2.54
AXIS_MARGIN_FRACTION = 0.005
LAYOUT_PADDING = 0.1
TREE_FLIP_LR = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chord.analysis import (
    build_label2expr_OG_adata,
    consensus_support_fuzzy_threshold,
    load_label_embeddings_from_h5ads,
    plot_dendrogram_support_colors,
    plot_tree_drift_fullcolor,
    resolve_species_gene,
)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read or aggregate one vector per label from h5ad files, construct "
            "a normalized concatenation consensus tree, and save the "
            "dendrogram plus machine-readable linkage/support tables."
        )
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        help="Number of consecutive seeds to expand in --path-template.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="First seed for consecutive seed discovery (default: 1).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit seed numbers; overrides --seed-start and --n-seeds.",
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--h5ad",
        type=Path,
        nargs="+",
        help="Explicit input h5ad paths.",
    )
    input_group.add_argument(
        "--path-template",
        required=False,
        help=(
            "Seed-dependent h5ad path containing one {seed} placeholder, "
            "for example '/results/seed_{seed}/adata.h5ad' or "
            "'/results/saturn_seed_{seed}.h5ad'."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help=(
            "Output path without an extension. If matching linkage, support, "
            "and leaves TSV files already exist, redraw the tree directly "
            "from them without loading embedding h5ads or rebuilding the tree. "
            "Otherwise defaults to the common input directory plus "
            "celltype_tree_<seed summary>."
        ),
    )
    parser.add_argument(
        "--representation",
        help="Vector source: X, obsm:<key>, or layer:<key>.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("none", "mean"),
        help="Whether selected rows are already label vectors or are averaged.",
    )
    parser.add_argument(
        "--filter-key",
        dest="filter_key",
        help="Optional obs column used to select rows before aggregation.",
    )
    parser.add_argument(
        "--filter-value",
        dest="filter_value",
        help="Required value in --filter-key.",
    )
    label_group = parser.add_mutually_exclusive_group(required=False)
    label_group.add_argument(
        "--label-key",
        help="obs label column used when --group-keys is omitted.",
    )
    label_group.add_argument(
        "--group-keys",
        nargs="+",
        help=(
            "obs columns defining aggregation groups and leaf labels, e.g. "
            "--group-keys species cell_type."
        ),
    )
    parser.add_argument(
        "--label-separator",
        default="_",
        help="Separator used to join multiple group values into a leaf label.",
    )
    parser.add_argument(
        "--gene",
        help=(
            "Optional gene whose mean expression is mapped to the tree as "
            "child-minus-parent change on each branch. Requires "
            "--expression-config with original expression and homology inputs."
        ),
    )
    parser.add_argument(
        "--expression-config",
        type=Path,
        help=(
            "YAML file specifying query_species and a training-style data "
            "block with species, homology, and cpm (positive number or null). "
            "Each species can set is_preprocessed. Gene mapping always "
            "disables HVG selection; n_HVG/select_hvg settings are ignored. "
            "Optional expression_layer selects an already processed layer."
        ),
    )
    parser.add_argument(
        "--species-key",
        default="species",
        help="obs column containing species names (default: species).",
    )
    parser.add_argument(
        "--species-map",
        action="append",
        nargs=2,
        metavar=("SOURCE", "TARGET"),
        help=(
            "Rename one species before filtering, grouping, and label "
            "construction. Repeat for multiple aliases, for example "
            "--species-map zebrafish fish. For cached trees, only displayed "
            "leaf names are changed; saved tables and calculations are retained."
        ),
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.8,
        help="Minimum fuzzy clade similarity contributing to support.",
    )
    parser.add_argument(
        "--support-metric",
        choices=("f1", "jaccard"),
        default="f1",
    )
    parser.add_argument(
        "--color-thr",
        type=float,
        nargs=2,
        metavar=("STRONG", "MID"),
        default=(80.0, 50.0),
        help="Support thresholds used for branch colors.",
    )
    parser.add_argument(
        "--orientation",
        choices=("top", "bottom", "left", "right"),
        default="right",
    )
    parser.add_argument(
        "--flip-lr",
        action=argparse.BooleanOptionalAction,
        default=TREE_FLIP_LR,
        help="Flip horizontal trees (default configured by TREE_FLIP_LR).",
    )
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> tuple[list[Path], list[int] | None]:
    if args.h5ad:
        if args.seeds or args.n_seeds is not None:
            raise ValueError(
                "--seeds and --n-seeds apply only to --path-template."
            )
        return [path.expanduser().resolve() for path in args.h5ad], None

    if args.seeds:
        seeds = list(args.seeds)
    else:
        if args.n_seeds is None:
            raise ValueError(
                "--path-template requires either --seeds or --n-seeds."
            )
        if args.n_seeds < 1:
            raise ValueError("--n-seeds must be at least 1.")
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed numbers must be unique.")

    fields = [
        field_name
        for _, field_name, _, _ in Formatter().parse(args.path_template)
        if field_name is not None
    ]
    if fields != ["seed"]:
        raise ValueError(
            "--path-template must contain exactly one {seed} placeholder."
        )
    paths = [
        Path(args.path_template.format(seed=seed)).expanduser()
        for seed in seeds
    ]
    return paths, seeds


def _default_output_prefix(
        paths: list[Path],
        seeds: list[int] | None,
    ) -> Path:
    if seeds:
        seed_summary = (
            f"seeds_{seeds[0]}-{seeds[-1]}"
            if seeds == list(range(seeds[0], seeds[-1] + 1))
            else "seeds_" + "-".join(map(str, seeds))
        )
    else:
        seed_summary = f"{len(paths)}_h5ads"
    common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
    return common_parent / f"celltype_tree_{seed_summary}"


def _parse_species_map(entries) -> dict[str, str]:
    mapping = {}
    for source, target in entries or ():
        previous = mapping.get(source)
        if previous is not None and previous != target:
            raise ValueError(
                f"Conflicting species mappings for {source!r}: "
                f"{previous!r} and {target!r}."
            )
        mapping[source] = target
    return mapping


def _rename_cached_leaf_labels(
        axis,
        labels_master: list[str],
        species_map: dict[str, str],
        *,
        label_separator: str,
        orientation: str,
    ) -> None:
    if not species_map:
        return
    if any(not source or not target for source, target in species_map.items()):
        raise ValueError("Species mapping names must be non-empty strings.")

    sources = sorted(species_map, key=len, reverse=True)

    def rename(label):
        for source in sources:
            if label == source:
                return species_map[source]
            prefix = f"{source}{label_separator}"
            if label.startswith(prefix):
                return f"{species_map[source]}{label_separator}{label[len(prefix):]}"
        return label

    label_map = {label: rename(label) for label in labels_master}
    if len(set(label_map.values())) != len(labels_master):
        raise ValueError("Species mapping would create duplicate tree leaf labels.")
    if all(label == renamed for label, renamed in label_map.items()):
        return

    horizontal = orientation in {"left", "right"}
    leaf_axis = axis.yaxis if horizontal else axis.xaxis
    ticks = leaf_axis.get_ticklabels()
    colors = [tick.get_color() for tick in ticks]
    labels = [label_map.get(tick.get_text(), tick.get_text()) for tick in ticks]
    leaf_axis.set_ticks(leaf_axis.get_ticklocs(), labels=labels)
    for tick, color in zip(leaf_axis.get_ticklabels(), colors):
        tick.set_color(color)


def _loader_options(args: argparse.Namespace) -> dict:
    return {
        "representation": args.representation,
        "aggregation": args.aggregation,
        "label_key": args.label_key,
        "group_keys": args.group_keys,
        "filter_key": args.filter_key,
        "filter_value": args.filter_value,
        "label_separator": args.label_separator,
        "species_key": args.species_key,
        "species_map": _parse_species_map(args.species_map),
    }


def _tree_table_paths(output_prefix: Path) -> tuple[Path, Path, Path, Path]:
    return (
        Path(f"{output_prefix}.linkage.tsv"),
        Path(f"{output_prefix}.support.tsv"),
        Path(f"{output_prefix}.leaves.tsv"),
        Path(f"{output_prefix}.inputs.tsv"),
    )


def _load_cached_tree(output_prefix: Path):
    linkage_path, support_path, leaves_path, manifest_path = (
        _tree_table_paths(output_prefix)
    )
    required_paths = (linkage_path, support_path, leaves_path)
    if not all(path.is_file() for path in required_paths):
        return None

    linkage_matrix = pd.read_csv(linkage_path, sep="\t")[[
        "left",
        "right",
        "distance",
        "leaf_count",
    ]].to_numpy(dtype=float)
    support_table = pd.read_csv(support_path, sep="\t")
    support = {
        frozenset(str(row.clade).split("|")): float(row.support_percent)
        for row in support_table.itertuples(index=False)
    }
    leaves_table = pd.read_csv(leaves_path, sep="\t").sort_values(
        "leaf_index"
    )
    labels_master = leaves_table["label"].astype(str).tolist()
    return (
        linkage_matrix,
        support,
        labels_master,
        linkage_path,
        support_path,
        leaves_path,
        manifest_path,
    )


def _load_expression_config(config_path: Path) -> dict:
    import yaml

    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Expression config must contain a YAML mapping.")

    query_species = config.get("query_species")
    if not isinstance(query_species, str) or not query_species:
        raise ValueError("Expression config requires query_species.")

    data_config = config.get("data", config)
    if not isinstance(data_config, dict):
        raise ValueError("Expression config data must be a mapping.")
    if "data" in config:
        duplicate_keys = {
            "species", "homology", "cpm", "n_HVG", "select_hvg"
        }.intersection(config)
        if duplicate_keys:
            raise ValueError(
                "Place these settings only inside data: "
                + ", ".join(sorted(duplicate_keys))
            )
    cpm = data_config.get("cpm", 1e4)
    if cpm is not None and (
        isinstance(cpm, bool) or not isinstance(cpm, (int, float))
        or not np.isfinite(cpm) or cpm <= 0
    ):
        raise ValueError("cpm must be null or a positive finite number.")

    def input_path(value, description):
        if not isinstance(value, str) or not value:
            raise ValueError(f"Expression config requires {description}.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
        return path

    species_entries = data_config.get("species")
    if not isinstance(species_entries, list) or not species_entries:
        raise ValueError("Expression config requires a non-empty species list.")
    species_inputs = {}
    preprocessed_species = {}
    for entry in species_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each expression species entry must be a mapping.")
        name = entry.get("name")
        label_col = entry.get("label_col")
        if not isinstance(name, str) or not name:
            raise ValueError("Each expression species requires a name.")
        if name in species_inputs:
            raise ValueError(f"Duplicate expression species: {name!r}.")
        if not isinstance(label_col, str) or not label_col:
            raise ValueError(f"Expression species {name!r} requires label_col.")
        is_preprocessed = entry.get("is_preprocessed", True)
        if not isinstance(is_preprocessed, bool):
            raise ValueError(
                f"{name} is_preprocessed must be a YAML boolean (true or false)."
            )
        species_inputs[name] = (
            input_path(entry.get("h5ad"), f"{name} expression h5ad"),
            label_col,
        )
        preprocessed_species[name] = is_preprocessed
    if query_species not in species_inputs:
        raise ValueError("query_species must occur in the expression species list.")

    homology_entries = data_config.get("homology")
    if not isinstance(homology_entries, list):
        raise ValueError("Expression config requires a homology list.")
    homology_files = {}
    for entry in homology_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each homology entry must be a mapping.")
        species1 = entry.get("species1")
        species2 = entry.get("species2")
        if (
            not isinstance(species1, str) or not species1
            or not isinstance(species2, str) or not species2
        ):
            raise ValueError("Each homology entry requires species1 and species2.")
        pair = (species1, species2)
        if pair in homology_files:
            raise ValueError(f"Duplicate homology pair: {pair!r}.")
        homology_files[pair] = input_path(
            entry.get("path"), f"homology table {species1}-{species2}"
        )

    layer = config.get("expression_layer")
    if layer is not None and (not isinstance(layer, str) or not layer):
        raise ValueError("expression_layer must be null or a layer name.")
    if layer is not None and not all(preprocessed_species.values()):
        raise ValueError(
            "CHORD preprocessing operates on .X. Set expression_layer: null "
            "to preprocess raw input. To read a stored layer, "
            "set every is_preprocessed: true."
        )

    return {
        "query_species": query_species,
        "species": species_inputs,
        "homology": homology_files,
        "expression_layer": layer,
        "preprocessing": {
            "cpm": cpm,
            "select_hvg": False,
            "is_preprocessed": preprocessed_species,
        },
    }


def _gene_expression_for_tree(
        expression_config: dict,
        gene: str,
        labels_master: list[str],
    ) -> dict[str, float]:
    from anndata import read_h5ad

    required_species = {label.split("_", 1)[0] for label in labels_master}
    missing_species = required_species - set(expression_config["species"])
    if missing_species:
        raise ValueError(
            "Expression config has no original h5ad for tree species: "
            + ", ".join(sorted(missing_species))
            + ". Config species names must match the tree leaf prefixes."
        )

    species_adatas = {}
    opened_adatas = []
    preprocessing = expression_config["preprocessing"]
    try:
        for species, (path, label_col) in expression_config["species"].items():
            is_preprocessed = preprocessing["is_preprocessed"][species]
            if not is_preprocessed:
                from chord.preprocessing import data_preprocess

                adata = data_preprocess(
                    path,
                    cpm=preprocessing["cpm"],
                    is_preprocessed=is_preprocessed,
                    select_hvg=False,
                )
            else:
                adata = read_h5ad(path, backed="r")
            opened_adatas.append(adata)
            print(
                f"Expression preprocessing [{species}]: "
                f"is_preprocessed={is_preprocessed}, "
                f"cpm={preprocessing['cpm']}, "
                "select_hvg=False (no HVG filtering)"
            )
            if label_col not in adata.obs:
                raise KeyError(f"{path} is missing obs column {label_col!r}.")
            layer = expression_config["expression_layer"]
            if layer is not None and layer not in adata.layers:
                raise KeyError(f"{path} is missing expression layer {layer!r}.")
            species_adatas[species] = (adata, label_col)

        species_gene = resolve_species_gene(
            expression_config["homology"],
            species_adatas,
            expression_config["query_species"],
            gene,
        )
        missing_genes = [
            species for species in sorted(required_species)
            if species_gene.get(species) is None
        ]
        if missing_genes:
            raise KeyError(
                f"Cannot resolve {expression_config['query_species']} gene "
                f"{gene!r} in expression h5ads for: "
                + ", ".join(missing_genes)
                + ". Check var_names and the original homology tables; "
                "the gene may be absent from the input or removed by QC."
            )

        label2expr = build_label2expr_OG_adata(
            labels_master,
            species_adatas,
            {
                species: index for species, index in species_gene.items()
                if index is not None
            },
            layer=expression_config["expression_layer"],
        )
        if not all(np.isfinite(value) for value in label2expr.values()):
            raise ValueError(f"Gene {gene!r} produces non-finite mean expression.")
        return label2expr
    finally:
        for adata in opened_adatas:
            adata.file.close()


def _write_tables(
        output_prefix: Path,
        linkage_matrix: np.ndarray,
        support: dict[frozenset[str], float],
        labels_master: list[str],
        paths: list[Path],
        seeds: list[int] | None,
        loader_options: dict,
) -> tuple[Path, Path, Path, Path]:
    linkage_path, support_path, leaves_path, manifest_path = (
        _tree_table_paths(output_prefix)
    )
    if not paths:
        return linkage_path, support_path, leaves_path, manifest_path

    pd.DataFrame(
        linkage_matrix,
        columns=("left", "right", "distance", "leaf_count"),
    ).to_csv(linkage_path, sep="\t", index=False)

    support_rows = [
        {
            "n_leaves": len(clade),
            "support_percent": float(value),
            "clade": "|".join(sorted(clade)),
        }
        for clade, value in support.items()
    ]
    pd.DataFrame(support_rows).sort_values(
        ["n_leaves", "clade"]
    ).to_csv(support_path, sep="\t", index=False)

    pd.DataFrame(
        {"leaf_index": np.arange(len(labels_master)), "label": labels_master}
    ).to_csv(leaves_path, sep="\t", index=False)

    pd.DataFrame(
        {
            "seed": seeds if seeds is not None else [None] * len(paths),
            "h5ad": [str(path) for path in paths],
            "representation": [loader_options["representation"]] * len(paths),
            "aggregation": [loader_options["aggregation"]] * len(paths),
            "label_key": [loader_options["label_key"]] * len(paths),
            "group_keys": [
                ",".join(loader_options["group_keys"] or ())
            ] * len(paths),
            "filter_key": [loader_options["filter_key"]] * len(paths),
            "filter_value": [loader_options["filter_value"]] * len(paths),
            "species_key": [loader_options["species_key"]] * len(paths),
            "species_map": [
                json.dumps(
                    loader_options["species_map"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ] * len(paths),
        }
    ).to_csv(manifest_path, sep="\t", index=False)
    return linkage_path, support_path, leaves_path, manifest_path


def main() -> None:
    args = parse_args()
    expression_config = None
    if args.gene is not None:
        if not args.gene.strip():
            raise ValueError("--gene must be a non-empty gene name.")
        if args.expression_config is None:
            raise ValueError(
                "--gene requires --expression-config with original expression "
                "h5ads and homology tables; embedding files are not expression."
            )
        expression_config = _load_expression_config(args.expression_config)
    output_prefix = (
        args.output_prefix.expanduser()
        if args.output_prefix is not None
        else None
    )
    cached_tree = (
        _load_cached_tree(output_prefix)
        if output_prefix is not None
        else None
    )
    loaded_from_cache = cached_tree is not None

    if loaded_from_cache:
        (
            linkage_matrix,
            support,
            labels_master,
            linkage_path,
            support_path,
            leaves_path,
            manifest_path,
        ) = cached_tree
        paths = []
        seeds = None
        loader_options = {
            "representation": "cached tree tables",
            "aggregation": "not recomputed",
            "label_key": None,
            "group_keys": None,
            "filter_key": None,
            "filter_value": None,
        }
    else:
        if not args.h5ad and not args.path_template:
            raise ValueError(
                "No complete cached tree was found for --output-prefix. "
                "Provide either --h5ad or --path-template to construct it."
            )
        if args.representation is None:
            raise ValueError(
                "--representation is required when constructing a tree."
            )
        if args.aggregation is None:
            raise ValueError(
                "--aggregation is required when constructing a tree."
            )
        if args.label_key is None and not args.group_keys:
            raise ValueError(
                "Either --label-key or --group-keys is required when "
                "constructing a tree."
            )

        paths, seeds = _resolve_inputs(args)
        loader_options = _loader_options(args)
        missing = [path for path in paths if not path.is_file()]
        if missing:
            preview = "\n".join(f"  - {path}" for path in missing[:20])
            suffix = (
                ""
                if len(missing) <= 20
                else f"\n  ... and {len(missing) - 20} more"
            )
            raise FileNotFoundError(
                f"Missing {len(missing)} embedding h5ad file(s):\n"
                f"{preview}{suffix}"
            )

        embeddings, labels_master, row_labels = (
            load_label_embeddings_from_h5ads(
                paths,
                **loader_options,
            )
        )
        linkage_matrix, support = consensus_support_fuzzy_threshold(
            embeddings,
            labels_master,
            row_labels_list=row_labels,
            method="ward",
            metric=args.support_metric,
            min_sim=args.min_sim,
        )
        if len(paths) == 1:
            print(
                "Warning: one input produces a valid tree, but its replicate "
                "support is trivially 100% and should not be interpreted as "
                "robustness."
            )

        if output_prefix is None:
            output_prefix = _default_output_prefix(paths, seeds)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure_size = (
        FIGURE_WIDTH_CM / CM_PER_INCH,
        FIGURE_HEIGHT_CM / CM_PER_INCH,
    )
    if args.gene is not None:
        label2expr = _gene_expression_for_tree(
            expression_config,
            args.gene,
            labels_master,
        )
        figure, axis = plot_tree_drift_fullcolor(
            linkage_matrix,
            labels_master,
            label2expr,
            cmap_name="bwr",
            size=figure_size,
            lw_h=0.6,
            lw_v=0.6,
            lw_base=0.6,
            flip_lr=args.flip_lr,
            set_arial=False,
            axis_labelsize=WARD_DISTANCE_LABEL_SIZE_PT,
            tick_labelsize=LEAF_LABEL_SIZE_PT,
            orientation=args.orientation,
        )
    else:
        figure, axis = plot_dendrogram_support_colors(
            linkage_matrix,
            support,
            labels_master,
            size=figure_size,
            color_thr=tuple(args.color_thr),
            fontsize=LEAF_LABEL_SIZE_PT,
            orientation=args.orientation,
            flip_lr=args.flip_lr,
        )
    if loaded_from_cache:
        _rename_cached_leaf_labels(
            axis,
            labels_master,
            _parse_species_map(args.species_map),
            label_separator=args.label_separator,
            orientation=args.orientation,
        )
    axis.set_title("")
    if figure._suptitle is not None:
        figure._suptitle.remove()
    if args.orientation in {"left", "right"}:
        axis.set_xlabel(
            "Ward distance", fontsize=WARD_DISTANCE_LABEL_SIZE_PT
        )
    else:
        axis.set_ylabel(
            "Ward distance", fontsize=WARD_DISTANCE_LABEL_SIZE_PT
        )
    axis.margins(
        x=AXIS_MARGIN_FRACTION,
        y=AXIS_MARGIN_FRACTION,
    )
    figure.tight_layout(pad=LAYOUT_PADDING)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    figure_prefix = output_prefix
    if args.gene is not None:
        gene_tag = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in args.gene
        )
        figure_prefix = Path(f"{output_prefix}.gene_{gene_tag}")
    pdf_path = Path(f"{figure_prefix}.pdf")
    png_path = Path(f"{figure_prefix}.png")
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=OUTPUT_DPI)
    plt.close(figure)
    linkage_path, support_path, leaves_path, manifest_path = _write_tables(
        output_prefix,
        linkage_matrix,
        support,
        labels_master,
        paths,
        seeds,
        loader_options,
    )

    if loaded_from_cache:
        print(f"Loaded cached tree tables for: {output_prefix}")
        print("Skipped h5ad loading and tree construction.")
    else:
        print(f"Loaded {len(paths)} embedding h5ad file(s).")
        print(
            "Preprocessing: "
            f"representation={loader_options['representation']}, "
            f"aggregation={loader_options['aggregation']}, "
            f"group_keys={loader_options['group_keys']}, "
            f"species_map={loader_options['species_map']}"
        )
    print(f"Tree leaves: {len(labels_master)}")
    if args.gene is not None:
        print(
            f"Expression mapping: {expression_config['query_species']} "
            f"gene {args.gene!r}; delta = child - parent."
        )
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print(f"Linkage: {linkage_path}")
    print(f"Support: {support_path}")
    print(f"Leaves: {leaves_path}")
    print(f"Inputs: {manifest_path}")


if __name__ == "__main__":
    main()
