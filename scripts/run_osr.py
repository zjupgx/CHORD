from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import scanpy as sc
import torch

from chord.data import (
    CrossSpeciesDataset,
    annotate_homology_withholding,
    build_homology_anchor_views,
    build_label_perturbation_manifest,
    build_osr_training_split_manifest,
    build_training_species_adatas,
    create_dataloader,
    get_intervention_block,
    get_intervention_tsv_filename,
    intervention_enabled,
    parse_fraction,
    perturb_training_labels,
    prepare_osr_species_views,
    split_osr_raw_species_adatas,
    validate_osr_training_config,
)
from chord.models import CrossSpeciesPrototypeLoss, CrossSpeciesPrototypeModel
from chord.preprocessing import get_homology_indices, prepare_multispecies_data
from chord.run_training import (
    build_homology_files,
    build_species_adatas,
    get_prototype_init_method,
    load_config,
    resolve_device,
    save_config_snapshot,
    set_random_seed,
    validate_config,
    get_hierarchy_config,
)
from chord.training import extract_embeddings_from_adatas, organize_result_from_adatas, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CHORD and export embeddings for later OSR evaluation."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def _save_raw_osr_split_h5ads(
    cfg: dict[str, Any],
    manifest,
    outdir: Path,
    *,
    seed: int,
) -> None:
    output_paths: dict[tuple[str, str], Path] = {}
    used_filenames: set[str] = set()
    for item in cfg["data"]["species"]:
        species = item["name"]
        if not isinstance(species, str) or not species.strip():
            raise ValueError("data.species[].name must be a non-empty string.")
        if species != species.strip() or Path(species).name != species:
            raise ValueError(
                f"Species name {species!r} cannot be used in an output filename."
            )
        for partition in ("train", "nontrain"):
            filename = f"{species}-{partition}-{seed}.h5ad"
            normalized = filename.casefold()
            if normalized in used_filenames:
                raise ValueError(f"Duplicate OSR split h5ad filename: {filename}")
            used_filenames.add(normalized)
            output_paths[(species, partition)] = outdir / filename

    for item in cfg["data"]["species"]:
        species = item["name"]
        raw_adata = sc.read_h5ad(item["h5ad"])
        species_manifest = manifest.loc[
            manifest["species"].astype(str) == species
        ].copy()
        partitions = split_osr_raw_species_adatas(
            {species: raw_adata},
            species_manifest,
        )[species]
        for partition, adata in partitions.items():
            path = output_paths[(species, partition)]
            adata.uns["osr_split_seed"] = seed
            adata.write_h5ad(path)
            print(f"Saved raw OSR {partition} h5ad to: {path}")


def run_osr_pipeline(cfg: dict[str, Any]) -> None:
    validate_config(cfg)
    if intervention_enabled(cfg, "cell_type_holdout"):
        raise ValueError("Cell-type hold-out is supported only by chord.run_training, not OSR training.")
    if get_hierarchy_config(cfg) is not None:
        raise ValueError("training.hierarchy is supported only by chord.run_training, not OSR training.")
    training_cfg = cfg["training"]
    prototype_init_method = get_prototype_init_method(cfg["model"])
    stage1_cfg = training_cfg["stage1"]
    stage2_cfg = training_cfg["stage2"]
    seed = int(training_cfg.get("seed", 42))
    osr_cfg = validate_osr_training_config(cfg, default_seed=seed)

    requested_tsvs: list[str] = []
    if osr_cfg["save_split_tsv"]:
        requested_tsvs.append(osr_cfg["split_tsv_filename"])
    for intervention in (
        "label_perturbation",
        "homology_anchor_withholding",
    ):
        filename = get_intervention_tsv_filename(cfg, intervention)
        if filename is not None:
            requested_tsvs.append(filename)
    normalized_names = [name.casefold() for name in requested_tsvs]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError(
            "OSR training TSV outputs must use distinct filenames: "
            f"{requested_tsvs}"
        )

    outdir = Path(cfg["output"]["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saved run config to: {save_config_snapshot(cfg, outdir)}")

    set_random_seed(seed)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)
    device = resolve_device(training_cfg["device"])
    batch_size = int(training_cfg["batch_size"])

    original_species_adatas = build_species_adatas(cfg, select_hvg=False)
    full_species_adatas, training_species_adatas, osr_metadata = (
        prepare_osr_species_views(
            original_species_adatas,
            osr_cfg,
            n_top_genes=cfg["data"].get("n_HVG", 2000),
        )
    )
    unknown_by_species = osr_metadata["unknown_cell_types_by_species"]
    print(
        "OSR types: "
        f"known={len(osr_metadata['known_cell_types'])}, "
        f"unknown={{{', '.join(f'{key}: {len(value)}' for key, value in unknown_by_species.items())}}}"
    )

    split_manifest = None
    if osr_cfg["save_split_tsv"] or osr_cfg["save_split_h5ad"]:
        split_manifest = build_osr_training_split_manifest(
            original_species_adatas,
            full_species_adatas,
        )
    if osr_cfg["save_split_tsv"]:
        path = outdir / osr_cfg["split_tsv_filename"]
        split_manifest.to_csv(path, sep="\t", index=False)
        print(f"Saved fixed OSR training split manifest to: {path}")
    if osr_cfg["save_split_h5ad"]:
        _save_raw_osr_split_h5ads(
            cfg,
            split_manifest,
            outdir,
            seed=osr_cfg["seed"],
        )

    label_perturbation_enabled = intervention_enabled(
        cfg, "label_perturbation"
    )
    intervention_metadata: dict[str, Any] = {}
    if label_perturbation_enabled:
        full_species_adatas, changed_counts = perturb_training_labels(
            full_species_adatas,
            get_intervention_block(cfg, "label_perturbation"),
            default_seed=seed,
        )
        intervention_metadata["label_perturbation_counts"] = changed_counts
        training_species_adatas = build_training_species_adatas(
            full_species_adatas,
            split_enabled=True,
            label_perturbation_enabled=True,
        )
        label_filename = get_intervention_tsv_filename(
            cfg, "label_perturbation"
        )
        if label_filename is not None:
            path = outdir / label_filename
            build_label_perturbation_manifest(full_species_adatas).to_csv(
                path, sep="\t", index=False
            )
            print(f"Saved label perturbation manifest to: {path}")

    species_data, species_config, species_expr_avg, global_cell_types = (
        prepare_multispecies_data(training_species_adatas)
    )
    homology_files = build_homology_files(cfg)
    homology_withholding_enabled = intervention_enabled(
        cfg, "homology_anchor_withholding"
    )
    withheld_homology_pairs = None
    if homology_withholding_enabled:
        homology_cfg = get_intervention_block(
            cfg, "homology_anchor_withholding"
        )
        _, homology_indices, withheld_homology_pairs = build_homology_anchor_views(
            species_config,
            homology_files,
            homology_cfg,
            default_seed=seed,
        )
        homology_filename = get_intervention_tsv_filename(
            cfg, "homology_anchor_withholding"
        )
        if homology_filename is not None:
            path = outdir / homology_filename
            withheld_homology_pairs.to_csv(path, sep="\t", index=False)
            print(f"Saved withheld homology-pair manifest to: {path}")
    else:
        homology_indices = get_homology_indices(species_config, homology_files)

    dataset = CrossSpeciesDataset(species_data)
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        generator=dataloader_generator,
    )
    model = CrossSpeciesPrototypeModel(
        species_config,
        cfg["model"]["embed_dim"],
        cfg["model"]["hidden_dim"],
        cfg["model"]["embed_dim"],
    ).to(device)
    model_save_path = outdir / osr_cfg["model_filename"]
    gene_masking_cfg = training_cfg.get("gene_masking", {})
    gene_masking_enabled = gene_masking_cfg.get("enabled", True)
    gene_mask_probability = parse_fraction(
        gene_masking_cfg.get("probability", 0.1),
        "training.gene_masking.probability",
    )

    criterion1 = CrossSpeciesPrototypeLoss(
        stage1_cfg["loss_weight"], stage1_cfg["temperature"]
    )
    optimizer1 = torch.optim.Adam(model.parameters(), lr=stage1_cfg["lr"])
    train(
        model,
        dataloader,
        species_expr_avg,
        homology_indices,
        device,
        optimizer1,
        criterion1,
        epochs=stage1_cfg["epochs"],
        model_save_path=model_save_path,
        gene_masking=gene_masking_enabled,
        gene_mask_probability=gene_mask_probability,
    )
    print("OSR Stage 1 training completed.")

    if prototype_init_method == "random_cell":
        model.initialize_prototypes_from_random_cells(species_data, seed)
        print("Initialized cell-type prototypes from random training cells after Stage 1.")
    elif prototype_init_method == "mean":
        model.initialize_prototypes_from_expression_means(species_expr_avg)
        print("Initialized cell-type prototypes from mean expression after Stage 1.")

    criterion2 = CrossSpeciesPrototypeLoss(
        stage2_cfg["loss_weight"], stage2_cfg["temperature"]
    )
    optimizer2 = torch.optim.Adam(model.parameters(), lr=stage2_cfg["lr"])
    train(
        model,
        dataloader,
        species_expr_avg,
        homology_indices,
        device,
        optimizer2,
        criterion2,
        epochs=stage2_cfg["epochs"],
        model_save_path=model_save_path,
        gene_masking=gene_masking_enabled,
        gene_mask_probability=gene_mask_probability,
    )
    print("OSR Stage 2 training completed.")

    best_state_dict = torch.load(model_save_path, map_location="cpu")
    model.load_state_dict(best_state_dict, strict=True)
    del best_state_dict

    result = extract_embeddings_from_adatas(
        model, full_species_adatas, device, batch_size
    )
    adata_cell, adata_gene = organize_result_from_adatas(
        result,
        full_species_adatas,
        species_config,
        species_expr_avg,
        global_cell_types,
        include_split=True,
        include_label_perturbation=label_perturbation_enabled,
    )
    adata_cell.uns["osr_known_cell_types"] = osr_metadata[
        "known_cell_types"
    ]
    adata_cell.uns["osr_unknown_min_cells"] = osr_cfg["unknown_min_cells"]
    adata_cell.uns["osr_known_cell_split_ratios"] = osr_cfg[
        "known_cell_split_ratios"
    ]
    type_table = osr_metadata["cell_type_table"].copy()
    type_table.index = [f"cell_type_{index}" for index in range(len(type_table))]
    adata_cell.uns["osr_cell_type_table"] = type_table

    if label_perturbation_enabled:
        label_cfg = get_intervention_block(cfg, "label_perturbation")
        adata_cell.uns["label_perturbation_enabled"] = True
        adata_cell.uns["label_perturbation_seed"] = int(
            label_cfg.get("seed", seed)
        )
        adata_cell.uns["label_perturbation_fraction"] = parse_fraction(
            label_cfg.get("fraction", 0.0),
            "interventions.label_perturbation.fraction",
        )
        adata_cell.uns["label_perturbation_counts"] = intervention_metadata[
            "label_perturbation_counts"
        ]

    if homology_withholding_enabled:
        homology_cfg = get_intervention_block(
            cfg, "homology_anchor_withholding"
        )
        annotate_homology_withholding(
            adata_gene,
            withheld_homology_pairs,
            fraction=parse_fraction(
                homology_cfg.get("fraction", 0.0),
                "interventions.homology_anchor_withholding.fraction",
            ),
        )
        adata_gene.uns["homology_anchor_withholding_seed"] = int(
            homology_cfg.get("seed", seed)
        )

    cell_path = outdir / osr_cfg["cell_embeddings_filename"]
    gene_path = outdir / osr_cfg["gene_embeddings_filename"]
    adata_cell.write_h5ad(cell_path)
    adata_gene.write_h5ad(gene_path)
    print(f"Saved OSR embeddings to: {cell_path}, {gene_path}")
    print(
        "Training/export completed. Run `python scripts/evaluate_osr.py --config ...` "
        "to evaluate the saved cell embeddings."
    )


def main() -> None:
    args = parse_args()
    run_osr_pipeline(load_config(args.config))


if __name__ == "__main__":
    main()
