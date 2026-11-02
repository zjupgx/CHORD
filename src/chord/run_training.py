from __future__ import annotations

import argparse
import random
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any
from itertools import combinations

import numpy as np
import torch
import yaml

from chord.models import CrossSpeciesPrototypeModel, CrossSpeciesPrototypeLoss
from chord.models.loss import validate_loss_weight
from chord.data import (
    CrossSpeciesDataset,
    create_dataloader,
    create_val_loaders,
    intervention_enabled,
    get_intervention_block,
    get_intervention_tsv_filename,
    parse_fraction,
    validate_interventions_config,
    build_label_perturbation_manifest,
    prepare_species_views,
    build_homology_anchor_views,
    annotate_homology_withholding,
    build_cell_type_holdout_views,
)
from chord.training import (
    train,
    validate,
    organize_result,
    extract_embeddings_from_adatas,
    organize_result_from_adatas,
)
from chord.preprocessing import (
    data_preprocess,
    prepare_multispecies_data,
    get_homology_indices,
)
from chord.training.hierarchy import (
    get_hierarchy_config,
    prepare_hierarchy_views,
    build_hierarchy_model,
)

warnings.filterwarnings("ignore")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CHORD training pipeline from a YAML config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config file must define a top-level dictionary.")
    return cfg


def _require_keys(d: dict[str, Any], keys: list[str], prefix: str = "") -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing required keys in {prefix or 'config'}: {missing}")


def normalize_stage_loss_weight(stage_cfg: dict[str, Any], *, prefix: str) -> None:
    if not isinstance(stage_cfg, dict):
        raise ValueError(f"{prefix} must be a dictionary.")

    has_loss_weight = "loss_weight" in stage_cfg
    has_named_weights = "lambda_g" in stage_cfg or "lambda_c" in stage_cfg
    if not has_loss_weight and not has_named_weights:
        raise KeyError(
            f"{prefix} requires either loss_weight or both lambda_g and lambda_c."
        )

    if has_loss_weight:
        validate_loss_weight(stage_cfg["loss_weight"], prefix=f"{prefix}.loss_weight")

    if has_named_weights:
        _require_keys(stage_cfg, ["lambda_g", "lambda_c"], prefix=prefix)
        named_weights = [
            stage_cfg["lambda_g"],
            stage_cfg["lambda_c"],
            stage_cfg["lambda_c"],
        ]
        validate_loss_weight(
            named_weights,
            prefix=f"{prefix}.loss_weight derived from lambda_g/lambda_c",
        )
        if has_loss_weight:
            if any(
                old_value != named_value
                for old_value, named_value in zip(stage_cfg["loss_weight"], named_weights)
            ):
                raise ValueError(
                    f"{prefix}.loss_weight={stage_cfg['loss_weight']!r} conflicts "
                    f"with [lambda_g, lambda_c, lambda_c]={named_weights!r}."
                )
        else:
            stage_cfg["loss_weight"] = named_weights


def get_prototype_init_method(model_cfg: dict[str, Any]) -> str:
    prototype_init = model_cfg.get("prototype_init", {})
    if not isinstance(prototype_init, dict):
        raise ValueError("model.prototype_init must be a dictionary.")
    unknown_keys = set(prototype_init) - {"method"}
    if unknown_keys:
        raise ValueError(
            "model.prototype_init supports only 'method'; unknown keys: "
            f"{sorted(unknown_keys)}"
        )
    method = prototype_init.get("method", "random")
    if not isinstance(method, str) or method not in {
        "random",
        "random_cell",
        "mean",
    }:
        raise ValueError(
            "model.prototype_init.method must be 'random', 'random_cell', "
            "or 'mean'."
        )
    return method


def validate_config(cfg: dict[str, Any]) -> None:
    _require_keys(cfg, ["model", "training", "data", "output"])

    _require_keys(cfg["model"], ["embed_dim", "hidden_dim"], prefix="model")
    get_prototype_init_method(cfg["model"])
    _require_keys(cfg["training"], ["batch_size", "device", "stage1", "stage2"], prefix="training")
    for stage_name in ("stage1", "stage2"):
        stage_cfg = cfg["training"][stage_name]
        stage_prefix = f"training.{stage_name}"
        normalize_stage_loss_weight(stage_cfg, prefix=stage_prefix)
        _require_keys(stage_cfg, ["temperature", "lr", "epochs"], prefix=stage_prefix)

    seed = cfg["training"].get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("training.seed must be a non-negative integer.")

    gene_masking_cfg = cfg["training"].get("gene_masking", {})
    if not isinstance(gene_masking_cfg, dict):
        raise ValueError("training.gene_masking must be a dictionary.")
    gene_masking_enabled = gene_masking_cfg.get("enabled", True)
    if not isinstance(gene_masking_enabled, bool):
        raise ValueError("training.gene_masking.enabled must be a boolean.")
    parse_fraction(
        gene_masking_cfg.get("probability", 0.1),
        "training.gene_masking.probability",
    )

    get_hierarchy_config(cfg)
    validate_interventions_config(cfg)

    _require_keys(cfg["data"], ["species", "homology"], prefix="data")
    select_hvg = cfg["data"].get("select_hvg", True)
    if not isinstance(select_hvg, bool):
        raise ValueError("data.select_hvg must be a boolean.")
    if not isinstance(cfg["data"]["species"], list) or len(cfg["data"]["species"]) < 2:
        raise ValueError("data.species must be a list with at least 2 species entries.")
    if not isinstance(cfg["data"]["homology"], list) or len(cfg["data"]["homology"]) < 1:
        raise ValueError("data.homology must be a non-empty list.")

    _require_keys(cfg["output"], ["outdir"], prefix="output")

    species_items = cfg["data"]["species"]
    homology_items = cfg["data"]["homology"]

    for item in species_items:
        _require_keys(item, ["name", "h5ad", "label_col"], prefix="data.species[]")

    for item in homology_items:
        _require_keys(item, ["species1", "species2", "path"], prefix="data.homology[]")

    species_names = [item["name"] for item in species_items]

    if len(species_names) != len(set(species_names)):
        raise ValueError(f"Duplicate species names found in data.species: {species_names}")

    species_name_set = set(species_names)

    for item in homology_items:
        s1 = item["species1"]
        s2 = item["species2"]

        if s1 not in species_name_set:
            raise ValueError(f"Homology species1 '{s1}' is not defined in data.species.")
        if s2 not in species_name_set:
            raise ValueError(f"Homology species2 '{s2}' is not defined in data.species.")
        if s1 == s2:
            raise ValueError(f"Homology pair cannot use the same species twice: ({s1}, {s2})")

    provided_pairs = [
        tuple(sorted((item["species1"], item["species2"])))
        for item in homology_items
    ]
    if len(provided_pairs) != len(set(provided_pairs)):
        raise ValueError(f"Duplicate homology pairs found: {provided_pairs}")

    required_pairs = {
        tuple(sorted(pair))
        for pair in combinations(species_names, 2)
    }
    provided_pair_set = set(provided_pairs)

    missing_pairs = required_pairs - provided_pair_set
    if missing_pairs:
        raise ValueError(
            "Missing homology files for species pairs: "
            f"{sorted(missing_pairs)}"
        )


def build_species_adatas(
    cfg: dict[str, Any],
    *,
    select_hvg: bool = True,
) -> dict[str, tuple[Any, str]]:
    species_adatas: dict[str, tuple[Any, str]] = {}

    n_HVG = cfg["data"].get("n_HVG", 2000)
    cpm = cfg["data"].get("cpm", 1e4)

    for item in cfg["data"]["species"]:
        _require_keys(item, ["name", "h5ad", "label_col"], prefix="data.species[]")

        species_name = item["name"]
        h5ad_path = item["h5ad"]
        label_col = item["label_col"]
        is_preprocessed = item.get("is_preprocessed", True)

        adata = data_preprocess(
            h5ad_path,
            n_top_genes=n_HVG,
            cpm=cpm,
            is_preprocessed=is_preprocessed,
            select_hvg=select_hvg,
        )
        species_adatas[species_name] = (adata, label_col)

    return species_adatas


def build_homology_files(cfg: dict[str, Any]) -> dict[tuple[str, str], str]:
    homology_files: dict[tuple[str, str], str] = {}

    for item in cfg["data"]["homology"]:
        _require_keys(item, ["species1", "species2", "path"], prefix="data.homology[]")
        s1 = item["species1"]
        s2 = item["species2"]
        path = item["path"]
        homology_files[(s1, s2)] = path

    return homology_files


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def save_config_snapshot(cfg: dict[str, Any], outdir: Path) -> Path:
    path = outdir / "config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
    return path


def run_hierarchy_finetuning(
    cfg, hierarchy_cfg, source_model, source_config, target_adatas,
    homology_indices, device,
) -> None:
    species_data, species_config, species_expr_avg, global_cell_types = (
        prepare_multispecies_data(target_adatas)
    )
    model = build_hierarchy_model(
        source_model, source_config, species_config, cfg["model"], device,
    )
    outdir = Path(cfg["output"]["outdir"]) / "hierarchy"
    outdir.mkdir(parents=True, exist_ok=True)

    source_snapshot = outdir / "source_config.yaml"
    with open(source_snapshot, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
    effective_cfg = deepcopy(cfg)
    for item in effective_cfg["data"]["species"]:
        item["label_col"] = hierarchy_cfg["target_label_col"]
    effective_cfg["training"]["hierarchy"] = {
        **hierarchy_cfg, "enabled": False,
    }
    effective_cfg["output"]["outdir"] = str(outdir)
    save_config_snapshot(effective_cfg, outdir)

    batch_size = cfg["training"]["batch_size"]
    generator = torch.Generator().manual_seed(cfg["training"].get("seed", 42))
    dataloader = create_dataloader(
        CrossSpeciesDataset(species_data), batch_size=batch_size,
        num_workers=4, generator=generator,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=hierarchy_cfg["lr"],
    )
    model_save_path = outdir / cfg["output"].get("model_filename", "best_model.pth")
    masking = cfg["training"].get("gene_masking", {})
    train(
        model, dataloader, species_expr_avg, homology_indices, device, optimizer,
        CrossSpeciesPrototypeLoss([0.0, 0.1, 0.1], hierarchy_cfg["temperature"]),
        epochs=hierarchy_cfg["epochs"], model_save_path=model_save_path,
        gene_masking=masking.get("enabled", True),
        gene_mask_probability=parse_fraction(
            masking.get("probability", 0.1), "training.gene_masking.probability",
        ),
    )
    model.load_state_dict(torch.load(model_save_path, map_location="cpu"), strict=True)
    for species in species_config:
        if not torch.equal(
            model.species_encoders[species].gene_embedding,
            source_model.species_encoders[species].gene_embedding,
        ):
            raise RuntimeError(f"[{species}] frozen gene embeddings changed during hierarchy finetuning.")
    result = validate(model, create_val_loaders(species_data, batch_size, num_workers=4), device)
    adata_cell, adata_gene = organize_result(
        result, species_config, species_expr_avg, global_cell_types,
    )
    adata_cell.write_h5ad(outdir / "adata_cell_embeddings.h5ad")
    adata_gene.write_h5ad(outdir / "adata_gene_embeddings.h5ad")
    print(f"Hierarchy finetuning completed. Output directory: {outdir}")


def run_pipeline(cfg: dict[str, Any]) -> None:
    validate_config(cfg)

    model_cfg = cfg["model"]
    prototype_init_method = get_prototype_init_method(model_cfg)
    training_cfg = cfg["training"]
    stage1_cfg = training_cfg["stage1"]
    stage2_cfg = training_cfg["stage2"]
    output_cfg = cfg["output"]

    seed = training_cfg.get("seed", 42)
    set_random_seed(seed)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)

    outdir = Path(output_cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = save_config_snapshot(cfg, outdir)
    print(f"Saved run config to: {config_snapshot_path}")

    model_filename = output_cfg.get("model_filename", "best_model.pth")
    model_save_path = outdir / model_filename

    device = resolve_device(training_cfg["device"])
    batch_size = training_cfg["batch_size"]

    gene_masking_cfg = training_cfg.get("gene_masking", {})
    gene_masking_enabled = gene_masking_cfg.get("enabled", True)
    gene_mask_probability = parse_fraction(
        gene_masking_cfg.get("probability", 0.1),
        "training.gene_masking.probability",
    )

    label_perturbation_enabled = intervention_enabled(
        cfg, "label_perturbation"
    )
    homology_withholding_enabled = intervention_enabled(
        cfg, "homology_anchor_withholding"
    )
    holdout_enabled = intervention_enabled(cfg, "cell_type_holdout")
    if holdout_enabled:
        full_species_adatas, training_species_adatas, holdout_metadata = (
            build_cell_type_holdout_views(cfg)
        )
        holdout_metadata["training_seed"] = seed
        species_adatas = full_species_adatas
        print(f"Globally held-out cell types: {holdout_metadata['cell_types']}")
    else:
        species_adatas = build_species_adatas(
            cfg,
            select_hvg=cfg["data"].get("select_hvg", True),
        )
    homology_files = build_homology_files(cfg)
    hierarchy_cfg = get_hierarchy_config(cfg)
    hierarchy_adatas = (
        prepare_hierarchy_views(species_adatas, hierarchy_cfg["target_label_col"])
        if hierarchy_cfg is not None else None
    )

    if label_perturbation_enabled:
        label_cfg = (
            get_intervention_block(cfg, "label_perturbation")
            if label_perturbation_enabled
            else None
        )
        full_species_adatas, training_species_adatas, intervention_metadata = (
            prepare_species_views(
                species_adatas,
                label_perturbation_config=label_cfg,
                default_seed=seed,
            )
        )
        for name, value in intervention_metadata.items():
            print(f"{name}: {value}")
    elif not holdout_enabled:
        full_species_adatas = species_adatas
        training_species_adatas = species_adatas

    if label_perturbation_enabled:
        label_tsv_filename = get_intervention_tsv_filename(
            cfg, "label_perturbation"
        )
        if label_tsv_filename is not None:
            label_manifest_path = outdir / label_tsv_filename
            build_label_perturbation_manifest(full_species_adatas).to_csv(
                label_manifest_path,
                sep="\t",
                index=False,
            )
            print(
                "Saved label perturbation manifest to: "
                f"{label_manifest_path}"
            )

    species_data, species_config, species_expr_avg, global_cell_types = (
        prepare_multispecies_data(training_species_adatas)
    )
    withheld_homology_pairs = None
    if homology_withholding_enabled:
        homology_cfg = get_intervention_block(
            cfg, "homology_anchor_withholding"
        )
        (
            _full_homology_indices,
            homology_indices,
            withheld_homology_pairs,
        ) = build_homology_anchor_views(
            species_config,
            homology_files,
            homology_cfg,
            default_seed=seed,
        )
        homology_tsv_filename = get_intervention_tsv_filename(
            cfg, "homology_anchor_withholding"
        )
        if homology_tsv_filename is not None:
            homology_manifest_path = outdir / homology_tsv_filename
            withheld_homology_pairs.to_csv(
                homology_manifest_path,
                sep="\t",
                index=False,
            )
            print(
                "Saved withheld homology-pair manifest to: "
                f"{homology_manifest_path}"
            )
    else:
        homology_indices = get_homology_indices(
            species_config,
            homology_files,
        )

    dataset = CrossSpeciesDataset(species_data)
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        generator=dataloader_generator,
    )
    val_loaders = (
        None
        if label_perturbation_enabled or holdout_enabled
        else create_val_loaders(species_data, batch_size)
    )

    model = CrossSpeciesPrototypeModel(
        species_config,
        model_cfg["embed_dim"],
        model_cfg["hidden_dim"],
        model_cfg["embed_dim"],
    ).to(device)

    criterion1 = CrossSpeciesPrototypeLoss(
        stage1_cfg["loss_weight"],
        stage1_cfg["temperature"],
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

    print("Stage 1 training completed.")

    if prototype_init_method == "random_cell":
        model.initialize_prototypes_from_random_cells(species_data, seed)
        print("Initialized cell-type prototypes from random training cells after Stage 1.")
    elif prototype_init_method == "mean":
        model.initialize_prototypes_from_expression_means(species_expr_avg)
        print("Initialized cell-type prototypes from mean expression after Stage 1.")

    criterion2 = CrossSpeciesPrototypeLoss(
        stage2_cfg["loss_weight"],
        stage2_cfg["temperature"],
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

    print("Stage 2 training completed.")

    best_state_dict = torch.load(model_save_path, map_location="cpu")
    model.load_state_dict(best_state_dict, strict=True)
    del best_state_dict
    print(f"Loaded best Stage 2 model from: {model_save_path}")
    
    if label_perturbation_enabled or holdout_enabled:
        result = extract_embeddings_from_adatas(
            model,
            full_species_adatas,
            device,
            batch_size,
        )
        adata_cell, adata_gene = organize_result_from_adatas(
            result,
            full_species_adatas,
            species_config,
            species_expr_avg,
            global_cell_types,
            include_split=holdout_enabled,
            include_label_perturbation=label_perturbation_enabled,
        )
    else:
        result = validate(model, val_loaders, device)
        adata_cell, adata_gene = organize_result(
            result,
            species_config,
            species_expr_avg,
            global_cell_types,
        )

    if holdout_enabled:
        adata_cell.uns["cell_type_holdout"] = dict(holdout_metadata)
        adata_gene.uns["cell_type_holdout"] = dict(holdout_metadata)

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

    adata_cell.write_h5ad(outdir / "adata_cell_embeddings.h5ad")
    adata_gene.write_h5ad(outdir / "adata_gene_embeddings.h5ad")

    if holdout_enabled:
        from chord.analysis.holdout import evaluate_cell_type_holdout

        metrics = evaluate_cell_type_holdout(adata_cell)
        metrics_path = outdir / "holdout_integration_metrics.tsv"
        metrics.to_csv(metrics_path, sep="\t", index=False)
        print(f"Saved joint hold-out-only integration metrics to: {metrics_path}")

    print(f"Saved AnnData results to: {outdir / 'adata_cell_embeddings.h5ad'}, {outdir / 'adata_gene_embeddings.h5ad'}")
    print(f"Output directory: {outdir}")
    if hierarchy_cfg is not None:
        run_hierarchy_finetuning(
            cfg, hierarchy_cfg, model, species_config, hierarchy_adatas,
            homology_indices, device,
        )
    print("Run completed.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
