from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any
from itertools import combinations

import numpy as np
import torch
import yaml

from chord.models import CrossSpeciesPrototypeModel, CrossSpeciesPrototypeLoss
from chord.data import CrossSpeciesDataset, create_dataloader, create_val_loaders
from chord.training import train, validate, organize_result
from chord.preprocessing import (
    data_preprocess,
    prepare_multispecies_data,
    get_homology_indices,
)

warnings.filterwarnings("ignore")


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


def validate_config(cfg: dict[str, Any]) -> None:
    _require_keys(cfg, ["model", "training", "data", "output"])

    _require_keys(cfg["model"], ["embed_dim", "hidden_dim"], prefix="model")
    _require_keys(cfg["training"], ["batch_size", "device", "stage1", "stage2"], prefix="training")
    _require_keys(cfg["training"]["stage1"], ["loss_weight", "temperature", "lr", "epochs"], prefix="training.stage1")
    _require_keys(cfg["training"]["stage2"], ["loss_weight", "temperature", "lr", "epochs"], prefix="training.stage2")

    _require_keys(cfg["data"], ["species", "homology"], prefix="data")
    if not isinstance(cfg["data"]["species"], list) or len(cfg["data"]["species"]) < 2:
        raise ValueError("data.species must be a list with at least 2 species entries.")
    if not isinstance(cfg["data"]["homology"], list) or len(cfg["data"]["homology"]) < 1:
        raise ValueError("data.homology must be a non-empty list.")

    _require_keys(cfg["output"], ["outdir"], prefix="output")

    # ------------------------------
    # extra checks for species / homology consistency
    # ------------------------------
    species_items = cfg["data"]["species"]
    homology_items = cfg["data"]["homology"]

    for item in species_items:
        _require_keys(item, ["name", "h5ad", "label_col"], prefix="data.species[]")

    for item in homology_items:
        _require_keys(item, ["species1", "species2", "path"], prefix="data.homology[]")

    species_names = [item["name"] for item in species_items]

    # 1) species name uniqueness
    if len(species_names) != len(set(species_names)):
        raise ValueError(f"Duplicate species names found in data.species: {species_names}")

    species_name_set = set(species_names)

    # 2) homology references must exist in species
    for item in homology_items:
        s1 = item["species1"]
        s2 = item["species2"]

        if s1 not in species_name_set:
            raise ValueError(f"Homology species1 '{s1}' is not defined in data.species.")
        if s2 not in species_name_set:
            raise ValueError(f"Homology species2 '{s2}' is not defined in data.species.")
        if s1 == s2:
            raise ValueError(f"Homology pair cannot use the same species twice: ({s1}, {s2})")

    # 3) duplicate homology pair detection (order-insensitive)
    provided_pairs = [
        tuple(sorted((item["species1"], item["species2"])))
        for item in homology_items
    ]
    if len(provided_pairs) != len(set(provided_pairs)):
        raise ValueError(f"Duplicate homology pairs found: {provided_pairs}")

    # 4) require full pair coverage for all species combinations
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


def build_species_adatas(cfg: dict[str, Any]) -> dict[str, tuple[Any, str]]:
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


def run_pipeline(cfg: dict[str, Any]) -> None:
    validate_config(cfg)

    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    stage1_cfg = training_cfg["stage1"]
    stage2_cfg = training_cfg["stage2"]
    output_cfg = cfg["output"]

    outdir = Path(output_cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    model_filename = output_cfg.get("model_filename", "best_model.pth")
    model_save_path = outdir / model_filename

    device = resolve_device(training_cfg["device"])
    batch_size = training_cfg["batch_size"]

    record_last_k_epochs = stage2_cfg.get("record_last_k_epochs", 50)

    # ------------------------------
    # Load input data
    # ------------------------------
    species_adatas = build_species_adatas(cfg)
    homology_files = build_homology_files(cfg)

    # ------------------------------
    # Prepare multi-species data
    # ------------------------------
    species_data, species_config, species_expr_avg, global_cell_types = (
        prepare_multispecies_data(species_adatas)
    )
    homology_indices = get_homology_indices(
        species_config,
        homology_files
    )

    # ------------------------------
    # Build loaders
    # ------------------------------
    dataset = CrossSpeciesDataset(species_data)
    dataloader = create_dataloader(dataset, batch_size=batch_size)
    val_loaders = create_val_loaders(species_data, batch_size)

    # ------------------------------
    # Build model
    # ------------------------------
    model = CrossSpeciesPrototypeModel(
        species_config,
        model_cfg["embed_dim"],
        model_cfg["hidden_dim"],
        model_cfg["embed_dim"],
    ).to(device)

    # ------------------------------
    # Stage 1 training
    # ------------------------------
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
        record_last_k_epochs=None,
    )

    print("Stage 1 training completed.")

    # ------------------------------
    # Stage 2 training
    # ------------------------------
    criterion2 = CrossSpeciesPrototypeLoss(
        stage2_cfg["loss_weight"],
        stage2_cfg["temperature"],
    )
    optimizer2 = torch.optim.Adam(model.parameters(), lr=stage2_cfg["lr"])

    cell_type_embedding_history = train(
        model,
        dataloader,
        species_expr_avg,
        homology_indices,
        device,
        optimizer2,
        criterion2,
        epochs=stage2_cfg["epochs"],
        model_save_path=model_save_path,
        record_last_k_epochs=record_last_k_epochs,
    )

    if cell_type_embedding_history is not None:
        np.save(outdir / "stage2_cell_type_embedding_history.npy", np.stack(cell_type_embedding_history, axis=0))
        print(f"Model record file: {outdir / 'stage2_cell_type_embedding_history.npy'}")
    
    print("Stage 2 training completed.")
    
    # ------------------------------
    # Validation and result packing
    # ------------------------------
    result = validate(model, val_loaders, device)
    adata_cell, adata_gene = organize_result(
        result,
        species_config,
        species_expr_avg,
        global_cell_types,
    )

    # ------------------------------
    # Save results
    # ------------------------------
    adata_cell.write_h5ad(outdir / "adata_cell_embeddings.h5ad")
    adata_gene.write_h5ad(outdir / "adata_gene_embeddings.h5ad")

    print(f"Saved AnnData results to: {outdir / 'adata_cell_embeddings.h5ad'}, {outdir / 'adata_gene_embeddings.h5ad'}")
    print(f"Output directory: {outdir}")
    print("Run completed.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()