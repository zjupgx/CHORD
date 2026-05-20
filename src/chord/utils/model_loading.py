from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models import CrossSpeciesPrototypeModel
from ..preprocessing import (
    data_preprocess,
    prepare_multispecies_data,
    get_homology_indices,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config file must define a top-level dictionary.")
    return cfg


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_species_adatas(cfg: dict[str, Any]) -> dict[str, tuple[Any, str]]:
    """
    Rebuild species_adatas in the same way as training.

    Returns
    -------
    dict
        {species_name: (adata, label_col)}
    """
    species_adatas: dict[str, tuple[Any, str]] = {}

    n_HVG = cfg["data"].get("n_HVG", 2000)
    cpm = cfg["data"].get("cpm", 1e4)

    for item in cfg["data"]["species"]:
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
    """
    Build homology file mapping:
    {(species1, species2): homology_path}
    """
    homology_files: dict[tuple[str, str], str] = {}

    for item in cfg["data"]["homology"]:
        s1 = item["species1"]
        s2 = item["species2"]
        path = item["path"]
        homology_files[(s1, s2)] = path

    return homology_files


def load_chord_model_and_homology(
    cfg: dict[str, Any],
    model_path: str | Path,
    device: str | None = None,
):
    """
    Rebuild CHORD model from config and load trained weights.
    Also rebuild homology_indices.

    Parameters
    ----------
    cfg
        Parsed YAML config.
    model_path
        Path to the saved model weights, e.g. outdir/best_model.pth
    device
        "cpu", "cuda", "auto", or None.
        If None, use cfg["training"]["device"].

    Returns
    -------
    model : CrossSpeciesPrototypeModel
    homology_indices : Any
    species_config : dict
    species_data : Any
    species_expr_avg : Any
    global_cell_types : Any
    """
    model_path = Path(model_path)

    if device is None:
        device = cfg["training"]["device"]
    device = resolve_device(device)

    # 1. Rebuild species inputs
    species_adatas = build_species_adatas(cfg)

    # 2. Rebuild homology file mapping
    homology_files = build_homology_files(cfg)

    # 3. Rebuild multi-species data and species_config
    species_data, species_config, species_expr_avg, global_cell_types = (
        prepare_multispecies_data(species_adatas)
    )

    # 4. Rebuild homology indices
    homology_indices = get_homology_indices(
        species_config,
        homology_files,
    )

    # 5. Rebuild model structure
    model = CrossSpeciesPrototypeModel(
        species_config,
        cfg["model"]["embed_dim"],
        cfg["model"]["hidden_dim"],
        cfg["model"]["embed_dim"],
    ).to(device)

    # 6. Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return (
        model,
        homology_indices,
        species_config,
        species_data,
        species_expr_avg,
        global_cell_types,
    )