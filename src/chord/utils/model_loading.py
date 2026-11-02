from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models import CrossSpeciesPrototypeModel
from ..data.holdout import get_cell_type_holdout_config, build_cell_type_holdout_views
from ..data.interventions import (
    intervention_enabled,
    get_intervention_block,
    prepare_species_views,
)
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


def build_species_adatas(
    cfg: dict[str, Any],
    *,
    select_hvg: bool = True,
) -> dict[str, tuple[Any, str]]:
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
            select_hvg=select_hvg,
        )
        species_adatas[species_name] = (adata, label_col)

    return species_adatas


def build_homology_files(cfg: dict[str, Any]) -> dict[tuple[str, str], str]:
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
    model_path = Path(model_path)

    if device is None:
        device = cfg["training"]["device"]
    device = resolve_device(device)

    label_perturbation_enabled = intervention_enabled(
        cfg, "label_perturbation"
    )
    if get_cell_type_holdout_config(cfg) is not None:
        _full_species_adatas, species_adatas, _metadata = build_cell_type_holdout_views(cfg)
    else:
        species_adatas = build_species_adatas(
            cfg,
            select_hvg=cfg["data"].get("select_hvg", True),
        )
    if label_perturbation_enabled:
        _full_species_adatas, species_adatas, _metadata = prepare_species_views(
            species_adatas,
            label_perturbation_config=(
                get_intervention_block(cfg, "label_perturbation")
                if label_perturbation_enabled
                else None
            ),
            default_seed=cfg["training"].get("seed", 42),
        )

    homology_files = build_homology_files(cfg)

    species_data, species_config, species_expr_avg, global_cell_types = (
        prepare_multispecies_data(species_adatas)
    )

    homology_indices = get_homology_indices(
        species_config,
        homology_files,
    )

    model = CrossSpeciesPrototypeModel(
        species_config,
        cfg["model"]["embed_dim"],
        cfg["model"]["hidden_dim"],
        cfg["model"]["embed_dim"],
    ).to(device)

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
