
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any


def get_hierarchy_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    block = cfg["training"].get("hierarchy")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("training.hierarchy must be a dictionary.")
    enabled = block.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("training.hierarchy.enabled must be a boolean.")
    if not enabled:
        return None

    interventions = cfg.get("interventions", {})
    if not isinstance(interventions, dict):
        raise ValueError("interventions must be a dictionary.")
    conflicts = [
        f"interventions.{name}"
        for name, intervention in interventions.items()
        if isinstance(intervention, dict) and intervention.get("enabled") is True
    ]
    if conflicts:
        raise ValueError(
            "training.hierarchy.enabled=true conflicts with enabled interventions: "
            + ", ".join(sorted(conflicts))
            + ". Disable the interventions before running hierarchy finetuning."
        )

    filename = cfg["output"].get("model_filename", "best_model.pth")
    if (
        not isinstance(filename, str) or not filename.strip()
        or "/" in filename or "\\" in filename or Path(filename).drive
        or filename.casefold() in {
            "hierarchy", "config.yaml", "source_config.yaml",
            "adata_cell_embeddings.h5ad", "adata_gene_embeddings.h5ad",
        }
    ):
        raise ValueError(
            "Hierarchy finetuning requires output.model_filename to be a filename "
            "that does not conflict with its other outputs."
        )

    target = block.get("target_label_col")
    if not isinstance(target, str) or not target.strip() or target != target.strip():
        raise ValueError("training.hierarchy.target_label_col must be a non-empty column name.")
    if all(item["label_col"] == target for item in cfg["data"]["species"]):
        raise ValueError("training.hierarchy.target_label_col must change the training labels.")

    resolved = dict(block)
    resolved["epochs"] = block.get("epochs", 100)
    epochs = resolved["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("training.hierarchy.epochs must be a positive integer.")
    for name, default in (
        ("lr", 5e-3),
        ("temperature", cfg["training"]["stage2"]["temperature"]),
    ):
        value = block.get(name, default)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.hierarchy.{name} must be a finite positive number.")
        resolved[name] = value
    return resolved


def prepare_hierarchy_views(species_adatas, target_label_col):
    views = {}
    for species, (adata, source_label_col) in species_adatas.items():
        for column in (source_label_col, target_label_col):
            if column not in adata.obs:
                raise ValueError(f"[{species}] hierarchy label column {column!r} is missing.")
            labels = adata.obs[column]
            if len(labels) == 0 or labels.isna().any():
                raise ValueError(f"[{species}] hierarchy label column {column!r} is empty or contains missing labels.")
        views[species] = (adata, target_label_col)
    return views


def build_hierarchy_model(source_model, source_config, target_config, model_cfg, device):
    from chord.models import CrossSpeciesPrototypeModel

    if set(source_config) != set(target_config):
        raise ValueError("Hierarchy finetuning must retain the same species.")
    for species in source_config:
        if list(source_config[species]["gene_names"]) != list(target_config[species]["gene_names"]):
            raise ValueError(f"[{species}] hierarchy finetuning changed the gene order.")

    model = CrossSpeciesPrototypeModel(
        target_config, model_cfg["embed_dim"], model_cfg["hidden_dim"],
        model_cfg["embed_dim"],
    ).to(device)
    source, target = source_model.state_dict(), model.state_dict()
    for name in target:
        if name.endswith(".gene_embedding") or name.startswith(("cell_mlp.", "celltype_mlp.")):
            if name not in source or source[name].shape != target[name].shape:
                raise ValueError(f"Hierarchy weight transfer failed for {name!r}.")
            target[name] = source[name].detach().clone()
    model.load_state_dict(target, strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.endswith(".gene_embedding")
    return model
