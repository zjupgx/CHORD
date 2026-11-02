
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from chord.data import CrossSpeciesDataset, create_dataloader, create_val_loaders
from chord.models import CrossSpeciesPrototypeModel
from chord.preprocessing import prepare_multispecies_data
from chord.run_training import set_random_seed
from chord.training import organize_result, validate
from chord.training.hierarchy import build_hierarchy_model, prepare_hierarchy_views
from chord.utils.model_loading import build_species_adatas, load_config, resolve_device


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Fit high-resolution prototypes on an existing CHORD model.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="Existing low-resolution run directory.")
    parser.add_argument("--config", type=Path, help="Default: source-dir/config.yaml; relative input paths follow the working directory.")
    parser.add_argument("--model", type=Path, help="Default: source-dir/<output.model_filename>.")
    parser.add_argument("--output-dir", type=Path, help="Default: source-dir/prototype_only/.")
    parser.add_argument("--target-label-col", default="Cross_species")
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--lr", type=positive_float, default=0.005)
    parser.add_argument("--temperature", type=positive_float, help="Default: source stage2.temperature.")
    parser.add_argument("--batch-size", type=positive_int, help="Default: source training.batch_size.")
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers. Default: 4 on all platforms.",
    )
    parser.add_argument("--seed", type=int, help="Default: source training.seed.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def fit_prototypes(model, species_data, epochs, lr, temperature, batch_size,
                   seed, device, masking, checkpoint, num_workers=4):
    generator = torch.Generator().manual_seed(seed)
    loader = create_dataloader(
        CrossSpeciesDataset(species_data), batch_size=batch_size,
        num_workers=num_workers, generator=generator,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr,
    )
    model.eval()
    rows, best_loss = [], None
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch, device, mask=masking[0], mask_prob=masking[1])
            losses = []
            for species in batch:
                cells = F.normalize(outputs[species]["cell_embed"], dim=1)
                prototypes = F.normalize(outputs[species]["cell_type_embed"], dim=1)
                logits = cells @ prototypes.T / temperature
                presence = model.species_encoders[species].ct_mask.bool()
                logits = logits.masked_fill(~presence.unsqueeze(0), -1e9)
                losses.append(F.cross_entropy(logits, batch[species][1].to(device)))
            loss = 0.1 * (len(batch) - 1) * torch.stack(losses).sum()
            loss.backward()
            optimizer.step()
            total += loss.item()
        mean_loss = total / len(loader)
        rows.append({"epoch": epoch + 1, "cell_loss": mean_loss})
        if best_loss is None or mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), checkpoint)
        print(f"Epoch {epoch + 1}/{epochs}: cell loss={mean_loss:.6f}", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model.eval()
    return rows


def add_annotation_columns(cells, species_adatas):
    source_labels, source_columns = {}, {}
    for species, (adata, source_col) in species_adatas.items():
        source_labels.update({
            (species, str(cell_id)): str(label)
            for cell_id, label in zip(adata.obs_names, adata.obs[source_col])
        })
        source_columns[species] = source_col
    cells.obs["cell_type_high"] = cells.obs["cell_type"].astype(str)
    low_labels = []
    for _, row in cells.obs.iterrows():
        if str(row["item_type"]) == "cell":
            low_labels.append(source_labels[(str(row["species"]), str(row["cell_id"]))])
        else:
            low_labels.append(None)
    cells.obs["cell_type_low"] = pd.Categorical(low_labels)
    cells.uns["source_label_columns"] = source_columns


def main():
    args = parse_args()
    source_dir = args.source_dir.resolve()
    config_path = (args.config or source_dir / "config.yaml").resolve()
    cfg = load_config(config_path)
    conflicts = [name for name, block in (cfg.get("interventions") or {}).items()
                 if isinstance(block, dict) and block.get("enabled") is True]
    if conflicts:
        raise ValueError(f"Prototype-only adaptation rejects enabled interventions: {sorted(conflicts)}")
    if len(cfg["data"]["species"]) < 2:
        raise ValueError("Prototype-only adaptation requires at least two species.")
    if all(item["label_col"] == args.target_label_col for item in cfg["data"]["species"]):
        raise ValueError("The target label column must differ from the source labels.")
    seed = cfg["training"].get("seed", 42) if args.seed is None else args.seed
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    temperature = args.temperature if args.temperature is not None else cfg["training"]["stage2"]["temperature"]
    if batch_size <= 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Batch size and temperature must be positive.")
    model_path = (args.model or source_dir / cfg["output"].get("model_filename", "best_model.pth")).resolve()
    outdir = (args.output_dir or source_dir / "prototype_only").resolve()
    if outdir == source_dir or outdir in source_dir.parents:
        raise ValueError("The adaptation output directory must not overwrite the source run.")
    if config_path in {outdir / "config.yaml", outdir / "source_config.yaml"} or model_path == outdir / "best_model.pth":
        raise ValueError("Adaptation output paths collide with source inputs.")

    set_random_seed(seed)
    device = resolve_device(args.device)
    species_adatas = build_species_adatas(cfg, select_hvg=cfg["data"].get("select_hvg", True))
    source_data, source_config, _, _ = prepare_multispecies_data(species_adatas)
    model_cfg = cfg["model"]
    source_model = CrossSpeciesPrototypeModel(
        source_config, model_cfg["embed_dim"], model_cfg["hidden_dim"], model_cfg["embed_dim"],
    ).to(device)
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    source_model.load_state_dict(state, strict=True)
    source_model.eval()
    target_adatas = prepare_hierarchy_views(species_adatas, args.target_label_col)
    target_data, target_config, target_means, target_labels = prepare_multispecies_data(target_adatas)
    del source_data
    model = build_hierarchy_model(source_model, source_config, target_config, model_cfg, device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.endswith(".prototype")

    from chord.data.interventions import parse_fraction
    mask_cfg = cfg["training"].get("gene_masking", {})
    mask_enabled = mask_cfg.get("enabled", True)
    if not isinstance(mask_enabled, bool):
        raise ValueError("training.gene_masking.enabled must be a boolean.")
    mask_probability = parse_fraction(mask_cfg.get("probability", 0.1), "training.gene_masking.probability")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = fit_prototypes(
        model, target_data, args.epochs, args.lr, temperature, batch_size,
        seed, device, (mask_enabled, mask_probability), outdir / "best_model.pth",
        num_workers=args.num_workers,
    )
    result = validate(model, create_val_loaders(target_data, batch_size, num_workers=args.num_workers), device)
    cells, genes = organize_result(result, target_config, target_means, target_labels)
    add_annotation_columns(cells, species_adatas)
    cells.uns["prototype_only_adaptation"] = True
    cells.write_h5ad(outdir / "adata_cell_embeddings.h5ad")
    genes.write_h5ad(outdir / "adata_gene_embeddings.h5ad")
    pd.DataFrame(rows).to_csv(outdir / "training_history.tsv", sep="\t", index=False)
    effective = deepcopy(cfg)
    for item in effective["data"]["species"]:
        item["label_col"] = args.target_label_col
    effective["training"]["hierarchy"] = {"enabled": False}
    effective["output"] = {**effective["output"], "outdir": str(outdir), "model_filename": "best_model.pth"}
    for filename, snapshot in (("config.yaml", effective), ("source_config.yaml", cfg)):
        with (outdir / filename).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)
    metadata = {
        "source_dir": str(source_dir), "source_config": str(config_path),
        "source_checkpoint": str(model_path), "target_label_col": args.target_label_col,
        "seed": seed, "epochs": args.epochs, "lr": args.lr, "temperature": temperature,
        "batch_size": batch_size, "num_workers": args.num_workers,
        "lambda_g": 0.0, "lambda_c": 0.1,
        "gene_masking": mask_enabled, "mask_probability": mask_probability,
        "mlp_dropout": "disabled; both frozen MLPs remain in eval mode",
        "trainable_parameters": [name for name, p in model.named_parameters() if p.requires_grad],
        "input_path_base": "process working directory, as in the original training command",
    }
    (outdir / "prototype_only_settings.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Prototype-only adaptation saved to: {outdir}")


if __name__ == "__main__":
    main()
