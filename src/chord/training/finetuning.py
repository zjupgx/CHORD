from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


def load_matching_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Load only parameters whose names exist in the current model and
    whose shapes match.

    Supports both:
    - a plain state_dict
    - a checkpoint dict containing "model_state_dict"

    Parameters
    ----------
    model
        Current model instance.
    checkpoint_path
        Path to pretrained checkpoint.
    map_location
        Device mapping for torch.load.
    verbose
        Whether to print loaded / skipped layers.

    Returns
    -------
    loaded_keys
        Parameter names successfully loaded.
    skipped_keys
        Parameter names skipped due to missing name or shape mismatch.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        pretrained_state_dict = ckpt["model_state_dict"]
    else:
        pretrained_state_dict = ckpt

    current_model_dict = model.state_dict()

    new_state_dict = OrderedDict()
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    for k, v in pretrained_state_dict.items():
        if k in current_model_dict and v.shape == current_model_dict[k].shape:
            new_state_dict[k] = v
            loaded_keys.append(k)
            if verbose:
                print(f"Loaded layer: {k} with shape {tuple(v.shape)}")
        else:
            skipped_keys.append(k)
            if verbose:
                print(f"Skipped layer: {k}")

    model.load_state_dict(new_state_dict, strict=False)

    if verbose:
        print("\n--- Partial weight loading complete ---")
        print(f"Loaded {len(loaded_keys)} layers.")
        print(f"Skipped {len(skipped_keys)} layers.")

    return loaded_keys, skipped_keys


def freeze_all_parameters(model: nn.Module) -> None:
    """
    Freeze all model parameters.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_parameters_by_prefix(
    model: nn.Module,
    prefixes: Iterable[str],
    verbose: bool = True,
) -> list[str]:
    """
    Unfreeze parameters whose names start with any prefix in prefixes.

    Parameters
    ----------
    model
        Model instance.
    prefixes
        Iterable of parameter name prefixes.
    verbose
        Whether to print unfrozen parameter names.

    Returns
    -------
    unfrozen_names
        Names of parameters that were unfrozen.
    """
    prefixes = tuple(prefixes)
    unfrozen_names: list[str] = []

    for name, param in model.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad = True
            unfrozen_names.append(name)
            if verbose:
                print(f"Unfrozen -> {name}")

    return unfrozen_names


def set_finetune_trainable_params(
    model: nn.Module,
    trainable_prefixes: Iterable[str],
    verbose: bool = True,
) -> list[str]:
    """
    Freeze all parameters first, then unfreeze selected parameter groups.

    Parameters
    ----------
    model
        Model instance.
    trainable_prefixes
        Parameter prefixes to unfreeze.
    verbose
        Whether to print logs.

    Returns
    -------
    unfrozen_names
        Names of trainable parameters after applying the strategy.
    """
    freeze_all_parameters(model)
    unfrozen_names = unfreeze_parameters_by_prefix(
        model=model,
        prefixes=trainable_prefixes,
        verbose=verbose,
    )

    if verbose:
        print("\n--- Parameter freezing complete ---")
        print(f"Trainable parameter groups matched: {len(unfrozen_names)}")

    return unfrozen_names


def get_trainable_parameters(
    model: nn.Module,
    verbose: bool = True,
) -> list[nn.Parameter]:
    """
    Collect parameters with requires_grad=True.

    Parameters
    ----------
    model
        Model instance.
    verbose
        Whether to print trainable parameter names.

    Returns
    -------
    params_to_update
        List of trainable parameters.
    """
    params_to_update: list[nn.Parameter] = []
    trainable_names: list[str] = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            trainable_names.append(name)

    if verbose:
        print("\nParameters to be trained:")
        for name in trainable_names:
            print(f"\t{name}")

    return params_to_update


def build_finetune_optimizer(
    model: nn.Module,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    """
    Build an Adam optimizer using only trainable parameters.

    Parameters
    ----------
    model
        Model instance.
    lr
        Learning rate.
    weight_decay
        Weight decay.
    verbose
        Whether to print trainable parameter names.

    Returns
    -------
    optimizer
        Adam optimizer over trainable parameters only.
    """
    params_to_update = get_trainable_parameters(model, verbose=verbose)

    optimizer = torch.optim.Adam(
        params_to_update,
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer