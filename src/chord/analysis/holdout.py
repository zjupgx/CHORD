
import json

import pandas as pd


def evaluate_cell_type_holdout(adata, *, n_neighbors=15, seed=0, rep="X"):
    from chord.data.holdout import HOLDOUT_KEY, select_cell_type_holdout_cells
    from .evaluation import evaluate_batch, evaluate_celltype

    cells = select_cell_type_holdout_cells(adata)
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 2:
        raise ValueError("Hold-out n_neighbors must be an integer >= 2.")
    n_neighbors = min(n_neighbors, cells.n_obs - 1)
    nmi, celltype_asw, ari = evaluate_celltype(
        cells, rep=rep, n_neighbors=n_neighbors, seed=seed, calculate_trajectory=False,
    )
    batch_asw, graph_connectivity = evaluate_batch(
        cells, batch_key="species", rep=rep, n_neighbors=n_neighbors, seed=seed,
    )
    metadata = adata.uns.get(HOLDOUT_KEY, {})
    return pd.DataFrame([{
        "cohort": "all_species_holdout_only",
        "training_seed": metadata.get("training_seed"),
        "holdout_seed": metadata.get("seed"),
        "n_cell_types": cells.obs["cell_type"].nunique(),
        "cell_types": json.dumps(sorted(set(cells.obs["cell_type"].astype(str)))),
        "n_cells": cells.n_obs,
        "n_species": cells.obs["species"].nunique(),
        "representation": rep, "metric_seed": seed, "n_neighbors": n_neighbors,
        "nmi": nmi, "ari": ari, "celltype_asw": celltype_asw,
        "batch_asw": batch_asw, "graph_connectivity": graph_connectivity,
    }])
