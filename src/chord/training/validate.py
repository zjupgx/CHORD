import torch
import anndata
import numpy as np
import pandas as pd


def validate(model, val_loaders, device):
    """
    Run validation for each species separately and collect model outputs.

    For each species-specific validation loader, this function:
    1. Runs the model without masking.
    2. Collects cell embeddings batch by batch.
    3. Stores labels for all validation cells.
    4. Saves cell-type embeddings and gene embeddings once per species.

    Args:
        model:
            Trained model.
        val_loaders:
            Dictionary mapping species name to its validation DataLoader.
        device:
            Target device for inference.

    Returns:
        Dictionary mapping species name to:
            - cell_embed: concatenated cell embeddings
            - cell_type_embed: cell-type embeddings
            - gene_embed: gene embeddings
            - labels: concatenated cell labels
    """
    model.eval()
    result = {}

    with torch.no_grad():
        for sp, loader in val_loaders.items():
            # Initialize storage for one species
            result[sp] = {
                'cell_embed': [],
                'cell_type_embed': None,
                'gene_embed': None,
                'labels': []
            }

            for data, labels in loader:
                # Build a single-species input batch
                batch = {sp: (data.to(device), labels.to(device))}
                outputs = model(batch, device, mask=False)

                sp_out = outputs[sp]
                result[sp]['cell_embed'].append(
                    sp_out['cell_embed'].cpu().numpy()
                )
                result[sp]['labels'].append(labels.cpu().numpy())

                # These two outputs are constant within one validation pass,
                # so they only need to be stored once
                if result[sp]['cell_type_embed'] is None:
                    result[sp]['cell_type_embed'] = sp_out['cell_type_embed'].cpu().numpy()
                if result[sp]['gene_embed'] is None:
                    result[sp]['gene_embed'] = sp_out['gene_embed'].cpu().numpy()

            # Concatenate all batches for the current species
            result[sp]['cell_embed'] = np.concatenate(result[sp]['cell_embed'], axis=0)
            result[sp]['labels'] = np.concatenate(result[sp]['labels'], axis=0)

    return result


def organize_result(
    result: dict,
    species_config: dict,
    species_expr_avg: dict,
    global_cell_types: list[str],
    species_time_dict=None
):
    """
    Organize validation outputs into AnnData objects for cells, cell types,
    and genes.

    This function creates:
    1. One AnnData object containing both cell embeddings and cell-type
       embeddings.
    2. One AnnData object containing gene embeddings.
    3. A list of species-specific cell-type labels.

    If `species_time_dict` is provided, the time annotation is added to
    cell observations and checked against the number of cells.

    Args:
        result:
            Output dictionary from `validate`.
        species_config:
            Per-species metadata, including gene names.
        species_expr_avg:
            Per-species average-expression metadata, including
            cell-type presence vectors.
        global_cell_types:
            Global list of cell-type names aligned to the shared encoder.
        species_time_dict:
            Optional dictionary mapping species name to a time series.

    Returns:
        adata_cell:
            AnnData containing both cell and cell-type embeddings.
        adata_gene:
            AnnData containing gene embeddings.
    """
    species_order, num_celltypes_per_species, all_type_labels = [], [], []
    cell_celltype_embeddings, obs_cell_celltype, index_cell_celltype = [], [], []
    gene_embeddings, obs_genes, index_genes = [], [], []

    cell_counter = celltype_counter = gene_counter = 0

    for species in result:
        species_order.append(species)

        # ---------------- Cell embeddings ----------------
        cell_embeddings = result[species]["cell_embed"]
        cell_labels = result[species]["labels"]
        n_cells = len(cell_embeddings)
        cell_celltype_embeddings.append(cell_embeddings)

        # Read and validate optional time annotations for cells
        time_series = None
        if species_time_dict is not None and species in species_time_dict:
            ts = species_time_dict[species]
            if len(ts) != n_cells:
                raise ValueError(
                    f"[organize_result] `{species}` time length {len(ts)} "
                    f"does not match its cell count {n_cells}"
                )
            time_series = ts.reset_index(drop=True)

        for i in range(n_cells):
            cell_type_idx = int(cell_labels[i])
            cell_type_name = global_cell_types[cell_type_idx]
            label_str = f"{species}_{cell_type_name}"

            row = {
                "species": species,
                "cell_type": cell_type_name,
                "item_type": "cell",
                "label_str": label_str,
            }

            if time_series is not None:
                row["Time"] = time_series[i]

            obs_cell_celltype.append(row)
            index_cell_celltype.append(f"cell_{cell_counter}")
            cell_counter += 1

        # ---------------- Cell-type embeddings ----------------
        sp_mask = torch.tensor(
            species_expr_avg[species]["cell_type_presence"], dtype=torch.bool
        )
        num_celltype = int(sp_mask.sum().item())
        num_celltypes_per_species.append(num_celltype)

        type_embeddings = result[species]["cell_type_embed"][sp_mask]
        cell_celltype_embeddings.append(type_embeddings)

        for i, present in enumerate(sp_mask):
            if present:
                cell_type_name = global_cell_types[i]
                label_str = f"{species}_{cell_type_name}"
                all_type_labels.append(label_str)

                obs_cell_celltype.append({
                    "species": species,
                    "cell_type": cell_type_name,
                    "item_type": "celltype",
                    "label_str": label_str,
                    "Time": None
                })
                index_cell_celltype.append(f"celltype_{celltype_counter}")
                celltype_counter += 1

        # ---------------- Gene embeddings ----------------
        gene_embeddings.append(result[species]["gene_embed"])
        for gene_id in species_config[species]["gene_names"]:
            obs_genes.append({
                "species": species,
                "gene_id": gene_id,
                "label_str": f"{species}_{gene_id}"
            })
            index_genes.append(f"gene_{gene_counter}")
            gene_counter += 1

    # ---------------- Cell + cell-type AnnData ----------------
    cell_celltype_embedding = np.concatenate(cell_celltype_embeddings)
    adata_cell = anndata.AnnData(
        X=cell_celltype_embedding,
        obs=pd.DataFrame(obs_cell_celltype, index=index_cell_celltype),
        dtype=np.float32
    )

    adata_cell.uns.update({
        "species_order": species_order,
        "global_cell_types": global_cell_types,
        "num_celltypes_per_species": num_celltypes_per_species,
        "all_type_labels": all_type_labels
    })

    adata_cell.obs["item_type"] = adata_cell.obs["item_type"].astype("category")
    adata_cell.obs["cell_type"] = adata_cell.obs["cell_type"].astype("category")
    adata_cell.obs["species"] = adata_cell.obs["species"].astype("category")
    adata_cell.obs["Time"] = adata_cell.obs["Time"].astype("category")

    # ---------------- Gene AnnData ----------------
    adata_gene = anndata.AnnData(
        X=np.concatenate(gene_embeddings),
        obs=pd.DataFrame(obs_genes, index=index_genes),
        dtype=np.float32
    )
    adata_gene.uns["species_order"] = species_order
    adata_gene.obs["species"] = adata_gene.obs["species"].astype("category")

    return adata_cell, adata_gene
