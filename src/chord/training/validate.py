import torch
import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data.interventions import (
    KNOWN_STATUS_COLUMN,
    LABEL_PERTURBED_COLUMN,
    SPLIT_COLUMN,
    TRAIN_LABEL_COLUMN,
    TRUE_LABEL_COLUMN,
)


def validate(model, val_loaders, device):
    model.eval()
    result = {}

    with torch.no_grad():
        for sp, loader in val_loaders.items():
            result[sp] = {
                'cell_embed': [],
                'cell_type_embed': None,
                'gene_embed': None,
                'labels': []
            }

            for data, labels in loader:
                batch = {sp: (data.to(device), labels.to(device))}
                outputs = model(batch, device, mask=False)

                sp_out = outputs[sp]
                result[sp]['cell_embed'].append(
                    sp_out['cell_embed'].cpu().numpy()
                )
                result[sp]['labels'].append(labels.cpu().numpy())

                if result[sp]['cell_type_embed'] is None:
                    result[sp]['cell_type_embed'] = sp_out['cell_type_embed'].cpu().numpy()
                if result[sp]['gene_embed'] is None:
                    result[sp]['gene_embed'] = sp_out['gene_embed'].cpu().numpy()

            result[sp]['cell_embed'] = np.concatenate(result[sp]['cell_embed'], axis=0)
            result[sp]['labels'] = np.concatenate(result[sp]['labels'], axis=0)

    return result


def extract_embeddings_from_adatas(
    model,
    species_adatas,
    device,
    batch_size: int,
):
    model.eval()
    result = {}

    with torch.no_grad():
        for species, (adata, _label_col) in species_adatas.items():
            cell_embeddings = []
            cell_type_embedding = None
            gene_embedding = None

            for start in range(0, adata.n_obs, batch_size):
                stop = min(start + batch_size, adata.n_obs)
                values = adata.X[start:stop]
                if sp.issparse(values):
                    values = values.toarray()
                values = np.asarray(values, dtype=np.float32)
                batch = {species: (torch.from_numpy(values), None)}
                outputs = model(batch, device, mask=False)
                species_output = outputs[species]
                cell_embeddings.append(
                    species_output["cell_embed"].detach().cpu().numpy()
                )
                if cell_type_embedding is None:
                    cell_type_embedding = (
                        species_output["cell_type_embed"].detach().cpu().numpy()
                    )
                    gene_embedding = (
                        species_output["gene_embed"].detach().cpu().numpy()
                    )

            result[species] = {
                "cell_embed": np.concatenate(cell_embeddings, axis=0),
                "cell_type_embed": cell_type_embedding,
                "gene_embed": gene_embedding,
            }
    return result


def organize_result_from_adatas(
    result: dict,
    species_adatas: dict,
    species_config: dict,
    species_expr_avg: dict,
    global_cell_types: list[str],
    *,
    include_split: bool,
    include_label_perturbation: bool,
):
    species_order, num_celltypes_per_species, all_type_labels = [], [], []
    cell_celltype_embeddings, obs_cell_celltype, index_cell_celltype = [], [], []
    gene_embeddings, obs_genes, index_genes = [], [], []

    cell_counter = celltype_counter = gene_counter = 0

    for species in result:
        species_order.append(species)
        adata, original_label_col = species_adatas[species]
        cell_embeddings = result[species]["cell_embed"]
        if len(cell_embeddings) != adata.n_obs:
            raise ValueError(
                f"[{species}] embedding count {len(cell_embeddings)} does not "
                f"match full-data cell count {adata.n_obs}."
            )
        cell_celltype_embeddings.append(cell_embeddings)

        if include_label_perturbation:
            true_labels = adata.obs[TRUE_LABEL_COLUMN].astype(str).to_numpy()
            train_labels = adata.obs[TRAIN_LABEL_COLUMN].astype(str).to_numpy()
            perturbed = adata.obs[LABEL_PERTURBED_COLUMN].astype(bool).to_numpy()
        else:
            true_labels = adata.obs[original_label_col].astype(str).to_numpy()
            train_labels = None
            perturbed = None

        split_values = (
            adata.obs[SPLIT_COLUMN].astype(str).to_numpy()
            if include_split
            else None
        )
        known_status_values = (
            adata.obs[KNOWN_STATUS_COLUMN].astype(str).to_numpy()
            if include_split
            else None
        )

        for position, cell_id in enumerate(adata.obs_names):
            cell_type_name = str(true_labels[position])
            row = {
                "species": species,
                "cell_type": cell_type_name,
                "item_type": "cell",
                "label_str": f"{species}_{cell_type_name}",
                "cell_id": str(cell_id),
                "Time": None,
            }
            if include_split:
                row[SPLIT_COLUMN] = str(split_values[position])
                row[KNOWN_STATUS_COLUMN] = str(known_status_values[position])
            if include_label_perturbation:
                row[TRUE_LABEL_COLUMN] = cell_type_name
                row[TRAIN_LABEL_COLUMN] = str(train_labels[position])
                row[LABEL_PERTURBED_COLUMN] = bool(perturbed[position])
            obs_cell_celltype.append(row)
            index_cell_celltype.append(f"cell_{cell_counter}")
            cell_counter += 1

        sp_mask = torch.tensor(
            species_expr_avg[species]["cell_type_presence"], dtype=torch.bool
        )
        num_celltypes_per_species.append(int(sp_mask.sum().item()))
        type_embeddings = result[species]["cell_type_embed"][sp_mask]
        cell_celltype_embeddings.append(type_embeddings)

        for index, present in enumerate(sp_mask):
            if not present:
                continue
            cell_type_name = global_cell_types[index]
            label_str = f"{species}_{cell_type_name}"
            all_type_labels.append(label_str)
            row = {
                "species": species,
                "cell_type": cell_type_name,
                "item_type": "celltype",
                "label_str": label_str,
                "cell_id": "",
                "Time": None,
            }
            if include_split:
                row[SPLIT_COLUMN] = "prototype"
                row[KNOWN_STATUS_COLUMN] = "known"
            if include_label_perturbation:
                row[TRUE_LABEL_COLUMN] = cell_type_name
                row[TRAIN_LABEL_COLUMN] = cell_type_name
                row[LABEL_PERTURBED_COLUMN] = False
            obs_cell_celltype.append(row)
            index_cell_celltype.append(f"celltype_{celltype_counter}")
            celltype_counter += 1

        gene_embeddings.append(result[species]["gene_embed"])
        for gene_id in species_config[species]["gene_names"]:
            obs_genes.append(
                {
                    "species": species,
                    "gene_id": gene_id,
                    "label_str": f"{species}_{gene_id}",
                }
            )
            index_genes.append(f"gene_{gene_counter}")
            gene_counter += 1

    adata_cell = anndata.AnnData(
        X=np.concatenate(cell_celltype_embeddings),
        obs=pd.DataFrame(obs_cell_celltype, index=index_cell_celltype),
        dtype=np.float32,
    )
    adata_cell.uns.update(
        {
            "species_order": species_order,
            "global_cell_types": global_cell_types,
            "num_celltypes_per_species": num_celltypes_per_species,
            "all_type_labels": all_type_labels,
        }
    )
    adata_cell.obs["item_type"] = adata_cell.obs["item_type"].astype("category")
    adata_cell.obs["cell_type"] = adata_cell.obs["cell_type"].astype("category")
    adata_cell.obs["species"] = adata_cell.obs["species"].astype("category")
    adata_cell.obs["Time"] = adata_cell.obs["Time"].astype("category")
    if include_split:
        adata_cell.obs[SPLIT_COLUMN] = adata_cell.obs[SPLIT_COLUMN].astype("category")
        adata_cell.obs[KNOWN_STATUS_COLUMN] = adata_cell.obs[
            KNOWN_STATUS_COLUMN
        ].astype("category")
    if include_label_perturbation:
        adata_cell.obs[TRUE_LABEL_COLUMN] = adata_cell.obs[
            TRUE_LABEL_COLUMN
        ].astype("category")
        adata_cell.obs[TRAIN_LABEL_COLUMN] = adata_cell.obs[
            TRAIN_LABEL_COLUMN
        ].astype("category")

    adata_gene = anndata.AnnData(
        X=np.concatenate(gene_embeddings),
        obs=pd.DataFrame(obs_genes, index=index_genes),
        dtype=np.float32,
    )
    adata_gene.uns["species_order"] = species_order
    adata_gene.obs["species"] = adata_gene.obs["species"].astype("category")
    return adata_cell, adata_gene


def organize_result(
    result: dict,
    species_config: dict,
    species_expr_avg: dict,
    global_cell_types: list[str],
    species_time_dict=None
):
    species_order, num_celltypes_per_species, all_type_labels = [], [], []
    cell_celltype_embeddings, obs_cell_celltype, index_cell_celltype = [], [], []
    gene_embeddings, obs_genes, index_genes = [], [], []

    cell_counter = celltype_counter = gene_counter = 0

    for species in result:
        species_order.append(species)

        cell_embeddings = result[species]["cell_embed"]
        cell_labels = result[species]["labels"]
        n_cells = len(cell_embeddings)
        cell_celltype_embeddings.append(cell_embeddings)

        cell_ids = species_config[species].get("cell_ids")
        if cell_ids is None:
            raise KeyError(
                f"species_config['{species}'] is missing 'cell_ids'; "
                "rebuild it with prepare_multispecies_data()."
            )
        if len(cell_ids) != n_cells:
            raise ValueError(
                f"[organize_result] `{species}` cell ID length {len(cell_ids)} "
                f"does not match its cell count {n_cells}"
            )

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
                "cell_id": str(cell_ids[i]),
            }

            if time_series is not None:
                row["Time"] = time_series[i]

            obs_cell_celltype.append(row)
            index_cell_celltype.append(f"cell_{cell_counter}")
            cell_counter += 1

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
                    "cell_id": "",
                    "Time": None
                })
                index_cell_celltype.append(f"celltype_{celltype_counter}")
                celltype_counter += 1

        gene_embeddings.append(result[species]["gene_embed"])
        for gene_id in species_config[species]["gene_names"]:
            obs_genes.append({
                "species": species,
                "gene_id": gene_id,
                "label_str": f"{species}_{gene_id}"
            })
            index_genes.append(f"gene_{gene_counter}")
            gene_counter += 1

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

    adata_gene = anndata.AnnData(
        X=np.concatenate(gene_embeddings),
        obs=pd.DataFrame(obs_genes, index=index_genes),
        dtype=np.float32
    )
    adata_gene.uns["species_order"] = species_order
    adata_gene.obs["species"] = adata_gene.obs["species"].astype("category")

    return adata_cell, adata_gene
