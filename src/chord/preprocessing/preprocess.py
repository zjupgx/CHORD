import torch
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp 
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData
from typing import Dict, Tuple


def data_preprocess(
    file_path,
    n_top_genes=2000,
    cpm=1e4,
    is_preprocessed=False,
    select_hvg=True,
):
    adata = sc.read_h5ad(file_path)
    if not is_preprocessed:
        sc.pp.filter_cells(adata, min_counts=200)
        sc.pp.filter_genes(adata, min_cells=10)
        
        sc.pp.normalize_total(adata, target_sum=cpm)
        
        sc.pp.log1p(adata)

    if select_hvg:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable].copy()
        print("HVG selection has been completed")

    return adata


def calculate_average_expression(data, cell_labels, global_cell_types):
    assert data.shape[0] == len(cell_labels), "Data and label lengths do not match"
    
    avg_matrix = np.zeros((len(global_cell_types), data.shape[1]))
    
    for i, ct in enumerate(global_cell_types):
        cell_mask = cell_labels == i
        if np.any(cell_mask):
            avg_matrix[i, :] = np.mean(data[cell_mask, :], axis=0)
    return avg_matrix, global_cell_types


def build_global_label_encoder(
        species_adatas: Dict[str, Tuple[AnnData, str]]
    ) -> LabelEncoder:
    le = LabelEncoder()
    all_labels = []
    for adata, ct_col in species_adatas.values():
        all_labels.extend(adata.obs[ct_col].tolist())
    le.fit(all_labels)
    print(f"[Global] Found {len(le.classes_)} unique cell types")
    return le


def prepare_multispecies_data(species_adatas, label_encoder=None):

    if label_encoder is None:
        label_encoder = LabelEncoder()
        all_labels = []
        for adata, ct_col in species_adatas.values():
            all_labels.extend(adata.obs[ct_col].tolist())
        label_encoder.fit(all_labels)
    
    global_cell_types = label_encoder.classes_.tolist()
        
    species_data = {}
    species_config = {}
    species_expr_avg = {}
    for name, (adata, ct_col) in species_adatas.items():
        labels = adata.obs[ct_col].tolist()
        labels_encoded = label_encoder.transform(labels)

        if sp.issparse(adata.X):
            expr = adata.X.toarray()
        else:
            expr = adata.X
        expr_tensor = torch.FloatTensor(expr)
        label_tensor = torch.LongTensor(labels_encoded)
    
        avg_expr_matrix, cell_types = calculate_average_expression(expr, labels_encoded, global_cell_types)

        current_cell_types = set(labels)
        cell_type_presence = [1 if ct in current_cell_types else 0 for ct in global_cell_types]
        
        species_data[name] = (expr_tensor, label_tensor)
        species_config[name] = {
            "num_genes": len(adata.var_names),
            "gene_names": adata.var_names,
            "cell_ids": [str(cell_id) for cell_id in adata.obs_names],
            "cell_type_column": ct_col,
            "cell_type_presence": cell_type_presence
        }
        species_expr_avg[name] = {
            "expr_avg": torch.FloatTensor(avg_expr_matrix),
            "cell_type_presence": cell_type_presence
        }
    
    return species_data, species_config, species_expr_avg, global_cell_types


def find_gene_matches(sp1_genes, sp2_genes, sp1_homo_genes, sp2_homo_genes):
    valid_matches = []
    
    for sp1_idx, sp1_gene in enumerate(sp1_genes):
        if sp1_gene in sp1_homo_genes:
            sp1_homo_indices = [i for i, gene in enumerate(sp1_homo_genes) if gene == sp1_gene]

            for i in sp1_homo_indices:
                sp2_gene = sp2_homo_genes[i]
                
                if sp2_gene in sp2_genes:
                    sp2_idx = sp2_genes.index(sp2_gene)
                    
                    valid_matches.append({
                        "sp1_index": sp1_idx,
                        "sp1_gene": sp1_gene,
                        "sp2_index": sp2_idx,
                        "sp2_gene": sp2_gene
                    })
    
    sp1_indices = [match['sp1_index'] for match in valid_matches]
    sp2_indices = [match['sp2_index'] for match in valid_matches]
    return valid_matches, sp1_indices, sp2_indices


def get_homology_indices(species_config, homology_files):
    homology_indices = {}
    for (species1, species2), file_path in homology_files.items():
        homo_df = pd.read_csv(file_path)
        sp1_homo_genes = list(homo_df[f"{species1}_gene"])
        sp2_homo_genes = list(homo_df[f"{species2}_gene"])
        
        sp1_genes = list(species_config[species1]["gene_names"])
        sp2_genes = list(species_config[species2]["gene_names"])

        valid_matches, sp1_indices, sp2_indices = find_gene_matches(
            sp1_genes, sp2_genes, sp1_homo_genes, sp2_homo_genes
        )
        print(f"{species1}, {species2}: found {len(valid_matches)} valid homologous gene matches")
        homology_indices[(species1, species2)] = (sp1_indices, sp2_indices)
    return homology_indices
