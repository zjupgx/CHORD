import torch
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp 
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData
from typing import Dict, Tuple


def data_preprocess(file_path, n_top_genes=2000, cpm=1e4, is_preprocessed=False):
    adata = sc.read_h5ad(file_path)
    if not is_preprocessed:
        # Basic quality control on cells and genes
        sc.pp.filter_cells(adata, min_counts=200)
        sc.pp.filter_genes(adata, min_cells=10)
        
        # Library-size normalization to a fixed total count per cell
        sc.pp.normalize_total(adata, target_sum=cpm)
        
        # Log-transform normalized expression values
        sc.pp.log1p(adata)

    # Select highly variable genes using the Seurat flavor
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()
    print("HVG selection has been completed")

    return adata


def calculate_average_expression(data, cell_labels, global_cell_types):
    """
    Compute the mean expression matrix for each globally defined cell type.

    The row order of the output matrix follows `global_cell_types`.
    Here, `cell_labels` is expected to contain integer-encoded labels
    whose values correspond to the indices of `global_cell_types`
    produced by the shared label encoder.

    If a given global cell type is absent in the current dataset,
    the corresponding row remains all zeros.

    Args:
        data (np.ndarray):
            Expression matrix of shape (n_cells, n_genes).
        cell_labels (list or np.ndarray):
            Integer-encoded cell-type labels of shape (n_cells,).
        global_cell_types (list):
            Global reference list of cell types. Its order defines
            the row order of the output matrix.

    Returns:
        avg_matrix (np.ndarray):
            Mean expression matrix of shape (n_global_types, n_genes),
            aligned to `global_cell_types`.
        global_cell_types (list):
            The same global cell-type list passed to the function.
    """
    # Check that the number of labels matches the number of cells
    assert data.shape[0] == len(cell_labels), "Data and label lengths do not match"
    
    # Initialize an all-zero matrix:
    # rows = global cell types, columns = genes
    avg_matrix = np.zeros((len(global_cell_types), data.shape[1]))
    
    # Fill each row according to the global encoded label index
    for i, ct in enumerate(global_cell_types):
        cell_mask = cell_labels == i
        if np.any(cell_mask):  # Compute the mean only if this type exists
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
    """
    Preprocess multi-species datasets with species-specific cell-type columns.

    This function:
    1. Builds or reuses a shared label encoder across all species.
    2. Converts expression matrices to dense tensors if needed.
    3. Encodes cell-type labels into a shared global label space.
    4. Computes per-cell-type average expression matrices.
    5. Records which global cell types are present in each species.

    Args:
        species_adatas:
            Dictionary mapping species name to a tuple of
            (AnnData object, cell-type column name).
        label_encoder:
            Optional pre-built global LabelEncoder.

    Returns:
        species_data:
            Dictionary mapping species name to
            (expression tensor, encoded label tensor).
        species_config:
            Dictionary containing species-level metadata.
        species_expr_avg:
            Dictionary containing average-expression matrices and
            cell-type presence vectors.
        global_cell_types:
            List of globally unified cell-type names.
    """

    # Build a shared label encoder from the species-specific cell-type columns
    if label_encoder is None:
        label_encoder = LabelEncoder()
        all_labels = []
        for adata, ct_col in species_adatas.values():
            all_labels.extend(adata.obs[ct_col].tolist())
        label_encoder.fit(all_labels)
    
    # Global union of cell types defined by the shared encoder
    global_cell_types = label_encoder.classes_.tolist()
        
    # Build outputs for each species
    species_data = {}
    species_config = {}
    species_expr_avg = {}
    for name, (adata, ct_col) in species_adatas.items():
        # Read species-specific cell-type labels
        labels = adata.obs[ct_col].tolist()
        labels_encoded = label_encoder.transform(labels)

        # Convert expression matrix to dense array if needed
        if sp.issparse(adata.X):
            expr = adata.X.toarray()
        else:
            expr = adata.X
        expr_tensor = torch.FloatTensor(expr)
        label_tensor = torch.LongTensor(labels_encoded)
    
        # Compute the average expression matrix in the global label space
        avg_expr_matrix, cell_types = calculate_average_expression(expr, labels_encoded, global_cell_types)

        # Build a binary presence vector indicating whether each global
        # cell type exists in the current species
        current_cell_types = set(labels)
        cell_type_presence = [1 if ct in current_cell_types else 0 for ct in global_cell_types]
        
        # Store processed tensors and metadata
        species_data[name] = (expr_tensor, label_tensor)
        species_config[name] = {
            "num_genes": len(adata.var_names),
            "gene_names": adata.var_names,
            "cell_type_column": ct_col,
            "cell_type_presence": cell_type_presence
        }
        species_expr_avg[name] = {
            "expr_avg": torch.FloatTensor(avg_expr_matrix),
            "cell_type_presence": cell_type_presence
        }
    
    return species_data, species_config, species_expr_avg, global_cell_types


def find_gene_matches(sp1_genes, sp2_genes, sp1_homo_genes, sp2_homo_genes):
    """
    Find valid cross-species gene matches based on a homology table.

    A match is considered valid if:
    1. A gene from species 1 appears in the homology gene list.
    2. Its mapped homolog gene from species 2 is also present in the
       species-2 gene list used by the current dataset/configuration.

    Args:
        sp1_genes:
            Gene list for species 1 in the current dataset/configuration.
        sp2_genes:
            Gene list for species 2 in the current dataset/configuration.
        sp1_homo_genes:
            Species-1 gene column from the homology table.
        sp2_homo_genes:
            Species-2 gene column from the homology table.

    Returns:
        valid_matches:
            List of dictionaries describing valid matched gene pairs.
        sp1_indices:
            Indices of matched genes in `sp1_genes`.
        sp2_indices:
            Indices of matched genes in `sp2_genes`.
    """    
    # Store all valid matched gene pairs
    valid_matches = []
    
    # Iterate over genes from species 1 in the current dataset
    for sp1_idx, sp1_gene in enumerate(sp1_genes):
        # Check whether the species-1 gene appears in the homology table
        if sp1_gene in sp1_homo_genes:
            # Find all occurrences of this gene in the homology table
            sp1_homo_indices = [i for i, gene in enumerate(sp1_homo_genes) if gene == sp1_gene]
            sp1_indices = [sp1_idx for _, gene in enumerate(sp1_homo_genes) if gene == sp1_gene]

            # Retrieve the corresponding species-2 homolog gene(s)
            for i in sp1_homo_indices:
                sp2_gene = sp2_homo_genes[i]
                
                # Keep the match only if the species-2 gene also exists
                # in the current dataset/configuration
                if sp2_gene in sp2_genes:
                    # Get the index of the matched species-2 gene
                    sp2_idx = sp2_genes.index(sp2_gene)
                    
                    # Record a valid gene match
                    valid_matches.append({
                        "sp1_index": sp1_idx,
                        "sp1_gene": sp1_gene,
                        "sp2_index": sp2_idx,
                        "sp2_gene": sp2_gene
                    })
    
    # Extract matched index lists for downstream use
    sp1_indices = [match['sp1_index'] for match in valid_matches]
    sp2_indices = [match['sp2_index'] for match in valid_matches]
    return valid_matches, sp1_indices, sp2_indices


def get_homology_indices(species_config, homology_files):
    homology_indices = {}
    for (species1, species2), file_path in homology_files.items():
        homo_df = pd.read_csv(file_path)
        sp1_homo_genes = list(homo_df[f"{species1}_gene"])
        sp2_homo_genes = list(homo_df[f"{species2}_gene"])
        
        # Read the gene lists stored in the species configuration
        sp1_genes = list(species_config[species1]["gene_names"])
        sp2_genes = list(species_config[species2]["gene_names"])

        # Find valid gene matches between the two species
        valid_matches, sp1_indices, sp2_indices = find_gene_matches(
            sp1_genes, sp2_genes, sp1_homo_genes, sp2_homo_genes
        )
        print(f"{species1}, {species2}: found {len(valid_matches)} valid homologous gene matches")
        homology_indices[(species1, species2)] = (sp1_indices, sp2_indices)
    return homology_indices