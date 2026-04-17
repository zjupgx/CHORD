import requests
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
from sklearn.metrics.pairwise import cosine_similarity
import time
from scipy.stats import hypergeom


def get_full_ppi(gene, species=9606, score_threshold=400):
    """Fetch the full STRING PPI network for one gene."""
    string_api_url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": gene,
        "species": species,
        "required_score": score_threshold
    }
    response = requests.get(string_api_url, params=params)
    df = pd.read_csv(io.StringIO(response.text), sep="\t")
    return df[['preferredName_A', 'preferredName_B', 'score']]


def visualize_ppi_with_embedding_filtered(adata_gene, target_gene, species="human", top_n=30, score_threshold=400):
    """
    Overlay embedding top-N neighbors on the real PPI network.

    Red: target gene
    Gold: overlap between embedding top-N and PPI
    Gray: PPI-only genes not present in the embedding space
    """
    
    # 1. Fetch PPI
    species_id = 9606 if species == "human" else 10090
    df_ppi = get_full_ppi(target_gene, species=species_id, score_threshold=score_threshold)
    
    # 2. Get genes in the embedding space
    embedding_genes = set(adata_gene[adata_gene.obs['species'] == species].obs['gene_id'])
    
    # 3. Get top-N embedding neighbors
    species_df = adata_gene[adata_gene.obs['species'] == species]
    gene_ids = species_df.obs['gene_id'].tolist()
    gene_matrix = species_df.X
    target_vec = species_df[species_df.obs['gene_id'] == target_gene].X.reshape(1, -1)
    
    sim = cosine_similarity(target_vec, gene_matrix).flatten()
    top_idx = sim.argsort()[::-1][1:top_n + 1]
    top_genes = set([gene_ids[i] for i in top_idx])
    
    # 4. Define node groups
    all_ppi_genes = set(df_ppi['preferredName_A']).union(df_ppi['preferredName_B'])
    all_ppi_genes.discard(target_gene)
    
    overlap_genes = top_genes & all_ppi_genes
    gray_genes = all_ppi_genes - embedding_genes
    
    # 5. Filter PPI edges
    df_ppi_filtered = df_ppi[
        df_ppi['preferredName_A'].isin(all_ppi_genes.union({target_gene})) &
        df_ppi['preferredName_B'].isin(all_ppi_genes.union({target_gene}))
    ]
    
    # 6. Build graph
    G = nx.from_pandas_edgelist(df_ppi_filtered, 'preferredName_A', 'preferredName_B', edge_attr='score')
    
    # 7. Set node colors
    color_map = []
    for node in G.nodes():
        if node == target_gene:
            color_map.append('#EA5941')
        elif node in overlap_genes:
            color_map.append('#FFD300')
        elif node in gray_genes:
            color_map.append('#BDD7E7')
        else:
            color_map.append('#79BB4C')
    
    # 8. Plot
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.family'] = 'Arial'
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=700, font_size=10, edge_color='gray', width=0.5)
    plt.title(f"{species.capitalize()} {target_gene} PPI Network")
    plt.savefig(f"{species.capitalize()} {target_gene} PPI Network.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    
    return G


def get_ppi_neighbors(protein, species, score_threshold=700, delay=1.0):
    """
    Fetch STRING PPI neighbors for one gene with a delay.
    """
    string_api_url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": protein,
        "species": species,
        "required_score": score_threshold,
    }
    try:
        response = requests.get(string_api_url, params=params)
        df = pd.read_csv(io.StringIO(response.text), sep="\t")
        neighbors = set(df['preferredName_A']).union(df['preferredName_B'])
        neighbors.discard(protein)
    except Exception as e:
        print(f"Warning: Failed to get PPI for {protein}: {e}")
        neighbors = set()
    time.sleep(delay)
    return neighbors


def build_ppi_dict(genes, species="human", score_threshold=700, delay=0.5):
    """
    Build a dictionary: gene -> PPI neighbor set.
    """
    species_map = {"human": 9606, "mouse": 10090}
    tax_id = species_map[species]
    ppi_dict = {}

    for i, gene in enumerate(genes):
        print(f"[{i+1}/{len(genes)}] Fetching PPI for {gene}...")
        neighbors = get_ppi_neighbors(gene, tax_id, score_threshold, delay)
        if neighbors:
            ppi_dict[gene] = neighbors
    return ppi_dict


def get_embedding_neighbors(adata_gene, target_gene, species="human", top_n=30):
    try:
        v_target = adata_gene[(adata_gene.obs["gene_id"] == target_gene) & (adata_gene.obs["species"] == species)].X.reshape(-1)
        V_species = adata_gene[(adata_gene.obs["species"] == species)].X
        gene_ids = adata_gene[(adata_gene.obs["species"] == species)].obs["gene_id"].tolist()

        sim = cosine_similarity(v_target.reshape(1, -1), V_species).flatten()
        top_idx = sim.argsort()[::-1][1:top_n + 1]
        top_genes = [gene_ids[i] for i in top_idx]
        return set(top_genes)
    except Exception:
        return set()


def compute_overlap_pval(gene, embedding_neighbors, ppi_neighbors, background_genes):
    """
    Hypergeometric test for overlap significance.

    M = number of background genes
    n = number of PPI neighbors in the background
    N = number of embedding neighbors
    k = overlap count
    """
    ppi_neighbors = set(ppi_neighbors) & set(background_genes)
    embedding_neighbors = set(embedding_neighbors) & set(background_genes)

    M = len(background_genes)
    n = len(ppi_neighbors)
    N = len(embedding_neighbors)
    k = len(ppi_neighbors & embedding_neighbors)

    if n == 0 or N == 0 or k == 0:
        return k, 1.0, []

    rv = hypergeom(M, n, N)
    pval = rv.sf(k - 1)

    return k, pval, list(ppi_neighbors & embedding_neighbors)


def batch_compute(adata_gene, ppi_dict, species="human", top_n=30):
    """
    Compute overlap statistics for all genes.
    """
    results = []
    background_genes = adata_gene[adata_gene.obs["species"] == species].obs["gene_id"].unique()

    for i, gene in enumerate(background_genes):
        if gene not in ppi_dict:
            continue
        emb_neighbors = get_embedding_neighbors(adata_gene, gene, species, top_n)
        k, pval, overlap_genes = compute_overlap_pval(gene, emb_neighbors, ppi_dict[gene], background_genes)
        results.append({
            "gene_id": gene,
            "overlap_count": k,
            "pval": pval,
            "overlap_genes": ",".join(overlap_genes)
        })
        if i % 100 == 0:
            print(f"[{i}/{len(background_genes)}] {gene} done.")
    return pd.DataFrame(results)


def query_gene_neighbors(adata_gene, gene_id, species, top_n=10):
    """
    Query the top-N nearest genes in the embedding space.

    Parameters
    ----------
    adata_gene : AnnData
        AnnData object with gene embeddings
    gene_id : str
        Target gene ID
    species : str
        Target species
    top_n : int, optional
        Number of neighbors to return (default=10)

    Returns
    -------
    list
        Top-N neighbor gene IDs, excluding the target gene itself
    """
    v_target = adata_gene[
        (adata_gene.obs["gene_id"] == gene_id) & (adata_gene.obs["species"] == species)
    ].X.reshape(1, -1)

    adata_species = adata_gene[adata_gene.obs["species"] == species]
    V_species = adata_species.X

    sim = cosine_similarity(v_target, V_species).flatten()

    top_idx = sim.argsort()[::-1]
    top_idx = top_idx[top_idx != np.argmax(sim)]
    top_idx = top_idx[:top_n]

    top_genes = adata_species.obs["gene_id"].iloc[top_idx].tolist()
    print(f"Top {top_n} neighbors of {gene_id} in {species} embedding:", top_genes)
    return top_genes


def human_mouse_ppi(df_results_human, df_results_mouse, homology_indices, adata_gene,
                    human_species="human", mouse_species="mouse"):
    """
    Align human and mouse PPI overlap results based on homolog mapping.

    Inputs
    ------
    df_results_human / df_results_mouse
        DataFrames containing gene_id, overlap_count, pval, overlap_genes, pval_adj
    homology_indices
        dict[(human, mouse)] -> tuple(human_idx, mouse_idx)
        Indices are relative to the species-specific subset
    adata_gene
        AnnData object with gene_id and species in .obs
    human_species / mouse_species
        Species names in adata_gene.obs["species"]
    """
    # 1. Build species-specific gene tables
    human_gene_df = adata_gene[adata_gene.obs['species'] == human_species].copy()
    mouse_gene_df = adata_gene[adata_gene.obs['species'] == mouse_species].copy()
    
    # Use continuous local indices to match homology_indices
    human_idx_to_gene = pd.Series(human_gene_df.obs['gene_id'].values, index=range(len(human_gene_df))).to_dict()
    mouse_idx_to_gene = pd.Series(mouse_gene_df.obs['gene_id'].values, index=range(len(mouse_gene_df))).to_dict()
    
    # 2. Build human -> mouse homolog mapping
    hm = homology_indices[(human_species, mouse_species)]
    h_to_m = {human_idx_to_gene[h]: mouse_idx_to_gene[m] for h, m in zip(hm[0], hm[1])}

    # 3. Align overlap results
    records = []
    for h_gene, m_gene in h_to_m.items():
        h_row = df_results_human[df_results_human['gene_id'] == h_gene]
        m_row = df_results_mouse[df_results_mouse['gene_id'] == m_gene]
        if not h_row.empty and not m_row.empty:
            records.append({
                "human_gene": h_gene,
                "mouse_gene": m_gene,
                "human_overlap": h_row['overlap_count'].values[0],
                "mouse_overlap": m_row['overlap_count'].values[0],
                "human_pval_adj": h_row['pval_adj'].values[0],
                "mouse_pval_adj": m_row['pval_adj'].values[0]
            })
    df_compare = pd.DataFrame(records)
  
    return df_compare