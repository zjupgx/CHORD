import requests
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
from sklearn.metrics.pairwise import cosine_similarity
import time
from scipy.stats import hypergeom
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests


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

def collect_homologous_triplets(homology_indices):
    hm = homology_indices[("human", "mouse")]
    triplets = []
    h_to_m = dict(zip(hm[0], hm[1]))
    for h_idx in h_to_m:
        triplets.append((h_idx, h_to_m[h_idx]))
    return triplets

def sanitize_symbol(x):
    return str(x).strip().upper()

def build_ortholog_map(species_config, homology_indices):
    pairs = collect_homologous_triplets(homology_indices)
    human_names = species_config["human"]["gene_names"]
    mouse_names = species_config["mouse"]["gene_names"]
    records = []
    for h_idx, m_idx in pairs:
        gh = str(human_names[h_idx])
        gm = str(mouse_names[m_idx])
        records.append({
            "human_idx": int(h_idx),
            "mouse_idx": int(m_idx),
            "gene_human": gh,
            "gene_mouse": gm,
            "gene_human_norm": sanitize_symbol(gh),
            "gene_mouse_norm": sanitize_symbol(gm),
        })
    map_df = pd.DataFrame(records).drop_duplicates(subset=["gene_human_norm", "gene_mouse_norm"], keep="first")
    return map_df

def fisher_meta(pvals):
    pvals = np.asarray(pvals, dtype=float)
    eps = 1e-300
    pvals = np.clip(pvals, eps, 1 - eps)
    X = -2.0 * np.sum(np.log(pvals))
    df = 2 * pvals.size
    p_meta = 1.0 - chi2.cdf(X, df)
    return X, df, p_meta

def preprocess_species_df(df):
    df = df.copy()
    expected_cols = ["gene_id", "overlap_count", "pval", "overlap_genes", "pval_adj"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in species df: {missing}")
    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    df["gene_id_norm"] = df["gene_id"].apply(sanitize_symbol)
    df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
    df["overlap_count"] = pd.to_numeric(df["overlap_count"], errors="coerce")
    df["pval_adj"] = pd.to_numeric(df["pval_adj"], errors="coerce")
    df = df.sort_values("pval", ascending=True).drop_duplicates(subset=["gene_id_norm"], keep="first")
    return df

def meta_on_ortholog_intersection(human_df, mouse_df, species_config, homology_indices):
    ortho_map = build_ortholog_map(species_config, homology_indices)
    human_df = preprocess_species_df(human_df)
    mouse_df = preprocess_species_df(mouse_df)

    merged = (
        ortho_map
        .merge(
            human_df[["gene_id_norm", "gene_id", "pval", "overlap_count", "pval_adj"]],
            left_on="gene_human_norm", right_on="gene_id_norm", how="left"
        )
        .merge(
            mouse_df[["gene_id_norm", "gene_id", "pval", "overlap_count", "pval_adj"]],
            left_on="gene_mouse_norm", right_on="gene_id_norm", how="left",
            suffixes=("_human", "_mouse")
        )
    )

    merged = merged.rename(columns={
        "gene_id_human": "gene_id_human_df",
        "gene_id_mouse": "gene_id_mouse_df",
        "pval_human": "p_human",
        "pval_mouse": "p_mouse",
        "overlap_count_human": "overlap_human",
        "overlap_count_mouse": "overlap_mouse",
        "pval_adj_human": "q_human",
        "pval_adj_mouse": "q_mouse",
    })

    merged["p_human"] = pd.to_numeric(merged["p_human"], errors="coerce")
    merged["p_mouse"] = pd.to_numeric(merged["p_mouse"], errors="coerce")

    valid = merged["p_human"].notna() & merged["p_mouse"].notna()

    merged["fisher_X"] = np.nan
    merged["fisher_df"] = np.nan
    merged["p_meta"] = np.nan

    def fisher_row(r):
        X, df_, pmeta = fisher_meta([r["p_human"], r["p_mouse"]])
        return pd.Series({"fisher_X": X, "fisher_df": df_, "p_meta": pmeta})

    res = merged.loc[valid, ["p_human", "p_mouse"]].apply(fisher_row, axis=1)
    merged.loc[valid, ["fisher_X", "fisher_df", "p_meta"]] = res

    valid_meta = merged["p_meta"].notna()
    merged["q_meta"] = np.nan
    if valid_meta.any():
        merged.loc[valid_meta, "q_meta"] = multipletests(merged.loc[valid_meta, "p_meta"], method="fdr_bh")[1]

    n_total = len(merged)
    n_human_match = int(merged["p_human"].notna().sum())
    n_mouse_match = int(merged["p_mouse"].notna().sum())
    n_meta_pairs = int(valid.sum())
    print(f"Ortholog pairs total: {n_total}")
    print(f"Matched with human df: {n_human_match} / {n_total}")
    print(f"Matched with mouse df: {n_mouse_match} / {n_total}")
    print(f"Meta-tested pairs (both species have p): {n_meta_pairs} / {n_total}")

    out_cols = [
        "human_idx", "mouse_idx",
        "gene_human", "gene_mouse", "gene_human_norm", "gene_mouse_norm",
        "gene_id_human_df", "gene_id_mouse_df",
        "p_human", "p_mouse",
        "overlap_human", "overlap_mouse",
        "q_human", "q_mouse",
        "fisher_X", "fisher_df", "p_meta", "q_meta"
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    return merged[out_cols]