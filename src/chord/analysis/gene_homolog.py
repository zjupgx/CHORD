import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from ..utils.plot import pretty_boxplot


def collect_homologous_triplets(homology_indices):
    """
    Collect shared human-mouse-marmoset homologous triplets.
    """
    hm = homology_indices[("human", "mouse")]
    hk = homology_indices[("human", "marmoset")]

    h_to_m = dict(zip(hm[0], hm[1]))
    h_to_k = dict(zip(hk[0], hk[1]))

    triplets = []
    for h_idx in h_to_m:
        if h_idx in h_to_k:
            triplets.append((h_idx, h_to_m[h_idx], h_to_k[h_idx]))

    return triplets


def cosine_similarity_paired(X1, X2, eps=1e-12):
    """
    Compute row-wise cosine similarity for paired vectors.
    """
    numerator = (X1 * X2).sum(axis=1)
    denominator = np.linalg.norm(X1, axis=1) * np.linalg.norm(X2, axis=1) + eps
    return numerator / denominator


def split_embedding_by_species(adata_gene, species_key="species"):
    """
    Split the gene embedding matrix by species.
    """
    X = adata_gene.X
    species_arr = adata_gene.obs[species_key].values
    species_to_global = {
        sp: np.where(species_arr == sp)[0]
        for sp in np.unique(species_arr)
    }
    return X, species_arr, species_to_global


def evaluate_homolog_embeddings(
    adata_gene,
    homology_indices,
    *,
    species_key: str = "species",
    random_seed: int = 42,
    make_plot: bool = False,
):
    """
    Compare homologous gene-pair similarities against random pairs.

    For each species pair in `homology_indices`, this function computes
    cosine similarities for homologous gene pairs and matched random
    non-homologous pairs, then applies a one-sided Mann-Whitney U test.

    Parameters
    ----------
    adata_gene
        AnnData object with gene embeddings in `.X`.
    homology_indices
        Dictionary mapping (species1, species2) to homologous indices.
        Each value can be either:
        - a tuple of two 1D arrays
        - a list of index tuples
    species_key
        Column name in `.obs` containing species labels.
    random_seed
        Random seed for random pair generation.
    make_plot
        Whether to generate boxplots.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per species pair.
    """
    rng = np.random.default_rng(random_seed)
    emb, _, species_to_global = split_embedding_by_species(
        adata_gene, species_key=species_key
    )

    rows = []
    axes = None

    if make_plot:
        nplot = len(homology_indices)
        fig, axes = plt.subplots(1, nplot, figsize=(3 * nplot, 4))
        if nplot == 1:
            axes = [axes]

    for i, ((sp1, sp2), pair_idx) in enumerate(homology_indices.items()):
        if len(pair_idx) == 2 and np.ndim(pair_idx[0]) == 1:
            loc1, loc2 = map(np.asarray, pair_idx)
        else:
            loc1, loc2 = np.asarray(pair_idx).T

        idx1_global = species_to_global[sp1][loc1]
        idx2_global = species_to_global[sp2][loc2]

        X1 = emb[idx1_global]
        X2 = emb[idx2_global]
        cos_hom = cosine_similarity_paired(X1, X2)

        n_rand = len(cos_hom)
        homolog_set = set(zip(idx1_global, idx2_global))

        rand_set = set()
        while len(rand_set) < n_rand:
            a = rng.choice(species_to_global[sp1])
            b = rng.choice(species_to_global[sp2])
            if (a, b) in homolog_set:
                continue
            rand_set.add((a, b))

        rand_pairs = np.array(list(rand_set))
        X1_rand = emb[rand_pairs[:, 0]]
        X2_rand = emb[rand_pairs[:, 1]]
        cos_rand = cosine_similarity_paired(X1_rand, X2_rand)

        U, p = mannwhitneyu(cos_hom, cos_rand, alternative="greater")

        rows.append(
            {
                "species1": sp1,
                "species2": sp2,
                "n_homolog": len(cos_hom),
                "median_homolog": np.median(cos_hom),
                "median_random": np.median(cos_rand),
                "U": U,
                "p": p,
            }
        )

        if make_plot:
            ax = axes[i]
            pretty_boxplot(
                ax,
                colors=["#F3756D", "#1FBDC3"],
                data=[cos_hom, cos_rand],
                labels=["Ortholog", "Random"],
                title=(
                    f"Gene embedding similarity\n"
                    f"between {sp1.capitalize()} and {sp2.capitalize()}"
                ),
                pval=p,
            )

    if make_plot:
        plt.tight_layout()
        fig.savefig("box.pdf")

    return pd.DataFrame(rows)


def compute_triplet_similarity_table(
    adata_gene,
    homology_indices,
    *,
    species_key: str = "species",
    species_order=("human", "mouse", "marmoset"),
):
    """
    Compute pairwise cosine similarities for homologous triplets.
    """
    sp_h, sp_m, sp_k = species_order

    X, species_arr, _ = split_embedding_by_species(adata_gene, species_key=species_key)

    idx_h = np.where(species_arr == sp_h)[0]
    idx_m = np.where(species_arr == sp_m)[0]
    idx_k = np.where(species_arr == sp_k)[0]

    X_h = X[idx_h]
    X_m = X[idx_m]
    X_k = X[idx_k]

    triplets = collect_homologous_triplets(homology_indices)

    rows = []
    for h_i, m_i, k_i in triplets:
        v_h = X_h[h_i]
        v_m = X_m[m_i]
        v_k = X_k[k_i]

        sim_hm = np.dot(v_h, v_m) / (np.linalg.norm(v_h) * np.linalg.norm(v_m) + 1e-12)
        sim_hk = np.dot(v_h, v_k) / (np.linalg.norm(v_h) * np.linalg.norm(v_k) + 1e-12)
        sim_mk = np.dot(v_m, v_k) / (np.linalg.norm(v_m) * np.linalg.norm(v_k) + 1e-12)

        rows.append(
            {
                f"{sp_h}_{sp_m}_sim": sim_hm,
                f"{sp_h}_{sp_k}_sim": sim_hk,
                f"{sp_m}_{sp_k}_sim": sim_mk,
            }
        )

    return pd.DataFrame(rows)