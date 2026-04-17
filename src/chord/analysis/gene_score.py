import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.colors import LinearSegmentedColormap


def compute_gene_grad_importance_all(
    model,
    dataloader,
    species,
    device,
    target_proto: int = 0,
    focus_on_type: bool = True,
):
    """
    Compute gene importance scores aggregated across batches.

    Returns
    -------
    np.ndarray or None
        Gene importance vector with shape (n_genes,).
    """
    scores = []
    counts = []

    for batch in dataloader:
        if species not in batch:
            continue

        cells_raw, labels = batch[species]
        cells_in = cells_raw.clone().detach().to(device)
        cells_in.requires_grad_(True)
        labels = labels.to(device)

        model.zero_grad()
        outputs = model({species: (cells_in, labels)}, device)
        cell_emb = outputs[species]["cell_embed"]
        proto_emb = outputs[species]["cell_type_embed"]

        proto_vec = proto_emb[target_proto].unsqueeze(0)

        if focus_on_type:
            mask = labels == target_proto
            if mask.sum() == 0:
                continue
            sim = F.cosine_similarity(cell_emb[mask], proto_vec, dim=-1).mean()
        else:
            sim = F.cosine_similarity(cell_emb, proto_vec, dim=-1).mean()

        sim.backward()

        grad = cells_in.grad
        if grad is None:
            grad = torch.autograd.grad(sim, cells_in, allow_unused=True)[0]
            if grad is None:
                continue

        if focus_on_type:
            grad_sel = grad[mask]
            x_sel = cells_in.detach()[mask]
        else:
            grad_sel = grad
            x_sel = cells_in.detach()

        imp = (grad_sel * x_sel).abs().sum(0).detach().cpu()
        scores.append(imp)
        counts.append(int(x_sel.shape[0]))

    if not scores:
        return None

    scores = torch.stack(scores)
    counts = torch.tensor(counts, dtype=torch.float32)
    weights = counts / (counts.sum() + 1e-8)

    return (scores.T @ weights).numpy()


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


def compare_gene_importance_cross_species(
    model,
    dataloader,
    species_config,
    homology_indices,
    device="cuda",
    target_proto: int = 0,
    focus_on_type: bool = True,
):
    """
    Compare homologous gene importance across species for importance_z heatmaps.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw importance and z-scored importance:
        [
            gene_human, gene_mouse, gene_monkey,
            imp_h, imp_m, imp_k,
            imp_h_z, imp_m_z, imp_k_z
        ]
    """
    per_species = {}
    for species in ["human", "mouse", "marmoset"]:
        imp = compute_gene_grad_importance_all(
            model=model,
            dataloader=dataloader,
            species=species,
            device=device,
            target_proto=target_proto,
            focus_on_type=focus_on_type,
        )
        per_species[species] = imp

    if any(v is None for v in per_species.values()):
        return pd.DataFrame()

    triplets = collect_homologous_triplets(homology_indices)
    rows = []

    for h_idx, m_idx, k_idx in triplets:
        rows.append(
            {
                "gene_human": species_config["human"]["gene_names"][h_idx],
                "gene_mouse": species_config["mouse"]["gene_names"][m_idx],
                "gene_monkey": species_config["marmoset"]["gene_names"][k_idx],
                "imp_h": float(per_species["human"][h_idx]),
                "imp_m": float(per_species["mouse"][m_idx]),
                "imp_k": float(per_species["marmoset"][k_idx]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    eps = 1e-8
    for sp in ["h", "m", "k"]:
        col = f"imp_{sp}"
        values = df[col].values.astype(float)
        df[f"imp_{sp}_z"] = (values - values.mean()) / (values.std() + eps)

    return df


def plot_importance_heatmap_for_celltype(
    ct2df,
    cell_type,
    top_n=30,
    max_genes=80,
    species_labels=("Human", "Mouse", "Marmoset"),
    figsize=(10, 3),
    gene_label_col="gene_human",
    save_pdf_path=None,
    pdf_dpi=300,
    font_size_base=8,
    figsize_cm=None,
    gene_whitelist=None,
):
    """
    Plot a heatmap of z-scored gene importance for one cell type.

    Parameters
    ----------
    ct2df
        Mapping from cell type to DataFrame returned by compare_gene_importance_cross_species.
    cell_type
        Target cell type.
    top_n
        Number of top positive genes selected per species before union.
    max_genes
        Maximum number of genes shown in the heatmap.
    species_labels
        Labels for the three species.
    figsize
        Figure size in inches.
    gene_label_col
        Column used for gene labels.
    save_pdf_path
        Output PDF path.
    pdf_dpi
        DPI for saving.
    font_size_base
        Base font size.
    figsize_cm
        Figure size in cm. Overrides figsize if provided.
    gene_whitelist
        Optional list of genes to plot directly.
    """
    if cell_type not in ct2df:
        raise ValueError(f"Cell type {cell_type} not found in ct2df")

    df = ct2df[cell_type].copy()
    value_cols = ("imp_h_z", "imp_m_z", "imp_k_z")

    for col in value_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in df")

    if gene_whitelist is not None:
        if gene_label_col not in df.columns:
            raise ValueError(f"gene_label_col '{gene_label_col}' not found in df")
        df_sel = df[df[gene_label_col].isin(gene_whitelist)].copy()
        if df_sel.empty:
            raise ValueError("No genes found matching gene_whitelist")
    else:
        top_idx_sets = []

        for col in value_cols:
            df_pos = df[df[col] > 0]
            if not df_pos.empty:
                top_idx = df_pos[col].sort_values(ascending=False).head(top_n).index
            else:
                top_idx = df[col].sort_values(ascending=False).head(top_n).index
            top_idx_sets.append(set(top_idx))

        union_idx = set().union(*top_idx_sets)
        if not union_idx:
            raise ValueError("No genes selected under current settings")

        if len(union_idx) > max_genes:
            tmp = df.loc[list(union_idx), list(value_cols)].copy()
            tmp["max_z"] = tmp[list(value_cols)].max(axis=1)
            keep_idx = tmp["max_z"].sort_values(ascending=False).head(max_genes).index
        else:
            keep_idx = list(union_idx)

        df_sel = df.loc[keep_idx].copy()

    if df_sel.empty:
        raise ValueError("No genes available for plotting")

    if gene_label_col in df_sel.columns:
        gene_labels = df_sel[gene_label_col].astype(str).values
    else:
        gene_labels = df_sel.index.astype(str).values

    heat_df = df_sel[list(value_cols)].copy()
    heat_df.columns = list(species_labels)
    heat_df.index = gene_labels

    if heat_df.shape[0] > 2:
        Z = linkage(heat_df.values, method="average", metric="euclidean")
        order = leaves_list(Z)
        heat_df = heat_df.iloc[order, :]

    if figsize_cm is not None:
        figsize_inch = (figsize_cm[0] / 2.54, figsize_cm[1] / 2.54)
    else:
        figsize_inch = figsize

    fig, ax = plt.subplots(figsize=figsize_inch)

    white_red = LinearSegmentedColormap.from_list(
        "white_red", ["#ffffff", "#d73027"]
    )

    hm = sns.heatmap(
        heat_df,
        cmap=white_red,
        linewidths=0.1,
        vmin=0,
        linecolor="white",
        cbar_kws={"label": "Z-scored importance"},
        ax=ax,
        xticklabels=heat_df.columns,
        yticklabels=heat_df.index,
    )

    title_fs = font_size_base + 2
    axis_label_fs = font_size_base + 1
    tick_fs = font_size_base

    ax.set_title(f"{cell_type}", fontsize=title_fs)
    ax.tick_params(axis="x", labelsize=tick_fs, rotation=45)
    ax.tick_params(axis="y", labelsize=tick_fs, rotation=0)

    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.set_label("Z-scored importance", fontsize=axis_label_fs)

    plt.tight_layout()

    if save_pdf_path is not None:
        fig.savefig(
            save_pdf_path,
            format="pdf",
            dpi=pdf_dpi,
            bbox_inches="tight",
        )

    plt.show()

    return heat_df