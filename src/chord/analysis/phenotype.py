import re
from typing import Union

import anndata
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests


def plot_developmental_alignment(
    adata: anndata.AnnData,
    species1: str,
    species2: str,
    time_col: str = "Time",
    species_col: str = "species",
    embedding_key: str = "X_umap",
    figsize: tuple = (5, 5),
    modified_align=None,
):
    print("--- 1. Prepare data and parse time points ---")

    def parse_time(time_str):
        match = re.search(r"\d+", str(time_str))
        return int(match.group()) if match else -1

    s1_times_unique = adata.obs.loc[adata.obs[species_col] == species1, time_col].unique()
    s1_times_sorted = sorted(s1_times_unique, key=parse_time)

    s2_times_unique = adata.obs.loc[adata.obs[species_col] == species2, time_col].unique()
    s2_times_sorted = sorted(s2_times_unique, key=parse_time)

    print(f"Species 1 ({species1}) time points: {s1_times_sorted}")
    print(f"Species 2 ({species2}) time points: {s2_times_sorted}")

    embeddings = adata.X if embedding_key == "X" else adata.obsm[embedding_key]

    print("\n--- 2. Compute cross-stage similarity matrix ---")
    similarity_matrix = np.zeros((len(s1_times_sorted), len(s2_times_sorted)), dtype=float)

    for i, t1 in enumerate(s1_times_sorted):
        for j, t2 in enumerate(s2_times_sorted):
            idx1 = adata.obs.index[(adata.obs[species_col] == species1) & (adata.obs[time_col] == t1)]
            idx2 = adata.obs.index[(adata.obs[species_col] == species2) & (adata.obs[time_col] == t2)]

            if len(idx1) == 0 or len(idx2) == 0:
                similarity_matrix[i, j] = 0.0
                continue

            loc1 = adata.obs.index.get_indexer(idx1)
            loc2 = adata.obs.index.get_indexer(idx2)
            emb1 = embeddings[loc1, :]
            emb2 = embeddings[loc2, :]

            nn1 = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(emb1)
            nn2 = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(emb2)
            dist1_to_2, _ = nn2.kneighbors(emb1)
            dist2_to_1, _ = nn1.kneighbors(emb2)

            avg_cos_dist = (np.mean(dist1_to_2) + np.mean(dist2_to_1)) / 2.0
            similarity_matrix[i, j] = 1.0 - avg_cos_dist

        print(f"Finished {species1} {t1} ...")

    print("\n--- 3. Plot alignment ---")

    def _map_time_to_frac_index(times_sorted_str, t_num):
        vals = np.array([parse_time(t) for t in times_sorted_str], dtype=float)
        n = len(vals)
        if n == 0:
            return 0.0, True
        if t_num <= vals[0]:
            return 0.0, True
        if t_num >= vals[-1]:
            return float(n - 1), True
        idx = np.searchsorted(vals, t_num)
        i0, i1 = idx - 1, idx
        v0, v1 = vals[i0], vals[i1]
        frac = i0 + (t_num - v0) / (v1 - v0)
        return float(frac), False

    best_match_indices = np.argmax(similarity_matrix, axis=1)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    hm = sns.heatmap(
        similarity_matrix,
        xticklabels=s2_times_sorted,
        yticklabels=s1_times_sorted,
        cmap="Blues",
        cbar=True,
        cbar_kws={"shrink": 0.8, "label": "Cosine similarity"},
        square=True,
        ax=ax,
    )

    if similarity_matrix.shape[1] > 0:
        ax.set_box_aspect(similarity_matrix.shape[0] / similarity_matrix.shape[1])

    ax.invert_yaxis()
    ax.set_title(f"Time similarity matrix with alignment path: {species1} → {species2}")
    ax.set_xlabel(f"{species2} time")
    ax.set_ylabel(f"{species1} time")

    ax.tick_params(axis="x", which="both", width=0.6, length=3, labelsize=8)
    ax.tick_params(axis="y", which="both", width=0.6, length=3, labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    cbar = hm.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(width=0.6, length=3, labelsize=8)
        try:
            cbar.outline.set_linewidth(0.6)
        except Exception:
            pass

    x_on_grid = best_match_indices
    y_on_grid = np.arange(len(s1_times_sorted))
    ax.plot(
        x_on_grid + 0.5,
        y_on_grid + 0.5,
        linestyle="--",
        linewidth=2,
        marker="o",
        markersize=4,
        color="#F3D266",
        markerfacecolor="#F3D266",
        markeredgecolor="#F3D266",
        markeredgewidth=0.6,
        zorder=10,
        label="Best match",
        path_effects=[pe.Stroke(linewidth=3.0, foreground="#F3D266"), pe.Normal()],
    )

    if modified_align is not None and len(modified_align) > 0:
        mod_y_real = [parse_time(k) for k in modified_align.keys()]
        mod_x_real = list(modified_align.values())

        mod_y_idx, mod_y_clip = [], []
        mod_x_idx, mod_x_clip = [], []
        for ty in mod_y_real:
            fi, clipped = _map_time_to_frac_index(s1_times_sorted, ty)
            mod_y_idx.append(fi)
            mod_y_clip.append(clipped)
        for tx in mod_x_real:
            fi, clipped = _map_time_to_frac_index(s2_times_sorted, tx)
            mod_x_idx.append(fi)
            mod_x_clip.append(clipped)

        order = np.argsort(mod_y_real)
        X = np.array(mod_x_idx)[order] + 0.5
        Y = np.array(mod_y_idx)[order] + 0.5

        Y_plot = Y.astype(float).copy()
        Y_plot[Y_plot > 7] = np.nan

        ax.plot(
            X,
            Y_plot,
            linestyle="-",
            linewidth=1.0,
            marker="o",
            markersize=0,
            color="#FC5252",
            alpha=0.2,
            markerfacecolor="#FC5252",
            markeredgecolor="#FC5252",
            markeredgewidth=0.6,
            zorder=11,
            label="Interpolated alignment",
            path_effects=[pe.Stroke(linewidth=3.0, foreground="#FC5252"), pe.Normal()],
        )

    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return similarity_matrix


def calculate_pseudo_tempo_for_species(
    adata_species: anndata.AnnData,
    time_col: str,
    embedding_key: str,
    time_conversion_table: dict,
) -> np.ndarray:
    """
    Compute normalized within-species pseudo-tempo.

    Pseudo-tempo is defined as the cosine distance from each cell
    to its nearest neighbor in the next time point, divided by
    the real time interval.
    """

    def parse_time(t_str):
        match = re.search(r"\d+\.?\d*", str(t_str))
        return float(match.group()) if match else -1

    unique_times = adata_species.obs[time_col].unique()
    sorted_times = sorted(unique_times, key=parse_time)

    embeddings = adata_species.X if embedding_key == "X" else adata_species.obsm[embedding_key]
    time_to_nn_model = {}
    time_to_indices = {}

    for t in sorted_times:
        time_mask = adata_species.obs[time_col] == t
        indices = np.where(time_mask)[0]
        if len(indices) > 0:
            time_to_indices[t] = indices
            nn_model = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
            nn_model.fit(embeddings[indices, :])
            time_to_nn_model[t] = nn_model

    tempos = np.full(adata_species.n_obs, np.nan)

    for i in range(len(sorted_times) - 1):
        current_time_str = sorted_times[i]
        next_time_str = sorted_times[i + 1]

        if current_time_str not in time_to_indices or next_time_str not in time_to_nn_model:
            continue

        current_indices = time_to_indices[current_time_str]
        current_embeddings = embeddings[current_indices, :]

        distances, _ = time_to_nn_model[next_time_str].kneighbors(current_embeddings)
        distances = distances.flatten()

        current_hpf = time_conversion_table.get(current_time_str, parse_time(current_time_str))
        next_hpf = time_conversion_table.get(next_time_str, parse_time(next_time_str))
        delta_t = next_hpf - current_hpf

        if delta_t > 0:
            true_tempos = distances / delta_t
            tempos[current_indices] = true_tempos
        else:
            tempos[current_indices] = 0

    return tempos


def calculate_pseudo_tempo(
    adata: anndata.AnnData,
    time_conversion_table,
    species_col: str = "species",
    time_col: str = "Time",
    embedding_key: str = "X_umap",
) -> anndata.AnnData:
    print("--- Start computing pseudo-tempo for all species ---")

    all_species = adata.obs[species_col].unique()
    print(f"Detected species: {list(all_species)}")

    adata.obs["pseudo_tempo"] = np.nan

    for species in all_species:
        print(f"\n--- Processing species: {species} ---")
        adata_species = adata[adata.obs[species_col] == species, :].copy()
        tempos = calculate_pseudo_tempo_for_species(
            adata_species,
            time_col,
            embedding_key,
            time_conversion_table[species],
        )
        adata.obs.loc[adata_species.obs.index, "pseudo_tempo"] = tempos

    n_nan_before = adata.obs["pseudo_tempo"].isna().sum()
    adata.obs["pseudo_tempo"].fillna(0, inplace=True)

    print(f"\n--- Done ---")
    print(f"Filled {n_nan_before} cells at the last time point of each species with 0.")

    return adata


def plot_and_compare_tempo(
    adata: anndata.AnnData,
    homologous_cell_type: str,
    species_col: str = "species",
    tempo_col: str = "pseudo_tempo",
    cell_type_col: str = "cell_type",
    ax: plt.Axes = None,
):
    """
    Plot and compare pseudo-tempo distributions across species.
    """
    adata_compare = adata[adata.obs[cell_type_col] == homologous_cell_type].copy()

    if adata_compare.n_obs < 10:
        print(f"Too few cells for '{homologous_cell_type}', skipped.")
        return

    species_list = adata_compare.obs[species_col].unique()
    if len(species_list) != 2:
        print(f"Error: expected 2 species, found {len(species_list)}.")
        return

    group1 = adata_compare.obs[adata_compare.obs[species_col] == species_list[0]][tempo_col]
    group2 = adata_compare.obs[adata_compare.obs[species_col] == species_list[1]][tempo_col]

    if len(group1) > 0 and len(group2) > 0:
        stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
    else:
        p_value = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 5))

    order = list(adata_compare.obs[species_col].unique())

    sns.violinplot(
        x=species_col,
        y=tempo_col,
        data=adata_compare.obs,
        ax=ax,
        palette="muted",
        width=0.5,
        cut=0,
        inner=None,
        linewidth=1.0,
    )

    sns.boxplot(
        x=species_col,
        y=tempo_col,
        data=adata_compare.obs,
        ax=ax,
        width=0.05,
        dodge=False,
        showfliers=False,
        showcaps=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.5),
        medianprops=dict(color="black", linewidth=0.5),
        whiskerprops=dict(color="black", linewidth=0.5),
        capprops=dict(color="black", linewidth=0.5),
    )

    for artist in ax.artists:
        artist.set_zorder(3)
    for line in ax.lines:
        line.set_zorder(4)

    ax.set_title(f"Tempo Comparison for {homologous_cell_type}")
    ax.set_xlabel("Species")
    ax.set_ylabel("Transcriptional Pseudo-Tempo")

    if not np.isnan(p_value):
        y_max = adata_compare.obs[tempo_col].max()
        y_top = y_max * 1.2

        ax.plot([0, 0, 1, 1], [y_top, y_top * 1.02, y_top * 1.02, y_top], lw=1.5, color="black")

        p_text = f"P-value = {p_value:.2e}" if p_value < 0.001 else f"P-value = {p_value:.3f}"
        ax.text(0.5, y_top * 1.03, p_text, ha="center", va="bottom", fontsize=12)
        ax.set_ylim(top=y_top * 1.1)

    print(f"Rank-sum test for '{homologous_cell_type}':")
    print(f"P-value = {p_value}")


def gene_tempo_correlation(adata, tempo_key="pseudo_tempo", method="spearman", fdr=True):
    """
    Compute gene-expression correlation with pseudo-tempo.
    """
    y = adata.obs[tempo_key].values
    X = adata.X

    if np.var(y) == 0:
        raise ValueError(f"The tempo vector '{tempo_key}' has zero variance.")

    if hasattr(X, "toarray"):
        X = X.toarray()

    cor_list = []
    pval_list = []

    for i in range(X.shape[1]):
        expr = X[:, i]

        if np.var(expr) == 0:
            cor, pval = np.nan, np.nan
        else:
            if method == "spearman":
                cor, pval = spearmanr(expr, y)
            else:
                cor, pval = np.corrcoef(expr, y)[0, 1], np.nan

        cor_list.append(cor)
        pval_list.append(pval)

    res = pd.DataFrame({
        "gene": adata.var_names,
        "cor": cor_list,
        "pval": pval_list,
    })

    res["pval"].fillna(1.0, inplace=True)

    if fdr:
        res["padj"] = multipletests(res["pval"], method="fdr_bh")[1]

    return res


def plot_correlation_comparison(
    df,
    x_col,
    y_col,
    x_label,
    y_label,
    output_filename=None,
    title="Correlation",
    padj_fish_col="padj_fish",
    padj_frog_col="padj_frog",
    padj_thresh=0.05,
    significance_mode="both",
    cor_fish_col="cor_fish",
    cor_frog_col="cor_frog",
    pos_cor_thresh=0.2,
    neg_cor_thresh=-0.2,
    color_pos="#d62728",
    color_neg="#1f77b4",
    color_other="#2ca02c",
    color_non_sig="lightgrey",
    alpha_pos=0.9,
    alpha_neg=0.9,
    alpha_other=0.42,
    alpha_non_sig=0.35,
    s_pos=12,
    s_neg=12,
    s_other=12,
    s_non_sig=12,
    x_padding_frac=0.05,
    y_padding_frac=0.05,
    create_inset=True,
    inset_width="60%",
    inset_height="28%",
    inset_loc=4,
    inset_pad_frac=0.08,
    inset_frame_color="0.5",
    inset_frame_linewidth=0.2,
    gene_name_col="fish_gene",
    max_labels_inset=None,
    label_score="both",
    blue_label_fontsize=8,
):
    import os
    import matplotlib as mpl
    import matplotlib.collections as mcoll

    d = df.copy()

    minimalist = False
    if output_filename:
        ext = os.path.splitext(str(output_filename))[-1].lower()
        minimalist = ext == ".png"

    valid = d[[x_col, y_col]].dropna()
    rho, pval = stats.spearmanr(valid[x_col], valid[y_col])
    n_points = len(valid)

    fish_sig = d[padj_fish_col] < padj_thresh
    frog_sig = d[padj_frog_col] < padj_thresh
    sig_mask = (fish_sig & frog_sig) if significance_mode == "both" else (fish_sig | frog_sig)

    pos_mask = sig_mask & (d[cor_fish_col] >= pos_cor_thresh) & (d[cor_frog_col] >= pos_cor_thresh)
    neg_mask = sig_mask & (d[cor_fish_col] <= neg_cor_thresh) & (d[cor_frog_col] <= neg_cor_thresh)
    other_sig_mask = sig_mask & ~(pos_mask | neg_mask)
    non_sig_mask = ~sig_mask

    fig, ax = plt.subplots(figsize=(6, 6))

    for sp in ax.spines.values():
        sp.set_linewidth(0.2)

    ax.tick_params(axis="both", which="both", width=0.2)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.2, zorder=0)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.2, zorder=0)

    if non_sig_mask.any():
        sns.scatterplot(
            data=d[non_sig_mask],
            x=x_col,
            y=y_col,
            color=color_non_sig,
            alpha=alpha_non_sig,
            s=s_non_sig,
            edgecolor=None,
            zorder=1,
            ax=ax,
            legend=False if minimalist else "auto",
        )
    if other_sig_mask.any():
        sns.scatterplot(
            data=d[other_sig_mask],
            x=x_col,
            y=y_col,
            color=color_other,
            alpha=alpha_other,
            s=s_other,
            edgecolor="none",
            zorder=2,
            ax=ax,
            legend=False if minimalist else "auto",
            label=None if minimalist else "Significant (others)",
        )
    if pos_mask.any():
        sns.scatterplot(
            data=d[pos_mask],
            x=x_col,
            y=y_col,
            color=color_pos,
            alpha=alpha_pos,
            s=s_pos,
            edgecolor="none",
            zorder=3,
            ax=ax,
            legend=False if minimalist else "auto",
            label=None if minimalist else f"Significant (+/+ ≥ {pos_cor_thresh})",
        )
    if neg_mask.any():
        sns.scatterplot(
            data=d[neg_mask],
            x=x_col,
            y=y_col,
            color=color_neg,
            alpha=alpha_neg,
            s=s_neg,
            edgecolor="none",
            zorder=3,
            ax=ax,
            legend=False if minimalist else "auto",
            label=None if minimalist else f"Significant (-/- ≤ {neg_cor_thresh})",
        )

    x_vals = d[x_col].values
    y_vals = d[y_col].values
    x_min = np.nanmin(x_vals)
    x_max = np.nanmax(x_vals)
    y_min = np.nanmin(y_vals)
    y_max = np.nanmax(y_vals)
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = x_padding_frac * (x_span if x_span > 0 else max(abs(x_min), abs(x_max), 1.0))
    y_pad = y_padding_frac * (y_span if y_span > 0 else max(abs(y_min), abs(y_max), 1.0))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_title(f"{title} (Spearman ρ={rho:.2f}, p={pval:.2g}, N={n_points})", fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    if not minimalist:
        ax.legend(frameon=False)

    ax.grid(False)

    if (not minimalist) and neg_mask.any() and (gene_name_col in d.columns):
        neg_df = d[neg_mask]
        for _, row in neg_df.iterrows():
            gx, gy = row[x_col], row[y_col]
            gname = str(row.get(gene_name_col, ""))
            if gname and gname != "nan":
                ax.text(
                    gx,
                    gy,
                    gname,
                    fontsize=blue_label_fontsize,
                    color="black",
                    ha="left",
                    va="bottom",
                    zorder=4,
                )

    effective_create_inset = (not minimalist) and create_inset and pos_mask.any()
    if effective_create_inset:
        red_df = d[pos_mask].copy()

        if (max_labels_inset is not None) and (len(red_df) > max_labels_inset):
            eps = 1e-300
            score_p = -np.log10(red_df[padj_fish_col].clip(lower=eps)) + -np.log10(red_df[padj_frog_col].clip(lower=eps))
            score_effect = red_df[cor_fish_col].abs() + red_df[cor_frog_col].abs()

            if label_score == "p":
                score = score_p
            elif label_score == "effect":
                score = score_effect
            else:
                score = score_p + score_effect

            red_df["_score"] = score
            red_df = red_df.sort_values("_score", ascending=False).head(max_labels_inset)

        rx = red_df[x_col].values
        ry = red_df[y_col].values
        rx_min = np.nanmin(rx)
        rx_max = np.nanmax(rx)
        ry_min = np.nanmin(ry)
        ry_max = np.nanmax(ry)
        rpx = inset_pad_frac * (rx_max - rx_min if (rx_max - rx_min) > 0 else max(abs(rx_min), abs(rx_max), 1.0))
        rpy = inset_pad_frac * (ry_max - ry_min if (ry_max - ry_min) > 0 else max(abs(ry_min), abs(ry_max), 1.0))
        xlim_inset = (rx_min - rpx, rx_max + rpx)
        ylim_inset = (ry_min - rpy, ry_max + rpy)

        axins = inset_axes(ax, width=inset_width, height=inset_height, loc=inset_loc, borderpad=0.8)
        sns.scatterplot(
            data=red_df,
            x=x_col,
            y=y_col,
            color=color_pos,
            alpha=0.95,
            s=s_pos,
            edgecolor="none",
            zorder=5,
            ax=axins,
            legend=False,
        )
        axins.set_xlim(*xlim_inset)
        axins.set_ylim(*ylim_inset)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_xlabel("")
        axins.set_ylabel("")
        axins.grid(False)

        for sp in axins.spines.values():
            sp.set_linewidth(0.2)

        texts = []
        try:
            from adjustText import adjust_text

            span_x = xlim_inset[1] - xlim_inset[0]
            span_y = ylim_inset[1] - ylim_inset[0]

            for _, row in red_df.iterrows():
                gx, gy = row[x_col], row[y_col]
                gname = str(row.get(gene_name_col, ""))
                if not gname or gname == "nan":
                    continue
                dx = 0.01 * span_x
                dy = 0.01 * span_y
                t = axins.text(
                    gx + dx,
                    gy + dy,
                    gname,
                    fontsize=8,
                    color="black",
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                    zorder=6,
                )
                texts.append(t)

            if texts:
                adjust_text(
                    texts,
                    ax=axins,
                    only_move={"text": "xy"},
                    autoalign="y",
                    expand_text=(1.05, 1.2),
                    expand_points=(1.05, 1.2),
                    arrowprops=dict(arrowstyle="-", linestyle="--", color="grey", linewidth=0.7),
                )
        except Exception:
            offsets = [(0.03, 0.03), (0.03, -0.03), (-0.03, 0.03), (-0.03, -0.03)]
            k = 0
            span_x = xlim_inset[1] - xlim_inset[0]
            span_y = ylim_inset[1] - ylim_inset[0]

            for _, row in red_df.iterrows():
                gx, gy = row[x_col], row[y_col]
                gname = str(row.get(gene_name_col, ""))
                if not gname or gname == "nan":
                    continue
                ox, oy = offsets[k % len(offsets)]
                k += 1
                dx = ox * span_x
                dy = oy * span_y
                axins.annotate(
                    gname,
                    xy=(gx, gy),
                    xytext=(gx + dx, gy + dy),
                    textcoords="data",
                    fontsize=8,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                    arrowprops=dict(arrowstyle="-", linestyle="--", color="grey", linewidth=0.7, shrinkA=0, shrinkB=0),
                    ha="left",
                    va="bottom",
                    zorder=6,
                )

        try:
            patch, lines = mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec=inset_frame_color, lw=inset_frame_linewidth)
            for ln in lines:
                ln.set_linestyle("--")
                ln.set_linewidth(0.2)
            for sp in axins.spines.values():
                sp.set_linewidth(0.2)
        except Exception:
            pass

    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = "figure"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    fig = plt.gcf()
    fig.canvas.draw()

    for col in ax.collections:
        if isinstance(col, mcoll.PathCollection):
            try:
                col.set_rasterized(False)
                col.set_antialiaseds([True])
                col.set_linewidths(0.0)
            except Exception:
                pass

    if output_filename:
        fig.savefig(output_filename, facecolor="white", bbox_inches=None)
        print(f"Figure saved to: {output_filename}")

    plt.show()
    plt.close(fig)