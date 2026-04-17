import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from collections import Counter, defaultdict, deque
from functools import lru_cache

from anndata import AnnData

from matplotlib import cm, colors
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.cluster.hierarchy import linkage, dendrogram, to_tree, ClusterNode
from scipy.sparse import issparse
from scipy.spatial.distance import squareform

from sklearn.metrics.pairwise import cosine_distances

# ------------------------------------------------------------------
# 1. Consensus support
# ------------------------------------------------------------------
def reorder_embeddings(emb, row_labels, label2idx):
    """
    Reorder embedding rows to match the order of `labels_master`.

    Args:
        emb:
            Embedding matrix whose rows correspond to `row_labels`.
        row_labels:
            Row labels for `emb`.
        label2idx:
            Mapping from label to target row index.

    Returns:
        Reordered embedding matrix.
    """
    order = [label2idx[l] for l in row_labels]
    return emb[order]


def extract_clusters_by_label(Z, labels_master):
    """
    Extract all non-root internal clusters from a linkage tree as label sets.

    Args:
        Z:
            Linkage matrix.
        labels_master:
            Master list of leaf labels.

    Returns:
        Set of frozensets, where each frozenset contains the leaf labels
        under one internal node, excluding the full root cluster.
    """
    n = len(labels_master)
    root, nodes = to_tree(Z, rd=True)
    return {
        frozenset(labels_master[j] for j in node.pre_order())
        for node in nodes if node.count < n
    }


def _similarity(set1, set2, metric="f1"):
    """
    Compute set similarity between two clusters.

    Supported metrics:
    - "jaccard": intersection / union
    - "f1": Dice/F1 overlap = 2 * intersection / (size1 + size2)
    """
    inter = len(set1 & set2)
    if metric == "jaccard":
        return inter / (len(set1 | set2))
    elif metric == "f1":
        return 2 * inter / (len(set1) + len(set2))
    else:
        raise ValueError("metric must be 'jaccard' or 'f1'")


# -------------------------------------------------------------
# Fuzzy consensus support with a similarity threshold
# -------------------------------------------------------------
def consensus_support_fuzzy_threshold(embeddings_list,
                                      labels_master,
                                      row_labels_list=None,
                                      method="ward",
                                      metric="f1",
                                      min_sim=0.8):
    """
    Compute a consensus dendrogram and fuzzy support scores.

    For each cluster in the main consensus tree, the function finds the most
    similar cluster in each replicate tree. A replicate contributes only if
    the best similarity is at least `min_sim`; otherwise it contributes 0.

    Args:
        embeddings_list:
            List of embedding matrices, one per replicate.
        labels_master:
            Master list of labels defining the target row order.
        row_labels_list:
            Optional list of row-label lists for each replicate embedding.
            If provided, each embedding is reordered to match `labels_master`.
        method:
            Linkage method.
        metric:
            Similarity metric used between clusters ("jaccard" or "f1").
        min_sim:
            Minimum cluster similarity required for a replicate to contribute.

    Returns:
        Z_main:
            Linkage matrix of the consensus tree.
        support:
            Dictionary mapping cluster leaf sets to support values in [0, 100].
    """
    R = len(embeddings_list)
    label2idx = {lab: i for i, lab in enumerate(labels_master)}
    D_sum, cluster_sets = None, []

    # --------- Process each replicate tree ---------
    for r, emb in enumerate(embeddings_list):
        if row_labels_list is not None:
            emb = reorder_embeddings(emb, row_labels_list[r], label2idx)

        euc = np.sqrt(2 * cosine_distances(emb))
        D_sum = euc if D_sum is None else D_sum + euc

        Z_i = linkage(squareform(euc, checks=False), method=method)
        # Keep all internal clusters plus the full root cluster
        cluster_sets.append(
            extract_clusters_by_label(Z_i, labels_master) |
            {frozenset(labels_master)}
        )

    # --------- Main consensus tree ---------
    D_avg = D_sum / R
    Z_main = linkage(squareform(D_avg, checks=False), method=method)
    all_clusters = (
        extract_clusters_by_label(Z_main, labels_master) |
        {frozenset(labels_master)}
    )

    # --------- Fuzzy support ---------
    support = {}
    for C in all_clusters:
        sim_sum = 0.0
        for Cset in cluster_sets:
            best_sim = max(_similarity(C, Cp, metric) for Cp in Cset)
            if best_sim >= min_sim:
                sim_sum += best_sim
        support[C] = sim_sum / R * 100.0

    return Z_main, support


# ------------------------------------------------------------------
# 2. Visualization helpers
# ------------------------------------------------------------------
def _leaf_x_positions(den):
    """
    Build a mapping from leaf id to x coordinate based on a dendrogram
    generated with `no_plot=True`.

    SciPy places the i-th leaf at x = 5 + 10 * i by default.
    """
    leaves = den['leaves']
    return {leaf_id: 5 + 10 * i for i, leaf_id in enumerate(leaves)}


def _annotate_support(ax,
                      node: ClusterNode,
                      labels_master,
                      leaf_xpos: dict,
                      support: dict,
                      fontsize=7):
    """
    Recursively annotate support values for all internal nodes.
    """
    if node.is_leaf():
        return
    leaves = frozenset(labels_master[j] for j in node.pre_order())
    supp = support.get(leaves, 0)

    # Place the text at the center of the node's horizontal branch
    xs = [leaf_xpos[leaf] for leaf in node.pre_order()]
    x = np.mean(xs)
    y = node.dist
    ax.text(x, y, f"{supp:.0f}%", ha='center', va='bottom',
            fontsize=fontsize, color='black')
    _annotate_support(ax, node.left, labels_master, leaf_xpos, support, fontsize)
    _annotate_support(ax, node.right, labels_master, leaf_xpos, support, fontsize)



# Shared dendrogram renderer
def _render_dendrogram_core(
        Z_main, support, labels_master,
        link_color_func,
        species_color=None,
        size=(7, 18),
        fontsize=8,
        orientation="right",
        flip_lr=False,
        axis_labelsize=None,
        tick_labelsize=None,
        x_major_step=None,
        set_arial=False,
        legend_cfg=None,
    ):
    """
    Render a dendrogram with externally supplied branch colors.

    Args:
        Z_main:
            Linkage matrix.
        support:
            Support dictionary, not always directly used here but kept to
            preserve a consistent interface with wrapper functions.
        labels_master:
            Master leaf labels.
        link_color_func:
            Function mapping a leaf-set frozenset to a branch color.
        species_color:
            Optional species-to-color mapping for leaf labels.
        legend_cfg:
            Optional config dict for a discrete legend.
    """
    if set_arial:
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['svg.fonttype'] = 'none'
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial']

    if axis_labelsize is None:
        axis_labelsize = fontsize + 3
    if tick_labelsize is None:
        tick_labelsize = fontsize

    root, nodes = to_tree(Z_main, rd=True)
    id2leaves = {nd.id: frozenset(labels_master[j] for j in nd.pre_order())
                 for nd in nodes}

    def _link_props(cid):
        return link_color_func(id2leaves[cid])

    fig, ax = plt.subplots(figsize=size)
    den = dendrogram(
        Z_main,
        orientation=orientation,
        leaf_label_func=lambda idx: labels_master[idx],
        link_color_func=_link_props,
        ax=ax
    )

    for coll in ax.collections:
        try:
            coll.set_linewidth(0.6)
            coll.set_capstyle('butt')
        except Exception:
            pass

    if orientation in ("right", "left") and flip_lr:
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x1, x0)

    species_list = sorted({lab.split('_', 1)[0] for lab in labels_master})
    palette = cm.get_cmap('tab10').colors
    species2color = (
        {sp: palette[i % len(palette)] for i, sp in enumerate(species_list)}
        if species_color is None else species_color
    )

    if orientation in ("top", "bottom"):
        fig.canvas.draw()
        for t in ax.get_xticklabels():
            sp = t.get_text().split('_', 1)[0]
            t.set_color(species2color.get(sp, "black"))
        ax.tick_params(axis='x', which='both', labelsize=tick_labelsize)
        ax.tick_params(axis='y', which='both', labelsize=tick_labelsize)
        ax.set_xlabel("Ward distance", fontsize=axis_labelsize)
    else:
        fig.canvas.draw()
        for t in ax.get_yticklabels():
            sp = t.get_text().split('_', 1)[0]
            t.set_color(species2color.get(sp, "black"))
        ax.tick_params(axis='x', which='both', labelsize=tick_labelsize, pad=2)
        ax.tick_params(axis='y', which='both', labelsize=tick_labelsize, pad=4)
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position('none')
        ax.tick_params(axis='y', which='both',
                       left=False, right=False, length=0, width=0,
                       labelleft=False, labelright=True, labelsize=tick_labelsize, pad=4)
        ax.minorticks_off()
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for t in ax.get_yticklabels():
            t.set_ha('left')
        ax.tick_params(axis='y', pad=2)
        ax.set_xlabel("Ward distance", fontsize=axis_labelsize)
        ax.yaxis.label.set_ha('left')
        ax.yaxis.set_label_coords(0.985, 0.5, transform=ax.transAxes)
        fig.canvas.draw()
        for t in ax.get_yticklabels():
            sp = t.get_text().split('_', 1)[0]
            t.set_color(species2color.get(sp, "black"))

    if orientation == "left":
        ax.invert_xaxis()

    if legend_cfg is not None:
        handles = [Line2D([0], [0], color=c, lw=2) for c in legend_cfg["colors"]]
        labels = legend_cfg["labels"]
        ax.legend(handles, labels, title=legend_cfg.get("title", ""),
                  loc=legend_cfg.get("loc", "lower right"),
                  frameon=False, fontsize=tick_labelsize,
                  title_fontsize=tick_labelsize)

    for side in ('right', 'top', 'left', 'bottom'):
        ax.spines[side].set_linewidth(0.6)

    if x_major_step is not None:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_major_step))
    ax.tick_params(axis='x', width=0.6, length=2)
    ax.tick_params(axis='y', width=0.6, length=0)
    ax.set_title("Hierarchical Clustering with Consensus Support", fontsize=axis_labelsize)
    plt.tight_layout()
    return fig, ax


# Version A: branch color = consensus support
def plot_dendrogram_support_colors(
        Z_main, support, labels_master,
        species_color=None,
        size=(7, 18), color_thr=(80, 50),
        fontsize=8,
        orientation="right",
        flip_lr=False,
        axis_labelsize=None,
        tick_labelsize=None,
        x_major_step=None,
        set_arial=False
    ):
    """
    Plot a dendrogram whose branch colors encode consensus support.
    """
    strong, mid, weak = "#941514", "#E3AEAA", "#C8C8C8"

    def _link_color_by_support(leaves_set):
        s = support.get(leaves_set, 0)
        if s >= color_thr[0]:
            return strong
        elif s >= color_thr[1]:
            return mid
        else:
            return weak

    legend_cfg = {
        "colors": [strong, mid, weak],
        "labels": [rf"$\geq$ {color_thr[0]} %",
                   rf"{color_thr[1]}–{color_thr[0]-1} %",
                   rf"$<$ {color_thr[1]} %"],
        "title": "Consensus support",
        "loc": "lower left" if (orientation in ("right", "left") and flip_lr) else "lower right"
    }

    return _render_dendrogram_core(
        Z_main, support, labels_master,
        link_color_func=_link_color_by_support,
        species_color=species_color,
        size=size, fontsize=fontsize,
        orientation=orientation, flip_lr=flip_lr,
        axis_labelsize=axis_labelsize, tick_labelsize=tick_labelsize,
        x_major_step=x_major_step, set_arial=set_arial,
        legend_cfg=legend_cfg
    )


# Version B: branch color = homolog-count bins
def resolve_species_gene(homology_files, species_all_adatas, query_species, query_gene):
    """
    Resolve homologous genes across species by on-demand multi-hop traversal.

    Starting from (`query_species`, `query_gene`), this function loads only
    homology CSV files relevant to the currently visited species, performs a
    BFS-style traversal across species, and finally matches candidate genes
    against each species AnnData `var_names`.

    Args:
        homology_files:
            Dictionary mapping species pairs to CSV paths.
        species_all_adatas:
            Dictionary mapping species name to (AnnData, cell-type-column).
        query_species:
            Species of the query gene.
        query_gene:
            Query gene name.

    Returns:
        Dictionary:
            {species_name: gene_index_or_None}
    """
    csv_cache = {}

    def get_df(pair):
        if pair in csv_cache:
            return csv_cache[pair]
        if pair in homology_files:
            csv_cache[pair] = pd.read_csv(homology_files[pair])
            return csv_cache[pair]
        return None

    visited = set()
    found = defaultdict(set)
    q = deque([(query_species, query_gene)])
    visited.add((query_species, query_gene))
    found[query_species].add(query_gene)

    pairs = list(homology_files.keys())

    while q:
        cur_sp, cur_gene = q.popleft()

        # Traverse every homology table that contains the current species
        for (a, b) in pairs:
            if cur_sp != a and cur_sp != b:
                continue

            df = get_df((a, b))
            if df is None:
                continue

            # Case 1: current species is on the left side
            if cur_sp == a:
                col_cur = f"{a}_gene"
                col_other = f"{b}_gene"
                if col_cur in df.columns and col_other in df.columns:
                    sub = df[df[col_cur] == cur_gene]
                    partners = sub[col_other].dropna().astype(str).unique().tolist()
                    for pg in partners:
                        node = (b, pg)
                        if node not in visited:
                            visited.add(node)
                            found[b].add(pg)
                            q.append(node)

            # Case 2: current species is on the right side
            if cur_sp == b:
                col_cur = f"{b}_gene"
                col_other = f"{a}_gene"
                if col_cur in df.columns and col_other in df.columns:
                    sub = df[df[col_cur] == cur_gene]
                    partners = sub[col_other].dropna().astype(str).unique().tolist()
                    for pg in partners:
                        node = (a, pg)
                        if node not in visited:
                            visited.add(node)
                            found[a].add(pg)
                            q.append(node)

    out = {}
    for sp, (adata, _ct_col) in species_all_adatas.items():
        idx = None
        candidate_genes = list(found.get(sp, []))
        if sp == query_species and query_gene not in candidate_genes:
            candidate_genes.append(query_gene)

        varnames = list(adata.var_names)
        pos = {g: i for i, g in enumerate(varnames)}
        for g in candidate_genes:
            if g in pos:
                idx = pos[g]
                break
        out[sp] = idx

    return out


def build_label2expr_OG_adata(
        labels_master: list[str],
        species_adata: dict[str, AnnData],
        species_gene: dict[str, int],
        layer=None,
):
    """
    Build a mapping:
        {leaf_label -> average expression of the orthologous gene}

    Args:
        labels_master:
            Leaf labels of the form "species_celltype".
        species_adata:
            Mapping from species name to (AnnData, cell-type-column-name).
        species_gene:
            Mapping from species name to gene index or gene name.
        layer:
            Optional AnnData layer to use instead of `.X`.

    Returns:
        Dictionary mapping each leaf label to the corresponding
        cell-type mean expression value.
    """
    label2expr = {}

    # 1) Pre-aggregate expression by species and cell type
    species_cache = defaultdict(dict)
    for sp, ad in species_adata.items():
        celltype_key = ad[1]
        if sp not in species_gene:
            continue
        gene_idx = species_gene[sp]
        if isinstance(gene_idx, str):
            gene_idx = ad[0].var_names.get_loc(gene_idx)

        X = ad[0].layers[layer] if layer else ad[0].X
        col = X[:, gene_idx]
        if issparse(col):
            gene_vec = col.toarray().ravel()
        else:
            gene_vec = np.asarray(col).ravel()

        ctype = ad[0].obs[celltype_key].to_numpy()
        if gene_vec.shape[0] != ctype.shape[0]:
            raise ValueError(
                f"[{sp}] Row mismatch: expr {gene_vec.shape[0]} vs obs {ctype.shape[0]}. "
                f"Check whether the selected layer and .X are aligned."
            )

        df_tmp = (
            pd.DataFrame({"expr": gene_vec, "ctype": ctype})
              .groupby("ctype")["expr"].mean().to_dict()
        )
        species_cache[sp] = df_tmp

    # 2) Read out one value per leaf label
    for lab in labels_master:
        sp, ctype = lab.split("_", 1)

        if sp not in species_cache or ctype not in species_cache[sp]:
            raise KeyError(f"Cannot find mean expression for {lab} in species {sp}")
        else:
            label2expr[lab] = float(species_cache[sp][ctype])

    return label2expr


def _compute_expr_and_delta(root, labels_master, label2expr):
    """
    Compute per-node mean expression and child-minus-parent deltas.

    This follows the current implementation exactly:
    - post-order traversal computes node-level mean expression and leaf count
    - pre-order traversal computes `delta = child.expr - parent.expr`
    """
    def post(nd):
        if nd.is_leaf():
            nd.expr = label2expr[labels_master[nd.id]]
            nd.count = 1
        else:
            post(nd.left)
            post(nd.right)
            nd.expr = (nd.left.expr * nd.left.count +
                       nd.right.expr * nd.right.count) / \
                      (nd.left.count + nd.right.count)
            nd.count = nd.left.count + nd.right.count

    def pre(nd):
        if nd.is_leaf():
            return
        for ch in (nd.left, nd.right):
            ch.delta = ch.expr - nd.expr
            pre(ch)

    root.delta = None
    post(root)
    pre(root)


def plot_tree_drift_fullcolor(
    Z_main, labels_master, label2expr,
    cmap_name="bwr", vmax=None,
    size=(7, 10),
    lw_h=0.5, lw_v=0.5,
    lw_base=0.5,
    flip_lr=True,
    set_arial=True,
    axis_labelsize=None,
    tick_labelsize=None,
    cbar_corner="auto",
    cbar_size=("2%", "10%"),
    cbar_pad=0.01
):
    """
    Plot a dendrogram in which all branch segments are colored by
    expression drift:
        Δe = child expression - parent expression

    Notes:
        - The base dendrogram is first drawn in gray, then recolored by Δe.
    """

    # --------------------------
    # 0) Font defaults
    # --------------------------
    if set_arial:
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['svg.fonttype'] = 'none'
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial']
    if axis_labelsize is None:
        axis_labelsize = 10
    if tick_labelsize is None:
        tick_labelsize = 8

    # --------------------------
    # 1) Tree and Δe values
    # --------------------------
    root, nodes = to_tree(Z_main, rd=True)
    _compute_expr_and_delta(root, labels_master, label2expr)

    abs_deltas = [abs(n.delta) for n in nodes if n.delta is not None]
    if vmax is None:
        vmax = np.percentile(abs_deltas, 95) if abs_deltas else 1e-2
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    # --------------------------
    # 2) Background dendrogram in gray
    # --------------------------
    fig, ax = plt.subplots(figsize=size)
    den = dendrogram(
        Z_main,
        orientation="right",
        leaf_label_func=lambda i: labels_master[i],
        link_color_func=lambda _: "#888888",
        above_threshold_color="#888888",
        color_threshold=-np.inf,
        ax=ax
    )

    for coll in ax.collections:
        try:
            coll.set_linewidth(lw_base)
            coll.set_capstyle('butt')
        except Exception:
            pass

    species_list = sorted({lab.split('_', 1)[0] for lab in labels_master})
    palette = cm.get_cmap('tab10').colors
    species2color = {sp: palette[i % len(palette)]
                     for i, sp in enumerate(species_list)}

    fig.canvas.draw()
    for t in ax.get_yticklabels():
        sp = t.get_text().split('_', 1)[0]
        t.set_color(species2color.get(sp, "black"))

    # --------------------------
    # 3) Recompute node coordinates
    # --------------------------
    y_leaf = {leaf_id: 5 + 10 * idx for idx, leaf_id in enumerate(den["leaves"])}

    def assign_coords(nd):
        if nd.is_leaf():
            nd.x = 0.0
            nd.y = y_leaf[nd.id]
        else:
            assign_coords(nd.left)
            assign_coords(nd.right)
            nd.x = nd.dist
            nd.y = (nd.left.y + nd.right.y) / 2.0
    assign_coords(root)

    # --------------------------
    # 4) Overlay Δe-colored branches
    # --------------------------
    def safe_color(delta):
        return cmap(norm(delta)) if delta is not None and not np.isnan(delta) \
               else (0.8, 0.8, 0.8, 1.0)

    for nd in nodes:
        if nd.is_leaf():
            continue
        ch = nd.left
        ax.hlines(ch.y, ch.x, nd.x, color=safe_color(ch.delta), linewidth=lw_h, zorder=3)
        ch = nd.right
        ax.hlines(ch.y, ch.x, nd.x, color=safe_color(ch.delta), linewidth=lw_h, zorder=3)

        y_low, y_mid, y_high = nd.left.y, nd.y, nd.right.y
        ax.vlines(nd.x, y_low, y_mid, color=safe_color(nd.left.delta), linewidth=lw_v, zorder=3)
        ax.vlines(nd.x, y_mid, y_high, color=safe_color(nd.right.delta), linewidth=lw_v, zorder=3)

    # --------------------------
    # 5) Axis formatting
    # --------------------------
    ax.yaxis.set_label_position('right')
    ax.yaxis.set_ticks_position('none')
    ax.tick_params(axis='y', which='both',
                   left=False, right=False, length=0, width=0,
                   labelleft=False, labelright=True, labelsize=tick_labelsize, pad=4)
    ax.tick_params(axis='x', which='both', labelsize=tick_labelsize, pad=2, width=0.3)
    ax.set_title("Orthologous gene drift", fontsize=axis_labelsize)
    ax.set_xlabel("Ward distance", fontsize=axis_labelsize)
    fig.canvas.draw()
    for t in ax.get_yticklabels():
        sp = t.get_text().split('_', 1)[0]
        t.set_color(species2color.get(sp, "black"))
        t.set_ha('left')

    if flip_lr:
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x1, x0)

    # --------------------------
    # 6) Corner colorbar
    # --------------------------
    if cbar_corner == "auto":
        corner = "lower left" if flip_lr else "lower right"
    else:
        corner = cbar_corner

    cax = inset_axes(
        ax,
        width=cbar_size[0], height=cbar_size[1],
        loc=corner,
        borderpad=0.8,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Δe = child − parent", fontsize=axis_labelsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)
    if hasattr(cbar, "outline") and (cbar.outline is not None):
        cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(which='both', width=0.3)

    if "left" in corner:
        tx_x, ha = 0.02 + cbar_pad, 'left'
    else:
        tx_x, ha = 0.98 - cbar_pad, 'right'

    ax.yaxis.set_label_coords(0.985, 0.5, transform=ax.transAxes)
    ax.tick_params(axis='x', width=0.6, length=2)

    for side in ('right', 'top', 'left', 'bottom'):
        ax.spines[side].set_linewidth(0.6)
    plt.tight_layout()
    return fig, ax