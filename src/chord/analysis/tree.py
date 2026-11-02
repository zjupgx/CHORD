import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Mapping, Sequence

from anndata import AnnData, read_h5ad

from matplotlib import cm, colors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.sparse import issparse

def reorder_embeddings(emb, row_labels, label2idx):
    emb = _dense_float_embedding(emb, source="embedding")
    row_labels = list(row_labels)
    if len(row_labels) != emb.shape[0]:
        raise ValueError(
            "The number of row labels does not match the number of "
            f"embedding rows: {len(row_labels)} != {emb.shape[0]}."
        )
    if len(set(row_labels)) != len(row_labels):
        raise ValueError("Embedding row labels must be unique.")

    master_items = sorted(label2idx.items(), key=lambda item: item[1])
    master_labels = [label for label, _ in master_items]
    expected_indices = list(range(len(master_labels)))
    if [index for _, index in master_items] != expected_indices:
        raise ValueError("label2idx values must be consecutive indices from zero.")

    source_index = {label: index for index, label in enumerate(row_labels)}
    missing = [label for label in master_labels if label not in source_index]
    extra = [label for label in row_labels if label not in label2idx]
    if missing or extra:
        raise ValueError(
            "Embedding labels do not match the master labels. "
            f"Missing={missing}; extra={extra}."
        )

    order = [source_index[label] for label in master_labels]
    return emb[order]


def _dense_float_embedding(embedding, *, source):
    if issparse(embedding):
        embedding = embedding.toarray()
    try:
        matrix = np.asarray(embedding, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must contain numeric values.") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{source} must be two-dimensional, got {matrix.ndim}D.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{source} contains NaN or infinite values.")
    return matrix


def _l2_normalize_embedding_rows(embedding, *, source):
    matrix = _dense_float_embedding(embedding, source=source)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms[:, 0] <= np.finfo(np.float64).eps)
    if zero_rows.size:
        raise ValueError(
            f"{source} contains zero-norm rows at indices "
            f"{zero_rows.tolist()}."
        )
    return matrix / norms


def _prepare_normalized_embeddings(
        embeddings_list,
        labels_master,
        row_labels_list=None,
        row_indices=None,
    ):
    embeddings = list(embeddings_list)
    labels_master = list(labels_master)
    if not embeddings:
        raise ValueError("embeddings_list must contain at least one embedding.")
    if len(labels_master) < 2:
        raise ValueError("At least two master labels are required to build a tree.")
    if len(set(labels_master)) != len(labels_master):
        raise ValueError("labels_master must contain unique labels.")

    if row_labels_list is not None:
        row_labels_list = list(row_labels_list)
        if len(row_labels_list) != len(embeddings):
            raise ValueError(
                "row_labels_list must have one entry per embedding: "
                f"{len(row_labels_list)} != {len(embeddings)}."
            )

    label2idx = {label: index for index, label in enumerate(labels_master)}
    normalized = []
    for replicate_index, embedding in enumerate(embeddings):
        source = f"embedding replicate {replicate_index + 1}"
        matrix = _dense_float_embedding(embedding, source=source)
        if row_labels_list is not None:
            matrix = reorder_embeddings(
                matrix,
                row_labels_list[replicate_index],
                label2idx,
            )
        elif matrix.shape[0] != len(labels_master):
            raise ValueError(
                f"{source} has {matrix.shape[0]} rows but "
                f"{len(labels_master)} master labels were provided."
            )
        if row_indices is not None:
            matrix = matrix[row_indices, :]
        normalized.append(
            _l2_normalize_embedding_rows(matrix, source=source)
        )
    return normalized


def _representation_source(adata, representation, *, path):
    if representation == "X":
        source = adata.X
    elif representation.startswith("obsm:"):
        key = representation.split(":", 1)[1]
        if not key or key not in adata.obsm:
            raise KeyError(
                f"{path} does not contain obsm representation {key!r}."
            )
        source = adata.obsm[key]
    elif representation.startswith("layer:"):
        key = representation.split(":", 1)[1]
        if not key or key not in adata.layers:
            raise KeyError(
                f"{path} does not contain layer representation {key!r}."
            )
        source = adata.layers[key]
    else:
        raise ValueError(
            "representation must be 'X', 'obsm:<key>', or 'layer:<key>'."
        )

    if len(source.shape) != 2 or source.shape[0] != adata.n_obs:
        raise ValueError(
            f"Representation {representation!r} in {path} must have shape "
            f"(n_obs, n_features); got {source.shape}."
        )
    return source


def _representation_rows(source, positions, *, source_name):
    values = _raw_representation_rows(source, positions)
    return _dense_float_embedding(values, source=source_name)


def _raw_representation_rows(source, positions):
    if hasattr(source, "iloc"):
        return source.iloc[positions].to_numpy()
    return source[positions, :]


def _normalize_species_map(species_map):
    if species_map is None:
        return {}
    if not isinstance(species_map, Mapping):
        raise TypeError("species_map must be a mapping of source to target names.")

    normalized = {}
    for source, target in species_map.items():
        source = str(source)
        target = str(target)
        if not source or not target:
            raise ValueError("species_map names must be non-empty strings.")
        normalized[source] = target
    return normalized


def _apply_species_map(obs, *, path, species_key, species_map):
    if not species_map:
        return obs
    if species_key not in obs:
        raise KeyError(
            f"{path} is missing species obs column {species_key!r}, which is "
            "required by species_map."
        )

    present_species = {
        str(value) for value in obs[species_key].dropna().unique().tolist()
    }
    collisions = [
        (source, target)
        for source, target in species_map.items()
        if source != target and source in present_species and target in present_species
    ]
    if collisions:
        formatted = ", ".join(
            f"{source!r} -> {target!r}" for source, target in collisions
        )
        raise ValueError(
            f"Species mapping for {path} would merge species that already coexist in "
            f"{species_key!r}: {formatted}. Rename or filter the inputs explicitly "
            "before building the tree."
        )

    mapped_obs = obs.copy()
    mapped_obs[species_key] = [
        value
        if pd.isna(value)
        else species_map.get(str(value), str(value))
        for value in obs[species_key]
    ]
    return mapped_obs


def _observation_labels(
        obs,
        positions,
        *,
        path,
        label_key,
        group_keys,
        label_separator,
        original_obs=None,
        species_key="species",
        species_map=None,
    ):
    selected_obs = obs.iloc[positions]
    if group_keys:
        missing = [key for key in group_keys if key not in selected_obs]
        if missing:
            raise KeyError(f"{path} is missing group obs columns: {missing}.")
        if selected_obs[list(group_keys)].isna().to_numpy().any():
            raise ValueError(
                f"{path} contains missing values in group columns "
                f"{list(group_keys)}."
            )
        group_values = [
            tuple(map(str, values))
            for values in selected_obs[list(group_keys)].itertuples(
                index=False, name=None
            )
        ]
        labels = [label_separator.join(values) for values in group_values]
        label_to_group = {}
        for label, group_value in zip(labels, group_values):
            previous = label_to_group.setdefault(label, group_value)
            if previous != group_value:
                raise ValueError(
                    f"Joining group columns with {label_separator!r} creates "
                    f"the ambiguous label {label!r} in {path}."
                )
        return labels, group_values

    if label_key not in selected_obs:
        raise KeyError(f"{path} is missing label obs column {label_key!r}.")
    if selected_obs[label_key].isna().any():
        raise ValueError(f"{path} contains missing values in {label_key!r}.")
    labels = selected_obs[label_key].astype(str).tolist()

    if species_map:
        original_selected = original_obs.iloc[positions]
        original_species = original_selected[species_key].astype(str).tolist()
        mapped_species = selected_obs[species_key].astype(str).tolist()
        remapped_labels = []
        for label, source_species, target_species in zip(
                labels, original_species, mapped_species
            ):
            if source_species == target_species:
                remapped_labels.append(label)
                continue
            prefix = f"{source_species}{label_separator}"
            if label == source_species:
                remapped_labels.append(target_species)
            elif label.startswith(prefix):
                remapped_labels.append(
                    f"{target_species}{label_separator}{label[len(prefix):]}"
                )
            else:
                raise ValueError(
                    f"Cannot apply species mapping {source_species!r} -> "
                    f"{target_species!r} to label {label!r} in {path}. Use "
                    "group_keys containing the species column, or labels of "
                    "the form 'species<separator>value'."
                )
        labels = remapped_labels
    return labels, labels


def _mean_rows_by_group(
        representation_source,
        positions,
        labels,
        group_values,
        *,
        source_name,
    ):
    group_to_positions = {}
    group_to_label = {}
    group_order = []
    for position, label, group_value in zip(positions, labels, group_values):
        if group_value not in group_to_positions:
            group_order.append(group_value)
            group_to_positions[group_value] = []
            group_to_label[group_value] = label
        group_to_positions[group_value].append(int(position))

    averaged_rows = []
    averaged_labels = []
    for group_value in group_order:
        group_positions = np.asarray(group_to_positions[group_value], dtype=int)
        group_matrix = _raw_representation_rows(
            representation_source, group_positions
        )
        if issparse(group_matrix):
            group_mean = np.asarray(group_matrix.mean(axis=0)).reshape(-1)
        else:
            dense_group = _dense_float_embedding(
                group_matrix, source=source_name
            )
            group_mean = dense_group.mean(axis=0)
        if not np.isfinite(group_mean).all():
            raise ValueError(f"{source_name} produces a non-finite group mean.")
        averaged_rows.append(group_mean)
        averaged_labels.append(group_to_label[group_value])
    return np.vstack(averaged_rows), averaged_labels


def load_label_embeddings_from_h5ads(
        h5ad_paths: Sequence[str | Path],
        *,
        representation: str = "X",
        aggregation: str = "none",
        label_key: str = "label_str",
        group_keys: Sequence[str] | None = None,
        filter_key: str | None = None,
        filter_value: str | None = None,
        label_separator: str = "_",
        species_key: str = "species",
        species_map: Mapping[str, str] | None = None,
    ):
    paths = [Path(path) for path in h5ad_paths]
    if not paths:
        raise ValueError("At least one h5ad path is required.")
    if len(set(paths)) != len(paths):
        raise ValueError("h5ad paths must be unique.")
    if aggregation not in {"none", "mean"}:
        raise ValueError("aggregation must be 'none' or 'mean'.")
    if (filter_key is None) != (filter_value is None):
        raise ValueError("filter_key and filter_value must be provided together.")
    if not isinstance(species_key, str) or not species_key:
        raise ValueError("species_key must be a non-empty string.")
    species_map = _normalize_species_map(species_map)
    if group_keys is not None:
        if isinstance(group_keys, str):
            group_keys = (group_keys,)
        group_keys = tuple(group_keys)
        if not group_keys:
            group_keys = None
        elif len(set(group_keys)) != len(group_keys):
            raise ValueError("group_keys must be unique.")

    embeddings_list = []
    row_labels_list = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Embedding h5ad file not found: {path}")
        adata = read_h5ad(path, backed="r")
        try:
            mapped_obs = _apply_species_map(
                adata.obs,
                path=path,
                species_key=species_key,
                species_map=species_map,
            )
            if filter_key is None:
                mask = np.ones(adata.n_obs, dtype=bool)
            else:
                if filter_key not in mapped_obs:
                    raise KeyError(
                        f"{path} is missing filter obs column {filter_key!r}."
                    )
                filter_values = mapped_obs[filter_key].astype(str).to_numpy()
                mask = filter_values == str(filter_value)
            positions = np.flatnonzero(mask)
            if positions.size < 2:
                raise ValueError(
                    f"{path} contains fewer than two selected observations."
                )

            labels, group_values = _observation_labels(
                mapped_obs,
                positions,
                path=path,
                label_key=label_key,
                group_keys=group_keys,
                label_separator=label_separator,
                original_obs=adata.obs,
                species_key=species_key,
                species_map=species_map,
            )
            source = _representation_source(
                adata, representation, path=path
            )
            source_name = f"{path}:{representation}"
            if aggregation == "mean":
                embedding, labels = _mean_rows_by_group(
                    source,
                    positions,
                    labels,
                    group_values,
                    source_name=source_name,
                )
            else:
                embedding = _representation_rows(
                    source,
                    positions,
                    source_name=source_name,
                )

            if len(labels) < 2:
                raise ValueError(
                    f"{path} produces fewer than two label vectors."
                )
            if len(set(labels)) != len(labels):
                duplicates = sorted(
                    label for label, count in Counter(labels).items() if count > 1
                )
                raise ValueError(
                    f"{path} produces duplicate labels: {duplicates}. Use "
                    "aggregation='mean' or provide more specific group keys."
                )
            embeddings_list.append(embedding)
            row_labels_list.append(labels)
        finally:
            adata.file.close()

    labels_master = list(row_labels_list[0])
    label2idx = {label: index for index, label in enumerate(labels_master)}
    for embedding, labels in zip(embeddings_list, row_labels_list):
        reorder_embeddings(embedding, labels, label2idx)

    return embeddings_list, labels_master, row_labels_list


def load_celltype_embeddings_from_h5ads(
        h5ad_paths: Sequence[str | Path],
        *,
        item_type_key: str = "item_type",
        celltype_value: str = "celltype",
        label_key: str = "label_str",
        species_key: str = "species",
        species_map: Mapping[str, str] | None = None,
    ):
    return load_label_embeddings_from_h5ads(
        h5ad_paths,
        representation="X",
        aggregation="none",
        label_key=label_key,
        filter_key=item_type_key,
        filter_value=celltype_value,
        species_key=species_key,
        species_map=species_map,
    )


def extract_clusters_by_label(Z, labels_master):
    n = len(labels_master)
    root, nodes = to_tree(Z, rd=True)
    return {
        frozenset(labels_master[j] for j in node.pre_order())
        for node in nodes if node.count < n
    }


def _similarity(set1, set2, metric="f1"):
    inter = len(set1 & set2)
    if metric == "jaccard":
        return inter / (len(set1 | set2))
    elif metric == "f1":
        return 2 * inter / (len(set1) + len(set2))
    else:
        raise ValueError("metric must be 'jaccard' or 'f1'")


def consensus_support_fuzzy_threshold(embeddings_list,
                                      labels_master,
                                      row_labels_list=None,
                                      method="ward",
                                      metric="f1",
                                      min_sim=0.8):
    labels_master = list(labels_master)
    if not 0.0 <= float(min_sim) <= 1.0:
        raise ValueError("min_sim must be between 0 and 1.")
    normalized_embeddings = _prepare_normalized_embeddings(
        embeddings_list,
        labels_master,
        row_labels_list=row_labels_list,
    )
    R = len(normalized_embeddings)
    cluster_sets = []

    for embedding in normalized_embeddings:
        Z_i = linkage(embedding, method=method, metric="euclidean")
        cluster_sets.append(
            extract_clusters_by_label(Z_i, labels_master) |
            {frozenset(labels_master)}
        )

    joint_embedding = (
        np.concatenate(normalized_embeddings, axis=1) / np.sqrt(float(R))
    )
    Z_main = linkage(joint_embedding, method=method, metric="euclidean")
    all_clusters = (
        extract_clusters_by_label(Z_main, labels_master) |
        {frozenset(labels_master)}
    )

    support = {}
    for C in all_clusters:
        sim_sum = 0.0
        for Cset in cluster_sets:
            best_sim = max(_similarity(C, Cp, metric) for Cp in Cset)
            if best_sim >= min_sim:
                sim_sum += best_sim
        support[C] = sim_sum / R * 100.0

    return Z_main, support


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
        "labels": [f"[{color_thr[0]:g}, 100]",
                   f"[{color_thr[1]:g}, {color_thr[0]:g})",
                   f"[0, {color_thr[1]:g})"],
        "title": "Support (%)",
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


def resolve_species_gene(homology_files, species_all_adatas, query_species, query_gene):
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

        for (a, b) in pairs:
            if cur_sp != a and cur_sp != b:
                continue

            df = get_df((a, b))
            if df is None:
                continue

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
    label2expr = {}

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

    for lab in labels_master:
        sp, ctype = lab.split("_", 1)

        if sp not in species_cache or ctype not in species_cache[sp]:
            raise KeyError(f"Cannot find mean expression for {lab} in species {sp}")
        else:
            label2expr[lab] = float(species_cache[sp][ctype])

    return label2expr


def _compute_expr_and_delta(root, labels_master, label2expr):
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
    cbar_pad=0.01,
    orientation="right",
):

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

    root, nodes = to_tree(Z_main, rd=True)
    _compute_expr_and_delta(root, labels_master, label2expr)

    abs_deltas = [abs(n.delta) for n in nodes if n.delta is not None]
    if vmax is None:
        vmax = np.percentile(abs_deltas, 95) if abs_deltas else 1e-2
        if vmax <= 0:
            vmax = max(abs_deltas, default=0.0) or 1e-2
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = mpl.colormaps[cmap_name]

    fig, ax = plt.subplots(figsize=size)
    den = dendrogram(
        Z_main,
        orientation=orientation,
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
    palette = mpl.colormaps['tab10'].colors
    species2color = {sp: palette[i % len(palette)]
                     for i, sp in enumerate(species_list)}

    fig.canvas.draw()
    leaf_ticklabels = (
        ax.get_yticklabels() if orientation in ("left", "right")
        else ax.get_xticklabels()
    )
    for t in leaf_ticklabels:
        sp = t.get_text().split('_', 1)[0]
        t.set_color(species2color.get(sp, "black"))

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

    def safe_color(delta):
        return cmap(norm(delta)) if delta is not None and not np.isnan(delta) \
               else (0.8, 0.8, 0.8, 1.0)

    for nd in nodes:
        if nd.is_leaf():
            continue
        for child in (nd.left, nd.right):
            if orientation in ("left", "right"):
                ax.hlines(child.y, child.x, nd.x,
                          color=safe_color(child.delta), linewidth=lw_h, zorder=3)
                ax.vlines(nd.x, min(child.y, nd.y), max(child.y, nd.y),
                          color=safe_color(child.delta), linewidth=lw_v, zorder=3)
            else:
                ax.vlines(child.y, child.x, nd.x,
                          color=safe_color(child.delta), linewidth=lw_v, zorder=3)
                ax.hlines(nd.x, min(child.y, nd.y), max(child.y, nd.y),
                          color=safe_color(child.delta), linewidth=lw_h, zorder=3)

    ax.set_title("Orthologous gene drift", fontsize=axis_labelsize)
    if orientation in ("left", "right"):
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position('none')
        ax.tick_params(axis='y', which='both',
                       left=False, right=False, length=0, width=0,
                       labelleft=False, labelright=True, labelsize=tick_labelsize, pad=4)
        ax.tick_params(axis='x', which='both', labelsize=tick_labelsize, pad=2, width=0.3)
        ax.set_xlabel("Ward distance", fontsize=axis_labelsize)
    else:
        ax.set_ylabel("Ward distance", fontsize=axis_labelsize)
        ax.tick_params(axis='x', which='both', labelsize=tick_labelsize, length=0)
        ax.tick_params(axis='y', which='both', labelsize=tick_labelsize, width=0.3)
    fig.canvas.draw()
    leaf_ticklabels = (
        ax.get_yticklabels() if orientation in ("left", "right")
        else ax.get_xticklabels()
    )
    for t in leaf_ticklabels:
        sp = t.get_text().split('_', 1)[0]
        t.set_color(species2color.get(sp, "black"))
        if orientation in ("left", "right"):
            t.set_ha('left')

    if flip_lr and orientation in ("left", "right"):
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x1, x0)

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

    if orientation in ("left", "right"):
        ax.yaxis.set_label_coords(0.985, 0.5, transform=ax.transAxes)
        ax.tick_params(axis='x', width=0.6, length=2)
    else:
        ax.tick_params(axis='y', width=0.6, length=2)

    for side in ('right', 'top', 'left', 'bottom'):
        ax.spines[side].set_linewidth(0.6)
    plt.tight_layout()
    return fig, ax
