from sklearn.metrics import silhouette_score, silhouette_samples
import scanpy as sc
import scib.metrics as sm
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


def evaluate_celltype(
        adata,
        rep: str = "X",
        n_neighbors: int = 15,
        seed: int = 0,
        gammas=np.arange(0.1, 2.05, 0.1),
):
    """
    Evaluate cell-type clustering quality.

    This function:
    1. Keeps only cells if `item_type` is available.
    2. Builds a KNN graph using the requested representation.
    3. Runs Leiden clustering across multiple resolutions.
    4. Reports the best NMI and ARI.
    5. Computes ASW on cell-type labels.

    Args:
        adata:
            AnnData containing embeddings and annotations.
        rep:
            Representation to use. Either "X" or a key in `adata.obsm`.
        n_neighbors:
            Number of neighbors for graph construction.
        seed:
            Random seed for Leiden clustering.
        gammas:
            Leiden resolutions to evaluate.

    Returns:
        best_nmi, asw, best_ari
    """
    # Keep only cells if both cells and cell-type embeddings are present
    if "item_type" in adata.obs.columns:
        adata_work = adata[adata.obs["item_type"] == "cell"].copy()
    else:
        adata_work = adata.copy()

    # Build a KNN graph
    sc.pp.neighbors(adata_work, n_neighbors=n_neighbors, use_rep=rep, random_state=seed)

    best_nmi = best_ari = 0.0
    for gamma in gammas:
        key = f"leiden_{gamma:.2f}"
        sc.tl.leiden(adata_work, resolution=gamma, key_added=key, random_state=seed)
        best_nmi = max(best_nmi, sm.nmi(adata_work, key, "cell_type"))
        best_ari = max(best_ari, sm.ari(adata_work, key, "cell_type"))

    # Compute ASW using only cell types with at least 2 cells
    X_use = adata_work.obsm[rep] if rep != "X" else adata_work.X
    uniq, cnt = np.unique(adata_work.obs["cell_type"], return_counts=True)
    mask = np.isin(adata_work.obs["cell_type"], uniq[cnt > 1])
    asw = (silhouette_score(X_use[mask], adata_work.obs["cell_type"][mask]) + 1) / 2

    print(f"NMI={best_nmi:.4f} | ASW={asw:.4f} | ARI={best_ari:.4f}")
    return best_nmi, asw, best_ari


def evaluate_batch(
        adata,
        batch_key: str = "species",
        rep: str = "X",
):
    """
    Evaluate batch mixing across species.

    This function:
    1. Keeps only cells if `item_type` is available.
    2. Computes batch ASW using the specified representation.
    3. Computes graph connectivity after neighbor graph construction.

    Args:
        adata:
            AnnData containing embeddings and annotations.
        batch_key:
            Observation column used as the batch label.
        rep:
            Representation to use. Either "X" or a key in `adata.obsm`.

    Returns:
        batch_asw, gc
    """
    if "item_type" in adata.obs.columns:
        adata_cells = adata[adata.obs["item_type"] == "cell"].copy()
    else:
        adata_cells = adata.copy()

    X = adata_cells.X if rep == "X" else adata_cells.obsm[rep]
    batch_labels = adata_cells.obs[batch_key].values
    cell_labels = adata_cells.obs["cell_type"].values

    # Batch ASW
    s = silhouette_samples(X, batch_labels)
    s_batch = (1 - s) / 2
    batch_asw = np.mean([
        np.mean(s_batch[cell_labels == lbl])
        for lbl in np.unique(cell_labels)
        if np.sum(cell_labels == lbl) > 1 and len(np.unique(batch_labels[cell_labels == lbl])) > 1
    ]) if len(cell_labels) else 0.0

    # Graph connectivity
    sc.pp.neighbors(adata_cells, use_rep=rep)
    gc = sm.graph_connectivity(adata_cells, label_key="cell_type")

    print(f"Batch ASW: {batch_asw:.4f}, Graph Connectivity: {gc:.4f}")
    return batch_asw, gc


def calculate_species_overlap(adata, species_Q, species_R):
    """
    Compute species overlap based on cell-type coverage.

    The score is defined as:
        1 - (number of cells in species_Q whose cell types do not have
             matching cell-type embeddings in species_R) / (total number of cells in species_Q)

    Args:
        adata:
            AnnData containing both cells and cell-type embeddings.
        species_Q:
            Query species whose cells are evaluated.
        species_R:
            Reference species providing cell-type embedding labels.

    Returns:
        overlap_score:
            Overlap score in [0, 1].
    """
    # Cell types present among cells of species Q
    q_cell_mask = (adata.obs['species'] == species_Q) & (adata.obs['item_type'] == 'cell')
    q_cell_types = set(adata.obs.loc[q_cell_mask, 'cell_type'].unique())
    
    # Cell types present among cell-type embeddings of species R
    r_celltype_mask = (adata.obs['species'] == species_R) & (adata.obs['item_type'] == 'celltype')
    r_cell_types = set(adata.obs.loc[r_celltype_mask, 'cell_type'].unique())
    
    # Cell types in species Q that are absent from species-R cell-type embeddings
    unique_q_types = q_cell_types - r_cell_types
    
    # Count cells in species Q that belong to these unmatched cell types
    unique_q_cells_mask = q_cell_mask & adata.obs['cell_type'].isin(unique_q_types)
    unique_q_cell_count = np.sum(unique_q_cells_mask)
    
    # Total number of cells in species Q
    total_q_cell_count = np.sum(q_cell_mask)
    
    # Avoid division by zero
    if total_q_cell_count == 0:
        return 0.0
    
    unique_proportion = unique_q_cell_count / total_q_cell_count
    overlap_score = 1 - unique_proportion
    
    print(f"Maximum annotation accuracy of {species_Q} using {species_R} is: {overlap_score:.4f}")
    
    return overlap_score


def transfer_labels_across_species(
    adata, species_Q, species_R,
    averages=("micro", "macro", "weighted"),
    restrict_to_intersection=True,
    return_confusion_matrix=True
):
    """
    Evaluate cross-species label transfer using cell-type embedding similarity.

    This function:
    1. Uses cells from `species_Q` as query samples.
    2. Uses cell-type embeddings from `species_R` as reference labels.
    3. Assigns each query cell to the most similar reference cell-type embedding
       using cosine similarity.
    4. Reports both intersection-based and union-based evaluation metrics.

    Args:
        adata:
            AnnData containing cells and cell-type embeddings.
        species_Q:
            Query species whose cells will be classified.
        species_R:
            Reference species providing cell-type embedding labels.
        averages:
            F1 averaging modes to report.
        restrict_to_intersection:
            If True, use only cells whose true labels are also present
            in the reference cell-type embedding label set when selecting the main result.
        return_confusion_matrix:
            Whether to include a confusion-matrix DataFrame.

    Returns:
        Dictionary containing:
            - the selected main metrics
            - label-coverage information
            - full intersection-based metrics
            - full union-based metrics
    """

    # 1) Query cells from species Q
    q_cell_mask = (adata.obs['species'] == species_Q) & (adata.obs['item_type'] == 'cell')
    if np.sum(q_cell_mask) == 0:
        print(f"Error: species {species_Q} has no cell data")
        return {}

    qX = adata[q_cell_mask].X
    if sparse.issparse(qX):
        q_cell_embeddings = qX.A
    else:
        q_cell_embeddings = np.asarray(qX)
    y_true_all = adata.obs.loc[q_cell_mask, 'cell_type'].astype(str).values

    # 2) Reference cell-type embeddings from species R
    r_celltype_mask = (adata.obs['species'] == species_R) & (adata.obs['item_type'] == 'celltype')
    if np.sum(r_celltype_mask) == 0:
        print(f"Error: species {species_R} has no cell-type embedding data")
        return {}

    rX = adata[r_celltype_mask].X
    if sparse.issparse(rX):
        r_celltype_embeddings = rX.A
    else:
        r_celltype_embeddings = np.asarray(rX)
    r_celltype_labels = adata.obs.loc[r_celltype_mask, 'cell_type'].astype(str).values

    # 3) Cosine similarity and nearest-cell-type assignment
    sim = cosine_similarity(q_cell_embeddings, r_celltype_embeddings)
    max_idx = np.argmax(sim, axis=1)
    y_pred_all = r_celltype_labels[max_idx]

    # Label-set relationships and coverage summary
    true_set = set(y_true_all)
    celltype_set = set(r_celltype_labels)
    inter_set = true_set & celltype_set
    coverage = {
        "true_in_celltype_coverage": (len(inter_set) / len(true_set)) if len(true_set) else 0.0,
        "true_only_labels": sorted(true_set - celltype_set),
        "celltype_only_labels": sorted(celltype_set - true_set),
    }

    # ---------- A) Intersection evaluation ----------
    # Evaluate only on cells whose true labels belong to the shared label set
    def eval_block(y_true, y_pred, labels_used, want_cm):
        acc = float(np.mean(y_true == y_pred))

        f1 = {}
        for avg in averages:
            f1[avg] = f1_score(y_true, y_pred, average=avg, labels=labels_used, zero_division=0)

        per_class_scores = f1_score(y_true, y_pred, average=None, labels=labels_used, zero_division=0)
        per_class_f1 = {lbl: sc for lbl, sc in zip(labels_used, per_class_scores)}

        report = classification_report(y_true, y_pred, labels=labels_used, digits=4, zero_division=0)

        cm_df = None
        if want_cm:
            cm = confusion_matrix(y_true, y_pred, labels=labels_used)
            cm_df = pd.DataFrame(
                cm,
                index=pd.Index(labels_used, name="True"),
                columns=pd.Index(labels_used, name="Pred")
            )
        return {
            "accuracy": acc,
            "f1": f1,
            "per_class_f1": per_class_f1,
            "report": report,
            "confusion_matrix": cm_df,
            "labels_used": list(labels_used),
        }

    inter_labels_sorted = sorted(inter_set)
    if len(inter_labels_sorted) > 0:
        inter_mask = np.isin(y_true_all, inter_labels_sorted)
        y_true_inter = y_true_all[inter_mask]
        y_pred_inter = y_pred_all[inter_mask]
        metrics_intersection = eval_block(
            y_true_inter, y_pred_inter, inter_labels_sorted, return_confusion_matrix
        )
    else:
        metrics_intersection = {
            "accuracy": np.nan, "f1": {avg: np.nan for avg in averages},
            "per_class_f1": {}, "report": "No intersecting labels.",
            "confusion_matrix": None, "labels_used": []
        }

    # ---------- B) Union evaluation ----------
    # Evaluate all query cells using the union of true and predicted labels
    union_labels_sorted = sorted(set(y_true_all) | set(y_pred_all))
    metrics_union = eval_block(
        y_true_all, y_pred_all, union_labels_sorted, return_confusion_matrix
    )

    # ---------- C) Main result selection ----------
    if restrict_to_intersection:
        main_metrics = metrics_intersection
        mode_name = f"{species_R} -> {species_Q} | intersection evaluation"
    else:
        main_metrics = metrics_union
        mode_name = f"{species_R} -> {species_Q} | union evaluation"

    print(f"Cross-species prediction {mode_name}")
    print(f"- true_in_celltype coverage: {coverage['true_in_celltype_coverage']:.4f}  "
          f"(true-only: {coverage['true_only_labels']}, celltype-only: {coverage['celltype_only_labels']})")
    print(f"- Accuracy: {main_metrics['accuracy']:.4f}")
    for avg in averages:
        print(f"- F1-{avg}: {main_metrics['f1'][avg]:.4f}")

    return {
        **main_metrics,
        "coverage": coverage,
        "metrics_intersection": metrics_intersection,
        "metrics_union": metrics_union,
    }

