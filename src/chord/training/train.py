import numpy as np
import torch
import torch.nn.functional as F


def row_cosine_similarity(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise row-wise cosine similarities between two matrices.

    Args:
        matrix1:
            Tensor of shape [m, d].
        matrix2:
            Tensor of shape [n, d].

    Returns:
        Tensor of shape [m, n], where entry (i, j) is the cosine similarity
        between row i of `matrix1` and row j of `matrix2`.
    """
    # Ensure both inputs are 2D and have the same feature dimension
    assert matrix1.dim() == 2 and matrix2.dim() == 2, "Inputs must be 2D tensors"
    assert matrix1.size(1) == matrix2.size(1), "Feature dimensions must match"
    
    # Compute L2 norms for each row
    norm1 = torch.norm(matrix1, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(matrix2, p=2, dim=1, keepdim=True)
    
    # Avoid division by zero
    eps = 1e-8
    norm1 = torch.clamp(norm1, min=eps)
    norm2 = torch.clamp(norm2, min=eps)
    
    # Normalize rows
    normalized_matrix1 = matrix1 / norm1
    normalized_matrix2 = matrix2 / norm2
    
    # Row-wise cosine similarity equals the dot product after normalization
    similarity = torch.matmul(normalized_matrix1, normalized_matrix2.transpose(0, 1))
    
    return similarity


def process_tensor(sp_ct_correlation, k=2):
    """
    Build a mutual top-k matching mask from a similarity matrix.

    For each pair (i, j), the output entry is set to 1 / k only if:
    - j is among the top-k neighbors of row i, and
    - i is among the top-k neighbors of column j.

    Otherwise, the output entry remains 0.

    Args:
        sp_ct_correlation:
            Similarity matrix, typically between selected cell-type averages
            from two species.
        k:
            Number of nearest neighbors used in each direction.

    Returns:
        Tensor of the same shape as `sp_ct_correlation`, containing a
        mutual top-k mask weighted by 1 / k.
    """
    # Find top-k neighbors in each direction
    _, topk_sp1_to_sp2 = torch.topk(sp_ct_correlation, k=k, dim=1)
    _, topk_sp2_to_sp1 = torch.topk(sp_ct_correlation, k=k, dim=0)
    
    # Build a mutual top-k mask
    result = torch.zeros_like(sp_ct_correlation)
    for i in range(sp_ct_correlation.shape[0]):
        for j in range(sp_ct_correlation.shape[1]):
            if (topk_sp1_to_sp2[i] == j).any() and (topk_sp2_to_sp1[:, j] == i).any():
                result[i, j] = 1 / k
    
    return result


def train(
    model,
    dataloader,
    expr_avg,
    homology_indices,
    device,
    optimizer,
    criterion,
    epochs=200,
    model_save_path=None,
    record_last_k_epochs=None,
):
    """
    Train the cross-species model for one stage.

    This function performs epoch-wise optimization over the provided dataloader,
    computes cross-species losses for all species pairs defined in
    ``homology_indices``, and saves the best model checkpoint according to the
    summed training loss.

    During training, the function can optionally record the cell-type embedding
    history from the last ``record_last_k_epochs`` epochs. These recorded
    cell-type embedding snapshots can be used for downstream consensus tree construction
    or stability analysis.

    Parameters
    ----------
    model : torch.nn.Module
        The cross-species model to train.
    dataloader : torch.utils.data.DataLoader
        Dataloader that yields one batch for each species.
    expr_avg : dict
        Dictionary containing species-level averaged expression and cell-type
        presence information. Typically includes fields such as
        ``expr_avg[species]["expr_avg"]`` and
        ``expr_avg[species]["cell_type_presence"]``.
    homology_indices : dict
        Mapping from species-pair tuples to homologous gene index pairs,
        e.g. ``(sp1, sp2) -> (sp1_indices, sp2_indices)``.
    device : torch.device
        Device used for model training.
    optimizer : torch.optim.Optimizer
        Optimizer used for parameter updates.
    criterion : callable
        Loss function object used to compute gene-level and cell-type-level loss.
    epochs : int, default=200
        Number of training epochs.
    model_save_path : str or pathlib.Path or None, default=None
        Path for saving the best model checkpoint. If None, the best model is
        not saved.
    record_last_k_epochs : int or None, default=None
        If not None, record cell-type embeddings from the last
        ``record_last_k_epochs`` epochs. If None, no cell-type embedding history is
        recorded.

    Returns
    -------
    cell_type_embedding_history : list[np.ndarray] or None
        If ``record_last_k_epochs`` is not None, returns a list of cell-type
        embedding snapshots from the last training epochs. Each element is
        typically a concatenated cell-type embedding array across species.
        If ``record_last_k_epochs`` is None, returns None.
    """
    model.train()
    min_loss = None
    combinations = list(homology_indices.keys())

    s = []
    for sp1, sp2 in combinations:
        sp1_gene_avg = expr_avg[sp1]["expr_avg"]
        sp1_cell_type = expr_avg[sp1]["cell_type_presence"]

        sp2_gene_avg = expr_avg[sp2]["expr_avg"]
        sp2_cell_type = expr_avg[sp2]["cell_type_presence"]

        sp1_ct_tensor = torch.tensor(sp1_cell_type, dtype=torch.bool)
        sp2_ct_tensor = torch.tensor(sp2_cell_type, dtype=torch.bool)

        sp1_indices, sp2_indices = homology_indices[(sp1, sp2)]
        sp1_indices = torch.tensor(sp1_indices)
        sp2_indices = torch.tensor(sp2_indices)

        sp1_gene_avg_selected = sp1_gene_avg[sp1_ct_tensor, :]
        sp2_gene_avg_selected = sp2_gene_avg[sp2_ct_tensor, :]

        if sp1_indices.numel() != 0:
            sp_ct_correlation = row_cosine_similarity(
                sp1_gene_avg_selected[:, sp1_indices],
                sp2_gene_avg_selected[:, sp2_indices]
            )
            sp_ct_correlation = process_tensor(sp_ct_correlation)

            s_truth = (
                F.normalize(sp1_gene_avg_selected, p=2, dim=0).T
                @ sp_ct_correlation
                @ F.normalize(sp2_gene_avg_selected, p=2, dim=0)
            )
        else:
            s_truth = torch.zeros(
                (sp1_gene_avg_selected.shape[-1], sp2_gene_avg_selected.shape[-1])
            )
        s.append(s_truth)

    cell_type_embedding_history = [] if record_last_k_epochs is not None else None

    for epoch in range(epochs):
        total_loss1 = 0.0
        total_loss2 = 0.0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            total_batch_loss = 0.0

            outputs = model(batch, device)
            for i, (sp1, sp2) in enumerate(combinations):
                sp1_cell_emb = outputs[sp1]["cell_embed"].to(device)
                sp1_cell_type_emb = outputs[sp1]["cell_type_embed"].to(device)
                sp1_gene_emb = outputs[sp1]["gene_embed"].to(device)
                sp1_labels = batch[sp1][1].to(device)
                sp1_cell_type = expr_avg[sp1]["cell_type_presence"]

                sp2_cell_emb = outputs[sp2]["cell_embed"].to(device)
                sp2_cell_type_emb = outputs[sp2]["cell_type_embed"].to(device)
                sp2_gene_emb = outputs[sp2]["gene_embed"].to(device)
                sp2_labels = batch[sp2][1].to(device)
                sp2_cell_type = expr_avg[sp2]["cell_type_presence"]

                sp1_gene_emb = F.normalize(sp1_gene_emb, p=2, dim=1)
                sp2_gene_emb = F.normalize(sp2_gene_emb, p=2, dim=1)
                s_pred = torch.mm(sp1_gene_emb, sp2_gene_emb.T)

                s_truth = s[i].to(device)

                sp1_ct_tensor = torch.tensor(sp1_cell_type, dtype=torch.bool, device=device)
                sp2_ct_tensor = torch.tensor(sp2_cell_type, dtype=torch.bool, device=device)

                loss1, loss2 = criterion(
                    sp1_cell_emb, sp2_cell_emb,
                    sp1_cell_type_emb, sp2_cell_type_emb,
                    sp1_labels, sp2_labels,
                    s_truth, s_pred,
                    sp1_ct_tensor, sp2_ct_tensor
                )

                total_batch_loss += (loss1 + loss2)
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()

            total_batch_loss.backward()
            optimizer.step()

        avg_loss1 = total_loss1 / len(dataloader)
        avg_loss2 = total_loss2 / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}] Gene loss: {avg_loss1:.4f}, Cell type loss: {avg_loss2:.4f}')

        if min_loss is None or min_loss > avg_loss1 + avg_loss2:
            min_loss = avg_loss1 + avg_loss2
            if model_save_path is not None:
                torch.save(model.state_dict(), model_save_path)
                print(f"Model has been saved to: {model_save_path}")

        if record_last_k_epochs is not None and epoch >= epochs - record_last_k_epochs:
            celltype_embeddings = []
            for species in expr_avg.keys():
                sp_mask = torch.tensor(expr_avg[species]["cell_type_presence"], dtype=torch.bool)
                type_embeddings = outputs[species]["cell_type_embed"][sp_mask]
                celltype_embeddings.append(type_embeddings.detach().cpu().numpy())

            cell_type_embedding_history.append(np.concatenate(celltype_embeddings, axis=0))

    return cell_type_embedding_history
