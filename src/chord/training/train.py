import torch
import torch.nn.functional as F


def row_cosine_similarity(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    assert matrix1.dim() == 2 and matrix2.dim() == 2, "Inputs must be 2D tensors"
    assert matrix1.size(1) == matrix2.size(1), "Feature dimensions must match"
    
    norm1 = torch.norm(matrix1, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(matrix2, p=2, dim=1, keepdim=True)
    
    eps = 1e-8
    norm1 = torch.clamp(norm1, min=eps)
    norm2 = torch.clamp(norm2, min=eps)
    
    normalized_matrix1 = matrix1 / norm1
    normalized_matrix2 = matrix2 / norm2
    
    similarity = torch.matmul(normalized_matrix1, normalized_matrix2.transpose(0, 1))
    
    return similarity


def process_tensor(sp_ct_correlation, k=2):
    effective_k = min(k, sp_ct_correlation.shape[0], sp_ct_correlation.shape[1])
    if effective_k == 0:
        return torch.zeros_like(sp_ct_correlation)

    _, topk_sp1_to_sp2 = torch.topk(
        sp_ct_correlation, k=effective_k, dim=1
    )
    _, topk_sp2_to_sp1 = torch.topk(
        sp_ct_correlation, k=effective_k, dim=0
    )
    
    result = torch.zeros_like(sp_ct_correlation)
    for i in range(sp_ct_correlation.shape[0]):
        for j in range(sp_ct_correlation.shape[1]):
            if (topk_sp1_to_sp2[i] == j).any() and (topk_sp2_to_sp1[:, j] == i).any():
                result[i, j] = 1 / effective_k
    
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
    *,
    gene_masking=True,
    gene_mask_probability=0.1,
):
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

    for epoch in range(epochs):
        total_loss1 = 0.0
        total_loss2 = 0.0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            total_batch_loss = 0.0

            if gene_masking is True and float(gene_mask_probability) == 0.1:
                outputs = model(batch, device)
            else:
                outputs = model(
                    batch,
                    device,
                    mask=gene_masking,
                    mask_prob=gene_mask_probability,
                )
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

