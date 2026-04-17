import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSpeciesPrototypeLoss(nn.Module):
    def __init__(self, weight, temperature=0.1):
        super().__init__()
        # Temperature used to scale prototype-classification logits
        self.temperature = temperature
        # Loss weights, typically [gene_mse_weight, sp1_ce_weight, sp2_ce_weight]
        self.weight = weight

    def forward(
        self,
        sp1_c,
        sp2_c,
        sp1_ct,
        sp2_ct,
        sp1_labels,
        sp2_labels,
        s_truth,
        s_pred,
        sp1_ct_tensor,
        sp2_ct_tensor,
    ):
        """
        Compute the weighted loss terms for the cross-species model.

        This forward pass contains two parts:
        1. A gene-level regression loss (MSE) between `s_pred` and `s_truth`.
        2. Species-specific prototype classification losses based on
           cell-to-prototype similarities.

        Args:
            sp1_c:
                Cell embeddings for species 1.
            sp2_c:
                Cell embeddings for species 2.
            sp1_ct:
                Cell-type prototype embeddings for species 1.
            sp2_ct:
                Cell-type prototype embeddings for species 2.
            sp1_labels:
                Encoded cell-type labels for species 1.
            sp2_labels:
                Encoded cell-type labels for species 2.
            s_truth:
                Ground-truth target for the gene-level regression term.
            s_pred:
                Predicted target for the gene-level regression term.
            sp1_ct_tensor:
                Boolean mask indicating which prototype columns are valid
                for species 1.
            sp2_ct_tensor:
                Boolean mask indicating which prototype columns are valid
                for species 2.

        Returns:
            A tuple of two weighted loss components:
            - weighted gene-level MSE loss
            - weighted sum of the two species-specific cross-entropy losses
        """
        # Unpack weights for the three loss terms
        a, b, c = self.weight[0], self.weight[1], self.weight[2]

        # Gene-level regression loss
        mse_loss_fn = torch.nn.MSELoss()
        loss1 = mse_loss_fn(s_pred, s_truth)

        
        # Prototype-based cell-type classification per species
        # L2-normalize embeddings so dot products become cosine similarities
        sp1_c = F.normalize(sp1_c, p=2, dim=1)
        sp2_c = F.normalize(sp2_c, p=2, dim=1)
        sp1_ct = F.normalize(sp1_ct, p=2, dim=1)
        sp2_ct = F.normalize(sp2_ct, p=2, dim=1)

        # Compute logits from cell embeddings to cell-type prototypes
        # Shape: [batch_size, num_global_cell_types]
        similarity1 = torch.matmul(sp1_c, sp1_ct.T) / self.temperature
        # Mask prototype columns that are not valid for species 1
        similarity1[:, ~sp1_ct_tensor] = -1e9

        similarity2 = torch.matmul(sp2_c, sp2_ct.T) / self.temperature
        # Mask prototype columns that are not valid for species 2
        similarity2[:, ~sp2_ct_tensor] = -1e9

        # Species-specific prototype classification losses
        loss2 = F.cross_entropy(similarity1, sp1_labels)
        loss3 = F.cross_entropy(similarity2, sp2_labels)

        return a * loss1, b * loss2 + c * loss3