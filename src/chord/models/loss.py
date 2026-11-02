import math
from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_loss_weight(weight, *, prefix="loss_weight"):
    if not isinstance(weight, (list, tuple)) or len(weight) != 3:
        raise ValueError(
            f"{prefix} must contain exactly three weights: "
            "[lambda_g, lambda_c, lambda_c]."
        )
    for index, value in enumerate(weight):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{prefix}[{index}] must be a finite number; received {value!r}.")
    if weight[1] != weight[2]:
        raise ValueError(
            f"{prefix}[1] and {prefix}[2] must be equal (lambda_c); "
            f"received {weight[1]!r} and {weight[2]!r}."
        )


class CrossSpeciesPrototypeLoss(nn.Module):
    def __init__(self, weight, temperature=0.1):
        super().__init__()
        validate_loss_weight(weight)
        self.temperature = temperature
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
        a, b, c = self.weight[0], self.weight[1], self.weight[2]

        mse_loss_fn = torch.nn.MSELoss()
        loss1 = mse_loss_fn(s_pred, s_truth)

        
        sp1_c = F.normalize(sp1_c, p=2, dim=1)
        sp2_c = F.normalize(sp2_c, p=2, dim=1)
        sp1_ct = F.normalize(sp1_ct, p=2, dim=1)
        sp2_ct = F.normalize(sp2_ct, p=2, dim=1)

        similarity1 = torch.matmul(sp1_c, sp1_ct.T) / self.temperature
        similarity1[:, ~sp1_ct_tensor] = -1e9

        similarity2 = torch.matmul(sp2_c, sp2_ct.T) / self.temperature
        similarity2[:, ~sp2_ct_tensor] = -1e9

        loss2 = F.cross_entropy(similarity1, sp1_labels)
        loss3 = F.cross_entropy(similarity2, sp2_labels)

        return a * loss1, b * loss2 + c * loss3
