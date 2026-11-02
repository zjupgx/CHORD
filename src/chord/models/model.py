import torch
import torch.nn as nn


class SpeciesEncoder(nn.Module):
    def __init__(self, cell_type_presence, num_genes, embed_dim):
        super().__init__()
        self.gene_embedding = nn.Parameter(torch.randn(num_genes, embed_dim))

        num_global_ct = len(cell_type_presence)
        self.prototype = nn.Parameter(torch.randn(num_global_ct, embed_dim))

        self.register_buffer(
            'ct_mask',
            torch.tensor(cell_type_presence, dtype=torch.float32)
        )

        with torch.no_grad():
            self.prototype.data *= self.ct_mask.unsqueeze(1)

    def forward(self, expr, mask=True, mask_prob=0.1):
        def add_mask_noise(x, probability):
            random_mask = (torch.rand(x.shape, device=x.device) > probability).float()
            return x * random_mask

        if mask:
            if isinstance(mask_prob, bool) or not 0.0 <= float(mask_prob) <= 1.0:
                raise ValueError("mask_prob must be between 0 and 1.")
            if float(mask_prob) > 0.0:
                expr = add_mask_noise(expr, float(mask_prob))

        masked_prototype = self.prototype * self.ct_mask.unsqueeze(1)

        cell_emb = torch.matmul(expr, self.gene_embedding)

        return cell_emb, masked_prototype


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class CrossSpeciesPrototypeModel(nn.Module):
    def __init__(self, species_config, embed_dim, hidden_dim, project_dim):
        super().__init__()
        self.species_encoders = nn.ModuleDict({
            sp: SpeciesEncoder(config['cell_type_presence'], config['num_genes'], embed_dim)
            for sp, config in species_config.items()})
        
        self.celltype_mlp = MLP(embed_dim, hidden_dim, project_dim)
        self.cell_mlp = MLP(embed_dim, hidden_dim, project_dim)

    def initialize_prototypes_from_expression_means(self, species_expr_avg):
        expected_species = set(self.species_encoders.keys())
        provided_species = set(species_expr_avg.keys())
        if provided_species != expected_species:
            missing = sorted(expected_species - provided_species)
            unexpected = sorted(provided_species - expected_species)
            raise ValueError(
                "Mean prototype initialization species mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )

        with torch.no_grad():
            for species, encoder in self.species_encoders.items():
                entry = species_expr_avg[species]
                if not isinstance(entry, dict) or "expr_avg" not in entry:
                    raise ValueError(
                        f"[{species}] mean prototype initialization requires "
                        "an 'expr_avg' matrix."
                    )

                mean_expression = torch.as_tensor(
                    entry["expr_avg"],
                    dtype=encoder.gene_embedding.dtype,
                    device=encoder.gene_embedding.device,
                )
                expected_shape = (
                    encoder.prototype.shape[0],
                    encoder.gene_embedding.shape[0],
                )
                if tuple(mean_expression.shape) != expected_shape:
                    raise ValueError(
                        f"[{species}] mean-expression shape "
                        f"{tuple(mean_expression.shape)} does not match "
                        f"{expected_shape}."
                    )
                if not torch.isfinite(mean_expression).all():
                    raise ValueError(
                        f"[{species}] mean expression contains non-finite values."
                    )

                mean_prototypes = mean_expression @ encoder.gene_embedding
                mean_prototypes *= encoder.ct_mask.unsqueeze(1)
                encoder.prototype.copy_(mean_prototypes)

    def initialize_prototypes_from_random_cells(self, species_data, seed):
        expected_species = set(self.species_encoders.keys())
        provided_species = set(species_data.keys())
        if provided_species != expected_species:
            missing = sorted(expected_species - provided_species)
            unexpected = sorted(provided_species - expected_species)
            raise ValueError(
                "Random-cell prototype initialization species mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(
                "Random-cell prototype initialization seed must be a "
                "non-negative integer."
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        with torch.no_grad():
            for species, encoder in self.species_encoders.items():
                inputs = species_data[species]
                if not isinstance(inputs, (tuple, list)) or len(inputs) < 2:
                    raise ValueError(
                        f"[{species}] random-cell prototype initialization "
                        "requires expression and encoded-label tensors."
                    )

                expression = torch.as_tensor(inputs[0])
                labels = torch.as_tensor(
                    inputs[1], dtype=torch.long, device="cpu"
                )
                if expression.ndim != 2:
                    raise ValueError(
                        f"[{species}] expression must be a two-dimensional matrix."
                    )
                if labels.ndim != 1 or labels.shape[0] != expression.shape[0]:
                    raise ValueError(
                        f"[{species}] encoded labels must contain one value per cell."
                    )
                if expression.shape[1] != encoder.gene_embedding.shape[0]:
                    raise ValueError(
                        f"[{species}] expression has {expression.shape[1]} genes, "
                        "but the model expects "
                        f"{encoder.gene_embedding.shape[0]}."
                    )

                present_type_indices = torch.nonzero(
                    encoder.ct_mask.detach().cpu() > 0,
                    as_tuple=False,
                ).flatten()
                selected_positions = []
                for type_index in present_type_indices.tolist():
                    candidates = torch.nonzero(
                        labels == type_index, as_tuple=False
                    ).flatten()
                    if candidates.numel() == 0:
                        raise ValueError(
                            f"[{species}] cell type index {type_index} is marked "
                            "present but has no training cells."
                        )
                    selected_offset = int(
                        torch.randint(
                            candidates.numel(),
                            (1,),
                            generator=generator,
                        ).item()
                    )
                    selected_positions.append(
                        int(candidates[selected_offset].item())
                    )

                selected_expression = torch.as_tensor(
                    expression[selected_positions],
                    dtype=encoder.gene_embedding.dtype,
                    device=encoder.gene_embedding.device,
                )
                if not torch.isfinite(selected_expression).all():
                    raise ValueError(
                        f"[{species}] selected expression contains non-finite values."
                    )

                selected_prototypes = (
                    selected_expression @ encoder.gene_embedding
                )
                target_indices = present_type_indices.to(
                    device=encoder.prototype.device
                )
                encoder.prototype.zero_()
                encoder.prototype.index_copy_(
                    0, target_indices, selected_prototypes
                )
                    
    def forward(self, species_inputs, device, mask=True, mask_prob=0.1):
        outputs = {}

        for sp, (x, _ ) in species_inputs.items():
            cell, cell_type = self.species_encoders[sp](
                x.to(device), mask=mask, mask_prob=mask_prob
            )
            
            cell = self.cell_mlp(cell)
            cell_type = self.celltype_mlp(cell_type)

            outputs[sp] = {
                "cell_embed": cell,
                "cell_type_embed": cell_type,
                "gene_embed": self.species_encoders[sp].gene_embedding
            }
        return outputs
