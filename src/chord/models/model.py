import torch
import torch.nn as nn


class SpeciesEncoder(nn.Module):
    def __init__(self, cell_type_presence, num_genes, embed_dim):
        super().__init__()
        # Learnable gene embedding matrix of shape [num_genes, embed_dim]
        self.gene_embedding = nn.Parameter(torch.randn(num_genes, embed_dim))

        # Learnable global cell-type prototype matrix of shape
        # [num_global_cell_types, embed_dim]
        num_global_ct = len(cell_type_presence)
        self.prototype = nn.Parameter(torch.randn(num_global_ct, embed_dim))

        # Non-trainable mask indicating which global cell types
        # are present in the current species
        self.register_buffer(
            'ct_mask',
            torch.tensor(cell_type_presence, dtype=torch.float32)
        )

        # Zero out prototypes for cell types that are absent in this species
        with torch.no_grad():
            self.prototype.data *= self.ct_mask.unsqueeze(1)  # [N_global, D]

    def forward(self, expr, mask=True):
        """
        Encode expression profiles into cell embeddings and return the
        masked prototype matrix for the current species.

        Args:
            expr:
                Expression matrix of shape [B, num_genes].
            mask:
                Whether to apply random element-wise masking noise to
                the input expression matrix.

        Returns:
            cell_emb:
                Cell embedding matrix of shape [B, embed_dim].
            masked_prototype:
                Prototype matrix of shape [num_global_cell_types, embed_dim],
                with absent cell-type prototypes zeroed out.
        """
        def add_mask_noise(x, mask_prob=0.1):
            # Randomly set a fraction of expression entries to zero
            mask = (torch.rand(x.shape, device=x.device) > mask_prob).float()
            return x * mask

        if mask:
            expr = add_mask_noise(expr)

        # Apply the species-specific cell-type presence mask to prototypes
        masked_prototype = self.prototype * self.ct_mask.unsqueeze(1)  # [N_global, D]

        # Compute cell embeddings by a linear projection from gene space
        cell_emb = torch.matmul(expr, self.gene_embedding)  # [B, E]

        return cell_emb, masked_prototype


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        # Two-layer projection MLP with LayerNorm, ReLU, and dropout
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Project input features through the MLP.

        Args:
            x:
                Input tensor of shape [..., input_dim].

        Returns:
            Output tensor of shape [..., output_dim].
        """
        return self.mlp(x)


class CrossSpeciesPrototypeModel(nn.Module):
    def __init__(self, species_config, embed_dim, hidden_dim, project_dim):
        super().__init__()
        # Build one encoder per species
        self.species_encoders = nn.ModuleDict({
            sp: SpeciesEncoder(config['cell_type_presence'], config['num_genes'], embed_dim)
            for sp, config in species_config.items()})
        
        # Separate projection heads for cell embeddings and cell-type prototypes
        self.celltype_mlp = MLP(embed_dim, hidden_dim, project_dim)
        self.cell_mlp = MLP(embed_dim, hidden_dim, project_dim)
                    
    def forward(self, species_inputs, device, mask=True):
        """
        Run species-specific encoders and project both cell embeddings
        and cell-type prototypes into a shared latent space.

        Args:
            species_inputs:
                Dictionary mapping species name to its model inputs.
                The first element, `data[0]`, is expected to be the
                expression tensor for that species.
            device:
                Target device for input tensors.
            mask:
                Whether to enable random masking in each species encoder.

        Returns:
            outputs:
                Dictionary of the form:
                {
                    species_name: {
                        "cell_embed": projected cell embeddings,
                        "cell_type_embed": projected prototype embeddings,
                        "gene_embed": species-specific gene embedding matrix,
                    },
                    ...
                }
        """
        outputs = {}

        for sp, (x, _ ) in species_inputs.items():
            cell, cell_type = self.species_encoders[sp](x.to(device), mask)
            
            # Project cell embeddings and prototype embeddings separately
            cell = self.cell_mlp(cell)
            cell_type = self.celltype_mlp(cell_type)

            outputs[sp] = {
                "cell_embed": cell,
                "cell_type_embed": cell_type,
                "gene_embed": self.species_encoders[sp].gene_embedding
            }
        return outputs