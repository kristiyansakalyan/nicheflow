import torch
from jaxtyping import Float
from torch import Tensor, nn

from nicheflow.models.backbones.pc_transformer import TimeEmbedding


class SinglePointMLP(nn.Module):
    """A feedforward MLP for modeling single-cell trajectories"""

    def __init__(
        self,
        pca_dim: int = 50,
        coord_dim: int = 2,
        ohe_dim: int = 3,
        time_emb_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 52,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.pca_dim = pca_dim

        # Embedding layers
        self.emb_x = nn.Linear(pca_dim, hidden_dim)
        self.emb_coord = nn.Linear(coord_dim, hidden_dim)
        self.emb_ohe = nn.Linear(ohe_dim, hidden_dim)

        self.time_embedding = TimeEmbedding(time_emb_dim=time_emb_dim, out_dim=hidden_dim)

        # [cond + target] + time
        concat_dim = 2 * (hidden_dim * 3) + hidden_dim

        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(concat_dim))
        layers += [
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Linear(concat_dim, output_dim),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_cond: Float[Tensor, "B N D_pca"],
        pos_cond: Float[Tensor, "B N D_coord"],
        ohe_cond: Float[Tensor, "B N D_ohe"],
        x_target: Float[Tensor, "B N D_pca"],
        pos_target: Float[Tensor, "B N D_coord"],
        ohe_target: Float[Tensor, "B N D_ohe"],
        t: Float[Tensor, "B"],
    ) -> tuple[Float[Tensor, "B N D_pca"], Float[Tensor, "B N D_coord"]]:
        # Condition embeddings
        x_emb_cond = self.emb_x(x_cond)
        coord_emb_cond = self.emb_coord(pos_cond)
        ohe_emb_cond = self.emb_ohe(ohe_cond)

        # Target embeddings
        x_emb_target = self.emb_x(x_target)
        coord_emb_target = self.emb_coord(pos_target)
        ohe_emb_target = self.emb_ohe(ohe_target)

        # Time embedding
        t_emb = self.time_embedding(t)
        t_emb = t_emb[:, None, :].expand(-1, x_emb_cond.size(1), -1)

        # Concatenate all embeddings
        z = torch.cat(
            [
                x_emb_cond,
                coord_emb_cond,
                ohe_emb_cond,
                x_emb_target,
                coord_emb_target,
                ohe_emb_target,
                t_emb,
            ],
            dim=-1,
        )

        out = self.mlp(z)
        return out[..., : self.pca_dim], out[..., self.pca_dim :]
