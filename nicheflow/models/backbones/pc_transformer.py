import torch
from jaxtyping import Float
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.LeakyReLU()

    def forward(
        self, x: Float[Tensor, "... {self.input_dim}"]
    ) -> Float[Tensor, "... {self.input_dim}"]:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ff_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(input_dim=embed_dim, hidden_dim=ff_hidden_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Float[Tensor, "B N_cells embed_dim"],
        mask: Float[Tensor, "B N_cells n_cells"] | None = None,
    ) -> Float[Tensor, "B N_cells embed_dim"]:
        key_padding_mask: Tensor | None = None
        if mask is not None:
            # The mask contains 1s for cells to be considered and 0s for cells
            # to be ignored. MultiHeadAttention expects True for tokens to be
            # ignored, and False for tokens to be kept. Therefore, we invert the mask.
            key_padding_mask = ~mask.bool()

        # Self attention
        attn_output, _ = self.attn.forward(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_output
        x = self.ln_1(x)

        # Feedforward
        x = self.ff(x)
        x = self.ln_2(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ff_hidden_dim: int,
    ) -> None:
        super().__init__()
        # Self attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_1 = nn.LayerNorm(embed_dim)

        # Cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

        # Feed forward
        self.ff = FeedForward(input_dim=embed_dim, hidden_dim=ff_hidden_dim)
        self.ln_3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Float[Tensor, "B N_cells_dec embed_dim"],
        enc_output: Float[Tensor, "B N_cells_enc embed_dim"],
        self_mask: Float[Tensor, "B N_cells_dec"] | None = None,
        cross_mask: Float[Tensor, "B N_cells_enc"] | None = None,
    ) -> Float[Tensor, "B N_cells_dec embed_dim"]:
        # The mask contains 1s for cells to be considered and 0s for cells
        # to be ignored. MultiHeadAttention expects True for tokens to be
        # ignored, and False for tokens to be kept. Therefore, we invert the mask.
        self_key_padding_mask = None
        if self_mask is not None:
            self_key_padding_mask = ~self_mask.bool()

        cross_key_padding_mask = None
        if cross_mask is not None:
            cross_key_padding_mask = ~cross_mask.bool()

        # Self attention + res. connection + layer norm
        attn_output, _ = self.self_attn(
            x, x, x, key_padding_mask=self_key_padding_mask, need_weights=False
        )
        x = x + attn_output
        x = self.ln_1(x)

        # Cross attention + res. connection + layer norm
        cross_attn_output, _ = self.cross_attn(
            x,
            enc_output,
            enc_output,
            key_padding_mask=cross_key_padding_mask,
            need_weights=False,
        )
        x = x + cross_attn_output
        x = self.ln_2(x)

        # Feed forward
        x = self.ff(x)
        x = self.ln_3(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int, out_dim: int) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.out_dim = out_dim
        self.out_linear = nn.Linear(2 * time_emb_dim, out_dim)

    def forward(self, t: Float[Tensor, "B ..."]) -> Float[Tensor, "B {self.out_dim}"]:
        if t.ndim == 1:
            t = t.unsqueeze(1)

        freqs = torch.arange(self.time_emb_dim, device=t.device, dtype=t.dtype)
        freqs = freqs.unsqueeze(0)
        t_freq = t * freqs
        # (B, 2 * D)
        t_emb = torch.cat([torch.cos(t_freq), torch.sin(t_freq)], dim=-1)
        return self.out_linear(t_emb)


class PointCloudTransformer(nn.Module):
    def __init__(
        self,
        pca_dim: int,
        ohe_dim: int,
        coord_dim: int,
        output_dim: int,
        embed_dim: int = 128,
        ff_hidden_dim: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pca_dim = pca_dim
        self.ohe_dim = ohe_dim

        # Recalculate the embedding dimension
        embed_dim = (3 * num_heads) * (embed_dim // (3 * num_heads))
        # Split it equally for gene expressions, time and positions
        concat_dim = embed_dim // 3

        self.x_emb = nn.Linear(pca_dim + ohe_dim, concat_dim)
        self.pos_emb = nn.Linear(coord_dim, concat_dim)
        self.time_emb = TimeEmbedding(embed_dim, concat_dim)

        self.enc_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_hidden_dim=ff_hidden_dim,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.out_linear = nn.Linear(embed_dim, output_dim)

    def forward(
        self,
        pcs_cond: Float[Tensor, "B N_points_t1 D_in"],
        pos_cond: Float[Tensor, "B N_points_t1 D_coord"],
        ohe_cond: Float[Tensor, "B N_points_t1 D_ohe"],
        pcs_target: Float[Tensor, "B N_points_t2 D_in"],
        pos_target: Float[Tensor, "B N_points_t2 D_coord"],
        ohe_target: Float[Tensor, "B N_points_t1 D_ohe"],
        t_target: Float[Tensor, "B ..."],
        mask_condition: Float[Tensor, "B N_points_t1"] | None = None,
        mask_target: Float[Tensor, "B N_points_t2"] | None = None,
    ) -> tuple[Float[Tensor, "B N_points_t2 D_in"], Float[Tensor, "B N_points_t2 D_coord"]]:
        # Concatenate the timestep one hot encoding to the PCs
        pcs_cond = torch.cat([pcs_cond, ohe_cond], dim=-1)
        pcs_target = torch.cat([pcs_target, ohe_target], dim=-1)

        # Embed condition
        x_cond = self.x_emb(pcs_cond)
        pos_cond = self.pos_emb(pos_cond)
        t_cond = torch.zeros_like(x_cond, device=x_cond.device, dtype=x_cond.dtype)

        # Encode
        enc_output = torch.cat([x_cond, t_cond, pos_cond], dim=-1)
        for block in self.enc_blocks:
            enc_output = block(x=enc_output, mask=mask_condition)

        # Embed target
        x_target = self.x_emb(pcs_target)
        pos_target = self.pos_emb(pos_target)
        t_target = self.time_emb(t_target)[:, None, :].expand(-1, x_target.size(1), -1)

        # Decode
        dec_output = torch.cat([x_target, t_target, pos_target], dim=-1)
        for block in self.dec_blocks:
            dec_output = block(
                x=dec_output,
                enc_output=enc_output,
                self_mask=mask_target,
                cross_mask=mask_condition,
            )

        out = self.out_linear(dec_output)

        x_pred = out[:, :, : self.pca_dim]
        pos_pred = out[:, :, self.pca_dim :]
        return x_pred, pos_pred
