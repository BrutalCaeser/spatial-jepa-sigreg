"""
JEPAPredictor — 6-layer transformer with Adaptive LayerNorm (AdaLN) conditioning.

Definition 1.3 (foundations.md):
    P_phi(z_c, y) = Transformer_AdaLN(z_c, Embed(y))

    - z_c in R^{B x N x d}: adapted context patch tokens
    - y in Z^B: clip-level action label (0..173 for SSv2 174 classes)
    - Embed: Z -> R^{d_action}: learned embedding lookup
    - AdaLN: action embedding modulates LayerNorm scale/shift at each layer
    - Output: z_hat in R^{B x N x d}

Architecture (build_spec.md Section 2.4):
    n_layers = 6, n_heads = 8, d_inner = 2*d (FFN hidden dim)
    d_action = 64 (action embedding dim), dropout = 0.1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """Adaptive LayerNorm: scale and shift conditioned on an external embedding.

    AdaLN(x, e) = gamma(e) * LayerNorm(x) + beta(e)

    where gamma, beta are linear projections of the conditioning embedding e.

    Args:
        d:       Feature dimension of x.
        d_cond:  Dimension of conditioning embedding e.
    """

    def __init__(self, d: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        # Single linear: output [2*d] -> split into gamma [d] and beta [d]
        self.proj = nn.Linear(d_cond, 2 * d)
        # Initialize projection near zero so conditioning starts neutral.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Apply AdaLN.

        Args:
            x: Input tensor, shape [B, N, d].
            e: Conditioning embedding, shape [B, d_cond].

        Returns:
            Normalized and conditioned tensor, shape [B, N, d].
        """
        gamma_beta = self.proj(e)               # [B, 2*d]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, d] each
        # Expand for broadcasting over N tokens.
        gamma = gamma.unsqueeze(1)              # [B, 1, d]
        beta  = beta.unsqueeze(1)               # [B, 1, d]
        return (1.0 + gamma) * self.norm(x) + beta


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm AdaLN and action conditioning.

    Structure:
        x = x + Attention(AdaLN1(x, e))
        x = x + FFN(AdaLN2(x, e))

    Args:
        d:       Token embedding dimension.
        n_heads: Number of attention heads.
        d_inner: FFN hidden dimension (2*d by default).
        d_cond:  Conditioning embedding dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d: int,
        n_heads: int,
        d_inner: int,
        d_cond: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"

        # AdaLN for attention and FFN branches.
        self.adaln1 = AdaLN(d, d_cond)
        self.adaln2 = AdaLN(d, d_cond)

        # Multi-head self-attention.
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # expects [B, N, d]
        )
        self.attn_drop = nn.Dropout(dropout)

        # Feed-forward network.
        self.ffn = nn.Sequential(
            nn.Linear(d, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token embeddings, shape [B, N, d].
            e: Action conditioning, shape [B, d_cond].

        Returns:
            Updated token embeddings, shape [B, N, d].
        """
        # Attention branch: pre-norm with AdaLN.
        x_norm = self.adaln1(x, e)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.attn_drop(attn_out)

        # FFN branch: pre-norm with AdaLN.
        x_norm = self.adaln2(x, e)
        x = x + self.ffn(x_norm)

        return x


class JEPAPredictor(nn.Module):
    """Patch-level JEPA predictor: transformer conditioned on action label.

    Definition 1.3 (foundations.md):
        P_phi(z_c, y) = Transformer_AdaLN(z_c, Embed(y))

    Takes adapted context patch tokens and an action label, predicts the
    adapted target patch tokens.

    Args:
        d:          Token dimension (adapter output dimension).
        n_layers:   Number of transformer blocks (default 6).
        n_heads:    Number of attention heads (default 8).
        n_classes:  Number of SSv2 action classes (default 174).
        d_action:   Action embedding dimension (default 64).
        dropout:    Dropout probability (default 0.1).
    """

    def __init__(
        self,
        d: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_classes: int = 174,
        d_action: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_inner = 2 * d

        # Action label embedding: y -> R^{d_action}
        self.action_embed = nn.Embedding(n_classes, d_action)
        nn.init.normal_(self.action_embed.weight, std=0.02)

        # Project d_action -> d_cond used for AdaLN conditioning across all layers.
        # Each layer receives the same conditioning vector.
        self.cond_proj = nn.Linear(d_action, d_action)

        # Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d=d,
                n_heads=n_heads,
                d_inner=d_inner,
                d_cond=d_action,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final LayerNorm before output projection.
        self.norm_out = nn.LayerNorm(d)

        # Output projection (identity-like initialization).
        self.proj_out = nn.Linear(d, d)
        nn.init.xavier_normal_(self.proj_out.weight, gain=0.1)
        nn.init.zeros_(self.proj_out.bias)

        # Learnable positional encoding for N=196 patch tokens.
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z_c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict target representations from context.

        Args:
            z_c: Adapted context patch tokens, shape [B, N, d].
            y:   Action class labels, shape [B], values in [0, n_classes-1].

        Returns:
            z_hat: Predicted target patch tokens, shape [B, N, d].
        """
        B, N, d = z_c.shape

        # Add positional encoding (truncate/expand if N != 196).
        pos = self.pos_embed[:, :N, :]  # [1, N, d]
        x = z_c + pos

        # Build action conditioning vector shared across all transformer blocks.
        e = self.action_embed(y)        # [B, d_action]
        e = self.cond_proj(e)           # [B, d_action]

        # Pass through transformer blocks.
        for block in self.blocks:
            x = block(x, e)

        # Final normalization and projection.
        x = self.norm_out(x)            # [B, N, d]
        z_hat = self.proj_out(x)        # [B, N, d]

        return z_hat
