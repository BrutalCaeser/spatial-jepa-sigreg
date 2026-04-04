"""
PatchAdapter — token-wise MLP: R^{D_in} -> R^{D_out}

Definition 1.2 (foundations.md):
    A_theta(f) = W2 * GELU(W1 * f + b1) + b2
    W1 in R^{D_in x D_in}, W2 in R^{D_in x D_out}

Critical constraints (CLAUDE.md Rule 4):
    - NO BatchNorm or LayerNorm in forward(): would mask collapse signals.
    - NO torch.compile / torch.jit: can hide gradient flow issues.
    - Initialization gain is configurable (default 0.1 for near-identity).

Note on BatchNorm: LeWorldModel (Maes et al. 2026) shows BN on the projector
is critical for SIGReg. When use_sigreg_bn=True in the loss config, BN is
applied OUTSIDE this module (in compute_loss) before SIGReg only. Metrics
are always computed on raw adapter output (no BN) to preserve collapse signals.
"""

import torch
import torch.nn as nn


class PatchAdapter(nn.Module):
    """Token-wise MLP adapter over frozen V-JEPA 2.1 patch features.

    Applied independently to each (batch, patch) token via shared weights.
    Shared weights across spatial positions allow collapse to occur (Theorem 2.1).

    Args:
        D_in:  Input feature dimension (1024 for V-JEPA 2.1 ViT-G).
        D_out: Output representation dimension (256 default, ablate 128/512).
        init_gain: Xavier gain for fc2 (output layer) initialization.
            0.1 = near-identity init (original; output std ≈ 0.2)
            1.0 = random init (output std ≈ 1.0, closer to N(0,I))
            Both papers (LeJEPA, LeWorldModel) use random init.
    """

    def __init__(self, D_in: int = 1024, D_out: int = 256, init_gain: float = 0.1):
        super().__init__()
        # Two-layer MLP: expand (D_in->D_in), then project (D_in->D_out).
        # NO normalization layers — collapse must be observable in outputs.
        self.fc1 = nn.Linear(D_in, D_in)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(D_in, D_out)

        self._init_gain = init_gain
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize adapter weights.

        fc1: W1 ~= I (identity), b1 = 0  (preserve input signal early in training)
        fc2: W2 ~ xavier_normal(gain=self._init_gain), b2 = 0
             gain=0.1 → near-identity (small initial perturbation)
             gain=1.0 → random init (output scale ≈ 1, closer to N(0,1) target)
        """
        # fc1: eye_ requires square matrix — D_in x D_in, so this works directly.
        nn.init.eye_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # fc2: xavier_normal with configurable gain.
        nn.init.xavier_normal_(self.fc2.weight, gain=self._init_gain)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply token-wise MLP independently to each patch.

        Args:
            f: Frozen V-JEPA 2.1 features, shape [B, N, D_in].
               B = batch size, N = 196 patch tokens (14x14), D_in = 1024.

        Returns:
            z: Adapted representations, shape [B, N, D_out].
        """
        # Applied token-wise: same fc1/fc2 weights applied to every (b, n) token.
        # Equivalent to reshaping [B*N, D_in], applying Linear, reshaping back.
        return self.fc2(self.act(self.fc1(f)))
