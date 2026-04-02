"""
All loss functions for GAP 1 experiment conditions.

Definitions (foundations.md):
    3.1  L_pred          — MSE prediction loss
    3.4  L_info (pooled) — cross-covariance trace, spatially pooled
    3.5  L_info_dense    — cross-covariance trace at each patch position
    3.6  L_cov           — covariance regularization (optional stabilizer)
    3.7  Combined loss   — dispatched by compute_loss() per condition config

CRITICAL RULES (CLAUDE.md):
    Rule 1: SIGReg is applied to ADAPTER output z_c, NEVER predictor output z_hat.
    Rule 2: L_info must involve z_hat (has gradient), not two frozen/detached tensors.
    Rule 3: Stop-gradient is z_t = adapter(f_t).detach(), not f_t.detach().
    Rule 5: Conditions A and F use weight_decay=0 in the optimizer (enforced in trainer).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from models.sigreg import apply_sigreg


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def l_pred(z_hat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Prediction loss: MSE between predicted and target representations.

    Definition 3.1 (foundations.md):
        L_pred = (1 / (B*N*d)) * sum_{b,n,j} (z_hat[b,n,j] - z_t[b,n,j])^2

    Equivalent to F.mse_loss with default 'mean' reduction.

    Args:
        z_hat: Predictor output, shape [B, N, d].  Always has gradient.
        z_t:   Target adapter output, shape [B, N, d].
               May be detached (stop-grad conditions) or live (no stop-grad).

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(z_hat, z_t)


def l_info_dense(z_hat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Dense cross-covariance information loss (preserves spatial structure).

    Definition 3.5 (foundations.md):
        For each patch position n in {0, ..., N-1}:
            Center per patch:
                z_hat_c[b,n,:] = z_hat[b,n,:] - mean_b(z_hat[:,n,:])
                z_t_c[b,n,:]   = z_t[b,n,:]   - mean_b(z_t[:,n,:])
            C_n = (1/B) * z_hat_c[:,n,:]^T @ z_t_c[:,n,:]   in R^{d x d}

        L_info_dense = -(1/N) * sum_n Tr(C_n)

    Computes cross-covariance AT EACH SPATIAL POSITION independently, preserving
    per-patch alignment. Negative sign: minimizing loss = maximizing Tr(C).

    Rule 2: z_hat MUST have gradient (it is always the predictor output).

    Args:
        z_hat: Predictor output, shape [B, N, d].  MUST have gradient.
        z_t:   Target adapter output, shape [B, N, d].

    Returns:
        Scalar loss <= 0 (more negative = better alignment).
    """
    B, N, d = z_hat.shape

    # Center each patch token across the batch dimension independently.
    z_hat_c = z_hat - z_hat.mean(dim=0, keepdim=True)  # [B, N, d]
    z_t_c   = z_t   - z_t.mean(dim=0, keepdim=True)    # [B, N, d]

    # Cross-covariance trace summed across patch positions.
    # C_n[i,j] = (1/B) * sum_b z_hat_c[b,n,i] * z_t_c[b,n,j]
    # Tr(C_n) = (1/B) * sum_b sum_j z_hat_c[b,n,j] * z_t_c[b,n,j]
    #         = (1/B) * sum_b (z_hat_c[b,n,:] · z_t_c[b,n,:])
    #
    # Efficient: element-wise product then sum over d and b.
    # xcov_per_patch: [N] — trace of C_n for each patch position n.
    xcov_per_patch = (z_hat_c * z_t_c).sum(dim=-1).mean(dim=0)  # [N]

    # Average over patch positions and negate (loss = -Tr).
    return -xcov_per_patch.mean()


def l_info_pooled(z_hat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Pooled cross-covariance information loss (collapses spatial dimension).

    Definition 3.4 (foundations.md):
        z_hat_pool = mean(z_hat, dim=1)  in R^{B x d}
        z_t_pool   = mean(z_t,   dim=1)  in R^{B x d}

        Center:
            z_hat_c = z_hat_pool - mean_b(z_hat_pool)   in R^{B x d}
            z_t_c   = z_t_pool   - mean_b(z_t_pool)     in R^{B x d}

        C = (1/B) * z_hat_c^T @ z_t_c    in R^{d x d}
        L_info = -Tr(C)

    This version loses per-patch spatial information by averaging tokens first.
    Used in condition E_pooled (alternative to E_dense).

    Args:
        z_hat: Predictor output, shape [B, N, d].
        z_t:   Target adapter output, shape [B, N, d].

    Returns:
        Scalar loss <= 0.
    """
    # Pool over patch tokens.
    z_hat_pool = z_hat.mean(dim=1)      # [B, d]
    z_t_pool   = z_t.mean(dim=1)        # [B, d]

    # Center across batch.
    z_hat_c = z_hat_pool - z_hat_pool.mean(dim=0, keepdim=True)  # [B, d]
    z_t_c   = z_t_pool   - z_t_pool.mean(dim=0, keepdim=True)    # [B, d]

    # Cross-covariance trace: Tr((1/B) * z_hat_c^T @ z_t_c)
    # = (1/B) * sum_{b,j} z_hat_c[b,j] * z_t_c[b,j]
    xcov = (z_hat_c * z_t_c).sum(dim=-1).mean(dim=0)   # scalar

    return -xcov


def l_cov(z_hat: torch.Tensor) -> torch.Tensor:
    """Covariance regularization (optional training stabilizer).

    Definition 3.6 (foundations.md):
        z_pool   = mean(z_hat, dim=1)         in R^{B x d}
        z_c      = z_pool - mean_b(z_pool)
        C        = (1/B) * z_c^T @ z_c        in R^{d x d}
        L_cov    = ||C - I||_F^2

    Penalizes deviation from identity covariance. Used only if training is
    unstable (FM4: NaN/Inf in L_info_dense with small batch size).

    Args:
        z_hat: Predictor output, shape [B, N, d].

    Returns:
        Scalar non-negative loss. 0 when covariance equals identity.
    """
    B, N, d = z_hat.shape
    z_pool = z_hat.mean(dim=1)                           # [B, d]
    z_c    = z_pool - z_pool.mean(dim=0, keepdim=True)  # [B, d]
    C      = (z_c.T @ z_c) / B                          # [d, d]
    I      = torch.eye(d, device=z_hat.device, dtype=z_hat.dtype)
    return ((C - I) ** 2).sum()


# ---------------------------------------------------------------------------
# Master loss dispatcher
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    """Configuration for compute_loss — read from YAML per condition."""
    # Stop-gradient on z_t (Conditions B, C only).
    stop_grad: bool = False
    # SIGReg weight. 0.0 disables SIGReg entirely.
    lambda_1: float = 0.0
    # L_info weight. 0.0 disables information loss entirely.
    lambda_2: float = 0.0
    # L_cov weight. 0.0 disables covariance regularization.
    lambda_3: float = 0.0
    # SIGReg axis: 'global', 'token', or 'channel'.
    sigreg_axis: str = "global"
    # If True, use L_info_dense; if False, use L_info_pooled.
    use_dense_info: bool = True
    # SIGReg hyperparameters.
    M_projections: int = 1024
    T_knots: int = 17
    bandwidth: float = 1.0


def compute_loss(
    z_c: torch.Tensor,
    z_t: torch.Tensor,
    z_hat: torch.Tensor,
    config: LossConfig,
) -> Dict[str, torch.Tensor]:
    """Compute the combined loss for one training step.

    This function is the single source of truth for all condition-specific
    loss logic. It enforces all critical rules from CLAUDE.md.

    CRITICAL RULE ENFORCEMENT:
        Rule 1: SIGReg always applied to z_c (adapter output).
                Never applied to z_hat (predictor output).
        Rule 2: L_info always takes z_hat as first argument (gradient guaranteed).
        Rule 3: Stop-grad applied to z_t here:
                    z_t = z_t.detach()  — correct
                NOT to frozen features f_t (which have no grad anyway).

    Args:
        z_c:    Adapter output for context frames, shape [B, N, d].
                Always has gradient (z_c = adapter(f_c), f_c has no grad
                but adapter params do).
        z_t:    Adapter output for target frames, shape [B, N, d].
                Gradient through z_t depends on the condition.
        z_hat:  Predictor output, shape [B, N, d].
                Always has gradient through predictor + adapter params.
        config: LossConfig specifying all condition-specific flags and weights.

    Returns:
        Dict with keys:
            'total'  — scalar total loss (for optimizer step)
            'l_pred' — prediction loss component
            'l_sig'  — SIGReg loss component (0.0 if disabled)
            'l_info' — information loss component (0.0 if disabled)
            'l_cov'  — covariance regularization (0.0 if disabled)
    """
    # --- Rule 3: Apply stop-gradient to adapter output z_t, not to frozen f_t ---
    # stop_grad=True: z_t = adapter(f_t).detach()
    # stop_grad=False: z_t = adapter(f_t)  (gradient flows through adapter params)
    if config.stop_grad:
        z_t = z_t.detach()

    # --- Prediction loss (always present) ---
    loss_pred = l_pred(z_hat, z_t)
    total = loss_pred

    # --- Rule 1: SIGReg on ADAPTER output z_c, NEVER on z_hat ---
    loss_sig = torch.tensor(0.0, device=z_c.device, dtype=z_c.dtype)
    if config.lambda_1 > 0.0:
        sig_val = apply_sigreg(
            z_c,
            axis=config.sigreg_axis,
            M=config.M_projections,
            T_knots=config.T_knots,
            bandwidth=config.bandwidth,
        )
        # Cast back to z_c dtype (sigreg runs in fp32 internally).
        loss_sig = sig_val.to(z_c.dtype)
        total = total + config.lambda_1 * loss_sig

    # --- Rule 2: L_info with z_hat as first argument (gradient guaranteed) ---
    loss_info = torch.tensor(0.0, device=z_hat.device, dtype=z_hat.dtype)
    if config.lambda_2 > 0.0:
        if config.use_dense_info:
            loss_info = l_info_dense(z_hat, z_t)
        else:
            loss_info = l_info_pooled(z_hat, z_t)
        total = total + config.lambda_2 * loss_info

    # --- Optional covariance regularization ---
    loss_cov = torch.tensor(0.0, device=z_hat.device, dtype=z_hat.dtype)
    if config.lambda_3 > 0.0:
        loss_cov = l_cov(z_hat)
        total = total + config.lambda_3 * loss_cov

    return {
        "total":  total,
        "l_pred": loss_pred.detach(),
        "l_sig":  loss_sig.detach(),
        "l_info": loss_info.detach(),
        "l_cov":  loss_cov.detach(),
    }


def loss_config_from_dict(cfg: dict) -> LossConfig:
    """Build a LossConfig from a parsed YAML config dict."""
    return LossConfig(
        stop_grad      = cfg.get("stop_grad", False),
        lambda_1       = cfg.get("lambda_1", 0.0),
        lambda_2       = cfg.get("lambda_2", 0.0),
        lambda_3       = cfg.get("lambda_3", 0.0),
        sigreg_axis    = cfg.get("sigreg_axis", "global"),
        use_dense_info = cfg.get("use_dense_info", True),
        M_projections  = cfg.get("M_projections", 1024),
        T_knots        = cfg.get("T_knots", 17),
        bandwidth      = cfg.get("bandwidth", 1.0),
    )
