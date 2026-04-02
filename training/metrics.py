"""
Evaluation metrics for GAP 1 experiment.

*** THIS FILE IS FROZEN AFTER THE FIRST EXPERIMENT JOB IS SUBMITTED ***
*** NEVER MODIFY AFTER FREEZE (CLAUDE.md Rule 6) ***
*** All conditions must be evaluated with identical metric code. ***

Definitions (foundations.md Section 4):
    4.1  effective_rank      — exp(Shannon entropy of singular value energy)
    4.2  cross_cov_trace     — diagnostic: Tr(Cov(z_c_pool, z_t_pool))
    4.3  neighbor_corr       — cosine similarity of adjacent patch tokens
    4.4  token_diversity     — 1 - mean off-diagonal token similarity
    4.5  (linear probe)      — see evaluation/linear_probe.py
    4.6  infonce_mi          — InfoNCE lower bound on I(z_c; z_t)

CRITICAL (CLAUDE.md — Common Mistakes #6):
    Metrics must be computed on the CONCATENATION of all eval batches,
    NOT on the mean of per-batch metrics.
    effective_rank(concat) != mean(effective_rank per batch)

All metrics are computed under torch.no_grad() and returned as Python floats
(for W&B logging).
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Definition 4.1 — Effective Rank
# ---------------------------------------------------------------------------

def effective_rank(Z: torch.Tensor) -> float:
    """Effective rank of a sample matrix Z.

    Definition 4.1 (foundations.md):
        sigma_i = singular values of Z
        p_i     = sigma_i^2 / sum_j sigma_j^2
        erank   = exp( -sum_i p_i * ln(p_i) )  [exponential of Shannon entropy]

    Range: [1, min(K, d)].
    - erank = 1  -> complete collapse (all energy in one component)
    - erank = d  -> maximum diversity (uniform energy distribution)

    Args:
        Z: Sample matrix, shape [K, d].  K samples, d features.

    Returns:
        Effective rank as Python float.
    """
    Z = Z.float()
    # SVD: singular values only.
    try:
        sv = torch.linalg.svdvals(Z)    # [min(K,d)]
    except Exception:
        # Fallback if svdvals fails (very degenerate matrix).
        return 1.0

    # Normalized energy per singular value.
    energy = sv ** 2
    total  = energy.sum()
    if total < 1e-12:
        # Numerically zero matrix — complete collapse.
        return 1.0

    p = energy / total                   # [min(K,d)]  — probability distribution
    # Remove zeros before log to avoid nan (0 * log(0) = 0 by convention).
    p_nonzero = p[p > 1e-12]
    entropy   = -(p_nonzero * torch.log(p_nonzero)).sum()
    return math.exp(entropy.item())


def min_singular_value(Z: torch.Tensor) -> float:
    """Smallest singular value of Z.

    Collapse severity metric. Approaches 0 under representation collapse.

    Args:
        Z: Sample matrix, shape [K, d].

    Returns:
        Min singular value as Python float.
    """
    Z = Z.float()
    try:
        sv = torch.linalg.svdvals(Z)
        return sv.min().item()
    except Exception:
        return 0.0


def variance_top1(Z: torch.Tensor) -> float:
    """Fraction of variance explained by the top singular value.

    = sigma_1^2 / sum_i sigma_i^2

    Range [0, 1]. Approaches 1.0 under collapse.

    Args:
        Z: Sample matrix, shape [K, d].

    Returns:
        Variance fraction in top singular value, as Python float.
    """
    Z = Z.float()
    try:
        sv = torch.linalg.svdvals(Z)
    except Exception:
        return 1.0

    energy = sv ** 2
    total  = energy.sum()
    if total < 1e-12:
        return 1.0
    return (energy[0] / total).item()


# ---------------------------------------------------------------------------
# Definition 4.2 — Cross-Covariance Trace (diagnostic, no negative sign)
# ---------------------------------------------------------------------------

def cross_cov_trace(z_c: torch.Tensor, z_t: torch.Tensor) -> float:
    """Cross-covariance trace between pooled context and target representations.

    Definition 4.2 (foundations.md):
        z_c_pool = mean(z_c, dim=1)  in R^{B x d}
        z_t_pool = mean(z_t, dim=1)  in R^{B x d}
        C        = (1/B) * z_c_pool_centered^T @ z_t_pool_centered
        result   = Tr(C)    [POSITIVE — diagnostic metric, NOT the loss]

    Higher = more linear dependence between context and target representations.

    Args:
        z_c: Context adapter output, shape [B, N, d].
        z_t: Target adapter output, shape [B, N, d].

    Returns:
        Cross-covariance trace as Python float.
    """
    with torch.no_grad():
        z_c_pool = z_c.float().mean(dim=1)   # [B, d]
        z_t_pool = z_t.float().mean(dim=1)   # [B, d]

        z_c_c = z_c_pool - z_c_pool.mean(dim=0, keepdim=True)
        z_t_c = z_t_pool - z_t_pool.mean(dim=0, keepdim=True)
        B = z_c_pool.shape[0]

        # Trace of cross-covariance = element-wise product sum / B
        trace = (z_c_c * z_t_c).sum() / B
        return trace.item()


def cross_cov_trace_dense(z_hat: torch.Tensor, z_t: torch.Tensor) -> float:
    """Dense cross-covariance trace across all patch positions.

    Same formula as l_info_dense but POSITIVE (diagnostic, not loss).

    = (1/N) * sum_n Tr(C_n)   where C_n is per-patch cross-covariance.

    Args:
        z_hat: Predictor output or context adapter output, shape [B, N, d].
        z_t:   Target adapter output, shape [B, N, d].

    Returns:
        Dense cross-covariance trace as Python float (positive).
    """
    with torch.no_grad():
        z_hat = z_hat.float()
        z_t   = z_t.float()

        z_hat_c = z_hat - z_hat.mean(dim=0, keepdim=True)
        z_t_c   = z_t   - z_t.mean(dim=0, keepdim=True)

        # Element-wise product, sum over d, average over B and N.
        trace = (z_hat_c * z_t_c).sum(dim=-1).mean()
        return trace.item()


# ---------------------------------------------------------------------------
# Definition 4.3 — Neighbor Token Correlation
# ---------------------------------------------------------------------------

def neighbor_corr(z: torch.Tensor, grid_size: int = 14) -> float:
    """Average cosine similarity between spatially adjacent patch tokens.

    Definition 4.3 (foundations.md):
        z_grid = z.reshape(B, 14, 14, d)
        cos_h  = cosine_similarity(z_grid[:, :, :-1, :], z_grid[:, :, 1:, :])
        cos_v  = cosine_similarity(z_grid[:, :-1, :, :], z_grid[:, 1:, :, :])
        result = (mean(cos_h) + mean(cos_v)) / 2

    Range: [-1, 1]. Typical useful range: [0.3, 0.9].
    - High (>0.7): strong spatial coherence
    - Low (<0.1):  spatial structure destroyed

    Assumes N = grid_size^2 patch tokens (default 14x14=196 for V-JEPA 2.1).

    Args:
        z:         Token representations, shape [B, N, d].
        grid_size: Spatial grid dimension (default 14 for 14x14 grid).

    Returns:
        Neighbor correlation as Python float.
    """
    with torch.no_grad():
        z = z.float()
        B, N, d = z.shape
        assert N == grid_size * grid_size, (
            f"Expected N={grid_size**2} but got N={N}. "
            f"Ensure grid_size={grid_size} matches spatial layout."
        )

        # Reshape to spatial grid.
        z_grid = z.reshape(B, grid_size, grid_size, d)

        # Horizontal neighbors: (h, w) vs (h, w+1).
        cos_h = F.cosine_similarity(
            z_grid[:, :, :-1, :],    # [B, G, G-1, d]
            z_grid[:, :, 1:, :],     # [B, G, G-1, d]
            dim=-1,
        )  # [B, G, G-1]

        # Vertical neighbors: (h, w) vs (h+1, w).
        cos_v = F.cosine_similarity(
            z_grid[:, :-1, :, :],    # [B, G-1, G, d]
            z_grid[:, 1:, :, :],     # [B, G-1, G, d]
            dim=-1,
        )  # [B, G-1, G]

        return ((cos_h.mean() + cos_v.mean()) / 2.0).item()


# ---------------------------------------------------------------------------
# Definition 4.4 — Token Diversity
# ---------------------------------------------------------------------------

def token_diversity(z: torch.Tensor) -> float:
    """Average pairwise dissimilarity between patch tokens within a sample.

    Definition 4.4 (foundations.md):
        z_norm = L2_normalize(z, dim=-1)          in R^{B x N x d}
        sim    = z_norm @ z_norm^T                in R^{B x N x N}
        result = 1 - mean(off_diagonal(sim))

    Range: [0, 1].
    - High: tokens are diverse (encoding different spatial content)
    - Low:  tokens are similar (collapse to spatially uniform representation)

    Args:
        z: Token representations, shape [B, N, d].

    Returns:
        Token diversity as Python float.
    """
    with torch.no_grad():
        z = z.float()
        B, N, d = z.shape

        # L2 normalize along feature dimension.
        z_norm = F.normalize(z, dim=-1)          # [B, N, d]

        # Pairwise cosine similarity matrix per sample.
        sim = torch.bmm(z_norm, z_norm.transpose(1, 2))  # [B, N, N]

        # Mask diagonal and compute mean of off-diagonal elements.
        mask = ~torch.eye(N, device=z.device, dtype=torch.bool).unsqueeze(0)  # [1, N, N]
        off_diag_mean = sim.masked_select(mask.expand(B, -1, -1)).mean()

        return (1.0 - off_diag_mean).item()


# ---------------------------------------------------------------------------
# Definition 4.6 — InfoNCE Mutual Information Lower Bound
# ---------------------------------------------------------------------------

def infonce_mi(z_c: torch.Tensor, z_t: torch.Tensor, tau: float = 0.1) -> float:
    """InfoNCE lower bound on I(z_c; z_t).

    Definition 4.6 (foundations.md):
        score(i, j) = z_c_pool^(i) · z_t_pool^(j) / tau
        I_NCE = log(B) - (1/B) * sum_i log(sum_j exp(score(i,j)))
                       + (1/B) * sum_i score(i,i)

    Does not assume Gaussianity. Used as a diagnostic complementary to
    cross-covariance trace.

    Args:
        z_c: Context adapter output, shape [B, N, d].
        z_t: Target adapter output, shape [B, N, d].
        tau: Temperature (default 0.1).

    Returns:
        InfoNCE MI bound as Python float.
    """
    with torch.no_grad():
        z_c_pool = z_c.float().mean(dim=1)   # [B, d]
        z_t_pool = z_t.float().mean(dim=1)   # [B, d]
        B = z_c_pool.shape[0]

        # Compute scores: [B, B] matrix.
        scores = (z_c_pool @ z_t_pool.T) / tau   # [B, B]

        # InfoNCE = log(B) + mean(diag(scores)) - mean(logsumexp(scores, dim=1))
        log_B    = math.log(B)
        diag_mean = scores.diagonal().mean()
        logsumexp = torch.logsumexp(scores, dim=1).mean()

        infonce = log_B + diag_mean - logsumexp
        return infonce.item()


# ---------------------------------------------------------------------------
# Master metrics function — called every eval_every steps
# ---------------------------------------------------------------------------

def compute_all_metrics(
    z_c_all: torch.Tensor,
    z_t_all: torch.Tensor,
    z_hat_all: torch.Tensor,
    include_dense_xcov: bool = True,
) -> Dict[str, float]:
    """Compute all evaluation metrics on concatenated validation data.

    CRITICAL (CLAUDE.md — Common Mistakes #6):
        All tensors must be the CONCATENATION of all validation batches,
        NOT separate per-batch results averaged afterward.
        Effective rank, neighbor correlation, etc. are non-linear in batch size.

    This function is called in trainer.py with pre-concatenated tensors.

    Args:
        z_c_all:   All context adapter outputs, shape [K, N, d].
        z_t_all:   All target adapter outputs, shape [K, N, d].
        z_hat_all: All predictor outputs, shape [K, N, d].
        include_dense_xcov: Whether to compute the expensive dense metric.

    Returns:
        Dict mapping W&B log keys to float metric values.
    """
    K, N, d = z_c_all.shape

    # Flatten to [K*N, d] for rank-based metrics on adapter output.
    z_c_flat = z_c_all.reshape(K * N, d)

    metrics: Dict[str, float] = {}

    # --- Collapse metrics (Section 5.1, build_spec.md) ---
    metrics["eval/erank"]     = effective_rank(z_c_flat)
    metrics["eval/min_sv"]    = min_singular_value(z_c_flat)
    metrics["eval/var_top1"]  = variance_top1(z_c_flat)

    # --- Information metrics (Section 5.2, build_spec.md) ---
    metrics["eval/xcov_trace"]      = cross_cov_trace(z_c_all, z_t_all)
    if include_dense_xcov:
        metrics["eval/xcov_trace_dense"] = cross_cov_trace_dense(z_hat_all, z_t_all)
    metrics["eval/infonce_mi"]      = infonce_mi(z_c_all, z_t_all)

    # --- Spatial metrics (Section 5.3, build_spec.md) ---
    metrics["eval/ncorr_adapter"]    = neighbor_corr(z_c_all)
    metrics["eval/ncorr_pred"]       = neighbor_corr(z_hat_all)
    metrics["eval/tokdiv_adapter"]   = token_diversity(z_c_all)

    return metrics


def compute_baseline_metrics(f_raw: torch.Tensor) -> Dict[str, float]:
    """Compute baseline metrics on raw V-JEPA 2.1 features (pre-adapter).

    Logged once before training begins as reference (Section 5.4, build_spec.md).

    Args:
        f_raw: Raw frozen features, shape [K, N, D] (D=1024).

    Returns:
        Dict with baseline/ prefixed keys.
    """
    K, N, D = f_raw.shape
    f_flat = f_raw.reshape(K * N, D)

    return {
        "baseline/raw_erank": effective_rank(f_flat),
        "baseline/raw_ncorr": neighbor_corr(f_raw),
    }
