"""
Tests for evaluation metrics — correctness of effective_rank, neighbor_corr, etc.

Required: 5 tests, zero failures, zero skips (build_spec.md Section 9.1).
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from training.metrics import (
    effective_rank,
    neighbor_corr,
    infonce_mi,
    cross_cov_trace,
    cross_cov_trace_dense,
    token_diversity,
    min_singular_value,
    variance_top1,
    compute_all_metrics,
)


# ---------------------------------------------------------------------------
# Test 1: effective_rank ≈ 1 for rank-1 input (collapsed)
# ---------------------------------------------------------------------------

def test_erank_collapsed():
    """effective_rank ≈ 1 for a rank-1 matrix (complete collapse).

    Definition 4.1 (foundations.md):
        erank = exp(-sum_i p_i * log(p_i))
        Under collapse, one singular value dominates -> p_1 ≈ 1 -> entropy ≈ 0 -> erank ≈ 1.
    """
    torch.manual_seed(0)
    K, d = 200, 32

    # Rank-1 matrix: outer product of two random vectors.
    u = torch.randn(K, 1)
    v = torch.randn(1, d)
    Z_rank1 = u @ v     # [K, d], rank=1

    erank = effective_rank(Z_rank1)

    assert abs(erank - 1.0) < 0.1, (
        f"effective_rank for rank-1 matrix should be ≈1.0, got {erank:.4f}. "
        "This is the expected value under complete representation collapse."
    )


# ---------------------------------------------------------------------------
# Test 2: effective_rank ≈ d for isotropic Gaussian (full rank)
# ---------------------------------------------------------------------------

def test_erank_full_rank():
    """effective_rank ≈ d for Z ~ N(0, I) (maximum diversity).

    For K >> d samples from N(0,I), singular values are approximately equal,
    so p_i ≈ 1/d for all i, entropy = log(d), erank = d.
    """
    torch.manual_seed(1)
    d = 16
    K = 5000    # K >> d for good estimation

    Z_gaussian = torch.randn(K, d)
    erank = effective_rank(Z_gaussian)

    # Should be close to d=16. Allow tolerance of 3 (batch variability).
    assert erank > d - 3, (
        f"effective_rank for Gaussian matrix should be ≈{d}, got {erank:.2f}. "
        "Isotropic Gaussian should achieve near-maximum effective rank."
    )
    assert erank <= d + 0.5, (
        f"effective_rank cannot exceed d={d}, got {erank:.2f}."
    )


# ---------------------------------------------------------------------------
# Test 3: neighbor_corr is high for spatially smooth features
# ---------------------------------------------------------------------------

def test_neighbor_corr_smooth():
    """neighbor_corr is high (>0.5) for spatially smooth representations.

    Definition 4.3 (foundations.md): cos similarity of adjacent patch tokens.
    If all tokens are identical (perfectly smooth), neighbor_corr = 1.0.
    If tokens vary slowly across the grid (smooth), neighbor_corr > 0.5.
    """
    torch.manual_seed(2)
    B, d = 4, 32
    G = 14   # 14x14 grid, N=196

    # Create smooth feature map: slow spatial variation (low-frequency).
    # Add small noise to unit vector -> high cos similarity between neighbors.
    base = torch.randn(B, 1, 1, d).expand(B, G, G, d)   # [B, G, G, d] constant
    noise = torch.randn(B, G, G, d) * 0.01              # very small noise
    z_smooth = (base + noise).reshape(B, G * G, d)       # [B, 196, d]

    ncorr = neighbor_corr(z_smooth, grid_size=G)

    assert ncorr > 0.5, (
        f"neighbor_corr for smooth features should be >0.5, got {ncorr:.4f}. "
        "Nearly-constant spatial features should have high neighbor similarity."
    )


# ---------------------------------------------------------------------------
# Test 4: neighbor_corr is low for random features
# ---------------------------------------------------------------------------

def test_neighbor_corr_random():
    """neighbor_corr is low (<0.3) for independent random features.

    Random features have no spatial structure — adjacent tokens are independent,
    so cosine similarity ≈ 0 in high dimensions.
    """
    torch.manual_seed(3)
    B, d = 8, 64
    G = 14

    Z_random = torch.randn(B, G * G, d)
    ncorr = neighbor_corr(Z_random, grid_size=G)

    # For random unit vectors in d=64 dimensions, E[cos_sim] ≈ 0.
    assert ncorr < 0.3, (
        f"neighbor_corr for random features should be <0.3, got {ncorr:.4f}. "
        "Independent random tokens should have near-zero neighbor similarity."
    )


# ---------------------------------------------------------------------------
# Test 5: InfoNCE is high for aligned pairs
# ---------------------------------------------------------------------------

def test_infonce_aligned_high():
    """InfoNCE MI bound is high when z_c and z_t are perfectly aligned.

    Definition 4.6 (foundations.md): InfoNCE lower bound on I(z_c; z_t).
    When z_c_pool = z_t_pool (identical representations), the score matrix
    is diagonal-dominant, and InfoNCE = log(B) (maximum bound).
    """
    torch.manual_seed(4)
    B, N, d = 16, 9, 32

    # Perfect alignment: z_c = z_t.
    z = torch.randn(B, N, d)
    z_c = z.clone()
    z_t = z.clone()

    mi_aligned = infonce_mi(z_c, z_t, tau=0.1)

    # Should be close to log(B) = log(16) ≈ 2.77.
    log_B = math.log(B)
    assert mi_aligned > log_B * 0.8, (
        f"InfoNCE for aligned pairs should be close to log(B)={log_B:.2f}, "
        f"got {mi_aligned:.4f}."
    )

    # For reference: random (unaligned) pairs should have lower MI.
    z_c_rand = torch.randn(B, N, d)
    z_t_rand = torch.randn(B, N, d)
    mi_random = infonce_mi(z_c_rand, z_t_rand, tau=0.1)

    assert mi_aligned > mi_random, (
        f"InfoNCE aligned ({mi_aligned:.4f}) should be > InfoNCE random ({mi_random:.4f})."
    )


# ---------------------------------------------------------------------------
# Additional: compute_all_metrics returns valid dict
# ---------------------------------------------------------------------------

def test_compute_all_metrics_structure():
    """compute_all_metrics() returns a dict with all required W&B keys."""
    torch.manual_seed(5)
    K, N, d = 32, 196, 32

    z_c   = torch.randn(K, N, d)
    z_t   = torch.randn(K, N, d)
    z_hat = torch.randn(K, N, d)

    metrics = compute_all_metrics(z_c, z_t, z_hat)

    required_keys = [
        "eval/erank",
        "eval/min_sv",
        "eval/var_top1",
        "eval/xcov_trace",
        "eval/xcov_trace_dense",
        "eval/infonce_mi",
        "eval/ncorr_adapter",
        "eval/ncorr_pred",
        "eval/tokdiv_adapter",
    ]

    for key in required_keys:
        assert key in metrics, f"Missing required metric key: {key}"
        val = metrics[key]
        assert isinstance(val, float), f"Metric {key} should be float, got {type(val)}"
        assert not math.isnan(val), f"Metric {key} is NaN"
        assert not math.isinf(val), f"Metric {key} is Inf"


# ---------------------------------------------------------------------------
# Additional: erank is in valid range
# ---------------------------------------------------------------------------

def test_erank_range():
    """effective_rank is in [1, min(K, d)] for any input."""
    torch.manual_seed(6)
    K, d = 50, 20
    Z = torch.randn(K, d)
    erank = effective_rank(Z)
    assert 1.0 <= erank <= min(K, d) + 0.1, (
        f"effective_rank {erank:.2f} out of range [1, {min(K,d)}]"
    )
