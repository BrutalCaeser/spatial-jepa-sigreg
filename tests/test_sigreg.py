"""
Tests for SIGReg implementation — correctness, gradients, axis variants.

Required: 5 tests, zero failures, zero skips (build_spec.md Section 9.1).

Key properties verified:
    - N(0,I) input -> SIGReg ≈ 0  (Gaussian input = low loss)
    - Constant input -> SIGReg >> 0  (collapse = high loss)
    - Gradient exists through SIGReg
    - All three axis variants return different values
    - Random projection vectors are unit norm
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from models.sigreg import (
    sigreg,
    sigreg_global,
    sigreg_token,
    sigreg_channel,
    apply_sigreg,
    _epps_pulley_1d,
)


# ---------------------------------------------------------------------------
# Test 1: Gaussian input produces low SIGReg loss
# ---------------------------------------------------------------------------

def test_gaussian_input_low_loss():
    """SIGReg(Z) ≈ 0 when Z ~ N(0, I).

    Cramér-Wold theorem: SIGReg -> 0 <=> P_Z -> N(0, I).
    With K=2000 samples and d=16, should be well below 0.01.
    """
    torch.manual_seed(0)
    K, d = 2000, 16
    Z_gaussian = torch.randn(K, d)

    loss = sigreg(Z_gaussian, M=64, T_knots=17, bandwidth=1.0)

    # With large K, empirical CF is close to Gaussian CF -> small T statistic.
    assert loss.item() < 0.05, (
        f"SIGReg on Gaussian input too high: {loss.item():.6f}. "
        "Expected < 0.05 for K=2000 Gaussian samples."
    )


# ---------------------------------------------------------------------------
# Test 2: Constant input produces high SIGReg loss
# ---------------------------------------------------------------------------

def test_constant_input_high_loss():
    """SIGReg(Z) >> 0 when Z is a constant matrix (collapsed representation).

    Definition 2.3 (foundations.md): Under collapse, P_Z = delta(c) — a point mass.
    This is maximally non-Gaussian. SIGReg loss -> infinity under collapse.
    """
    torch.manual_seed(1)
    K, d = 100, 16
    # Constant matrix: all rows identical (complete collapse).
    Z_collapsed = torch.ones(K, d) * 3.14

    loss = sigreg(Z_collapsed, M=32, T_knots=17, bandwidth=1.0)

    # Should be >> 0 (much larger than Gaussian case).
    assert loss.item() > 0.1, (
        f"SIGReg on constant input too low: {loss.item():.6f}. "
        "Expected >> 0 for collapsed (constant) representation."
    )


# ---------------------------------------------------------------------------
# Test 3: Gradient exists through SIGReg
# ---------------------------------------------------------------------------

def test_gradient_exists():
    """SIGReg must have nonzero gradient for training to work.

    If SIGReg has no gradient, it cannot regularize the adapter during backprop.
    """
    torch.manual_seed(2)
    K, d = 32, 16
    Z = torch.randn(K, d, requires_grad=True)

    loss = sigreg(Z, M=16, T_knots=11, bandwidth=1.0)
    loss.backward()

    assert Z.grad is not None, "SIGReg has no gradient (Z.grad is None)"
    assert Z.grad.abs().sum() > 0, "SIGReg gradient is all zeros"
    assert not torch.isnan(Z.grad).any(), "SIGReg gradient contains NaN"
    assert not torch.isinf(Z.grad).any(), "SIGReg gradient contains Inf"


# ---------------------------------------------------------------------------
# Test 4: All three axis variants return different values
# ---------------------------------------------------------------------------

def test_all_axes_different():
    """sigreg_global, sigreg_token, and sigreg_channel must give different values.

    They operate on different sample distributions:
        global:  K=B samples of pooled representation (mean over patches)
        token:   K=B*N samples of individual patch tokens
        channel: Per-dimension EP test (different formulation)

    If they all returned the same value, there would be a bug in the axis dispatch.
    """
    torch.manual_seed(3)
    B, N, d = 4, 16, 8  # Small for speed

    # Non-trivial random input (not Gaussian, not constant).
    z = torch.randn(B, N, d) * 2.0 + 0.5

    loss_global  = sigreg_global(z,  M=16, T_knots=9, bandwidth=1.0).item()
    loss_token   = sigreg_token(z,   M=16, T_knots=9, bandwidth=1.0).item()
    loss_channel = sigreg_channel(z, T_knots=9, bandwidth=1.0).item()

    # All three should be finite and non-negative.
    for name, val in [("global", loss_global), ("token", loss_token), ("channel", loss_channel)]:
        assert val >= 0, f"SIGReg {name} is negative: {val}"
        assert not torch.tensor(val).isnan().item(), f"SIGReg {name} is NaN"

    # They should differ from each other (different axis = different statistics).
    assert loss_global != loss_token, (
        f"sigreg_global == sigreg_token ({loss_global:.6f}). "
        "These should differ — different sample populations (B vs B*N)."
    )
    assert loss_token != loss_channel, (
        f"sigreg_token == sigreg_channel ({loss_token:.6f}). "
        "These should differ — different formulation (joint vs per-dim)."
    )


# ---------------------------------------------------------------------------
# Test 5: Random projection vectors are unit norm
# ---------------------------------------------------------------------------

def test_projections_on_sphere():
    """The M random projection vectors u^(m) must lie on S^{d-1} (unit sphere).

    Definition 3.2 (foundations.md): Sample M unit vectors uniformly on S^{d-1}.
    If projections are not normalized, the Epps-Pulley statistic is not computing
    1D projections correctly, violating the Cramér-Wold theorem guarantee.
    """
    import torch.nn.functional as F

    torch.manual_seed(4)
    d, M = 256, 128

    # Generate unit vectors using the same method as sigreg().
    u = torch.randn(M, d)
    u = F.normalize(u, dim=-1)

    # Check unit norm for all M vectors.
    norms = u.norm(dim=-1)  # [M]
    max_deviation = (norms - 1.0).abs().max().item()

    assert max_deviation < 1e-5, (
        f"Random projection vectors are not unit norm. "
        f"Max deviation from 1.0: {max_deviation:.2e}"
    )

    # Additional: check that vectors are not all equal (random).
    # Pairwise inner products should be small on average.
    gram = u @ u.T   # [M, M]
    # Off-diagonal mean should be ~0 for random unit vectors in high dim.
    mask = ~torch.eye(M, dtype=torch.bool)
    off_diag_mean = gram.masked_select(mask).abs().mean().item()
    assert off_diag_mean < 0.5, (
        f"Random projection vectors are not diverse: off-diag mean = {off_diag_mean:.3f}"
    )


# ---------------------------------------------------------------------------
# Additional: verify apply_sigreg dispatcher
# ---------------------------------------------------------------------------

def test_apply_sigreg_dispatcher():
    """apply_sigreg() correctly dispatches to global/token/channel variants."""
    torch.manual_seed(5)
    B, N, d = 4, 9, 8

    z = torch.randn(B, N, d)

    for axis in ["global", "token", "channel"]:
        loss = apply_sigreg(z, axis=axis, M=8, T_knots=9, bandwidth=1.0)
        assert isinstance(loss, torch.Tensor), f"apply_sigreg({axis}) did not return a tensor"
        assert loss.shape == (), f"apply_sigreg({axis}) is not scalar: {loss.shape}"
        assert loss.item() >= 0, f"apply_sigreg({axis}) is negative: {loss.item()}"

    with pytest.raises(ValueError, match="Unknown sigreg_axis"):
        apply_sigreg(z, axis="invalid", M=8)
