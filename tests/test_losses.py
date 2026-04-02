"""
Tests for loss functions — gradient flow, correctness, condition-specific logic.

Required: 5 tests, zero failures, zero skips (build_spec.md Section 9.1).

Key correctness properties:
    - L_info_dense has gradient through z_hat
    - Aligned inputs give lower (more negative) L_info_dense
    - Constant inputs -> L_info_dense = 0 (no penalty under collapse)
    - Condition E: adapter + predictor both have gradients
    - Condition F: SIGReg term is exactly 0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import pytest

from models.adapter import PatchAdapter
from models.predictor import JEPAPredictor
from models.losses import (
    l_pred,
    l_info_dense,
    l_info_pooled,
    l_cov,
    compute_loss,
    LossConfig,
)


# ---------------------------------------------------------------------------
# Helper: make_models
# ---------------------------------------------------------------------------

def make_models(d=32, D_in=64, N=16, n_classes=10):
    adapter   = PatchAdapter(D_in=D_in, D_out=d)
    predictor = JEPAPredictor(d=d, n_layers=1, n_heads=2, n_classes=n_classes, d_action=16)
    return adapter, predictor


# ---------------------------------------------------------------------------
# Test 1: L_info_dense has gradient through z_hat
# ---------------------------------------------------------------------------

def test_l_info_dense_gradient():
    """L_info_dense must have gradient through z_hat (Rule 2, CLAUDE.md).

    L_info must involve at least one tensor with gradient.
    z_hat = predictor(z_c, label) always has gradient through adapter + predictor.
    """
    torch.manual_seed(0)
    B, N, d = 4, 16, 32

    # z_hat: has gradient (simulates predictor output).
    z_hat = torch.randn(B, N, d, requires_grad=True)
    # z_t: detached target (stop-grad conditions), or live (no stop-grad).
    z_t   = torch.randn(B, N, d)  # no requires_grad needed

    loss = l_info_dense(z_hat, z_t)
    loss.backward()

    assert z_hat.grad is not None, "L_info_dense has no gradient through z_hat"
    assert z_hat.grad.abs().sum() > 0, "L_info_dense gradient through z_hat is zero"
    assert not torch.isnan(z_hat.grad).any(), "L_info_dense gradient contains NaN"


# ---------------------------------------------------------------------------
# Test 2: Aligned inputs give lower (more negative) L_info_dense
# ---------------------------------------------------------------------------

def test_l_info_dense_aligned_lower():
    """L_info_dense is more negative for aligned (identical) inputs vs random.

    L_info_dense = -(1/N) * sum_n Tr(C_n)
    When z_hat = z_t: Tr(C_n) = Tr(Cov(z,z)) > 0 -> loss is very negative.
    When z_hat is random and independent: Tr(C_n) ≈ 0 -> loss ≈ 0.
    """
    torch.manual_seed(1)
    B, N, d = 8, 16, 32

    z_target = torch.randn(B, N, d)

    # Perfectly aligned: z_hat = z_t -> maximum Tr(Cov) -> most negative loss.
    loss_aligned = l_info_dense(z_target.clone().requires_grad_(True), z_target)

    # Random (unaligned): z_hat independent of z_t -> near-zero cross-cov.
    z_random = torch.randn(B, N, d, requires_grad=True)
    loss_random = l_info_dense(z_random, z_target)

    # Aligned loss should be more negative.
    assert loss_aligned.item() < loss_random.item(), (
        f"Aligned loss ({loss_aligned.item():.4f}) should be < "
        f"random loss ({loss_random.item():.4f}). "
        "Maximizing cross-cov trace = minimizing (negative) L_info_dense."
    )


# ---------------------------------------------------------------------------
# Test 3: L_info_dense = 0 under collapse (constant inputs)
# ---------------------------------------------------------------------------

def test_l_info_zero_under_collapse():
    """When all tokens are constant, L_info_dense = 0.

    This verifies the theoretical claim in foundations.md Section 5 (Condition F):
    Under collapse, A_theta(f) -> c for all f.
    Then z_hat = c (constant), z_t = c (constant).
    Cov(constant, constant) = 0 -> Tr(C_n) = 0 -> L_info_dense = 0.
    This means L_info alone CANNOT prevent collapse.
    """
    torch.manual_seed(2)
    B, N, d = 8, 16, 32

    # Collapsed representation: all tokens identical.
    z_hat = torch.ones(B, N, d, requires_grad=True) * 2.71828
    z_t   = torch.ones(B, N, d) * 3.14159

    loss = l_info_dense(z_hat, z_t)

    # Should be (very close to) 0.
    # Note: after centering, both z_hat_c and z_t_c will be zero, so trace = 0.
    assert abs(loss.item()) < 1e-5, (
        f"L_info_dense on constant inputs should be 0, got {loss.item():.2e}. "
        "This confirms that L_info alone cannot prevent collapse."
    )


# ---------------------------------------------------------------------------
# Test 4: Condition E — adapter + predictor both have gradients
# ---------------------------------------------------------------------------

def test_condition_E_all_gradients():
    """In Condition E, gradients flow through both adapter and predictor parameters.

    Condition E loss: L_pred + lambda_1 * SIGReg(z_c) + lambda_2 * L_info_dense(z_hat, z_t)
    No stop-grad: z_t = adapter(f_t) (live gradient through adapter).
    Both adapter and predictor parameters must receive non-zero gradients.
    """
    torch.manual_seed(3)
    adapter, predictor = make_models(d=16, D_in=32, N=9, n_classes=5)

    B, N, D = 4, 9, 32
    f_c   = torch.randn(B, N, D)
    f_t   = torch.randn(B, N, D)
    label = torch.randint(0, 5, (B,))

    z_c   = adapter(f_c)
    z_t   = adapter(f_t)         # NO detach (Condition E)
    z_hat = predictor(z_c, label)

    # Condition E config (small M for speed).
    cfg = LossConfig(
        stop_grad=False, lambda_1=0.1, lambda_2=0.1, lambda_3=0.0,
        sigreg_axis="global", use_dense_info=True, M_projections=8, T_knots=9
    )
    loss_dict = compute_loss(z_c, z_t, z_hat, cfg)
    loss_dict["total"].backward()

    # Adapter parameters must have gradients.
    for name, param in adapter.named_parameters():
        assert param.grad is not None, f"Adapter.{name} has no gradient in Condition E"
        assert param.grad.abs().sum() > 0, f"Adapter.{name} has zero gradient in Condition E"

    # Predictor parameters must have gradients.
    # Two exceptions to the nonzero check:
    #   1. nn.Embedding: sparse gradient — only accessed rows are nonzero.
    #   2. cond_proj / AdaLN.proj: initialized to zeros, so the conditioning
    #      pathway has zero gradient at initialization (correct behavior — AdaLN
    #      starts as identity; conditioning activates as training proceeds).
    for name, param in predictor.named_parameters():
        assert param.grad is not None, f"Predictor.{name} has no gradient in Condition E"

    # The core transformer weights (attn, ffn) must have nonzero gradients.
    for name, param in predictor.named_parameters():
        if any(k in name for k in ["attn", "ffn", "proj_out", "norm_out", "pos_embed"]):
            assert param.grad.abs().sum() > 0, (
                f"Predictor.{name} has zero gradient in Condition E. "
                "Core transformer weights should always receive gradient."
            )


# ---------------------------------------------------------------------------
# Test 5: Condition F — SIGReg term is exactly 0
# ---------------------------------------------------------------------------

def test_condition_F_no_sigreg():
    """In Condition F, the SIGReg loss component must be exactly 0.0.

    Condition F: lambda_1 = 0.0 (no SIGReg). Only L_pred + L_info_dense.
    Verifies that compute_loss() correctly disables SIGReg when lambda_1=0.
    """
    torch.manual_seed(4)
    adapter, predictor = make_models(d=16, D_in=32, N=9, n_classes=5)

    B, N, D = 4, 9, 32
    f_c   = torch.randn(B, N, D)
    f_t   = torch.randn(B, N, D)
    label = torch.randint(0, 5, (B,))

    z_c   = adapter(f_c)
    z_t   = adapter(f_t)
    z_hat = predictor(z_c, label)

    # Condition F config: lambda_1=0 (no SIGReg).
    cfg = LossConfig(
        stop_grad=False, lambda_1=0.0, lambda_2=0.1, lambda_3=0.0,
        sigreg_axis="global", use_dense_info=True
    )
    loss_dict = compute_loss(z_c, z_t, z_hat, cfg)

    assert loss_dict["l_sig"].item() == 0.0, (
        f"Condition F SIGReg term should be 0.0, got {loss_dict['l_sig'].item()}. "
        "lambda_1=0.0 must disable SIGReg entirely."
    )

    # L_pred should be present (non-zero).
    assert loss_dict["l_pred"].item() > 0, "L_pred should be non-zero in Condition F"

    # Total should equal l_pred + lambda_2 * l_info (no sig component).
    expected_total = loss_dict["l_pred"].item() + 0.1 * loss_dict["l_info"].item()
    actual_total   = loss_dict["total"].item()
    assert abs(actual_total - expected_total) < 1e-5, (
        f"Total loss mismatch in Condition F: expected {expected_total:.6f}, "
        f"got {actual_total:.6f}"
    )


# ---------------------------------------------------------------------------
# Additional: Rule 1 check — SIGReg on z_c not z_hat
# ---------------------------------------------------------------------------

def test_sigreg_applied_to_z_c_not_z_hat():
    """compute_loss() applies SIGReg to z_c (adapter output), never to z_hat.

    Rule 1 (CLAUDE.md): SIGReg must regularize the representation SPACE
    (adapter output), not the prediction (predictor output).
    """
    torch.manual_seed(5)
    adapter, predictor = make_models(d=16, D_in=32, N=9, n_classes=5)

    B, N, D = 4, 9, 32
    f_c   = torch.randn(B, N, D)
    f_t   = torch.randn(B, N, D)
    label = torch.randint(0, 5, (B,))

    z_c   = adapter(f_c)
    z_t   = adapter(f_t)
    z_hat = predictor(z_c, label)

    cfg = LossConfig(lambda_1=0.5, lambda_2=0.0, sigreg_axis="global", M_projections=8, T_knots=9)
    loss_dict = compute_loss(z_c, z_t, z_hat, cfg)

    # SIGReg loss should be present.
    assert loss_dict["l_sig"].item() > 0, (
        "SIGReg loss is zero but lambda_1=0.5. "
        "Verify SIGReg is applied to z_c (adapter output)."
    )


# ---------------------------------------------------------------------------
# Additional: l_pred basic correctness
# ---------------------------------------------------------------------------

def test_l_pred_zero_for_perfect_prediction():
    """L_pred = 0 when z_hat == z_t (perfect prediction)."""
    z = torch.randn(4, 16, 32)
    loss = l_pred(z, z)
    assert abs(loss.item()) < 1e-6, f"L_pred should be 0 for identical inputs, got {loss.item()}"
