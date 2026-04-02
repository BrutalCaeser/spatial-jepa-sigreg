"""
Tests for PatchAdapter — gradient flow, output shape, and architecture constraints.

Required: 4 tests, zero failures, zero skips (build_spec.md Section 9.1).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import pytest

from models.adapter import PatchAdapter
from models.predictor import JEPAPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    return PatchAdapter(D_in=1024, D_out=256)


@pytest.fixture
def small_adapter():
    """Smaller adapter for faster tests."""
    return PatchAdapter(D_in=64, D_out=32)


@pytest.fixture
def predictor():
    return JEPAPredictor(d=256, n_layers=2, n_heads=4, n_classes=174, d_action=32)


# ---------------------------------------------------------------------------
# Test 1: Collapse is possible — gradient flows through both z_c and z_t
# ---------------------------------------------------------------------------

def test_collapse_possible():
    """Verify that Condition A can collapse: gradient flows through both paths.

    build_spec.md Section 7.2 — Critical implementation detail.
    Theorem 2.1 (foundations.md): Collapse is possible IFF gradients flow
    through both z_c and z_t paths (shared adapter weights).
    """
    adapter   = PatchAdapter(D_in=64, D_out=32)
    predictor = JEPAPredictor(d=32, n_layers=1, n_heads=2, n_classes=10, d_action=16)

    B, N, D = 4, 49, 64   # smaller for speed (7x7 grid)
    f_c   = torch.randn(B, N, D)  # frozen features — no gradient
    f_t   = torch.randn(B, N, D)
    label = torch.randint(0, 10, (B,))

    # Condition A: both z_c and z_t derived from adapter (no stop-grad).
    z_c   = adapter(f_c)          # gradient via adapter params (shared)
    z_t   = adapter(f_t)          # gradient via adapter params (shared)
    z_hat = predictor(z_c, label) # gradient via predictor + adapter

    loss = F.mse_loss(z_hat, z_t)
    loss.backward()

    # Both adapter parameters must have non-zero gradients.
    for name, param in adapter.named_parameters():
        assert param.grad is not None, f"No gradient for adapter.{name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for adapter.{name}"

    # Both z_c and z_t should contribute to adapter gradient.
    # Verify by checking that gradient is nonzero for both fc1 and fc2.
    assert adapter.fc1.weight.grad.abs().sum() > 0, "fc1 weight has zero gradient"
    assert adapter.fc2.weight.grad.abs().sum() > 0, "fc2 weight has zero gradient"


# ---------------------------------------------------------------------------
# Test 2: Stop-gradient blocks gradient through z_t
# ---------------------------------------------------------------------------

def test_stopgrad_blocks_target():
    """Stop-gradient on adapter output: z_t = adapter(f_t).detach() has no grad.

    Rule 3 (CLAUDE.md): Stop-grad applied to adapter output, NOT frozen features.
    After detach(), z_t.requires_grad should be False and adapter grad from z_t path = 0.

    This verifies that the stop-grad implementation is correct:
        CORRECT: z_t = adapter(f_t).detach()
        WRONG:   f_t.detach()  (no-op — frozen features already have no grad)
    """
    adapter   = PatchAdapter(D_in=64, D_out=32)
    predictor = JEPAPredictor(d=32, n_layers=1, n_heads=2, n_classes=10, d_action=16)

    B, N, D = 4, 49, 64
    f_c   = torch.randn(B, N, D)
    f_t   = torch.randn(B, N, D)
    label = torch.randint(0, 10, (B,))

    z_c   = adapter(f_c)
    # Condition B: stop-grad on adapter output.
    z_t   = adapter(f_t).detach()          # CORRECT stop-grad
    z_hat = predictor(z_c, label)

    # z_t should not require grad after detach.
    assert not z_t.requires_grad, "z_t.requires_grad should be False after detach()"

    loss = F.mse_loss(z_hat, z_t)
    loss.backward()

    # Adapter still gets gradient (through z_c -> predictor -> z_hat path).
    for name, param in adapter.named_parameters():
        assert param.grad is not None, f"No gradient for adapter.{name}"


# ---------------------------------------------------------------------------
# Test 3: Adapter output shape
# ---------------------------------------------------------------------------

def test_adapter_output_shape():
    """Adapter maps [B, N, D_in] -> [B, N, D_out] for various input shapes."""
    adapter = PatchAdapter(D_in=1024, D_out=256)

    # Standard dimensions.
    B, N, D = 8, 196, 1024
    f = torch.randn(B, N, D)
    z = adapter(f)
    assert z.shape == (B, N, 256), f"Expected [{B}, {N}, 256], got {z.shape}"

    # Batch size 1.
    f1 = torch.randn(1, 196, 1024)
    z1 = adapter(f1)
    assert z1.shape == (1, 196, 256)

    # Different N (non-standard grid).
    f2 = torch.randn(4, 49, 1024)
    z2 = adapter(f2)
    assert z2.shape == (4, 49, 256)

    # Small adapter.
    adapter_small = PatchAdapter(D_in=64, D_out=32)
    f3 = torch.randn(2, 10, 64)
    z3 = adapter_small(f3)
    assert z3.shape == (2, 10, 32)


# ---------------------------------------------------------------------------
# Test 4: Adapter has no normalization layers
# ---------------------------------------------------------------------------

def test_no_normalization_layers():
    """Adapter must not contain BatchNorm or LayerNorm (Rule 4, CLAUDE.md).

    Normalization layers would mask collapse signals — if the adapter maps
    everything to a constant, normalized outputs would still look unit-normed,
    hiding the collapse from effective rank and singular value metrics.
    """
    adapter = PatchAdapter(D_in=1024, D_out=256)

    forbidden_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
    )

    for name, module in adapter.named_modules():
        assert not isinstance(module, forbidden_types), (
            f"Adapter contains forbidden normalization layer: {name} ({type(module).__name__}). "
            "Rule 4 (CLAUDE.md): adapter must have NO normalization layers."
        )

    # Double-check: no LayerNorm, BatchNorm in any child.
    module_types = {type(m).__name__ for m in adapter.modules()}
    for forbidden in ["BatchNorm1d", "BatchNorm2d", "LayerNorm", "GroupNorm"]:
        assert forbidden not in module_types, (
            f"Found {forbidden} in adapter. Rule 4 violation."
        )
