"""
SIGReg — Sketched Isotropic Gaussian Regularizer

Definition 3.2 (foundations.md):
    SIGReg(Z) = (1/M) sum_m T^(m)
    where T^(m) is the Epps-Pulley test statistic applied to 1D projections Z @ u_m.

    Theoretical guarantee (Cramér-Wold theorem):
        SIGReg(Z) -> 0  <=>  P_Z -> N(0, I)  (in distribution, as M -> inf)

Axis variants (Definition 3.3, foundations.md):
    - sigreg_global(z):   pool patches first, then regularize  [B, d]
    - sigreg_token(z):    flatten patches, regularize each token  [B*N, d]
    - sigreg_channel(z):  EP test on each feature dimension independently

CRITICAL (CLAUDE.md):
    - ALL SIGReg computation must be in fp32 — empirical characteristic functions
      involve complex exponentials that are numerically sensitive in fp16/bf16.
    - SIGReg is always applied to the ADAPTER output z_c, NEVER predictor output.
    - Projection vectors u_m are resampled per forward pass (not learned parameters).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core EP test statistic
# ---------------------------------------------------------------------------

def _epps_pulley_1d(h: torch.Tensor, T_knots: int = 17, bandwidth: float = 1.0) -> torch.Tensor:
    """Epps-Pulley goodness-of-fit statistic against N(0,1) for a 1D sample.

    Computes:
        T = integral_[0.2, 4.0] w(t) * |phi_K(t) - phi_0(t)|^2 dt

    where:
        phi_K(t) = (1/K) * sum_k exp(i*t*h_k)  [empirical characteristic function]
        phi_0(t) = exp(-t^2/2)                  [N(0,1) characteristic function]
        w(t)     = exp(-t^2 / (2*bw^2))         [Gaussian weight]

    Integration via trapezoidal rule.

    Args:
        h:         Standardized 1D projection values, shape [K].  Must be float32.
        T_knots:   Number of quadrature nodes in [0.2, 4.0] (default 17).
        bandwidth: Gaussian weight bandwidth lambda (default 1.0).

    Returns:
        Scalar tensor: EP test statistic T >= 0.
    """
    K = h.shape[0]
    device = h.device

    # Quadrature nodes in [0.2, 4.0]
    t = torch.linspace(0.2, 4.0, T_knots, device=device)  # [T_knots]

    # Empirical characteristic function: phi_K(t) = mean_k exp(i*t*h_k)
    # Split into real and imaginary parts to avoid complex tensor ops in fp16.
    # h: [K], t: [T_knots]
    # outer product: t_h[k, j] = t[j] * h[k]
    t_h = torch.outer(h, t)               # [K, T_knots]
    cos_part = torch.cos(t_h).mean(dim=0)  # [T_knots]  Re(phi_K)
    sin_part = torch.sin(t_h).mean(dim=0)  # [T_knots]  Im(phi_K)

    # Gaussian characteristic function: phi_0(t) = exp(-t^2/2)  (real-valued)
    phi_0 = torch.exp(-0.5 * t ** 2)      # [T_knots]

    # |phi_K(t) - phi_0(t)|^2 = (Re(phi_K) - phi_0)^2 + Im(phi_K)^2
    diff_sq = (cos_part - phi_0) ** 2 + sin_part ** 2  # [T_knots]

    # Gaussian weight: w(t) = exp(-t^2 / (2*bw^2))
    w = torch.exp(-t ** 2 / (2.0 * bandwidth ** 2))    # [T_knots]

    # Trapezoidal integration: integral w(t) * diff_sq dt
    integrand = w * diff_sq               # [T_knots]
    dt = (4.0 - 0.2) / (T_knots - 1)
    T_stat = torch.trapezoid(integrand, dx=dt)

    return T_stat


def sigreg(
    Z: torch.Tensor,
    M: int = 1024,
    T_knots: int = 17,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """SIGReg: Sketched Isotropic Gaussian Regularizer.

    Definition 3.2 (foundations.md), modified to test against N(0,1):
        1. Sample M unit vectors u^(m) uniformly on S^{d-1}
        2. Project: h^(m) = Z @ u^(m)  in R^K
        3. Epps-Pulley test statistic T^(m) against N(0,1) via trapezoidal quadrature
        4. Return (1/M) * sum_m T^(m)

    No standardization: tests against N(0,1) directly, not "any Gaussian".
    This ensures nonzero gradient at the collapsed state (see CHANGELOG Session 7).

    Args:
        Z:         Sample matrix, shape [K, d].
                   K = number of samples, d = feature dimension.
        M:         Number of random projections (default 1024).
        T_knots:   Number of quadrature nodes for EP test (default 17).
        bandwidth: EP weight bandwidth (default 1.0).

    Returns:
        Scalar loss >= 0. Lower means more Gaussian.
    """
    # Force fp32 for numerical stability of complex exponentials.
    Z = Z.float()
    K, d = Z.shape
    device = Z.device

    # Step 1: Sample M unit vectors on S^{d-1}
    u = torch.randn(M, d, device=device, dtype=torch.float32)
    u = F.normalize(u, dim=-1)  # [M, d]  — unit vectors

    # Step 2: Project Z onto each unit vector.  h: [K, M]
    h = Z @ u.T  # [K, M]

    # NO standardization. We test against N(0,1) directly, not "any Gaussian".
    #
    # Why: Standardization (h - mean) / (std + eps) creates a gradient dead zone
    # at collapse. When all K samples are identical, h_tilde = 0 for all k, and
    # ∂T/∂h_tilde = 0 exactly (both sin_part=0 and centering cancellation).
    # SIGReg reports nonzero loss but zero gradient — it cannot push apart
    # collapsed representations.
    #
    # Without standardization, SIGReg tests "is P_h = N(0,1)?" instead of
    # "is P_h some Gaussian?". By Cramér-Wold, SIGReg(Z) → 0 ⟺ P_Z → N(0,I).
    # This simultaneously prevents collapse AND anchors scale (unit variance).
    #
    # See CHANGELOG.md Session 7 for full derivation of the gradient dead zone.

    # Step 3–4: Apply EP test to each raw projection column and average.
    T_stats = torch.stack([
        _epps_pulley_1d(h[:, m], T_knots=T_knots, bandwidth=bandwidth)
        for m in range(M)
    ])  # [M]

    return T_stats.mean()


# ---------------------------------------------------------------------------
# Axis variants (Definition 3.3, foundations.md)
# ---------------------------------------------------------------------------

def sigreg_global(
    z: torch.Tensor,
    M: int = 1024,
    T_knots: int = 17,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Global SIGReg: pool patches then regularize.

    Definition 3.3 — Global:
        z_pool = mean(z, dim=1) in R^{B x d}
        L_sig_global = SIGReg(z_pool)

    K = B samples.  Weakest spatial intervention — only the spatially-averaged
    representation is pushed toward N(0,I). Closest to LeWM style.

    Args:
        z: Adapter output, shape [B, N, d].

    Returns:
        Scalar SIGReg loss.
    """
    z_pool = z.mean(dim=1)  # [B, d]
    return sigreg(z_pool, M=M, T_knots=T_knots, bandwidth=bandwidth)


def sigreg_token(
    z: torch.Tensor,
    M: int = 1024,
    T_knots: int = 17,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Per-token SIGReg: flatten patches then regularize.

    Definition 3.3 — Per-token:
        z_flat = z.reshape(B*N, d) in R^{B*N x d}
        L_sig_token = SIGReg(z_flat)

    K = B*N samples.  Strongest spatial intervention — each (batch, position)
    token is independently pushed toward N(0,I), destroying spatial coherence.

    Args:
        z: Adapter output, shape [B, N, d].

    Returns:
        Scalar SIGReg loss.
    """
    B, N, d = z.shape
    z_flat = z.reshape(B * N, d)  # [B*N, d]
    return sigreg(z_flat, M=M, T_knots=T_knots, bandwidth=bandwidth)


def sigreg_channel(
    z: torch.Tensor,
    T_knots: int = 17,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Per-channel SIGReg: EP test on each feature dimension independently.

    Definition 3.3 — Per-channel:
        z_flat = z.reshape(B*N, d)
        L_sig_channel = (1/d) * sum_{j=0}^{d-1} T(z_flat[:, j])

    K = B*N scalar values per dimension.  No random projections needed.
    Moderate spatial intervention — each feature dimension is independently
    pushed toward N(0,1), but no cross-dimension structure is enforced.

    Args:
        z: Adapter output, shape [B, N, d].

    Returns:
        Scalar SIGReg loss.
    """
    z = z.float()
    B, N, d = z.shape
    z_flat = z.reshape(B * N, d)  # [B*N, d]

    T_stats = torch.stack([
        _ep_test_1d_scalar(z_flat[:, j], T_knots=T_knots, bandwidth=bandwidth)
        for j in range(d)
    ])  # [d]

    return T_stats.mean()


def _ep_test_1d_scalar(
    x: torch.Tensor,
    T_knots: int = 17,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """EP test on a 1D sample against N(0,1): no standardization.

    Tests raw values directly against N(0,1), ensuring nonzero gradient
    at the collapsed state. See sigreg() docstring for rationale.

    Args:
        x: Raw 1D sample values, shape [K].

    Returns:
        Scalar EP test statistic.
    """
    x = x.float()
    return _epps_pulley_1d(x, T_knots=T_knots, bandwidth=bandwidth)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_sigreg(
    z_c: torch.Tensor,
    axis: str,
    M: int = 1024,
    T_knots: int = 17,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Dispatch SIGReg to the correct axis variant.

    CRITICAL (CLAUDE.md Rule 1): z_c must be adapter output, never predictor output.

    Args:
        z_c:  Adapter output for context frames, shape [B, N, d].
        axis: One of 'global', 'token', 'channel'.
        M:    Number of random projections (used by global and token variants).

    Returns:
        Scalar SIGReg loss.
    """
    if axis == "global":
        return sigreg_global(z_c, M=M, T_knots=T_knots, bandwidth=bandwidth)
    elif axis == "token":
        return sigreg_token(z_c, M=M, T_knots=T_knots, bandwidth=bandwidth)
    elif axis == "channel":
        return sigreg_channel(z_c, T_knots=T_knots, bandwidth=bandwidth)
    else:
        raise ValueError(f"Unknown sigreg_axis '{axis}'. Choose: global, token, channel.")
