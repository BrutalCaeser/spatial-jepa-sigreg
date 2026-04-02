# foundations.md — Mathematical Foundations

**Version:** 1.0
**Scope:** All mathematical objects, definitions, and derivations for GAP 1.
**Rule:** Every formula in build_spec.md or code must trace back to a numbered definition here.

---

## 0. Notation

| Symbol | Meaning | Shape |
|--------|---------|-------|
| B | Batch size | scalar |
| N | Number of patch tokens per frame (14×14 = 196 for V-JEPA 2.1 ViT-G at 224²) | scalar |
| D | Frozen V-JEPA 2.1 embedding dimension | 1024 |
| d | Adapter output dimension (hyperparameter) | 256 or 512 |
| f | Frozen V-JEPA 2.1 feature tensor | ℝ^{B×N×D} |
| z | Adapter output (learned representation) | ℝ^{B×N×d} |
| ẑ | Predictor output | ℝ^{B×N×d} |
| A_θ | Trainable adapter (encoder surrogate) | ℝ^D → ℝ^d |
| P_φ | Predictor network | (ℝ^{B×N×d}, ℝ^B) → ℝ^{B×N×d} |
| sg(·) | Stop-gradient operator (detach from computation graph) | identity on values, zero on gradients |
| ⊙ | Element-wise product | — |

Subscript conventions:
- `c` = context (frame at time t)
- `t` = target (frame at time t+1)
- `pool` = spatially averaged (mean over N patch tokens)

---

## 1. Architecture Pipeline

### Definition 1.1 — Full Forward Pass

Given consecutive frozen V-JEPA 2.1 features f_c, f_t ∈ ℝ^{B×N×D} and clip label y ∈ ℤ^B:

```
Step 1 (Adapter):   z_c = A_θ(f_c)    ∈ ℝ^{B×N×d}
                    z_t = A_θ(f_t)    ∈ ℝ^{B×N×d}

Step 2 (Predictor): ẑ_t = P_φ(z_c, y) ∈ ℝ^{B×N×d}

Step 3 (Loss):      L = L(ẑ_t, z_t, z_c; condition)
```

The same adapter A_θ (shared weights) processes both context and target features.
Whether gradients flow through z_t depends on the experimental condition.

### Definition 1.2 — Adapter Architecture

The adapter is a token-wise MLP applied independently to each of the B×N tokens:

```
A_θ(f) = W₂ · GELU(W₁ · f + b₁) + b₂
```

Where:
- W₁ ∈ ℝ^{D×D}, b₁ ∈ ℝ^D (expand)
- W₂ ∈ ℝ^{D×d}, b₂ ∈ ℝ^d (project)
- GELU activation (matching V-JEPA 2.1 convention)

The adapter is applied token-wise: for each (b, n) pair independently.
No BatchNorm, no LayerNorm in the adapter — normalization would mask collapse signals.

**Parameter count:** D² + D + D·d + d = 1024² + 1024 + 1024·d + d ≈ 1.3M (for d=256)

**Why this design:**
- Token-wise processing preserves spatial independence — the adapter does not mix tokens
- Two layers provide enough capacity to remap the feature space
- No normalization ensures that if the adapter maps everything to a constant,
  that constant is observable in the outputs (BatchNorm would hide it)

### Definition 1.3 — Predictor Architecture

The predictor is a small transformer with Adaptive Layer Normalization (AdaLN) conditioning:

```
P_φ(z_c, y) = Transformer_AdaLN(z_c, Embed(y))
```

Where:
- z_c ∈ ℝ^{B×N×d} — adapted context patch tokens
- y ∈ ℤ^B — clip-level action label (one of 174 SSv2 classes)
- Embed: ℤ → ℝ^{d_action} — learned embedding lookup
- AdaLN: action embedding modulates LayerNorm scale/shift at each transformer layer

Output: ẑ_t ∈ ℝ^{B×N×d} — predicted target representation

---

## 2. Collapse Mechanism

### Definition 2.1 — Representation Collapse

Collapse occurs when the adapter maps all inputs to a constant:

```
∃ c ∈ ℝ^d such that A_θ(f) → c  for all f
```

Under collapse, z_c = z_t = c (constant matrix), so:
- Any predictor that outputs c achieves L_pred = 0
- All representations are identical regardless of input

### Theorem 2.1 — Collapse Is Possible If and Only If Gradients Flow Through Both Paths

**Claim:** The trivial solution z_c = z_t = c, ẑ_t = c, L_pred = 0 is a critical
point of L_pred when gradients flow through both z_c and z_t.

**Proof sketch:**
Let L_pred = (1/BNd) Σ (ẑ_t - z_t)². At the trivial solution:
- ∂L/∂φ = 0 (predictor outputs constant regardless of input, matches constant target)
- ∂L/∂θ via z_t path: ∂L/∂z_t = -(2/BNd)(ẑ_t - z_t) = 0 at the trivial point
- ∂L/∂θ via z_c path: ∂L/∂z_c flows through predictor, but predictor output is constant → 0

Therefore the trivial solution is a fixed point of gradient descent.

**When stop-gradient is applied to z_t:**
The gradient ∂L/∂θ comes ONLY through the z_c → P_φ → ẑ_t path.
The predictor must now actually model the (non-constant) targets to reduce loss.
The adapter must produce informative representations for the predictor to succeed.
The trivial solution is no longer a useful fixed point because the targets z_t remain
diverse (the adapter still produces diverse targets even if the loss can't update through them).

Wait — this needs more care. With stop-gradient on z_t:
- z_t = sg(A_θ(f_t)) — values computed but gradients blocked
- The TARGETS are A_θ(f_t) with current weights, but treated as fixed constants for this step
- If the adapter moves toward collapse, the targets z_t also become constant
  (they use the same weights), so the loss still approaches 0

**Correction:** Stop-gradient alone does NOT prevent collapse in this architecture.
With shared adapter weights, if A_θ → constant, then sg(A_θ(f_t)) is also constant.
The predictor trivially outputs that constant. Loss = 0.

This matches the JEPA literature: stop-gradient alone is insufficient.
V-JEPA uses EMA (exponential moving average) target encoder to break the symmetry.
LeWM uses SIGReg instead of EMA.

### Definition 2.2 — Why EMA Prevents Collapse (for reference, not used in our experiment)

In V-JEPA, the target encoder uses weights θ̄ updated as:
```
θ̄ ← τ·θ̄ + (1-τ)·θ     where τ ∈ [0.996, 0.999]
```

This creates temporal asymmetry: the target encoder changes slowly while the
context encoder (same architecture, weights θ) changes fast. The targets are
"stale" — they reflect a previous version of the representation. The encoder
must produce representations useful for predicting these slowly-moving targets,
preventing the trivial collapse equilibrium.

### Definition 2.3 — Why SIGReg Prevents Collapse (used in our experiment)

SIGReg enforces P_Z → N(0, I). A collapsed representation where all z's are
constant has P_Z = δ(c) (point mass), which is maximally non-Gaussian.
Therefore SIGReg loss → ∞ under collapse, creating an infinite barrier.

**Critical insight:** SIGReg replaces the role of both stop-gradient AND EMA.
In conditions that use SIGReg, we do NOT need stop-gradient. This is why
LeWM trains without stop-gradient or EMA.

---

## 3. Loss Functions

### Definition 3.1 — Prediction Loss

```
L_pred = (1 / (B · N · d)) · Σ_{b,n,j} (ẑ_t[b,n,j] − z_t[b,n,j])²
```

Equivalent to: `F.mse_loss(z_hat, z_target)` (PyTorch default is mean reduction).

### Definition 3.2 — SIGReg (Sketched Isotropic Gaussian Regularizer)

Given a matrix Z ∈ ℝ^{K×d} (K samples, d dimensions):

**Step 1 — Random projection:**
Sample M unit vectors u^(m) ∈ S^{d-1} (uniform on the unit hypersphere).
Compute 1D projections: h^(m) = Z · u^(m) ∈ ℝ^K

**Step 2 — Standardization:**
For each projection m:
```
h̃^(m) = (h^(m) − mean(h^(m))) / std(h^(m))
```

**Step 3 — Epps-Pulley test statistic via empirical characteristic function:**

The Epps-Pulley statistic measures the L² distance between the empirical
characteristic function (ECF) and the standard Gaussian characteristic function:

```
T^(m) = ∫_{-∞}^{∞} w(t) |φ_K(t; h̃^(m)) − φ₀(t)|² dt
```

Where:
- φ_K(t; h̃) = (1/K) Σ_{k=1}^K exp(i·t·h̃_k)  — empirical characteristic function
- φ₀(t) = exp(-t²/2)  — standard Gaussian characteristic function
- w(t) = exp(-t² / (2λ²))  — weighting function (λ is a bandwidth parameter)
- The integral is computed numerically via quadrature with T_knots nodes
  uniformly spaced in [0.2, 4.0]

**Step 4 — Aggregate:**
```
SIGReg(Z) = (1/M) Σ_{m=1}^M T^(m)
```

**Theoretical guarantee (Cramér-Wold theorem):**
In the limit M → ∞:
```
SIGReg(Z) → 0  ⟺  P_Z → N(0, I)  (in distribution)
```

Matching all 1D marginals of a distribution is equivalent to matching
the full joint distribution.

**Default hyperparameters (from LeWM):** M = 1024 projections, T_knots = 17,
bandwidth λ = 1.0, integration range [0.2, 4.0].

### Definition 3.3 — SIGReg Axis Variants

Given adapter output z ∈ ℝ^{B×N×d}:

**Global (pool then regularize):**
```
z_pool = mean(z, dim=1)             ∈ ℝ^{B×d}
L_sig_global = SIGReg(z_pool)
```
K = B samples, each of dimension d.

**Per-token (flatten then regularize):**
```
z_flat = z.reshape(B·N, d)          ∈ ℝ^{(B·N)×d}
L_sig_token = SIGReg(z_flat)
```
K = B·N samples, each of dimension d.
This treats each spatial token independently.

**Per-channel (regularize each feature dimension independently):**
```
z_flat = z.reshape(B·N, d)          ∈ ℝ^{(B·N)×d}
L_sig_channel = (1/d) Σ_{j=0}^{d-1} T(z_flat[:, j])
```
For each feature dimension j, apply the EP test to the K = B·N scalar values.
No random projections needed — each column is already 1D.

**Per-token vs Global — spatial structure implications:**

Per-token SIGReg enforces that each (batch_item, patch_position) sample is drawn
from N(0,I). This means:
- Token at position (3, 7) and token at position (3, 8) are each pushed toward
  the same isotropic Gaussian
- Their CORRELATION is not explicitly controlled, but the isotropy pressure
  discourages structured correlations between neighbors
- Prediction: per-token SIGReg destroys spatial coherence

Global SIGReg enforces that the spatially-averaged representation is Gaussian.
Averaging preserves low-frequency spatial information (the mean value across all patches)
but destroys high-frequency spatial detail. Therefore:
- Some coarse spatial structure may survive
- Fine-grained spatial relationships are not directly regulated

### Definition 3.4 — Cross-Covariance Information Loss (L_info)

This term encourages alignment between the predictor output and the target representation.

```
ẑ_pool = mean(ẑ_t, dim=1)           ∈ ℝ^{B×d}     (predictor output, pooled)
z_t_pool = mean(z_t, dim=1)          ∈ ℝ^{B×d}     (target adapter output, pooled)

ẑ_centered = ẑ_pool − mean(ẑ_pool, dim=0)    ∈ ℝ^{B×d}
z_centered = z_t_pool − mean(z_t_pool, dim=0)  ∈ ℝ^{B×d}

C = (1/B) · ẑ_centered^T · z_centered         ∈ ℝ^{d×d}

L_info = −Tr(C) = −Σ_{j=0}^{d-1} C[j, j]
```

**Why predictor output, not context adapter output:**
L_info must have gradient flowing through learnable parameters. The predictor
output ẑ_t is a function of both adapter parameters θ and predictor parameters φ.
The target z_t may or may not have gradients depending on the condition.
By computing L_info between (ẑ_pool, z_t_pool), we guarantee at least one side
always has gradients.

**Relationship to mutual information:**
Tr(C) = Σ_j Cov(ẑ_j, z_j) is the sum of per-dimension linear covariances.
Under joint Gaussianity, mutual information is:
```
I(ẑ; z) = −(1/2) log det(I − Σ_ẑ^{-1/2} C Σ_z^{-1/2} C^T Σ_ẑ^{-1/2})
```
The trace Tr(C) is a LOWER BOUND on mutual information only when both
representations have unit covariance (Σ_ẑ = Σ_z = I). SIGReg pushes toward
this condition. Therefore:

**When SIGReg is active, Tr(C) is a tighter proxy for MI.**
**When SIGReg is not active, Tr(C) is a weaker proxy.**

This justifies the combination of SIGReg + L_info as more principled than either alone.

**We do NOT claim L_info equals mutual information.** We claim it is a
linear-dependence-maximizing objective that, under the near-Gaussian conditions
enforced by SIGReg, approximates a lower bound on MI.

### Definition 3.5 — Dense Information Loss (L_info_dense)

Alternative to the pooled version. Preserves spatial structure:

```
ẑ_centered[b,n,:] = ẑ_t[b,n,:] − mean_b(ẑ_t[:,n,:])
z_centered[b,n,:] = z_t[b,n,:] − mean_b(z_t[:,n,:])

For each patch position n ∈ {0, ..., N-1}:
    C_n = (1/B) · ẑ_centered[:,n,:]^T · z_centered[:,n,:]    ∈ ℝ^{d×d}

L_info_dense = −(1/N) Σ_n Tr(C_n)
```

This computes the cross-covariance trace AT EACH spatial position independently,
then averages. It encourages each patch token in the prediction to align with
its corresponding patch token in the target.

**Why this matters:** The pooled L_info collapses spatial information before
measuring alignment. L_info_dense measures alignment while preserving the
spatial correspondence. If we want to test whether information is preserved
at the patch level (not just globally), this is the correct metric.

### Definition 3.6 — Covariance Regularization (L_cov, optional stabilizer)

```
z_pool = mean(ẑ_t, dim=1)           ∈ ℝ^{B×d}
z_centered = z_pool − mean(z_pool, dim=0)
C = (1/B) · z_centered^T · z_centered    ∈ ℝ^{d×d}

L_cov = ||C − I||_F² = Σ_{i,j} (C[i,j] − δ_{ij})²
```

Penalizes deviation from identity covariance. Use only if training is unstable.

### Definition 3.7 — Combined Loss Per Condition

Let λ₁, λ₂, λ₃ be non-negative hyperparameters.

| Condition | Total Loss |
|-----------|-----------|
| A | L_pred(ẑ, z_t) — no stop-grad, no regularization |
| B | L_pred(ẑ, sg(z_t)) — stop-grad only |
| C | L_pred(ẑ, sg(z_t)) + λ₁·SIGReg_global(z_c) |
| D1 | L_pred(ẑ, z_t) + λ₁·SIGReg_token(z_c) |
| D2 | L_pred(ẑ, z_t) + λ₁·SIGReg_channel(z_c) |
| D3 | L_pred(ẑ, z_t) + λ₁·SIGReg_global(z_c) |
| E_pooled | L_pred(ẑ, z_t) + λ₁·SIGReg_global(z_c) + λ₂·L_info(ẑ_pool, z_t_pool) |
| E_dense | L_pred(ẑ, z_t) + λ₁·SIGReg_global(z_c) + λ₂·L_info_dense(ẑ, z_t) |
| F | L_pred(ẑ, z_t) + λ₂·L_info_dense(ẑ, z_t) — ablation: no SIGReg |

**SIGReg is always applied to the ADAPTER OUTPUT z_c, not the predictor output ẑ.**
This regularizes the representation space, not the predictions.

**Conditions without stop-gradient (A, D1, D2, D3, E, F):**
z_t = A_θ(f_t) with gradients flowing back to θ.

**Conditions with stop-gradient (B, C):**
z_t = sg(A_θ(f_t)) — values are computed but no gradient flows to θ through this path.

---

## 4. Evaluation Metrics

### Definition 4.1 — Effective Rank

Given Z ∈ ℝ^{K×d} with singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_r:

```
p_i = σ_i² / Σ_j σ_j²          (normalized energy per component)
erank(Z) = exp(−Σ_i p_i · ln(p_i))    (exponential of Shannon entropy)
```

Range: [1, min(K, d)].
- erank = 1 → complete collapse (all energy in one component)
- erank = d → maximum diversity (uniform energy distribution)

### Definition 4.2 — Cross-Covariance Trace (diagnostic, no gradient)

```
eval_xcov_trace(z_c, z_t) = Tr(Cov(z_c_pool, z_t_pool))
```

Same formula as L_info but without the negative sign, computed with torch.no_grad().
Higher = more linear dependence between context and target representations.

### Definition 4.3 — Neighbor Token Correlation

Given z ∈ ℝ^{B×N×d} with N = 14² = 196:

```
z_grid = z.reshape(B, 14, 14, d)

cos_h = cosine_similarity(z_grid[:, :, :-1, :], z_grid[:, :, 1:, :])
cos_v = cosine_similarity(z_grid[:, :-1, :, :], z_grid[:, 1:, :, :])

neighbor_corr = (mean(cos_h) + mean(cos_v)) / 2
```

Range: [-1, 1]. Typical range for useful representations: [0.3, 0.9].
- High (>0.7): strong spatial coherence (neighbors encode similar content)
- Low (<0.1): spatial structure destroyed (each token independent)

### Definition 4.4 — Token Diversity

```
z_norm = L2_normalize(z, dim=-1)         ∈ ℝ^{B×N×d}
sim = z_norm @ z_norm.T                  ∈ ℝ^{B×N×N}  (per sample)
token_diversity = 1 − mean(off_diagonal(sim))
```

Range: [0, 1]. Measures how different patch tokens are from each other.
- High: tokens are diverse (encoding different spatial content)
- Low: tokens are similar (possible collapse to spatially uniform representation)

### Definition 4.5 — Linear Probe Accuracy

Train a logistic regression classifier on frozen representations:
```
Input:  z_pool = mean(A_θ(f), dim=1) ∈ ℝ^{B×d}  (pooled adapter output)
Target: SSv2 clip class ∈ {0, ..., 173}
```

The probe is trained on a held-out portion of validation clips.
The probe itself has NO gradient flowing back to the adapter or predictor.
This measures how much task-relevant information is linearly accessible
in the learned representation space.

### Definition 4.6 — InfoNCE Mutual Information Lower Bound

As a secondary MI estimate, independent of Gaussianity assumptions:

```
Given aligned pairs (z_c_pool^(i), z_t_pool^(i)) for i = 1..B:

score(i, j) = z_c_pool^(i) · z_t_pool^(j) / τ

I_NCE = log(B) - (1/B) Σ_i log(Σ_j exp(score(i,j))) + (1/B) Σ_i score(i,i)
```

This is the InfoNCE lower bound on I(z_c; z_t). It does not assume Gaussianity.
We use it as a diagnostic metric (no gradient), complementary to cross-covariance trace.

---

## 5. Theoretical Predictions Per Condition

### Condition A — No Regularization, No Stop-Gradient

**Mechanism:** Both ẑ_t and z_t are functions of θ. Trivial solution: A_θ → constant.
**Prediction:** Adapter outputs collapse to a constant vector within O(1000) steps.
**Observable:** erank → 1, min singular value → 0, neighbor_corr → 1 (all tokens identical).

### Condition B — Stop-Gradient Only

**Mechanism:** z_t = sg(A_θ(f_t)). But A_θ is shared: if θ moves toward collapse,
the targets also become constant (computed with same weights, just detached).
**Prediction:** Collapse also occurs, possibly slower than A.
**Justification:** Stop-gradient breaks gradient flow but not the weight-sharing
symmetry. This is why V-JEPA needs EMA in addition to stop-gradient.

**Alternative possibility:** If collapse is slower or doesn't occur, it reveals
that stop-gradient creates enough optimization asymmetry (context path optimized,
target path stale-by-one-step) to partially stabilize. This would be an
interesting finding in itself.

### Condition C — Stop-Gradient + Global SIGReg

**Mechanism:** SIGReg on z_c prevents the adapter from collapsing (Gaussianity
enforced on context representations). Stop-gradient on z_t means targets are
computed but not optimized.
**Prediction:** erank high, collapse prevented. But:
- SIGReg pushes z_c toward isotropic Gaussian
- The targets z_t (from same adapter) are also becoming more Gaussian
- Whether this preserves spatial structure or task-relevant information is the
  central experimental question.

### Conditions D1, D2, D3 — SIGReg Axis Variants (No Stop-Gradient)

D1 (per-token): Strongest spatial destruction. Each token independently pushed to Gaussian.
D2 (per-channel): Moderate. Each feature dimension independently Gaussian.
D3 (global): Weakest spatial intervention. Only the pooled representation is Gaussian.

**Ordering prediction:** Spatial coherence: D3 > D2 > D1.
Probe accuracy: D3 ≥ D2 > D1 (if spatial structure helps downstream tasks).

### Condition E — SIGReg + Information Constraint

**Mechanism:** SIGReg prevents collapse. L_info (dense or pooled) encourages
the predictor output to align with target representations, providing an additional
training signal beyond MSE prediction.
**Prediction:** If the hypothesis is correct, E should have:
- erank comparable to C/D conditions (SIGReg maintaining rank)
- Higher probe accuracy than C/D (information constraint preserving useful structure)
- Higher cross-covariance trace than C/D (by construction)

### Condition F — Information Constraint Only (No SIGReg)

**Ablation purpose:** Isolate the contribution of L_info.
Without SIGReg, collapse may occur unless L_info alone prevents it.
L_info alone may NOT prevent collapse because:
- If A_θ → constant, then ẑ_pool and z_t_pool are both constant
- Cov(constant, constant) = 0 → L_info = 0 (not penalizing collapse)
- So L_info is NOT a collapse prevention mechanism

**Prediction:** F collapses or produces poor representations, confirming that
SIGReg is necessary (but not sufficient) and L_info is helpful (but not alone sufficient).

---

## 6. What This Experiment Can and Cannot Establish

### CAN establish (if results align with predictions):
1. SIGReg prevents collapse of the adapter representation (measurable: erank)
2. SIGReg on patch tokens does/does not destroy spatial structure (measurable: neighbor_corr)
3. The axis of SIGReg application matters for spatial preservation (D1 vs D2 vs D3)
4. Adding a cross-covariance alignment term improves task performance beyond SIGReg alone
5. The combination {SIGReg + L_info} outperforms either alone

### CANNOT establish:
1. That these results generalize to end-to-end pixel training (we use frozen features + adapter)
2. That Tr(C) equals mutual information (it is a proxy, not MI itself)
3. That our findings transfer beyond SSv2 to other datasets
4. That our adapter architecture is optimal — findings may depend on adapter design

### The honest framing:
"We study the interaction between distributional regularization (SIGReg) and
information preservation in patch-level JEPA representations, using a lightweight
trainable adapter over frozen V-JEPA 2.1 features as a controlled experimental
setting. Our results characterize when SIGReg helps, when it hurts, and what
additional constraints are needed for useful representations at the patch level."
