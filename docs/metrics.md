# GAP 1 Experiment — Metrics Reference

**Source of truth:** All metric definitions trace to `foundations.md` Section 4.
**Implementation:** `training/metrics.py` — **FROZEN after first job submission (CLAUDE.md Rule 6).**
All conditions are evaluated with identical metric code.

---

## Overview

Metrics fall into three groups, each testing a different aspect of the representations:

| Group | Metrics | Question answered |
|-------|---------|-------------------|
| **Collapse** | `erank`, `min_sv`, `var_top1` | Did the adapter collapse? |
| **Spatial Structure** | `ncorr_adapter`, `ncorr_pred`, `tokdiv_adapter` | Is spatial coherence preserved? |
| **Information** | `xcov_trace`, `xcov_trace_dense`, `infonce_mi` | Does the representation carry useful signal? |

All metrics are computed on the **concatenation of all validation batches**, never averaged per-batch (non-linear metrics like effective rank are not additive across batches — see CLAUDE.md Rule 6).

---

## 1. Collapse Metrics

### `eval/erank` — Effective Rank

**Definition (foundations.md §4.1):**

The adapter output `z_c` is flattened to shape `[K×N, d]` (K clips × 196 patches × 256 dims). SVD gives singular values `σ₁ ≥ σ₂ ≥ ... ≥ σ_r`. Define the normalised energy distribution:

```
pᵢ = σᵢ² / Σⱼ σⱼ²

erank = exp( −Σᵢ pᵢ · ln pᵢ )
```

This is the exponential of the Shannon entropy of the singular value energy spectrum.

**Range:** `[1, min(K×N, d)]` — in practice `[1, 256]`.

| Value | Interpretation |
|-------|---------------|
| `erank ≈ 1` | Complete collapse — all energy in one direction |
| `erank ≈ d` | Uniform energy — maximally diverse representation |
| `erank = 71.92` | Raw V-JEPA 2.1 features (our quality ceiling) |

**Why this detects collapse:** When the adapter maps all inputs to a constant, the matrix Z has rank 1 — one nonzero singular value, all others exactly zero — giving entropy = 0 and erank = exp(0) = 1.

**Key experimental values:**
- Condition A at step 2000: `erank = 1.01` — textbook collapse ✅
- V-JEPA 2.1 baseline: `erank = 71.92`

---

### `eval/min_sv` — Minimum Singular Value

**Definition:** The smallest singular value of the flattened adapter output.

```
min_sv = σ_min(Z)    where Z ∈ ℝ^{K×N × d}
```

**Range:** `[0, ∞)`

Under collapse, `min_sv → 0` as the representation degenerates to a subspace. More sensitive than erank to *partial* collapse where only some dimensions die (a few small singular values won't move erank much, but will visibly drop min_sv).

---

### `eval/var_top1` — Variance in Top Singular Direction

**Definition:** Fraction of total variance explained by the dominant singular value.

```
var_top1 = σ₁² / Σᵢ σᵢ²
```

**Range:** `[1/rank, 1.0]`

| Value | Interpretation |
|-------|---------------|
| `→ 1.0` | Collapse — one direction explains everything |
| `≈ 0.004` | Uniform (`1/256`) — no dominant direction |

Complements erank: a high erank with high var_top1 indicates that one direction is still disproportionately dominant even if the distribution appears spread.

---

## 2. Spatial Structure Metrics

### `eval/ncorr_adapter` — Neighbour Token Correlation

**Definition (foundations.md §4.3):**

The 196 adapter output tokens are reshaped to a `14×14` spatial grid matching the ViT patch layout:

```
z_grid = z_c.reshape(B, 14, 14, d)

cos_h = cosine_similarity(z_grid[:, :, :-1, :], z_grid[:, :, 1:, :])   # horizontal pairs
cos_v = cosine_similarity(z_grid[:, :-1, :, :], z_grid[:, 1:, :, :])   # vertical pairs

ncorr = (mean(cos_h) + mean(cos_v)) / 2
```

**Range:** `[-1, 1]` — useful range for this experiment: `[0.1, 0.9]`

| Value | Interpretation |
|-------|---------------|
| `> 0.7` | Strong spatial coherence — neighbouring patches encode similar content |
| `0.3–0.7` | Moderate — some spatial structure preserved |
| `< 0.1` | Spatial structure destroyed |
| `→ 1.0` | All 196 patches identical (trivial — see note below) |

**Critical note on interpreting high ncorr:** Condition A shows `ncorr → 0.994` because all 196 patches collapse to the *same constant vector*, making them trivially maximally similar. High ncorr is only meaningful when `erank` is also high. Always interpret ncorr and erank jointly.

**What this tests for the paper (Claim 2):** Whether SIGReg's regularization axis (per-token D1 vs per-channel D2 vs global D3) destroys the natural spatial coherence present in V-JEPA 2.1 features (`raw_ncorr = 0.683`). Expected ordering: `D1 < D2 < D3 ≈ C < E`.

---

### `eval/ncorr_pred` — Neighbour Correlation of Predictor Output

Same computation as `ncorr_adapter` but applied to the predictor output `ẑ` rather than `z_c`.

Measures whether the *predicted* target representation preserves spatial structure. Under Condition E, SIGReg on z_c combined with L_info_dense should force ẑ to track z_t spatially, yielding high ncorr_pred. Under Condition A, ẑ also collapses so ncorr_pred → 1 trivially.

---

### `eval/tokdiv_adapter` — Token Diversity

**Definition (foundations.md §4.4):**

```
z_norm = L2_normalize(z_c, dim=-1)           # [B, 196, 256]
S      = z_norm @ z_norm.transpose(1, 2)     # [B, 196, 196] pairwise cosine sim
tokdiv = 1 − mean(off-diagonal elements of S)
```

**Range:** `[0, 1]`

| Value | Interpretation |
|-------|---------------|
| `→ 0` | All 196 patches identical within each sample (within-sample spatial collapse) |
| `→ 1` | All 196 patches mutually orthogonal (maximally diverse) |

**Distinction from ncorr:** ncorr measures similarity between *adjacent* patches (spatial coherence). tokdiv measures dissimilarity between *all* patches (within-sample diversity). A healthy representation has moderate ncorr (neighbours are similar) and moderate-high tokdiv (patches still vary across the image). Condition A: `tokdiv ≈ 0.001` (all patches identical), `ncorr ≈ 0.994` (trivially — they're the same vector).

---

## 3. Information Metrics

### `eval/xcov_trace` — Cross-Covariance Trace (Pooled)

**Definition (foundations.md §4.2):**

Spatial pooling first — average the 196 patch tokens to get a single vector per clip:

```
z_c_pool = mean(z_c, dim=1)    # [B, 256]
z_t_pool = mean(z_t, dim=1)    # [B, 256]

z_c_c = z_c_pool − mean(z_c_pool, dim=0)    # centre
z_t_c = z_t_pool − mean(z_t_pool, dim=0)

C = (1/B) · z_c_cᵀ z_t_c      # [256, 256] cross-covariance matrix
xcov_trace = Tr(C)
```

**Range:** `(-∞, +∞)` — positive values indicate linear dependence.

**Interpretation:** How much does the pooled context representation linearly predict the pooled target representation? Under collapse, both become constant → centred versions are zero → `C = 0` → `xcov_trace = 0`. Under a healthy representation, context and target from the same clip should be strongly correlated.

This is the **diagnostic** version of L_info (positive, not negated). High values in Condition E confirm that SIGReg + L_info produces context representations that covary with targets.

---

### `eval/xcov_trace_dense` — Dense Cross-Covariance Trace

**Definition:** Same formula as `xcov_trace` but computed independently for each of the 196 patch positions and averaged:

```
xcov_dense = (1/N) Σₙ Tr(Cov_n(ẑ, z_t))
```

where Cov_n is the cross-covariance at patch position n across the batch.

**Why this matters for the paper:** The paper's thesis is about *spatial* structure. The dense version operates patch-by-patch, directly measuring whether each spatial position in the prediction carries information about the corresponding target position. Condition E uses `L_info_dense` in training, so `xcov_trace_dense` should be highest for Condition E. If it's also high for C or D3 (which don't use L_info), that's evidence that SIGReg alone preserves spatial alignment.

---

### `eval/infonce_mi` — InfoNCE Mutual Information Bound

**Definition (foundations.md §4.6):**

A non-parametric lower bound on mutual information `I(z_c; z_t)` that does not assume Gaussianity:

```
scores(i, j) = z_c_pool⁽ⁱ⁾ · z_t_pool⁽ʲ⁾ / τ      (B×B similarity matrix, τ=0.1)

I_NCE = log(B) + mean(diag(scores)) − mean(log-sum-exp(scores, dim=1))
```

Maximum value = `log(B)` (perfect alignment of context and target in every pair). Value near 0 or negative → representations carry no mutual information.

**Why this complements xcov_trace:** xcov_trace is linear — it detects linear correlations between z_c and z_t. InfoNCE is non-linear (contrastive) — it detects any structure that makes same-clip pairs more similar than cross-clip pairs, regardless of whether the relationship is linear. Together they provide a fuller picture of representational quality.

**Key values:**
- Condition A: `infonce ≈ 0` — collapsed representations carry no mutual information
- Condition B: `infonce = −1793` to `−4295` (at steps 2000–4000) — large negative because both z_c and z_t are collapsing, making all similarity scores equally large regardless of pairing

---

## 4. Baseline Metrics (Pre-Training Reference)

Computed once before training on raw V-JEPA 2.1 features (no adapter, no training):

### `baseline/raw_erank`
Effective rank of raw features `f_c` flattened to `[K×196, 1024]`.
- **Value: 71.92** — V-JEPA 2.1 features span ~72 effective dimensions in 1024-dim space.
- This is the quality ceiling. No condition can exceed this (adapter reduces dimensionality to 256).

### `baseline/raw_ncorr`
Neighbour token correlation of raw features.
- **Value: 0.683** — strong natural spatial coherence from V-JEPA 2.1's video pretraining.
- This is what we are trying to preserve through the adapter. Conditions that destroy spatial structure will show `ncorr_adapter << 0.683`.

---

## 5. Metrics Logged Per Run

| Metric key (W&B) | Logged at | Shape input | Notes |
|-----------------|-----------|-------------|-------|
| `train/loss_total` | every step | scalar | Sum of all active loss components |
| `train/loss_pred` | every step | scalar | MSE(ẑ, z_t) |
| `train/loss_sig` | every step | scalar | λ₁ · SIGReg(z_c) |
| `train/loss_info` | every step | scalar | λ₂ · L_info_dense |
| `train/loss_cov` | every step | scalar | λ₃ · L_cov (optional) |
| `train/grad_norm` | every step | scalar | Gradient norm before clipping |
| `train/lr` | every step | scalar | Current learning rate |
| `eval/erank` | every 500 steps | [K×N, d] | Primary collapse indicator |
| `eval/min_sv` | every 500 steps | [K×N, d] | |
| `eval/var_top1` | every 500 steps | [K×N, d] | |
| `eval/ncorr_adapter` | every 500 steps | [K, N, d] | Primary spatial metric |
| `eval/ncorr_pred` | every 500 steps | [K, N, d] | |
| `eval/tokdiv_adapter` | every 500 steps | [K, N, d] | |
| `eval/xcov_trace` | every 500 steps | [K, N, d] | Primary information metric |
| `eval/xcov_trace_dense` | every 500 steps | [K, N, d] | |
| `eval/infonce_mi` | every 500 steps | [K, N, d] | |
| `baseline/raw_erank` | step 0 only | [K, N, D] | V-JEPA 2.1 reference |
| `baseline/raw_ncorr` | step 0 only | [K, N, D] | V-JEPA 2.1 reference |
| `probe/acc_train` | every 5000 steps | — | Linear probe train accuracy |
| `probe/acc_val` | every 5000 steps | — | Linear probe val accuracy |

---

## 6. Interpreting Results: What to Look For

### The three paper claims and which metrics test them

**Claim 1 — SIGReg prevents collapse:**
Compare erank across conditions. Must see: `erank_A ≈ 1`, `erank_{C,D1,D2,D3,E} >> 1`.

**Claim 2 — SIGReg axis controls spatial structure:**
Compare ncorr_adapter across D1, D2, D3. Must see: `ncorr_D1 < ncorr_D2 < ncorr_D3`.

**Claim 3 — L_info recovers information:**
Compare D3 vs E on probe accuracy and xcov_trace_dense. Must see: `probe_E > probe_D3` and `xcov_dense_E > xcov_dense_D3`.

### Reading a metric pair

Always read ncorr alongside erank:

| erank | ncorr | Diagnosis |
|-------|-------|-----------|
| ≈ 1 | ≈ 1 | Collapsed — all patches same constant. ncorr is trivially 1. |
| High | High | Healthy — diverse representations, spatial coherence preserved. |
| High | Low | SIGReg destroyed spatial structure (expected for D1). |
| Low | Moderate | Partial collapse — some diversity but dimensions dying. |
