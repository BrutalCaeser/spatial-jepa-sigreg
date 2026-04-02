# build_spec.md — GAP 1 Research Build Specification

**Version:** 3.0 (corrected architecture, mathematically validated)
**Platform:** Northeastern University Discovery Cluster (SLURM, A100 80GB)
**Language:** Python 3.10
**Mathematical reference:** All formulas trace to `foundations.md` definitions.

---

## 0. Read This First

This document supersedes all prior GAP 1 specifications.

The previous spec had three fatal flaws (documented in review):
1. Collapse cannot occur with frozen features and no trainable encoder
2. L_info on frozen features produces zero gradient
3. Stop-gradient on pre-extracted features is a no-op

This spec fixes all three by introducing a **trainable adapter** between frozen
V-JEPA 2.1 features and the predictor. The adapter serves as a controlled
encoder surrogate: it can collapse, it can be regularized, and it creates
a representation space where our hypotheses are testable.

---

## 1. Scientific Objective

### 1.1 Research Question

**Primary:** Does distributional regularization (SIGReg), when applied to
patch-level representations, preserve the spatial structure and task-relevant
information present in the input features?

**Secondary:** Can an explicit cross-view alignment constraint (L_info) recover
information that SIGReg alone does not preserve?

### 1.2 The Hypothesis (Three Independent Claims)

**Claim 1 — Collapse Prevention:**
SIGReg prevents representation collapse of the adapter (measured by effective rank
and singular value spectrum).

**Claim 2 — Spatial Structure:**
SIGReg applied at the token level destroys spatial coherence between neighboring
patch tokens. Applied at the global (pooled) level, it partially preserves
spatial structure. (Measured by neighbor token correlation.)

**Claim 3 — Information Preservation:**
SIGReg alone does NOT maximize cross-view information alignment.
Adding L_info (dense cross-covariance trace) alongside SIGReg yields higher
downstream probe accuracy than SIGReg alone.

### 1.3 Paper Positioning

This is a **diagnostic paper** studying the interaction between distributional
regularization and information preservation in patch-level JEPA representations.

Core message: **Preventing collapse via Gaussianity enforcement is necessary but
creates a tension with spatial structure preservation. Cross-view alignment
resolves this tension.**

The paper has three contributions:
1. Identifying that SIGReg's Gaussianity pressure can destroy spatial structure
   at the patch token level (negative result, principled analysis)
2. Characterizing how the axis of SIGReg application (global vs per-token vs
   per-channel) controls the trade-off between collapse prevention and
   spatial preservation
3. Proposing and validating that a cross-covariance alignment term, combined
   with SIGReg, achieves both collapse prevention and information preservation

### 1.4 Target Venue and Timeline

**Primary target:** NeurIPS 2026 (abstract deadline: ~May 15, paper deadline: ~May 22, 2026)
**Backup target:** ICLR 2027 workshop/main conference (October 2026)

**Honest assessment:** NeurIPS 2026 is 7 weeks away. This is feasible ONLY if:
- Week 1: Setup, data, adapter implementation, unit tests
- Week 2: All conditions running, A confirmed collapsed
- Week 3: All conditions complete, probe evaluations
- Week 4: Analysis, figures, core results table
- Week 5: Writing (intro, method, results)
- Week 6: Writing (related work, conclusion), internal review
- Week 7: Polish, final figures, submission

If any week slips by more than 2 days, fall back to NeurIPS workshop paper
(shorter, fewer conditions: A, B, C, E only).

---

## 2. Architecture

### 2.1 Data Flow

```
Frozen V-JEPA 2.1 Features (.pt files)
       │
       ▼
┌──────────────┐
│ Trainable    │  A_θ: ℝ^{1024} → ℝ^{d}  (token-wise MLP, shared weights)
│ Adapter      │  Applied independently to each of B×N tokens
└──────┬───────┘
       │
       ├──── z_c = A_θ(f_c)  ∈ ℝ^{B×196×d}  (context, always has gradient)
       │
       ├──── z_t = A_θ(f_t)  ∈ ℝ^{B×196×d}  (target, gradient depends on condition)
       │
       ▼
┌──────────────┐
│ Predictor    │  P_φ: (z_c, label) → ẑ_t  (transformer with AdaLN)
│              │  Input: context adapted features + action embedding
└──────┬───────┘
       │
       ▼
    ẑ_t ∈ ℝ^{B×196×d}  (predicted target representation)
       │
       ▼
   Loss computation (condition-dependent)
```

### 2.2 Why This Architecture Tests Our Hypothesis

| Component | Role in experiment |
|-----------|-------------------|
| Frozen V-JEPA 2.1 features | Provide high-quality patch tokens WITH spatial structure (V-JEPA 2.1 dense features) |
| Trainable adapter | Creates a representation space that CAN collapse — the locus of our study |
| SIGReg on adapter output | Tests whether Gaussianity enforcement preserves or destroys the spatial structure inherited from V-JEPA 2.1 |
| Predictor | Provides the prediction loss that creates the collapse incentive |
| L_info on predictor output | Tests whether cross-view alignment recovers lost information |

### 2.3 Adapter Specification

```python
class PatchAdapter(nn.Module):
    def __init__(self, D_in=1024, D_out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_in, D_in),
            nn.GELU(),
            nn.Linear(D_in, D_out),
        )
        # Initialize near-identity for stability:
        # First layer ≈ identity, second layer ≈ random projection
        nn.init.eye_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_normal_(self.net[2].weight, gain=0.1)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, f):
        # f: [B, N, D_in]
        return self.net(f)  # [B, N, D_out] — applied token-wise
```

**No normalization layers.** BatchNorm or LayerNorm would mask collapse signals.
**Near-identity initialization.** Ensures early training starts with features
close to the original V-JEPA 2.1 features (projected to lower dim).

### 2.4 Predictor Specification

Same as previous spec: 6-layer transformer with AdaLN, 8 heads, d_inner = 2*d.
Action conditioning: `nn.Embedding(174, d_action)` → AdaLN scale/shift.

### 2.5 Hyperparameters

```yaml
# Adapter
D_in: 1024          # V-JEPA 2.1 ViT-G hidden dim
d: 256               # adapter output dim (ablate: 128, 256, 512)

# Predictor
d_inner: 512         # predictor FFN hidden dim
d_action: 64         # action embedding dim
n_layers: 6
n_heads: 8
dropout: 0.1

# Training
batch_size: 32
n_steps: 50000
learning_rate: 3e-4
weight_decay: 0.01
warmup_steps: 2000
lr_schedule: cosine_annealing

# Regularization (defaults, overridden per condition)
lambda_1: 0.1        # SIGReg weight
lambda_2: 0.1        # L_info weight
lambda_3: 0.0        # L_cov weight (optional stabilizer)
M_projections: 1024  # SIGReg random projections
```

---

## 3. Experimental Conditions

### 3.1 Condition Table

| ID | Stop-Grad | SIGReg | L_info | What it tests |
|----|-----------|--------|--------|---------------|
| A  | No  | None   | No  | Collapse baseline — confirms collapse is possible |
| B  | Yes | None   | No  | Stop-gradient alone — does it prevent collapse? |
| C  | Yes | Global | No  | SIGReg + stop-grad — standard approach, measures spatial impact |
| D1 | No  | Token  | No  | Per-token SIGReg — maximal spatial destruction |
| D2 | No  | Channel| No  | Per-channel SIGReg — intermediate |
| D3 | No  | Global | No  | Global SIGReg — minimal spatial intervention (= LeWM style) |
| E  | No  | Global | Dense | **Proposed method:** SIGReg + dense cross-view alignment |
| F  | No  | None   | Dense | Ablation: L_info alone — does it prevent collapse? |

### 3.2 Minimum Viable Experiment (5 conditions, 40 GPU-hours)

If compute-constrained: **A → B → D3 → E → F**

This tests:
- A: collapse exists → B: stop-grad insufficient
- D3 vs E: SIGReg alone vs SIGReg + L_info
- F: L_info alone insufficient

### 3.3 Predicted Outcomes Table

| ID | erank | probe_acc | neighbor_corr | xcov_trace | Collapse? |
|----|-------|-----------|---------------|------------|-----------|
| A  | →1    | ~chance   | →1 (all same) | ~0         | Yes       |
| B  | Low-moderate | Low | Low-moderate | Low       | Likely slow collapse |
| C  | High  | Moderate  | Moderate      | Moderate   | No        |
| D1 | High  | Low       | **Low**       | Low        | No        |
| D2 | High  | Moderate-low | Moderate-low | Low-moderate | No   |
| D3 | High  | Moderate  | Moderate      | Moderate   | No        |
| E  | High  | **Highest** | Moderate-high | **Highest** | No    |
| F  | Low   | Low       | Variable      | Variable   | Likely yes |

### 3.4 Condition A — Collapse Baseline

**Loss:** L_pred(ẑ, z_t) — no stop-gradient, no regularization.

**Mechanism:** Gradient flows through both ẑ (via predictor and adapter) and z_t
(directly through adapter). Adapter can satisfy L_pred = 0 by mapping everything
to a constant.

**Prediction:** Collapse within 2000 steps. erank → 1.

**Halt rule:** If A does NOT collapse after 5000 steps (erank > 5),
something is wrong. Possible causes:
- Implicit regularization from weight decay (try weight_decay=0)
- Near-identity initialization preventing collapse (try random init)
- Gradient clipping preventing large adapter updates (check grad norm)

### 3.5 Condition B — Stop-Gradient Only

**Loss:** L_pred(ẑ, sg(z_t))

**Mechanism:** Target z_t computed with current adapter weights but detached.
However, adapter weights are shared — if they drift toward collapse, targets
also become constant (computed with same θ, just one step stale).

**Two possible outcomes (both are informative):**
1. B also collapses → confirms stop-grad alone is insufficient for shared-weight adapter
   (consistent with JEPA literature requiring EMA)
2. B does NOT collapse → the optimization asymmetry from stop-grad provides
   enough stabilization (interesting finding about stop-grad dynamics)

Either way, we document the result honestly.

### 3.6 Conditions D1, D2, D3 — SIGReg Axis Study

All use NO stop-gradient. SIGReg is the sole collapse prevention mechanism.

**D1 (per-token):** `SIGReg(z_c.reshape(B*N, d))`
Each (sample, token) pair independently pushed toward Gaussian.
Strongest isotropy pressure. Prediction: destroys spatial coherence.

**D2 (per-channel):** `(1/d) Σ_j EP_test(z_flat[:, j])`
Each feature dimension independently Gaussian across all B*N samples.
Moderate pressure. No inter-dimension correlation control.

**D3 (global):** `SIGReg(z_c.mean(dim=1))`
Only the spatially-pooled representation is Gaussian.
Weakest pressure. Most spatial structure preserved.
This is closest to what LeWM does (but at patch level rather than CLS).

### 3.7 Condition E — SIGReg + Dense Information Constraint

**Loss:** L_pred(ẑ, z_t) + λ₁·SIGReg_global(z_c) + λ₂·L_info_dense(ẑ, z_t)

**No stop-gradient.** SIGReg prevents collapse. L_info_dense encourages
per-patch-position alignment between predictor output and target.

**Why L_info_dense rather than L_info_pooled:**
The paper's thesis is about SPATIAL structure. Using a pooled information term
would be inconsistent — we'd be claiming to preserve spatial information
using a loss that discards spatial information. L_info_dense operates at each
patch position independently, directly measuring and encouraging spatial alignment.

### 3.8 Condition F — Information Constraint Only (Ablation)

**Loss:** L_pred(ẑ, z_t) + λ₂·L_info_dense(ẑ, z_t)

No SIGReg. Tests whether L_info alone prevents collapse.
**Expected result:** F collapses or produces poor representations.
Reason: under collapse, both ẑ and z_t are constant, so Cov(ẑ, z_t) = 0
and L_info_dense = 0. L_info does not penalize collapse.

---

## 4. Dataset

### 4.1 Something-Something v2 (SSv2) — Pre-extracted Features

**Source:** V-JEPA 2.1 ViT-G encoder, frozen.
**Features per clip:** T frames × 196 patches × 1024 dimensions.
**We extract:** k=2 consecutive frames (f_t, f_{t+1}) per clip.
**Labels:** 174 action classes (used for predictor AdaLN conditioning + probe evaluation).
**Subset:** ~20K clips from SSv2 part-00. Sufficient for all 8 conditions.

**Feature extraction pipeline:**
1. Load V-JEPA 2.1 ViT-G (from Meta's public release)
2. For each clip: sample 2 consecutive frames, extract patch tokens [196, 1024]
3. Save as .pt files: `{clip_id}_frame{t}.pt`, `{clip_id}_frame{t+1}.pt`
4. Save metadata: `{clip_id}_label.json`

**Storage estimate:** 20K clips × 2 frames × 196 × 1024 × 4 bytes ≈ 30 GB

### 4.2 Train/Val/Test Split

- Train: 16K clips (for training all conditions)
- Val: 2K clips (for metrics computed during training: erank, neighbor_corr, xcov_trace)
- Test: 2K clips (for final linear probe evaluation, touched ONCE per condition)

The linear probe uses a 50/50 split of the test set (1K train probe, 1K eval probe).

### 4.3 V-JEPA 2.1 Feature Quality Verification

Before any experiment, verify that frozen features have spatial structure:

```python
# Load a batch of frozen features
f = load_features(val_set, batch_size=32)  # [32, 196, 1024]

# Check neighbor correlation of RAW frozen features
from metrics import neighbor_correlation
raw_ncorr = neighbor_correlation(f)
print(f"Raw V-JEPA 2.1 neighbor correlation: {raw_ncorr:.3f}")
# Expected: > 0.3 (V-JEPA 2.1 has spatial structure per their paper)

# Check effective rank of raw features
from metrics import effective_rank
raw_erank = effective_rank(f.reshape(-1, 1024))
print(f"Raw V-JEPA 2.1 effective rank: {raw_erank:.1f}")
# Expected: > 20 (diverse, non-collapsed features)
```

**If raw neighbor_corr < 0.1:** The V-JEPA 2.1 features don't have spatial structure.
Our experiment cannot study preservation of something that doesn't exist.
Action: verify you're using V-JEPA 2.1 (not V-JEPA 2), verify patch tokens
(not CLS), verify ViT-G (not smaller variant).

---

## 5. Metrics — Complete Specification

All metrics computed every 500 training steps on validation set.
All metrics logged to W&B. Metrics code is FROZEN before first job.

### 5.1 Collapse Metrics

| Metric | Key | When | Purpose |
|--------|-----|------|---------|
| Effective rank | `eval/erank` | Every 500 steps | Collapse detection (primary) |
| Min singular value | `eval/min_sv` | Every 500 steps | Collapse severity |
| Variance explained by top SV | `eval/var_top1` | Every 500 steps | Concentration measure |

### 5.2 Information Metrics

| Metric | Key | When | Purpose |
|--------|-----|------|---------|
| Cross-cov trace (pooled) | `eval/xcov_trace` | Every 500 steps | Information alignment (global) |
| Cross-cov trace (dense) | `eval/xcov_trace_dense` | Every 500 steps | Information alignment (spatial) |
| InfoNCE MI bound | `eval/infonce_mi` | Every 2500 steps | Non-parametric MI estimate |
| Linear probe top-1 | `eval/probe_top1` | Every 5000 steps | Downstream task utility |
| Linear probe top-5 | `eval/probe_top5` | Every 5000 steps | Downstream task utility |

### 5.3 Spatial Metrics

| Metric | Key | When | Purpose |
|--------|-----|------|---------|
| Neighbor correlation (adapter) | `eval/ncorr_adapter` | Every 500 steps | Spatial structure of z_c |
| Neighbor correlation (predictor) | `eval/ncorr_pred` | Every 500 steps | Spatial structure of ẑ |
| Token diversity (adapter) | `eval/tokdiv_adapter` | Every 500 steps | Token differentiation |

### 5.4 Baseline Metrics (computed once before training)

| Metric | Key | Purpose |
|--------|-----|---------|
| Raw feature erank | `baseline/raw_erank` | Reference for frozen V-JEPA 2.1 features |
| Raw feature neighbor_corr | `baseline/raw_ncorr` | Reference spatial structure |
| Raw feature probe accuracy | `baseline/raw_probe` | Reference information content |

---

## 6. Failure Modes

### FM1 — Condition A Does Not Collapse

**Detection:** erank > 5 after 5000 steps in Condition A.

**Possible causes:**
1. Weight decay acts as implicit regularizer → set weight_decay=0 for Condition A
2. Near-identity init is too stable → use random init for Condition A
3. Adapter bottleneck (d << D) naturally prevents collapse → check if d matters
4. Gradient clipping at 1.0 prevents adapter from reaching degenerate solution → raise to 10.0

**Action:** Systematically eliminate each cause. If A still doesn't collapse,
this is itself a finding: the adapter architecture has implicit collapse resistance.
Document it, remove A from the main narrative, and reframe: "we study
how SIGReg modifies already-stable representations."

### FM2 — Condition B Does Not Collapse

**Not a failure.** If B is stable without SIGReg, that is an interesting finding
about stop-gradient dynamics with shared-weight adapters. Document it:
"Stop-gradient provides sufficient optimization asymmetry to stabilize
adapter training, even without EMA."

### FM3 — SIGReg Conditions Have Lower Probe Accuracy Than B

**Expected for some conditions (D1).** If true for ALL SIGReg conditions
including E, then L_info is not strong enough.

**Action:**
- Increase λ₂ by 5× (from 0.1 to 0.5)
- Try L_info_dense instead of L_info_pooled (or vice versa)
- If still no improvement: this IS the result. Publish it honestly.
  "Neither SIGReg nor cross-covariance alignment improves on stop-gradient
  for patch-level JEPA representations."

### FM4 — NaN/Inf Loss

**Detection:** Automatic assertion in training loop.

**Likely cause:** L_info_dense can be numerically unstable when batch size is
small relative to d (covariance matrix ill-conditioned).

**Action:**
- Add L_cov regularizer (λ₃ = 0.01)
- Reduce λ₂ by 5×
- Increase batch size to 64

### FM5 — V-JEPA 2.1 Features Lack Spatial Structure

**Detection:** Raw feature neighbor_corr < 0.1 during baseline verification.

**Action:** Verify model version (must be V-JEPA 2.1, NOT V-JEPA 2).
Verify using patch tokens (not CLS or pooled). Verify correct checkpoint.

If confirmed that V-JEPA 2.1 features genuinely lack spatial structure,
the experiment cannot test spatial preservation. Pivot to: "Can SIGReg + L_info
induce spatial structure that isn't present in the input?"

### FM6 — Adapter Is Too Expressive (Memorizes Without Learning)

**Detection:** Train loss → 0 but eval metrics don't improve.

**Action:** Reduce adapter capacity (d=128), add dropout, increase weight decay.

---

## 7. Implementation

### 7.1 Repository Structure

```
gap1-experiment/
├── CLAUDE.md                    ← Agent instructions
├── build_spec.md                ← This document
├── foundations.md               ← Mathematical foundations
├── requirements.txt
│
├── configs/
│   ├── base.yaml
│   ├── condition_A.yaml
│   ├── condition_B.yaml
│   ├── condition_C.yaml
│   ├── condition_D1.yaml
│   ├── condition_D2.yaml
│   ├── condition_D3.yaml
│   ├── condition_E.yaml
│   └── condition_F.yaml
│
├── models/
│   ├── adapter.py               ← PatchAdapter (Definition 1.2)
│   ├── predictor.py             ← JEPAPredictor with AdaLN
│   ├── sigreg.py                ← SIGReg + all axis variants (Definition 3.2)
│   └── losses.py                ← All loss functions (Definitions 3.1–3.7)
│
├── data/
│   ├── ssv2_dataset.py          ← Feature loader
│   └── preextract_ssv2.py       ← One-time V-JEPA 2.1 extraction
│
├── training/
│   ├── trainer.py               ← Main training loop
│   └── metrics.py               ← All evaluation metrics (FREEZE before first job)
│
├── evaluation/
│   └── linear_probe.py          ← Post-hoc probe
│
├── tests/
│   ├── test_adapter_collapse.py ← Verify collapse is possible
│   ├── test_sigreg.py           ← Verify SIGReg implementation
│   ├── test_losses.py           ← Verify gradient flow per condition
│   ├── test_metrics.py          ← Verify metric computation
│   └── test_smoke.py            ← End-to-end smoke test
│
├── scripts/
│   ├── preextract_ssv2.sh
│   ├── run_condition.sh
│   ├── submit_gap1.sh
│   └── verify_baseline.sh       ← Raw feature quality check
│
└── analysis/
    ├── plot_erank_vs_probe.py   ← Central figure
    ├── plot_spatial_metrics.py
    ├── plot_eigenspectrum.py
    └── generate_results_table.py
```

### 7.2 Critical Implementation Details

**Adapter gradient flow test (must pass before any experiment):**

```python
def test_collapse_possible():
    """Verify that Condition A can collapse: gradient flows through both paths."""
    adapter = PatchAdapter(1024, 256)
    predictor = JEPAPredictor(...)

    f_c = torch.randn(4, 196, 1024)  # frozen features (no grad)
    f_t = torch.randn(4, 196, 1024)
    label = torch.randint(0, 174, (4,))

    z_c = adapter(f_c)       # has grad via adapter params
    z_t = adapter(f_t)       # has grad via adapter params (shared weights)
    z_hat = predictor(z_c, label)

    loss = F.mse_loss(z_hat, z_t)
    loss.backward()

    # Both adapter parameters must have gradients
    for name, param in adapter.named_parameters():
        assert param.grad is not None, f"No gradient for adapter.{name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for adapter.{name}"

    # Gradient should flow through BOTH z_c path and z_t path
    # To verify: compute gradient with z_t detached, should be different
    adapter.zero_grad()
    z_c2 = adapter(f_c)
    z_t2 = adapter(f_t).detach()  # NOW detached
    z_hat2 = predictor(z_c2, label)
    loss2 = F.mse_loss(z_hat2, z_t2)
    loss2.backward()

    grad_both = {n: p.grad.clone() for n, p in adapter.named_parameters()}

    # The gradients should be DIFFERENT (stop-grad removes one path)
    # If they're identical, the z_t path wasn't contributing
    for name in grad_both:
        diff = (grad_both[name] - adapter.state_dict()[name]).abs().sum()
        # We just check gradients exist; exact comparison is fragile
```

**SIGReg on adapter output, NOT predictor output:**

```python
# CORRECT — regularize the representation space
sig_loss = sigreg(adapter(f_c).mean(dim=1))  # [B, d]

# WRONG — regularize predictions (different semantics)
sig_loss = sigreg(predictor_output.mean(dim=1))  # DON'T DO THIS
```

**L_info on predictor output (has gradient) vs target (may not):**

```python
# Condition E: no stop-grad, so z_t has gradient too
z_hat = predictor(z_c, label)   # gradient via φ AND θ
z_t = adapter(f_t)              # gradient via θ (no stop-grad in E)
info_loss = l_info_dense(z_hat, z_t)  # gradient through both
```

### 7.3 Training Loop (Pseudocode)

```python
CONDITIONS_WITH_STOPGRAD = {'B', 'C'}

for step in range(n_steps):
    f_c, f_t, label = next(dataloader)  # frozen features + label

    # Adapter forward (always)
    z_c = adapter(f_c)

    # Target: stop-gradient depends on condition
    if condition in CONDITIONS_WITH_STOPGRAD:
        z_t = adapter(f_t).detach()
    else:
        z_t = adapter(f_t)

    # Predictor forward
    z_hat = predictor(z_c, label)

    # Loss (condition-specific)
    loss, components = compute_loss(z_hat, z_t, z_c, condition, config)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm_(
        list(adapter.parameters()) + list(predictor.parameters()),
        max_norm=1.0
    )
    optimizer.step()
    scheduler.step()
```

**Key difference from previous spec:** `adapter.parameters()` is included in
the optimizer. Both adapter and predictor are trained jointly.

---

## 8. Configuration Files

```yaml
# configs/base.yaml
feature_dir: "/scratch/${USER}/data/ssv2_vjepa21_features"
output_dir: "/scratch/${USER}/outputs/gap1"

# Adapter
D_in: 1024
d: 256

# Predictor
d_inner: 512
d_action: 64
n_layers: 6
n_heads: 8
dropout: 0.1

# Training
batch_size: 32
n_steps: 50000
learning_rate: 3.0e-4
weight_decay: 0.01
warmup_steps: 2000

# Defaults (overridden per condition)
lambda_1: 0.1
lambda_2: 0.1
lambda_3: 0.0
use_cov_reg: false
sigreg_axis: "global"
use_dense_info: true

# Logging
log_every: 100
eval_every: 500
probe_every: 5000
save_every: 10000
```

```yaml
# configs/condition_A.yaml — collapse baseline
condition: "A"
lambda_1: 0.0
lambda_2: 0.0
weight_decay: 0.0   # remove implicit regularization
```

```yaml
# configs/condition_B.yaml — stop-gradient only
condition: "B"
lambda_1: 0.0
lambda_2: 0.0
```

```yaml
# configs/condition_C.yaml — stop-grad + global SIGReg
condition: "C"
lambda_1: 0.1
lambda_2: 0.0
sigreg_axis: "global"
```

```yaml
# configs/condition_D1.yaml — per-token SIGReg
condition: "D1"
lambda_1: 0.1
lambda_2: 0.0
sigreg_axis: "token"
```

```yaml
# configs/condition_D2.yaml — per-channel SIGReg
condition: "D2"
lambda_1: 0.1
lambda_2: 0.0
sigreg_axis: "channel"
```

```yaml
# configs/condition_D3.yaml — global SIGReg, no stop-grad
condition: "D3"
lambda_1: 0.1
lambda_2: 0.0
sigreg_axis: "global"
```

```yaml
# configs/condition_E.yaml — SIGReg + dense info constraint
condition: "E"
lambda_1: 0.1
lambda_2: 0.1
sigreg_axis: "global"
use_dense_info: true
```

```yaml
# configs/condition_F.yaml — info constraint only (ablation)
condition: "F"
lambda_1: 0.0
lambda_2: 0.1
use_dense_info: true
```

---

## 9. Testing Protocol

**All tests must pass before ANY SLURM job is submitted.**

### 9.1 Unit Tests

```
test_adapter_collapse.py
  ├── test_collapse_possible()          — gradient flows both paths in Condition A
  ├── test_stopgrad_blocks_target()     — no gradient through z_t in Condition B
  ├── test_adapter_output_shape()       — [B, N, d] output shape
  └── test_no_normalization_layers()    — adapter has no BN/LN

test_sigreg.py
  ├── test_gaussian_input_low_loss()    — SIGReg ≈ 0 for N(0,I) samples
  ├── test_constant_input_high_loss()   — SIGReg >> 0 for identical samples
  ├── test_gradient_exists()            — SIGReg has nonzero gradient
  ├── test_all_axes_different()         — global/token/channel give different values
  └── test_projections_on_sphere()      — random directions are unit norm

test_losses.py
  ├── test_l_info_dense_gradient()      — L_info_dense has gradient through z_hat
  ├── test_l_info_dense_aligned_lower() — aligned inputs → lower (more negative) loss
  ├── test_l_info_zero_under_collapse() — constant inputs → L_info = 0
  ├── test_condition_E_all_gradients()  — adapter + predictor both have gradients
  └── test_condition_F_no_sigreg()      — SIGReg term is exactly 0.0

test_metrics.py
  ├── test_erank_collapsed()            — erank ≈ 1 for rank-1 input
  ├── test_erank_full_rank()            — erank ≈ d for isotropic Gaussian
  ├── test_neighbor_corr_smooth()       — high for spatially smooth features
  ├── test_neighbor_corr_random()       — low for random features
  └── test_infonce_aligned_high()       — InfoNCE high for aligned pairs

test_smoke.py
  ├── test_100_steps_no_nan()           — 100 steps of each condition, no NaN
  ├── test_100_steps_loss_decreases()   — loss at step 100 < loss at step 0
  └── test_all_metrics_computable()     — compute_all_metrics returns valid dict
```

### 9.2 Integration Verification (before full runs)

```bash
# Step 1: Verify raw feature quality
python scripts/verify_baseline.py  # prints raw erank, ncorr, probe acc

# Step 2: Smoke test all conditions (100 steps each, CPU or single GPU)
for COND in A B C D1 D2 D3 E F; do
    python -m training.trainer --config configs/base.yaml \
           --override configs/condition_${COND}.yaml --smoke_test --max_steps 100
done

# Step 3: Verify Condition A collapses in extended smoke test (2000 steps)
python -m training.trainer --config configs/base.yaml \
       --override configs/condition_A.yaml --max_steps 2000
# Check: erank should drop below 3 by step 2000
```

---

## 10. Execution Timeline (7 weeks to NeurIPS 2026)

```
WEEK 1 (Apr 2–8) — SETUP & DATA

Day 1   SSH to Discovery. Create gap1-experiment/.
        Set up venv: python3.10 -m venv venv && source venv/bin/activate
        pip install torch torchvision wandb scikit-learn
        Confirm GPU: srun --partition=gpu --gres=gpu:a100:1 --pty bash

Day 2   Download V-JEPA 2.1 ViT-G checkpoint from Meta's release.
        Register for SSv2 if needed. Begin download of part-00.
        IMPORTANT: verify this is V-JEPA 2.1, not V-JEPA 2.

Day 3   Implement feature extraction (preextract_ssv2.py).
        Run extraction: sbatch scripts/preextract_ssv2.sh
        Expected: 6-8 hours on 1 GPU.

Day 4   Extraction complete. Verify features:
        - Shape: [196, 1024] per frame
        - Neighbor correlation > 0.3
        - Effective rank > 20
        If verification fails: STOP. Debug feature extraction.

Day 5   Implement adapter.py, predictor.py, sigreg.py, losses.py, metrics.py.
        Run pytest tests/ -v — all tests must pass.
        FREEZE metrics.py (commit with tag "metrics-frozen-v1").

WEEK 2 (Apr 9–15) — CORE EXPERIMENTS

Day 6   Submit Condition A (collapse baseline): 8 GPU-hours.
        Submit Condition B in parallel: 8 GPU-hours.
        Monitor A every hour for first 4 hours.

Day 7   A should have collapsed by now. Verify: erank < 3 by step 2000.
        If A did NOT collapse: execute FM1 recovery (Section 6).
        If A collapsed: submit C, D3, E, F.

Day 8   Conditions C, D3, E, F running.
        Monitor W&B: erank, ncorr, xcov_trace for each.

Day 9   Submit D1, D2 (axis study — lower priority).
        Begin checking FM3 (SIGReg conditions vs B probe accuracy).

Day 10  All conditions complete or nearly complete.
        Run linear probe evaluations for completed conditions.

WEEK 3 (Apr 16–22) — ANALYSIS

Day 11  All conditions complete. Run final linear probe for all.
        Generate results table: generate_results_table.py

Day 12  Central figure: erank vs probe accuracy scatter plot.
        Eigenspectrum plot (SVD of adapter output per condition).
        Spatial metrics comparison (neighbor_corr per condition).

Day 13  Training curves figure (loss components over steps).
        Cross-covariance trace evolution figure.

Day 14  Compile all results. Does the hypothesis hold?
        Honest assessment: which claims are supported, which are not?

WEEK 4 (Apr 23–29) — WRITING (RESULTS + METHOD)

Day 15  Write Section 3 (Method): architecture, adapter, loss functions.
Day 16  Write Section 4 (Experiments): conditions, dataset, metrics.
Day 17  Write Section 5 (Results): tables, figures, analysis.
Day 18  Draft abstract and introduction outline.

WEEK 5 (Apr 30–May 6) — WRITING (INTRO + RELATED)

Day 19  Write Section 2 (Related Work): LeWM, V-JEPA family, VICReg, SIGReg.
Day 20  Write Section 1 (Introduction): motivation, gap, contribution.
Day 21  Write Section 6 (Discussion + Limitations): honest framing.

WEEK 6 (May 7–13) — REVISION

Day 22  Internal read-through. Fix logical gaps.
Day 23  Verify every number in the paper against W&B logs.
Day 24  Re-run any failed or ambiguous conditions if needed.
Day 25  Final figure polish. LaTeX formatting.

WEEK 7 (May 14–22) — SUBMISSION

Day 26  Abstract submission (if required before paper).
Day 27  Full paper draft complete. Share with advisor.
Day 28  Address advisor feedback.
Day 29  Final proofread. Verify supplementary material.
Day 30  SUBMIT.
```

---

## 11. Paper Outline

```
Title: "Distributional Regularization Meets Spatial Structure:
        A Diagnostic Study of SIGReg in Patch-Level JEPA Representations"

Abstract (150 words)
  - JEPA world models need collapse prevention
  - LeWM shows SIGReg works for CLS tokens
  - We study SIGReg at patch-token level using V-JEPA 2.1 features
  - Finding: SIGReg prevents collapse but can destroy spatial structure
  - Finding: axis of application matters (per-token worst, global best)
  - Finding: adding cross-view alignment recovers information
  - Contribution: first systematic study of SIGReg × spatial structure interaction

1. Introduction (1.5 pages)
2. Related Work (1 page)
   - JEPA and collapse prevention
   - SIGReg and distributional regularization
   - V-JEPA family and spatial features
3. Method (2 pages)
   - Architecture (adapter + predictor)
   - Loss functions (SIGReg variants + L_info_dense)
   - Experimental conditions
4. Experimental Setup (1 page)
   - Dataset, metrics, hyperparameters
5. Results (2 pages)
   - Table 1: full condition × metric results
   - Figure 1: eigenspectrum per condition (the money figure)
   - Figure 2: erank vs probe accuracy scatter
   - Figure 3: neighbor_corr across conditions
   - Figure 4: training curves (stability)
6. Discussion (0.5 pages)
   - Limitations: adapter is not full end-to-end, SSv2 only, etc.
   - Implications for LeWM-style systems at patch level
7. Conclusion (0.25 pages)

Total: ~8 pages + references + appendix
```

---

## 12. Definition of Success

The experiment succeeds if we can show **any two** of the following three:

1. **SIGReg prevents adapter collapse** — erank under SIGReg conditions >> erank under A/B
2. **SIGReg axis affects spatial structure** — neighbor_corr(D1) < neighbor_corr(D3)
3. **L_info improves over SIGReg alone** — probe_acc(E) > probe_acc(D3)

If all three hold: strong paper.
If only two hold: solid paper with honest negative result on the third.
If only one holds: workshop paper with exploratory framing.
If zero hold: we have learned something important about what DOESN'T work,
and we document it as a negative result paper (still publishable at NeurIPS
workshop track or TMLR).

---

## 13. Compute Budget

| Condition | GPU-hours (est.) | Priority |
|-----------|-----------------|----------|
| A         | 8               | Critical |
| B         | 8               | Critical |
| C         | 8               | High     |
| D1        | 8               | Medium   |
| D2        | 8               | Medium   |
| D3        | 8               | Critical |
| E         | 8               | Critical |
| F         | 8               | High     |
| Feature extraction | 8     | Critical |
| Probes + analysis  | 4     | Critical |
| **Total** | **76**          |          |

On Discovery with A100: approximately 76 GPU-hours.
At ~2 jobs per day (scheduler contention): ~5 days of wall-clock GPU time.

Minimum viable (A, B, D3, E, F): 44 GPU-hours, ~3 days.
