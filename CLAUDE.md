# CLAUDE.md — Agent Instructions for GAP 1 Experiment

**Read this ENTIRE file before writing any code or making any decision.**

---

## Project Identity

This is a NeurIPS 2026 research project studying how distributional regularization
(SIGReg) interacts with spatial structure in patch-level JEPA representations.

The two canonical reference documents are:
- `foundations.md` — All mathematics. Every formula in code must trace to a definition here.
- `build_spec.md` — All experimental design, conditions, metrics, timeline.

If there is any conflict between this file, build_spec.md, and foundations.md,
**foundations.md wins** (it is the mathematical source of truth).

---

## Architecture Summary

```
Frozen V-JEPA 2.1 features (.pt) → Trainable Adapter (MLP) → Predictor (Transformer) → Loss
```

**Three trainable components:**
1. `PatchAdapter` — token-wise MLP: ℝ^1024 → ℝ^d (d=256 default)
2. `JEPAPredictor` — transformer with AdaLN conditioning on action labels
3. Loss-specific parameters (e.g., SIGReg projection vectors are resampled, not learned)

**Frozen components:**
- V-JEPA 2.1 ViT-G features (loaded from .pt files)
- SSv2 labels (loaded from metadata)

---

## Critical Rules — Violations of These Are Bugs

### Rule 1: SIGReg is applied to ADAPTER output, never predictor output.
```python
# CORRECT
sig_loss = sigreg(adapter(f_c))

# WRONG — this regularizes predictions, not representations
sig_loss = sigreg(predictor_output)
```
Why: SIGReg regularizes the representation SPACE (the encoder's output distribution).
The adapter is our encoder surrogate. The predictor makes predictions within that space.

### Rule 2: L_info must involve at least one tensor with gradient.
```python
# CORRECT — z_hat has gradient through predictor + adapter
info_loss = l_info_dense(z_hat, z_t)

# WRONG — both are frozen features, zero gradient
info_loss = l_info_dense(frozen_f_c, frozen_f_t)
```

### Rule 3: Stop-gradient is applied to the ADAPTER output, not the frozen features.
```python
# CORRECT
z_t = adapter(f_t).detach()  # stop-grad on adapter output

# MEANINGLESS — frozen features already have no gradient
z_t = f_t.detach()  # this is a no-op
```

### Rule 4: The adapter has NO normalization layers (no BatchNorm, no LayerNorm).
Why: Normalization can mask collapse. If the adapter maps everything to a constant,
we need the downstream metrics to see that constant, not a normalized version of it.

### Rule 5: Conditions A and F must have weight_decay=0 in the optimizer.
Why: Weight decay is an implicit regularizer that can prevent collapse.
For conditions testing whether collapse occurs, we must remove all implicit regularizers.

### Rule 6: metrics.py is FROZEN after the first experiment job is submitted.
Never modify metrics.py after any condition has started training.
All conditions must be evaluated with identical metric code.

### Rule 7: Every number in the paper comes from W&B logs, never from memory.
When writing analysis code, read values from W&B API. Never hardcode results.

---

## Condition-Specific Implementation Notes

### Condition A (collapse baseline)
- `z_t = adapter(f_t)` — NO detach
- Loss: `F.mse_loss(z_hat, z_t)` only
- `weight_decay = 0.0` in optimizer
- Expected: erank → 1 within 2000 steps
- If erank stays > 5 after 5000 steps: HALT. Read FM1 in build_spec.md.

### Condition B (stop-gradient)
- `z_t = adapter(f_t).detach()`
- Loss: `F.mse_loss(z_hat, z_t)` only
- May or may not collapse (both outcomes are valid findings)

### Conditions C (stop-grad + SIGReg)
- `z_t = adapter(f_t).detach()`
- `z_c = adapter(f_c)` — SIGReg applied to this
- Loss: `F.mse_loss(z_hat, z_t) + lambda_1 * sigreg_global(z_c)`

### Conditions D1/D2/D3 (SIGReg axis study)
- NO stop-gradient: `z_t = adapter(f_t)` (with gradient)
- SIGReg applied to z_c with the appropriate axis function
- D1: `sigreg(z_c.reshape(B*N, d))` — per-token
- D2: `sigreg_channel(z_c)` — per-channel (EP test on each dimension)
- D3: `sigreg(z_c.mean(dim=1))` — global (pool then regularize)

### Condition E (SIGReg + dense info)
- NO stop-gradient
- Loss: `L_pred + lambda_1 * sigreg_global(z_c) + lambda_2 * l_info_dense(z_hat, z_t)`
- Both adapter and predictor optimized
- This is the proposed method

### Condition F (info only, ablation)
- NO stop-gradient, NO SIGReg
- Loss: `L_pred + lambda_2 * l_info_dense(z_hat, z_t)`
- `weight_decay = 0.0`
- Expected to collapse (L_info = 0 under collapse)

---

## File Descriptions

| File | Purpose | Modify after first job? |
|------|---------|------------------------|
| `models/adapter.py` | PatchAdapter MLP | No |
| `models/predictor.py` | Transformer + AdaLN | No |
| `models/sigreg.py` | SIGReg + axis variants | No |
| `models/losses.py` | All loss functions + compute_loss | No |
| `training/metrics.py` | All evaluation metrics | **NEVER** |
| `training/trainer.py` | Main training loop | Only for bug fixes |
| `data/ssv2_dataset.py` | Feature loader | No |
| `configs/*.yaml` | Hyperparameters per condition | Yes (before submission) |
| `analysis/*.py` | Post-hoc analysis scripts | Yes (freely) |

---

## Common Mistakes to Avoid

1. **Do NOT use `torch.compile` or `torch.jit` on the adapter.** It can hide
   gradient flow issues that are critical for collapse detection.

2. **Do NOT add a learning rate warmup for Condition A.** Warmup slows down
   the initial collapse dynamics, making it harder to confirm collapse.

3. **Do NOT use mixed precision (fp16/bf16) for SIGReg computation.** The
   Epps-Pulley test involves characteristic functions (complex exponentials)
   that are numerically sensitive. Compute SIGReg in fp32 always.

4. **Do NOT normalize adapter outputs before computing metrics.** L2-normalization
   would prevent effective rank from detecting collapse (normalized vectors
   always have unit norm, masking the collapse signal).

5. **Do NOT share the optimizer across conditions.** Each condition must start
   from the same random initialization with a fresh optimizer.

6. **Do NOT average metrics across batches during evaluation.** Compute metrics
   on the concatenation of all eval batches, not the mean of per-batch metrics.
   (Effective rank of concatenated data ≠ mean of per-batch effective ranks.)

---

## Testing Checklist (Run Before Every Job Submission)

```bash
# 1. All unit tests pass
python -m pytest tests/ -v --timeout=120
# Zero failures, zero skips allowed.

# 2. Smoke test for the specific condition being submitted
python -m training.trainer --config configs/base.yaml \
       --override configs/condition_X.yaml --smoke_test --max_steps 100
# Must complete without NaN/Inf, loss must be finite.

# 3. Verify feature baseline (run once after extraction)
python scripts/verify_baseline.py
# Must print: raw_erank > 20, raw_ncorr > 0.3
```

---

## W&B Configuration

```
Project: gap1-sigreg-spatial
Entity: YOUR_WANDB_ENTITY
Tags per run: [condition_{X}, gap1, v3]
```

Every run must log:
- All metrics from compute_all_metrics() at eval_every steps
- All loss components at log_every steps
- Gradient norm at log_every steps
- Learning rate at log_every steps
- Linear probe results at probe_every steps

---

## When Things Go Wrong

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| Condition A erank stays high | Implicit regularization | Set weight_decay=0, random init, raise grad clip to 10.0 |
| NaN in SIGReg loss | Zero-variance projections | Add eps=1e-8 to standardization: std + eps |
| NaN in L_info_dense | Degenerate covariance | Add L_cov regularizer (lambda_3=0.01) |
| All conditions have same probe accuracy | Adapter too weak to differentiate | Increase d (try 512), increase adapter depth |
| Probe accuracy at chance for all | Features don't encode SSv2 actions | Verify V-JEPA 2.1 features, try ImageNet probe instead |
| D3 and C give identical results | Stop-grad has no effect (expected if A collapses) | This IS the result for shared-weight adapter — document it |
| SLURM job killed (OOM) | Batch size too large for A100 80GB | Reduce batch_size to 16, or reduce d to 128 |

---

## Key Equations (Quick Reference)

All derive from foundations.md. See that document for full definitions.

```
L_pred        = MSE(ẑ_t, z_t)
SIGReg(Z)     = (1/M) Σ_m T(Z · u_m)       # T = Epps-Pulley statistic
L_info_dense  = -(1/N) Σ_n Tr(Cov_n(ẑ, z_t))
L_cov         = ||Cov(ẑ_pool) - I||_F²

Condition E:  L = L_pred + λ₁·SIGReg(z_c_pool) + λ₂·L_info_dense(ẑ, z_t)
```
