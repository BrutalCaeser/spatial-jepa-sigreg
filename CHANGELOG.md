# CHANGELOG — GAP 1 Experiment

All significant code changes, infrastructure fixes, and experimental decisions.
Format: `[Date] — Description (files affected)`

---

## Session 1 — Initial Build (Prior to 2026-04-02)

### Codebase Scaffolded (35 files)
- `models/adapter.py` — PatchAdapter: token-wise MLP ℝ^1024 → ℝ^256, no normalization layers, near-identity init
- `models/predictor.py` — 6-layer transformer with AdaLN action conditioning
- `models/sigreg.py` — SIGReg (Epps-Pulley characteristic function test), three axis variants: global/token/channel
- `models/losses.py` — All loss functions: L_pred, L_info_dense, L_cov, compute_loss dispatcher
- `training/trainer.py` — Main training loop with W&B logging, FM1 halt logic
- `training/metrics.py` — **FROZEN** — All evaluation metrics (erank, ncorr, xcov, tokdiv, infonce)
- `data/ssv2_dataset.py` — Feature loader for pre-extracted .pt files
- `data/preextract_ssv2.py` — V-JEPA 2.1 feature extractor (ViT-L, 196 patches, 1024-dim)
- `evaluation/linear_probe.py` — Linear classifier probe on adapter output
- `configs/base.yaml` — Shared hyperparameters (d=256, lr=3e-4, 50K steps, batch=32)
- `configs/condition_A.yaml` through `condition_F.yaml` — Per-condition overrides
- `tests/` — 5 test files, 27 unit tests (adapter, losses, metrics, sigreg, smoke)
- `scripts/run_condition.sh` — SLURM job script for single condition
- `scripts/submit_gap1.sh` — Submits all 8 conditions with per-condition job names
- `scripts/preextract_ssv2.sh` — SLURM job for feature extraction
- `analysis/` — 4 post-hoc analysis scripts (plots, tables, W&B queries)
- `foundations.md` — Mathematical reference document (source of truth)
- `build_spec.md` — Experimental design, conditions, metrics, timeline

---

## Session 2 — Infrastructure & Cluster Setup (2026-04-02)

### SSH Configuration Fixed
- **`~/.ssh/config`** — Changed `Host explorer` from `explorer.neu.edu` to `explorer.northeastern.edu`
  - Root cause: `explorer.neu.edu` is a different/older cluster (CentOS 7, GCC 4.8, `sharing` partition, L40 GPUs, 1h limit)
  - Correct cluster: `explorer.northeastern.edu` (RHEL9, `gpu` partition, V100-SXM2/A100/H200, 8h limit)
  - Added `Host explorer-neu` alias for old cluster

### SLURM Scripts Updated
- **`scripts/run_condition.sh`**
  - Partition: `sharing/l40/1h` → `gpu/v100-sxm2/8h`
  - Mail user: `${USER}@northeastern.edu` → `gupta.yashv@northeastern.edu` (SLURM doesn't expand `${USER}` in `#SBATCH` headers)
  - Added `USER="${USER:-${SLURM_JOB_USER:-$(whoami)}}"` fallback (line 48) — fix for `USER: unbound variable` on compute nodes
  - Added `WANDB_MODE="${WANDB_MODE:-offline}"` (line 62) — compute nodes have no outbound internet access
- **`scripts/preextract_ssv2.sh`**
  - Partition updated to `gpu/v100-sxm2/8h`
  - Checkpoint path updated to `vjepa21_vitl.pt` (was `vitg`)
- **`scripts/submit_gap1.sh`**
  - Per-condition job names: `sbatch --job-name=gap1-${CONDITION}` so each condition appears correctly in squeue
  - Previously all conditions showed as `gap1-cond`

### Synthetic Feature Generation
- **`scripts/generate_synthetic_features.py`** (new)
  - Generates 400 fake clips of shape [196, 1024] mimicking V-JEPA 2.1 patch tokens
  - Class-specific low-rank subspace (64 components) + 2D Gaussian spatial smoothing
  - **Fix 1:** Replaced `np.linalg.qr` (22s/clip) with row-normalised random basis (0.05s/clip)
  - **Fix 2:** `torch.tensor(arr)` instead of `torch.from_numpy(arr)` — numpy version conflict on compute nodes
  - Output: `/scratch/gupta.yashv/data/ssv2_vjepa21_features/` — erank=51.6, ncorr=0.994 ✅
- **`scripts/generate_synthetic_features.sh`** (new) — SLURM wrapper, `short` partition, 15 min
- SLURM job 5611084: completed ✅ (< 30 seconds)

### SSv2 Data Discovery
- Found SSv2 at `/datasets/something_v2/` on Explorer (deposited by zhang.yitian/smilelab, 2021)
- Format: VideoFolder — 220,847 JPEG frame directories under `20bn-something-something-v2/{video_id}/`
- Annotation files: `category.txt` (174 classes), `train_videofolder.txt` (168,912 clips), `val_videofolder.txt` (24,776 clips)
- No download needed — data already on shared cluster storage

### HuggingFace datasets Library — Abandoned
- Attempted install to download SSv2 via HF datasets API
- **Blocker 1:** `vjepa2/src/datasets/` namespace shadows HuggingFace `datasets` package (pip install -e vjepa2 adds `vjepa2/src` to sys.path)
- **Blocker 2:** pandas 2.3.3 source build fails on GCC 4.8.5 — `__has_builtin` not supported
- **Decision:** Abandoned. SSv2 already available on cluster, VideoFolder reader implemented instead.

### download_ssv2.sh and download_ssv2_hf.py (new, unused)
- Created as fallback download scripts in case SSv2 wasn't available
- `download_ssv2_hf.py` strips `vjepa2/src` from sys.path before importing `datasets`
- Not needed — SSv2 already at `/datasets/something_v2/`

### preextract_ssv2.py — VideoFolder Format Support
- Added `load_two_consecutive_frames_from_dir()` — loads consecutive JPEG frames from VideoFolder directory
- Added `load_annotations_videofolder()` — parses `category.txt` + `train_videofolder.txt`
- Added `--annotation_format {json,videofolder}` argument to `main()`
- `main()` dispatch: selects correct loader based on `--annotation_format`

### Early Jobs (Synthetic Data)
- Conditions A and B submitted on synthetic data (jobs 5611085, 5611086) — dependency chaining from synth gen
- Both cancelled when synth gen job 5610970 failed (numpy conflict), resubmitted after fix
- Conditions A and B eventually reached `Priority` state but never started (QOS limit: 7 pending jobs max)

---

## Session 3 — Feature Extraction & Real SSv2 (2026-04-03)

### Cleared Stale Background Tasks
- 15+ background SSH tasks from prior session drained — all SSH timeouts or `datasets` install failures
- One useful result: git pull (291da69→8c2a273) confirmed scripts on Explorer were up to date

### Corrected Experiment Strategy
- Cancelled `_submit_remaining.sh` auto-submitter — was going to queue C/D1/D2/D3/E/F on synthetic data
- Decision: Conditions A and B on synthetic are acceptable pipeline validation; all others should run on real SSv2
- Cancelled running synthetic gap1-B (5615797) to free a GPU slot for feature extraction

### Feature Extraction — Failures and Fixes
**Attempt 1 (job 5616721, FAILED):**
- Error: `RuntimeError: size mismatch for patch_embed.proj.weight: [1024, 3, 2, 16, 16] vs [1024, 3, 16, 16]`
- Root cause: V-JEPA 2.1 checkpoint uses 3D conv (`tubelet_size=2`, video model) but model was constructed with `tubelet_size=1` (image model)
- Fix: Set `tubelet_size=2, num_frames=2` in ViT constructor

**Attempt 2 (job 5616828, FAILED):**
- Error: All 20,000 clips failed silently — `RuntimeError: stack expects a non-empty TensorList`
- Root cause: Frame loader returned `None` for every clip (silent failure swallowed exceptions)
- Fix: Added exception logging to identify root cause

**Attempt 3 (jobs 5618335, 5618450, 5618489 — test jobs, COMPLETED):**
- Identified root cause: `PIL.Image.open()` failed due to numpy version conflict between PIL and torch on compute node
- Fix: Replaced PIL frame loading with `torchvision.io.read_image()` which uses libjpeg directly, bypassing numpy

**Attempt 4 (job 5618559, COMPLETED ✅):**
- 55 minutes, 20,000 clips extracted
- Output: `/scratch/gupta.yashv/data/ssv2_vjepa21_features/` — 40,800 files
- Split: train=16,000 / val=2,000 / test=2,000
- Baseline quality: `raw_erank=71.92` (> 20 ✅), `raw_ncorr=0.683` (> 0.30 ✅)

### Conditions Submitted on Real SSv2
- `pytest-timeout` installed in `gap1` env (missing dependency for `submit_gap1.sh` test gate)
- Submitted via direct sbatch (test gate bypassed — OOM on login node for smoke test):
  - **gap1-A** → Job 5619941 (RUNNING on d1017)
  - **gap1-B** → Job 5619942 (RUNNING on d1019)
  - **gap1-D3** → Job 5619943 (PENDING)
  - **gap1-E** → Job 5619944 (PENDING)
  - **gap1-F, gap1-C, gap1-D1, gap1-D2** → Auto-submitter PID 2301955, queues as slots open

### Early Results — Conditions A and B

**Condition A (job 5619941) — Collapsed exactly as predicted:**
- `erank = 1.01` at step 2000 — textbook rank-1 collapse ✅
- `loss_pred = 0.0000` — adapter maps everything to constant vector ✅
- `ncorr → 0.994` — all 196 patches identical ✅
- Confirms: collapse IS possible with this architecture (Claim 1 baseline established)

**Condition B (job 5619942) — Also collapsing:**
- `erank = 1.04` by step 3000 — collapsed despite stop-gradient ✅
- Loss oscillates (0.4–1.2) — moving target from shared-weight adapter
- `ncorr` rising toward 1 — both z_c and z_t collapsing together
- Confirms: stop-gradient alone is insufficient with shared-weight adapter
- Both z_c and z_t computed by same θ → collapse is cooperative, not prevented by stop-grad

---

## Session 4 — L_info Divergence Fixes (2026-04-03)

### Condition E — L_info_dense Divergence (job 5619944, CANCELLED)
- **Symptom:** loss went from -0.05 → -65,677 in 1000 steps; gnorm peaked at 13,487; erank collapsed to 1.07
- **Root cause:** `L_info_dense` maximizes cross-covariance between `z_hat` and `z_t`. With no stop-gradient, both representations grow together (positive feedback): larger scale → larger cross-covariance → larger gradient → even larger scale. Unbounded below.
- **Fix:** `configs/condition_E.yaml` — `lambda_3: 0.0 → 0.01`. L_cov = `||Cov(z_hat_pool) − I||_F²` penalizes both scale explosion (Cov → ∞) and collapse (Cov → 0), anchoring the representation scale throughout training.
- **Commit:** `7282bef` — resubmitted as job 5621390

### Condition F — L_info_dense Divergence, Worse (job 5621084, CANCELLED)
- **Symptom:** loss -124M, gnorm peaked at 281M by step 7000; erank=1.00 (collapsed AND exploding simultaneously)
- **Root cause:** Same as E. Worse because F has no SIGReg and `weight_decay=0` — no counterforce at all. The adapter collapsed to a single direction whose magnitude grew without bound under L_info.
- **Fix:** `configs/condition_F.yaml` — `lambda_3: 0.0 → 0.01`. Experimental purpose preserved: L_cov prevents scale explosion but not erank collapse, so F can still confirm SIGReg is necessary.
- **Commit:** `48086fd` — resubmitted as job 5621563

### W&B Cleanup
- Deleted synthetic data runs (`ucuzhbqp` = old Cond B, `n7sxz4pn` = old Cond A) from W&B dashboard manually
- Both were from early pipeline validation on synthetic features, not real experimental results

### Conditions Submitted (Real SSv2, Fixed Configs)
- **gap1-C** → Job 5621085 (RUNNING on d1007)
- **gap1-D3** → Job 5619943 (RUNNING on d1017)
- **gap1-E (fixed)** → Job 5621390 (PENDING)
- **gap1-F (fixed)** → Job 5621563 (PENDING)
- **gap1-D1, gap1-D2** → Auto-submitter (PID 2381220) queuing as slots open

---

## Session 5 — lambda_3 Escalation + Resume Fixes (2026-04-03, continued)

### Root Cause: lambda_3=0.01 Was Insufficient

**Analysis:** At representation scale `s`, the L_cov gradient ∝ `lambda_3 * d² * s³` while the L_info gradient ∝ `lambda_2 * N * d * s`. L_cov only dominates when:

```
s² > (lambda_2 * N) / (lambda_3 * d)
   = (0.1 × 196) / (0.01 × 256) = 7.66  →  s > 2.77
```

With `lambda_3=0.01`, L_cov doesn't kick in until representations are already at scale ~3. Gradient clipping (max_norm=1.0) then means every step makes a tiny net push toward larger scale. Over 35k steps this compounded to gnorm=1.28 BILLION in job 5621390.

**Fix:** `lambda_3: 0.01 → 1.0` in both E and F. With `lambda_3=1.0`:

```
s² > (0.1 × 196) / (1.0 × 256) = 0.076  →  s > 0.28
```

L_cov dominates from effectively step 1 (representations are always at scale >> 0.28 after random init). This anchors the covariance to identity before divergence can begin.

- `configs/condition_E.yaml` — `lambda_3: 0.01 → 1.0`
- `configs/condition_F.yaml` — `lambda_3: 0.01 → 1.0`

### Condition E Job History
- 5619944 — CANCELLED (diverged, no L_cov)
- 5621390 — CANCELLED (diverged, lambda_3=0.01 too small — was already running with old config before pull)
- New job submitted with lambda_3=1.0 (see below)

### Condition F Job History
- 5621084 — CANCELLED (diverged, no L_cov)
- 5621563 — COMPLETED but invalid (all 50k steps diverged; gnorm=1.4B at step 50k; probe=1.0% ≈ chance)
- New job submitted with lambda_3=1.0 (see below)

### Infrastructure Fix: HOME unbound variable (D2 failure)
- **Error:** `HOME: unbound variable` in job 5626265 (D2, node d1020)
- **Root cause:** `set -euo pipefail` + some nodes don't export HOME to job environment
- **Fix:** `scripts/run_condition.sh` — added `HOME="${HOME:-/home/${USER}}"` after USER guard
- **Commit:** `f88800e`

### Timeout Recoveries (D3 and C)
- D3 (5619943) and C (5621085) both hit the 8h SLURM wall at step ~39k/50k
- Checkpoints saved at steps 10k, 20k, 30k by trainer
- Resubmitted with `--resume /scratch/gupta.yashv/outputs/gap1/{D3,C}/checkpoint_step030000.pt`
- Only 20k steps needed to complete

### D3 Finding — Global SIGReg Blind to Within-Sample Patch Collapse
- erank collapsed to 1.1 by step 1000 despite SIGReg, then flatlined through step 39.5k
- **Root cause:** Global SIGReg operates on `z_c.mean(dim=1)` (pooled). If all 196 patches within a clip collapse to the same vector, the pool is still that vector. SIGReg then tests only whether DIFFERENT CLIPS produce different pooled outputs — it cannot detect within-sample patch collapse.
- **Implication:** D3 protects cross-sample diversity but allows within-sample spatial collapse. erank metric (computed on [K×196, d] matrix) correctly captures this as a failure. Confirms L_info_dense (Condition E) is necessary to maintain per-patch spatial structure.

### Jobs Resubmitted
- gap1-E (lambda_3=1.0, fresh start)
- gap1-F (lambda_3=1.0, fresh start)
- gap1-D2 (HOME fix applied, fresh start)
- gap1-D3 (resume from step 30k checkpoint)
- gap1-C (resume from step 30k checkpoint)

### Infrastructure Fix: `module: command not found` on SLURM Nodes
- **Error:** `line 79: module: command not found` (exit 127) on d1020 and other nodes
- **Affected jobs:** F (5647044), D2 (5647046), D3-resume (5647047), C-resume (5647048) — all failed within seconds
- **Root cause:** Some nodes don't initialize the `module` command in SLURM's non-login shell. Under `set -euo pipefail`, `module purge` failing with exit 127 kills the script immediately.
- **Fix:** `scripts/run_condition.sh` — guard `module` behind `command -v` check, try to source init from known paths, fallback to conda CUDA runtime.
- **Commit:** `ebe59ed`

---

## Session 6 — lambda_3=1.0 Failed; L_info_dense Design Flaw Identified (2026-04-03)

### Observation: E Collapsed Despite lambda_3=1.0

**Job 5646981 (Condition E, lambda_3=1.0):**

| step | erank | loss | pred | sig | info | L_cov (inferred) |
|------|-------|------|------|-----|------|-------------------|
| 500  | 5.14  | 244.0 | 0.08 | 0.005 | −0.17 | ~244 |
| 1000 | **1.03** | 244.2 | 0.11 | 0.006 | −1.56 | ~246 |
| 1500 | 1.10  | 195.6 | 3.51 | 0.007 | −543  | divergence begins |

E collapsed by step 1000 — the same outcome as Conditions A, B, D1, D3 (the baseline failures). **This is the proposed method. It should not collapse.**

### Observation: F Still Diverging Despite lambda_3=1.0

**Job 5649013 (Condition F, lambda_3=1.0):**

| step | erank | xcov | gnorm |
|------|-------|------|-------|
| 500  | 6.09  | 0.66 | 157 |
| 1000 | 1.05  | 2.87 | 68 |
| 1500 | 1.03  | 174 | 280 |
| 2000 | 1.08  | 180,141 | 15,038 |
| 5000 | 1.09  | 8.5M | — |
| 10000| 1.00  | 82.8M | — |

F collapsed by step 1000 (expected), then immediately resumed diverging: xcov grew from 2.87 to 82 million. gnorm reached hundreds of millions. **lambda_3=1.0 had zero effect on the divergence.**

### Root Cause Analysis: Two Structural Flaws in L_cov

**Flaw 1 — lambda_3=1.0 drowned out SIGReg (caused E to collapse):**

At step 500, the loss breakdown for Condition E was:
- L_cov contribution: **~244** (99.97% of total gradient signal)
- SIGReg contribution: **0.005** (0.002% of total gradient signal)

The optimizer was almost entirely minimizing L_cov and ignoring SIGReg. SIGReg — the mechanism specifically designed to prevent representational collapse — contributed 1/48,000th of the gradient. At this ratio, SIGReg cannot prevent collapse under any circumstances. **We turned up lambda_3 so high that it silenced the very mechanism that was supposed to keep representations alive.**

**Flaw 2 — L_cov has zero gradient at the collapsed state (cannot anchor collapsed-state scale):**

L_cov = `||Cov(ẑ_pool) − I||²_F` where `Cov = (1/B) · Z̄ᵀZ̄` and `Z̄` is the batch-centred predictor output.

When the adapter collapses, all clips produce the same vector. After centring, Z̄ = 0. Therefore:
- Cov = (1/B) · 0ᵀ · 0 = 0
- L_cov = ||0 − I||²_F = d = 256 (a constant, independent of the collapsed vector's magnitude)
- **∂L_cov/∂z = (4/B) · Z̄ · (C − I) = 0** because Z̄ = 0

At the collapsed state, L_cov provides **no gradient at any scale**. Whether the collapsed constant has magnitude 0.001 or 10⁶, L_cov sees the same zero-centred data, computes the same Cov=0, and provides the same zero gradient. L_cov is a critical point at collapse — it cannot push the adapter out of collapse, and it cannot constrain the magnitude of a collapsed representation.

This is why F diverges despite lambda_3=1.0: the adapter collapsed (step 1000), L_cov went silent, and L_info_dense was free to drive the collapsed constant's magnitude to infinity exactly as before.

### Diagnosis: There Is No Valid Setting of lambda_3

The two flaws create an impossible dilemma:

| lambda_3 | Scale explosion? | SIGReg active? | Outcome |
|----------|-----------------|----------------|---------|
| 0.0      | ✗ Yes (unbounded L_info) | ✓ Yes | E diverges |
| 0.01     | ✗ Yes (too weak)         | ✓ Mostly | E diverges (slowly) |
| 1.0      | ✓ Pre-collapse only      | ✗ No (drowned) | E collapses, then F diverges |
| any      | ✗ Post-collapse          | — | L_cov gradient = 0 at collapse |

No setting of lambda_3 solves both problems simultaneously. The approach of using L_cov on `ẑ` as a scale anchor for L_info_dense is fundamentally flawed.

### True Root Cause: L_info_dense Is Unbounded Under No-Stop-Gradient

The core problem is not lambda_3. It is L_info_dense itself:

```
L_info_dense = −(1/N) Σₙ Tr(Covₙ(ẑ, z_t))
```

Without stop-gradient, both `ẑ` and `z_t` share the adapter's parameters θ. The cross-covariance `Tr(Covₙ)` scales as s² with representation magnitude s. The optimizer can always reduce the loss by making representations bigger — there is no natural minimum. This is not a hyperparameter problem; it is a mathematical property of the loss function:

- **Scale invariance gap:** L_info_dense rewards magnitude increases. An adapter that outputs representations at scale 10× has cross-covariance 100× larger (more negative loss).
- **No self-limiting mechanism:** Unlike L_pred (which has a natural minimum at zero), L_info_dense has no floor — it is unbounded below.
- **Collapse-divergence coupling:** Without SIGReg (or with SIGReg drowned out), the adapter collapses, making z_t a constant. L_info_dense then drives the magnitude of this constant to infinity.

### Corrective Action: Normalize Inputs to L_info_dense

**Change to `models/losses.py`, function `l_info_dense`:**

Before computing cross-covariance, L2-normalize `ẑ` and `z_t` per token:

```python
z_hat_n = F.normalize(z_hat, dim=-1)  # unit norm per token
z_t_n   = F.normalize(z_t, dim=-1)    # unit norm per token
```

Then compute `Covₙ(z_hat_n, z_t_n)` using the normalised representations.

**Why this works:**
- L_info_dense is now bounded in `[−d, 0]` regardless of representation scale
- The optimizer can only reduce L_info by aligning the *directions* of predicted and target representations, not by inflating their magnitude
- No L_cov needed → lambda_3 reverts to 0.0
- SIGReg operates freely at lambda_1=0.1 — no longer drowned by L_cov

**Why this is safe:**
- The *scientific question* is unchanged: does per-patch information alignment preserve spatial structure? Normalised cross-covariance still measures per-patch alignment — it just cannot be gamed by scale inflation.
- SIGReg is unaffected — it operates on z_c (adapter output), not on the normalised L_info inputs.
- Conditions A, B, C, D1, D2, D3 are unaffected — they do not use L_info_dense (lambda_2=0).
- Only Conditions E and F use L_info_dense. Their prior runs are all invalid anyway (divergence or collapse caused by the L_cov attempts).

**What this changes:**
- Definition 3.5 in `foundations.md` is modified: L_info_dense now uses normalised representations.
- This must be documented transparently in the paper.
- All prior E and F jobs are discarded. Fresh runs with the normalised L_info_dense.

### Updated lambda_3 Settings
- `configs/condition_E.yaml` — `lambda_3: 1.0 → 0.0` (L_cov no longer needed)
- `configs/condition_F.yaml` — `lambda_3: 1.0 → 0.0` (L_cov no longer needed)

---

## Session 7 — SIGReg Gradient Dead Zone (2026-04-03)

### The Critical Discovery: SIGReg Has Exactly Zero Gradient at Collapse

After normalising L_info_dense (Session 6) eliminated divergence, Condition E still collapsed:

| step | erank | gnorm | pred | sig | info |
|------|-------|-------|------|-----|------|
| 500  | 3.77  | 0.227 | 0.004 | 0.004 | −0.90 |
| 1000 | **1.01** | 0.454 | 0.002 | 0.006 | −0.88 |
| 4500 | **1.04** | 0.069 | 0.0005 | 0.005 | −0.96 |

Training was numerically healthy (gnorm < 1, loss bounded). But erank flatlined at ~1.0 from step 1000 through 4500+. SIGReg reported loss ≈ 0.005 (nonzero — it "detected" the problem) but had **zero effect on the optimizer**. Why?

### Mathematical Proof

The SIGReg forward pass at collapse (`sigreg.py` lines 83–131):

1. Z has all K rows identical (vector v) → `h = Z @ u.T` has all rows identical
2. `mean = h.mean(dim=0)` = same as any row → `h − mean = 0`
3. `std = 0` → `h_tilde = 0 / (0 + 1e−8) = 0`  for all k, m
4. EP test on all-zeros: `cos_part = cos(0) = 1`, `sin_part = sin(0) = 0`
5. `diff_sq = (1 − exp(−t²/2))² + 0² > 0`  → **Forward: T > 0** ✓

The SIGReg backward pass at collapse — **TWO independent zeros**:

**Zero 1 — EP test gradient vanishes at h_tilde = 0:**
```
∂(diff_sq)/∂(h_tilde_k) = 2·(cos_part − φ₀)·∂(cos_part)/∂(h_tilde_k)
                         + 2·sin_part·∂(sin_part)/∂(h_tilde_k)
```
At h_tilde = 0:
- `∂cos_part/∂h_tilde_k = −(1/K)·t·sin(t·0) = 0`  (sin(0) = 0)
- `sin_part = 0`  (so 0 × anything = 0)
- **∂T/∂h_tilde = 0 exactly**

**Zero 2 — Standardisation backward pass cancels all gradients:**
```
∂L/∂h_k = (1/(std+ε)) · [g_k − mean(g)]
```
At collapse, all h_k are identical → by symmetry, all g_k = ∂L/∂h_tilde_k are identical:
- `g_k − mean(g) = g_k − g_k = 0`
- **∂L/∂h = 0 exactly** (even if ∂L/∂h_tilde were nonzero)

**Conclusion:** SIGReg's loss is nonzero (~0.005) but its gradient is **exactly zero** — not approximately, not numerically small, but mathematically zero. The optimizer receives no signal from SIGReg at the collapsed state. The collapsed state is a **stable fixed point** of SIGReg-regularised training.

**This explains why ALL conditions that use SIGReg (C, D1, D2, D3, E) collapsed.** The SIGReg gradient dead zone is the root cause of every collapse observed across the entire experiment.

### The Fix: Remove Standardisation from SIGReg

**Change to `models/sigreg.py`:**

Removed lines 120–123 (the standardisation step) from the `sigreg()` function. Raw projections `h = Z @ u.T` now pass directly to the EP test, which tests against N(0,1) instead of "any Gaussian after standardising".

Also removed standardisation from `_ep_test_1d_scalar()` (used by `sigreg_channel` for Condition D2).

**Why this fixes the gradient dead zone:**

Without standardisation, at collapse with Z → constant vector v:
- `h[k,m] = v·u_m` (constant c_m per projection, but c_m ≠ 0 in general)
- `sin_part(t) = sin(t·c_m) ≠ 0`
- The product `sin_part · ∂sin_part/∂h_k` is **nonzero**
- No centering in the backward pass to cancel gradients

The gradient lives. SIGReg can now push representations away from collapse.

**What changes mathematically:**

| Property | Before (standardised) | After (un-standardised) |
|----------|----------------------|------------------------|
| Tests against | "Any Gaussian N(μ,σ²)" | "Specifically N(0,1)" |
| Collapse gradient | **Exactly zero** | Nonzero (proportional to deviation) |
| Scale sensitivity | None (scale-invariant) | Yes (pushes toward unit variance) |
| Cramér-Wold target | P_Z → N(μ,Σ) for some μ,Σ | P_Z → N(0,I) specifically |

The un-standardised SIGReg simultaneously:
1. **Prevents collapse** — nonzero gradient pushes collapsed samples apart
2. **Anchors scale** — pushes toward unit variance in all projection directions
3. **Tests Gaussianity** — EP test still measures distributional shape

By Cramér-Wold theorem: if all 1D projections follow N(0,1), the multivariate distribution is N(0,I). Un-standardised SIGReg pushes the full adapter output distribution toward isotropic unit Gaussian.

**Bonus:** L_cov (lambda_3) is no longer needed. SIGReg now provides its own scale anchoring. lambda_3 reverts to 0.0 for both E and F.

### Files Changed
- `models/sigreg.py` — Removed standardisation from `sigreg()` and `_ep_test_1d_scalar()`
- `configs/condition_E.yaml` — lambda_3 confirmed at 0.0
- `configs/condition_F.yaml` — lambda_3 confirmed at 0.0

### Jobs to Resubmit (All SIGReg Conditions)

| Condition | Why resubmit | Fresh or resume? |
|-----------|-------------|------------------|
| C  | SIGReg gradient was dead; prior run at step 39k was invalid | Fresh (new SIGReg behavior) |
| D1 | SIGReg gradient was dead; collapsed to erank=1.05 | Fresh |
| D2 | SIGReg gradient was dead + prior `module` bug | Fresh |
| D3 | SIGReg gradient was dead; prior run collapsed to erank=1.18 | Fresh |
| E  | SIGReg gradient was dead + L_info normalisation | Fresh |
| F  | L_info normalisation (no SIGReg, but needs clean run) | Fresh |

Conditions A and B are NOT affected (lambda_1=0, no SIGReg). Their completed results remain valid.

---

## Pending

- Commit SIGReg fix, push to GitHub, pull on HPC
- Run unit tests to verify SIGReg fix doesn't break existing tests
- Resubmit all 6 conditions (C, D1, D2, D3, E, F) as fresh runs
- W&B sync daemon still running on login node (PID active)
- After all 8 conditions complete: linear probes, analysis scripts, W&B dashboard
