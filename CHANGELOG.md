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

## Session 8 — SIGReg Gradient Analysis & Lambda Sweep (2026-04-03)

### Critical Finding: SIGReg Loss Is Bounded at Collapse (foundations.md Error)

**Definition 2.3 of foundations.md claims:** "SIGReg loss → ∞ under collapse, creating an infinite barrier."

**This is mathematically false.** Proof:

At collapse, all samples z_k = c (constant vector). For any projection direction u:
```
φ_K(t) = (1/K) Σ exp(ith_k) = exp(it·c·u)    [point mass CF]
φ₀(t)  = exp(-t²/2)                              [Gaussian CF]

|φ_K(t) - φ₀(t)|² = |exp(it·c·u) - exp(-t²/2)|² ≤ 4
```

Bounded integrand × bounded weight × finite interval [0.2, 4.0] → **SIGReg is bounded at collapse.**
Empirically confirmed: SIGReg stabilises at **0.2043** across ALL collapsed conditions.

### Critical Finding: SIGReg Gradient Is Negligible at Collapse

The gradient of the EP test w.r.t. sample h_k at collapse (all h_k = c·u):

```
∂T/∂h_k = ∫ w(t) · (2t/K) · exp(-t²/2) · sin(t·c·u) dt
```

Three problems:
1. **1/K factor** — each sample's gradient is diluted by 1/(batch_size)
2. **At exact collapse (c=0):** sin(0) = 0, gradient is **exactly zero**
3. **Near collapse:** all K gradients are identical → gradient is a **translation** (shifts mean), not a **dispersion** (spreads samples). Dispersion relies on adapter Jacobian differences across inputs, which shrink as the adapter converges to a constant mapping.

### Evidence from Resubmitted Jobs (5657048–5657053)

All jobs used the Session 7 fix (un-standardised SIGReg, lambda_1=0.1):

| Condition | Job | Steps seen | erank trajectory | SIGReg trajectory | Verdict |
|-----------|-----|-----------|-----------------|-------------------|---------|
| C (sg+sig) | 5657048 | 0→2000 | 76.8→4.01→1.05 | 0.209→0.151→0.070 | Collapsed |
| D1 (token) | 5657049 | 0→1500 | 76.8→5.10→1.02 | 0.199→0.204→0.204 | Collapsed |
| D2 (channel) | 5657050 | 0→6500 | 76.8→4.73→1.02-1.17 | 0.200→0.204→0.204 | Collapsed |
| D3 (global) | 5657051 | 0→2000 | 76.8→5.21→1.04 | 0.207→0.205→0.204 | Collapsed |
| E | 5657052 | — | PENDING (never ran) | — | Cancelled |
| F | 5657053 | — | PENDING (never ran) | — | Cancelled |

**Key observation:** SIGReg loss is STUCK at 0.204 while erank → 1. The optimizer completely ignores SIGReg because its gradient (gnorm ≈ 0.003) is negligible compared to L_pred's gradient.

**Condition C anomaly:** SIGReg *decreases* to 0.07 while erank drops to 1.05. This is likely a **small-sample artifact** — with K=32 (batch size for global SIGReg), the EP test has insufficient statistical power. The 32 pooled samples may appear "more Gaussian" to the test even though the representation is collapsing.

### Root Cause Analysis

The competition dynamics (D1 as example):

| Step | L_pred | SIGReg | λ₁×SIGReg | gnorm | erank |
|------|--------|--------|-----------|-------|-------|
| 100  | 0.0113 | 0.199  | 0.020     | 0.065 | ~5    |
| 500  | 0.0015 | 0.204  | 0.020     | 0.015 | 5.10  |
| 1000 | 0.0002 | 0.204  | 0.020     | 0.005 | 1.08  |

L_pred drops 56× while SIGReg barely moves. The **loss surface is a plateau**: high SIGReg value (0.204) but negligible gradient. L_pred has a steep slope toward collapse.

At lambda_1=0.1, SIGReg contributes ~92% of total loss but ~5% of gradient signal.

### Lambda Sweep 1: Global SIGReg (K=32) — FAILED

**Hypothesis:** At lambda_1=10+ (100× current), SIGReg gradient dominates L_pred, preventing collapse.

**Result: ALL THREE FAILED.** erank collapsed to ~1.1 by step 400 at every lambda value.

| lambda_1 | erank@200 | erank@400 | erank@800 | SIGReg@200 | SIGReg@800 |
|----------|-----------|-----------|-----------|------------|------------|
| 10  | 5.93 | **1.10** | 1.14 | 0.199 | **0.047** |
| 25  | 5.60 | **1.13** | ~1.0 | 0.184 | **0.045** |
| 50  | 5.23 | **1.06** | 1.03 | 0.183 | **0.044** |

### Critical Finding: Global SIGReg Is Cheatable (1D Variance Trick)

**SIGReg DECREASED while erank DECREASED.** The optimizer is REDUCING SIGReg by collapsing — the opposite of what should happen.

**Mechanism:** The adapter collapses to a 1D subspace with direction v and variance σ₁². Random projections u see variance ≈ σ₁²·(u·v)². In expectation over random u on S^{d-1}, E[(u·v)²] = 1/d. If the adapter sets σ₁² ≈ d (= 256), then E[var(projection)] ≈ 1, making projections look like N(0,1).

With only K=32 samples, the EP test has insufficient statistical power to detect that 255/256 projection directions have near-zero variance (those few samples at ~0 look compatible with N(0,1) at small sample size).

**Conclusion:** Global SIGReg (K=B=32) is **structurally broken** for collapse prevention when d >> K. Higher lambda makes it WORSE (optimizer exploits loophole faster).

**Note on Condition C anomaly (Session 7):** The "SIGReg decreasing while collapsing" seen in Condition C (0.209→0.070) was NOT a small-sample artifact — it was the 1D variance trick, same mechanism now confirmed at lambda=10/25/50.

### Lambda Sweep 2: Token SIGReg (K=6272) — FAILED

Token SIGReg uses K=B×N=32×196=6,272 samples. Hypothesis: with 6272 samples
the EP test has enough power that the 1D variance trick cannot work.

**Jobs:** 5657700 (lam10), 5657701 (lam25), 5657702 (lam50)

**Result: ALSO COLLAPSED.** Same pattern — SIGReg decreases while erank decreases:

| Step | erank | SIGReg | L_pred | (token lam10) |
|------|-------|--------|--------|---------------|
| 200  | 4.40  | 0.071  | 0.208  |               |
| 400  | **1.12** | **0.025** | 0.145 |            |
| 600  | **1.05** | **0.019** | 0.095 |            |
| 800  | ~1.0  | **0.015** | 0.089  |               |

The 1D variance trick works even at K=6272. The adapter collapses to a 1D
subspace where the distribution along that direction is approximately Gaussian
with variance tuned so random projections see ≈ N(0,1). With 6272 samples
the distribution is even MORE convincingly Gaussian (CLT), not less.

**Conclusion:** Increasing lambda or K alone does not fix SIGReg. The failure
is not about gradient strength or statistical power — it's about our experimental
setup differing from how the SIGReg papers (LeJEPA, LeWorldModel) use it.

### Root Cause Analysis: Why Our Setup Differs From the Papers

We compared our setup with the original SIGReg papers (LeJEPA, LeWorldModel):

| Setting | Our setup | LeJEPA / LeWorldModel |
|---------|-----------|----------------------|
| Batch size | 32 | 2048–4096 |
| Init | Near-identity (gain=0.1, std≈0.08) | Random (std≈1.0) |
| BatchNorm on projector | No | **Yes** (LeWorldModel: "critical") |
| Adapter output dim | 256 | 256–8192 |

Three key differences identified:

1. **Batch size (K=32 vs K=2048+):** Both papers use large batches. The EP test
   on 32 samples has far less statistical power than on 2048. More importantly,
   the 1D variance trick is harder to exploit with more samples.

2. **Init gain (0.1 vs 1.0):** Our gain=0.1 gives initial output std≈0.08,
   which is 13× too small for the N(0,1) target. SIGReg must simultaneously
   increase scale AND prevent collapse — two competing objectives. Random init
   (gain=1.0) starts at std≈0.87, much closer to the target.

3. **BatchNorm before SIGReg:** LeWorldModel explicitly states BN on the
   projector is critical for SIGReg to work. BN normalises each feature
   dimension to zero mean and unit variance across the batch. This means
   SIGReg only needs to enforce the SHAPE of the distribution (Gaussianity),
   not the scale. Without BN, the adapter can satisfy SIGReg by tuning the
   variance of a 1D projection without achieving true isotropy.

### Diagnostic Sweep: Isolating the Three Factors

Three controlled 2000-step runs on Condition D3 (global SIGReg, no stop-grad,
lambda_1=0.1). Each adds one factor cumulatively:

| Run | batch_size | init_gain | BN  | What it isolates | Job ID |
|-----|-----------|-----------|-----|-----------------|--------|
| A (done) | 32 | 0.1 | No | Baseline — collapsed | (prior) |
| B | **128** | 0.1 | No | Batch size alone? | 5659135 |
| C | **128** | **1.0** | No | + random init? | 5659136 |
| D | **128** | **1.0** | **Yes** | + BatchNorm? (full fix) | 5659137 |

**Reading guide:**
- B works → batch size was the issue (K=32 too few)
- B fails, C works → random init is key (starting near N(0,I))
- B+C fail, D works → BN is critical (confirms LeWorldModel finding)
- All fail → SIGReg genuinely doesn't work for frozen-feature adapter setup.
  **That IS a valid paper finding.**

### Diagnostic Sweep Results: ALL THREE FAILED

| Run | Intervention | erank@200 | erank@400 | erank@600 | SIGReg@200 | SIGReg@500 |
|-----|-------------|-----------|-----------|-----------|------------|------------|
| A (baseline) | none (B=32, gain=0.1, no BN) | ~5 | ~1.0 | ~1.0 | 0.204 | 0.204 |
| B | batch=128 | 5.75 | **1.03** | 1.06 | 0.207 | 0.205 |
| C | batch=128 + gain=1.0 | 4.89 | **1.05** | (running) | 0.232 | 0.208 |
| D | batch=128 + gain=1.0 + BN | 4.90 | **1.05** | 1.04 | 0.123 | 0.113 |

**Every combination collapsed to erank ≈ 1.03–1.05 by step 400.**

Per-run analysis:

**Run B (batch=128 only):** SIGReg stuck at 0.205 — completely ignored by the
optimizer. L_pred drops 0.018→0.001 while gnorm falls to 0.011. Batch size
alone doesn't help because at lambda_1=0.1, SIGReg contributes only 0.02 to
the total loss (L_pred ≈ 0.5–2.5 initially). The optimizer has no reason to
even look at SIGReg.

**Run C (batch=128 + gain=1.0):** Initial SIGReg is much higher (0.463) because
random init + GELU produces a non-Gaussian distribution. SIGReg drops rapidly
(0.463→0.208) as the adapter learns to be more Gaussian — but this trajectory
IS the collapse. The adapter converges to a collapsed point that also happens
to be somewhat Gaussian-looking.

**Run D (batch=128 + gain=1.0 + BN):** SIGReg starts LOW (0.122) because BN
normalises each feature to mean=0, std=1 before the EP test. This is
**counterproductive**: at 1D collapse, BN normalises the single dominant
direction across all 256 features, making the collapsed representation LOOK
256-dimensional to SIGReg. The erank (computed on raw z_c, not BN-normalised)
still drops to 1.05. BN masks the collapse rather than preventing it.

### Definitive Conclusion

**SIGReg at lambda_1=0.1 cannot prevent collapse** regardless of batch size,
init gain, or BatchNorm. The weighted SIGReg contribution (λ₁ × 0.20 = 0.02)
is negligible compared to L_pred (0.5–2.5 initially). The optimizer ignores it.

Combined with the earlier lambda sweep result (lambda_1=10–50 enables the 1D
variance trick where the optimizer REDUCES SIGReg by collapsing), the full
picture is:

- **Low lambda (0.1):** SIGReg is too weak — ignored by the optimizer
- **High lambda (10–50):** SIGReg has spurious minima at 1D — exploited by the optimizer
- **BatchNorm:** Masks collapse from SIGReg rather than preventing it
- **Batch size / init:** Secondary factors that don't address the structural issue

**SIGReg cannot replace stop-gradient/EMA as a collapse prevention mechanism
in a shared-weight JEPA adapter architecture.** This is a genuine negative
finding suitable for the paper.

### Code Changes

- `models/adapter.py` — Added `init_gain` parameter (default 0.1, backward-compatible).
  gain=0.1 gives near-identity init (std≈0.08), gain=1.0 gives random init (std≈0.87).
- `models/losses.py` — Added `use_sigreg_bn` flag in LossConfig. When True,
  a BatchNorm1d layer normalises z_c per-feature across the batch BEFORE SIGReg.
  Metrics are always computed on raw z_c (no BN) so collapse is still observable.
- `training/trainer.py` — Creates optional BN module, includes its parameters
  in the optimizer, passes it to compute_loss. Prints `use_sigreg_bn` and
  `adapter_init_gain` in the config summary.
- `configs/diag_B_batchsize.yaml` — batch_size=128 only
- `configs/diag_C_initgain.yaml` — batch_size=128 + init_gain=1.0
- `configs/diag_D_withbn.yaml` — batch_size=128 + init_gain=1.0 + use_sigreg_bn=True

### Files Changed (Session 8 cumulative)
- `configs/sweep_lambda{10,25,50}.yaml` (new) — global lambda sweep
- `configs/sweep_token_lam{10,25,50}.yaml` (new) — token lambda sweep
- `configs/diag_{B,C,D}_*.yaml` (new) — diagnostic sweep configs
- `scripts/run_lambda_sweep.sh` (new) — lambda sweep SLURM script
- `scripts/run_diagnostic.sh` (new) — diagnostic sweep SLURM script
- `models/adapter.py` — configurable init_gain
- `models/losses.py` — optional use_sigreg_bn
- `training/trainer.py` — BN module creation, init_gain passthrough
- `CHANGELOG.md` — this entry

### Path 1: Stop-Gradient + SIGReg (lambda=1.0, batch=128) — FAILED

**Hypothesis:** Stop-grad breaks the direct collapse pathway. With the direct
force removed, SIGReg at lambda=1.0 should dominate the weak indirect collapse
force and prevent collapse.

**Jobs:** 5660181 (stop_grad=True), 5660182 (stop_grad=False, control)

**Result: BOTH COLLAPSED by step 400.** But with critically different dynamics:

**p1-nosg (no stop-grad):** Classic pattern. SIGReg stuck at 0.204 (ignored),
L_pred → 0, gnorm → 0. Silent collapse.

| Step | L_pred | SIGReg | erank | gnorm |
|------|--------|--------|-------|-------|
| 200  | 0.007  | 0.204  | 5.70  | 0.048 |
| 400  | 0.003  | 0.204  | 1.13  | 0.016 |
| 600  | 0.001  | 0.204  | 1.07  | 0.010 |

**p1-sg (stop-grad):** Phase transition at step 250–300. SIGReg DROPS sharply
(0.19→0.06), L_pred SPIKES (0.04→0.86), erank crashes (5.9→1.1).

| Step | L_pred | SIGReg | erank | gnorm |
|------|--------|--------|-------|-------|
| 200  | 0.019  | 0.201  | 5.91  | 0.040 |
| 300  | 0.857  | 0.057  | ~1    | 0.745 |
| 400  | 0.604  | 0.043  | 1.08  | 0.501 |
| 600  | 0.348  | 0.036  | 1.13  | 0.450 |

**Critical finding:** With stop-grad, SIGReg is not just failing to prevent
collapse — it is **actively driving the collapse**. The optimizer finds that
the 1D Gaussian configuration (erank≈1, SIGReg≈0.04) has lower SIGReg than
the initial 5D configuration (erank≈6, SIGReg≈0.20). SIGReg's gradient
points TOWARD the 1D local minimum, not away from it.

The L_pred spike (0.04→0.86) at step 300 is the adapter restructuring to
satisfy SIGReg. The predictor, trained on the old representation, suddenly
faces a collapsed input and produces large errors. It then re-learns the
collapsed representation (L_pred recovers from 0.86→0.35 by step 600).

**Conclusion: The 1D variance trick is the fundamental, insurmountable blocker
for SIGReg.** Stop-grad, high lambda, batch size, init gain, BN — none address
it because the problem is in the SIGReg loss landscape itself. The EP test on
random projections has spurious local minima at low-dimensional distributions.

### Summary of All Attempts (Session 8)

| Attempt | What we tried | Result | Why it failed |
|---------|--------------|--------|---------------|
| Lambda sweep (global) | lambda=10/25/50, B=32 | Collapsed | 1D variance trick (SIGReg↓ while erank↓) |
| Lambda sweep (token) | lambda=10, K=6272 | Collapsed | Same trick at larger K |
| Diag-B | B=128, gain=0.1, no BN | Collapsed | SIGReg ignored (too weak vs L_pred) |
| Diag-C | B=128, gain=1.0, no BN | Collapsed | Collapse = convergence to 1D Gaussian |
| Diag-D | B=128, gain=1.0, BN | Collapsed | BN masks collapse from SIGReg |
| Path 1 (sg) | stop_grad + lambda=1.0, B=128 | Collapsed | SIGReg DRIVES collapse to 1D minimum |
| Path 1 (nosg) | no stop_grad + lambda=1.0, B=128 | Collapsed | Classic: SIGReg ignored |

**7 experiments, 0 successes.** SIGReg cannot prevent collapse in this setup.

---

## Session 9 — Path 2: L_cov + SIGReg (2026-04-03)

### Code Changes

- **`models/losses.py`** — Changed `l_cov(z_hat)` → `l_cov(z_c)` in `compute_loss()`.
  L_cov must regularize the adapter output (the representation we want full-rank),
  not the predictor output. At 1D collapse: L_cov(z_c) ≈ d-1 ≈ 255 → massive
  anti-collapse gradient. At Cov=I: L_cov=0 → steps aside, lets L_pred + SIGReg work.

- **`configs/path2_lcov_sigreg.yaml`** (new) — Path 2 experiment config:
  - `lambda_1=0.1` (SIGReg for distributional shape)
  - `lambda_3=0.01` (L_cov as collapse barrier; effective weight ≈ 0.01×253 ≈ 2.53 at init)
  - `batch_size=128`, `stop_grad=false`, `sigreg_axis=global`
  - Key idea: L_cov has NO 1D local minimum (unlike SIGReg). Its only minimum is
    Cov(z)=I (full rank). Once L_cov prevents collapse, SIGReg shapes the distribution.

### Results — FAILED (job 5660722, 2026-04-03)

Collapse by step 400. erank=1.04 despite L_cov dominating the total loss (2.35/2.40).

| Step | erank | L_pred | SIGReg | L_cov (inferred) |
|------|-------|--------|--------|-----------------|
| 50   | —     | 0.018  | 0.207  | ≈255            |
| 200  | 5.62  | 0.022  | 0.195  | ≈251            |
| 400  | 1.04  | 0.038  | 0.155  | ≈235            |
| 800  | 1.07  | 0.042  | 0.123  | ≈219            |
| 1000 | 1.07  | 0.038  | 0.113  | ≈217            |

**Root cause — L_cov is NOT a collapse barrier:**

At near-zero init (gain=0.1), Cov(z_pool) ≈ 0, so L_cov ≈ d = 256. As the
adapter collapses toward 1D with growing variance σ₁²:

    L_cov = (σ₁² - 1)² + (d - 1)

As σ₁²: 0 → 1, L_cov: 256 → 255. L_cov **decreases** on the collapse path.

The 1D unit-variance configuration is the minimum of L_cov on the collapse
trajectory, not a maximum. L_cov cooperates with collapse rather than
resisting it. Additionally, the L_cov gradient ∝ z_pool (near-zero at init),
so it is weakest exactly when it needs to be strongest.

Both L_cov and SIGReg share the same structural failure: both have spurious
"acceptable" configurations at low-dimensional structures. Neither is a true
gradient-sense collapse barrier.

**Mechanism distinguishing VICReg variance term from L_cov:**

VICReg variance: V = Σᵢ max(0, γ - std(z_pool[:, i]))

Gradient in dimension i: ∂V/∂z[b,i] ∝ -1/std_i (huge as std_i → 0).
This applies independent, growing gradient to each dead dimension. A 1D
unit-variance collapse still leaves 255 dimensions with std≈0 → 255 terms
with enormous gradient. V cannot be satisfied by any low-dimensional structure.

L_cov gradient: ∝ (Cov - I) * z_pool (small when z_pool is small). No
amplification near collapse — the gradient shrinks exactly when most needed.

---

## Pending

- **Path 3 (candidate): VICReg variance term** as true collapse barrier.
  V = Σᵢ max(0, γ - std(z_c_pool[:, i])). Gradient ∝ 1/std_i → ∞ as collapse.
  No low-dimensional local minimum. This is directional (per-dim), unlike L_cov (global).
- Correct foundations.md Definition 2.3 (SIGReg is bounded, not infinite, at collapse)
- After collapse prevention works: resubmit all 8 conditions with revised losses
- Linear probes, analysis scripts, W&B dashboard
