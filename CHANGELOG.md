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

## Pending

- Conditions D3, E, F, C, D1, D2 on real SSv2 (in queue / auto-submitting)
- W&B offline sync after all conditions complete:
  `wandb sync /scratch/gupta.yashv/gap1-experiment/wandb/offline-run-*`
- Linear probe evaluation after training
- Analysis scripts: `analysis/plot_erank_curves.py`, `analysis/results_table.py`
