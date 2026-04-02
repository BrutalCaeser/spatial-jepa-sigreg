"""
Main training loop for GAP 1 experiment.

Supports all 8 conditions (A–F) via LossConfig dispatched from YAML.
Logs all metrics to W&B and console.

Usage:
    # Full run:
    python -m training.trainer \
        --config configs/base.yaml \
        --override configs/condition_E.yaml \
        --wandb_project gap1-sigreg-spatial \
        --wandb_entity YOUR_ENTITY

    # Smoke test (100 steps, no W&B):
    python -m training.trainer \
        --config configs/base.yaml \
        --override configs/condition_A.yaml \
        --smoke_test --max_steps 100

Critical constraints enforced here (CLAUDE.md):
    - NO torch.compile or torch.jit on adapter (hides gradient flow issues)
    - NO LR warmup for Condition A (slows collapse detection)
    - weight_decay=0 for Conditions A and F (loaded from YAML)
    - Metrics logged on CONCATENATED val data (not per-batch mean)
    - All metrics from W&B logs — never hardcode results
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml

# Allow running as module from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.adapter import PatchAdapter
from models.predictor import JEPAPredictor
from models.losses import LossConfig, compute_loss, loss_config_from_dict
from training.metrics import compute_all_metrics, compute_baseline_metrics


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(base_path: str, override_path: Optional[str] = None) -> dict:
    """Load base YAML config, optionally merging a condition-specific override.

    Args:
        base_path:     Path to configs/base.yaml.
        override_path: Path to configs/condition_X.yaml (optional).

    Returns:
        Merged config dict.
    """
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    if override_path is not None:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        cfg.update(override)

    return cfg


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_annealing_with_warmup(
    step: int,
    n_steps: int,
    warmup_steps: int,
    lr: float,
) -> float:
    """Cosine annealing LR schedule with linear warmup.

    Returns:
        Current learning rate scaling factor (multiply by base lr).
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Main trainer for all GAP 1 experimental conditions.

    Args:
        adapter:      PatchAdapter module.
        predictor:    JEPAPredictor module.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        cfg:          Full config dict (merged base + condition override).
        wandb_run:    Active W&B run (or None in smoke-test mode).
        smoke_test:   If True, skip saving and limit steps.
        max_steps:    Override n_steps (used for smoke testing).
    """

    def __init__(
        self,
        adapter: PatchAdapter,
        predictor: JEPAPredictor,
        train_loader,
        val_loader,
        cfg: dict,
        wandb_run=None,
        smoke_test: bool = False,
        max_steps: Optional[int] = None,
    ):
        self.adapter    = adapter
        self.predictor  = predictor
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg        = cfg
        self.wandb_run  = wandb_run
        self.smoke_test = smoke_test

        # Core training hyperparameters.
        self.n_steps      = max_steps if max_steps is not None else cfg["n_steps"]
        self.lr           = cfg["learning_rate"]
        self.weight_decay = cfg["weight_decay"]
        self.warmup_steps = cfg.get("warmup_steps", 2000)
        self.lr_warmup    = cfg.get("lr_warmup", True)  # False for Condition A
        self.grad_clip    = cfg.get("grad_clip", 1.0)

        # Logging intervals.
        self.log_every   = cfg.get("log_every", 100)
        self.eval_every  = cfg.get("eval_every", 500)
        self.probe_every = cfg.get("probe_every", 5000)
        self.save_every  = cfg.get("save_every", 10000)

        # Loss config parsed from YAML.
        self.loss_cfg = loss_config_from_dict(cfg)

        # Output directory.
        self.output_dir = Path(cfg.get("output_dir", "outputs")) / cfg.get("condition", "unknown")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter.to(self.device)
        self.predictor.to(self.device)

        # Build optimizer with BOTH adapter and predictor parameters.
        # DO NOT share optimizer across conditions (each gets a fresh one).
        self.optimizer = torch.optim.AdamW(
            list(self.adapter.parameters()) + list(self.predictor.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Track step count.
        self.step = 0

        print(f"[Trainer] Condition: {cfg.get('condition', 'unknown')}")
        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Steps: {self.n_steps}")
        print(f"[Trainer] weight_decay: {self.weight_decay}")
        print(f"[Trainer] stop_grad: {self.loss_cfg.stop_grad}")
        print(f"[Trainer] lambda_1 (SIGReg): {self.loss_cfg.lambda_1}")
        print(f"[Trainer] lambda_2 (L_info): {self.loss_cfg.lambda_2}")
        print(f"[Trainer] sigreg_axis: {self.loss_cfg.sigreg_axis}")
        print(f"[Trainer] use_dense_info: {self.loss_cfg.use_dense_info}")
        print(f"[Trainer] lr_warmup: {self.lr_warmup}")

    # ------------------------------------------------------------------
    # LR update
    # ------------------------------------------------------------------

    def _update_lr(self) -> float:
        """Update optimizer learning rate according to schedule.

        Condition A has lr_warmup=False: flat lr from step 0.
        All others: cosine annealing with warmup.

        Returns:
            Current lr value (for logging).
        """
        if not self.lr_warmup:
            # No warmup (Condition A): flat lr throughout.
            lr = self.lr
        else:
            scale = cosine_annealing_with_warmup(
                self.step, self.n_steps, self.warmup_steps, self.lr
            )
            lr = self.lr * scale

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        return lr

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _step_fn(self, batch: Tuple) -> Dict[str, float]:
        """Execute one training step.

        Args:
            batch: (f_c, f_t, label) from DataLoader.

        Returns:
            Dict of scalar loss values for logging.
        """
        f_c, f_t, label = batch
        f_c   = f_c.to(self.device)      # [B, N, D]
        f_t   = f_t.to(self.device)      # [B, N, D]
        label = label.to(self.device)    # [B]

        self.optimizer.zero_grad()

        # Forward pass through adapter (shared weights for context and target).
        # CRITICAL: z_c always has gradient (adapter params).
        #           z_t gradient depends on condition (stop_grad flag in LossConfig).
        z_c  = self.adapter(f_c)         # [B, N, d]  — gradient through adapter
        z_t  = self.adapter(f_t)         # [B, N, d]  — may be detached inside compute_loss

        # Forward pass through predictor.
        z_hat = self.predictor(z_c, label)  # [B, N, d]  — gradient through predictor + adapter

        # Compute loss (dispatches based on condition config).
        loss_dict = compute_loss(z_c, z_t, z_hat, self.loss_cfg)
        total_loss = loss_dict["total"]

        # Gradient check (smoke test mode only).
        if self.smoke_test and torch.isnan(total_loss):
            raise RuntimeError(f"NaN loss at step {self.step}")
        if torch.isinf(total_loss):
            raise RuntimeError(f"Inf loss at step {self.step}")

        # Backward pass.
        total_loss.backward()

        # Gradient clipping.
        grad_norm = nn.utils.clip_grad_norm_(
            list(self.adapter.parameters()) + list(self.predictor.parameters()),
            max_norm=self.grad_clip,
        ).item()

        self.optimizer.step()

        return {
            "train/loss_total":  total_loss.item(),
            "train/loss_pred":   loss_dict["l_pred"].item(),
            "train/loss_sig":    loss_dict["l_sig"].item(),
            "train/loss_info":   loss_dict["l_info"].item(),
            "train/loss_cov":    loss_dict["l_cov"].item(),
            "train/grad_norm":   grad_norm,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval(self) -> Dict[str, float]:
        """Compute all validation metrics on concatenated val data.

        CRITICAL: concatenate ALL val batches before computing metrics.
        Never average per-batch metrics (effective rank is non-linear).
        """
        self.adapter.eval()
        self.predictor.eval()

        z_c_list, z_t_list, z_hat_list = [], [], []

        for f_c, f_t, label in self.val_loader:
            f_c   = f_c.to(self.device)
            f_t   = f_t.to(self.device)
            label = label.to(self.device)

            z_c  = self.adapter(f_c)
            z_t  = self.adapter(f_t)
            z_hat = self.predictor(z_c, label)

            z_c_list.append(z_c.cpu())
            z_t_list.append(z_t.cpu())
            z_hat_list.append(z_hat.cpu())

        # Concatenate all batches before computing metrics.
        z_c_all   = torch.cat(z_c_list,   dim=0)   # [K, N, d]
        z_t_all   = torch.cat(z_t_list,   dim=0)
        z_hat_all = torch.cat(z_hat_list, dim=0)

        metrics = compute_all_metrics(z_c_all, z_t_all, z_hat_all)

        self.adapter.train()
        self.predictor.train()

        return metrics

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Save adapter and predictor checkpoints."""
        if self.smoke_test:
            return
        ckpt = {
            "step":      self.step,
            "adapter":   self.adapter.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg":       self.cfg,
        }
        path = self.output_dir / f"checkpoint_step{self.step:06d}.pt"
        torch.save(ckpt, path)
        print(f"[Trainer] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint to resume training."""
        ckpt = torch.load(path, map_location=self.device)
        self.adapter.load_state_dict(ckpt["adapter"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        print(f"[Trainer] Resumed from step {self.step}")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop for n_steps gradient steps.

        Logging (W&B + console):
            - Loss components every log_every steps
            - Validation metrics every eval_every steps
            - Linear probe every probe_every steps
            - Checkpoint every save_every steps
        """
        self.adapter.train()
        self.predictor.train()

        # Infinite data iterator (cycle over DataLoader).
        data_iter = iter(self.train_loader)
        t0 = time.time()

        print(f"[Trainer] Starting training. Total steps: {self.n_steps}")

        while self.step < self.n_steps:
            # Fetch next batch, cycling if needed.
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Update LR before step.
            lr = self._update_lr()

            # Execute training step.
            log_dict = self._step_fn(batch)
            log_dict["train/lr"] = lr
            log_dict["step"] = self.step

            self.step += 1

            # --- Logging ---
            if self.step % self.log_every == 0:
                elapsed = time.time() - t0
                steps_per_sec = self.log_every / elapsed
                t0 = time.time()

                print(
                    f"[step {self.step:6d}/{self.n_steps}] "
                    f"loss={log_dict['train/loss_total']:.4f} "
                    f"pred={log_dict['train/loss_pred']:.4f} "
                    f"sig={log_dict['train/loss_sig']:.4f} "
                    f"info={log_dict['train/loss_info']:.4f} "
                    f"gnorm={log_dict['train/grad_norm']:.3f} "
                    f"lr={lr:.2e} "
                    f"({steps_per_sec:.1f} steps/s)"
                )

                if self.wandb_run is not None:
                    self.wandb_run.log(log_dict, step=self.step)

            # --- Evaluation metrics ---
            if self.step % self.eval_every == 0:
                eval_metrics = self._eval()
                eval_metrics["step"] = self.step

                print(
                    f"[eval step {self.step}] "
                    f"erank={eval_metrics['eval/erank']:.2f} "
                    f"ncorr={eval_metrics['eval/ncorr_adapter']:.3f} "
                    f"xcov={eval_metrics['eval/xcov_trace']:.3f} "
                    f"infonce={eval_metrics['eval/infonce_mi']:.3f}"
                )

                if self.wandb_run is not None:
                    self.wandb_run.log(eval_metrics, step=self.step)

                # Halt rule for Condition A (build_spec.md FM1):
                # If erank > 5 after 5000 steps, halt and diagnose.
                if (
                    self.cfg.get("condition") == "A"
                    and self.step >= 5000
                    and eval_metrics["eval/erank"] > 5.0
                ):
                    print(
                        "[HALT FM1] Condition A erank > 5 after 5000 steps. "
                        "Likely implicit regularization. See build_spec.md FM1. "
                        "Check weight_decay, init, grad_clip."
                    )
                    # Log halt event to W&B.
                    if self.wandb_run is not None:
                        self.wandb_run.log({"fm1_halt": 1}, step=self.step)
                    break

            # --- Linear probe ---
            if not self.smoke_test and self.step % self.probe_every == 0:
                self._run_probe()

            # --- Checkpoint ---
            if not self.smoke_test and self.step % self.save_every == 0:
                self._save()

        print(f"[Trainer] Training complete at step {self.step}.")

    def _run_probe(self) -> None:
        """Run linear probe evaluation and log results."""
        try:
            from evaluation.linear_probe import run_linear_probe
            # Build a test loader from the val loader for now.
            # In production, use the separate test set from build_dataloaders().
            top1, top5 = run_linear_probe(self.adapter, self.val_loader, self.device)
            print(f"[probe step {self.step}] top1={top1:.3f} top5={top5:.3f}")
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "eval/probe_top1": top1,
                    "eval/probe_top5": top5,
                }, step=self.step)
        except Exception as e:
            print(f"[probe] WARNING: probe failed at step {self.step}: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GAP 1 Experiment Trainer")
    parser.add_argument("--config",   required=True, help="Path to base.yaml")
    parser.add_argument("--override", default=None,  help="Path to condition_X.yaml")
    parser.add_argument("--wandb_project", default="gap1-sigreg-spatial")
    parser.add_argument("--wandb_entity",  default=None)
    parser.add_argument("--smoke_test",    action="store_true", help="Run 100 steps, no W&B")
    parser.add_argument("--max_steps",     type=int, default=None, help="Override n_steps")
    parser.add_argument("--resume",        default=None, help="Checkpoint path to resume from")
    parser.add_argument("--no_wandb",      action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    # Load config.
    cfg = load_config(args.config, args.override)

    # Seed for reproducibility.
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # Build models.
    adapter = PatchAdapter(
        D_in=cfg.get("D_in", 1024),
        D_out=cfg.get("d", 256),
    )
    predictor = JEPAPredictor(
        d=cfg.get("d", 256),
        n_layers=cfg.get("n_layers", 6),
        n_heads=cfg.get("n_heads", 8),
        n_classes=cfg.get("n_classes", 174),
        d_action=cfg.get("d_action", 64),
        dropout=cfg.get("dropout", 0.1),
    )

    # Build data loaders.
    feature_dir = cfg.get("feature_dir", None)
    if feature_dir is None or args.smoke_test:
        print("[Trainer] Using synthetic data (smoke test or no feature_dir).")
        from data.ssv2_dataset import build_synthetic_dataloaders
        train_loader, val_loader, _ = build_synthetic_dataloaders(
            batch_size=cfg.get("batch_size", 4) if args.smoke_test else cfg.get("batch_size", 32),
            n_train=64 if args.smoke_test else 512,
            n_val=32 if args.smoke_test else 128,
        )
    else:
        from data.ssv2_dataset import build_dataloaders
        train_loader, val_loader, _ = build_dataloaders(
            feature_dir=feature_dir,
            batch_size=cfg.get("batch_size", 32),
            num_workers=cfg.get("num_workers", 4),
        )

    # Initialize W&B.
    wandb_run = None
    if not args.smoke_test and not args.no_wandb:
        try:
            import wandb
            condition = cfg.get("condition", "unknown")
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"condition_{condition}",
                config=cfg,
                tags=[f"condition_{condition}", "gap1", "v3"],
            )
        except ImportError:
            print("[Trainer] W&B not installed. Logging to console only.")

    # Build trainer.
    trainer = Trainer(
        adapter=adapter,
        predictor=predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        wandb_run=wandb_run,
        smoke_test=args.smoke_test,
        max_steps=args.max_steps,
    )

    # Resume from checkpoint if specified.
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Log baseline metrics (run once before training).
    if not args.smoke_test and wandb_run is not None:
        _log_baseline(trainer, val_loader, wandb_run)

    # Run training.
    trainer.train()

    if wandb_run is not None:
        wandb_run.finish()


def _log_baseline(trainer: Trainer, val_loader, wandb_run) -> None:
    """Log baseline V-JEPA 2.1 feature metrics before training starts."""
    device = trainer.device
    f_list = []
    with torch.no_grad():
        for f_c, f_t, _ in val_loader:
            f_list.append(f_c)
            if len(f_list) * f_c.shape[0] >= 512:
                break
    f_all = torch.cat(f_list, dim=0).to(device)

    baseline = compute_baseline_metrics(f_all)
    print(f"[baseline] raw_erank={baseline['baseline/raw_erank']:.2f} "
          f"raw_ncorr={baseline['baseline/raw_ncorr']:.3f}")
    wandb_run.log(baseline, step=0)


if __name__ == "__main__":
    main()
