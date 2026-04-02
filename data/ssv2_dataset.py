"""
SSv2 Feature Dataset — loads pre-extracted V-JEPA 2.1 patch features from disk.

Data format (build_spec.md Section 4):
    - Features saved as .pt files: {clip_id}_frame{t}.pt, {clip_id}_frame{t+1}.pt
    - Each .pt file: torch.Tensor of shape [196, 1024]  (N=196 patches, D=1024)
    - Labels saved as integers in a JSON index file

Split sizes (build_spec.md Section 4.2):
    Train:  16K clips
    Val:    2K clips  (used for eval metrics during training)
    Test:   2K clips  (touched ONCE — final linear probe)

Usage:
    dataset = SSv2FeatureDataset(feature_dir, split="train")
    f_c, f_t, label = dataset[0]
    # f_c: [196, 1024], f_t: [196, 1024], label: int
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class SSv2FeatureDataset(Dataset):
    """Dataset of pre-extracted V-JEPA 2.1 patch features for SSv2 clips.

    Expects the following directory layout under feature_dir:
        feature_dir/
            index.json          ← {clip_id: {"label": int, "t": int}} for all clips
            {clip_id}_fc.pt     ← context frame features [196, 1024]
            {clip_id}_ft.pt     ← target frame features [196, 1024]

    The index.json is split into train/val/test by preextract_ssv2.py.
    Each split entry lists clip IDs belonging to that split.

    Args:
        feature_dir: Root directory containing .pt feature files and index.json.
        split:       One of 'train', 'val', 'test'.
        dtype:       Dtype for returned tensors (default torch.float32).
    """

    def __init__(
        self,
        feature_dir: str,
        split: str = "train",
        dtype: torch.dtype = torch.float32,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.dtype = dtype

        # Load split index.
        index_path = self.feature_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"index.json not found at {index_path}. "
                "Run data/preextract_ssv2.py first."
            )
        with open(index_path) as f:
            index = json.load(f)

        self.clips: List[dict] = index[split]
        # clips: [{"clip_id": str, "label": int}, ...]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Load one (context, target, label) triple.

        Returns:
            f_c:   Context frame patch features, shape [196, 1024].
            f_t:   Target frame patch features, shape [196, 1024].
            label: SSv2 action class index (0..173).
        """
        entry = self.clips[idx]
        clip_id = entry["clip_id"]
        label   = entry["label"]

        fc_path = self.feature_dir / f"{clip_id}_fc.pt"
        ft_path = self.feature_dir / f"{clip_id}_ft.pt"

        f_c = torch.load(fc_path, map_location="cpu").to(self.dtype)  # [196, 1024]
        f_t = torch.load(ft_path, map_location="cpu").to(self.dtype)  # [196, 1024]

        return f_c, f_t, label

    def __repr__(self) -> str:
        return (
            f"SSv2FeatureDataset(split={self.split}, "
            f"n_clips={len(self.clips)}, "
            f"feature_dir={self.feature_dir})"
        )


def build_dataloaders(
    feature_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders.

    Args:
        feature_dir: Root directory with pre-extracted features.
        batch_size:  Batch size (default 32 per build_spec.md).
        num_workers: DataLoader worker processes.
        pin_memory:  Pin memory for faster GPU transfer.

    Returns:
        (train_loader, val_loader, test_loader)

    Batch output:
        f_c:   [B, 196, 1024] — context features
        f_t:   [B, 196, 1024] — target features
        label: [B]            — SSv2 action class labels
    """
    train_ds = SSv2FeatureDataset(feature_dir, split="train")
    val_ds   = SSv2FeatureDataset(feature_dir, split="val")
    test_ds  = SSv2FeatureDataset(feature_dir, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,     # drop incomplete batches to avoid metric edge cases
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def build_synthetic_dataloaders(
    batch_size: int = 32,
    n_train: int = 512,
    n_val: int = 128,
    N: int = 196,
    D: int = 1024,
    n_classes: int = 174,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build synthetic dataloaders for unit tests and smoke tests.

    Generates random Gaussian features without requiring real SSv2 data.
    Used by tests/test_smoke.py and pytest fixtures.

    Args:
        batch_size: Batch size.
        n_train:    Number of synthetic training samples.
        n_val:      Number of synthetic val samples.
        N:          Number of patch tokens (default 196).
        D:          Feature dimension (default 1024).
        n_classes:  Number of classes (default 174).

    Returns:
        (train_loader, val_loader, test_loader) with same API as build_dataloaders.
    """
    class SyntheticDataset(Dataset):
        def __init__(self, n: int):
            self.n = n
            self.f_c   = torch.randn(n, N, D)
            self.f_t   = torch.randn(n, N, D)
            self.labels = torch.randint(0, n_classes, (n,))

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int):
            return self.f_c[idx], self.f_t[idx], self.labels[idx].item()

    train_ds = SyntheticDataset(n_train)
    val_ds   = SyntheticDataset(n_val)
    test_ds  = SyntheticDataset(n_val)

    kw = dict(batch_size=batch_size, num_workers=0, drop_last=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    return train_loader, val_loader, test_loader
