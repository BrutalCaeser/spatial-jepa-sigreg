#!/usr/bin/env python3
"""
Download Something-Something v2 (SSv2) from HuggingFace and prepare the
directory layout expected by data/preextract_ssv2.py.

Dataset: https://huggingface.co/datasets/HuggingFaceM4/something_something_v2
License: Qualcomm research license (non-commercial).

Output layout:
    $DATA_DIR/
        labels.json       ← {"template text": label_int, ...}  (174 entries)
        annotations.json  ← [{"id": str, "template": str}, ...]
        videos/
            {video_id}.mp4

Usage (on HPC login node or via SLURM):
    python scripts/download_ssv2_hf.py \
        --output_dir /scratch/$USER/data/ssv2 \
        --split train \
        --n_clips 20000 \
        --hf_cache /scratch/$USER/hf_cache

Then run:
    sbatch scripts/preextract_ssv2.sh
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


HF_DATASET_NAME = "HuggingFaceM4/something_something_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SSv2 from HuggingFace")
    parser.add_argument(
        "--output_dir",
        default=f"/scratch/{os.environ.get('USER', 'user')}/data/ssv2",
        help="Root directory for SSv2 data",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to download (use 'train' for 168K clips)",
    )
    parser.add_argument(
        "--n_clips",
        type=int,
        default=20000,
        help="Max clips to download (default 20000 for GAP 1 experiment)",
    )
    parser.add_argument(
        "--hf_cache",
        default=f"/scratch/{os.environ.get('USER', 'user')}/hf_cache",
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace token (if dataset requires authentication)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Parallel workers for dataset processing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for clip selection",
    )
    return parser.parse_args()


def load_ssv2_hf(split: str, hf_cache: str, hf_token: str | None):
    """Load SSv2 from HuggingFace with streaming."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' library not installed.")
        print("        Run: pip install datasets")
        sys.exit(1)

    os.environ["HF_DATASETS_CACHE"] = hf_cache
    Path(hf_cache).mkdir(parents=True, exist_ok=True)

    print(f"[dl] Loading {HF_DATASET_NAME} split='{split}' (streaming)...")
    print(f"[dl] HF cache: {hf_cache}")
    print(f"[dl] This may take several minutes for the first run.")

    try:
        ds = load_dataset(
            HF_DATASET_NAME,
            split=split,
            streaming=True,
            token=hf_token,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        print()
        print("If you see an authentication error, run:")
        print("  huggingface-cli login")
        print("  # or pass --hf_token YOUR_TOKEN")
        print()
        print("If you see a license agreement error, visit:")
        print(f"  https://huggingface.co/datasets/{HF_DATASET_NAME}")
        print("  Click 'Agree and access repository'")
        sys.exit(1)

    return ds


def build_labels_map(ds, n_probe: int = 5000) -> dict[str, int]:
    """
    Build the labels.json mapping: {"template text": label_int, ...}.

    Streams n_probe samples to collect all class names observed.
    SSv2 has exactly 174 classes; all should appear within 5K samples.
    """
    print(f"[dl] Building label map (probing {n_probe} samples)...")
    label_map: dict[str, int] = {}

    for i, sample in enumerate(ds):
        if i >= n_probe:
            break
        text = sample.get("text", sample.get("label", ""))
        label_int = sample.get("label", -1)
        if isinstance(label_int, int) and text and text not in label_map:
            label_map[text] = label_int

    print(f"[dl] Found {len(label_map)} unique action classes.")
    return label_map


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    labels_path = output_dir / "labels.json"
    annotations_path = output_dir / "annotations.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("[dl] SSv2 Download — HuggingFaceM4/something_something_v2")
    print(f"[dl] Output dir:  {output_dir}")
    print(f"[dl] Split:       {args.split}")
    print(f"[dl] Max clips:   {args.n_clips}")
    print("=" * 50)

    # ── 1. Load dataset (streaming) ──────────────────────────────────────────
    ds = load_ssv2_hf(args.split, args.hf_cache, args.hf_token)

    # ── 2. Build label map ───────────────────────────────────────────────────
    if labels_path.exists():
        print(f"[dl] Labels already exist: {labels_path}")
        with open(labels_path) as f:
            label_map = json.load(f)
    else:
        # Probe a fresh iterator for label map
        ds_probe = load_ssv2_hf(args.split, args.hf_cache, args.hf_token)
        label_map = build_labels_map(ds_probe, n_probe=min(5000, args.n_clips * 2))
        with open(labels_path, "w") as f:
            json.dump(label_map, f, indent=2, sort_keys=True)
        print(f"[dl] Saved labels to {labels_path}")

    # ── 3. Download video clips ───────────────────────────────────────────────
    # Check which clips we already have
    existing = {p.stem for p in video_dir.glob("*.mp4")}
    print(f"[dl] Already downloaded: {len(existing)} videos")

    annotations = []
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0

    pbar = tqdm(total=args.n_clips, desc="Downloading SSv2 clips", unit="clip")

    import random
    rng = random.Random(args.seed)

    for sample in ds:
        if n_downloaded + len(existing) >= args.n_clips:
            break

        video_id = str(sample.get("video_id", sample.get("id", "")))
        text = sample.get("text", sample.get("label", ""))
        label_int = sample.get("label", label_map.get(text, -1))

        if not video_id or label_int < 0:
            n_failed += 1
            continue

        # Record annotation regardless of whether we need to download
        annotations.append({"id": video_id, "template": text})

        # Check if already downloaded
        video_path = video_dir / f"{video_id}.mp4"
        if video_path.exists():
            n_skipped += 1
            pbar.update(1)
            continue

        # Get raw video bytes / path from HuggingFace
        video_data = sample.get("video", None)
        if video_data is None:
            n_failed += 1
            continue

        try:
            # HF datasets returns video as a dict with 'path' or 'bytes'
            if isinstance(video_data, dict):
                if "path" in video_data and video_data["path"]:
                    shutil.copy2(video_data["path"], video_path)
                elif "bytes" in video_data and video_data["bytes"]:
                    with open(video_path, "wb") as f:
                        f.write(video_data["bytes"])
                else:
                    n_failed += 1
                    continue
            elif isinstance(video_data, bytes):
                with open(video_path, "wb") as f:
                    f.write(video_data)
            elif hasattr(video_data, "read"):
                with open(video_path, "wb") as f:
                    f.write(video_data.read())
            else:
                # Last resort: try to get path attribute
                src = getattr(video_data, "path", None) or str(video_data)
                if os.path.exists(src):
                    shutil.copy2(src, video_path)
                else:
                    n_failed += 1
                    continue

            n_downloaded += 1
            pbar.update(1)

        except Exception as e:
            print(f"\n[dl] WARNING: Failed to save video {video_id}: {e}")
            n_failed += 1
            continue

    pbar.close()

    # ── 4. Save annotations ───────────────────────────────────────────────────
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    n_videos = len(list(video_dir.glob("*.mp4")))
    print()
    print("=" * 50)
    print("[dl] SSv2 Download Complete")
    print(f"[dl] Videos in {video_dir}: {n_videos}")
    print(f"[dl] Downloaded this run:   {n_downloaded}")
    print(f"[dl] Skipped (existing):    {n_skipped}")
    print(f"[dl] Failed:                {n_failed}")
    print(f"[dl] Labels:   {labels_path}")
    print(f"[dl] Annotations: {annotations_path}")
    print()
    if n_videos >= args.n_clips * 0.9:
        print("[dl] ✓ Sufficient videos downloaded.")
        print("[dl] Next: sbatch scripts/preextract_ssv2.sh")
    else:
        print(f"[dl] ✗ Only {n_videos}/{args.n_clips} videos downloaded.")
        print("[dl]   Re-run this script to resume from where it left off.")
    print("=" * 50)


if __name__ == "__main__":
    main()
