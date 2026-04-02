"""
Pre-extraction script: V-JEPA 2.1 ViT-G patch features from SSv2 video clips.

Run once before any training:
    python data/preextract_ssv2.py \
        --video_dir /path/to/ssv2_part00 \
        --label_file /path/to/something-something-v2-labels.json \
        --annotation_file /path/to/something-something-v2-train.json \
        --output_dir /scratch/$USER/data/ssv2_vjepa21_features \
        --checkpoint /path/to/vjepa21_vitg.pt \
        --n_clips 20000

Output structure (build_spec.md Section 4.1):
    output_dir/
        index.json              <- split index: {train: [...], val: [...], test: [...]}
        {clip_id}_fc.pt         <- context frame features [196, 1024]
        {clip_id}_ft.pt         <- target frame features [196, 1024]

Storage estimate: 20K clips × 2 frames × 196 × 1024 × 4 bytes ≈ 30 GB

CRITICAL:
    - Extract PATCH tokens [196, 1024], NOT the CLS token (index 0).
    - This must be V-JEPA 2.1 ViT-G, NOT V-JEPA 2.
    - Verify: raw_erank > 20, raw_ncorr > 0.3 (FM5 detection).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# V-JEPA 2.1 ViT-G feature extractor (frozen)
# ---------------------------------------------------------------------------

class VJEPAFeatureExtractor(nn.Module):
    """Wrapper around V-JEPA 2.1 ViT-G that extracts patch tokens.

    The V-JEPA 2.1 ViT-G outputs token_list where index 0 is the CLS token
    and indices 1..196 are patch tokens. We extract the patch tokens only.

    Args:
        checkpoint_path: Path to V-JEPA 2.1 ViT-G checkpoint (.pt).
        device: Device to run extraction on.
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        self.device = device
        self.encoder = self._load_encoder(checkpoint_path, device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def _load_encoder(self, checkpoint_path: str, device: torch.device) -> nn.Module:
        """Load V-JEPA 2.1 encoder from checkpoint.

        V-JEPA 2.1 checkpoints follow Meta's public release format.
        Adjust key loading logic if using a different checkpoint format.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # V-JEPA 2.1 checkpoint may store encoder under 'encoder' or 'target_encoder'.
        if "encoder" in checkpoint:
            state_dict = checkpoint["encoder"]
        elif "target_encoder" in checkpoint:
            state_dict = checkpoint["target_encoder"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix from DataParallel if present.
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Import V-JEPA 2.1 model (assumes vjepa package is installed).
        try:
            from vjepa.models.vision_transformer import vit_giant
            encoder = vit_giant()
        except ImportError:
            raise ImportError(
                "V-JEPA 2.1 package not found. Clone from Meta's public repo and "
                "install: pip install -e /path/to/vjepa2"
            )

        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARNING] Missing keys in V-JEPA 2.1 checkpoint: {missing[:5]}...")
        return encoder.to(device)

    @torch.no_grad()
    def extract_patches(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from a single frame.

        Args:
            frame: Preprocessed frame tensor, shape [1, C, H, W].

        Returns:
            Patch features, shape [196, 1024].
            CLS token (index 0) is excluded.
        """
        output = self.encoder(frame)
        # V-JEPA 2.1 ViT-G returns patch tokens: [1, N+1, D] or [1, N, D]
        if isinstance(output, (list, tuple)):
            tokens = output[-1]   # take last layer output
        else:
            tokens = output       # [1, N+1, D] or [1, N, D]

        # If CLS token is present (N+1=197), strip it.
        if tokens.shape[1] == 197:
            tokens = tokens[:, 1:, :]   # [1, 196, D]
        elif tokens.shape[1] == 196:
            pass
        else:
            raise ValueError(
                f"Unexpected token count {tokens.shape[1]} from V-JEPA 2.1. "
                f"Expected 196 (no CLS) or 197 (with CLS)."
            )

        return tokens.squeeze(0).cpu()  # [196, 1024]


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def build_transform(image_size: int = 224) -> T.Compose:
    """Standard V-JEPA 2.1 preprocessing for a single frame."""
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Video frame loader
# ---------------------------------------------------------------------------

def load_two_consecutive_frames(
    video_path: str,
    transform: T.Compose,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load two random consecutive frames from a video.

    Args:
        video_path: Path to video file (.webm or .mp4).
        transform:  Image preprocessing transform.

    Returns:
        (frame_c, frame_t) tensors each [1, C, H, W], or None on failure.
    """
    try:
        import torchvision.io as tvio
        # Read video as [T, H, W, C] uint8 tensor.
        video, _, _ = tvio.read_video(video_path, output_format="TCHW", pts_unit="sec")
        # video: [T, C, H, W]  uint8
        T_total = video.shape[0]
        if T_total < 2:
            return None

        # Sample a random consecutive pair.
        t = random.randint(0, T_total - 2)
        frame_c = video[t].float() / 255.0     # [C, H, W]
        frame_t = video[t + 1].float() / 255.0  # [C, H, W]

        # Apply preprocessing (resize + normalize).
        # Transform expects PIL or tensor [C, H, W] in [0,1].
        import torchvision.transforms.functional as TF
        from PIL import Image
        import numpy as np

        img_c = Image.fromarray(frame_c.permute(1, 2, 0).numpy().astype("uint8"))
        img_t = Image.fromarray(frame_t.permute(1, 2, 0).numpy().astype("uint8"))

        fc = transform(img_c).unsqueeze(0)  # [1, C, 224, 224]
        ft = transform(img_t).unsqueeze(0)  # [1, C, 224, 224]

        return fc, ft

    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-extract V-JEPA 2.1 features for SSv2")
    parser.add_argument("--video_dir", required=True, help="Directory with SSv2 video clips")
    parser.add_argument("--label_file", required=True, help="SSv2 labels JSON file")
    parser.add_argument("--annotation_file", required=True, help="SSv2 train annotation JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory for features")
    parser.add_argument("--checkpoint", required=True, help="V-JEPA 2.1 ViT-G checkpoint (.pt)")
    parser.add_argument("--n_clips", type=int, default=20000, help="Max clips to extract")
    parser.add_argument("--image_size", type=int, default=224, help="Frame resize target")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Val fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[preextract] Using device: {device}")

    # Load label mapping.
    with open(args.label_file) as f:
        label_map: Dict[str, int] = json.load(f)
    # label_map: {"action text": index, ...}

    # Load annotations.
    with open(args.annotation_file) as f:
        annotations: List[dict] = json.load(f)
    # annotations: [{"id": str, "template": str, "label": str, ...}, ...]

    print(f"[preextract] Found {len(annotations)} clips in annotation file.")

    # Shuffle and limit.
    random.shuffle(annotations)
    annotations = annotations[:args.n_clips]

    # Load V-JEPA 2.1 extractor.
    print(f"[preextract] Loading V-JEPA 2.1 from {args.checkpoint}...")
    extractor = VJEPAFeatureExtractor(args.checkpoint, device)
    transform = build_transform(args.image_size)
    print("[preextract] Encoder loaded and frozen.")

    # Extract features.
    success_clips = []
    failed = 0

    for ann in tqdm(annotations, desc="Extracting features"):
        clip_id = ann["id"]
        label_text = ann.get("template", ann.get("label", ""))
        label = label_map.get(label_text, -1)
        if label < 0:
            failed += 1
            continue

        # Skip if already extracted.
        fc_path = output_dir / f"{clip_id}_fc.pt"
        ft_path = output_dir / f"{clip_id}_ft.pt"
        if fc_path.exists() and ft_path.exists():
            success_clips.append({"clip_id": clip_id, "label": label})
            continue

        # Find video file.
        video_path = None
        for ext in [".webm", ".mp4", ".avi"]:
            candidate = Path(args.video_dir) / f"{clip_id}{ext}"
            if candidate.exists():
                video_path = str(candidate)
                break
        if video_path is None:
            failed += 1
            continue

        # Load two consecutive frames.
        frames = load_two_consecutive_frames(video_path, transform)
        if frames is None:
            failed += 1
            continue
        frame_c, frame_t = frames

        # Extract patch features.
        frame_c = frame_c.to(device)
        frame_t = frame_t.to(device)

        try:
            fc_feat = extractor.extract_patches(frame_c)   # [196, 1024]
            ft_feat = extractor.extract_patches(frame_t)   # [196, 1024]
        except Exception as e:
            print(f"[preextract] ERROR for clip {clip_id}: {e}")
            failed += 1
            continue

        # Save.
        torch.save(fc_feat, fc_path)
        torch.save(ft_feat, ft_path)
        success_clips.append({"clip_id": clip_id, "label": label})

    print(f"[preextract] Extracted {len(success_clips)} clips. Failed: {failed}.")

    # Create train/val/test splits.
    random.shuffle(success_clips)
    n_total = len(success_clips)
    n_train = int(n_total * args.train_frac)
    n_val   = int(n_total * args.val_frac)
    n_test  = n_total - n_train - n_val

    index = {
        "train": success_clips[:n_train],
        "val":   success_clips[n_train:n_train + n_val],
        "test":  success_clips[n_train + n_val:],
    }
    print(f"[preextract] Split: train={n_train}, val={n_val}, test={n_test}")

    # Save index.
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[preextract] Saved index to {index_path}")

    # Verify baseline feature quality (FM5 check).
    print("[preextract] Verifying baseline feature quality...")
    _verify_baseline(output_dir, success_clips[:32])


def _verify_baseline(output_dir: Path, sample_clips: List[dict]) -> None:
    """Verify that V-JEPA 2.1 features have spatial structure (FM5 check).

    Required: raw_erank > 20, raw_ncorr > 0.3 (build_spec.md Section 4.3).
    If these fail, the experiment cannot proceed (V-JEPA version wrong or
    using wrong token type).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.metrics import effective_rank, neighbor_corr

    features = []
    for clip in sample_clips:
        clip_id = clip["clip_id"]
        fc_path = output_dir / f"{clip_id}_fc.pt"
        if fc_path.exists():
            f = torch.load(fc_path, map_location="cpu")  # [196, 1024]
            features.append(f)

    if not features:
        print("[verify] No features found for baseline check.")
        return

    f_all = torch.stack(features)  # [K, 196, 1024]

    raw_erank = effective_rank(f_all.reshape(-1, 1024))
    raw_ncorr = neighbor_corr(f_all)

    print(f"[verify] raw_erank = {raw_erank:.2f}  (required > 20)")
    print(f"[verify] raw_ncorr = {raw_ncorr:.3f} (required > 0.30)")

    if raw_erank < 20:
        print("[verify] FAIL: raw_erank < 20. Check V-JEPA 2.1 version and token type.")
    elif raw_ncorr < 0.3:
        print("[verify] FAIL: raw_ncorr < 0.3. Features may lack spatial structure.")
    else:
        print("[verify] PASS: Features are high-quality V-JEPA 2.1 patch tokens.")


if __name__ == "__main__":
    main()
