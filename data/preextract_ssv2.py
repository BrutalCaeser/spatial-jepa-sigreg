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
        """Load V-JEPA 2.1 ViT-L encoder from checkpoint.

        Uses vjepa2_1_vit_large_384 (ViT-L, embed_dim=1024) with img_size=224
        so output is 196 patch tokens at 1024-dim (matching build_spec D_in=1024, N=196).

        RoPE interpolation (interpolate_rope=True in backbones.py) allows 224px
        inference from a model trained at 384px.

        checkpoint_key = "ema_encoder" (EMA-averaged teacher weights, highest quality).

        Checkpoint: vjepa2_1_vitl_dist_vitG_384.pt (~1.2 GB)
        URL: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
        """
        import sys
        # Add vjepa2 repo to path (installed via pip install -e or sys.path)
        vjepa2_repo = os.path.expanduser("~/vjepa2") if not os.path.exists(
            "/scratch/gupta.yashv/vjepa2") else "/scratch/gupta.yashv/vjepa2"
        if os.path.exists(vjepa2_repo):
            sys.path.insert(0, vjepa2_repo)

        try:
            from app.vjepa_2_1.models import vision_transformer as vit_encoder_mod
            # V-JEPA 2.1 ViT-L: embed_dim=1024, with img_size=224 → 196 spatial patches
            encoder = vit_encoder_mod.vit_large(
                patch_size=16,
                img_size=(224, 224),
                num_frames=1,
                tubelet_size=1,
                use_sdpa=True,
                use_SiLU=False,
                wide_SiLU=True,
                uniform_power=False,
                use_rope=True,
                img_temporal_dim_size=1,
                interpolate_rope=True,
            )
        except (ImportError, Exception) as e:
            raise ImportError(
                f"V-JEPA 2.1 package not found: {e}\n"
                "Clone: git clone https://github.com/facebookresearch/vjepa2.git\n"
                "Install: pip install -e /scratch/$USER/vjepa2 --no-deps"
            )

        # Load checkpoint — key is 'ema_encoder' for V-JEPA 2.1 ViT-L
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key in ("ema_encoder", "target_encoder", "encoder", "model"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint  # checkpoint IS the state dict

        # Strip DataParallel and backbone prefixes.
        state_dict = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state_dict.items()
        }

        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        n_missing = len([k for k in missing if "pos_embed" not in k])
        if n_missing > 5:
            print(f"[WARNING] {n_missing} non-pos_embed keys missing from checkpoint.")
        print(f"[extractor] Loaded V-JEPA 2.1 ViT-L: "
              f"embed_dim={encoder.embed_dim}, img_size=224, N=196")
        return encoder.to(device)

    @torch.no_grad()
    def extract_patches(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from a single frame (224×224).

        V-JEPA 2.1 ViT-L with img_size=224, patch_size=16, num_frames=1
        produces exactly 196 patch tokens at 1024-dim.

        The model returns a list of layer outputs (hierarchical_layers).
        We take the LAST element which is the final encoder output.

        Args:
            frame: Preprocessed frame tensor, shape [1, 1, C, H, W] (video format)
                   or [1, C, H, W] (image format). Will be reshaped as needed.

        Returns:
            Patch features, shape [196, 1024].
        """
        # V-JEPA 2.1 expects video input: [B, C, T, H, W] or [B, T, C, H, W]
        # For single-frame: unsqueeze time dim if needed
        if frame.dim() == 4:
            # [1, C, H, W] → [1, C, 1, H, W]
            frame = frame.unsqueeze(2)

        output = self.encoder(frame)

        # V-JEPA 2.1 returns a list from hierarchical_layers (last = final output)
        if isinstance(output, (list, tuple)):
            tokens = output[-1]   # [1, N, D]
        else:
            tokens = output       # [1, N, D]

        # tokens shape: [1, N, D] where N=196, D=1024
        # V-JEPA 2.1 ViT-L does NOT prepend CLS token; pure patch tokens only.
        # Strip CLS if accidentally present (N+1 case).
        N = tokens.shape[1]
        if N == 197:
            tokens = tokens[:, 1:, :]   # strip CLS → [1, 196, 1024]
        elif N != 196:
            raise ValueError(
                f"Unexpected token count {N} from V-JEPA 2.1 ViT-L (224px). "
                f"Expected 196. Check img_size and patch_size."
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
# Video frame loaders
# ---------------------------------------------------------------------------

def load_two_consecutive_frames(
    video_path: str,
    transform: T.Compose,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load two random consecutive frames from a video file (.webm/.mp4).

    Args:
        video_path: Path to video file.
        transform:  Image preprocessing transform.

    Returns:
        (frame_c, frame_t) tensors each [1, C, H, W], or None on failure.
    """
    try:
        import torchvision.io as tvio
        video, _, _ = tvio.read_video(video_path, output_format="TCHW", pts_unit="sec")
        T_total = video.shape[0]
        if T_total < 2:
            return None
        t = random.randint(0, T_total - 2)
        frame_c = video[t].float() / 255.0
        frame_t = video[t + 1].float() / 255.0
        from PIL import Image
        img_c = Image.fromarray(frame_c.permute(1, 2, 0).numpy().astype("uint8"))
        img_t = Image.fromarray(frame_t.permute(1, 2, 0).numpy().astype("uint8"))
        return transform(img_c).unsqueeze(0), transform(img_t).unsqueeze(0)
    except Exception:
        return None


def load_two_consecutive_frames_from_dir(
    frame_dir: Path,
    n_frames: int,
    transform: T.Compose,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load two random consecutive JPEG frames from a VideoFolder directory.

    Explorer HPC SSv2 format: each video is a directory of JPEG frames named
    000001.jpg, 000002.jpg, ... (6-digit zero-padded, 1-indexed).

    Args:
        frame_dir: Directory containing {NNNNNN}.jpg frame files.
        n_frames:  Total frame count as declared in the annotation file.
        transform: Image preprocessing transform (resize + normalize).

    Returns:
        (frame_c, frame_t) tensors each [1, C, H, W], or None on failure.
    """
    try:
        from PIL import Image
        if n_frames < 2:
            return None
        # Sample random consecutive pair (1-indexed frames)
        t = random.randint(1, n_frames - 1)  # t and t+1, both valid
        fc_path = frame_dir / f"{t:06d}.jpg"
        ft_path = frame_dir / f"{t + 1:06d}.jpg"
        if not fc_path.exists() or not ft_path.exists():
            # Fall back: list actual files and pick pair
            jpgs = sorted(frame_dir.glob("*.jpg"))
            if len(jpgs) < 2:
                return None
            idx = random.randint(0, len(jpgs) - 2)
            fc_path, ft_path = jpgs[idx], jpgs[idx + 1]
        img_c = Image.open(fc_path).convert("RGB")
        img_t = Image.open(ft_path).convert("RGB")
        return transform(img_c).unsqueeze(0), transform(img_t).unsqueeze(0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Annotation parsers (JSON and VideoFolder formats)
# ---------------------------------------------------------------------------

def load_annotations_json(
    label_file: str,
    annotation_file: str,
) -> Tuple[Dict[str, int], List[Dict]]:
    """Load SSv2 annotations from the standard JSON format.

    label_file:      {"action template": label_int, ...}
    annotation_file: [{"id": str, "template": str, ...}, ...]
    """
    with open(label_file) as f:
        label_map: Dict[str, int] = json.load(f)
    with open(annotation_file) as f:
        annotations: List[Dict] = json.load(f)
    return label_map, annotations


def load_annotations_videofolder(
    base_dir: str,
    annotation_file: str,
) -> Tuple[Dict[str, int], List[Dict]]:
    """Load SSv2 annotations from the VideoFolder format used on Explorer HPC.

    Explorer HPC path: /datasets/something_v2/
      category.txt          — one class name per line (index = line number, 0-based)
      train_videofolder.txt — "{rel_path} {n_frames} {label_idx}" per line
      val_videofolder.txt   — same format

    Returns:
        label_map:   {"class name": label_int, ...}  (174 entries)
        annotations: [{"id": str, "n_frames": int, "label": int}, ...]
    """
    base = Path(base_dir)
    category_path = base / "category.txt"

    # Build label map from category.txt (0-indexed, one class per line)
    with open(category_path) as f:
        classes = [line.strip() for line in f if line.strip()]
    label_map = {name: idx for idx, name in enumerate(classes)}

    # Parse annotation file: "{rel_path} {n_frames} {label_idx}"
    annotations = []
    with open(annotation_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rel_path, n_frames_str, label_str = parts[0], parts[1], parts[2]
            clip_id = rel_path.rstrip("/").split("/")[-1]  # e.g. "78687"
            annotations.append({
                "id": clip_id,
                "rel_path": rel_path,
                "n_frames": int(n_frames_str),
                "label": int(label_str),
            })

    return label_map, annotations


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-extract V-JEPA 2.1 features for SSv2")
    parser.add_argument("--video_dir", required=True,
                        help="Directory with SSv2 video clips (VideoFolder: parent of 20bn-something-something-v2/)")
    parser.add_argument("--label_file", required=True,
                        help="SSv2 labels: JSON file (json format) or base_dir containing category.txt (videofolder)")
    parser.add_argument("--annotation_file", required=True,
                        help="SSv2 annotations: JSON file or train_videofolder.txt")
    parser.add_argument("--output_dir", required=True, help="Output directory for features")
    parser.add_argument("--checkpoint", required=True, help="V-JEPA 2.1 ViT-L checkpoint (.pt)")
    parser.add_argument("--n_clips", type=int, default=20000, help="Max clips to extract")
    parser.add_argument("--image_size", type=int, default=224, help="Frame resize target")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Val fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument(
        "--annotation_format",
        default="json",
        choices=["json", "videofolder"],
        help=(
            "Annotation format: 'json' (standard SSv2 JSON files) or "
            "'videofolder' (Explorer HPC: category.txt + train_videofolder.txt)"
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[preextract] Using device: {device}")
    print(f"[preextract] Annotation format: {args.annotation_format}")

    # ── Load annotations (format-dispatched) ─────────────────────────────────
    if args.annotation_format == "videofolder":
        # Explorer HPC: label_file is the dataset base dir (contains category.txt)
        label_map, annotations = load_annotations_videofolder(
            base_dir=args.label_file,
            annotation_file=args.annotation_file,
        )
        print(f"[preextract] VideoFolder: {len(label_map)} classes, "
              f"{len(annotations)} clips in annotation file.")
    else:
        # Standard SSv2 JSON format
        label_map, annotations = load_annotations_json(
            label_file=args.label_file,
            annotation_file=args.annotation_file,
        )
        print(f"[preextract] JSON: {len(label_map)} classes, "
              f"{len(annotations)} clips in annotation file.")

    # Shuffle and limit.
    random.shuffle(annotations)
    annotations = annotations[:args.n_clips]
    print(f"[preextract] Extracting up to {len(annotations)} clips.")

    # Load V-JEPA 2.1 extractor.
    print(f"[preextract] Loading V-JEPA 2.1 from {args.checkpoint}...")
    extractor = VJEPAFeatureExtractor(args.checkpoint, device)
    transform = build_transform(args.image_size)
    print("[preextract] Encoder loaded and frozen.")

    # Extract features.
    success_clips = []
    failed = 0

    for ann in tqdm(annotations, desc="Extracting features"):
        clip_id = str(ann["id"])

        # ── Resolve label ─────────────────────────────────────────────────────
        if args.annotation_format == "videofolder":
            # VideoFolder annotations already have integer label index
            label = ann["label"]
        else:
            # JSON format: label stored as action text, look up in label_map
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

        # ── Load two consecutive frames (format-dispatched) ───────────────────
        if args.annotation_format == "videofolder":
            # VideoFolder: each clip is a directory of JPEG frames
            # video_dir is the parent; rel_path is e.g. "20bn-something-something-v2/78687"
            rel_path = ann.get("rel_path", f"20bn-something-something-v2/{clip_id}")
            frame_dir = Path(args.video_dir) / rel_path
            if not frame_dir.is_dir():
                # Fallback: try video_dir/clip_id directly
                frame_dir = Path(args.video_dir) / clip_id
            if not frame_dir.is_dir():
                failed += 1
                continue
            frames = load_two_consecutive_frames_from_dir(
                frame_dir=frame_dir,
                n_frames=ann.get("n_frames", 0),
                transform=transform,
            )
        else:
            # JSON format: clip is a video file (.webm/.mp4/.avi)
            video_path = None
            for ext in [".webm", ".mp4", ".avi"]:
                candidate = Path(args.video_dir) / f"{clip_id}{ext}"
                if candidate.exists():
                    video_path = str(candidate)
                    break
            if video_path is None:
                failed += 1
                continue
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
