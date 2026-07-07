"""
BraTS-PED 2026 — inference and submission zip generator.

Runs sliding-window inference (+ optional 4-flip TTA) on every subject in
data/BraTS26_PED_validation, converts predictions to BraTS label format,
resamples back to the original image space, and writes a submission.zip.

Label convention (BraTS-PEDs 2026 — RAPNO 4-label):
  0 → background
  1 → ET  (enhancing tumor)
  2 → NET (nonenhancing tumor)
  3 → CC  (cystic component)
  4 → ED  (peritumoral edema)

Output filenames inside the zip: {subject_id}-seg.nii.gz

Usage:
    python predict.py --ckpt checkpoints/last.pt
    python predict.py --ckpt checkpoints/epoch_0300.pt --no_tta
    python predict.py --ckpt checkpoints/last.pt --sw_overlap 0.75 --out my_submission.zip
    python predict.py --ckpt_dir checkpoints          # ensemble fold*_best.pt (cross-val)
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path
from typing import List

import nibabel as nib
import nibabel.processing
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from data_loader import MODALITY_SUFFIXES, inference_transforms
from models import IN_CHANNELS, MedSwinNet, OUT_CHANNELS, PATCH_SIZE, remove_small_components

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cuda.matmul.allow_tf32 = True


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("brats_predict")


# ─────────────────────────────────────────────────────────────────────────────
# Subject discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_subjects(val_dir: Path) -> List[dict]:
    subjects = sorted(
        p for p in val_dir.iterdir()
        if p.is_dir() and p.name.startswith("BraTS-PED-")
    )
    samples = []
    for subj in subjects:
        sid   = subj.name
        paths = [str(subj / f"{sid}-{m}.nii.gz") for m in MODALITY_SUFFIXES]
        if all(Path(p).exists() for p in paths):
            samples.append({"image": paths, "subject_id": sid})
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_one(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    log = logging.getLogger("brats_predict")
    # deep_supervision=True must match training; forward_inference returns full-res only.
    # use_checkpoint=True is safe: the flag is a no-op in model.eval() mode.
    model = MedSwinNet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        deep_supervision=True,
        use_checkpoint=True,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw   = state.get("model", state)
    # strip torch.compile prefix if present
    clean = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}
    model.load_state_dict(clean)
    model.eval()

    epoch = state.get("epoch", "?")
    loss  = state.get("loss",  None)
    info  = f"epoch={epoch}" + (f"  loss={loss:.4f}" if loss is not None else "")
    log.info(f"  Loaded {ckpt_path.name}  {info}")
    return model


def load_models(args: argparse.Namespace, device: torch.device) -> List[torch.nn.Module]:
    """Resolve which checkpoint(s) to use and return eval-mode models."""
    if args.ckpt:
        return [_load_one(Path(args.ckpt), device)]

    ckpt_dir = Path(args.ckpt_dir)

    if args.fold is not None:
        candidates = [ckpt_dir / f"fold{args.fold}_best.pt"]
    else:
        # Priority: fold ensemble → epoch snapshots → last.pt
        candidates = sorted(ckpt_dir.glob("fold*_best.pt"))
        if not candidates:
            candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
        if not candidates:
            last = ckpt_dir / "last.pt"
            if last.exists():
                candidates = [last]

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir}. "
            "Use --ckpt <path> to point to a specific file."
        )
    return [_load_one(p, device) for p in candidates]


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict_volume(
    models:        List[torch.nn.Module],
    image:         torch.Tensor,      # (1, 4, H, W, D) on CPU or GPU
    device:        torch.device,
    sw_batch_size: int   = 4,
    sw_overlap:    float = 0.75,
    use_tta:       bool  = True,
) -> np.ndarray:
    """
    Ensemble + optional 4-flip TTA → BraTS-PED uint8 label map (H, W, D).
    """
    image = image.to(device)
    # 4-flip TTA: identity + one flip per spatial axis (drops the 4 multi-axis
    # combos of the full 8-flip set to roughly halve runtime).
    flip_sets = (
        [[], [2], [3], [4]]
        if use_tta else [[]]
    )

    all_probs: List[torch.Tensor] = []
    for model in models:
        for axes in flip_sets:
            inp = torch.flip(image, axes) if axes else image
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = sliding_window_inference(
                    inputs=inp,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=sw_batch_size,
                    predictor=model.forward_inference,
                    overlap=sw_overlap,
                    mode="gaussian",
                )
            probs = torch.sigmoid(logits.float())
            if axes:
                probs = torch.flip(probs, axes)
            all_probs.append(probs.cpu())

    mean_probs = torch.stack(all_probs).mean(0)          # (1, 4, H, W, D)

    p   = mean_probs[0].numpy()                            # (4, H, W, D) — [ET, NET, CC, ED]
    et  = p[0] >= 0.5
    net = p[1] >= 0.5
    cc  = p[2] >= 0.5
    ed  = p[3] >= 0.4    # slightly lower: edema is diffuse, recall > precision

    # Remove small spurious ET connected components (BraTS-style clean-up).
    # Done on the already-thresholded boolean mask so it doesn't clobber the
    # raw probabilities the other channels' thresholds above depend on.
    et = remove_small_components(et, min_voxels=10)

    # Priority ET > NET > CC > ED (higher-priority class overwrites on overlap)
    seg       = np.zeros(et.shape, dtype=np.uint8)
    seg[ed]   = 4    # ED  — peritumoral edema
    seg[cc]   = 3    # CC  — cystic component
    seg[net]  = 2    # NET — nonenhancing tumor
    seg[et]   = 1    # ET  — enhancing tumor

    return seg


# ─────────────────────────────────────────────────────────────────────────────
# Space inversion — preprocessed 1-mm RAS → original subject space
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_original(
    seg:                 np.ndarray,
    preprocessed_affine: np.ndarray,
    orig_nib:            nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Nearest-neighbour resample from preprocessed space → original voxel grid.

    The output uses orig_nib's affine and header *exactly* so that array
    dimensions, voxel spacing, origin, and orientation are identical to the
    source image — required by BraTS 2026 validation.
    """
    pred_nib  = nib.Nifti1Image(seg, affine=preprocessed_affine)
    resampled = nibabel.processing.resample_from_to(pred_nib, orig_nib, order=0, cval=0)
    seg_out   = np.asarray(resampled.dataobj, dtype=np.uint8)

    # Force exact shape — resample_from_to should match, but guard against rounding
    orig_shape = tuple(orig_nib.header.get_data_shape()[:3])
    if seg_out.shape != orig_shape:
        padded = np.zeros(orig_shape, dtype=np.uint8)
        slices = tuple(slice(0, min(s, o)) for s, o in zip(seg_out.shape, orig_shape))
        padded[slices] = seg_out[slices]
        seg_out = padded

    # Use the original header verbatim so spacing/origin/orientation are identical
    out_header = orig_nib.header.copy()
    out_header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(seg_out, orig_nib.affine, out_header)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    log    = _setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        f"Device: {device}"
        + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    val_dir = Path(args.val_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(val_dir)
    if not subjects:
        raise RuntimeError(f"No BraTS-PED subjects found in {val_dir}")
    log.info(f"Subjects: {len(subjects)}")

    models = load_models(args, device)
    log.info(
        f"Ensemble: {len(models)} model(s)"
        f"  TTA: {'ON (4 flips)' if args.use_tta else 'OFF'}"
        f"  sw_overlap: {args.sw_overlap}"
    )

    transforms = inference_transforms()

    pbar = tqdm(subjects, unit="subj", desc="Predicting")
    for subj in pbar:
        sid = subj["subject_id"]
        pbar.set_postfix(sid=sid[-12:])

        # Load original t1c as spatial reference (affine + header for final output)
        orig_nib = nib.load(subj["image"][0])

        # Preprocess — MONAI MetaTensor tracks the updated affine through transforms
        sample              = transforms({"image": subj["image"]})
        image               = sample["image"]                        # MetaTensor (4, H, W, D)
        preprocessed_affine = image.meta["affine"].numpy()           # (4, 4) in 1-mm RAS
        image               = image.unsqueeze(0)                     # (1, 4, H, W, D)

        seg = predict_volume(
            models, image, device,
            sw_batch_size=args.sw_batch_size,
            sw_overlap=args.sw_overlap,
            use_tta=args.use_tta,
        )

        out_nib  = resample_to_original(seg, preprocessed_affine, orig_nib)
        out_path = out_dir / f"{sid}.nii.gz"
        nib.save(out_nib, out_path)

        unique = np.unique(np.asarray(out_nib.dataobj))
        log.info(f"  {sid}  labels present: {unique.tolist()}")

    pbar.close()

    # ── Package submission zip ────────────────────────────────────────────────
    zip_path = Path(args.out)
    seg_files = sorted(out_dir.glob("*.nii.gz"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in seg_files:
            zf.write(f, f.name)   # flat structure inside zip

    size_mb = zip_path.stat().st_size / 1e6
    log.info(f"\nSubmission zip: {zip_path}  ({len(seg_files)} files, {size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS-PED 2026 — inference & submission zip")
    p.add_argument("--ckpt",          default="checkpoints/best.pt",
                   help="Single checkpoint file (e.g. checkpoints/last.pt). "
                        "Takes priority over --ckpt_dir.")
    p.add_argument("--ckpt_dir",      default="checkpoints",
                   help="Directory searched for fold*_best.pt → epoch_*.pt → last.pt")
    p.add_argument("--fold",          type=int, default=None,
                   help="Use only fold N checkpoint (ignored when --ckpt is set)")
    p.add_argument("--val_dir",       default="data/BraTS26_PED_validation",
                   help="Validation subjects directory")
    p.add_argument("--out_dir",       default="predictions",
                   help="Directory for per-subject NIfTI files")
    p.add_argument("--out",           default="submission.zip",
                   help="Output zip file path")
    p.add_argument("--sw_batch_size", type=int,   default=8,
                   help="Sliding-window patch batch size (bf16 autocast + no-grad "
                        "inference allows a larger batch than training)")
    p.add_argument("--sw_overlap",    type=float, default=0.75,
                   help="Sliding-window overlap (0.75 recommended for submission)")
    p.add_argument("--no_tta",        action="store_true",
                   help="Disable 4-flip TTA (faster but ~1-2%% lower Dice)")
    args         = p.parse_args()
    args.use_tta = not args.no_tta
    return args


if __name__ == "__main__":
    run(parse_args())
