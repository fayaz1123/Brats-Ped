"""
BraTS-PED 2025 data loader with MONAI-based augmentation pipeline.

Scans data/BraTS-PEDs_Batch2_Release and data/BraTS26_PED_training (if present).
Modalities loaded as 4-channel input: t1c, t1n, t2f, t2w.

BraTS-PEDs 2026 label convention — RAPNO 4-label (NOT skull-stripped, only defaced):
  0 → background
  1 → ET  (Enhancing Tumor)
  2 → NET (Nonenhancing Tumor)
  3 → CC  (Cystic Component)
  4 → ED  (Peritumoral Edema)

Output: 4-channel binary masks in order [ET, NET, CC, ED]:
  ET  = label 1             (channel 0) — enhancing tumor
  NET = label 2             (channel 1) — nonenhancing tumor
  CC  = label 3             (channel 2) — cystic component
  ED  = label 4             (channel 3) — peritumoral edema

Derived for submission:
  TC = ET | NET | CC        = labels 1|2|3
  WT = ET | NET | CC | ED   = labels 1|2|3|4
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from monai.data import CacheDataset, list_data_collate
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    SpatialPadd,
    Spacingd,
    LoadImaged,
)


# ---------------------------------------------------------------------------
# Label conversion — BraTS-PED 2025 specific
# ---------------------------------------------------------------------------

class ConvertBratsPed2025Labelsd(MapTransform):
    """
    Convert scalar BraTS-PEDs 2026 (RAPNO 4-label) label map to 4 binary channels:
      channel 0 → ET  (label 1)   — enhancing tumor
      channel 1 → NET (label 2)   — nonenhancing tumor
      channel 2 → CC  (label 3)   — cystic component
      channel 3 → ED  (label 4)   — peritumoral edema

    Each sub-region is a direct prediction target so every channel gets its own
    Dice loss. TC and WT are derived at inference exactly as the challenge
    defines them:
      TC = ET | NET | CC      = labels 1|2|3
      WT = ET | NET | CC | ED = labels 1|2|3|4

    BraTS-PEDs 2026 label convention (4 labels):
      1 → ET  (Enhancing Tumor)
      2 → NET (Nonenhancing Tumor)
      3 → CC  (Cystic Component)
      4 → ED  (Peritumoral Edema)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            # squeeze channel dim added by EnsureChannelFirstd
            if lbl.ndim == 4 and lbl.shape[0] == 1:
                lbl = lbl.squeeze(0)
            et  = lbl == 1
            net = lbl == 2
            cc  = lbl == 3
            ed  = lbl == 4
            if isinstance(lbl, torch.Tensor):
                d[key] = torch.stack([et, net, cc, ed], dim=0).float()
            else:
                d[key] = np.stack([et, net, cc, ed], axis=0).astype(np.float32)
        return d


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

MODALITY_SUFFIXES = ("t1c", "t1n", "t2f", "t2w")
SEG_SUFFIX = "seg"
DATA_ROOTS = [
    "data/BraTS-PEDs_Batch2_Release",
    "data/BraTS26_PED_training",
]


def _find_subjects(base_dir: Path) -> List[Path]:
    """Return sorted list of subject directories under *base_dir*."""
    if not base_dir.exists():
        return []
    return sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and p.name.startswith("BraTS-PED-")
    )


def _build_file_list(project_root: Path, require_seg: bool = True) -> List[dict]:
    """
    Walk all DATA_ROOTS and collect per-subject dicts with keys:
      image  (list of 4 modality paths)
      label  (seg path, only when require_seg=True)

    Args:
        project_root:  Root of the project (parent of 'data/').
        require_seg:   If False, subjects without a seg file are included
                       (useful for inference on validation / test sets).
    """
    samples = []
    for root_rel in DATA_ROOTS:
        root = project_root / root_rel
        for subj in _find_subjects(root):
            sid = subj.name
            modality_paths = [
                str(subj / f"{sid}-{m}.nii.gz") for m in MODALITY_SUFFIXES
            ]
            seg_path = str(subj / f"{sid}-{SEG_SUFFIX}.nii.gz")

            if not all(os.path.isfile(p) for p in modality_paths):
                continue

            if require_seg and not os.path.isfile(seg_path):
                continue

            entry = {"image": modality_paths}
            if os.path.isfile(seg_path):
                entry["label"] = seg_path
            samples.append(entry)

    return samples


# ---------------------------------------------------------------------------
# MONAI transform pipelines
# ---------------------------------------------------------------------------

PATCH_SIZE = (128, 128, 128)
TARGET_SPACING = (1.0, 1.0, 1.0)  # isotropic 1 mm


def _base_transforms() -> List:
    """
    Shared preprocessing: load → orient → resample → convert labels → normalize.

    Note: data is defaced but NOT skull-stripped (BraTS-PED 2025 policy).
    Skull voxels remain; CropForegroundd trims air padding using a threshold
    above background so skull tissue is preserved in the crop.
    Per-channel z-score normalization with nonzero=True correctly ignores only
    the background air, leaving skull and brain intensities intact.
    """
    return [
        LoadImaged(keys=["image", "label"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Consistent RAS orientation across scanners
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        # Resample to isotropic 1 mm spacing
        Spacingd(
            keys=["image", "label"],
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest"),
        ),
        # BraTS-PEDs 2026 labels 1/2/3/4 → 4-channel binary masks [ET, NET, CC, ED]
        ConvertBratsPed2025Labelsd(keys=["label"]),
        # Remove background air padding (skull is preserved — threshold > 0 keeps
        # all tissue voxels including skull since data is not skull-stripped)
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=list(PATCH_SIZE),
        ),
        # Per-channel z-score over non-zero (non-air) voxels
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]


def _base_transforms_infer() -> List:
    """Base transforms for unlabelled data (validation/test sets, no seg file)."""
    return [
        LoadImaged(keys=["image"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=TARGET_SPACING, mode="bilinear"),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            k_divisible=list(PATCH_SIZE),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ]


def _train_transforms() -> Compose:
    return Compose(
        _base_transforms()
        + [
            # ── Spatial augmentations ──────────────────────────────────────
            # Foreground-biased patch sampling (2:1 positive-to-negative ratio)
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=2,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            # Axis-aligned flips — p=0.5 per axis
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # 90° lattice rotations (lossless for cubic patches)
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            # Continuous rotation ±15° all axes
            RandRotated(
                keys=["image", "label"],
                range_x=0.26,
                range_y=0.26,
                range_z=0.26,
                prob=0.3,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            # Random zoom 0.85×–1.15×
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.85,
                max_zoom=1.15,
                prob=0.3,
                mode=("trilinear", "nearest"),
                padding_mode="constant",
            ),
            # ── Intensity augmentations ───────────────────────────────────
            # Gaussian noise — simulates scanner thermal noise
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            # Gaussian blur — simulates inter-scanner resolution differences
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.15,
            ),
            # Intensity scale / shift — simulates gain / offset variability
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            # Gamma contrast — simulates non-linear scanner response
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            # track_meta=False strips MetaTensor → plain Tensor so torch.compile
            # doesn't hit cache_size_limit from MetaTensor.__torch_function__ dispatch
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )


def _val_transforms() -> Compose:
    """
    Validation transforms: full-volume preprocessing without spatial cropping.
    CropForegroundd (k_divisible=PATCH_SIZE) ensures each dim is divisible by
    the patch size; sliding-window inference in validate_epoch handles the rest.
    """
    return Compose(
        _base_transforms()
        + [
            # Pad only if any dimension is smaller than one patch (rare for brain MRI)
            SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )


def inference_transforms() -> Compose:
    """
    Full-volume transforms for challenge submission / sliding-window inference.
    No cropping — the model receives the entire resampled, normalised volume.
    Use with monai.inferers.SlidingWindowInferer in the evaluation loop.
    """
    return Compose(_base_transforms_infer())


# ---------------------------------------------------------------------------
# Dataset & DataLoader builders
# ---------------------------------------------------------------------------

def build_datasets(
    project_root: str = ".",
    val_fraction: float = 0.2,
    seed: int = 42,
    cache_rate: float = 0.0,
    num_workers: int = 4,
) -> Tuple[CacheDataset, CacheDataset]:
    """
    Build train/val CacheDatasets from labelled subjects.

    Args:
        project_root:  Root directory of the project (contains 'data/').
        val_fraction:  Fraction of subjects held out for validation.
        seed:          Random seed for the train/val split.
        cache_rate:    Fraction of data to cache in RAM (0.0 = no caching,
                       1.0 = full dataset — requires sufficient RAM).
        num_workers:   Workers used by CacheDataset for pre-loading.

    Returns:
        (train_dataset, val_dataset)
    """
    root = Path(project_root)
    all_samples = _build_file_list(root, require_seg=True)

    if not all_samples:
        raise RuntimeError(
            f"No valid BraTS-PED subjects found under {root / 'data'}. "
            "Check that the expected directory structure exists."
        )

    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    print(f"Dataset split → train: {len(train_samples)}, val: {len(val_samples)}")

    train_ds = CacheDataset(
        data=train_samples,
        transform=_train_transforms(),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_samples,
        transform=_val_transforms(),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    return train_ds, val_ds


def build_dataloaders(
    project_root: str = ".",
    batch_size: int = 1,
    val_fraction: float = 0.2,
    seed: int = 42,
    cache_rate: float = 0.0,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience wrapper that returns ready-to-use DataLoaders.

    Args:
        project_root:  Root directory of the project.
        batch_size:    Samples per batch (keep at 1 for full 3D volumes
                       unless you have ≥32 GB GPU VRAM).
        val_fraction:  Fraction held out for validation.
        seed:          Split seed.
        cache_rate:    Fraction cached in RAM.
        num_workers:   DataLoader workers.

    Returns:
        (train_loader, val_loader)
    """
    train_ds, val_ds = build_datasets(
        project_root=project_root,
        val_fraction=val_fraction,
        seed=seed,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    project_root = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"Scanning {Path(project_root).resolve() / 'data'} ...")
    train_loader, val_loader = build_dataloaders(
        project_root=project_root,
        batch_size=1,
        cache_rate=0.0,
        num_workers=0,
    )

    print(f"Train batches: {len(train_loader)},  Val batches: {len(val_loader)}")

    batch = next(iter(train_loader))
    img = batch["image"]
    lbl = batch["label"]

    print(f"Image shape  : {img.shape}  dtype: {img.dtype}")
    print(f"Label shape  : {lbl.shape}  dtype: {lbl.dtype}")
    print(f"Image range  : [{img.min():.3f}, {img.max():.3f}]")
    # Each label channel is binary; report positive-voxel fraction per channel
    ch_names = ["ET", "NET", "CC", "ED"]
    for i, name in enumerate(ch_names):
        frac = lbl[0, i].float().mean().item()
        print(f"  {name} positive fraction: {frac:.4f}")
    print("Sanity check passed.")
