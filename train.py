"""
BraTS-PED 2025 — Full training script

Priority order (as set in models.py):
  1. Stage-1 coarse WT (single run, no CV)
  2. Stage-2 Model A — TaskEnhancedDynUNet — 5-fold CV  [biggest Dice gain]
  3. Stage-2 Model B — SwinUNETR           — 5-fold CV  [frequency-aware ensemble]
  4. Post-processing: remove ET components < MIN_ET_VOXELS
  5. HFF module + SPADE conditioning active inside TaskEnhancedDynUNet

Usage
─────
  python train.py --stage 1                   # train Stage-1 WT segmenter
  python train.py --stage 2 --model dynunet   # 5-fold CV for Model A
  python train.py --stage 2 --model swin      # 5-fold CV for Model B
  python train.py --stage all                 # run all stages in order

Checkpoints
───────────
  checkpoints/stage1_best.pt
  checkpoints/dynunet_fold{0-4}_best.pt
  checkpoints/swin_fold{0-4}_best.pt
"""

import argparse
import logging
import math
import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress MONAI FutureWarnings in the main process (e.g. Orientationd labels).
warnings.filterwarnings("ignore", category=FutureWarning, module="monai")


def _worker_init(_: int) -> None:
    """Install warning filters inside each DataLoader worker process."""
    warnings.filterwarnings("ignore", category=FutureWarning, module="monai")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.model_selection import KFold
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from monai.data import CacheDataset, list_data_collate
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    Spacingd,
)

from data_loader import (
    _build_file_list,
    _train_transforms,
    _val_transforms,
)
from models import (
    ETFocalLoss,
    IN_CHANNELS,
    OUT_CHANNELS,
    PATCH_SIZE,
    TaskEnhancedDynUNet,
    build_stage1_net,
    build_swin_unetr,
    deep_supervision_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Paths
    project_root:   str  = "."
    ckpt_dir:       str  = "checkpoints"

    # Dataset
    val_fraction:   float = 0.2           # used only when n_folds=1
    n_folds:        int   = 5
    seed:           int   = 42

    # Stage-1 training (coarse WT at 3 mm)
    stage1_epochs:  int   = 100
    stage1_lr:      float = 1e-4
    stage1_wd:      float = 1e-5
    stage1_batch:   int   = 2
    stage1_spacing: Tuple[float, ...] = (3.0, 3.0, 3.0)
    stage1_roi:     Tuple[int, ...]   = (96, 96, 96)  # pad-or-crop to this (must be mult of 16)

    # Stage-2 training (fine 128^3 patches)
    stage2_epochs:  int   = 300
    stage2_lr:      float = 1e-4
    stage2_wd:      float = 1e-5
    stage2_batch:   int   = 1
    warmup_epochs:  int   = 10
    grad_clip:      float = 1.0

    # Loss weights
    dice_ce_weight: float = 0.7
    et_focal_weight: float = 0.3
    et_focal_gamma:  float = 2.0
    et_focal_channel_weight: float = 3.0
    ds_weights:     List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])

    # Inference
    sw_batch_size:  int   = 2
    sw_overlap:     float = 0.5

    # Post-processing
    min_et_voxels:  int   = 10

    # Data loading
    cache_rate:     float = 0.0
    num_workers:    int   = 4

    # Misc
    amp:            bool  = True    # automatic mixed precision
    log_every:      int   = 10      # log every N batches


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 data transforms  (3 mm, binary WT label)
# ─────────────────────────────────────────────────────────────────────────────

class ConvertToWTMaskd(MapTransform):
    """Convert scalar seg label to binary WT mask (any label > 0 = tumour)."""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            lbl = d[key]
            d[key] = (lbl > 0).float()
        return d


def _stage1_train_transforms(cfg: Config) -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["image", "label"],
            pixdim=cfg.stage1_spacing,
            mode=("bilinear", "nearest"),
        ),
        ConvertToWTMaskd(keys=["label"]),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
        ),
        # Crop-or-pad to a fixed size so batching works across variable-size volumes.
        # 96^3 is large enough for any brain at 3 mm; divisible by 16 (4 downsampling steps).
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=cfg.stage1_roi),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])


def _stage1_val_transforms(cfg: Config) -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["image", "label"],
            pixdim=cfg.stage1_spacing,
            mode=("bilinear", "nearest"),
        ),
        ConvertToWTMaskd(keys=["label"]),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
        ),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=cfg.stage1_roi),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("brats_ped")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def poly_lr_lambda(epoch: int, max_epochs: int, exponent: float = 0.9):
    """Polynomial LR decay as used in nnU-Net."""
    return (1.0 - epoch / max_epochs) ** exponent


def warmup_poly_lambda(
    epoch: int, warmup_epochs: int, max_epochs: int, exponent: float = 0.9
):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return poly_lr_lambda(epoch - warmup_epochs, max_epochs - warmup_epochs, exponent)


def post_process_et(
    pred: torch.Tensor,
    threshold: float = 0.5,
    min_et_voxels: int = 10,
) -> torch.Tensor:
    """
    Threshold predictions and remove ET connected components < min_et_voxels.

    Args:
        pred:          (B, 3, H, W, D) sigmoid probability map.
        threshold:     Binarisation threshold.
        min_et_voxels: Minimum voxel count for a valid ET component.

    Returns:
        Binary tensor same shape as pred.
    """
    binary = (pred >= threshold).cpu().numpy()
    for b in range(binary.shape[0]):
        et = binary[b, 0]                          # ET channel
        labeled, n_comp = ndimage.label(et)
        for comp_id in range(1, n_comp + 1):
            if (labeled == comp_id).sum() < min_et_voxels:
                et[labeled == comp_id] = 0
        binary[b, 0] = et
    return torch.from_numpy(binary.astype(np.float32)).to(pred.device)


def save_checkpoint(state: dict, is_best: bool, path_last: Path, path_best: Path):
    torch.save(state, path_last)
    if is_best:
        shutil.copy(path_last, path_best)


def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return f"{n:.1f} M"


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 prior helper (used inside Stage-2 training loop)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def get_stage1_prior(
    images: torch.Tensor,
    stage1_net: nn.Module,
    spacing_scale: float = 1.0 / 3.0,
) -> torch.Tensor:
    """
    Run Stage-1 on a batch of full-resolution patches to produce a WT prior.

    The prior is upsampled back to the original patch resolution and returned
    as a (B, 1, H, W, D) float tensor in [0, 1].

    This is called inside the Stage-2 training loop so Stage-2 models learn
    to condition on the Stage-1 WT probability map — the SPADE-inspired
    spatial conditioning described in models.py.
    """
    orig_size = images.shape[2:]
    x_low = F.interpolate(images, scale_factor=spacing_scale,
                          mode="trilinear", align_corners=False)

    # Pad to multiple of 16
    _, _, H, W, D = x_low.shape
    pH = (16 - H % 16) % 16
    pW = (16 - W % 16) % 16
    pD = (16 - D % 16) % 16
    if pH or pW or pD:
        x_low = F.pad(x_low, (0, pD, 0, pW, 0, pH))

    wt_low = torch.sigmoid(stage1_net(x_low))
    wt_low = wt_low[:, :, :H, :W, :D]            # remove padding
    return F.interpolate(wt_low, size=orig_size,
                         mode="trilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Core training / validation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    et_criterion: Optional[nn.Module],
    scaler:      GradScaler,
    device:      torch.device,
    cfg:         Config,
    logger:      logging.Logger,
    epoch:       int,
    stage1_net:  Optional[nn.Module] = None,
    deep_sup:    bool = True,
) -> float:
    """
    One training epoch.

    Args:
        model:       The segmentation model (may return list if deep_sup=True).
        loader:      Training DataLoader.
        criterion:   DiceCELoss (applied to each deep-supervision scale).
        et_criterion: ETFocalLoss (applied to full-resolution output only).
        stage1_net:  If provided, generate the WT prior and concatenate as
                     channel 5 before forwarding through model.
        deep_sup:    Whether the model returns a list (DynUNet) or single tensor.
    """
    model.train()
    if stage1_net is not None:
        stage1_net.eval()

    total_loss = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)   # (B, 4, H, W, D)
        labels = batch["label"].to(device)   # (B, 3, H, W, D)

        # Stage-1 cascade conditioning
        if stage1_net is not None:
            wt_prior = get_stage1_prior(images, stage1_net)
            images = torch.cat([images, wt_prior], dim=1)  # → (B, 5, H, W, D)

        optimizer.zero_grad()

        with autocast("cuda", enabled=cfg.amp):
            outputs = model(images)

            if deep_sup and isinstance(outputs, (list, tuple)):
                loss = deep_supervision_loss(
                    outputs, criterion, labels, cfg.ds_weights
                )
                logits_full = outputs[0]
            else:
                loss = criterion(outputs, labels)
                logits_full = outputs

            if et_criterion is not None:
                loss = (cfg.dice_ce_weight * loss
                        + cfg.et_focal_weight * et_criterion(logits_full, labels))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

        if (batch_idx + 1) % cfg.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch:>3d}  batch {batch_idx+1:>4d}/{len(loader)}"
                f"  loss={loss.item():.4f}  lr={lr:.2e}"
            )

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def validate_epoch(
    model:         nn.Module,
    loader:        DataLoader,
    metric:        DiceMetric,
    device:        torch.device,
    cfg:           Config,
    stage1_net:    Optional[nn.Module] = None,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Validate with patch-based forward pass; return per-channel and mean Dice.

    channel_names controls the keys returned. Use ["WT"] for Stage-1 (1 output
    channel) and ["ET", "TC", "WT"] (default) for Stage-2.
    """
    if channel_names is None:
        channel_names = ["ET", "TC", "WT"]
    model.eval()
    metric.reset()

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if stage1_net is not None:
            stage1_net.eval()
            wt_prior = get_stage1_prior(images, stage1_net)
            images = torch.cat([images, wt_prior], dim=1)

        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        probs = torch.sigmoid(outputs)
        if probs.shape[1] >= 3:
            preds = post_process_et(probs, min_et_voxels=cfg.min_et_voxels)
        else:
            preds = (probs >= 0.5).float()
        metric(preds, labels)

    scores, not_nans = metric.aggregate()
    metric.reset()

    ch = {name: scores[i].item() for i, name in enumerate(channel_names)}
    ch["mean"] = float(np.nanmean([v for v in ch.values()]))
    return ch


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 training
# ─────────────────────────────────────────────────────────────────────────────

def train_stage1(cfg: Config, logger: logging.Logger):
    """Train the coarse whole-tumour segmenter at 3 mm isotropic spacing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    root     = Path(cfg.project_root)
    all_data = _build_file_list(root, require_seg=True)
    n_val    = max(1, int(len(all_data) * cfg.val_fraction))
    rng      = np.random.default_rng(cfg.seed)
    idx      = rng.permutation(len(all_data))
    train_data = [all_data[i] for i in idx[n_val:]]
    val_data   = [all_data[i] for i in idx[:n_val]]

    logger.info(
        f"Stage-1  subjects: {len(all_data)}  "
        f"train={len(train_data)}  val={len(val_data)}"
    )

    train_ds = CacheDataset(
        data=train_data,
        transform=_stage1_train_transforms(cfg),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    val_ds = CacheDataset(
        data=val_data,
        transform=_stage1_val_transforms(cfg),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.stage1_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=_worker_init,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = build_stage1_net(
        in_channels=IN_CHANNELS,
        out_channels=1,
    ).to(device)
    logger.info(f"Stage-1 model params: {count_params(model)}")

    criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer = AdamW(model.parameters(), lr=cfg.stage1_lr, weight_decay=cfg.stage1_wd)
    scheduler = LambdaLR(
        optimizer,
        lambda ep: poly_lr_lambda(ep, cfg.stage1_epochs),
    )
    scaler    = GradScaler("cuda", enabled=cfg.amp)
    metric    = DiceMetric(include_background=True, reduction="mean_batch",
                           get_not_nans=True)

    best_dice = -1.0
    path_last = ckpt_dir / "stage1_last.pt"
    path_best = ckpt_dir / "stage1_best.pt"

    for epoch in range(1, cfg.stage1_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            et_criterion=None, scaler=scaler, device=device,
            cfg=cfg, logger=logger, epoch=epoch,
            stage1_net=None, deep_sup=False,
        )
        scheduler.step()

        scores = validate_epoch(model, val_loader, metric, device, cfg,
                                channel_names=["WT"])
        mean_dice = scores["mean"]

        logger.info(
            f"Stage-1  epoch {epoch:>3d}/{cfg.stage1_epochs}"
            f"  train_loss={train_loss:.4f}"
            f"  val_WT_dice={scores['WT']:.4f}"
        )

        is_best = mean_dice > best_dice
        if is_best:
            best_dice = mean_dice

        save_checkpoint(
            {"epoch": epoch, "model": model.state_dict(), "best_dice": best_dice},
            is_best, path_last, path_best,
        )

    logger.info(f"Stage-1 complete. Best WT Dice = {best_dice:.4f}")
    return str(path_best)


# ─────────────────────────────────────────────────────────────────────────────
# Stage-2 training — one fold
# ─────────────────────────────────────────────────────────────────────────────

def _build_stage2_model(
    model_name: str,
    in_channels: int,
    deep_supervision: bool,
    device: torch.device,
) -> nn.Module:
    if model_name == "dynunet":
        return TaskEnhancedDynUNet(
            in_channels=in_channels,
            out_channels=OUT_CHANNELS,
            deep_supervision=deep_supervision,
        ).to(device)
    elif model_name == "swin":
        return build_swin_unetr(
            in_channels=in_channels,
            out_channels=OUT_CHANNELS,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose 'dynunet' or 'swin'.")


def train_stage2_fold(
    fold:        int,
    model_name:  str,
    train_data:  list,
    val_data:    list,
    stage1_ckpt: Optional[str],
    cfg:         Config,
    logger:      logging.Logger,
) -> float:
    """
    Train one fold of Stage-2.

    Returns the best validation mean Dice for this fold.
    """
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg.ckpt_dir)

    # Stage-1 conditioning: load the pre-trained coarse WT segmenter
    stage1_net = None
    use_5ch    = False
    if stage1_ckpt is not None and Path(stage1_ckpt).exists():
        stage1_net = build_stage1_net(in_channels=IN_CHANNELS, out_channels=1).to(device)
        state = torch.load(stage1_ckpt, map_location=device)
        stage1_net.load_state_dict(state.get("model", state))
        stage1_net.eval()
        for p in stage1_net.parameters():
            p.requires_grad_(False)
        use_5ch = True
        logger.info(f"  Fold {fold}: Stage-1 prior enabled (5-channel input)")
    else:
        logger.info(f"  Fold {fold}: No Stage-1 prior (4-channel input)")

    in_channels  = 5 if use_5ch else IN_CHANNELS
    deep_sup     = model_name == "dynunet"
    model        = _build_stage2_model(model_name, in_channels, deep_sup, device)
    logger.info(f"  Fold {fold} | {model_name} | params: {count_params(model)}")

    train_ds = CacheDataset(
        data=train_data,
        transform=_train_transforms(),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    val_ds = CacheDataset(
        data=val_data,
        transform=_val_transforms(),
        cache_rate=cfg.cache_rate,
        num_workers=cfg.num_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.stage2_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=list_data_collate,
        worker_init_fn=_worker_init,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    criterion    = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    et_criterion = ETFocalLoss(
        et_weight=cfg.et_focal_channel_weight,
        gamma=cfg.et_focal_gamma,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.stage2_lr,
        weight_decay=cfg.stage2_wd,
    )
    scheduler = LambdaLR(
        optimizer,
        lambda ep: warmup_poly_lambda(ep, cfg.warmup_epochs, cfg.stage2_epochs),
    )
    scaler = GradScaler("cuda", enabled=cfg.amp)
    metric = DiceMetric(
        include_background=True,
        reduction="mean_batch",
        get_not_nans=True,
    )

    best_dice  = -1.0
    path_last  = ckpt_dir / f"{model_name}_fold{fold}_last.pt"
    path_best  = ckpt_dir / f"{model_name}_fold{fold}_best.pt"

    for epoch in range(1, cfg.stage2_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            et_criterion=et_criterion, scaler=scaler, device=device,
            cfg=cfg, logger=logger, epoch=epoch,
            stage1_net=stage1_net, deep_sup=deep_sup,
        )
        scheduler.step()

        scores    = validate_epoch(model, val_loader, metric, device, cfg,
                                   stage1_net=stage1_net)
        mean_dice = scores["mean"]

        logger.info(
            f"  Fold {fold} | epoch {epoch:>3d}/{cfg.stage2_epochs}"
            f"  loss={train_loss:.4f}"
            f"  ET={scores['ET']:.4f}  TC={scores['TC']:.4f}"
            f"  WT={scores['WT']:.4f}  mean={mean_dice:.4f}"
        )

        is_best = mean_dice > best_dice
        if is_best:
            best_dice = mean_dice
            logger.info(f"  ** New best: {best_dice:.4f}")

        save_checkpoint(
            {
                "epoch":     epoch,
                "fold":      fold,
                "model":     model.state_dict(),
                "best_dice": best_dice,
                "scores":    scores,
            },
            is_best, path_last, path_best,
        )

    return best_dice


# ─────────────────────────────────────────────────────────────────────────────
# Stage-2 5-fold cross-validation orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def train_kfold(
    model_name:  str,
    stage1_ckpt: Optional[str],
    cfg:         Config,
    logger:      logging.Logger,
):
    """
    Run 5-fold cross-validation for one architecture.

    Each fold saves:
      checkpoints/<model>_fold{i}_best.pt
      checkpoints/<model>_fold{i}_last.pt

    Prints a summary table at the end.
    """
    root     = Path(cfg.project_root)
    all_data = _build_file_list(root, require_seg=True)

    logger.info(
        f"\n{'='*60}\n"
        f"Stage-2  model={model_name}  n_folds={cfg.n_folds}  "
        f"subjects={len(all_data)}\n"
        f"{'='*60}"
    )

    kfold  = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    splits = list(kfold.split(all_data))

    fold_dices: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_data = [all_data[i] for i in train_idx]
        val_data   = [all_data[i] for i in val_idx]

        logger.info(
            f"\nFold {fold}/{cfg.n_folds-1}  "
            f"train={len(train_data)}  val={len(val_data)}"
        )

        best = train_stage2_fold(
            fold=fold,
            model_name=model_name,
            train_data=train_data,
            val_data=val_data,
            stage1_ckpt=stage1_ckpt,
            cfg=cfg,
            logger=logger,
        )
        fold_dices.append(best)
        logger.info(f"Fold {fold} best mean Dice = {best:.4f}")

    logger.info(
        f"\n{'='*60}\n"
        f"{model_name} 5-fold CV summary\n"
        + "\n".join(f"  fold {i}: {d:.4f}" for i, d in enumerate(fold_dices))
        + f"\n  mean : {np.mean(fold_dices):.4f} +/- {np.std(fold_dices):.4f}\n"
        f"{'='*60}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS-PED 2025 training")
    p.add_argument(
        "--stage", choices=["1", "2", "all"], default="all",
        help="Which stage to run: '1' (coarse WT), '2' (fine ET/TC/WT), 'all'"
    )
    p.add_argument(
        "--model", choices=["dynunet", "swin", "both"], default="both",
        help="Stage-2 architecture (used only when --stage 2 or all)"
    )
    p.add_argument("--project_root",   default=".",          help="Project root dir")
    p.add_argument("--ckpt_dir",       default="checkpoints", help="Checkpoint directory")
    p.add_argument("--n_folds",        type=int,   default=5)
    p.add_argument("--stage1_epochs",  type=int,   default=100)
    p.add_argument("--stage2_epochs",  type=int,   default=300)
    p.add_argument("--stage2_batch",   type=int,   default=1)
    p.add_argument("--stage1_batch",   type=int,   default=2)
    p.add_argument("--stage2_lr",      type=float, default=1e-4)
    p.add_argument("--stage1_lr",      type=float, default=1e-4)
    p.add_argument("--cache_rate",     type=float, default=0.0,
                   help="Fraction of data to cache in RAM (0=none, 1=all)")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--no_amp",         action="store_true",
                   help="Disable automatic mixed precision")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    cfg             = Config()
    cfg.project_root = args.project_root
    cfg.ckpt_dir     = args.ckpt_dir
    cfg.n_folds      = args.n_folds
    cfg.stage1_epochs = args.stage1_epochs
    cfg.stage2_epochs = args.stage2_epochs
    cfg.stage2_batch  = args.stage2_batch
    cfg.stage1_batch  = args.stage1_batch
    cfg.stage2_lr     = args.stage2_lr
    cfg.stage1_lr     = args.stage1_lr
    cfg.cache_rate    = args.cache_rate
    cfg.num_workers   = args.num_workers
    cfg.amp           = not args.no_amp
    cfg.seed          = args.seed

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(cfg.ckpt_dir) / "train.log")
    logger.info(f"Config: {cfg}")
    logger.info(f"CUDA: {torch.cuda.is_available()}  "
                f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    stage1_ckpt = str(Path(cfg.ckpt_dir) / "stage1_best.pt")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if args.stage in ("1", "all"):
        logger.info("\n" + "="*60 + "\nTRAINING STAGE-1 (coarse WT)\n" + "="*60)
        train_stage1(cfg, logger)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if args.stage in ("2", "all"):
        if not Path(stage1_ckpt).exists():
            logger.warning(
                f"Stage-1 checkpoint not found at {stage1_ckpt}. "
                "Stage-2 will train without cascade conditioning (4-channel input)."
            )
            stage1_ckpt = None

        models_to_train = (
            ["dynunet", "swin"] if args.model == "both" else [args.model]
        )
        for model_name in models_to_train:
            logger.info(f"\nTRAINING STAGE-2: {model_name}")
            train_kfold(
                model_name=model_name,
                stage1_ckpt=stage1_ckpt,
                cfg=cfg,
                logger=logger,
            )

    logger.info("\nAll training complete.")


if __name__ == "__main__":
    main()
