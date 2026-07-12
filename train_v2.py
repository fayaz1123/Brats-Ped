"""
BraTS-PED 2025 — full-dataset training (no validation).

Trains on all labelled subjects (BraTS26_PED_training + BraTS-PEDs_Batch2_Release).
Checkpoints saved every --save_every epochs and at the final epoch.

Usage:
  python train_v2.py                         # safe defaults
  python train_v2.py --batch 2 --accum 2    # slightly faster if VRAM allows
  python train_v2.py --no_grad_ckpt         # disable checkpointing (faster, more VRAM)
  python train_v2.py --epochs 500 --save_every 10
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore", category=FutureWarning, module="monai")
# Benign: RandCropByLabelClassesd zeroes a class's crop ratio for a given
# subject when that subject has none of that label (expected given BraTS-PED's
# per-class rarity — see ConvertBratsPed2025Labelsd docstring), not an error.
warnings.filterwarnings("ignore", message="no available indices of class.*",
                         category=UserWarning)


def _worker_init(_: int) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="monai")
    warnings.filterwarnings("ignore", message="no available indices of class.*",
                             category=UserWarning)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cuda.matmul.allow_tf32 = True

from monai.data import PersistentDataset, pad_list_data_collate
from monai.losses import DiceCELoss

from data_loader import _build_file_list, _train_transforms
from models import (
    ETFocalLoss,
    IN_CHANNELS,
    OUT_CHANNELS,
    MedSwinNet,
    deep_supervision_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    project_root: str   = "."
    ckpt_dir:     str   = "checkpoints"

    epochs:           int   = 500
    lr:               float = 1e-4
    weight_decay:     float = 1e-5
    # RTX 6000 Ada (48 GB): batch=4 @ 128³ BF16 + grad-ckpt ≈ 12-15 GB VRAM.
    # Try batch=8 if the logged peak stays under 40 GB after epoch 1.
    batch_size:       int   = 4
    warmup_epochs:    int   = 10
    grad_clip:        float = 1.0
    # No accumulation needed: batch=4 fits directly, so grad_accum=1 is most efficient.
    grad_accum_steps: int   = 1

    dice_ce_weight:  float = 0.7
    et_focal_weight: float = 0.2
    aux_wt_weight:   float = 0.1
    ds_weights: List[float] = field(default_factory=lambda: [1.0, 0.5])

    # Disk-backed cache (monai.data.PersistentDataset) for the deterministic
    # preprocessing prefix (load, orient, resample to 1mm iso, crop-foreground,
    # normalize) — only the random crop + augment steps re-run each epoch.
    # Without this, that whole expensive pipeline re-runs from raw NIfTI on
    # every sample access, every epoch, starving the GPU waiting on CPU
    # preprocessing. Uses disk, not RAM.
    cache_dir:     str   = "cache"
    # 12 workers keeps the Ada busy; reduce if system RAM is limited.
    num_workers:   int   = 12
    amp:           bool  = True
    use_bf16:      bool  = True
    compile_model: bool  = False
    log_every:     int   = 5
    save_every:    int   = 10

    # Gradient checkpointing halves activation memory at ~20% compute cost.
    # Keeps batch=4 comfortably under 20 GB so there is headroom for spikes.
    use_grad_ckpt: bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("brats_ped")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    for h in [logging.StreamHandler(), logging.FileHandler(log_path)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def warmup_poly(epoch: int, warmup: int, total: int, exp: float = 0.9) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    return (1.0 - (epoch - warmup) / max(total - warmup, 1)) ** exp


def count_params(model: nn.Module) -> str:
    return f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f} M"


def log_gpu_mem(logger: logging.Logger, tag: str = "") -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    resv  = torch.cuda.memory_reserved()  / 1e9
    peak  = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"  [GPU{' ' + tag if tag else ''}]  alloc={alloc:.2f} GB  "
                f"reserved={resv:.2f} GB  peak={peak:.2f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    et_criterion: nn.Module,
    scaler:       GradScaler,
    device:       torch.device,
    cfg:          Config,
    logger:       logging.Logger,
    epoch:        int,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    accum = cfg.grad_accum_steps
    pbar  = tqdm(loader, desc=f"Ep {epoch:>3d}/{cfg.epochs}", unit="batch",
                 dynamic_ncols=True, leave=False)

    for i, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if i % accum == 0:
            optimizer.zero_grad(set_to_none=True)

        dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16
        with autocast("cuda", dtype=dtype, enabled=cfg.amp):
            outputs, wt_aux_logit = model(images)

            if isinstance(outputs, (list, tuple)):
                main_loss   = deep_supervision_loss(outputs, criterion, labels, cfg.ds_weights)
                logits_full = outputs[0]
            else:
                main_loss   = criterion(outputs, labels)
                logits_full = outputs

            et_loss  = et_criterion(logits_full, labels)
            # WT = union of all sub-region channels [ET, NET, CC, ED]
            wt_binary = labels.amax(dim=1, keepdim=True).float()
            wt_gt     = F.interpolate(wt_binary, size=wt_aux_logit.shape[2:], mode="nearest")
            aux_loss  = F.binary_cross_entropy_with_logits(wt_aux_logit, wt_gt)

            loss = (cfg.dice_ce_weight  * main_loss
                  + cfg.et_focal_weight * et_loss
                  + cfg.aux_wt_weight   * aux_loss) / accum

        scaler.scale(loss).backward()

        # Cheap per-channel visibility, computed only on logged steps: without
        # a held-out validation loop, a channel silently collapsing (e.g. ED
        # never crossing the sigmoid threshold) is invisible behind the
        # aggregate scalar loss. Comparing predicted vs. ground-truth positive
        # voxel fraction per sub-region catches that immediately.
        do_log = (i + 1) % cfg.log_every == 0
        if do_log:
            with torch.no_grad():
                pred_frac = torch.sigmoid(logits_full.detach().float()).ge(0.5).float().mean(dim=(0, 2, 3, 4))
                gt_frac   = labels.float().mean(dim=(0, 2, 3, 4))
            channel_stats = list(zip(["ET", "NET", "CC", "ED"], pred_frac.tolist(), gt_frac.tolist()))

        del images, labels, outputs, wt_aux_logit, logits_full, wt_binary, wt_gt

        is_last = (i + 1) == len(loader)
        if (i + 1) % accum == 0 or is_last:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * accum
        n += 1

        if do_log:
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item() * accum:.4f}", lr=f"{lr:.2e}")
            stats_str = "  ".join(f"{name}:pred={p:.4f}/gt={g:.4f}" for name, p, g in channel_stats)
            logger.info(
                f"  epoch {epoch:>3d}  batch {i+1:>4d}/{len(loader)}"
                f"  loss={loss.item() * accum:.4f}  lr={lr:.2e}  {stats_str}"
            )

    pbar.close()
    return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    all_data: list,
    cfg:      Config,
    logger:   logging.Logger,
) -> None:
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg.ckpt_dir)

    model = MedSwinNet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        deep_supervision=True,
        use_checkpoint=cfg.use_grad_ckpt,
    ).to(device)
    logger.info(f"Params: {count_params(model)}  grad_ckpt={'ON' if cfg.use_grad_ckpt else 'OFF'}")

    if cfg.compile_model:
        import sys
        if sys.platform == "win32":
            logger.warning("  torch.compile skipped: Triton unavailable on Windows.")
        else:
            logger.info("  Compiling model (max-autotune)...")
            model = torch.compile(model, mode="max-autotune")

    ds = PersistentDataset(
        data=all_data, transform=_train_transforms(),
        cache_dir=Path(cfg.cache_dir),
    )

    # Warm the disk cache single-process, sequentially, before any parallel
    # workers touch it. PersistentDataset writes each subject's cache file
    # lazily on first access with no cross-process locking — if the real
    # num_workers>0 training loader below hit an un-cached subject, two
    # workers could race to write the same file and corrupt it (torn write ->
    # "failed finding central directory" on read). One single-threaded pass
    # here guarantees every file exists before workers ever read it.
    logger.info(f"Warming persistent disk cache for {len(all_data)} subjects "
                f"(single-process, avoids concurrent-write races)...")
    for i in tqdm(range(len(ds)), desc="Cache warm-up", unit="subject"):
        ds[i]

    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=pad_list_data_collate,
        worker_init_fn=_worker_init, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    logger.info(
        f"Training on {len(all_data)} subjects | {len(loader)} batches/epoch"
        f"  eff_batch={cfg.batch_size * cfg.grad_accum_steps}"
        f"  workers={cfg.num_workers}  cache_dir={cfg.cache_dir}"
    )

    criterion    = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    et_criterion = ETFocalLoss(et_weight=2.5, ed_weight=3.0, gamma=2.0)
    optimizer    = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler    = LambdaLR(
        optimizer,
        lambda ep: warmup_poly(ep, cfg.warmup_epochs, cfg.epochs),
    )
    scaler   = GradScaler("cuda", enabled=cfg.amp and not cfg.use_bf16)
    path_last = ckpt_dir / "last.pt"
    path_best = ckpt_dir / "best.pt"
    best_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        torch.cuda.reset_peak_memory_stats()

        train_loss = train_epoch(
            model, loader, optimizer, criterion, et_criterion,
            scaler, device, cfg, logger, epoch,
        )
        scheduler.step()
        torch.cuda.empty_cache()

        log_gpu_mem(logger, tag=f"ep{epoch}")
        logger.info(f"epoch {epoch:>3d}/{cfg.epochs}  loss={train_loss:.4f}")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": train_loss,
        }
        torch.save(state, path_last)

        # No validation set in full-dataset training, so "best" tracks the
        # lowest training loss seen so far.
        if train_loss < best_loss:
            best_loss = train_loss
            shutil.copy(path_last, path_best)
            logger.info(f"  ** New best (train loss {best_loss:.4f}) — saved best.pt")

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            shutil.copy(path_last, ckpt_dir / f"epoch_{epoch:04d}.pt")
            logger.info(f"  Saved checkpoint: epoch_{epoch:04d}.pt")

    logger.info("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS-PED 2025 — MedSwinNet full-dataset training")
    p.add_argument("--project_root", default=".")
    p.add_argument("--ckpt_dir",     default="checkpoints")
    p.add_argument("--epochs",       type=int,   default=500)
    p.add_argument("--batch",        type=int,   default=4,
                   help="Per-step batch size. RTX 6000 Ada: 4 @ 128³ BF16+ckpt ≈ 15 GB; try 8 for more throughput.")
    p.add_argument("--accum",        type=int,   default=1,
                   help="Gradient accumulation steps (1 = no accumulation)")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--cache_dir",    default="cache",
                   help="Disk directory caching the deterministic preprocessing prefix "
                        "(load/resample/crop) so it isn't redone from raw NIfTI every epoch")
    p.add_argument("--num_workers",  type=int,   default=12)
    p.add_argument("--no_amp",       action="store_true")
    p.add_argument("--no_bf16",      action="store_true")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="Disable gradient checkpointing (faster but uses more VRAM)")
    p.add_argument("--save_every",   type=int,   default=10,
                   help="Save a dated checkpoint every N epochs (default 10)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config(
        project_root=args.project_root,
        ckpt_dir=args.ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        lr=args.lr,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        use_bf16=not args.no_bf16,
        use_grad_ckpt=not args.no_grad_ckpt,
        save_every=args.save_every,
    )

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(cfg.ckpt_dir) / "train_v2.log")
    logger.info(f"Config: {cfg}")
    logger.info(
        f"CUDA: {torch.cuda.is_available()}  "
        f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"VRAM total: {total_gb:.1f} GB  |  batch={cfg.batch_size}  "
                    f"accum={cfg.grad_accum_steps}  eff_batch={cfg.batch_size * cfg.grad_accum_steps}  "
                    f"grad_ckpt={'ON' if cfg.use_grad_ckpt else 'OFF'}")

    all_data = _build_file_list(Path(cfg.project_root), require_seg=True)
    logger.info(f"Total subjects: {len(all_data)}")

    train_all(all_data, cfg, logger)


if __name__ == "__main__":
    main()
