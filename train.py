"""
BraTS-PED 2025 — training script for MedSwinNet

Trains on a single held-out validation split (--val_count subjects, default
40, randomly selected) rather than cross-validation — one run, one set of
subjects the model never trains on, giving a real per-region validation Dice
(ET, NET, CC, ED, TC, WT) each epoch.

Usage:
  python train.py                          # 40 held out, rest train
  python train.py --val_count 25          # override held-out count
  python train.py --epochs 300 --batch 2  # override defaults

Checkpoints saved to:
  checkpoints/fold0_best.pt
  checkpoints/fold0_last.pt
"""

import os
# Must be set before `import torch` — the CUDA allocator reads this at context
# init. Without it, PyTorch's caching allocator can leave several GB
# "reserved but unallocated" (fragmentation) instead of reusing/returning it,
# artificially inflating peak usage until an allocation fails even though
# the actual working set would otherwise fit.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore", category=FutureWarning, module="monai")


def _worker_init(_: int) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="monai")


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── RTX 6000 Ada / Ampere+ GPU optimizations ─────────────────────────────────
# TF32 gives ~2× faster matmuls on Ada tensor cores with negligible accuracy loss.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark      = True   # auto-tune conv algorithms per shape
torch.backends.cudnn.allow_tf32     = True
torch.backends.cuda.matmul.allow_tf32 = True

from monai.data import CacheDataset, pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from data_loader import _build_file_list, _train_transforms, _val_transforms
from models import (
    ETFocalLoss,
    IN_CHANNELS,
    OUT_CHANNELS,
    PATCH_SIZE,
    MedSwinNet,
    deep_supervision_loss,
    post_process_et,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    project_root:  str   = "."
    ckpt_dir:      str   = "checkpoints"

    seed:          int   = 42
    val_count:     int   = 40         # subjects held out for validation; rest train

    epochs:        int   = 300
    # 1e-4 is the from-scratch peak LR train_v2.py used to produce best.pt.
    # Fine-tuning that already-converged checkpoint at the same peak LR
    # regressed validation Dice as soon as warmup ramped up to it (best
    # result was at ~2e-5, still in warmup); lowered to match.
    lr:            float = 2e-5
    weight_decay:  float = 1e-5
    batch_size:    int   = 4      # RTX 6000 Ada (48 GB): fits 4× 128³; reduce to 2 if OOM with scaled model
    warmup_epochs: int   = 10
    grad_clip:     float = 1.0
    grad_accum_steps: int = 1     # gradient accumulation: effective batch = batch_size × grad_accum_steps

    # Loss weights — sum should be 1.0
    dice_ce_weight:  float = 0.7
    et_focal_weight: float = 0.2
    aux_wt_weight:   float = 0.1
    ds_weights: List[float] = field(default_factory=lambda: [1.0, 0.5])

    # predict.py measured 8 -> 4 saving ~21 GB peak VRAM with identical
    # throughput (sliding-window patches are small relative to a full batch
    # forward); train.py predates that fix, so it inherited the old default.
    sw_batch_size:  int   = 4
    sw_overlap:     float = 0.75  # 0.75 reduces stitching artefacts at patch boundaries (competition standard)
    min_et_voxels:  int   = 10
    use_tta:        bool  = False  # 8-flip TTA at inference; ~8× slower — enable for final eval/submission

    # Hard-cap the allocator at this fraction of total VRAM. Without any cap,
    # this architecture can push right up to the card's ceiling with ~0
    # headroom, and Windows' WDDM driver responds to that by paging GPU
    # memory instead of raising a clean OOM — which looks like a hang, not a
    # crash. But this model's own measured peak (train_v2.py) is ~44 GB on a
    # 48 GB card — 0.9 (43.2 GB) cuts it too close and OOMs during a normal
    # backward pass; 0.95 (~45.6 GB) leaves the model room to actually fit
    # while still keeping ~2.4 GB free so the driver never has to page.
    gpu_mem_fraction: float = 0.95

    cache_rate:    float = 0.1    # fraction of dataset cached in RAM after deterministic transforms
    num_workers:   int   = 12     # more workers feeds the GPU without stalling
    amp:           bool  = True
    use_bf16:      bool  = True   # BF16 on Ada: same speed as FP16, wider dynamic range
    compile_model: bool  = False  # torch.compile requires Triton (Linux only); disabled on Windows
    log_every:     int   = 5

    # Warm-start each fold's model weights from an existing checkpoint (e.g.
    # train_v2.py's full-dataset checkpoints/best.pt) instead of random init.
    # Only the model weights are loaded — each fold still gets its own fresh
    # optimizer/scheduler and trains for cfg.epochs, since the checkpoint's
    # optimizer/scheduler state belongs to a different training run (full
    # dataset, different total epoch count) and doesn't correspond to any
    # single fold here. Set to "" to disable and train from scratch.
    init_ckpt:     str   = "checkpoints/best.pt"


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
    """nnU-Net polynomial LR decay with linear warm-up."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    return (1.0 - (epoch - warmup) / max(total - warmup, 1)) ** exp


def save_checkpoint(state: dict, is_best: bool, path_last: Path, path_best: Path):
    torch.save(state, path_last)
    if is_best:
        shutil.copy(path_last, path_best)


def count_params(model: nn.Module) -> str:
    return f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f} M"


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
    pbar = tqdm(loader, desc=f"Ep {epoch:>3d}", unit="batch", dynamic_ncols=True, leave=False)
    for i, batch in enumerate(pbar):
        images = batch["image"].to(device)   # (B, 4, H, W, D)
        labels = batch["label"].to(device)   # (B, 4, H, W, D) — [ET, NET, CC, ED]

        if i % accum == 0:
            optimizer.zero_grad()

        dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16
        with autocast("cuda", dtype=dtype, enabled=cfg.amp):
            outputs, wt_aux_logit = model(images)

            # Weighted DiceCE across deep-supervision scales
            if isinstance(outputs, (list, tuple)):
                main_loss   = deep_supervision_loss(outputs, criterion, labels, cfg.ds_weights)
                logits_full = outputs[0]
            else:
                main_loss   = criterion(outputs, labels)
                logits_full = outputs

            # ET-focal loss on full-resolution output
            et_loss = et_criterion(logits_full, labels)

            # Auxiliary WT BCE (coarse WT self-supervision for SPADE conditioning).
            # WT = union of all sub-region channels [ET, NET, CC, ED].
            wt_full = labels.amax(dim=1, keepdim=True).float()
            wt_gt   = F.interpolate(
                wt_full, size=wt_aux_logit.shape[2:], mode="nearest"
            )
            aux_loss = F.binary_cross_entropy_with_logits(wt_aux_logit, wt_gt)

            loss = (cfg.dice_ce_weight  * main_loss
                  + cfg.et_focal_weight * et_loss
                  + cfg.aux_wt_weight   * aux_loss) / accum

        scaler.scale(loss).backward()

        is_last_batch = (i + 1) == len(loader)
        if (i + 1) % accum == 0 or is_last_batch:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * accum   # undo normalization for logging
        n += 1

        if (i + 1) % cfg.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item() * accum:.4f}", lr=f"{lr:.2e}")
            logger.info(
                f"  epoch {epoch:>3d}  batch {i+1:>4d}/{len(loader)}"
                f"  loss={loss.item() * accum:.4f}  lr={lr:.2e}"
            )

    pbar.close()
    return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Validation loop
# ─────────────────────────────────────────────────────────────────────────────

def _sw_predict(model: nn.Module, images: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Sliding-window inference returning sigmoid probabilities.

    Gradient checkpointing is a no-op in eval mode, so without autocast this
    ran sw_batch_size patches through in full FP32 with the full activation
    memory of every layer — that's what produced the 16 GiB single-tensor
    OOM. bf16 halves that footprint, matching train_epoch and predict.py.
    """
    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16
    with autocast("cuda", dtype=dtype, enabled=cfg.amp):
        logits = sliding_window_inference(
            inputs=images,
            roi_size=PATCH_SIZE,
            sw_batch_size=cfg.sw_batch_size,
            predictor=model.forward_inference,
            overlap=cfg.sw_overlap,
            mode="gaussian",
        )
    return torch.sigmoid(logits.float())


@torch.inference_mode()
def tta_predict(model: nn.Module, images: torch.Tensor, cfg: Config) -> torch.Tensor:
    """8-flip test-time augmentation: average sigmoid probs over all axis-flip combos."""
    flip_sets = [
        [],
        [2], [3], [4],
        [2, 3], [2, 4], [3, 4],
        [2, 3, 4],
    ]
    preds = []
    for axes in flip_sets:
        inp = torch.flip(images, axes) if axes else images
        prob = _sw_predict(model, inp, cfg)
        preds.append(torch.flip(prob, axes) if axes else prob)
    return torch.stack(preds).mean(0)


def _with_derived_regions(x: torch.Tensor) -> torch.Tensor:
    """Append derived TC and WT channels to a 4-channel [ET, NET, CC, ED] mask.

    Returns 6 channels [ET, NET, CC, ED, TC, WT] so DiceMetric reports every
    region scored on the BraTS-PEDs 2026 leaderboard:
      TC = ET | NET | CC       (channels 0,1,2)
      WT = ET | NET | CC | ED   (channels 0,1,2,3)
    """
    tc = x[:, 0:3].amax(dim=1, keepdim=True)
    wt = x[:, 0:4].amax(dim=1, keepdim=True)
    return torch.cat([x, tc, wt], dim=1)


@torch.inference_mode()
def validate_epoch(
    model:  nn.Module,
    loader: DataLoader,
    metric: DiceMetric,
    device: torch.device,
    cfg:    Config,
    logger: logging.Logger,
) -> Dict[str, float]:
    model.eval()
    metric.reset()
    n_val = len(loader)

    for idx, batch in enumerate(loader, 1):
        if idx % 10 == 0 or idx == n_val:
            logger.info(f"  [val] {idx}/{n_val}")

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if cfg.use_tta:
            probs = tta_predict(model, images, cfg)
        else:
            probs = _sw_predict(model, images, cfg)

        preds = post_process_et(probs, min_et_voxels=cfg.min_et_voxels)
        # Score all six leaderboard regions: 4 subregions + derived TC and WT
        metric(_with_derived_regions(preds), _with_derived_regions(labels))

    scores, _ = metric.aggregate()
    metric.reset()

    names = ["ET", "NET", "CC", "ED", "TC", "WT"]
    ch = {name: scores[i].item() for i, name in enumerate(names)}
    ch["mean"] = float(np.nanmean([ch[n] for n in names]))
    return ch


# ─────────────────────────────────────────────────────────────────────────────
# One fold
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold:       int,
    train_data: list,
    val_data:   list,
    cfg:        Config,
    logger:     logging.Logger,
) -> float:
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg.ckpt_dir)

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.gpu_mem_fraction, torch.cuda.current_device())
        logger.info(f"Fold {fold} | GPU memory capped at {cfg.gpu_mem_fraction:.0%} of total VRAM")

    model = MedSwinNet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        deep_supervision=True,
        # Without this, batch=4 pushes this architecture right up against a
        # 49 GB card's ceiling (measured: ~48 GB, effectively no headroom) —
        # WDDM then pages GPU memory under that pressure instead of erroring,
        # which looks like a hang rather than an OOM. train_v2.py measured
        # ~44 GB peak at batch=4 *with* checkpointing enabled.
        use_checkpoint=True,
    ).to(device)
    logger.info(f"Fold {fold} | params: {count_params(model)}")

    if cfg.init_ckpt:
        init_path = Path(cfg.init_ckpt)
        if init_path.exists():
            state = torch.load(init_path, map_location=device, weights_only=False)
            raw   = state.get("model", state)
            clean = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}
            model.load_state_dict(clean)
            logger.info(
                f"Fold {fold} | warm-started weights from {init_path} "
                f"(epoch={state.get('epoch', '?')}, loss={state.get('loss', '?')})"
            )
        else:
            logger.info(f"Fold {fold} | --init_ckpt {init_path} not found — training from scratch.")

    if cfg.compile_model:
        import sys
        if sys.platform == "win32":
            # torch.compile inductor backend requires Triton, which is Linux-only.
            # Fall back to eager — TF32, BF16, cuDNN benchmark, and Flash Attention
            # are still active and provide significant speedup without Triton.
            logger.warning("  torch.compile skipped: Triton is not available on Windows. Running in eager mode.")
        else:
            logger.info("  Compiling model (max-autotune) — first epoch will be slow...")
            model = torch.compile(model, mode="max-autotune")

    train_ds = CacheDataset(
        train_data, _train_transforms(),
        cache_rate=cfg.cache_rate, num_workers=cfg.num_workers,
    )
    val_ds = CacheDataset(
        val_data, _val_transforms(),
        cache_rate=cfg.cache_rate, num_workers=cfg.num_workers,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=pad_list_data_collate,
        worker_init_fn=_worker_init, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    logger.info(
        f"Fold {fold} | train={len(train_loader)} batches  val={len(val_loader)} subjects"
        f"  workers={cfg.num_workers}  cache_rate={cfg.cache_rate}"
    )

    criterion    = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    et_criterion = ETFocalLoss(et_weight=3.0, gamma=2.0)
    optimizer    = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler    = LambdaLR(
        optimizer,
        lambda ep: warmup_poly(ep, cfg.warmup_epochs, cfg.epochs),
    )
    # GradScaler is only needed for FP16; BF16 has wide enough dynamic range.
    scaler = GradScaler("cuda", enabled=cfg.amp and not cfg.use_bf16)
    metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)

    best_dice = -1.0
    path_last = ckpt_dir / f"fold{fold}_last.pt"
    path_best = ckpt_dir / f"fold{fold}_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        logger.info(f"Fold {fold} | epoch {epoch}/{cfg.epochs} starting")
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, et_criterion,
            scaler, device, cfg, logger, epoch,
        )
        scheduler.step()

        scores    = validate_epoch(model, val_loader, metric, device, cfg, logger)
        mean_dice = scores["mean"]

        logger.info(
            f"Fold {fold} | epoch {epoch:>3d}/{cfg.epochs}"
            f"  loss={train_loss:.4f}"
            f"  ET={scores['ET']:.3f} NET={scores['NET']:.3f} CC={scores['CC']:.3f}"
            f" ED={scores['ED']:.3f} TC={scores['TC']:.3f} WT={scores['WT']:.3f}"
            f"  mean={mean_dice:.4f}"
        )

        is_best = mean_dice > best_dice
        if is_best:
            best_dice = mean_dice
            logger.info(f"  ** New best: {best_dice:.4f}")

        save_checkpoint(
            {"epoch": epoch, "fold": fold, "model": model.state_dict(),
             "best_dice": best_dice, "scores": scores},
            is_best, path_last, path_best,
        )

    return best_dice


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS-PED 2025 — MedSwinNet")
    p.add_argument("--project_root", default=".")
    p.add_argument("--ckpt_dir",     default="checkpoints")
    p.add_argument("--val_count",    type=int,   default=40,
                   help="Number of subjects held out for validation (rest train)")
    p.add_argument("--epochs",       type=int,   default=300)
    p.add_argument("--batch",        type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-5,
                   help="Peak LR after warmup. Lowered from the 1e-4 used for "
                        "from-scratch training -- this run fine-tunes an "
                        "already-converged checkpoint, and 1e-4 regressed "
                        "validation Dice as soon as warmup reached it.")
    p.add_argument("--cache_rate",   type=float, default=0.0)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no_amp",       action="store_true", help="Disable AMP")
    p.add_argument("--no_bf16",      action="store_true", help="Use FP16 instead of BF16")
    # torch.compile disabled by default on Windows (no Triton); set compile_model=True in Config if on Linux
    p.add_argument("--use_tta",      action="store_true", help="Enable 8-flip TTA at validation (slow)")
    p.add_argument("--grad_accum",   type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--init_ckpt",    default="checkpoints/best.pt",
                   help="Warm-start each fold's model weights from this checkpoint "
                        "(empty string to train from scratch)")
    p.add_argument("--gpu_mem_fraction", type=float, default=0.95,
                   help="Cap the CUDA allocator at this fraction of total VRAM "
                        "(default 0.95) — leaves a small safety margin against "
                        "WDDM paging-induced stalls without cutting into this "
                        "model's own ~44 GB measured peak")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Config(
        project_root=args.project_root,
        ckpt_dir=args.ckpt_dir,
        val_count=args.val_count,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        use_bf16=not args.no_bf16,
        use_tta=args.use_tta,
        grad_accum_steps=args.grad_accum,
        init_ckpt=args.init_ckpt,
        gpu_mem_fraction=args.gpu_mem_fraction,
    )

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(cfg.ckpt_dir) / "train.log")
    logger.info(f"Config: {cfg}")
    logger.info(
        f"CUDA: {torch.cuda.is_available()}  "
        f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    all_data = _build_file_list(Path(cfg.project_root), require_seg=True)
    logger.info(f"Total subjects: {len(all_data)}")

    n_val = max(1, min(cfg.val_count, len(all_data) - 1))
    rng   = np.random.default_rng(cfg.seed)
    idx   = rng.permutation(len(all_data))
    logger.info(
        f"Held-out split: {len(all_data) - n_val} train, {n_val} val "
        f"(seed={cfg.seed})"
    )

    best = train_fold(
        0,
        [all_data[i] for i in idx[n_val:]],
        [all_data[i] for i in idx[:n_val]],
        cfg, logger,
    )
    logger.info(f"Best mean Dice: {best:.4f}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
